import csv
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from gpiozero import DigitalInputDevice

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows fallback
    msvcrt = None

MIN_BPM, MAX_BPM = 30, 220
BOUNCE_TIME = 0.15          # debounce: min 150ms between accepted beats
AVG_WINDOW = 6               # beats to average for a smoothed bpm
STALE_WARN_S = 5.0           # warn if no beat seen for this long

sensor = DigitalInputDevice(15, bounce_time=BOUNCE_TIME)

last_beat_time: Optional[float] = None
last_report_time = time.time()
ibi_history = deque(maxlen=AVG_WINDOW)

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

capture_active = False
capture_file: Optional[object] = None
capture_writer: Optional[csv.writer] = None
capture_path: Optional[Path] = None
latest_display = ""
stdin_fd: Optional[int] = None
stdin_old_attrs = None
state_lock = threading.Lock()

capture_start_time: Optional[float] = None
valid_beats = 0
rejected_beats = 0


def _write_header(writer):
    writer.writerow(["ts", "ibi_s", "instant_bpm", "avg_bpm"])


def start_capture() -> Optional[Path]:
    global capture_active, capture_file, capture_writer, capture_path, capture_start_time
    if capture_active:
        return capture_path

    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    capture_path = output_dir / f"ground_truth_{ts_tag}.csv"

    capture_file = open(capture_path, "w", newline="")
    capture_writer = csv.writer(capture_file)
    _write_header(capture_writer)
    capture_file.flush()
    capture_active = True
    capture_start_time = time.time()
    return capture_path


def stop_capture() -> None:
    global capture_active, capture_file, capture_writer
    if not capture_active:
        return

    capture_active = False
    if capture_file is not None:
        capture_file.flush()
        try:
            os.fsync(capture_file.fileno())
        except OSError:
            pass
        capture_file.close()
    capture_file = None
    capture_writer = None


def write_beat_row(ibi: float, instant_bpm: float, avg_bpm: float) -> None:
    global capture_active
    now_ts = f"{time.time():.3f}"
    row = [now_ts, f"{ibi:.4f}", f"{instant_bpm:.2f}", f"{avg_bpm:.2f}"]
    if capture_active and capture_writer is not None:
        capture_writer.writerow(row)
        if capture_file is not None:
            capture_file.flush()


def render_status(message: str) -> None:
    global latest_display
    if not message:
        message = "idle"
    clear_tail = ""
    if len(latest_display) > len(message):
        clear_tail = " " * (len(latest_display) - len(message))
    sys.stdout.write(f"\r{message}{clear_tail}")
    sys.stdout.flush()
    latest_display = message


def build_status_line() -> str:
    global valid_beats, rejected_beats, capture_start_time
    if capture_active and capture_start_time is not None:
        elapsed = time.time() - capture_start_time
        return (
            f"REC ON  BUSY CAPTURING  beats={valid_beats}  rejected={rejected_beats}  "
            f"elapsed={elapsed:5.1f}s"
        )
    return f"REC OFF  beats={valid_beats}  rejected={rejected_beats}"


def beat() -> None:
    global last_beat_time, last_report_time, valid_beats, rejected_beats
    now = time.time()

    if last_beat_time is not None:
        ibi = now - last_beat_time
        instant_bpm = 60.0 / ibi

        if not (MIN_BPM <= instant_bpm <= MAX_BPM):
            rejected_beats += 1
            render_status(f"[reject] ibi={ibi:.3f}s bpm={instant_bpm:.1f} (out of range)  {build_status_line()}")
        else:
            ibi_history.append(ibi)
            avg_bpm = 60.0 / (sum(ibi_history) / len(ibi_history))
            write_beat_row(ibi, instant_bpm, avg_bpm)
            valid_beats += 1
            last_report_time = now
            render_status(
                f"ts={now:.3f}  instant={instant_bpm:6.1f} bpm  avg({len(ibi_history)})={avg_bpm:6.1f} bpm  {build_status_line()}"
            )

    last_beat_time = now


def watchdog() -> None:
    """Periodic check for signal dropout (finger lifted, bad contact, etc)."""
    if time.time() - last_report_time > STALE_WARN_S:
        render_status(f"[warn] no valid beat in {time.time() - last_report_time:.1f}s -- check sensor contact  {build_status_line()}")


def _setup_keyboard() -> None:
    global stdin_fd, stdin_old_attrs
    if msvcrt is not None:
        return
    try:
        import termios
        import tty
    except ImportError:
        return

    if not sys.stdin.isatty():
        return

    fd = sys.stdin.fileno()
    try:
        stdin_old_attrs = termios.tcgetattr(fd)
    except Exception:
        return

    new_attrs = stdin_old_attrs[:]
    new_attrs[3] &= ~(termios.ICANON | termios.ECHO)
    new_attrs[6][termios.VMIN] = 0
    new_attrs[6][termios.VTIME] = 0
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new_attrs)
        stdin_fd = fd
    except Exception:
        stdin_fd = None
        stdin_old_attrs = None


def _restore_keyboard() -> None:
    global stdin_fd, stdin_old_attrs
    if msvcrt is not None or stdin_fd is None or stdin_old_attrs is None:
        return
    try:
        import termios
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, stdin_old_attrs)
    except Exception:
        pass
    stdin_fd = None
    stdin_old_attrs = None


def poll_key() -> Optional[str]:
    if msvcrt is not None:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            # ignore extended keys like arrows
            if ch in ("\x00", "\xe0"):
                msvcrt.getwch()
                return None
            return ch.lower()
        return None

    try:
        import select

        if stdin_fd is None:
            return None

        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return None
        ch = sys.stdin.read(1)
        return ch.lower() if ch else None
    except Exception:
        return None


sensor.when_activated = beat

try:
    _setup_keyboard()
    print("Heartbeat capture ready. Press 'c' to start/stop recording, 'q' to quit.")
    while True:
        key = poll_key()
        if key == "c":
            if capture_active:
                stop_capture()
                render_status(f"REC OFF  capture stopped  {build_status_line()}")
            else:
                path = start_capture()
                render_status(f"REC ON   capture started  {build_status_line()}  file={path.name if path else 'unknown'}")
        elif key == "q":
            break

        watchdog()
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    stop_capture()
    _restore_keyboard()
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(f"Stopped. Latest capture written to {capture_path or output_dir / 'ground_truth_<timestamp>.csv'}")