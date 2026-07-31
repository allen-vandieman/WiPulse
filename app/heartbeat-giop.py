import csv
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
latest_file: Optional[object] = None
capture_writer: Optional[csv.writer] = None
latest_writer: Optional[csv.writer] = None
capture_path: Optional[Path] = None
latest_path: Optional[Path] = None
latest_display = ""
state_lock = threading.Lock()

capture_start_time: Optional[float] = None
valid_beats = 0
rejected_beats = 0


def _write_header(writer):
    writer.writerow(["ts", "ibi_s", "instant_bpm", "avg_bpm"])


def start_capture() -> Optional[Path]:
    global capture_active, capture_file, latest_file, capture_writer, latest_writer, capture_path, latest_path
    if capture_active:
        return capture_path

    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    capture_path = output_dir / f"ground_truth_{ts_tag}.csv"
    latest_path = output_dir / "ground_truth.csv"

    capture_file = open(capture_path, "w", newline="")
    latest_file = open(latest_path, "w", newline="")
    capture_writer = csv.writer(capture_file)
    latest_writer = csv.writer(latest_file)
    _write_header(capture_writer)
    _write_header(latest_writer)
    capture_active = True
    capture_start_time = time.time()
    return capture_path


def stop_capture() -> None:
    global capture_active, capture_file, latest_file, capture_writer, latest_writer
    if not capture_active:
        return

    capture_active = False
    for handle in (capture_file, latest_file):
        if handle is not None:
            handle.flush()
            handle.close()
    capture_file = None
    latest_file = None
    capture_writer = None
    latest_writer = None


def write_beat_row(ibi: float, instant_bpm: float, avg_bpm: float) -> None:
    global capture_active
    now_ts = f"{time.time():.3f}"
    row = [now_ts, f"{ibi:.4f}", f"{instant_bpm:.2f}", f"{avg_bpm:.2f}"]
    if capture_active and capture_writer is not None and latest_writer is not None:
        capture_writer.writerow(row)
        latest_writer.writerow(row)
        if capture_file is not None:
            capture_file.flush()
        if latest_file is not None:
            latest_file.flush()


def render_status(message: str) -> None:
    global latest_display
    if not message:
        message = "idle"
    padding = " " * max(0, len(latest_display) - len(message))
    sys.stdout.write(f"\r{message}{padding}")
    sys.stdout.flush()
    latest_display = message


def build_status_line() -> str:
    global valid_beats, rejected_beats, capture_start_time
    if capture_active and capture_start_time is not None:
        elapsed = time.time() - capture_start_time
        return (
            f"REC ON  beats={valid_beats}  rejected={rejected_beats}  "
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


def poll_key() -> Optional[str]:
    if msvcrt is not None:
        if msvcrt.kbhit():
            return msvcrt.getwch().lower()
        return None

    try:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return None
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            return sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        return None


sensor.when_activated = beat

try:
    print("Heartbeat capture ready. Press 'c' to start/stop recording, 'q' to quit.")
    while True:
        key = poll_key()
        if key == "c":
            if capture_active:
                stop_capture()
                render_status(f"REC OFF  capture stopped  {build_status_line()}")
            else:
                start_capture()
                render_status(f"REC ON   capture started  {build_status_line()}")
        elif key == "q":
            break

        watchdog()
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    stop_capture()
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(f"Stopped. Latest capture written to {latest_path or output_dir / 'ground_truth.csv'}")