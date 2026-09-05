import csv
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import max30102

try:
    import msvcrt
except ImportError:
    msvcrt = None


# ============================================================
# HEART RATE SETTINGS
# ============================================================

MIN_BPM = 30
MAX_BPM = 220

AVG_WINDOW = 6
STALE_WARN_S = 5.0

# Prevent unrealistically fast duplicate detections
MIN_BEAT_INTERVAL = 60.0 / MAX_BPM

# Number of IR samples used for beat detection
SIGNAL_WINDOW = 100


# ============================================================
# MAX30102 SENSOR
# ============================================================

print("Initializing MAX30102...")

sensor = max30102.MAX30102()

print("MAX30102 initialized.")


# ============================================================
# HEARTBEAT STATE
# ============================================================

last_beat_time: Optional[float] = None
last_report_time = time.time()

ibi_history = deque(maxlen=AVG_WINDOW)

signal_history = deque(maxlen=SIGNAL_WINDOW)

previous_value = None
previous_derivative = 0.0

valid_beats = 0
rejected_beats = 0


# ============================================================
# OUTPUT / CAPTURE STATE
# ============================================================

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

capture_active = False
capture_file: Optional[object] = None
capture_writer: Optional[csv.writer] = None
capture_path: Optional[Path] = None

latest_display = ""

stdin_fd: Optional[int] = None
stdin_old_attrs = None

capture_start_time: Optional[float] = None


# ============================================================
# CSV
# ============================================================

def _write_header(writer):
    writer.writerow([
        "ts",
        "ibi_s",
        "instant_bpm",
        "avg_bpm"
    ])


def start_capture() -> Optional[Path]:
    global capture_active
    global capture_file
    global capture_writer
    global capture_path
    global capture_start_time

    if capture_active:
        return capture_path

    ts_tag = time.strftime("%Y%m%d_%H%M%S")

    capture_path = (
        output_dir /
        f"ground_truth_{ts_tag}.csv"
    )

    capture_file = open(
        capture_path,
        "w",
        newline=""
    )

    capture_writer = csv.writer(capture_file)

    _write_header(capture_writer)

    capture_file.flush()

    capture_active = True
    capture_start_time = time.time()

    return capture_path


def stop_capture() -> None:

    global capture_active
    global capture_file
    global capture_writer

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


def write_beat_row(
    ibi: float,
    instant_bpm: float,
    avg_bpm: float
) -> None:

    now_ts = f"{time.time():.3f}"

    row = [
        now_ts,
        f"{ibi:.4f}",
        f"{instant_bpm:.2f}",
        f"{avg_bpm:.2f}"
    ]

    if capture_active and capture_writer is not None:

        capture_writer.writerow(row)

        if capture_file is not None:
            capture_file.flush()


# ============================================================
# DISPLAY
# ============================================================

def render_status(message: str) -> None:

    global latest_display

    if not message:
        message = "idle"

    clear_tail = ""

    if len(latest_display) > len(message):
        clear_tail = (
            " " *
            (len(latest_display) - len(message))
        )

    sys.stdout.write(
        f"\r{message}{clear_tail}"
    )

    sys.stdout.flush()

    latest_display = message


def build_status_line() -> str:

    if (
        capture_active and
        capture_start_time is not None
    ):

        elapsed = (
            time.time() -
            capture_start_time
        )

        return (
            f"REC ON  "
            f"beats={valid_beats}  "
            f"rejected={rejected_beats}  "
            f"elapsed={elapsed:5.1f}s"
        )

    return (
        f"REC OFF  "
        f"beats={valid_beats}  "
        f"rejected={rejected_beats}"
    )


# ============================================================
# BEAT PROCESSING
# ============================================================

def register_beat() -> None:

    global last_beat_time
    global last_report_time
    global valid_beats
    global rejected_beats

    now = time.time()

    if last_beat_time is not None:

        ibi = now - last_beat_time

        if ibi <= 0:
            return

        instant_bpm = 60.0 / ibi

        if not (
            MIN_BPM <=
            instant_bpm <=
            MAX_BPM
        ):

            rejected_beats += 1

            render_status(
                f"[reject] "
                f"ibi={ibi:.3f}s "
                f"bpm={instant_bpm:.1f} "
                f"{build_status_line()}"
            )

            return

        ibi_history.append(ibi)

        avg_ibi = (
            sum(ibi_history) /
            len(ibi_history)
        )

        avg_bpm = 60.0 / avg_ibi

        write_beat_row(
            ibi,
            instant_bpm,
            avg_bpm
        )

        valid_beats += 1

        last_report_time = now

        render_status(
            f"ts={now:.3f}  "
            f"instant={instant_bpm:6.1f} bpm  "
            f"avg({len(ibi_history)})="
            f"{avg_bpm:6.1f} bpm  "
            f"{build_status_line()}"
        )

    last_beat_time = now


# ============================================================
# SIMPLE PEAK DETECTOR
# ============================================================

def process_ir_sample(ir_value: int) -> None:

    global previous_value
    global previous_derivative

    signal_history.append(ir_value)

    if len(signal_history) < 20:
        previous_value = ir_value
        return

    signal = np.array(
        signal_history,
        dtype=float
    )

    baseline = np.mean(signal)

    centered = (
        ir_value -
        baseline
    )

    if previous_value is None:
        previous_value = centered
        return

    derivative = (
        centered -
        previous_value
    )

    # Detect a local maximum:
    # derivative changes from positive to negative

    peak_detected = (
        previous_derivative > 0 and
        derivative <= 0
    )

    # Dynamic threshold based on signal variation

    threshold = (
        np.std(signal) *
        0.5
    )

    if (
        peak_detected and
        centered > threshold
    ):

        now = time.time()

        if (
            last_beat_time is None or
            now - last_beat_time >=
            MIN_BEAT_INTERVAL
        ):

            register_beat()

    previous_derivative = derivative
    previous_value = centered


# ============================================================
# SENSOR LOOP
# ============================================================

def read_sensor() -> None:

    while True:

        try:

            red, ir = sensor.read_sequential()

            for ir_value in ir:

                process_ir_sample(ir_value)

        except Exception as e:

            render_status(
                f"[sensor error] {e}"
            )

            time.sleep(1.0)


# ============================================================
# WATCHDOG
# ============================================================

def watchdog() -> None:

    elapsed = (
        time.time() -
        last_report_time
    )

    if elapsed > STALE_WARN_S:

        render_status(
            f"[warn] no valid beat in "
            f"{elapsed:.1f}s "
            f"-- check sensor contact  "
            f"{build_status_line()}"
        )


# ============================================================
# KEYBOARD
# ============================================================

def _setup_keyboard() -> None:

    global stdin_fd
    global stdin_old_attrs

    if msvcrt is not None:
        return

    try:
        import termios
    except ImportError:
        return

    if not sys.stdin.isatty():
        return

    fd = sys.stdin.fileno()

    try:
        stdin_old_attrs = (
            termios.tcgetattr(fd)
        )
    except Exception:
        return

    new_attrs = stdin_old_attrs[:]

    new_attrs[3] &= ~(
        termios.ICANON |
        termios.ECHO
    )

    new_attrs[6][termios.VMIN] = 0
    new_attrs[6][termios.VTIME] = 0

    try:

        termios.tcsetattr(
            fd,
            termios.TCSADRAIN,
            new_attrs
        )

        stdin_fd = fd

    except Exception:

        stdin_fd = None
        stdin_old_attrs = None


def _restore_keyboard() -> None:

    global stdin_fd
    global stdin_old_attrs

    if (
        msvcrt is not None or
        stdin_fd is None or
        stdin_old_attrs is None
    ):
        return

    try:

        import termios

        termios.tcsetattr(
            stdin_fd,
            termios.TCSADRAIN,
            stdin_old_attrs
        )

    except Exception:
        pass

    stdin_fd = None
    stdin_old_attrs = None


def poll_key() -> Optional[str]:

    if msvcrt is not None:

        if msvcrt.kbhit():

            ch = msvcrt.getwch()

            if ch in (
                "\x00",
                "\xe0"
            ):
                msvcrt.getwch()
                return None

            return ch.lower()

        return None

    try:

        import select

        if stdin_fd is None:
            return None

        if not select.select(
            [sys.stdin],
            [],
            [],
            0.0
        )[0]:

            return None

        ch = sys.stdin.read(1)

        return (
            ch.lower()
            if ch
            else None
        )

    except Exception:
        return None


# ============================================================
# MAIN
# ============================================================

try:

    _setup_keyboard()

    sensor_thread = threading.Thread(
        target=read_sensor,
        daemon=True
    )

    sensor_thread.start()

    print(
        "MAX30102 heartbeat capture ready.\n"
        "Place your finger on the sensor.\n"
        "Press 'c' to start/stop recording.\n"
        "Press 'q' to quit."
    )

    while True:

        key = poll_key()

        if key == "c":

            if capture_active:

                stop_capture()

                render_status(
                    "REC OFF  capture stopped  "
                    f"{build_status_line()}"
                )

            else:

                path = start_capture()

                render_status(
                    "REC ON   capture started  "
                    f"{build_status_line()}  "
                    f"file="
                    f"{path.name if path else 'unknown'}"
                )

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

    print(
        "Stopped. Latest capture written to "
        f"{capture_path or output_dir / 'ground_truth_<timestamp>.csv'}"
    )