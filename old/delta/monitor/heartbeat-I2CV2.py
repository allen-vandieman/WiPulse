#!/usr/bin/env python3

import csv
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

try:
    import msvcrt
except ImportError:
    msvcrt = None

from DFRobot_BloodOxygen_S import *


# ============================================================
# SENSOR CONFIGURATION
# ============================================================

I2C_BUS = 0x01
I2C_ADDRESS = 0x57

MIN_BPM = 30
MAX_BPM = 220

AVG_WINDOW = 6
STALE_WARN_S = 5.0

# How often to ask the sensor for its latest result.
POLL_INTERVAL = 0.25


# ============================================================
# CREATE SENSOR
# ============================================================

sensor = DFRobot_BloodOxygen_S_i2c(
    I2C_BUS,
    I2C_ADDRESS
)


# ============================================================
# HEART RATE STATE
# ============================================================

bpm_history = deque(maxlen=AVG_WINDOW)

last_valid_measurement_time: Optional[float] = None

valid_measurements = 0
rejected_measurements = 0

last_sensor_bpm: Optional[float] = None
last_sensor_spo2: Optional[float] = None


# ============================================================
# OUTPUT / CAPTURE STATE
# ============================================================

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

capture_active = False

capture_file: Optional[object] = None
capture_writer: Optional[csv.writer] = None
capture_path: Optional[Path] = None

capture_start_time: Optional[float] = None

latest_display = ""

stdin_fd: Optional[int] = None
stdin_old_attrs = None

state_lock = threading.Lock()

running = True


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

    with state_lock:

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

    with state_lock:

        if not capture_active:
            return

        capture_active = False

        if capture_file is not None:

            capture_file.flush()

            try:
                os.fsync(
                    capture_file.fileno()
                )
            except OSError:
                pass

            capture_file.close()

        capture_file = None
        capture_writer = None


def write_measurement(
    timestamp: float,
    bpm: float,
    avg_bpm: float
) -> None:

    if bpm <= 0:
        return

    # Estimated beat interval derived from sensor BPM.
    # This is NOT a directly measured beat-to-beat IBI.
    estimated_ibi = 60.0 / bpm

    row = [
        f"{timestamp:.3f}",
        f"{estimated_ibi:.4f}",
        f"{bpm:.2f}",
        f"{avg_bpm:.2f}"
    ]

    with state_lock:

        if (
            capture_active and
            capture_writer is not None
        ):

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
            f"measurements={valid_measurements}  "
            f"rejected={rejected_measurements}  "
            f"elapsed={elapsed:5.1f}s"
        )

    return (
        f"REC OFF  "
        f"measurements={valid_measurements}  "
        f"rejected={rejected_measurements}"
    )


# ============================================================
# PROCESS SENSOR MEASUREMENT
# ============================================================

def process_measurement(
    bpm: float,
    spo2: float
) -> None:

    global last_valid_measurement_time
    global valid_measurements
    global rejected_measurements
    global last_sensor_bpm
    global last_sensor_spo2

    now = time.time()

    if not (
        MIN_BPM <= bpm <= MAX_BPM
    ):

        rejected_measurements += 1

        render_status(
            f"[reject] "
            f"bpm={bpm:.1f} "
            f"spo2={spo2:.1f} "
            f"{build_status_line()}"
        )

        return

    bpm_history.append(bpm)

    avg_bpm = (
        sum(bpm_history) /
        len(bpm_history)
    )

    last_sensor_bpm = bpm
    last_sensor_spo2 = spo2

    last_valid_measurement_time = now

    valid_measurements += 1

    write_measurement(
        now,
        bpm,
        avg_bpm
    )

    render_status(
        f"ts={now:.3f}  "
        f"HR={bpm:6.1f} bpm  "
        f"avg({len(bpm_history)})="
        f"{avg_bpm:6.1f} bpm  "
        f"SpO2={spo2:5.1f}%  "
        f"{build_status_line()}"
    )


# ============================================================
# SENSOR THREAD
# ============================================================

def sensor_loop() -> None:

    global running

    while running:

        try:

            # Update sensor.heartbeat and sensor.SPO2
            sensor.get_heartbeat_SPO2()

            bpm = sensor.heartbeat
            spo2 = sensor.SPO2

            if bpm is not None:

                process_measurement(
                    float(bpm),
                    float(spo2)
                )

        except Exception as e:

            render_status(
                f"[sensor error] {e}"
            )

        time.sleep(POLL_INTERVAL)


# ============================================================
# WATCHDOG
# ============================================================

def watchdog() -> None:

    if last_valid_measurement_time is None:

        return

    elapsed = (
        time.time() -
        last_valid_measurement_time
    )

    if elapsed > STALE_WARN_S:

        render_status(
            f"[warn] no valid measurement "
            f"in {elapsed:.1f}s "
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
# SENSOR SETUP
# ============================================================

def setup_sensor() -> None:

    print(
        "Initializing DFRobot MAX30102..."
    )

    while not sensor.begin():

        print(
            "Sensor initialization failed. "
            "Checking again in 1 second..."
        )

        time.sleep(1)

    print(
        "Sensor connected."
    )

    print(
        "Starting measurement..."
    )

    sensor.sensor_start_collect()

    time.sleep(1)


# ============================================================
# MAIN
# ============================================================

try:

    setup_sensor()

    _setup_keyboard()

    sensor_thread = threading.Thread(
        target=sensor_loop,
        daemon=True
    )

    sensor_thread.start()

    print(
        "\nDFRobot MAX30102 heartbeat "
        "capture ready."
    )

    print(
        "Place your finger on the sensor."
    )

    print(
        "Press 'c' to start/stop recording."
    )

    print(
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

    running = False

    try:

        sensor.sensor_end_collect()

    except Exception:

        pass

    stop_capture()

    _restore_keyboard()

    sys.stdout.write("\n")
    sys.stdout.flush()

    print(
        "Stopped. Latest capture written to "
        f"{capture_path or output_dir / 'ground_truth_<timestamp>.csv'}"
    )