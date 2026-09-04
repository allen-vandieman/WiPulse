"""
Reads agreed heart-rate values sent by the Arduino over USB serial.

The Arduino only sends a line when both sensors agree, formatted as:
    HR,<bpm>\n

Records ground truth BPM values to a CSV file with timestamps.

Install pyserial first:
    pip3 install pyserial
"""

import copy
import csv
import os
import signal
import sys
import time
from pathlib import Path

import serial

# ============================================================
# CONFIGURATION
# ============================================================

PORT = "/dev/ttyACM0"  # run `ls /dev/tty*` after plugging in the Arduino to confirm
BAUD = 115200

# Output directory for ground truth CSVs
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Heart rate range validation (sanity check)
MIN_BPM = 30
MAX_BPM = 220

# ============================================================
# GROUND TRUTH RECORDER
# ============================================================

class GroundTruthRecorder:
    """Records agreed BPM values from Arduino to CSV."""

    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None
        self.recording = False
        self.start_time = None
        self.beat_count = 0
        self.rejected_count = 0

    def start_recording(self):
        """Start a new ground truth CSV file."""
        if self.recording:
            return self.csv_path

        ts_tag = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"ground_truth_{ts_tag}.csv"

        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # Write header matching heartbeat-giop.py format
        self.csv_writer.writerow(["ts", "bpm"])
        self.csv_file.flush()

        self.recording = True
        self.start_time = time.time()
        self.beat_count = 0
        self.rejected_count = 0

        return self.csv_path

    def stop_recording(self):
        """Stop recording and close the CSV file."""
        if not self.recording:
            return

        self.recording = False

        if self.csv_file is not None:
            self.csv_file.flush()
            try:
                os.fsync(self.csv_file.fileno())
            except OSError:
                pass
            self.csv_file.close()

        self.csv_file = None
        self.csv_writer = None

    def write_bpm(self, bpm):
        """Write a BPM value to the CSV with timestamp."""
        if not self.recording or self.csv_writer is None:
            return False

        now = time.time()
        self.csv_writer.writerow([f"{now:.3f}", bpm])
        self.csv_file.flush()
        self.beat_count += 1
        return True

    def get_status(self):
        """Get status string for display."""
        if self.recording:
            elapsed = time.time() - self.start_time
            return f"REC ON  beats={self.beat_count}  rejected={self.rejected_count}  elapsed={elapsed:5.1f}s"
        return f"REC OFF  beats={self.beat_count}  rejected={self.rejected_count}"


# ============================================================
# MAIN
# ============================================================

def main():
    # Set up signal handler for clean exit
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\nShutting down...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    recorder = GroundTruthRecorder()

    print(f"Listening on {PORT} at {BAUD} baud...")
    print("Press 'c' to start/stop recording, 'q' to quit.")
    print("-" * 60)

    with serial.Serial(PORT, BAUD, timeout=0.1) as ser:
        # Clear any stale data
        ser.reset_input_buffer()

        # Non-blocking keyboard input setup. Falls back gracefully (keyboard
        # control disabled) on non-UNIX platforms or when stdin isn't a real
        # TTY (e.g. run under systemd, nohup, or with redirected input).
        try:
            import termios
            import tty
        except ImportError:
            termios = None
            tty = None

        stdin_fd = None
        stdin_old_attrs = None

        if termios is not None:
            try:
                fd = sys.stdin.fileno()
                old_attrs = termios.tcgetattr(fd)
                new_attrs = copy.deepcopy(old_attrs)   # deep copy - old_attrs must stay untouched
                new_attrs[3] &= ~(termios.ICANON | termios.ECHO)
                new_attrs[6][termios.VMIN] = 0
                new_attrs[6][termios.VTIME] = 0
                termios.tcsetattr(fd, termios.TCSADRAIN, new_attrs)
                stdin_fd = fd
                stdin_old_attrs = old_attrs
            except Exception:
                stdin_fd = None
                stdin_old_attrs = None

        if stdin_fd is None:
            print("(Interactive keyboard control unavailable - 'c'/'q' won't work here.")
            print(" Use Ctrl+C to stop.)")

        def poll_key():
            """Poll for a keypress without blocking."""
            if stdin_fd is None:
                return None

            try:
                import select
                if not select.select([sys.stdin], [], [], 0.0)[0]:
                    return None
                ch = sys.stdin.read(1)
                return ch.lower() if ch else None
            except Exception:
                return None

        status_line = ""

        try:
            while running:
                # Check for keyboard input
                key = poll_key()
                if key == "c":
                    if recorder.recording:
                        recorder.stop_recording()
                        status_line = f"REC OFF  capture stopped  {recorder.get_status()}"
                    else:
                        path = recorder.start_recording()
                        status_line = f"REC ON   capture started  {recorder.get_status()}  file={path.name if path else 'unknown'}"
                    print(f"\n{status_line}")
                elif key == "q":
                    print("\nQuitting...")
                    break

                # Read from serial (non-blocking with timeout)
                try:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                except serial.SerialException:
                    if not running:
                        break
                    continue

                if not line or not line.startswith("HR,"):
                    continue

                try:
                    bpm = int(line.split(",")[1])
                except (IndexError, ValueError):
                    continue

                # Validate BPM range
                if bpm < MIN_BPM or bpm > MAX_BPM:
                    recorder.rejected_count += 1
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] REJECTED: {bpm} bpm (out of range)  {recorder.get_status()}")
                    continue

                # Record valid BPM
                recorder.write_bpm(bpm)

                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] Agreed heart rate: {bpm} bpm  {recorder.get_status()}")
        finally:
            # Guaranteed to run whether we exit normally, via 'q', Ctrl+C,
            # or an unexpected exception - so we never lose unflushed data
            # or leave the terminal stuck in raw mode.
            if recorder.recording:
                recorder.stop_recording()

            if stdin_old_attrs is not None:
                try:
                    termios.tcsetattr(stdin_fd, termios.TCSADRAIN, stdin_old_attrs)
                except Exception:
                    pass

    print(f"\nStopped. Latest capture written to {recorder.csv_path or OUTPUT_DIR / 'ground_truth_<timestamp>.csv'}")


if __name__ == "__main__":
    main()
