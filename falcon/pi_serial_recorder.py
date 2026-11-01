"""
Reads agreed heart-rate values sent by the Arduino over USB serial.

The Arduino only sends a line when both sensors agree, formatted as:
    HR,<bpm>\n

Records ground truth BPM values to a CSV file with timestamps.

Install pyserial first:
    pip3 install pyserial

Port selection
--------------
Instead of a hardcoded PORT, this script now auto-detects a usable
serial port at startup, and will re-detect / reconnect if the Arduino
is unplugged and replugged (which can flip it between /dev/ttyACM0,
/dev/ttyACM1, etc.).

You can still force a specific port:

    python3 pi_serial_recorder.py --port /dev/ttyACM1

Or just list what's currently available:

    python3 pi_serial_recorder.py --list-ports
"""

import argparse
import copy
import csv
import os
import signal
import sys
import time
from pathlib import Path

import serial
import serial.tools.list_ports

# ============================================================
# CONFIGURATION
# ============================================================

BAUD = 115200

# Output directory for ground truth CSVs
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Heart rate range validation (sanity check)
MIN_BPM = 30
MAX_BPM = 220

# How long to wait (seconds) for a matching port to show up before
# giving up, and how often to poll while waiting.
PORT_WAIT_TIMEOUT = 30.0
PORT_POLL_INTERVAL = 1.0

# How long to wait before trying to reconnect after the serial link
# drops mid-session (e.g. Arduino unplugged/replugged).
RECONNECT_DELAY = 2.0


# ============================================================
# PORT DETECTION
# ============================================================

def list_available_ports():
    """Return the list of serial ports currently visible to the OS."""
    return list(serial.tools.list_ports.comports())


def print_available_ports(ports=None):
    ports = ports if ports is not None else list_available_ports()
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for p in ports:
        extra = p.description or "unknown device"
        if p.manufacturer:
            extra += f" ({p.manufacturer})"
        print(f"  {p.device:20s} {extra}")


def find_serial_port(preferred=None, ports=None):
    """
    Pick the best candidate serial port.

    Priority:
        1. `preferred`, if it currently exists on the system.
        2. Any port whose description/manufacturer mentions "arduino".
        3. Any port matching common patterns (ttyACM*, ttyUSB*, usbmodem*,
           usbserial*, COM* on Windows), lowest-numbered first.

    Returns the device path/string, or None if nothing usable was found.
    """
    ports = ports if ports is not None else list_available_ports()

    if preferred:
        if any(p.device == preferred for p in ports):
            return preferred
        # Fall back to checking the filesystem directly (list_ports can
        # occasionally miss devices depending on platform/driver).
        if Path(preferred).exists():
            return preferred
        return None

    # Prefer anything self-identifying as an Arduino.
    for p in ports:
        text = f"{p.description or ''} {p.manufacturer or ''}".lower()
        if "arduino" in text:
            return p.device

    # Fall back to generic USB-serial naming patterns.
    patterns = ("ttyacm", "ttyusb", "usbmodem", "usbserial", "com")
    candidates = sorted(
        (p.device for p in ports if any(pat in p.device.lower() for pat in patterns))
    )
    if candidates:
        return candidates[0]

    return None


def wait_for_port(preferred=None, timeout=PORT_WAIT_TIMEOUT):
    """
    Block (with periodic status prints) until a usable serial port
    appears, or `timeout` seconds elapse.

    Returns the chosen port device string, or None on timeout.
    """
    deadline = time.time() + timeout if timeout else None
    last_message_time = 0

    while True:
        port = find_serial_port(preferred=preferred)
        if port:
            return port

        now = time.time()
        if now - last_message_time >= PORT_POLL_INTERVAL:
            target = preferred or "an Arduino / USB-serial device"
            print(f"Waiting for {target} to appear... (plug it in now)")
            last_message_time = now

        if deadline and now >= deadline:
            return None

        time.sleep(PORT_POLL_INTERVAL)


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
# ARGUMENTS
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Record Arduino ground-truth heart rate over serial.",
    )
    p.add_argument(
        "--port",
        default=None,
        help=(
            "Serial port to use, e.g. /dev/ttyACM0 or /dev/ttyACM1. "
            "If omitted, the script auto-detects a port (preferring "
            "anything that identifies itself as an Arduino)."
        ),
    )
    p.add_argument(
        "--baud",
        type=int,
        default=BAUD,
        help="Baud rate (default: %(default)s)",
    )
    p.add_argument(
        "--port-timeout",
        type=float,
        default=PORT_WAIT_TIMEOUT,
        help=(
            "Seconds to wait for the requested/auto-detected port to "
            "appear before giving up. 0 waits forever. "
            "(default: %(default)s)"
        ),
    )
    p.add_argument(
        "--list-ports",
        action="store_true",
        help="List available serial ports and exit.",
    )
    return p.parse_args(argv)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.list_ports:
        print_available_ports()
        return

    # Set up signal handler for clean exit
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\nShutting down...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    recorder = GroundTruthRecorder()

    print("Press 'c' to start/stop recording, 'q' to quit.")
    print("-" * 60)

    # Non-blocking keyboard input setup. Falls back gracefully (keyboard
    # control disabled) on non-UNIX platforms or when stdin isn't a real
    # TTY (e.g. run under systemd, nohup, or with redirected input).
    try:
        import termios
        import tty  # noqa: F401  (imported for parity with original script)
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
        # ------------------------------------------------------------
        # Outer loop: (re)connect to a serial port, run until it drops
        # or the user quits, then try to reconnect (unless quitting).
        # ------------------------------------------------------------
        while running:

            port = wait_for_port(
                preferred=args.port,
                timeout=args.port_timeout,
            )

            if port is None:
                target = args.port or "a serial port"
                print(f"Timed out waiting for {target}. Giving up.")
                print_available_ports()
                break

            print(f"Connecting to {port} at {args.baud} baud...")

            try:
                ser = serial.Serial(port, args.baud, timeout=0.1)
            except serial.SerialException as e:
                print(f"Could not open {port}: {e}")
                time.sleep(RECONNECT_DELAY)
                continue

            with ser:
                print(f"Listening on {port} at {args.baud} baud...")
                ser.reset_input_buffer()

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
                            running = False
                            break

                        # Read from serial (non-blocking with timeout)
                        try:
                            line = ser.readline().decode("utf-8", errors="ignore").strip()
                        except (serial.SerialException, OSError) as e:
                            print(f"\nLost connection to {port}: {e}")
                            break

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

                except KeyboardInterrupt:
                    running = False

            # If we broke out because the link dropped (not because the
            # user quit), pause briefly and try to reconnect. Recording
            # state (and the open CSV) is preserved across reconnects.
            if running:
                print(f"Attempting to reconnect in {RECONNECT_DELAY:.0f}s...")
                time.sleep(RECONNECT_DELAY)

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