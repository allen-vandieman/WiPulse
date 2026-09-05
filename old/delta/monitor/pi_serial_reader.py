"""
Reads agreed heart-rate values sent by the Arduino over USB serial.

The Arduino only sends a line when both sensors agree, formatted as:
    HR,<bpm>\n

Install pyserial first:
    pip3 install pyserial
"""

import serial
import time

PORT = "/dev/ttyACM0"  # run `ls /dev/tty*` after plugging in the Arduino to confirm
BAUD = 115200


def main():
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        print(f"Listening on {PORT} at {BAUD} baud...")
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("HR,"):
                continue
            try:
                bpm = int(line.split(",")[1])
            except (IndexError, ValueError):
                continue
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Agreed heart rate: {bpm} bpm")
            # TODO: do whatever you need with bpm here
            # (write to a log, push to a dashboard, trigger something, etc.)


if __name__ == "__main__":
    main()
