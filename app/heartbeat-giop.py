import csv
import time
from collections import deque
from pathlib import Path
from gpiozero import DigitalInputDevice
from signal import pause

MIN_BPM, MAX_BPM = 30, 220
BOUNCE_TIME = 0.15          # debounce: min 150ms between accepted beats
AVG_WINDOW = 6               # beats to average for a smoothed bpm
STALE_WARN_S = 5.0           # warn if no beat seen for this long

sensor = DigitalInputDevice(15, bounce_time=BOUNCE_TIME)

last_beat_time = None
last_report_time = time.time()
ibi_history = deque(maxlen=AVG_WINDOW)

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)
csv_path = output_dir / "ground_truth.csv"

csvfile = open(csv_path, "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["timestamp", "ibi_s", "instant_bpm", "avg_bpm"])


def beat():
    global last_beat_time, last_report_time
    now = time.time()

    if last_beat_time is not None:
        ibi = now - last_beat_time
        instant_bpm = 60.0 / ibi

        if not (MIN_BPM <= instant_bpm <= MAX_BPM):
            # Implausible interval -- almost certainly a noise glitch
            # or lost/poor sensor contact. Don't let it corrupt the
            # ground truth or the rolling average.
            print(f"[reject] {now:.3f}  ibi={ibi:.3f}s  bpm={instant_bpm:.1f} (out of range)")
        else:
            ibi_history.append(ibi)
            avg_bpm = 60.0 / (sum(ibi_history) / len(ibi_history))
            writer.writerow([f"{now:.3f}", f"{ibi:.4f}", f"{instant_bpm:.2f}", f"{avg_bpm:.2f}"])
            csvfile.flush()
            print(f"{now:.3f}  instant={instant_bpm:6.1f} bpm  avg({len(ibi_history)})={avg_bpm:6.1f} bpm")
            last_report_time = now

    last_beat_time = now


def watchdog():
    """Periodic check for signal dropout (finger lifted, bad contact, etc)."""
    if time.time() - last_report_time > STALE_WARN_S:
        print(f"[warn] no valid beat in {time.time() - last_report_time:.1f}s -- check sensor contact")


sensor.when_activated = beat

try:
    while True:
        watchdog()
        time.sleep(1.0)
except KeyboardInterrupt:
    print(f"\nStopped. Wrote ground truth to {csvfile.name}")
finally:
    csvfile.close()