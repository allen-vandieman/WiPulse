## Woring Folder and envirment

```bash
cd ~/work

source env/bin/activate

cd WiPulse/echo
```

### Recording live CSI (for ML)

Press `c` in the dashboard to start recording every decoded CSI packet to a
timestamped CSV; press `c` again to stop. This lets you capture data
live -- no separate tcpdump-then-decode step -- and, by default, writes to
the same `output/` folder that `pi_serial_recorder.py` writes
`ground_truth.csv` to, so both data streams land side by side, ready to be
paired up for the ML step:

```bash
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80 --capture-mode complex
# press 'c' to start recording, 'c' again to stop, 'q' to quit
```

## Ground truth data

```bash
python3 pi_serial_recorder.py
```

## Live monitoring

For a real-time terminal dashboard instead of capture-then-analyze:

```bash
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80
```

Useful flags:
- `--output-dir` -- where captures are written (default: `output/` next to
  `live_monitor.py`, matching `heartbeat-giop.py`).
- `--capture-mode {amplitude,phase,complex}` -- representation to record per
  subcarrier (default `amplitude`; `complex` keeps full I/Q if you need
  phase for a later step).

Live view with filter

```bash
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80 --hampel
```

## Ground truth data

```bash
python3 pi_serial_recorder.py
```

## Setup

```bash
# Create an enviroment
python3 -m venv csi_env

# Activate the enviroment
source csi_env/bin/activate

# Install pre-requirements to enviroment
pip install -r requirements.txt
```
