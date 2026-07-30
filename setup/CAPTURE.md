## Woring Folder and envirment

```bash
cd ~/work

source env/bin/activate

cd WiPulse/app
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
python3 heartbeat-giop.py
```

## Setup

```bash
# Create an enviroment
python3 -m venv csi_env

# Activate the enviroment
source test/bin/activate

# Install pre-requirements to enviroment
pip install -r requirements.txt
```

## Prep data for ML

```bash
python align_and_baseline.py \
    --csi-csv output/csi_capture_20260728_101500.csv \
    --gt-csv output/ground_truth.csv \
    --out-dir results/baseline_ml
```