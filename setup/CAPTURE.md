## Woring Folder and envirment

```bash
cd ~/work

source env/bin/activate

cd WiPulse/app
```

### Recording live CSI (for ML)

Press `c` in the dashboard to start recording every decoded CSI packet to a
timestamped CSV; press `c` again to stop. This lets you capture data
live -- no separate tcpdump-then-decode step -- and, by default, writes to
the same `output/` folder that `heartbeat-giop.py` writes
`ground_truth.csv` to, so both data streams land side by side, ready to be
paired up for the ML step:

```bash
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80 --capture-mode complex
# press 'c' to start recording, 'c' again to stop, 'q' to quit
```

## Ground truth data

```bash
python3 heartbeat-giop.py
```

## Baseline heart-rate check

Once you have both a CSI capture (`output/csi_capture_*.csv` from
`live_monitor.py`) and a ground-truth recording (`output/ground_truth.csv`
from `heartbeat-giop.py`) covering the same session, `align_and_baseline.py`
answers the key question before you build anything fancier: **does the CSI
signal actually contain a usable heart-rate signature?**

```bash
python align_and_baseline.py \
    --csi-csv output/csi_capture_20260728_101500.csv \
    --gt-csv output/ground_truth.csv \
    --out-dir results/baseline_ml
```

- Or use phase data

```bash
# 1. record a fresh capture with full I/Q (phase needs this -- an
#    amplitude-only or raw sc*_pha capture doesn't carry enough
#    information to remove the SFO/CFO phase ramp)
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80 \
    --capture-mode complex
# press 'c' to start/stop recording, 'q' to quit

# 2. run the baseline check on phase instead of amplitude
python align_and_baseline.py \
    --csi-csv output/csi_capture_<new_timestamp>.csv \
    --gt-csv output/ground_truth.csv \
    --signal-type phase \
    --search-subcarriers \
    --out-dir results/baseline_ml_phase
```

`--signal-type phase` unwraps phase across the subcarrier axis per packet,
then removes the linear SFO/CFO ramp with a per-packet least-squares fit
against subcarrier index (`align_and_baseline.sanitize_phase`) -- what's
left is (approximately) the true channel phase, everything downstream
(resampling, band-pass, windowed FFT, ground-truth matching, plots) is
identical to the amplitude path.

### Searching all subcarriers

`--search-subcarriers` evaluates every subcarrier individually (ranked by
a `corr / (1 + mae)` score) and also the average of the top 1/3/5/10/20
performers, which is usually a better-conditioned signal than any single
subcarrier alone:

```bash
python align_and_baseline.py \
  --csi-csv output/csi_capture.csv \
  --gt-csv output/ground_truth.csv \
  --search-subcarriers \
  --out-dir results/baseline_ml
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