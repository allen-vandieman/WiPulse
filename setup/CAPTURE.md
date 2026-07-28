## Woring Folder and envirment

```bash
cd work

source env/bin/activate

cd WiPulse/app
```

## Live monitoring

For a real-time terminal dashboard instead of capture-then-analyze:

```bash
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80
```

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