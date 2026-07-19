# Setup CSI

Content

1. [Find Channel](#find-channel)
1. [TX](#transmitter)
1. [RX - tester](#receiver)
1. [Restore](#restore)
1. [RX - Extractor](#extractor)
    1. [Boot](#boot)
    1. [Monitor Mode](#monitor-mode)
    1. [Configure](#configure)
    1. [Capture](#capture)
    1. [Unload](#unload)
1. [Errors](#errors)
    1. [Transmitter](#transmiter)
    1. [Extractor](#extractor-1)
        1. [Debug Monitor](#debug-monitor)
    1. [Common](#common)

## Find Channel

```bash
# Use on managed device
nmcli device wifi list

# Or airmon
sudo airodump-ng wlan0mon

# Search specific channel
sudo airodump-ng --channel 36 wlan0mon
```

## Transmitter

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
# Run twice for loading error
sudo airmon-ng start wlan0
sudo iw dev wlan0mon set channel 36 80MHz

iw dev wlan0mon info

sudo python3 optimized-fast-sender.py
```

## Receiver

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
# Run twice for loading error
sudo airmon-ng start wlan0
sudo iw dev wlan0mon set channel 36 80MHz

iw dev wlan0mon info

sudo python3 rssi-receiver.py
```

## Restore

```bash
sudo airmon-ng stop wlan0mon
sudo systemctl restart wpa_supplicant
sudo systemctl restart NetworkManager
sudo ifconfig wlan0 up

# Best to reboot WIFI AP
sudo reboot now
# Then verify
nmcli con show --active
# Or bring up
sudo nmcli con up AP
```

## Extractor

### Boot

```bash
cd ~/nexmon
source setup_env.sh
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189/nexmon_csi

# Kill wlan0
sudo ifconfig wlan0 down

# Build and install the patched firmware:
make -f Makefile.rpi install-firmware
make -f Makefile.rpi unmanage
make -f Makefile.rpi reload-full
```

### Monitor mode

```bash
# bring up wlan0
sudo ifconfig wlan0 up
# enable monitoring
sudo nexutil -Iwlan0 -m1
# Verify monitor
nexutil -m
# bring up wlan0 (make sure its up)
sudo ifconfig wlan0 up
# Quick test to see if it is up
sudo tcpdump -i wlan0

# If tcpdump fail, wlan0 was not working proper before Makefile.rpi reload-full
```

### Configure

```bash
#CSIPARAMS=$(makecsiparams -c 1/20 -C 1 -N 1)
#CSIPARAMS=$(makecsiparams -c 36/80 -C 1 -N 1 -m b8:27:eb:74:f2:6c,02:11:22:33:44:55)
# Defenitly use the mac i create in my script when i create the frames !!!!!!!
CSIPARAMS=$(makecsiparams -c 36/80 -C 1 -N 1 -m 02:11:22:33:44:55)
echo "Generated params: $CSIPARAMS"
sudo nexutil -Iwlan0 -s500 -b -l34 -v$CSIPARAMS
```

### Capture

```bash
# Move to working folder
cd /home/allen/work

# All data
sudo tcpdump -i wlan0

# Port 5500 data
sudo tcpdump -i wlan0 udp port 5500

# More port 5500
sudo tcpdump -i wlan0 dst port 5500 -vv

# To file
sudo tcpdump -i wlan0 udp port 5500 -w csi_capture.pcap
```

### Python Stuff

```bash
# Create python enviroment
python3 -m venv csi_env
# Activate
source csi_env/bin/activate

# Then
cd WiPulse/claude/csi_analysis/


# Do stuff
pip install --upgrade pip
pip install "numpy>=2.0" "scapy>=2.5" "pandas>=2.2"
python3 -c "import numpy, scapy, pandas; print(f'NumPy: {numpy.__version__}\nScapy: {scapy.__version__}\nPandas: {pandas.__version__}')"

python3 main.py ../../csi_capture.pcap

# Leave enviroment
deactivate
```

### Unload

```bash
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189/nexmon_csi

make -f Makefile.rpi restore-wifi

sudo systemctl restart wpa_supplicant
sudo systemctl restart NetworkManager
```

## Errors

### Transmiter

```bash
# Monitor not working
sudo iw dev wlan0 set power_save off
sudo systemctl stop NetworkManager
sudo pkill wpa_supplicant

# Wifi not working
sudo systemctl restart wpa_supplicant
sudo systemctl restart NetworkManager
```

### Extractor

#### Debug monitor

```bash
# Disable Power save
sudo iw dev wlan0 set power_save off
# RF-kill status
sudo rfkill unblock all
sudo nmcli radio wifi on
sudo nmcli device wifi rescan
sleep 3
nmcli device wifi list
sudo systemctl restart wpa_supplicant
sudo systemctl restart NetworkManager

# Setup monitor manual without filter
sudo nexutil -k6/20          # match AP_CHANNEL above
sudo nexutil -m1
iw dev
```

### Common

```bash
# Check RF-kill status
# list all
rfkill list
# unblock wifi
sudo rfkill unblock wifi
# or unblock all
sudo rfkill unblock all

#Ensure no service is blocking it
sudo systemctl stop wpa_supplicant
sudo systemctl stop dhcpcd
sudo systemctl status wpa_supplicant
sudo systemctl status dhcpcd
```

- More

```bash
# More options

nmcli device wifi rescan
nmcli device wifi list

# Querry device
nmcli radio

# Switch device on
nmcli radio wifi on

# ! run in order
rfkill unblock wifi
nmcli radio wifi on
nmcli device wifi rescan
nmcli device wifi list

# On error force rescan
sudo nmcli device wifi rescan
sleep 3
nmcli device wifi list
```