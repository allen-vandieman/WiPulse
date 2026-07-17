# Setup CSI

Content

1. [Find Channel](#find-channel)
1. [TX](#transmitter)
1. [RX - tester](#receiver)
1. [Restore](#restore)
1. [RX - Extractor](#extractor)
    1. [Boot](#boot)
    1. [Monitor Mode](#monitor-mode)
    1. [Debug Monitor](#debug-monitor)
    1. [Configure](#configure)
    1. [Capture](#capture)
    1. [Unload](#unload)

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
#sudo iw dev wlan0 set power_save off
#sudo systemctl stop NetworkManager
#sudo pkill wpa_supplicant
sudo airmon-ng check kill
sudo airmon-ng start wlan0
sudo iw dev wlan0mon set channel 36 80MHz

iw dev wlan0mon info

sudo python3 optimized-fast-sender.py
```

## Receiver

```bash
sudo airmon-ng check kill
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

### Debug monitor

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

### Unload

```bash
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189/nexmon_csi

make -f Makefile.rpi restore-wifi

sudo systemctl restart wpa_supplicant
sudo systemctl restart NetworkManager
```