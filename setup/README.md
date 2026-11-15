# Setup CSI extractor

## Find a nice channel

```bash
# Pre-patch
nmcli device wifi list

# Use airmon on all channels
sudo airodump-ng wlan0mon

# Search specific channel
sudo airodump-ng --channel 36 wlan0mon

# Check just the 5G channels or band a
sudo airodump-ng --band a wlan0mon

# Check specific channels
sudo airodump-ng -c 36,40,44,48 wlan0mon
```

## Transmiter

Setup transmitter and tester Pi

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
# Run twice for loading error
sudo airmon-ng start wlan0
```

Load Pi on that channel
- 80 MHz
```bash
sudo iw dev wlan0mon set channel 36 80MHz

iw dev wlan0mon info
```
- 40 MHz
```bash
sudo iw dev wlan0mon set channel 44 HT40+

iw dev wlan0mon info
```
- 20 MHZ
```bash
sudo iw dev wlan0mon set channel 40

iw dev wlan0mon info
```

Sender

```bash
sudo python3 optimized-fast-sender.py
```

Tester

```bash
sudo python3 rssi-receiver.py
```

## Receiver RX

### Setup

```bash
sudo ifconfig wlan0 up
sudo nexutil -Iwlan0 -m1
```

### Load
- 80 MHz
```bash
CSIPARAMS=$(makecsiparams -c 36/80 -C 1 -N 1 -m 02:11:22:33:44:55)
sudo nexutil -Iwlan0 -s500 -b -l34 -v$CSIPARAMS
```
- 40 MHz
```bash
CSIPARAMS=$(makecsiparams -c 44/40 -C 1 -N 1 -m 02:11:22:33:44:55)
sudo nexutil -Iwlan0 -s500 -b -l34 -v$CSIPARAMS
```
- 20 MHz
```bash
CSIPARAMS=$(makecsiparams -c 40/20 -C 1 -N 1 -m 02:11:22:33:44:55)
sudo nexutil -Iwlan0 -s500 -b -l34 -v$CSIPARAMS
```

### Capture

Raw capture

```bash
sudo tcpdump -i wlan0 udp port 5500 -w csi_capture.pcap
```

or use the [FALCON application](/falcon/README.md).

```bash
cd ~/work

source env/bin/activate

cd WiPulse/falcon
```

#### Live Monitor

```bash
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80 --capture-mode complex
# press 'c' to start recording, 'c' again to stop, 'q' to quit
```

#### Servo Monitor

```bash
sudo python servo_frequency_monitor.py   --iface wlan0   --chip 43455c0   --bw 80
```

- Dashboard

```
http://25.0.0.77:8080
```

#### Serial Ground truth recorder

```bash
python pi_serial_recorder.py
```

#### Quick start both monitor and serial recoder

```bash
./start_wipulse_capture.sh
```

### Verify RX

```bash
nexutil -m
sudo tcpdump -i wlan0
echo "Generated params: $CSIPARAMS"
lsmod | grep brcm
iw dev
```

### Reload RX driver

```bash
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189/nexmon_csi

make -f Makefile.rpi reload-full
```

## Restore TX and Tester

```bash
# Restore lite
sudo airmon-ng stop wlan0mon
sudo systemctl restart wpa_supplicant
sudo systemctl restart NetworkManager
sudo ifconfig wlan0 up


# Restore medium
sudo rfkill unblock all
sudo nmcli radio wifi on
sudo nmcli device wifi rescan
sleep 3
nmcli device wifi list
sudo systemctl restart wpa_supplicant
sudo systemctl restart NetworkManager

# Restore Heavy
sudo reboot now
```
