# Setup CSI extractor

## Transmiter

Setup transmitter and tester Pi

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
# Run twice for loading error
sudo airmon-ng start wlan0
```
Find a nice channel

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

Load Pi on that channel

```bash
sudo iw dev wlan0mon set channel 36 80MHz

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

### Load

```bash
sudo ifconfig wlan0 up
sudo nexutil -Iwlan0 -m1

CSIPARAMS=$(makecsiparams -c 36/80 -C 1 -N 1 -m 02:11:22:33:44:55)
sudo nexutil -Iwlan0 -s500 -b -l34 -v$CSIPARAMS
```

### Capture

```bash
sudo tcpdump -i wlan0 udp port 5500 -w csi_capture.pcap
```

### Verify

```bash
nexutil -m
sudo tcpdump -i wlan0
echo "Generated params: $CSIPARAMS"
lsmod | grep brcm
iw dev
```

### Reload driver

```bash
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189/nexmon_csi

make -f Makefile.rpi reload-full
```