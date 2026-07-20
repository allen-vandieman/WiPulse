# Setup CSI extractor

## Receiver RX

### Load

```bash
source ~/nexmon/setup_env.sh

cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189/nexmon_csi

make -f Makefile.rpi reload-full

sudo ifconfig wlan0 up
sudo nexutil -Iwlan0 -m1

CSIPARAMS=$(makecsiparams -c 36/80 -C 1 -N 1 -m 02:11:22:33:44:55)
sudo nexutil -Iwlan0 -s500 -b -l34 -v$CSIPARAMS

sudo tcpdump -i wlan0 udp port 5500 -w csi_capture.pcap
```

### Unload

```bash
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189/nexmon_csi

make -f Makefile.rpi restore-wifi

sudo systemctl restart wpa_supplicant
sudo systemctl restart NetworkManager
```

### Verify

```bash
nexutil -m
sudo tcpdump -i wlan0
echo "Generated params: $CSIPARAMS"
```