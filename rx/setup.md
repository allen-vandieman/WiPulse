# RX

Contents

1. [Setup](#setup)
    1. [Patch Kernel](#patch-kernel)
    1. [Install Pre-req](#install-pre-req)
    1. [Python 2.7](#python-27)
1. [Repo](#repo)
    1. [Clone and build](#clone-and-build)
    1. [nexutil](#build-nexutil)
    1. [Fetch firmware](#fetch-firmware)
    1. [Build Utils](#build-utils-build-and-setup-makecsiparams)
    1. [Build and patch firmware](#build-and-patch-firmware)
1. [Monitor mode](#monitor-mode-after-boot)
1. [Configure firmware](#configure)
1. [Unload firmware](#unload)
1. [Errors](#on-error)
    1. [RF Kill](#rf-kill-status)
    1. [Services](#services)

## Setup

### Patch Kernel

```bash
sudo echo "kernel=kernel8.img" | sudo tee -a /boot/firmware/config.txt
sudo reboot
```

### Install pre-req

```bash
sudo apt update -y
sudo apt full-upgrade -y
sudo apt install git libgmp3-dev gawk qpdf bison flex make autoconf libtool texinfo xxd libnl-3-dev libnl-genl-3-dev bc libssl-dev tcpdump -y

sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git libgmp3-dev gawk qpdf bison flex make automake \
  autoconf libtool texinfo libncurses5-dev tcpdump tmux nano vim htop

sudo dpkg --add-architecture armhf
sudo apt update
sudo apt-get install libc6:armhf libisl23:armhf libmpfr6:armhf libmpc3:armhf libstdc++6:armhf -y
sudo ln -s /usr/lib/arm-linux-gnueabihf/libisl.so.23 /usr/lib/arm-linux-gnueabihf/libisl.so.10
sudo ln -s /usr/lib/arm-linux-gnueabihf/libmpfr.so.6 /usr/lib/arm-linux-gnueabihf/libmpfr.so.4
```

### Python 2.7

```bash
sudo cp /etc/apt/sources.list /tmp/
echo 'deb http://archive.debian.org/debian/ stretch contrib main non-free' | sudo tee -a /etc/apt/sources.list
sudo apt update
sudo apt install python2.7 -y
sudo mv /tmp/sources.list /etc/apt/
sudo apt update
python2.7 --version
```

## Repo

### Clone and build

```bash
git clone --depth=1 https://github.com/seemoo-lab/nexmon.git
cd nexmon
source setup_env.sh
sed -i '1 s/$/2.7/' $NEXMON_ROOT/buildtools/b43-v3/debug/b43-beautifier
make
```

### Build nexutil

```bash
cd $NEXMON_ROOT/utilities/nexutil
sudo -E make install USE_VENDOR_CMD=1
sudo setcap cap_net_admin+ep /usr/bin/nexutil
```

### Fetch firmware

```bash
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189
git clone --depth=1 https://github.com/seemoo-lab/nexmon_csi.git
cd nexmon_csi
```

### Build utils, Build and Setup makecsiparams

```bash
cd ~/nexmon/patches/bcm43455c0/7_45_189/nexmon_csi/utils/makecsiparams
make

sudo cp makecsiparams /usr/local/bin/
```

### Build and patch firmware

```bash
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189/nexmon_csi

make -f Makefile.rpi install-firmware
make -f Makefile.rpi unmanage
make -f Makefile.rpi reload-full
```

## Monitor mode after boot

```bash
cd ~/nexmon
source setup_env.sh
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189/nexmon_csi

# Build and install the patched firmware:

make -f Makefile.rpi install-firmware
make -f Makefile.rpi unmanage
make -f Makefile.rpi reload-full

# bring up wlan0
sudo ifconfig wlan0 up
# enable monitoring
sudo nexutil -Iwlan0 -m1

# Verify
nexutil -m

# Setup monitor
sudo nexutil -k40/80          # match AP_CHANNEL above
sudo nexutil -m1
```
## Configure

```bash
# Configure the CSI extractor and activate monitor mode:

nexutil -s500 -b -l34 -v<your-config-generated-with-makecsiparams>
nexutil -m1

# Capture CSI UDPs, for example using tcpdump:

sudo tcpdump -i wlan0 dst port 5500


### or

# -m <mac> for mac filter, only accept that mac
# -b 0x88 for byte filter, first payload byte
# both not needed

CSIPARAMS=$(makecsiparams -c 6/20 -C 1 -N 1 -m 2C:CF:67:32:57:F0 -b 0x88)
echo "Generated params: $CSIPARAMS"
sudo nexutil -Iwlan0 -s500 -b -l34 -v$CSIPARAMS

sudo tcpdump -i wlan0 udp port 5500 -w csi_capture.pcap

# Example
CSIPARAMS=$(makecsiparams -c 40/80 -C 1 -N 1)
echo "Generated params: $CSIPARAMS"
sudo nexutil -Iwlan0 -s500 -b -l34 -v$CSIPARAMS

sudo tcpdump -i wlan0 udp port 5500 -w csi_capture.pcap
```

## Unload

```bash
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189/nexmon_csi

make -f Makefile.rpi restore-wifi
```

## On Error

### RF-Kill status

```bash
# list all
rfkill list
# unblock wifi
sudo rfkill unblock wifi
# or unblock all
sudo rfkill unblock all
```

### Services

```bash
sudo systemctl stop wpa_supplicant
sudo systemctl stop dhcpcd

# Cut power manager
sudo iw dev wlan0 set power_save off
```