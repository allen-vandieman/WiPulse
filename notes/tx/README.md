# TX

Setup kali linux on raspberry pi 5

# Contents

1. [WiFi](#wifi)
    1. [Find an empty channel](#find-an-empty--quite-channel)
    1. [Setup basic AP](#setup-basic-access-point-ap-wifi)
    1. [Change AP channel](#modify--change-channel)
        1. [Channel Information](#channel-information)
        1. [Change Channel](#change-channel)
    1. [Speed Test](#speed-test)
1. [Monitor](#monitor)
    1. [Setup monitor](#create-monitor)

## WiFi

### Find an empty / quite channel

```bash
nmcli device wifi list
iw phy phy0 channels
iw list | grep -A20 Frequencies
```

### Setup basic Access Point (AP) wifi

```bash
sudo nmcli con add type wifi ifname wlan0 con-name AP autoconnect yes ssid wipulse

sudo nmcli con modify AP \
    802-11-wireless.mode ap \
    802-11-wireless.band a \
    802-11-wireless.channel 36 \
    802-11-wireless.channel-width 80 \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "123456789" \
    ipv4.addresses 192.168.4.1/24 \
    ipv4.method shared

sudo nmcli con up AP
```

### Modify / change channel

#### Channel Information

```bash
# Possible 5G channels in ZA (South Africa)
# (80MHZ) 36 , 44  , 100 , 108 , 116 , 124
# (20MHz) 40 , 48 , 104 , 112 , 120 , 128 , 136

# HT20 // 20 MHz channel. Typical PHY rate: up to ~72–150 Mbps per stream.
# HT40+ // Uses 40 MHz by bonding two adjacent 20 MHz channels. 
#       // Primary channel is the lower one. Secondary channel is above (+).
# HT40- // Primary channel is the upper one. Secondary channel is below (-).
# Example
# HT40- usually means channels 36+40.
# HT40+ usually means channels 40+44.
# VHT80  // Uses 80 MHz total width. Bonds four 20 MHz channels.
#	 // Requires 802.11ac support.
# 80 MHz  // These are often used interchangeably.
#	  // 80 MHz = channel width.
#	  // VHT80 = 802.11ac mode using an 80 MHz width.
```

#### Change channel

```bash
sudo nmcli con modify AP \
    802-11-wireless.channel 108 \
    802-11-wireless.channel-width 80

sudo nmcli con down AP
sudo nmcli con up AP

#  Verify
iw dev
```

### Speed Test

```bash
iw dev wlan0 link
iw dev wlan0 station dump

# iperf3
# On Server
iperf3 -s

# From client
iperf3 -c 192.168.4.1
```

## Monitor

### Create monitor

```bash
# Kill problems and Start
sudo airmon-ng check kill
sudo airmon-ng start wlan0

# Verify
iw dev

# Change channel
sudo iw dev wlan0mon set channel 40 80MHz

# Stop monitor
sudo airmon-ng stop wlan0mon 
```