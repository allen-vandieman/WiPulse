# Channels

For 5 GHz Wi-Fi, if you configure:

Primary channel: 36
Channel width: VHT80 (80 MHz)

then you are using 4 contiguous 20 MHz channels:

**Channel 36 (5180 MHz)**

**Channel 40 (5200 MHz)**

**Channel 44 (5220 MHz)**

**Channel 48 (5240 MHz)**

How many channels are you using?

4 x 20 MHz channels combined into one 80 MHz channel.

What is the center frequency?

The center frequency is halfway across the 80 MHz block:

5180 MHz + 30 MHz = 5210 MHz 

So the center frequency is 5210 MHz.

## Summary

| Setting |	Value |
|--|--|
| Primary channel |	36 |
| Width |	VHT80 (80 MHz) |
| 20 MHz channels used |	36, 40, 44, 48 |
| Frequency range |	5180 MHz to 5260 MHz |
| Center frequency |	5210 MHz |


In 802.11ac/ax terminology, this is the 80 MHz channel block centered at 5210 MHz (often referred to as the channel group 36-48).

## Channel Information

### Possible 5G channels in ZA (South Africa)
| Setting | Info |
|--|--|
| 80MHZ | 36 , 44  , 100 , 108 , 116 , 124 |
| 20MHz | 40 , 48 , 104 , 112 , 120 , 128 , 136 |
| HT20 | 20 MHz channel. Typical PHY rate: up to ~72–150 Mbps per stream. |
| HT40+ | Uses 40 MHz by bonding two adjacent 20 MHz channels. |
||       // Primary channel is the lower one. Secondary channel is above (+).|
|HT40- | Primary channel is the upper one. Secondary channel is below (-). |
| Example||
| HT40- | usually means channels 36+40. |
| HT40+ | usually means channels 40+44. |
| VHT80  | Uses 80 MHz total width. Bonds four 20 MHz channels.|
|| Requires 802.11ac support.|
| 80 MHz | These are often used interchangeably.|
| Example ||
| 80 MHz | channel width.|
| VHT80 | 802.11ac mode using an 80 MHz width.|