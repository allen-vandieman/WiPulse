# WiPulse

WiPulse - Contactless Vital Sign Extraction

[Start here](/setup/testing.md)

## Contents

1. [Brief](#brief)
1. [Goals](#goals)
1. [Drawing](#drawing)
1. [Progress](#progress)


## Brief

Contact-based vital-sign extraction, while reliable, is uncomfortable, non
portable, and impractical in many situations — particularly for infants and 
elderly individuals. This project proposes WiPulse, a contactless vital-sign 
extraction system that uses Wi-Fi Channel State Information (CSI) to 
monitor heart rate without physical contact. Rather than relying on 
hardware that is difficult to source, such as the Intel 5300 NIC, the system 
uses an ESP32 — a low-cost, readily available IoT device — to extract CSI 
amplitude data. A Raspberry Pi 5 serves as the processing platform, where 
Fourier transforms and Fourier series are applied to convert the extracted 
signals from the time domain to the frequency domain, and noise is filtered 
from the resulting data. A Long Short-Term Memory (LSTM) machine 
learning model is then trained using ground-truth heart-rate readings 
obtained from a PPG optical sensor and smartwatch. The processed output 
is displayed on a screen and delivered through a web API for mobile 
monitoring. This project demonstrates that contactless vital-sign extraction 
is achievable using minimal, affordable, and commercially available 
consumer hardware, offering a less intrusive alternative for monitoring 
vulnerable individuals. 

## Goals

1. Use consumer of the shevle IOT devices
2. Patch firmware to allow (a) transmission and (b) capture  of raw frames on (c) any channel
3. Extract Received Signal Strength Indicator (RSSI) and  Channel State Information (CSI)
4. Transform CSI data in a meaningful way
5. Process CSI data to extract usefull information
6. Output or display data in a userfriendly way

## Drawing

![Drawing](img/Wi-Pulse_Drawing-V2.png)

## Progress

Progress thus far:

![Progress](img/Progress.jpg)

1. [Setup TX](tx/setup.md)
1. [Setup RX](rx/setup.md)
1. [Testing](setup/testing.md)