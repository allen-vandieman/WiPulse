# Understanding Frequency Domain Analysis and Power Spectral Density (PSD) for Wi-Fi CSI

## Introduction

When working with Wi-Fi Channel State Information (CSI), each packet contains complex-valued measurements that describe how the wireless channel affects a transmitted signal. These complex values can be used to obtain both amplitude and phase information, which are the primary features used in many wireless sensing and communication applications.

A common question is:

> Why perform frequency domain analysis or calculate the Power Spectral Density (PSD) when CSI already contains frequency information?

This document explains the concepts and motivation behind frequency domain analysis and PSD in the context of Wi-Fi CSI.

---

# 1. What Are CSI Complex Numbers?

For each OFDM subcarrier, CSI provides a complex value:

<center>
<h4>
H(f<sub>k</sub>) = I<sub>k</sub> + jQ<sub>k</sub>
</h4>
</center>

Where:

- I<sub>k</sub> = In-phase component
- Q<sub>k</sub> = Quadrature component
- f<sub>k</sub> = Frequency of subcarrier (k)

Each complex value describes how the wireless channel affects that particular subcarrier frequency.

## Amplitude

The amplitude (or magnitude) is calculated as:

\[
|H(f_k)| = \sqrt{I_k^2 + Q_k^2}
\]

Amplitude represents how much the channel attenuates or amplifies the signal at a specific frequency.

### Interpretation

- High amplitude → strong channel response
- Low amplitude → signal fading or attenuation

---

## Phase

The phase is calculated as:

\[
\angle H(f_k) = \tan^{-1}\left(\frac{Q_k}{I_k}\right)
\]

Phase represents the phase shift introduced by the propagation channel.

### Interpretation

Phase contains information about:

- Signal propagation delay
- Multipath effects
- Movement within the environment

---

# 2. CSI Is Already in the Frequency Domain

A CSI vector for a single packet looks like:

\[
[H(f_1), H(f_2), H(f_3), ... , H(f_N)]
\]

where each element corresponds to a different OFDM subcarrier frequency.

Therefore:

> CSI itself is already a frequency-domain representation of the wireless channel.

This is because the channel response has already been measured separately for each frequency component.

---

# 3. What Does Frequency Domain Analysis Mean?

There are two different types of frequency-domain analysis commonly performed on CSI data.

## Type A: Analysis Across Subcarriers

For a single packet, we can examine:

\[
H(f)
\]

and observe how amplitude or phase changes across subcarrier frequencies.

Example:

```text
Subcarrier:  -28  -27  -26  ... +27 +28
Amplitude:    1.2  0.8 2.1      0.7 1.5
```

### What This Reveals

- Frequency-selective fading
- Multipath reflections
- Deep fades (nulls)
- Channel quality variations

This is the most direct interpretation of CSI data.

---

## Type B: Analysis Across Time

Suppose CSI packets are collected continuously.

For a fixed subcarrier:

\[
H(f_k,t)
\]

we can extract the amplitude:

\[
A(t)=|H(f_k,t)|
\]

This produces a time series.

Example:

```text
Packet 1: 1.20
Packet 2: 1.21
Packet 3: 1.18
Packet 4: 1.25
...
```

Now the question becomes:

> What frequencies exist within these temporal fluctuations?

To answer this, a Fourier Transform (FFT) is applied over time.

---

# 4. Why Does CSI Change Over Time?

Changes in the environment cause CSI values to fluctuate.

Common causes include:

- Human movement
- Breathing
- Heartbeat
- Opening doors
- Moving furniture
- Mechanical vibrations
- Rotating fans

These physical activities introduce periodic variations into the CSI measurements.

### Typical Motion Frequencies

| Activity | Frequency Range |
|-----------|----------------|
| Breathing | ~0.1–0.5 Hz |
| Walking | ~1–3 Hz |
| Running | ~3–6 Hz |
| Mechanical vibration | 10–50 Hz |

These frequencies become visible when CSI is analyzed over time.

---

# 5. Frequency Domain Analysis of a CSI Time Series

From a CSI amplitude time series:

\[
A(t)
\]

we compute:

\[
FFT(A(t))
\]

The FFT decomposes the signal into sinusoidal frequency components.

Instead of viewing:

```text
Amplitude vs Time
```

we obtain:

```text
Power vs Frequency
```

Example:

| Frequency (Hz) | Power |
|---------------|--------|
| 0.2 | High |
| 1.0 | Low |
| 2.5 | Medium |
| 10.0 | Small |

This immediately shows where most of the signal variation occurs.

---

# 6. What Is Power Spectral Density (PSD)?

PSD stands for:

**Power Spectral Density**

PSD describes how a signal's power is distributed across frequencies.

Mathematically:

\[
PSD(f)
\]

indicates how much energy exists at each frequency component.

A PSD can be interpreted as:

| Frequency | Power |
|------------|---------|
| 0.1 Hz | Small |
| 0.2 Hz | Large |
| 0.3 Hz | Small |

The dominant peaks reveal the strongest periodic behaviours in the signal.

---

# 7. FFT vs PSD

These concepts are closely related but not identical.

## FFT

FFT provides complex-valued frequency components:

\[
X(f)
\]

The FFT retains:

- Magnitude
- Phase

information.

---

## PSD

PSD represents the energy or power of those frequency components:

\[
PSD(f) \propto |X(f)|^2
\]

Typically PSD is normalized and smoothed using estimation methods such as Welch's method.

### Why PSD Is Often Preferred

PSD focuses on energy content and removes phase information that may not be important for motion analysis.

As a result:

- Easier to visualize
- Easier to interpret
- More robust to noise

---

# 8. Why Calculate PSD From CSI?

PSD helps identify dominant motions and periodic behaviours present in CSI measurements.

Suppose a CSI amplitude signal appears noisy:

```text
wiggle wiggle wiggle wiggle
```

It may be difficult to identify the source of these fluctuations directly.

After computing the PSD:

```text
Peak at 0.25 Hz
```

we immediately know the dominant variation occurs at:

\[
0.25\,Hz
\]

which corresponds to:

\[
0.25 \times 60 = 15
\]

breaths per minute.

PSD transforms noisy time-domain data into a more interpretable representation.

---

# 9. Example: Breathing Detection

Suppose CSI amplitudes are collected for one minute.

After calculating the PSD:

```text
Frequency (Hz)

0.1
0.2
0.3  <-- Peak
0.4
0.5
```

The dominant frequency is:

\[
0.3\,Hz
\]

Breathing rate:

\[
0.3 \times 60 = 18
\]

breaths per minute.

Using PSD makes respiration frequency easy to identify even when the raw CSI signal appears noisy.

---

# 10. Example: Motion Detection

## Static Environment

PSD:

```text
Low power across all frequencies
```

This indicates little or no movement.

---

## Walking Person

PSD:

```text
Strong energy between 1 Hz and 3 Hz
```

This corresponds to typical walking motion.

---

## Running Person

PSD:

```text
Strong energy over a broader frequency range
```

More motion produces stronger and wider spectral content.

---

# 11. Delay-Domain Analysis Using IFFT

Although CSI is already a frequency-domain representation, it can be transformed into the delay domain using an Inverse FFT (IFFT):

\[
h(\tau)=IFFT(H(f))
\]

This produces an approximation of the channel impulse response.

### What This Reveals

- Direct propagation path
- Reflected paths
- Multipath delays
- Channel structure

In many cases, delay-domain information is easier to interpret than raw CSI measurements.

---

# 12. Typical CSI Processing Pipeline

```text
CSI Packets
      |
      v
Complex CSI H(f,t)
      |
      +--> Amplitude
      |
      +--> Phase
      |
      v
Filtering / Denoising
      |
      v
Subcarrier Selection
      |
      v
Time Series A(t)
      |
      +--> FFT
      |
      +--> PSD
      |
      +--> Dominant Frequency Detection
      |
      +--> Classification
```

Applications include:

- Human activity recognition
- Respiration monitoring
- Heart-rate estimation
- Occupancy sensing
- Gesture recognition
- Motion detection
- Doppler analysis

---

# Key Takeaway

CSI already tells us:

> **How the wireless channel varies across frequency.**

PSD tells us:

> **How the wireless channel varies across time frequencies.**

By analyzing the PSD of CSI amplitude or phase measurements, we can determine:

- What periodic motions exist
- How strong those motions are
- The dominant frequency of movement
- Is there breathing, walking, vibration, or noise?