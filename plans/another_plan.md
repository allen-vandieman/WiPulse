# Processing Pipeline for Heartbeat Monitoring using Raspberry Pi and NEXMON CSI

## Summary Table of Possible Processing Pipelines

| **Stage**                    | **Description**                                                                                                  | **Reference**                                                                                       |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **Amplitude Conversion**     | Extract amplitude information from CSI data.                                                              | Hou Xiaoqing et al., "A study on methods for extracting vital signs based on CSI signals" [11] |
| **Stationary Noise Removal** | Remove the DC component (raw noise) to stabilize the signal.                                               | Xie Y, Li Z, Li M, [2]                                                                                   |
| **Pulse Extraction**         | Use a third-order Butterworth bandpass filter set at 0.8-2.17 Hz (corresponding to 48-130 BPM) for heart rate detection. | Hou Xiaoqing et al., "A study on methods for extracting vital signs based on CSI signals" [11] |
| **Pulse Shaping**             | Apply Savitzky-Golay filtering for smoothing the signal while preserving features.                               | Not specifically mentioned, but inspired by Hou Xiaoqing's work.                                             |
| **Segmentation and Normalization** | Segment the processed data into overlapping windows for further analysis.                                    | Not specifically mentioned, but a typical step in signal processing pipelines.                            |

## Processing Pipeline Summary

### Steps for Signal Processing

1. **Amplitude Conversion**
   - Extract amplitude information from raw CSI data.
   
2. **Stationary Noise Removal**
   - Remove DC components to stabilize the signal.
   
3. **Pulse Extraction**
   - Use a third-order Butterworth bandpass filter at 0.8-2.17 Hz (corresponding to 48-130 BPM) for heart rate detection.
   
4. **Pulse Shaping**
   - Apply Savitzky-Golay filtering for smoothing the signal while preserving important features.
   
5. **Segmentation and Normalization**
   - Segment the processed data into overlapping windows for further analysis.

### References

- Hou Xiaoqing et al., "A study on methods for extracting vital signs based on CSI signals" [11]
- Xie Y, Li Z, Li M, [2]

## Processing Pipeline Implementation Plan

### Steps to Implement the Processing Pipeline

1. **Amplitude Conversion**
   - Use NEXMON CSI data.
   
2. **Stationary Noise Removal**
   - Set up a Butterworth bandpass filter at 0.8-2.17 Hz on both devices for consistent signal stabilization.
   
3. **Pulse Extraction**
   - Configure third-order Butterworth filters specifically for heart rate extraction (0.8-2.17 Hz).
   
4. **Pulse Shaping**
   - Apply Savitzky-Golay filtering on both devices.
   
5. **Segmentation and Normalization**
   - Segment the data into overlapping windows.

### Notes

- Ensure both Raspberry Pi devices use consistent filter settings for accurate heart rate monitoring.

# My uncle that is Deep in China recommends

## What's Missing: Frequency Analysis

To confirm a heartbeat, I need to see **frequency domain analysis**. A heartbeat at ~60-90 BPM would appear as a peak at **1.0-1.5 Hz**.

Can you provide:

1. FFT or Power Spectral Density (PSD) of the filtered CSI for selected subcarriers?

1. Frequency domain comparison between baseline and heartbeat attempts?

1. Peak detection showing the dominant frequency in the signal?

## Recommendations

### 1. Extract the Heartbeat Component

The heartbeat signal is very subtle (~1-2% of total signal). You need to:

- **Bandpass filter** between 0.8-2.5 Hz

- **Average across subcarriers** that show the strongest response

- Apply **moving average** to smooth the signal

### 2. Check Specific Subcarriers

The subcarriers showing the **most variation** in the filtered time series are likely the ones carrying the heartbeat. 

Try:

- **Subcarriers 0, 63, and 123** (high original magnitude)

- **Middle subcarriers** (100-150) are often most sensitive

### 3. Breathing vs Heartbeat

The signal might be showing *breathing* (0.2-0.4 Hz) rather than *heartbeat* (1.0-1.5 Hz). To differentiate:

- Look for **two distinct frequencies** in the signal

- Breathing causes **larger amplitude** variations

- Heartbeat causes **smaller, faster** variations

### 4. Run a Longer Capture

80 seconds at 93.5 Hz gives ~7,500 samples - this is good for frequency analysis. For even better heart rate estimation, 2-3 minutes would be ideal.

## Next Steps

1. Can you share the **frequency domain analysis (FFT/PSD)** for this capture?

1. **Compare the filtered time series** of subcarrier 0, 63, and 123 to see which shows the most rhythmic pattern

1. **Check the phase time series** - heartbeats often cause both amplitude and phase changes

Based on what I see, this capture **looks promising** - the signal is strong, the subject is still, and the filtering is working. The key now is to extract the subtle heartbeat component from the signal.

## Key Recommendations Based on Your Results

### 1. The CSI signal DOES contain heartbeat information

- 52/100 beats matched shows there's a real signal

- The moderate F1-score (0.542) confirms this

### 2. The main issues are:

- Too many false positives (47) → The filter is detecting noise as heartbeats

- Poor heart rate correlation (0.059) → The extracted signal is noisy

### 3. Suggested improvements:

```bash
# Try narrower bandpass filter (focus on typical heart rates)
python validate_heartbeat.py --csi .\captures\anusha1-normal\csi_capture_20260812_191830.csv --groundtruth .\captures\anusha1-normal\ground_truth_20260812_191826.csv --heart-low 0.8 --heart-high 1.8

# Try higher order filter for better noise rejection
python validate_heartbeat.py --csi .\captures\anusha1-normal\csi_capture_20260812_191830.csv --groundtruth .\captures\anusha1-normal\ground_truth_20260812_191826.csv --heart-order 5

# Try wider range if heart rate varies
python validate_heartbeat.py --csi .\captures\anusha1-normal\csi_capture_20260812_191830.csv --groundtruth .\captures\anusha1-normal\ground_truth_20260812_191826.csv --heart-low 0.7 --heart-high 2.2
```

### 4. For the best results, you need:

- Multiple captures to find optimal filter parameters

- Different subcarrier selection - try manually selecting subcarriers 29, 23, 24, 30, 46 (the top performers from auto-selection)

- Signal preprocessing - consider removing the DC component and normalizing before peak detection