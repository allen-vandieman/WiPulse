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
