# ECHO

## Application

```
project/
├── live_monitor.py          # Main dashboard (UPDATED with integration)
├── heartbeat_extraction.py  # Enhanced with occupancy/motion (UPDATED)
├── filters.py               # CSI filtering utilities (unchanged)
├── decoder.py               # CSI decoding (unchanged)
├── read_csv_csi.py          # CSV loading (unchanged)
├── windowed_psd_analysis.py # Offline analysis (unchanged)
├── pi_serial_recorder.py    # Ground truth recording (unchanged)
└── output/                  # Output directory
```

## Running the Live Monitor

```bash
sudo python3 live_monitor.py \
    --iface wlan0 \
    --chip 43455c0 \
    --bw 80 \
    --hr-window 30 \
    --hr-update 2 \
    --hr-low 0.7 \
    --hr-high 2.5 \
    --hr-subcarriers 12 \
    --hr-percentile 20 \
    --capture-mode complex
```

## Live Monitor 

The live estimator below uses a 30-second rolling CSI window, recalculates approximately every 2 seconds, and estimates BPM from the dominant PSD peak in the configured heartbeat band. It also reports a confidence value based on peak prominence relative to the surrounding PSD.

The pipeline is:

complex CSI → bad-subcarrier cleaning/interpolation → amplitude → Hampel → optional notch → heartbeat bandpass → drift removal → normalization → Welch PSD → BPM

```
/live_monitor.py 
|
|-decoder.py
|
|-filters.py
```

## Serial Ground truth recorder

`pi_serial_recorder.py` receivers heartbeat information from two sensors managed by an Arduino Uno R4 Wifi, and send the average Heartrate to the Raspberry PI.

# Quick Start

## Launch CSI capture

Quick launch CSI and ground truth capture with 

```bash
./start_wipulse_capture.sh
```

# Validation

```
/multi_capture_validation.py
|
|-read_csv_csi.py
|-read_pcap.py
|-filters.py
|-heartbeat_extraction.py
|-windowed_psd_analysis.py
```

When i updated multi_capture_validation.py i also had to update windowed_psd_analysis.py

## Multi capture validation V2

```bash
# Run all (its hard coded)
python multi_capture_validation.py
```

### Add files to test

```python
# In the code, change:

# CAPTURES
        (
            r"captures\04-09-2026\Allen-S,0,1\csi_capture_20260904_232607.csv",
            r"captures\04-09-2026\Allen-S,0,1\ground_truth_20260904_232615.csv"
        ),
```

### For Old Beat Timestamp CSVs

```bash
# In the code, change:
results, windows, summary = run_test(gt_type="beats")
```

## Key Parameters to Tune

|Parameter|	Current|	Range to Try|	Rationale|
|---|---|---|---|
|Low cutoff|	0.8 Hz|	0.5-1.0 Hz|	Catch lower heart rates|
|High cutoff|	2.5 Hz|	2.0-3.5 Hz|	Include harmonics|
|Filter order|	3|	2-5|	Balance sharpness vs stability|
|Peak detection threshold|	0.3 std|	0.2-0.6 std|	Adjust sensitivity|
|Min beat distance|	300ms|	250-400ms|	Physiological range|

`How to convert from frequency to Beats per minute.`

0.8 Hz => 0.8 beats per second

0.8 x 60 = 48 beats per minute

 - 0.8 Hz = 48 bpm

 - 2.5 Hz = 150 bpm


## Success Criteria

|Metric|	Target|
|---|---|
|F1-Score|	> 0.75|
|HR MAE|	< 5 BPM|
|HR Correlation|	> 0.8|
|Timing Error|	< 50 ms|

### Metrics Explained

|Metric|	Description|
|---|---|
|Precision|	% of CSI detections that are correct|
|Recall|	% of actual beats that were detected|
|F1-Score|	Harmonic mean of precision and recall|
|MAE|	Mean absolute error in beat timing|
|RMSE|	Root mean square error of heart rate|
|Correlation|	Pearson correlation of heart rates|