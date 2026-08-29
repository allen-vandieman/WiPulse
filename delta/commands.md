# Structure

```
Analysis/
├── validate_heartbeat.py               # Aligns ground truth heartbeat data with CSI data 
├── psd_subcarrier_analysis.py          # PSD instead of FFT averaging
UNIT TEST/
├── test_psd_vs_auto.py                 # PSD vs Auto averaging
├── unit_test.py                        # Command to run in CMD to test psd or something
├── run_best_method.py                  # Test if hold breath improved HR by adding extra notch filter
├── run_best_method_adaptive.py         # Use gain control instead of notch
├── diagnose_correlation.py             # Comparing absolute HR, compare the HRV (Heart Rate Variability) pattern
├── test_psd_vs_auto_fixed.py           # Add freq low,high and filter order to command
├── fix_peak_detection.py               # Test intervals between peaks adjusted test
├── test_phase_methods.py               # Test Phase
├── test_packet_timing.py               # Test packet uniformity
├── windowed_psd_analysis.py            # use windowed analysis of CSI data
├── windowed_psd_analysis(old).py       # Uncle D gave me old and uncle c gave me new
├── test_packet_timing.py               # TEST PACKET TIMING
├── segment_analysis.py                 # Test packets in segments
├── multi_capture_validation.py         # Broad test for best pipeline  #---> Best test ever !!!!!!!!!!
├── multi_capture_validation_test.py    # Quick review multi_capture_validation results
├── combined_amplitude_phase_test.py    # Combined Amplitude + Phase Heartbeat Detection Test
Tool/
├── tune_heartbeat_filter.py            # Aligns ground truth heartbeat data with CSI data 
├── align_beats.py                      # PSD instead of FFT averaging
└── phase_processing.py                 # Output directory

```

# 1. Capture

```bash
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80 --capture-mode complex
# press 'c' to start recording, 'c' again to stop, 'q' to quit

python3 heartbeat-giop.py
# press 'c' to start recording, 'c' again to stop, 'q' to quit
```

# 2. Pipeline (Filter and graphs)

```bash
# CSV Analysis
python run_pipeline.py captures/anusha1-normal/csi_capture_20260812_191830.csv

# pcap Analysis (weaker because of time stamp)
python run_pipeline.py captures/capture.pcap

# Flags
# No filtering
--analysis-only

# Specify output
--output-dir results

# Custom Filter Parameters
--notch-q 40.0 --hampel-window 15 --dc-threshold 0.2
```

# 3. Validate heartbeat

```bash
# Full validation
python validate_heartbeat.py --csi capture.pcap --groundtruth output/ground_truth_20260817_143022.csv

# Quick alignment check
python align_beats.py --csi capture.pcap --groundtruth output/ground_truth_20260817_143022.csv

# Custom filter parameters
python validate_heartbeat.py --csi capture.pcap --groundtruth output/ground_truth_*.csv --heart-low 0.7 --heart-high 2.0 --heart-order 4

# Flags
--heart-low FLOAT      Lower cutoff (Hz), default: 0.8
--heart-high FLOAT     Upper cutoff (Hz), default: 2.5
--heart-order INT      Filter order, default: 3
--no-heartbeat         Disable heartbeat extraction
--heart-auto-select    Auto-select best subcarriers
``` 

## Ideas

```bash
# Try narrower bandpass filter (focus on typical heart rates)
python validate_heartbeat.py --csi .\captures\anusha1-normal\csi_capture_20260812_191830.csv --groundtruth .\captures\anusha1-normal\ground_truth_20260812_191826.csv --heart-low 0.8 --heart-high 1.8

# Try higher order filter for better noise rejection
python validate_heartbeat.py --csi .\captures\anusha1-normal\csi_capture_20260812_191830.csv --groundtruth .\captures\anusha1-normal\ground_truth_20260812_191826.csv --heart-order 5

# Try wider range if heart rate varies
python validate_heartbeat.py --csi .\captures\anusha1-normal\csi_capture_20260812_191830.csv --groundtruth .\captures\anusha1-normal\ground_truth_20260812_191826.csv --heart-low 0.7 --heart-high 2.2
```

# 4. Fine-Tune Bandpass Filter Systematically

```bash
# 1. Collect a longer run (60+ seconds)
# Make sure subject is sitting still

# 2. Run the filter sweep
python tune_heartbeat_filter.py --csi capture.pcap --groundtruth ground_truth.csv

# 3. Try different subcarrier selection
python main.py capture.pcap --heart-auto-select --heart-low 0.6 --heart-high 2.0

# 4. Check alignment visually
python align_beats.py --csi capture.pcap --groundtruth ground_truth.csv
```

# 5. Test PSD-based selection vs auto-selection

```bash
# psd_subcarrier_analysis.py - Analize using PSD
# Usage: 
python psd_subcarrier_analysis.py --csi captures/Alicia/csi_capture_20260818_193158.csv --groundtruth captures/Alicia/ground_truth_20260818_193154.csv

# test_psd_vs_auto.py
# Compare PSD-based selection vs auto-selection

python test_psd_vs_auto.py --csi captures/Alicia/csi_capture_20260818_193158.csv --groundtruth captures/Alicia/ground_truth_20260818_193154.csv

```

# 6. Multi capture validation

```bash
# Run all (its hard coded)
python multi_capture_validation.py

# Quick summary
multi_capture_validation_test.py
```

# 7. Combined Amplitude + Phase Heartbeat Detection Test

This is the last badboy i am busy with, trying to get capture as close to hearbeat as possible

## 7.1. Validate Combined Methods on a Single Capture

```bash
python combined_amplitude_phase_test.py --csi captures/Allen-S,0,1/csi_capture_20260820_135250.csv --groundtruth captures/Allen-S,0,1/ground_truth_20260820_135247.csv
```

## 7.2. Position Optimization (Find Best Sitting Position)

```bash
# With explicit position names and files
python combined_amplitude_phase_test.py --position-optimization --positions "position1" captures/pos1/csi_capture.csv "position2" captures/pos2/csi_capture.csv "position3" captures/pos3/csi_capture.csv

# Auto-detect captures in current directory
python combined_amplitude_phase_test.py --position-optimization
```

## 7.3. Quick Quality Check for a Capture (No Ground Truth)

```bash
python combined_amplitude_phase_test.py --csi captures/unknown/csi_capture.csv
```

## 7.4 Integration with Existing Workflow

This test fits into your existing workflow:

```bash
# 1. First, collect captures at different positions
# 2. Run position optimization to find best spot
python combined_amplitude_phase_test.py --position-optimization

# 3. Once best position is found, run full validation
python validate_heartbeat.py --csi best_position/csi_capture.csv --groundtruth best_position/ground_truth.csv

# 4. Run multi-capture validation across different subjects
python multi_capture_validation.py
```

## 7.5 Run the Improved Version

```bash
python combined_amplitude_phase_test_v2.py --csi captures/Allen-S,0,1/csi_capture_20260820_135250.csv --groundtruth captures/Allen-S,0,1/ground_truth_20260820_135247.csv
```

### For Position Optimization

- To find the best sitting position, collect captures at different positions and run:


```bash
# With explicit positions
python combined_amplitude_phase_test_v2.py --position-optimization --positions "center" captures/center/csi.csv "left" captures/left/csi.csv "right" captures/right/csi.csv

# Or auto-detect
python combined_amplitude_phase_test_v2.py --position-optimization
```

## Key Parameters to Tune

|Parameter|	Current|	Range to Try|	Rationale|
|---|---|---|---|
|Low cutoff|	0.8 Hz|	0.5-1.0 Hz|	Catch lower heart rates|
|High cutoff|	2.5 Hz|	2.0-3.5 Hz|	Include harmonics|
|Filter order|	3|	2-5|	Balance sharpness vs stability|
|Peak detection threshold|	0.3 std|	0.2-0.6 std|	Adjust sensitivity|
|Min beat distance|	300ms|	250-400ms|	Physiological range|

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