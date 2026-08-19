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