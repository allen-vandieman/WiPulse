# Delta Version Plan

## Objective

1. Valid heartbeat

## Capture

```bash
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80 --capture-mode complex
# press 'c' to start recording, 'c' again to stop, 'q' to quit

python3 heartbeat-giop.py
# press 'c' to start recording, 'c' again to stop, 'q' to quit
```

## Usage

### `run_pipeline.py` -> apply all filters and plot graphs

```bash
# CSV Analysis
python run_pipeline.py captures/anusha1-normal/csi_capture_20260812_191830.csv

# pcap Analysis (weaker because of time stamp)
python run_pipeline.py captures/capture.pcap
```

- flags

```bash
# No filtering
--analysis-only

# Specify output
--output-dir results

# Custom Filter Parameters
--notch-q 40.0 --hampel-window 15 --dc-threshold 0.2
```

### Heartbeat Extraction Module `heartbeat_extraction.py`

```bash
# Full validation
python validate_heartbeat.py --csi capture.pcap --groundtruth output/ground_truth_20260817_143022.csv

# Quick alignment check
python align_beats.py --csi capture.pcap --groundtruth output/ground_truth_20260817_143022.csv

# Custom filter parameters
python validate_heartbeat.py --csi capture.pcap --groundtruth output/ground_truth_*.csv --heart-low 0.7 --heart-high 2.0 --heart-order 4
```

- flags

```bash
--heart-low FLOAT      Lower cutoff (Hz), default: 0.8
--heart-high FLOAT     Upper cutoff (Hz), default: 2.5
--heart-order INT      Filter order, default: 3
--no-heartbeat         Disable heartbeat extraction
--heart-auto-select    Auto-select best subcarriers
```

## Current Issues Identified

Root causes likely:

1. Filter passband too narrow/incorrect - 0.8-2.5 Hz may not match your actual heart rate

1. Subcarrier selection - Auto-selection may be picking noisy subcarriers

1. Signal-to-noise ratio - CSI signal may be contaminated by respiration/motion

1. Alignment error - 99ms timing error suggests alignment could be improved

1. Ground truth vs CSI sampling mismatch - Different rates cause interpolation errors

## Recommended Next Steps

### Collect Longer, Cleaner Data (High Priority)

```bash
# Run for 60-90 seconds with subject resting
# Ensure:
# - Subject is still (no movement)
# - Sensor is stable
# - No other people nearby
# - Capture both CSI and ground truth simultaneously
```
### 2. Fine-Tune Bandpass Filter Systematically

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
### 3. Improve Subcarrier Selection

```python
def select_subcarriers_manually(csi_data):
    """Manual subcarrier selection strategies"""
    magnitudes = np.abs(csi_data)
    mean_mag = np.mean(magnitudes, axis=0)
    std_mag = np.std(magnitudes, axis=0)
    
    # Strategy 1: Highest SNR
    snr = mean_mag / (std_mag + 1e-6)
    top_snr = np.argsort(snr)[-10:]  # Top 10 by SNR
    
    # Strategy 2: Medium amplitude (avoid DC, avoid clipping)
    sorted_by_amplitude = np.argsort(mean_mag)
    middle_range = sorted_by_amplitude[len(sorted_by_amplitude)//3:2*len(sorted_by_amplitude)//3]
    
    # Strategy 3: Lowest variance (stable subcarriers)
    low_variance = np.argsort(std_mag)[:10]
    
    return {
        'top_snr': top_snr,
        'middle_range': middle_range,
        'low_variance': low_variance
    }
```

### 4. Respiratory Filtering

Add a notch filter at respiratory frequency (0.15-0.4 Hz):

```python
def add_respiratory_notch(signal_data, fs=100.0):
    """Remove respiratory component"""
    from scipy import signal
    
    # Notch at 0.2 Hz (typical respiration)
    b, a = signal.iirnotch(0.2 / (fs/2), 10)  # Q=10
    filtered = signal.filtfilt(b, a, signal_data)
    return filtered
```

### 5. Multi-Subcarrier Weighted Averaging

Weight subcarriers by their signal quality:

```python
def weighted_heartbeat_average(csi_data):
    """Weighted average across subcarriers based on quality"""
    magnitudes = np.abs(csi_data)
    
    # Calculate quality metric per subcarrier
    quality_scores = []
    for sc in range(csi_data.shape[1]):
        mag = magnitudes[:, sc]
        # Variance in heartbeat band
        quality = np.var(mag) / (np.mean(mag) + 1e-6)
        quality_scores.append(quality)
    
    quality_scores = np.array(quality_scores)
    weights = quality_scores / (np.sum(quality_scores) + 1e-6)
    
    # Weighted average of filtered signals
    weighted_signal = np.zeros(csi_data.shape[0])
    for sc in range(csi_data.shape[1]):
        weighted_signal += weights[sc] * np.abs(csi_data[:, sc])
    
    return weighted_signal
```

## Recommended Testing Plan

### Phase 1: Data Collection (1 day)

1. Collect 3-5 sessions of 60-90 seconds each

1. Vary heart rate (resting, walking recovery, etc.)

1. Document subject's actual heart rate

### Phase 2: Filter Sweep (1 day)

1. Run sweep script on collected data

1. Identify optimal filter parameters

1. Validate on held-out sessions

### Phase 3: Feature Engineering (2-3 days)

1. Test weighted subcarrier averaging

1. Add respiratory notch filtering

1. Experiment with different peak detection thresholds

### Phase 4: Validation (1 day)

1. Run full validation pipeline on all sessions

1. Generate final performance metrics

1. Document optimal parameters

## Immediate Actions (Today)

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

# WiPulse Project Plan

## Objective
Build a contactless heart-rate monitor using Wi-Fi CSI on Raspberry Pi, with real-time signal processing and validation against ground-truth pulse measurements.

## Scope
- Reuse existing Nexmon CSI capture and analysis tools in `app/`
- Add a dedicated HR estimation pipeline in `app/hr_pipeline.py`
- Provide documentation and usage examples in `README.md` and `app/README.md`
- Add regression tests that verify HR extraction and subcarrier selection

## Approach
1. Decode recorded CSI or load live-monitor CSVs.
2. Preprocess CSI amplitude:
   - Hampel outlier removal
   - Heartbeat bandpass filtering (`0.7-3.7 Hz`)
   - DC drift removal
3. Score and select informative subcarriers.
4. Combine selected subcarriers and estimate heart rate using windowed FFT.
5. Save per-window BPM estimates and optional ground-truth comparison.

## Verification
- Synthetic unit tests in `tests/test_hr_pipeline.py`
- Ground-truth validation via `align_and_baseline.py` output and `app/ground_truth.csv`
- Documentation examples for offline and live use

## Files Added
- `app/hr_pipeline.py`
- `plan.md`
- `tests/test_hr_pipeline.py`

## Usage
- Offline pcap: `python app/hr_pipeline.py --pcap csi_capture.pcap --chip 43455c0 --bw 80 --out-dir results/hr`
- Live CSV: `python app/hr_pipeline.py --csi-csv output/csi_capture_20260728_101500.csv --gt-csv output/ground_truth.csv --out-dir results/hr`

## Phase 2: Advanced Research & Implementation

Building on the successful CSI extraction pipeline (Phase 1), Phase 2 focuses on integrating advanced signal processing methods derived from recent academic research found in the `research/` folder. The goal is to improve HR estimation accuracy and robustify performance against challenging environmental conditions.

**Next Steps:**
1. **Research Integration:** Systematically review papers and models stored in the `research/` directory (e.g., advanced feature extraction, deep learning approaches).
2. **Method Implementation:** Select promising techniques (e.g., non-linear regression, specific spectral analysis) and implement wrappers or modules within `app/utils.py` or a new dedicated module to test their efficacy against current data.
3. **Module Enhancement:** Update core modules like `hr_pipeline.py` and potentially add a new module for complex signal modeling.
4. **Testing & Benchmarking:** Develop specific benchmarks in the `test/` directory that compare the standard FFT approach (Phase 1) against newly implemented research-based methods, using datasets from `app/output/`.

**Target Output:** A validated, high-accuracy HR estimation pipeline ready for field deployment considerations.
