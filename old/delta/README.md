# Directory Structure

```
csi_processing/
├── __init__.py              # Package initialization
├── read_csv_csi.py          # CSV data loader
├── read_pcap.py             # PCAP data loader (NEW)
├── filters.py               # Filter functions
├── plot_csv.py              # Plotting functions  
├── main.py                  # Main application (updated)
├── run_pipeline.py          # Simple runner (updated)
└── graphs/                  # Output directory
    ├── csi_analysis.png
    ├── csi_timeseries.png
    ├── csi_statistics.png
    ├── motion_detection.png
    ├── filter_comparison.png
    ├── filtered_csi_analysis.png
    └── filtered_problematic_subcarriers.png
```
# Setup

```bash
# Create virtual enviroment
python3 -m venv .csi

# Windows: 
.csi\Scripts\activate

# Linux
source .csi/bin/activate

# Temp working dir
cd delta

# Install requirements
pip install -r requirements.txt
```

# Usage Examples

```bash
# Run with default heartbeat extraction
python main.py capture.pcap

# Adjust bandpass range for different heart rates
python main.py capture.pcap --heart-low 0.7 --heart-high 2.0

# Higher order filter for sharper cutoff
python main.py capture.pcap --heart-order 4

# Disable heartbeat extraction
python main.py capture.pcap --no-heartbeat

# Auto-select best subcarriers
python main.py capture.pcap --heart-auto-select
```

## 1. Process CSV File

```bash
python run_pipeline.py captures/anusha1-normal/csi_capture_20260812_191830.csv
```

## 2. Process PCAP File

```bash
python run_pipeline.py captures/capture.pcap
```

## 3. Analysis Only (No Filtering)

```bash
python run_pipeline.py captures/capture.pcap --analysis-only
```

## 4. Custom Output Directory

```bash
python run_pipeline.py captures/capture.pcap --output-dir results
```

## 5. Custom Filter Parameters

```bash
python run_pipeline.py captures/capture.pcap --notch-q 40.0 --hampel-window 15 --dc-threshold 0.2
```

## Process Multiple Files (Using Shell Loop)

```bash
# Process all CSV files in a directory
for file in captures/*.csv; do
    python run_pipeline.py "$file" --analysis-only
done

# Process all PCAP files
for file in captures/*.pcap; do
    python run_pipeline.py "$file"
done
```

# Summary of New Features

## Heartbeat Extraction Module (heartbeat_extraction.py)

| Feature |	Description |
|---|---|
|Bandpass Filter |	Third-order Butterworth, adjustable (0.8-2.5 Hz default)|
|Auto-subcarrier Selection |	Automatically picks best subcarriers for heartbeat detection|
|Heart Rate Estimation |	FFT-based frequency analysis with confidence scoring|
|Peak Detection |	Identifies individual heartbeats in the waveform|
|Spectrogram |	Time-frequency visualization for heart rate over time|
|Sliding Window Analysis |	Tracks heart rate changes over the capture duration|
|Comprehensive Plots |	6-panel analysis dashboard with summary statistics |

## New CLI Arguments

```text
--heart-low FLOAT      Lower cutoff (Hz), default: 0.8
--heart-high FLOAT     Upper cutoff (Hz), default: 2.5
--heart-order INT      Filter order, default: 3
--no-heartbeat         Disable heartbeat extraction
--heart-auto-select    Auto-select best subcarriers
```

# Summary of alignment

I've created two scripts for your validation workflow:

### 1. validate_heartbeat.py - Complete Validation Pipeline

|Feature|	Description|
|---|---|
|Data Loading|	Supports both PCAP and CSV CSI files|
|Ground Truth Loading|	Reads CSV from heartbeat-giop.py|
|CSI Heartbeat Extraction|	Uses your bandpass filter (0.8-2.5 Hz)|
|Timestamp Alignment|	Auto-aligns using cross-correlation|
|Beat Detection|	Compares CSI peaks vs ground truth beats|
|Validation Metrics|	Precision, Recall, F1-Score, MAE, RMSE, Correlation|
|Comprehensive Plots|	8-panel validation dashboard|
|Detailed Analysis|	Zoomed waveform comparison, error distribution|
|Report Generation|	Text report with confidence assessment|

### 2. align_beats.py - Quick Alignment Helper

|Feature|	Description|
|---|---|
|Fast Alignment|	Finds optimal time offset quickly|
|Visual Feedback|	Event plot of aligned beats|
|Match Score|	Shows percentage of matched beats|
|Interactive|	Plots match score vs offset|


## Alignement Usage Examples

```bash
# Full validation
python validate_heartbeat.py --csi capture.pcap --groundtruth output/ground_truth_20260817_143022.csv

# Quick alignment check
python align_beats.py --csi capture.pcap --groundtruth output/ground_truth_20260817_143022.csv

# Custom filter parameters
python validate_heartbeat.py --csi capture.pcap --groundtruth output/ground_truth_*.csv --heart-low 0.7 --heart-high 2.0 --heart-order 4
```

### Output Files

|File|	Description|
|---|---|
|`validation_results.png`|	Main 8-panel validation dashboard|
|`validation_detailed.png`|	Zoomed waveform comparison|
|`validation_report.txt`|	Text report with all metrics|
|`alignment_results.png`|	Quick alignment visualization|

### Metrics Explained

|Metric|	Description|
|---|---|
|Precision|	% of CSI detections that are correct|
|Recall|	% of actual beats that were detected|
|F1-Score|	Harmonic mean of precision and recall|
|MAE|	Mean absolute error in beat timing|
|RMSE|	Root mean square error of heart rate|
|Correlation|	Pearson correlation of heart rates|

## Extraction Usage Examples

```bash
# Run with default heartbeat extraction
python main.py capture.pcap

# Adjust bandpass range for different heart rates
python main.py capture.pcap --heart-low 0.7 --heart-high 2.0

# Higher order filter for sharper cutoff
python main.py capture.pcap --heart-order 4

# Disable heartbeat extraction
python main.py capture.pcap --no-heartbeat

# Auto-select best subcarriers
python main.py capture.pcap --heart-auto-select
```

### Output Files

- `heartbeat_analysis.png` - Comprehensive 6-panel analysis

- `subcarrier_comparison.png` - Comparison across subcarriers

- `heartbeat_signal.txt` - Extracted heartbeat waveform

- `heartbeat_summary.txt` - Statistics and parameters

# Version Summary

## V0.5

### Key Features Added

1. Align CSI and Groundtruth data

## V0.4

### Key Features Added

1. Third-order Butterworth bandpass filter (0.8-2.5 Hz, adjustable)

1. Subcarrier averaging for optimal signal-to-noise ratio

1. Frequency domain analysis for heart rate estimation

1. Integration with your existing pipeline

## V0.3

### Key Features Added

1. Automatic File Type Detection: Detects CSV vs PCAP based on file extension

1. Unified Interface: Same commands work for both file types

1. PCAP Metadata Extraction: Extracts sequence, core, NSS, chanspec, chipver

1. Strongest Subcarrier Display: Shows 15 strongest subcarriers like the original read_pcap.py

1. Flexible Output: PCAP data can be saved as CSV after filtering

1. Backward Compatibility: Maintains original read_pcap.py function

## V0.2

### Key Changes Made:

1. Fixed Import: Added import numpy as np to main.py

1. Less Aggressive Filtering: By default, filters are only applied to problematic subcarriers (DC and permanent zeros), not all 256 subcarriers

1. Added Configuration Option: aggressive_filtering flag controls whether to apply filters to all or only problematic subcarriers

1. Better Statistics: Shows modifications only for problematic subcarriers

1. Added dc_threshold Parameter: Allows tuning of DC detection sensitivity

## V0.1

This modular architecture provides:

1. Separation of Concerns: Each module has a single responsibility

1. Reusability: Each module can be used independently

1. Extensibility: Easy to add new filters, plot types, or data sources

1. Configurability: Filter parameters can be customized via command line

1. Flexibility: Run full pipeline or individual steps

1. Maintainability: Clean, well-organized code structure