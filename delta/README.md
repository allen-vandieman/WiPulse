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
# Full pipeline (load, filter, plot)
python run_pipeline.py csi_capture_20240101_120000.csv

# Analysis only (no filtering)
python run_pipeline.py csi_capture_*.csv --analysis-only

# Custom output directory
python run_pipeline.py csi_capture_*.csv --output-dir my_results

# Custom filter parameters
python run_pipeline.py csi_capture_*.csv --notch-q 40.0 --hampel-window 15

# Using main.py directly
python main.py csi_capture_*.csv --analysis-only
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

# Summary

## V0.1

This modular architecture provides:

1. Separation of Concerns: Each module has a single responsibility

1. Reusability: Each module can be used independently

1. Extensibility: Easy to add new filters, plot types, or data sources

1. Configurability: Filter parameters can be customized via command line

1. Flexibility: Run full pipeline or individual steps

1. Maintainability: Clean, well-organized code structure

# Key Changes Made:

## V0.2

1. Fixed Import: Added import numpy as np to main.py

1. Less Aggressive Filtering: By default, filters are only applied to problematic subcarriers (DC and permanent zeros), not all 256 subcarriers

1. Added Configuration Option: aggressive_filtering flag controls whether to apply filters to all or only problematic subcarriers

1. Better Statistics: Shows modifications only for problematic subcarriers

1. Added dc_threshold Parameter: Allows tuning of DC detection sensitivity

# Key Features Added

## V0.3

1. Automatic File Type Detection: Detects CSV vs PCAP based on file extension

1. Unified Interface: Same commands work for both file types

1. PCAP Metadata Extraction: Extracts sequence, core, NSS, chanspec, chipver

1. Strongest Subcarrier Display: Shows 15 strongest subcarriers like the original read_pcap.py

1. Flexible Output: PCAP data can be saved as CSV after filtering

1. Backward Compatibility: Maintains original read_pcap.py function