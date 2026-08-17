# Directory Structure

```
csi_processing/
├── __init__.py
├── read_csv_csi.py      # Data loading module
├── filters.py           # Filter functions
├── plot_csv.py          # Plotting functions
├── main.py              # Main application
├── run_pipeline.py      # Simple runner script
└── graphs/              # Output directory for plots
    ├── csi_analysis.png
    ├── csi_timeseries.png
    ├── csi_statistics.png
    ├── motion_detection.png
    ├── filter_comparison.png
    ├── filtered_csi_analysis.png
    └── filtered_problematic_subcarriers.png
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

# Summary

This modular architecture provides:

1. Separation of Concerns: Each module has a single responsibility

1. Reusability: Each module can be used independently

1. Extensibility: Easy to add new filters, plot types, or data sources

1. Configurability: Filter parameters can be customized via command line

1. Flexibility: Run full pipeline or individual steps

1. Maintainability: Clean, well-organized code structure