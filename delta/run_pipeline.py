#!/usr/bin/env python3

"""
Simple runner script for the CSI processing pipeline.
Usage: python run_pipeline.py csi_capture_*.csv
"""

import sys
from main import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py csi_capture_*.csv [options]")
        print("\nOptions:")
        print("  --analysis-only    Only load and plot, no filtering")
        print("  --output-dir DIR   Output directory for plots")
        print("  --no-save-filtered Don't save filtered CSV")
        print("\nExample:")
        print("  python run_pipeline.py csi_capture_20240101_120000.csv")
        sys.exit(1)
    
    main()