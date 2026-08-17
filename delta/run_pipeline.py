#!/usr/bin/env python3

"""
Simple runner script for the CSI processing pipeline.
Supports both CSV and PCAP files.

Usage: python run_pipeline.py [input_file] [options]
  input_file: .csv or .pcap file
"""

import sys
from pathlib import Path
from main import main

def print_usage():
    print("Usage: python run_pipeline.py [input_file] [options]")
    print("\nInput file can be:")
    print("  - CSV file (csi_capture_*.csv) from live_monitor.py --capture-mode complex")
    print("  - PCAP file (capture.pcap) from live_monitor.py")
    print("\nOptions:")
    print("  --analysis-only        Only load and plot, no filtering")
    print("  --output-dir DIR       Output directory for plots")
    print("  --no-save-filtered     Don't save filtered data")
    print("  --notch-q Q            Quality factor for notch filter (default: 30.0)")
    print("  --hampel-window W      Window size for Hampel filter (default: 11)")
    print("  --hampel-threshold T   Threshold for Hampel filter (default: 3.0)")
    print("  --dc-threshold T       Threshold for DC detection (default: 0.1)")
    print("\nExamples:")
    print("  # Process CSV file")
    print("  python run_pipeline.py captures/csi_capture_20260812_191830.csv")
    print("\n  # Process PCAP file")
    print("  python run_pipeline.py captures/capture.pcap")
    print("\n  # Analysis only (no filtering)")
    print("  python run_pipeline.py captures/capture.pcap --analysis-only")
    print("\n  # Custom output directory")
    print("  python run_pipeline.py captures/capture.pcap --output-dir my_results")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    # Check if file exists
    input_file = sys.argv[1]
    if not Path(input_file).exists():
        print(f"Error: File '{input_file}' not found!")
        sys.exit(1)
    
    # Run main with the arguments
    main()