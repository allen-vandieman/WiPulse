#!/usr/bin/env python3
"""
Quick alignment helper for CSI and ground truth data.
Finds the optimal time offset between CSI and ground truth signals.

Usage: python align_beats.py --csi captures/capture.pcap --groundtruth output/ground_truth_*.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from read_pcap import PCAPDataLoader
from read_csv_csi import CSIDataLoader
from filters import CSIFilter, FilterConfig
from heartbeat_extraction import HeartbeatExtractor, HeartbeatConfig


def quick_align(csi_path, groundtruth_path):
    """Quick alignment of CSI and ground truth data"""
    
    print("\nLoading CSI data...")
    csi_path = Path(csi_path)
    if csi_path.suffix.lower() == '.csv':
        loader = CSIDataLoader(str(csi_path))
        csi_data, metadata = loader.load()
    else:
        loader = PCAPDataLoader(str(csi_path))
        csi_data, metadata = loader.load()
    
    print(f"  CSI: {metadata['n_packets']} packets, {metadata['rate']:.1f} Hz")
    
    print("\nLoading ground truth...")
    gt_df = pd.read_csv(groundtruth_path)
    print(f"  Ground truth: {len(gt_df)} beats")
    
    # Extract CSI heartbeat
    print("\nExtracting CSI heartbeat...")
    config = HeartbeatConfig()
    config.sampling_rate = metadata['rate']
    config.bandpass_low = 0.8
    config.bandpass_high = 2.5
    
    extractor = HeartbeatExtractor(config)
    filter_obj = CSIFilter(FilterConfig())
    filtered_csi = filter_obj.apply_all_filters(csi_data, verbose=False)
    heartbeat = extractor.apply_bandpass_filter(filtered_csi)
    peaks = extractor.detect_peaks(heartbeat)['peaks']
    
    # Get beat times
    csi_beat_times = np.arange(len(heartbeat)) / metadata['rate']
    csi_beat_times = csi_beat_times[peaks]
    
    # Get ground truth times
    if 'ts' in gt_df.columns:
        gt_times = gt_df['ts'].values
    else:
        gt_times = np.arange(len(gt_df))
    
    gt_times = gt_times - gt_times[0]
    
    print(f"  CSI beats: {len(csi_beat_times)}")
    print(f"  GT beats: {len(gt_times)}")
    
    # Find optimal offset by cross-correlation
    print("\nFinding optimal alignment offset...")
    
    offsets = np.arange(-10, 10, 0.05)
    best_offset = 0
    best_score = 0
    scores = []
    
    for offset in offsets:
        aligned = csi_beat_times + offset
        # Count matches
        matches = 0
        for gt_time in gt_times:
            if np.any(np.abs(aligned - gt_time) < 0.15):
                matches += 1
        score = matches / len(gt_times) if len(gt_times) > 0 else 0
        scores.append(score)
        if score > best_score:
            best_score = score
            best_offset = offset
    
    print(f"\nBest offset: {best_offset:.2f} seconds")
    print(f"Match rate: {best_score*100:.1f}%")
    
    # Plot alignment
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot aligned beat times
    ax1.eventplot([gt_times], colors='red', lineoffsets=0.2, linewidths=2, label='Ground Truth')
    ax1.eventplot([csi_beat_times + best_offset], colors='blue', lineoffsets=-0.2, linewidths=2, label='CSI (aligned)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('')
    ax1.set_title('Aligned Beat Times')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot match score vs offset
    ax2.plot(offsets, scores, 'b-', linewidth=1.5)
    ax2.axvline(x=best_offset, color='r', linestyle='--', label=f'Best: {best_offset:.2f}s')
    ax2.set_xlabel('Offset (seconds)')
    ax2.set_ylabel('Match Score')
    ax2.set_title('Alignment Score vs Offset')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Beat Alignment Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = Path('alignment_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nAlignment plot saved to '{output_file}'")
    plt.close()
    
    # Return results
    return {
        'best_offset': best_offset,
        'match_score': best_score,
        'n_csi_beats': len(csi_beat_times),
        'n_gt_beats': len(gt_times),
        'aligned_csi_beats': csi_beat_times + best_offset,
        'gt_beats': gt_times
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick alignment of CSI and ground truth data')
    parser.add_argument('--csi', required=True, help='CSI file (.pcap or .csv)')
    parser.add_argument('--groundtruth', required=True, help='Ground truth CSV file')
    
    args = parser.parse_args()
    
    if not Path(args.csi).exists():
        print(f"Error: CSI file '{args.csi}' not found!")
        sys.exit(1)
    
    if not Path(args.groundtruth).exists():
        print(f"Error: Ground truth file '{args.groundtruth}' not found!")
        sys.exit(1)
    
    try:
        results = quick_align(args.csi, args.groundtruth)
        
        print("\n" + "="*50)
        print("ALIGNMENT SUMMARY")
        print("="*50)
        print(f"Offset:     {results['best_offset']:.3f} seconds")
        print(f"Match Rate: {results['match_score']*100:.1f}%")
        print(f"CSI Beats:  {results['n_csi_beats']}")
        print(f"GT Beats:   {results['n_gt_beats']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)