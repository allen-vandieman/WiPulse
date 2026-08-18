# test_psd_vs_auto.py
# Compare PSD-based selection vs auto-selection

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from read_csv_csi import CSIDataLoader
from filters import CSIFilter, FilterConfig
from heartbeat_extraction import HeartbeatExtractor, HeartbeatConfig
from validate_heartbeat import HeartbeatValidator


def quick_validate_signal(signal_data, gt_df, fs):
    """Quick validation of a signal against ground truth"""
    try:
        # Bandpass filter (using best params from sweep)
        nyquist = fs / 2
        low = max(0.001, 1.0 / nyquist)
        high = min(0.999, 3.15 / nyquist)
        sos = signal.butter(2, [low, high], btype='band', output='sos')
        filtered = signal.sosfiltfilt(sos, signal_data - np.mean(signal_data))
        
        # Detect peaks
        peaks, _ = signal.find_peaks(
            filtered,
            distance=int(0.3 * fs),
            prominence=np.std(filtered) * 0.4,
            height=np.mean(filtered) + np.std(filtered) * 0.2
        )
        
        if len(peaks) < 3:
            return {'f1': 0, 'precision': 0, 'recall': 0, 'hr_mae': 999, 'n_peaks': 0}
        
        csi_beat_times = np.arange(len(filtered)) / fs
        csi_beat_times = csi_beat_times[peaks]
        
        # Ground truth times
        if 'ts' in gt_df.columns:
            gt_times = gt_df['ts'].values
        else:
            gt_times = np.arange(len(gt_df))
        gt_times = gt_times - gt_times[0]
        
        # Quick alignment
        best_offset = 0
        best_matches = 0
        for offset in np.arange(-10, 10, 0.2):
            aligned = csi_beat_times + offset
            matches = 0
            for gt_time in gt_times:
                if np.any(np.abs(aligned - gt_time) < 0.15):
                    matches += 1
            if matches > best_matches:
                best_matches = matches
                best_offset = offset
        
        aligned_beats = csi_beat_times + best_offset
        
        # Match beats
        used_gt = set()
        matched_pairs = []
        for csi_time in aligned_beats:
            best_match = None
            best_dist = np.inf
            for j, gt_time in enumerate(gt_times):
                if j in used_gt:
                    continue
                dist = abs(csi_time - gt_time)
                if dist < best_dist and dist < 0.2:
                    best_dist = dist
                    best_match = j
            if best_match is not None:
                matched_pairs.append(best_dist)
                used_gt.add(best_match)
        
        tp = len(matched_pairs)
        fp = len(csi_beat_times) - tp
        fn = len(gt_times) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Heart rate MAE
        if len(matched_pairs) > 3 and len(gt_times) > 3:
            csi_hr = 60.0 / (np.diff(peaks[:len(matched_pairs)]) / fs)
            gt_hr = 60.0 / np.diff(gt_times[:len(matched_pairs)])
            min_len = min(len(csi_hr), len(gt_hr))
            if min_len > 1:
                hr_mae = np.mean(np.abs(csi_hr[:min_len] - gt_hr[:min_len]))
            else:
                hr_mae = 999
        else:
            hr_mae = 999
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'hr_mae': hr_mae,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'n_peaks': len(peaks),
            'best_offset': best_offset
        }
        
    except Exception as e:
        print(f"Error in validation: {e}")
        return {'f1': 0, 'precision': 0, 'recall': 0, 'hr_mae': 999, 'n_peaks': 0}


def main():
    parser = argparse.ArgumentParser(description='Compare PSD vs Auto subcarrier selection')
    parser.add_argument('--csi', required=True, help='CSI data file')
    parser.add_argument('--groundtruth', required=True, help='Ground truth CSV')
    parser.add_argument('--output-dir', default='comparison_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading CSI from: {args.csi}")
    loader = CSIDataLoader(args.csi)
    csi_data, metadata = loader.load()
    print(f"  {metadata['n_packets']} packets at {metadata['rate']:.1f} Hz")
    
    print(f"Loading ground truth: {args.groundtruth}")
    gt_df = pd.read_csv(args.groundtruth)
    print(f"  {len(gt_df)} beats")
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Apply basic filtering
    print("\nApplying basic filters...")
    filter_obj = CSIFilter(FilterConfig())
    filtered_csi = filter_obj.apply_all_filters(csi_data, verbose=False)
    
    # Get magnitudes
    magnitudes = np.abs(filtered_csi)
    
    # Define subcarrier sets to test
    subcarrier_sets = {
        'PSD Top 10': [187, 2, 1, 188, 161, 159, 0, 162, 158, 163],
        'PSD Top 5': [187, 2, 1, 188, 161],
        'PSD Top 3': [187, 2, 1],
        'Auto-Select': [232, 228, 49, 60, 50],
        'Subcarrier 187': [187],  # Best individual
        'Subcarrier 2': [2],      # Second best
        'Subcarrier 1': [1],      # Third best
        'All Subcarriers': list(range(256)),  # All
    }
    
    results = []
    
    print("\n" + "="*70)
    print("COMPARING SUBCARRIER SELECTION METHODS")
    print("="*70)
    
    for name, sc_list in subcarrier_sets.items():
        print(f"\nTesting: {name} ({len(sc_list)} subcarriers)")
        
        # Extract signal
        if len(sc_list) == 1:
            signal_data = magnitudes[:, sc_list[0]]
        else:
            signal_data = np.mean(magnitudes[:, sc_list], axis=1)
        
        # Validate
        metrics = quick_validate_signal(signal_data, gt_df, metadata['rate'])
        metrics['method'] = name
        metrics['n_subcarriers'] = len(sc_list)
        metrics['subcarriers'] = str(sc_list)
        results.append(metrics)
        
        print(f"  F1: {metrics['f1']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
        print(f"  HR MAE: {metrics['hr_mae']:.1f} BPM, Peaks: {metrics['n_peaks']}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1', ascending=False)
    
    # Save results
    results_df.to_csv(f'{args.output_dir}/comparison_results.csv', index=False)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. F1 Score comparison
    ax1 = axes[0, 0]
    methods = results_df['method'].values
    f1_scores = results_df['f1'].values
    colors = ['green' if f1 == max(f1_scores) else 'blue' for f1 in f1_scores]
    y_pos = np.arange(len(methods))
    ax1.barh(y_pos, f1_scores, color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods)
    ax1.set_xlabel('F1 Score')
    ax1.set_title('F1 Score by Subcarrier Selection Method')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Good threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision vs Recall
    ax2 = axes[0, 1]
    for idx, row in results_df.iterrows():
        ax2.scatter(row['precision'], row['recall'], s=100, label=row['method'])
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.set_title('Precision vs Recall')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # 3. HR MAE comparison
    ax3 = axes[1, 0]
    hr_mae = results_df['hr_mae'].values
    colors = ['green' if mae == min(hr_mae) else 'blue' for mae in hr_mae]
    ax3.barh(y_pos, hr_mae, color=colors)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(methods)
    ax3.set_xlabel('HR MAE (BPM)')
    ax3.set_title('HR MAE (lower is better)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Number of peaks detected
    ax4 = axes[1, 1]
    n_peaks = results_df['n_peaks'].values
    ax4.barh(y_pos, n_peaks, color='purple')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(methods)
    ax4.set_xlabel('Number of Peaks Detected')
    ax4.set_title('Peaks Detected vs Ground Truth (135 beats)')
    ax4.axvline(x=135, color='red', linestyle='--', alpha=0.5, label='GT beats')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Subcarrier Selection Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/comparison_plot.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to '{args.output_dir}/comparison_plot.png'")
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - BEST PERFORMERS")
    print("="*70)
    print("\nTop 3 Methods by F1 Score:")
    for idx, row in results_df.head(3).iterrows():
        print(f"  {row['method']}: F1={row['f1']:.3f}, "
              f"P={row['precision']:.3f}, R={row['recall']:.3f}, "
              f"HR_MAE={row['hr_mae']:.1f}")
    
    # Recommendation
    best = results_df.iloc[0]
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"Best method: {best['method']}")
    print(f"  F1: {best['f1']:.3f}")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall: {best['recall']:.3f}")
    print(f"  HR MAE: {best['hr_mae']:.1f} BPM")
    print(f"  Subcarriers: {best['subcarriers']}")


if __name__ == "__main__":
    main()