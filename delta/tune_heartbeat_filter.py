# tune_heartbeat_filter.py
# Fine-Tune Bandpass Filter Systematically
# Usage: python tune_heartbeat_filter.py --csi <csi_file> --groundtruth <gt_file>

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from datetime import datetime

from read_csv_csi import CSIDataLoader
from read_pcap import PCAPDataLoader
from filters import CSIFilter, FilterConfig
from heartbeat_extraction import HeartbeatExtractor, HeartbeatConfig
from validate_heartbeat import HeartbeatValidator


def sweep_filter_parameters(csi_data, gt_df, metadata, output_dir='sweep_results'):
    """Sweep bandpass parameters to find optimal settings"""
    Path(output_dir).mkdir(exist_ok=True)
    results = []
    
    # Test ranges - expand for broader search
    low_freqs = np.arange(0.3, 1.5, 0.1)
    high_freqs = np.arange(1.5, 4.0, 0.15)
    orders = [2, 3, 4, 5]
    
    best_score = 0
    best_params = None
    best_metrics = {}
    total_combinations = 0
    
    print("\n" + "="*70)
    print("FILTER PARAMETER SWEEP")
    print("="*70)
    print(f"Low freqs:  {low_freqs}")
    print(f"High freqs: {high_freqs}")
    print(f"Orders:     {orders}")
    print(f"Total combos: {len(low_freqs) * len(high_freqs) * len(orders)}")
    print("="*70 + "\n")
    
    # Pre-filter CSI data once
    print("Pre-filtering CSI data (DC removal)...")
    filter_obj = CSIFilter(FilterConfig())
    filtered_csi = filter_obj.apply_all_filters(csi_data, verbose=False)
    
    for low in low_freqs:
        for high in high_freqs:
            if low >= high:
                continue
            for order in orders:
                try:
                    # Quick validation with current params
                    config = HeartbeatConfig()
                    config.bandpass_low = low
                    config.bandpass_high = high
                    config.filter_order = order
                    config.sampling_rate = metadata['rate']
                    config.use_all_subcarriers = False
                    
                    extractor = HeartbeatExtractor(config)
                    heartbeat_signal = extractor.apply_bandpass_filter(filtered_csi)
                    
                    # Detect peaks
                    peak_result = extractor.detect_peaks(heartbeat_signal)
                    csi_peaks = peak_result['peaks']
                    csi_beat_times = np.arange(len(heartbeat_signal)) / metadata['rate']
                    csi_beat_times = csi_beat_times[csi_peaks]
                    
                    # Ground truth times
                    if 'ts' in gt_df.columns:
                        gt_times = gt_df['ts'].values
                    else:
                        gt_times = np.arange(len(gt_df))
                    gt_times = gt_times - gt_times[0]
                    
                    # Find best alignment
                    best_offset = 0
                    best_matches = 0
                    
                    # Coarse search
                    offsets = np.arange(-10, 10, 0.1)
                    for offset in offsets:
                        aligned = csi_beat_times + offset
                        matches = 0
                        for gt_time in gt_times:
                            if np.any(np.abs(aligned - gt_time) < 0.15):
                                matches += 1
                        if matches > best_matches:
                            best_matches = matches
                            best_offset = offset
                    
                    # Calculate metrics with best alignment
                    aligned_beats = csi_beat_times + best_offset
                    
                    # Match beats
                    matched_pairs = []
                    used_gt = set()
                    
                    for i, csi_time in enumerate(aligned_beats):
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
                            matched_pairs.append({
                                'csi_idx': i,
                                'gt_idx': best_match,
                                'error': best_dist
                            })
                            used_gt.add(best_match)
                    
                    tp = len(matched_pairs)
                    fp = len(csi_beat_times) - tp
                    fn = len(gt_times) - tp
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # Heart rate accuracy
                    if len(matched_pairs) > 2:
                        csi_hr = 60.0 / (np.diff([m['error'] + 1 for m in matched_pairs]) / metadata['rate'])
                        # Use ground truth intervals
                        if len(gt_times) > 2:
                            gt_intervals = np.diff(gt_times)
                            gt_hr = 60.0 / gt_intervals
                            min_len = min(len(csi_hr), len(gt_hr))
                            if min_len > 1:
                                csi_hr = csi_hr[:min_len]
                                gt_hr = gt_hr[:min_len]
                                hr_mae = np.mean(np.abs(csi_hr - gt_hr))
                            else:
                                hr_mae = 999
                        else:
                            hr_mae = 999
                    else:
                        hr_mae = 999
                    
                    result = {
                        'low': low,
                        'high': high,
                        'order': order,
                        'tp': tp,
                        'fp': fp,
                        'fn': fn,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'hr_mae': hr_mae,
                        'matches': len(matched_pairs),
                        'offset': best_offset
                    }
                    results.append(result)
                    total_combinations += 1
                    
                    if f1 > best_score:
                        best_score = f1
                        best_params = (low, high, order)
                        best_metrics = result
                        best_metrics['heartbeat_signal'] = heartbeat_signal
                        best_metrics['peaks'] = csi_peaks
                        best_metrics['gt_times'] = gt_times
                        best_metrics['best_offset'] = best_offset
                        best_metrics['aligned_beats'] = aligned_beats
                    
                    # Print progress
                    status = "★ BEST" if f1 == best_score and f1 > 0 else " "
                    print(f"low={low:.1f} high={high:.2f} order={order}: "
                          f"F1={f1:.3f} (P={precision:.3f} R={recall:.3f}) "
                          f"HR_MAE={hr_mae:.1f} {status}")
                    
                except Exception as e:
                    print(f"Error with low={low}, high={high}, order={order}: {str(e)[:50]}")
                    continue
    
    print("\n" + "="*70)
    print("SWEEP COMPLETE!")
    print(f"Total combinations tested: {total_combinations}")
    print("="*70)
    
    if best_params is not None:
        print(f"\nBEST PARAMETERS:")
        print(f"  Low cutoff:  {best_params[0]:.1f} Hz")
        print(f"  High cutoff: {best_params[1]:.2f} Hz")
        print(f"  Order:       {best_params[2]}")
        print(f"  F1-Score:    {best_metrics['f1']:.3f}")
        print(f"  Precision:   {best_metrics['precision']:.3f}")
        print(f"  Recall:      {best_metrics['recall']:.3f}")
        print(f"  HR MAE:      {best_metrics['hr_mae']:.1f} BPM")
    
    # Save results
    results_df = pd.DataFrame(results)
    
    # Sort by F1 score
    results_df = results_df.sort_values('f1', ascending=False)
    results_df.to_csv(f'{output_dir}/sweep_results.csv', index=False)
    print(f"\nResults saved to '{output_dir}/sweep_results.csv'")
    
    # Create heatmap visualization
    create_sweep_heatmap(results_df, output_dir)
    
    return results, best_params, best_metrics


def create_sweep_heatmap(results_df, output_dir):
    """Create heatmap visualization of sweep results"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Pivot for heatmap (max F1 for each low/high combo)
    pivot_f1 = results_df.pivot_table(
        values='f1', 
        index='low', 
        columns='high', 
        aggfunc='max'
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Heatmap 1: F1 Score
    im1 = axes[0].imshow(pivot_f1.values, aspect='auto', cmap='RdYlGn', 
                         extent=[pivot_f1.columns.min(), pivot_f1.columns.max(), 
                                pivot_f1.index.min(), pivot_f1.index.max()],
                         origin='lower', vmin=0, vmax=0.7)
    axes[0].set_xlabel('High Cutoff (Hz)')
    axes[0].set_ylabel('Low Cutoff (Hz)')
    axes[0].set_title('F1 Score by Filter Parameters')
    plt.colorbar(im1, ax=axes[0], label='F1 Score')
    
    # Heatmap 2: Precision
    pivot_prec = results_df.pivot_table(
        values='precision', 
        index='low', 
        columns='high', 
        aggfunc='max'
    )
    im2 = axes[1].imshow(pivot_prec.values, aspect='auto', cmap='RdYlGn',
                         extent=[pivot_prec.columns.min(), pivot_prec.columns.max(),
                                pivot_prec.index.min(), pivot_prec.index.max()],
                         origin='lower', vmin=0, vmax=0.7)
    axes[1].set_xlabel('High Cutoff (Hz)')
    axes[1].set_ylabel('Low Cutoff (Hz)')
    axes[1].set_title('Precision by Filter Parameters')
    plt.colorbar(im2, ax=axes[1], label='Precision')
    
    # Heatmap 3: HR MAE
    pivot_hr = results_df.pivot_table(
        values='hr_mae', 
        index='low', 
        columns='high', 
        aggfunc='min'
    )
    im3 = axes[2].imshow(pivot_hr.values, aspect='auto', cmap='RdYlGn_r',
                         extent=[pivot_hr.columns.min(), pivot_hr.columns.max(),
                                pivot_hr.index.min(), pivot_hr.index.max()],
                         origin='lower', vmin=0, vmax=100)
    axes[2].set_xlabel('High Cutoff (Hz)')
    axes[2].set_ylabel('Low Cutoff (Hz)')
    axes[2].set_title('HR MAE by Filter Parameters (lower is better)')
    plt.colorbar(im3, ax=axes[2], label='HR MAE (BPM)')
    
    plt.suptitle('Filter Parameter Sweep Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/sweep_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to '{output_dir}/sweep_heatmap.png'")
    plt.close()
    
    # Also create parameter ranking bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    top_n = min(15, len(results_df))
    top_results = results_df.head(top_n)
    
    labels = [f"{row['low']:.1f}/{row['high']:.2f}/o{row['order']}" 
              for _, row in top_results.iterrows()]
    colors = ['green' if i == 0 else 'blue' for i in range(len(labels))]
    
    ax.barh(labels, top_results['f1'].values, color=colors)
    ax.set_xlabel('F1 Score')
    ax.set_title(f'Top {top_n} Parameter Combinations')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Good threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sweep_ranking.png', dpi=300, bbox_inches='tight')
    print(f"Ranking chart saved to '{output_dir}/sweep_ranking.png'")
    plt.close()


def create_best_filter_plot(best_metrics, output_dir):
    """Create detailed visualization of best filter configuration"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Heartbeat signal with peaks
    ax1 = axes[0, 0]
    heartbeat = best_metrics['heartbeat_signal']
    fs = 100  # Assuming 100 Hz
    time = np.arange(len(heartbeat)) / fs
    
    # Normalize
    heartbeat_norm = (heartbeat - np.mean(heartbeat)) / np.std(heartbeat)
    
    ax1.plot(time, heartbeat_norm, 'b-', alpha=0.7, linewidth=0.8)
    peaks = best_metrics['peaks']
    ax1.scatter(time[peaks], heartbeat_norm[peaks], c='red', s=20, label='CSI Peaks')
    
    # Ground truth
    gt_times = best_metrics['gt_times']
    for gt in gt_times:
        if gt < time[-1]:
            ax1.axvline(x=gt, color='green', alpha=0.3, linewidth=0.5)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Normalized Amplitude')
    ax1.set_title('Best Filter: Heartbeat Signal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Beat timing comparison
    ax2 = axes[0, 1]
    aligned = best_metrics['aligned_beats']
    gt_times = best_metrics['gt_times']
    
    # Match pairs
    pairs = []
    used_gt = set()
    for i, csi_time in enumerate(aligned):
        best_dist = np.inf
        best_j = None
        for j, gt_time in enumerate(gt_times):
            if j in used_gt:
                continue
            dist = abs(csi_time - gt_time)
            if dist < best_dist and dist < 0.2:
                best_dist = dist
                best_j = j
        if best_j is not None:
            pairs.append((gt_times[best_j], csi_time))
            used_gt.add(best_j)
    
    if pairs:
        gt_vals, csi_vals = zip(*pairs)
        ax2.scatter(gt_vals, csi_vals, alpha=0.6, s=30)
        min_val = min(min(gt_vals), min(csi_vals))
        max_val = max(max(gt_vals), max(csi_vals))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        ax2.set_xlabel('Ground Truth Time (s)')
        ax2.set_ylabel('CSI Detected Time (s)')
        ax2.set_title('Beat Timing Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add error stats
        errors = [abs(a-b) for a, b in pairs]
        ax2.text(0.05, 0.95, f"Mean Error: {np.mean(errors)*1000:.1f} ms", 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top')
    
    # 3. Frequency response
    ax3 = axes[1, 0]
    # Plot FFT of heartbeat signal
    from scipy.fft import fft, fftfreq
    n = len(heartbeat)
    fft_vals = np.abs(fft(heartbeat - np.mean(heartbeat)))
    freqs = fftfreq(n, 1/fs)
    
    pos_mask = (freqs > 0) & (freqs < 5)
    ax3.plot(freqs[pos_mask], fft_vals[pos_mask])
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Frequency Spectrum (0-5 Hz)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    metrics_text = f"""BEST FILTER CONFIGURATION
    
    Low Cutoff:     {best_metrics['low']:.1f} Hz
    High Cutoff:    {best_metrics['high']:.2f} Hz
    Filter Order:   {best_metrics['order']}
    
    PERFORMANCE:
    F1-Score:       {best_metrics['f1']:.3f}
    Precision:      {best_metrics['precision']:.3f}
    Recall:         {best_metrics['recall']:.3f}
    
    True Positives: {best_metrics['tp']}
    False Positives:{best_metrics['fp']}
    False Negatives:{best_metrics['fn']}
    
    HR MAE:         {best_metrics['hr_mae']:.1f} BPM
    """
    
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Best Filter Configuration Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/best_filter_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Best filter analysis saved to '{output_dir}/best_filter_analysis.png'")
    plt.close()


def load_csi_and_gt(csi_path, groundtruth_path):
    """Load CSI and ground truth data"""
    print(f"\nLoading CSI data from: {csi_path}")
    csi_path = Path(csi_path)
    
    if csi_path.suffix.lower() == '.csv':
        loader = CSIDataLoader(str(csi_path))
        csi_data, metadata = loader.load()
    else:
        loader = PCAPDataLoader(str(csi_path))
        csi_data, metadata = loader.load()
    
    print(f"  Loaded {metadata['n_packets']} packets at {metadata['rate']:.1f} Hz")
    
    print(f"\nLoading ground truth from: {groundtruth_path}")
    gt_df = pd.read_csv(groundtruth_path)
    print(f"  Loaded {len(gt_df)} beats")
    print(f"  Columns: {list(gt_df.columns)}")
    
    return csi_data, gt_df, metadata


def main():
    parser = argparse.ArgumentParser(
        description='Sweep filter parameters to find optimal heartbeat detection settings'
    )
    parser.add_argument('--csi', required=True, help='CSI data file (.pcap or .csv)')
    parser.add_argument('--groundtruth', required=True, help='Ground truth CSV file')
    parser.add_argument('--output-dir', default='sweep_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.csi).exists():
        print(f"Error: CSI file '{args.csi}' not found!")
        sys.exit(1)
    if not Path(args.groundtruth).exists():
        print(f"Error: Ground truth file '{args.groundtruth}' not found!")
        sys.exit(1)
    
    try:
        # Load data
        csi_data, gt_df, metadata = load_csi_and_gt(args.csi, args.groundtruth)
        
        # Run sweep
        results, best_params, best_metrics = sweep_filter_parameters(
            csi_data, gt_df, metadata, args.output_dir
        )
        
        # Create best filter visualization
        if best_metrics:
            best_metrics['low'] = best_params[0]
            best_metrics['high'] = best_params[1]
            best_metrics['order'] = best_params[2]
            create_best_filter_plot(best_metrics, args.output_dir)
        
        print("\n" + "="*70)
        print("SWEEP COMPLETE!")
        print("="*70)
        print(f"Results saved to '{args.output_dir}/'")
        print(f"  - sweep_results.csv: Full results table")
        print(f"  - sweep_heatmap.png: Heatmap visualization")
        print(f"  - sweep_ranking.png: Top parameter ranking")
        if best_params:
            print(f"  - best_filter_analysis.png: Best filter details")
            print(f"\nRecommended parameters:")
            print(f"  --heart-low {best_params[0]:.1f} --heart-high {best_params[1]:.2f} --heart-order {best_params[2]}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()