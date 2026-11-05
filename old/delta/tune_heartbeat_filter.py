# tune_heartbeat_filter_fast.py
# Optimized filter sweep - uses smarter search strategy
# Usage: python tune_heartbeat_filter_fast.py --csi <file> --groundtruth <file>

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
from read_pcap import PCAPDataLoader
from filters import CSIFilter, FilterConfig
from heartbeat_extraction import HeartbeatExtractor, HeartbeatConfig


def quick_validate(signal_data, gt_df, fs):
    """Quick validation without creating full validator object"""
    try:
        # Bandpass filter
        nyquist = fs / 2
        low = 0.8 / nyquist
        high = 2.5 / nyquist
        low = max(0.001, low)
        high = min(0.999, high)
        sos = signal.butter(3, [low, high], btype='band', output='sos')
        filtered = signal.sosfiltfilt(sos, signal_data - np.mean(signal_data))
        
        # Detect peaks
        peaks, _ = signal.find_peaks(
            filtered,
            distance=int(0.3 * fs),
            prominence=np.std(filtered) * 0.4,
            height=np.mean(filtered) + np.std(filtered) * 0.2
        )
        
        if len(peaks) < 3:
            return {'f1': 0, 'precision': 0, 'recall': 0, 'hr_mae': 999}
        
        csi_beat_times = np.arange(len(filtered)) / fs
        csi_beat_times = csi_beat_times[peaks]
        
        # Ground truth times
        if 'ts' in gt_df.columns:
            gt_times = gt_df['ts'].values
        else:
            gt_times = np.arange(len(gt_df))
        gt_times = gt_times - gt_times[0]
        
        # Quick alignment - coarse then fine
        best_offset = 0
        best_matches = 0
        
        # Coarse search
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
        
        # Heart rate MAE (only if enough beats)
        if len(matched_pairs) > 3 and len(gt_times) > 3:
            # Calculate HR from intervals
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
            'n_peaks': len(peaks)
        }
        
    except Exception as e:
        return {'f1': 0, 'precision': 0, 'recall': 0, 'hr_mae': 999}


def fast_sweep(csi_data, gt_df, metadata, output_dir='sweep_results'):
    """Fast sweep using smarter parameter ranges"""
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("FAST FILTER PARAMETER SWEEP")
    print("="*70)
    
    # First, find approximate heart rate using FFT
    magnitudes = np.abs(csi_data)
    mean_mag = np.mean(magnitudes, axis=1)
    
    # Remove DC
    mean_mag = mean_mag - np.mean(mean_mag)
    
    # FFT
    n = len(mean_mag)
    fft_vals = np.abs(np.fft.fft(mean_mag))
    freqs = np.fft.fftfreq(n, 1/metadata['rate'])
    
    # Find peak in 0.5-3.5 Hz range
    mask = (freqs > 0.5) & (freqs < 3.5)
    if np.any(mask):
        peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
        hr_estimate = peak_freq * 60
        print(f"Estimated heart rate from FFT: {hr_estimate:.1f} BPM ({peak_freq:.2f} Hz)")
    else:
        peak_freq = 1.2
        print("Could not estimate HR, using default 1.2 Hz")
    
    # Smart parameter ranges based on HR estimate
    # Low cutoff: 0.5x to 1.5x of HR frequency
    # High cutoff: 2x to 4x of HR frequency
    low_min = max(0.3, peak_freq * 0.5)
    low_max = min(1.5, peak_freq * 1.5)
    high_min = max(1.5, peak_freq * 2.0)
    high_max = min(4.5, peak_freq * 4.0)
    
    print(f"\nSmart parameter ranges:")
    print(f"  Low:  {low_min:.2f} - {low_max:.2f} Hz")
    print(f"  High: {high_min:.2f} - {high_max:.2f} Hz")
    
    # Use finer steps for better precision
    low_freqs = np.arange(low_min, low_max + 0.05, 0.1)
    high_freqs = np.arange(high_min, high_max + 0.05, 0.15)
    orders = [2, 3, 4]
    
    print(f"Testing {len(low_freqs) * len(high_freqs) * len(orders)} combinations...")
    
    results = []
    best_score = 0
    best_params = None
    best_metrics = None
    
    # Pre-filter CSI once
    filter_obj = CSIFilter(FilterConfig())
    filtered_csi = filter_obj.apply_all_filters(csi_data, verbose=False)
    
    total = 0
    for low in low_freqs:
        for high in high_freqs:
            if low >= high:
                continue
            for order in orders:
                total += 1
                try:
                    # Apply bandpass filter to each subcarrier
                    config = HeartbeatConfig()
                    config.bandpass_low = low
                    config.bandpass_high = high
                    config.filter_order = order
                    config.sampling_rate = metadata['rate']
                    
                    extractor = HeartbeatExtractor(config)
                    heartbeat = extractor.apply_bandpass_filter(filtered_csi)
                    
                    # Quick validate
                    metrics = quick_validate(heartbeat, gt_df, metadata['rate'])
                    
                    result = {
                        'low': low,
                        'high': high,
                        'order': order,
                        **metrics
                    }
                    results.append(result)
                    
                    if metrics['f1'] > best_score:
                        best_score = metrics['f1']
                        best_params = (low, high, order)
                        best_metrics = metrics
                        best_metrics['heartbeat'] = heartbeat
                    
                    # Print progress every 20 combos
                    if total % 20 == 0:
                        print(f"  Tested {total} combos... Best F1: {best_score:.3f}")
                    
                except Exception as e:
                    continue
    
    print(f"\nTotal tested: {total}")
    print("\n" + "="*70)
    print("BEST PARAMETERS:")
    print("="*70)
    print(f"  Low cutoff:  {best_params[0]:.2f} Hz")
    print(f"  High cutoff: {best_params[1]:.2f} Hz")
    print(f"  Order:       {best_params[2]}")
    print(f"  F1-Score:    {best_score:.3f}")
    print(f"  Precision:   {best_metrics['precision']:.3f}")
    print(f"  Recall:      {best_metrics['recall']:.3f}")
    print(f"  HR MAE:      {best_metrics['hr_mae']:.1f} BPM")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1', ascending=False)
    results_df.to_csv(f'{output_dir}/fast_sweep_results.csv', index=False)
    
    # Create visualizations
    create_sweep_plots(results_df, best_params, best_metrics, output_dir)
    
    return results, best_params, best_metrics


def create_sweep_plots(results_df, best_params, best_metrics, output_dir):
    """Create sweep visualizations"""
    
    # Filter results for heatmap
    pivot = results_df.pivot_table(values='f1', index='low', columns='high', aggfunc='max')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    im = axes[0].imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                        extent=[pivot.columns.min(), pivot.columns.max(),
                                pivot.index.min(), pivot.index.max()],
                        origin='lower', vmin=0, vmax=0.6)
    axes[0].set_xlabel('High Cutoff (Hz)')
    axes[0].set_ylabel('Low Cutoff (Hz)')
    axes[0].set_title('F1 Score Heatmap')
    plt.colorbar(im, ax=axes[0], label='F1 Score')
    
    # Mark best
    axes[0].scatter(best_params[1], best_params[0], c='red', s=100, 
                    marker='*', label='Best')
    axes[0].legend()
    
    # Top rankings
    top_n = min(10, len(results_df))
    top_results = results_df.head(top_n)
    
    labels = [f"{row['low']:.2f}/{row['high']:.2f}/o{row['order']}" 
              for _, row in top_results.iterrows()]
    colors = ['green' if i == 0 else 'blue' for i in range(len(labels))]
    
    axes[1].barh(labels, top_results['f1'].values, color=colors)
    axes[1].set_xlabel('F1 Score')
    axes[1].set_title(f'Top {top_n} Parameter Combinations')
    axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Fast Filter Parameter Sweep Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fast_sweep_results.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to '{output_dir}/fast_sweep_results.png'")
    plt.close()
    
    # Best filter analysis
    if 'heartbeat' in best_metrics:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        heartbeat = best_metrics['heartbeat']
        fs = 100  # Assuming 100 Hz
        
        # Normalize
        heartbeat_norm = (heartbeat - np.mean(heartbeat)) / np.std(heartbeat)
        time = np.arange(len(heartbeat)) / fs
        
        axes[0].plot(time, heartbeat_norm, 'b-', alpha=0.7, linewidth=0.8)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Normalized Amplitude')
        axes[0].set_title('Best Filter: Heartbeat Signal')
        axes[0].grid(True, alpha=0.3)
        
        # FFT
        n = len(heartbeat)
        fft_vals = np.abs(np.fft.fft(heartbeat - np.mean(heartbeat)))
        freqs = np.fft.fftfreq(n, 1/fs)
        mask = (freqs > 0) & (freqs < 5)
        
        axes[1].plot(freqs[mask], fft_vals[mask])
        axes[1].axvline(x=best_params[0], color='r', linestyle='--', label=f'Low: {best_params[0]:.2f}Hz')
        axes[1].axvline(x=best_params[1], color='g', linestyle='--', label=f'High: {best_params[1]:.2f}Hz')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Magnitude')
        axes[1].set_title('Frequency Spectrum with Filter Bands')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 5)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/best_filter_analysis_fast.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Fast filter parameter sweep')
    parser.add_argument('--csi', required=True, help='CSI data file')
    parser.add_argument('--groundtruth', required=True, help='Ground truth CSV')
    parser.add_argument('--output-dir', default='sweep_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading CSI from: {args.csi}")
    csi_path = Path(args.csi)
    if csi_path.suffix.lower() == '.csv':
        loader = CSIDataLoader(str(csi_path))
        csi_data, metadata = loader.load()
    else:
        loader = PCAPDataLoader(str(csi_path))
        csi_data, metadata = loader.load()
    
    print(f"  {metadata['n_packets']} packets at {metadata['rate']:.1f} Hz")
    
    print(f"Loading ground truth: {args.groundtruth}")
    gt_df = pd.read_csv(args.groundtruth)
    print(f"  {len(gt_df)} beats")
    
    # Run fast sweep
    results, best_params, best_metrics = fast_sweep(
        csi_data, gt_df, metadata, args.output_dir
    )
    
    print("\n" + "="*70)
    print("RECOMMENDED COMMAND:")
    print("="*70)
    print(f"python main.py {args.csi} --heart-low {best_params[0]:.2f} --heart-high {best_params[1]:.2f} --heart-order {best_params[2]}")


if __name__ == "__main__":
    main()