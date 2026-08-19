# run_best_method_adaptive.py
# Uses adaptive gain control instead of notch filter

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


def remove_amplitude_modulation(signal_data, fs):
    """
    Remove respiratory amplitude modulation by normalizing
    the signal envelope over time.
    """
    # Extract envelope using Hilbert transform
    analytic_signal = signal.hilbert(signal_data - np.mean(signal_data))
    envelope = np.abs(analytic_signal)
    
    # Smooth the envelope (remove high-frequency variations)
    window = int(0.5 * fs)  # 0.5 second smoothing
    if window < 3:
        window = 3
    kernel = np.ones(window) / window
    smooth_envelope = np.convolve(envelope, kernel, mode='same')
    
    # Avoid division by zero and clipping
    smooth_envelope = np.maximum(smooth_envelope, np.percentile(smooth_envelope, 10))
    
    # Normalize the signal by the envelope
    normalized_signal = (signal_data - np.mean(signal_data)) / smooth_envelope
    
    # Remove any remaining DC offset
    normalized_signal = normalized_signal - np.mean(normalized_signal)
    
    return normalized_signal, smooth_envelope


def quick_validate_signal(signal_data, gt_df, fs):
    """Quick validation of a signal against ground truth"""
    try:
        # Bandpass filter (best params: 1.0-3.15 Hz, order 2)
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
            'n_peaks': len(peaks)
        }
        
    except Exception as e:
        print(f"Error in validation: {e}")
        return {'f1': 0, 'precision': 0, 'recall': 0, 'hr_mae': 999, 'n_peaks': 0}


def run_adaptive_method(csi_path, gt_path, output_dir='best_results'):
    """Run adaptive method with amplitude normalization"""
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("ADAPTIVE METHOD: Amplitude Normalization")
    print("="*70)
    
    # Load data
    print(f"\nLoading CSI from: {csi_path}")
    loader = CSIDataLoader(str(csi_path))
    csi_data, metadata = loader.load()
    fs = metadata['rate']
    print(f"  {metadata['n_packets']} packets at {fs:.1f} Hz")
    
    print(f"Loading ground truth: {gt_path}")
    gt_df = pd.read_csv(str(gt_path))
    print(f"  {len(gt_df)} beats")
    
    # Apply basic filters
    print("\nApplying basic filters...")
    filter_obj = CSIFilter(FilterConfig())
    filtered_csi = filter_obj.apply_all_filters(csi_data, verbose=False)
    magnitudes = np.abs(filtered_csi)
    
    # Best subcarriers from PSD analysis
    best_subcarriers = [187, 2, 1, 188, 161, 159, 0, 162, 158, 163]
    print(f"Using PSD Top 10 subcarriers: {best_subcarriers}")
    
    # Test configurations
    configs = {
        'PSD Top 10 (baseline)': {
            'subcarriers': best_subcarriers,
            'normalize': False
        },
        'PSD Top 10 + Amplitude Norm': {
            'subcarriers': best_subcarriers,
            'normalize': True
        },
        'SC 187 + Amplitude Norm': {
            'subcarriers': [187],
            'normalize': True
        },
        'SC 2 + Amplitude Norm': {
            'subcarriers': [2],
            'normalize': True
        }
    }
    
    results = []
    
    for name, config in configs.items():
        print(f"\nTesting: {name}")
        
        # Extract signal from selected subcarriers
        sc_list = config['subcarriers']
        if len(sc_list) == 1:
            signal_data = magnitudes[:, sc_list[0]]
        else:
            signal_data = np.mean(magnitudes[:, sc_list], axis=1)
        
        # Remove amplitude modulation if requested
        if config['normalize']:
            signal_data, envelope = remove_amplitude_modulation(signal_data, fs)
            print(f"  Applied amplitude normalization")
        
        # Validate
        metrics = quick_validate_signal(signal_data, gt_df, fs)
        metrics['method'] = name
        results.append(metrics)
        
        print(f"  F1: {metrics['f1']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
        print(f"  HR MAE: {metrics['hr_mae']:.1f} BPM, Peaks: {metrics['n_peaks']}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1', ascending=False)
    results_df.to_csv(f'{output_dir}/adaptive_results.csv', index=False)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # F1 Score
    ax1 = axes[0]
    methods = results_df['method'].values
    f1_scores = results_df['f1'].values
    colors = ['green' if f1 == max(f1_scores) else 'blue' for f1 in f1_scores]
    y_pos = np.arange(len(methods))
    ax1.barh(y_pos, f1_scores, color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontsize=9)
    ax1.set_xlabel('F1 Score')
    ax1.set_title('Adaptive Method Comparison')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # HR MAE
    ax2 = axes[1]
    hr_mae = results_df['hr_mae'].values
    colors = ['green' if mae == min(hr_mae) else 'blue' for mae in hr_mae]
    ax2.barh(y_pos, hr_mae, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods, fontsize=9)
    ax2.set_xlabel('HR MAE (BPM)')
    ax2.set_title('HR MAE (lower is better)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Adaptive Amplitude Normalization Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/adaptive_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to '{output_dir}/adaptive_comparison.png'")
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nTop 3 Methods:")
    for idx, row in results_df.head(3).iterrows():
        print(f"  {row['method']}: F1={row['f1']:.3f}, "
              f"P={row['precision']:.3f}, R={row['recall']:.3f}, "
              f"HR_MAE={row['hr_mae']:.1f}")
    
    best = results_df.iloc[0]
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"Best method: {best['method']}")
    print(f"  F1: {best['f1']:.3f}")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall: {best['recall']:.3f}")
    print(f"  HR MAE: {best['hr_mae']:.1f} BPM")


def main():
    parser = argparse.ArgumentParser(description='Run adaptive method with amplitude normalization')
    parser.add_argument('--csi', required=True, help='CSI data file')
    parser.add_argument('--groundtruth', required=True, help='Ground truth CSV')
    parser.add_argument('--output-dir', default='best_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.csi).exists():
        print(f"Error: CSI file '{args.csi}' not found!")
        sys.exit(1)
    if not Path(args.groundtruth).exists():
        print(f"Error: Ground truth file '{args.groundtruth}' not found!")
        sys.exit(1)
    
    run_adaptive_method(args.csi, args.groundtruth, args.output_dir)


if __name__ == "__main__":
    main()