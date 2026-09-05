# fix_peak_detection.py
# Test different peak detection parameters

import numpy as np
import pandas as pd
from scipy import signal
from read_csv_csi import CSIDataLoader
from filters import CSIFilter, FilterConfig
import matplotlib.pyplot as plt

def test_peak_params(csi_path, gt_path, low=1.2, high=2.8, order=4):
    # Load data
    loader = CSIDataLoader(csi_path)
    csi_data, metadata = loader.load()
    fs = metadata['rate']
    
    gt_df = pd.read_csv(gt_path)
    gt_times = gt_df['ts'].values - gt_df['ts'].values[0]
    
    # Apply filters
    filter_obj = CSIFilter(FilterConfig())
    filtered_csi = filter_obj.apply_all_filters(csi_data, verbose=False)
    magnitudes = np.abs(filtered_csi)
    
    # Use PSD Top 3 subcarriers
    sc_list = [187, 2, 1]
    signal_data = np.mean(magnitudes[:, sc_list], axis=1)
    
    # Bandpass filter
    nyquist = fs / 2
    sos = signal.butter(order, [low/nyquist, high/nyquist], btype='band', output='sos')
    filtered = signal.sosfiltfilt(sos, signal_data - np.mean(signal_data))
    
    # Test different peak detection parameters
    param_sets = [
        # (min_distance, prominence_factor, height_factor)
        (0.25, 0.5, 0.3),   # Current-ish
        (0.30, 0.6, 0.4),   # Higher thresholds
        (0.35, 0.7, 0.5),   # Even higher
        (0.25, 0.8, 0.6),   # High prominence, low distance
        (0.30, 0.5, 0.5),   # Balanced
        (0.35, 0.6, 0.4),   # Conservative
    ]
    
    results = []
    
    for dist, prom_factor, height_factor in param_sets:
        min_dist = int(dist * fs)
        prom = np.std(filtered) * prom_factor
        height = np.mean(filtered) + np.std(filtered) * height_factor
        
        peaks, props = signal.find_peaks(filtered, distance=min_dist, 
                                         prominence=prom, height=height)
        
        csi_times = np.arange(len(filtered)) / fs
        csi_beat_times = csi_times[peaks]
        
        # Align
        best_offset = 0
        best_matches = 0
        for offset in np.arange(-10, 10, 0.2):
            aligned = csi_beat_times + offset
            matches = sum(1 for gt in gt_times if np.any(np.abs(aligned - gt) < 0.15))
            if matches > best_matches:
                best_matches = matches
                best_offset = offset
        
        aligned = csi_beat_times + best_offset
        
        # Match beats
        used_gt = set()
        matched = []
        for t in aligned:
            best = None
            best_dist = np.inf
            for j, gt in enumerate(gt_times):
                if j in used_gt:
                    continue
                d = abs(t - gt)
                if d < best_dist and d < 0.2:
                    best_dist = d
                    best = j
            if best is not None:
                matched.append(best_dist)
                used_gt.add(best)
        
        tp = len(matched)
        fp = len(aligned) - tp
        fn = len(gt_times) - tp
        
        precision = tp / (tp + fp) if (tp+fp) > 0 else 0
        recall = tp / (tp + fn) if (tp+fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0
        
        # HR calculation
        if len(matched) > 2:
            csi_hr = 60.0 / (np.diff(peaks[:len(matched)]) / fs)
            gt_hr = 60.0 / np.diff(gt_times[:len(matched)])
            min_len = min(len(csi_hr), len(gt_hr))
            hr_mae = np.mean(np.abs(csi_hr[:min_len] - gt_hr[:min_len])) if min_len > 1 else 999
            corr = np.corrcoef(csi_hr[:min_len], gt_hr[:min_len])[0,1] if min_len > 2 else 0
        else:
            hr_mae = 999
            corr = 0
        
        results.append({
            'min_dist': dist,
            'prom_factor': prom_factor,
            'height_factor': height_factor,
            'n_peaks': len(peaks),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'hr_mae': hr_mae,
            'correlation': corr
        })
        
        print(f"dist={dist:.2f}s, prom={prom_factor:.1f}σ, height={height_factor:.1f}σ")
        print(f"  Peaks: {len(peaks)}, F1: {f1:.3f}, HR_MAE: {hr_mae:.1f}, Corr: {corr:.3f}")
    
    # Show best
    df = pd.DataFrame(results)
    best = df.loc[df['f1'].idxmax()]
    print(f"\nBEST: dist={best['min_dist']:.2f}s, prom={best['prom_factor']:.1f}σ, height={best['height_factor']:.1f}σ")
    print(f"  F1: {best['f1']:.3f}, HR_MAE: {best['hr_mae']:.1f}, Corr: {best['correlation']:.3f}")
    
    return results

if __name__ == "__main__":
    test_peak_params(r"captures\Allen-S,0,1\csi_capture_20260820_135250.csv",
                     r"captures\Allen-S,0,1\ground_truth_20260820_135247.csv")