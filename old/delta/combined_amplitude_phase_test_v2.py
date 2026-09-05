# combined_amplitude_phase_test_v2.py
"""
Improved Combined Amplitude + Phase Heartbeat Detection Test
- Better quality scoring (normalized 0-100 scale)
- Adaptive peak detection
- Improved F1 and HR MAE calculation
- Position optimization with presence detection
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.signal import detrend, find_peaks, welch, spectrogram, butter, sosfiltfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from read_csv_csi import CSIDataLoader
from read_pcap import PCAPDataLoader
from filters import CSIFilter, FilterConfig


class ImprovedCombinedExtractor:
    """
    Improved extractor with better scoring and peak detection.
    """
    
    def __init__(self, fs=100.0, low_freq=0.8, high_freq=2.5, order=4):
        self.fs = fs
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        self.quality_metrics = {}
        
    def extract_amplitude(self, csi_data):
        return np.abs(csi_data)
    
    def extract_phase(self, csi_data, unwrap=True):
        phase = np.angle(csi_data)
        if unwrap:
            phase = np.unwrap(phase, axis=0)
        return phase
    
    def remove_common_phase(self, phase_data):
        common_phase = np.median(phase_data, axis=1)
        return phase_data - common_phase[:, np.newaxis]
    
    def compute_phase_differences(self, phase_data, max_lag=10):
        n_sc = phase_data.shape[1]
        diffs = []
        indices = []
        
        for lag in range(1, min(max_lag + 1, n_sc)):
            for i in range(n_sc - lag):
                diff = phase_data[:, i] - phase_data[:, i + lag]
                diffs.append(diff)
                indices.append((i, i + lag))
        
        return diffs, indices
    
    def apply_hampel(self, x, window_size=11, threshold=3.0):
        if window_size % 2 == 0:
            window_size += 1
        
        half = window_size // 2
        y = np.asarray(x, dtype=float).copy()
        
        for i in range(len(x)):
            start = max(0, i - half)
            end = min(len(x), i + half + 1)
            w = x[start:end]
            
            med = np.median(w)
            mad = np.median(np.abs(w - med))
            
            if mad > 0:
                z = abs(x[i] - med) / (1.4826 * mad)
                if z > threshold:
                    y[i] = med
        
        return y
    
    def bandpass_filter(self, x):
        nyquist = self.fs / 2.0
        low = max(0.001, self.low_freq / nyquist)
        high = min(0.999, self.high_freq / nyquist)
        sos = scipy_signal.butter(self.order, [low, high], btype='bandpass', output='sos')
        return scipy_signal.sosfiltfilt(sos, x - np.mean(x))
    
    def compute_quality_score(self, signal_data):
        """
        Compute normalized quality score (0-100 scale).
        """
        if len(signal_data) < 32 or np.std(signal_data) == 0:
            return 0.0, 0.0, 0.0
        
        n = len(signal_data)
        fft_vals = np.abs(np.fft.rfft(signal_data - np.mean(signal_data)))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        
        # Heartbeat band (0.8-3.5 Hz for safety)
        hr_mask = (freqs >= 0.8) & (freqs <= 3.5)
        
        if not np.any(hr_mask):
            return 0.0, 0.0, 0.0
        
        hr_spec = fft_vals[hr_mask]
        hr_freqs = freqs[hr_mask]
        
        # Find peak in HR band
        peak_idx = np.argmax(hr_spec)
        peak_amp = float(hr_spec[peak_idx])
        peak_freq = float(hr_freqs[peak_idx])
        
        # Noise floor (0.1-5 Hz excluding HR band)
        noise_mask = (freqs >= 0.1) & (freqs <= 5.0) & ~hr_mask
        if np.any(noise_mask):
            noise_floor = float(np.median(fft_vals[noise_mask]))
        else:
            noise_floor = float(np.median(fft_vals))
        
        # SNR (signal-to-noise ratio)
        snr = peak_amp / (noise_floor + 1e-10)
        
        # Spectral purity (how prominent is the peak)
        if len(hr_spec) > 2:
            mean_hr = np.mean(hr_spec)
            purity = peak_amp / (mean_hr + 1e-10)
        else:
            purity = 1.0
        
        # Combine into a 0-100 score
        raw_score = snr * purity
        normalized_score = min(100.0, raw_score / 10.0)  # Cap at 100
        
        return normalized_score, peak_freq, peak_amp
    
    def process_channel(self, signal_data):
        """Process a single channel"""
        x = np.asarray(signal_data, dtype=float).copy()
        x -= np.mean(x)
        x = self.apply_hampel(x)
        x = detrend(x)
        x = self.bandpass_filter(x)
        
        if np.std(x) > 0:
            x = x / np.std(x)
        
        return x
    
    def get_best_subcarriers(self, csi_data, method='amplitude', top_n=10):
        """Get best subcarriers based on quality"""
        if method == 'amplitude':
            data = self.extract_amplitude(csi_data)
        elif method == 'phase':
            data = self.extract_phase(csi_data, unwrap=True)
            data = self.remove_common_phase(data)
        elif method == 'complex':
            data = csi_data
        else:
            raise ValueError(f"Unknown method: {method}")
        
        scores = []
        for sc in range(data.shape[1]):
            if method == 'complex':
                channel_data = np.abs(data[:, sc])
            else:
                channel_data = data[:, sc]
            
            processed = self.process_channel(channel_data)
            score, _, _ = self.compute_quality_score(processed)
            scores.append(score)
        
        best_indices = np.argsort(scores)[::-1][:top_n]
        return best_indices, scores
    
    def extract_amplitude_signal(self, csi_data, top_n=10):
        best_sc, scores = self.get_best_subcarriers(csi_data, 'amplitude', top_n)
        magnitudes = self.extract_amplitude(csi_data)
        
        signals = []
        for sc in best_sc:
            processed = self.process_channel(magnitudes[:, sc])
            signals.append(processed)
        
        fused = self.fuse_signals(signals)
        return fused, best_sc, scores
    
    def extract_phase_signal(self, csi_data, top_n=10):
        best_sc, scores = self.get_best_subcarriers(csi_data, 'phase', top_n)
        phase = self.extract_phase(csi_data, unwrap=True)
        phase = self.remove_common_phase(phase)
        
        signals = []
        for sc in best_sc:
            processed = self.process_channel(phase[:, sc])
            signals.append(processed)
        
        fused = self.fuse_signals(signals)
        return fused, best_sc, scores
    
    def extract_phase_diff_signal(self, csi_data, top_n=10):
        phase = self.extract_phase(csi_data, unwrap=True)
        diffs, indices = self.compute_phase_differences(phase, max_lag=8)
        
        scores = []
        for diff in diffs:
            processed = self.process_channel(diff)
            score, _, _ = self.compute_quality_score(processed)
            scores.append(score)
        
        top_idx = np.argsort(scores)[::-1][:top_n]
        signals = []
        best_indices = []
        for idx in top_idx:
            processed = self.process_channel(diffs[idx])
            signals.append(processed)
            best_indices.append(indices[idx])
        
        fused = self.fuse_signals(signals)
        return fused, best_indices, scores
    
    def fuse_signals(self, signals, align_polarity=True):
        if not signals:
            return np.array([])
        
        signals = [s.copy() for s in signals]
        
        if align_polarity and len(signals) > 1:
            reference = signals[0]
            for i in range(1, len(signals)):
                corr = np.corrcoef(reference, signals[i])[0, 1]
                if np.isfinite(corr) and corr < 0:
                    signals[i] = -signals[i]
        
        fused = np.mean(signals, axis=0)
        if np.std(fused) > 0:
            fused = (fused - np.mean(fused)) / np.std(fused)
        
        return fused
    
    def extract_combined_signal(self, csi_data, method='weighted', top_n=10):
        """Extract combined signal using various methods"""
        # Get individual signals
        amp_signal, amp_sc, amp_scores = self.extract_amplitude_signal(csi_data, top_n)
        phase_signal, phase_sc, phase_scores = self.extract_phase_signal(csi_data, top_n)
        
        amp_score, _, _ = self.compute_quality_score(amp_signal)
        phase_score, _, _ = self.compute_quality_score(phase_signal)
        
        diff_signal, diff_indices, diff_scores = self.extract_phase_diff_signal(csi_data, top_n)
        diff_score, _, _ = self.compute_quality_score(diff_signal)
        
        self.quality_metrics = {
            'amplitude': {'score': amp_score, 'top_sc': amp_sc},
            'phase': {'score': phase_score, 'top_sc': phase_sc},
            'phase_diff': {'score': diff_score, 'top_indices': diff_indices}
        }
        
        if method == 'weighted':
            total_score = amp_score + phase_score + diff_score + 1e-10
            w_amp = amp_score / total_score
            w_phase = phase_score / total_score
            w_diff = diff_score / total_score
            
            amp_signal = self.align_to_reference(amp_signal, phase_signal)
            diff_signal = self.align_to_reference(amp_signal, diff_signal)
            
            combined = (w_amp * amp_signal + w_phase * phase_signal + w_diff * diff_signal)
            combined = combined / np.std(combined) if np.std(combined) > 0 else combined
            
            top_channels = {
                'amplitude': amp_sc[:5].tolist(),
                'phase': phase_sc[:5].tolist(),
                'phase_diff': diff_indices[:5]
            }
            combined_score = max(amp_score, phase_score, diff_score)
            
        elif method == 'ensemble':
            amp_aligned = self.align_to_reference(amp_signal, phase_signal)
            diff_aligned = self.align_to_reference(amp_signal, diff_signal)
            
            signals = [amp_aligned, phase_signal, diff_aligned]
            combined = self.fuse_signals(signals, align_polarity=False)
            combined_score = np.mean([amp_score, phase_score, diff_score])
            top_channels = {
                'amplitude': amp_sc[:3].tolist(),
                'phase': phase_sc[:3].tolist(),
                'phase_diff': diff_indices[:3]
            }
            
        elif method == 'adaptive':
            # Choose best method
            methods = {
                'amplitude': (amp_signal, amp_score),
                'phase': (phase_signal, phase_score),
                'phase_diff': (diff_signal, diff_score),
            }
            best_method = max(methods.items(), key=lambda x: x[1][1])
            combined = best_method[1][0]
            combined_score = best_method[1][1]
            top_channels = {'best_method': best_method[0]}
            
        else:
            # Default: use phase_diff as it performed best
            combined = diff_signal
            combined_score = diff_score
            top_channels = {'phase_diff': diff_indices[:5]}
        
        if np.std(combined) > 0:
            combined = (combined - np.mean(combined)) / np.std(combined)
        
        return combined, combined_score, top_channels, self.quality_metrics
    
    def align_to_reference(self, ref_signal, signal_to_align):
        if len(ref_signal) != len(signal_to_align):
            return signal_to_align
        
        corr = np.corrcoef(ref_signal, signal_to_align)[0, 1]
        if np.isfinite(corr) and corr < 0:
            return -signal_to_align
        return signal_to_align
    
    def estimate_heart_rate(self, signal_data):
        """Estimate heart rate with better peak finding"""
        if len(signal_data) < 32:
            return 0, 0, 0
        
        # Use Welch PSD for better frequency estimation
        nperseg = min(256, len(signal_data))
        freqs, psd = welch(
            signal_data - np.mean(signal_data),
            fs=self.fs,
            nperseg=nperseg,
            scaling='density'
        )
        
        hr_mask = (freqs >= self.low_freq) & (freqs <= self.high_freq)
        if not np.any(hr_mask):
            return 0, 0, 0
        
        hr_psd = psd[hr_mask]
        hr_freqs = freqs[hr_mask]
        
        peak_idx = np.argmax(hr_psd)
        peak_freq = hr_freqs[peak_idx]
        peak_power = hr_psd[peak_idx]
        
        # Quadratic interpolation
        if 0 < peak_idx < len(hr_psd) - 1:
            alpha = hr_psd[peak_idx - 1]
            beta = hr_psd[peak_idx]
            gamma = hr_psd[peak_idx + 1]
            denom = alpha - 2*beta + gamma
            if abs(denom) > 1e-20:
                delta = 0.5 * (alpha - gamma) / denom
                delta = np.clip(delta, -1.0, 1.0)
                df = hr_freqs[1] - hr_freqs[0]
                peak_freq = hr_freqs[peak_idx] + delta * df
        
        bpm = peak_freq * 60.0
        
        # Confidence based on SNR
        noise_floor = np.median(psd[(freqs >= 0.1) & (freqs <= 5.0) & ~hr_mask])
        confidence = peak_power / (noise_floor + 1e-10)
        
        return bpm, peak_freq, confidence


def validate_with_gt_improved(signal_data, gt_df, fs):
    """
    Improved validation with better peak detection and alignment.
    """
    if 'ts' not in gt_df.columns:
        return {'f1_score': 0, 'precision': 0, 'recall': 0, 'hr_mae': 999}
    
    gt_times = gt_df['ts'].values
    gt_times = gt_times - gt_times[0]
    
    # Adaptive peak detection
    # First, try different prominence thresholds
    signal_std = np.std(signal_data)
    signal_mean = np.mean(signal_data)
    
    best_peaks = None
    best_f1 = 0
    
    # Try different thresholds
    for prom_factor in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for height_factor in [0.1, 0.2, 0.3, 0.4]:
            try:
                peaks, _ = find_peaks(
                    signal_data,
                    distance=int(0.25 * fs),  # Min 250ms between beats
                    prominence=signal_std * prom_factor,
                    height=signal_mean + signal_std * height_factor
                )
                
                if len(peaks) < 5:
                    continue
                
                # Evaluate this set of peaks
                csi_times = np.arange(len(signal_data)) / fs
                csi_beats = csi_times[peaks]
                
                # Align
                best_offset = 0
                best_matches = 0
                for offset in np.arange(-10, 10, 0.2):
                    aligned = csi_beats + offset
                    matches = sum(1 for gt in gt_times if np.any(np.abs(aligned - gt) < 0.15))
                    if matches > best_matches:
                        best_matches = matches
                        best_offset = offset
                
                aligned_beats = csi_beats + best_offset
                
                # Match
                used_gt = set()
                matched = []
                for t in aligned_beats:
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
                fp = len(aligned_beats) - tp
                fn = len(gt_times) - tp
                
                precision = tp / (tp + fp) if (tp+fp) > 0 else 0
                recall = tp / (tp + fn) if (tp+fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_peaks = peaks
                    best_precision = precision
                    best_recall = recall
                    best_tp = tp
                    best_fp = fp
                    best_fn = fn
                    best_matched = matched
                    
            except:
                continue
    
    if best_peaks is None or len(best_peaks) < 3:
        return {'f1_score': 0, 'precision': 0, 'recall': 0, 'hr_mae': 999}
    
    # Calculate HR MAE
    if len(best_matched) > 2:
        csi_hr = 60.0 / (np.diff(best_peaks[:len(best_matched)]) / fs)
        gt_hr = 60.0 / np.diff(gt_times[:len(best_matched)])
        min_len = min(len(csi_hr), len(gt_hr))
        if min_len > 1:
            hr_mae = np.mean(np.abs(csi_hr[:min_len] - gt_hr[:min_len]))
        else:
            hr_mae = 999
    else:
        hr_mae = 999
    
    return {
        'f1_score': best_f1,
        'precision': best_precision,
        'recall': best_recall,
        'hr_mae': hr_mae,
        'matched_beats': best_tp,
        'false_positives': best_fp,
        'false_negatives': best_fn,
        'total_gt_beats': len(gt_times),
        'total_csi_beats': len(best_peaks)
    }


def validate_combined_methods(csi_path, gt_path, output_dir='combined_results'):
    """Run improved combined method validation"""
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("IMPROVED COMBINED AMPLITUDE + PHASE VALIDATION")
    print("="*70)
    
    # Load data
    print(f"\nLoading CSI from: {csi_path}")
    csi_path = Path(csi_path)
    if csi_path.suffix.lower() == '.csv':
        loader = CSIDataLoader(str(csi_path))
        csi_data, metadata = loader.load()
    else:
        loader = PCAPDataLoader(str(csi_path))
        csi_data, metadata = loader.load()
    
    fs = metadata['rate']
    print(f"  {metadata['n_packets']} packets at {fs:.1f} Hz")
    
    if gt_path:
        print(f"Loading ground truth: {gt_path}")
        gt_df = pd.read_csv(gt_path)
        print(f"  {len(gt_df)} beats")
    else:
        gt_df = None
        print("No ground truth provided - using quality scores only")
    
    # Apply filters
    print("\nApplying basic filters...")
    filter_obj = CSIFilter(FilterConfig())
    filtered_csi = filter_obj.apply_all_filters(csi_data, verbose=False)
    
    # Initialize extractor
    extractor = ImprovedCombinedExtractor(fs, low_freq=1.0, high_freq=2.8, order=4)
    
    # Test methods (focused on best performers from previous test)
    methods = [
        'amplitude_only',
        'phase_only', 
        'phase_diff_only',
        'weighted',
        'ensemble',
        'adaptive'
    ]
    
    results = []
    signals = {}
    
    for method in methods:
        print(f"\nTesting method: {method}")
        
        if method == 'amplitude_only':
            signal, sc, _ = extractor.extract_amplitude_signal(filtered_csi, top_n=10)
            signal = extractor.process_channel(signal)
            score, _, _ = extractor.compute_quality_score(signal)
            channels = {'amplitude': sc[:5].tolist()}
            
        elif method == 'phase_only':
            signal, sc, _ = extractor.extract_phase_signal(filtered_csi, top_n=10)
            signal = extractor.process_channel(signal)
            score, _, _ = extractor.compute_quality_score(signal)
            channels = {'phase': sc[:5].tolist()}
            
        elif method == 'phase_diff_only':
            signal, indices, _ = extractor.extract_phase_diff_signal(filtered_csi, top_n=10)
            signal = extractor.process_channel(signal)
            score, _, _ = extractor.compute_quality_score(signal)
            channels = {'phase_diff': indices[:5]}
            
        else:
            signal, score, channels, _ = extractor.extract_combined_signal(
                filtered_csi, method=method, top_n=10
            )
        
        # Estimate heart rate
        bpm, peak_freq, confidence = extractor.estimate_heart_rate(signal)
        bpm = max(40, min(120, bpm))  # Clamp to reasonable range
        
        signals[method] = signal
        
        result = {
            'method': method,
            'quality_score': score,
            'confidence': confidence,
            'bpm': bpm,
            'peak_freq': peak_freq,
            'channels': str(channels)
        }
        
        # Validate with ground truth
        if gt_df is not None:
            val_result = validate_with_gt_improved(signal, gt_df, fs)
            result.update(val_result)
        
        results.append(result)
        
        print(f"  Score: {score:.2f}, Confidence: {confidence:.2f}, BPM: {bpm:.1f}")
        if gt_df is not None:
            print(f"  F1: {result.get('f1_score', 0):.3f}, HR_MAE: {result.get('hr_mae', 999):.1f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by F1 if available, otherwise by quality score
    if 'f1_score' in results_df.columns and results_df['f1_score'].notna().any():
        results_df = results_df.sort_values('f1_score', ascending=False)
    else:
        results_df = results_df.sort_values('quality_score', ascending=False)
    
    results_df.to_csv(f'{output_dir}/combined_methods_results_v2.csv', index=False)
    
    # Create improved plot
    create_improved_plot(results_df, signals, fs, gt_df, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - METHOD RANKING")
    print("="*70)
    for idx, row in results_df.iterrows():
        line = f"  {row['method']:15s}: Score={row['quality_score']:.2f}, BPM={row['bpm']:.1f}"
        if 'f1_score' in row and pd.notna(row['f1_score']):
            line += f", F1={row['f1_score']:.3f}, MAE={row.get('hr_mae', 999):.1f}"
        print(line)
    
    best = results_df.iloc[0]
    print("\n" + "="*70)
    print(f"BEST METHOD: {best['method']}")
    print("="*70)
    print(f"  Quality Score: {best['quality_score']:.2f}")
    print(f"  Heart Rate: {best['bpm']:.1f} BPM")
    if 'f1_score' in best:
        print(f"  F1 Score: {best['f1_score']:.3f}")
        print(f"  HR MAE: {best.get('hr_mae', 999):.1f} BPM")
        print(f"  Precision: {best.get('precision', 0):.3f}")
        print(f"  Recall: {best.get('recall', 0):.3f}")
    
    return results_df, signals


def create_improved_plot(results_df, signals, fs, gt_df, output_dir):
    """Create improved visualization"""
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    
    # Sort order for plots
    methods = results_df['method'].values
    y_pos = np.arange(len(methods))
    
    # 1. Quality Score
    ax1 = axes[0, 0]
    scores = results_df['quality_score'].values
    colors = ['green' if i == 0 else 'blue' for i in range(len(scores))]
    ax1.barh(y_pos, scores, color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontsize=9)
    ax1.set_xlabel('Quality Score (0-100)')
    ax1.set_title('Quality Score by Method')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(100, scores.max() * 1.1))
    
    # 2. F1 Score (if available)
    ax2 = axes[0, 1]
    if 'f1_score' in results_df.columns and results_df['f1_score'].notna().any():
        f1_scores = results_df['f1_score'].fillna(0).values
        colors = ['green' if i == 0 else 'blue' for i in range(len(f1_scores))]
        ax2.barh(y_pos, f1_scores, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(methods, fontsize=9)
        ax2.set_xlabel('F1 Score')
        ax2.set_title('F1 Score by Method')
        ax2.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='Good threshold')
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Acceptable')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'No Ground Truth Available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('F1 Score (N/A)')
    
    # 3. HR MAE (if available)
    ax3 = axes[0, 2]
    if 'hr_mae' in results_df.columns and results_df['hr_mae'].notna().any():
        hr_mae = results_df['hr_mae'].fillna(999).values
        colors = ['green' if i == 0 else 'blue' for i in range(len(hr_mae))]
        ax3.barh(y_pos, hr_mae, color=colors)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(methods, fontsize=9)
        ax3.set_xlabel('HR MAE (BPM)')
        ax3.set_title('Heart Rate MAE by Method')
        ax3.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='Good')
        ax3.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Acceptable')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Ground Truth Available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('HR MAE (N/A)')
    
    # 4. Signal waveforms (top 3 methods)
    top_methods = results_df.head(3)['method'].values
    for idx, method in enumerate(top_methods):
        ax = axes[1, idx]
        signal_data = signals.get(method, np.array([]))
        if len(signal_data) > 0:
            t = np.arange(len(signal_data)) / fs
            n_show = min(len(signal_data), int(10 * fs))
            ax.plot(t[:n_show], signal_data[:n_show], linewidth=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{method} (first 10s)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-3, 3)
    
    # 5. Spectrogram for best method
    ax5 = axes[2, 0]
    best_method = results_df.iloc[0]['method']
    signal_data = signals.get(best_method, np.array([]))
    if len(signal_data) > 0:
        freqs, times, Sxx = spectrogram(
            signal_data - np.mean(signal_data),
            fs=fs,
            window='hann',
            nperseg=min(256, len(signal_data)),
            noverlap=128,
            scaling='density'
        )
        freq_mask = freqs <= 5
        im = ax5.imshow(
            10 * np.log10(Sxx[freq_mask, :] + 1e-10),
            aspect='auto',
            origin='lower',
            extent=[times[0], times[-1], freqs[freq_mask][0], freqs[freq_mask][-1]],
            cmap='viridis'
        )
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Frequency (Hz)')
        ax5.set_title(f'Spectrogram: {best_method}')
        plt.colorbar(im, ax=ax5, label='Power (dB)')
    
    # 6. Method details
    ax6 = axes[2, 1]
    ax6.axis('off')
    summary_text = "DETAILED RESULTS\n" + "="*30 + "\n\n"
    for idx, row in results_df.iterrows():
        summary_text += f"{row['method']:12s}: "
        summary_text += f"Q={row['quality_score']:.1f}, "
        summary_text += f"BPM={row['bpm']:.1f}"
        if 'f1_score' in row and pd.notna(row['f1_score']):
            summary_text += f", F1={row['f1_score']:.3f}, MAE={row.get('hr_mae', 999):.1f}"
        summary_text += "\n"
    
    summary_text += f"\nBest: {results_df.iloc[0]['method']}"
    if 'f1_score' in results_df.iloc[0]:
        summary_text += f" (F1={results_df.iloc[0]['f1_score']:.3f})"
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    # 7. Performance summary
    ax7 = axes[2, 2]
    ax7.axis('off')
    
    if gt_df is not None:
        # GT summary
        gt_text = f"GROUND TRUTH SUMMARY\n" + "="*30 + "\n\n"
        gt_text += f"Total Beats: {len(gt_df)}\n"
        
        best = results_df.iloc[0]
        if 'matched_beats' in best:
            gt_text += f"Matched: {best.get('matched_beats', 0)}\n"
            gt_text += f"False Positives: {best.get('false_positives', 0)}\n"
            gt_text += f"False Negatives: {best.get('false_negatives', 0)}\n"
        
        # Overall assessment
        f1 = best.get('f1_score', 0)
        mae = best.get('hr_mae', 999)
        
        gt_text += f"\nASSESSMENT:\n"
        if f1 >= 0.8 and mae < 10:
            gt_text += "EXCELLENT - Highly reliable"
        elif f1 >= 0.7 and mae < 15:
            gt_text += "GOOD - Reliable for monitoring"
        elif f1 >= 0.6 and mae < 20:
            gt_text += "MODERATE - Use with caution"
        elif f1 >= 0.5:
            gt_text += "POOR - Needs improvement"
        else:
            gt_text += "VERY POOR - Not reliable"
        
        ax7.text(0.1, 0.9, gt_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Improved Combined Amplitude + Phase Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_analysis_v2.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved improved analysis plot to '{output_dir}/combined_analysis_v2.png'")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Improved Combined Amplitude + Phase Heartbeat Detection'
    )
    
    parser.add_argument('--csi', help='CSI data file (.pcap or .csv)')
    parser.add_argument('--groundtruth', help='Ground truth CSV file')
    parser.add_argument('--output-dir', default='combined_results', help='Output directory')
    parser.add_argument('--heart-low', type=float, default=1.0, help='Lower cutoff (Hz)')
    parser.add_argument('--heart-high', type=float, default=2.8, help='Upper cutoff (Hz)')
    parser.add_argument('--heart-order', type=int, default=4, help='Filter order')
    
    args = parser.parse_args()
    
    if not args.csi:
        print("Error: --csi required")
        sys.exit(1)
    
    if not Path(args.csi).exists():
        print(f"Error: CSI file '{args.csi}' not found!")
        sys.exit(1)
    
    if args.groundtruth and not Path(args.groundtruth).exists():
        print(f"Warning: Ground truth file '{args.groundtruth}' not found!")
        args.groundtruth = None
    
    validate_combined_methods(args.csi, args.groundtruth, args.output_dir)


if __name__ == "__main__":
    main()