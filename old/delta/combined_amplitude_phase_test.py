# combined_amplitude_phase_test.py - FIXED VERSION
"""
Combined Amplitude + Phase Heartbeat Detection Test

Evaluates different fusion strategies for combining amplitude and phase
information to improve heartbeat detection accuracy. Also detects
whether a person is present and the signal quality for different positions.
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


class CombinedAmplitudePhaseExtractor:
    """
    Extract and combine amplitude and phase information from CSI.
    
    Fusion strategies:
    1. 'weighted': Weighted combination based on SNR
    2. 'product': Product of amplitude and phase signals
    3. 'complex': Use complex CSI directly
    4. 'adaptive': Adaptively choose best combination
    5. 'ensemble': Ensemble average of multiple methods
    """
    
    def __init__(self, fs=100.0, low_freq=0.8, high_freq=2.5, order=4):
        self.fs = fs
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        self.quality_metrics = {}
        
    def extract_amplitude(self, csi_data):
        """Extract amplitude from complex CSI"""
        return np.abs(csi_data)
    
    def extract_phase(self, csi_data, unwrap=True):
        """Extract phase from complex CSI"""
        phase = np.angle(csi_data)
        if unwrap:
            phase = np.unwrap(phase, axis=0)
        return phase
    
    def remove_common_phase(self, phase_data):
        """Remove common phase trend across subcarriers"""
        common_phase = np.median(phase_data, axis=1)
        return phase_data - common_phase[:, np.newaxis]
    
    def compute_phase_differences(self, phase_data, max_lag=10):
        """Compute phase differences between subcarriers"""
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
        """Apply Hampel filter for outlier removal"""
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
        """Apply bandpass filter"""
        nyquist = self.fs / 2.0
        
        low = max(0.001, self.low_freq / nyquist)
        high = min(0.999, self.high_freq / nyquist)
        
        sos = scipy_signal.butter(self.order, [low, high], btype='bandpass', output='sos')
        return scipy_signal.sosfiltfilt(sos, x - np.mean(x))
    
    def compute_quality_score(self, signal_data):
        """
        Compute quality score for a signal.
        Higher score = better heartbeat signal.
        """
        if len(signal_data) < 32 or np.std(signal_data) == 0:
            return 0.0, 0.0, 0.0
        
        # FFT
        n = len(signal_data)
        fft_vals = np.abs(np.fft.rfft(signal_data - np.mean(signal_data)))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        
        # Heartbeat band
        hr_mask = (freqs >= self.low_freq) & (freqs <= self.high_freq)
        
        if not np.any(hr_mask):
            return 0.0, 0.0, 0.0
        
        # Peak in HR band
        hr_spec = fft_vals[hr_mask]
        hr_freqs = freqs[hr_mask]
        
        peak_idx = np.argmax(hr_spec)
        peak_amp = float(hr_spec[peak_idx])
        peak_freq = float(hr_freqs[peak_idx])
        
        # Noise floor
        noise_mask = (freqs >= 0.1) & (freqs <= 5.0) & ~hr_mask
        if np.any(noise_mask):
            noise_floor = float(np.median(fft_vals[noise_mask]))
        else:
            noise_floor = float(np.median(fft_vals))
        
        # SNR-like score
        score = peak_amp / (noise_floor + 1e-10)
        
        # Also compute spectral purity (peak prominence)
        if len(hr_spec) > 2:
            mean_hr = np.mean(hr_spec)
            purity = peak_amp / (mean_hr + 1e-10)
        else:
            purity = 1.0
        
        combined_score = score * (1.0 + 0.5 * purity)
        
        return combined_score, peak_freq, peak_amp
    
    def process_channel(self, signal_data):
        """Process a single channel through the full pipeline"""
        x = np.asarray(signal_data, dtype=float).copy()
        x -= np.mean(x)
        x = self.apply_hampel(x)
        x = detrend(x)
        x = self.bandpass_filter(x)
        
        if np.std(x) > 0:
            x = x / np.std(x)
        
        return x
    
    def get_best_subcarriers(self, csi_data, method='amplitude', top_n=10):
        """
        Get best subcarriers based on quality scoring.
        """
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
                # For complex, use magnitude
                channel_data = np.abs(data[:, sc])
            else:
                channel_data = data[:, sc]
            
            processed = self.process_channel(channel_data)
            score, _, _ = self.compute_quality_score(processed)
            scores.append(score)
        
        best_indices = np.argsort(scores)[::-1][:top_n]
        return best_indices, scores
    
    def extract_amplitude_signal(self, csi_data, top_n=10):
        """Extract best amplitude signal"""
        best_sc, scores = self.get_best_subcarriers(csi_data, 'amplitude', top_n)
        magnitudes = self.extract_amplitude(csi_data)
        
        # Get top signals
        signals = []
        for sc in best_sc:
            processed = self.process_channel(magnitudes[:, sc])
            signals.append(processed)
        
        # Fuse with polarity alignment
        fused = self.fuse_signals(signals)
        return fused, best_sc, scores
    
    def extract_phase_signal(self, csi_data, top_n=10):
        """Extract best phase signal (with common phase removal)"""
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
        """Extract best phase difference signal"""
        phase = self.extract_phase(csi_data, unwrap=True)
        diffs, indices = self.compute_phase_differences(phase, max_lag=8)
        
        # Score all differences
        scores = []
        for diff in diffs:
            processed = self.process_channel(diff)
            score, _, _ = self.compute_quality_score(processed)
            scores.append(score)
        
        # Get top differences
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
        """Fuse multiple signals with polarity alignment"""
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
        """
        Extract heartbeat signal using combined amplitude + phase.
        
        Args:
            csi_data: Complex CSI data (packets x subcarriers)
            method: Fusion method
            top_n: Number of channels to use
        
        Returns:
            heartbeat_signal, quality_score, top_channels, metadata
        """
        # Get individual signals
        amp_signal, amp_sc, amp_scores = self.extract_amplitude_signal(csi_data, top_n)
        phase_signal, phase_sc, phase_scores = self.extract_phase_signal(csi_data, top_n)
        
        # Compute quality scores for each
        amp_score, _, _ = self.compute_quality_score(amp_signal)
        phase_score, _, _ = self.compute_quality_score(phase_signal)
        
        # Also get phase difference
        diff_signal, diff_indices, diff_scores = self.extract_phase_diff_signal(csi_data, top_n)
        diff_score, _, _ = self.compute_quality_score(diff_signal)
        
        # Store metrics
        self.quality_metrics = {
            'amplitude': {'score': amp_score, 'top_sc': amp_sc},
            'phase': {'score': phase_score, 'top_sc': phase_sc},
            'phase_diff': {'score': diff_score, 'top_indices': diff_indices}
        }
        
        # Combine based on method
        if method == 'weighted':
            # Weighted combination based on quality scores
            total_score = amp_score + phase_score + diff_score + 1e-10
            w_amp = amp_score / total_score
            w_phase = phase_score / total_score
            w_diff = diff_score / total_score
            
            # Ensure signals are aligned
            amp_signal = self.align_to_reference(amp_signal, phase_signal)
            diff_signal = self.align_to_reference(amp_signal, diff_signal)
            
            combined = (w_amp * amp_signal + w_phase * phase_signal + w_diff * diff_signal)
            combined = combined / np.std(combined) if np.std(combined) > 0 else combined
            
            top_channels = {
                'amplitude': amp_sc[:5].tolist(),
                'phase': phase_sc[:5].tolist(),
                'phase_diff': diff_indices[:5]
            }
            combined_score = np.max([amp_score, phase_score, diff_score])
            
        elif method == 'product':
            # Product of amplitude and phase (enhances coherent components)
            amp_aligned = self.align_to_reference(amp_signal, phase_signal)
            combined = amp_aligned * phase_signal
            combined = self.process_channel(combined)
            top_channels = {'amplitude': amp_sc[:5].tolist(), 'phase': phase_sc[:5].tolist()}
            combined_score = (amp_score * phase_score) ** 0.5
            
        elif method == 'complex':
            # Use complex CSI directly
            best_sc, scores = self.get_best_subcarriers(csi_data, 'complex', top_n)
            signals = []
            for sc in best_sc:
                processed = self.process_channel(np.abs(csi_data[:, sc]))
                signals.append(processed)
            combined = self.fuse_signals(signals)
            combined_score, _, _ = self.compute_quality_score(combined)
            top_channels = {'complex': best_sc[:5].tolist()}
            
        elif method == 'ensemble':
            # Ensemble average of all methods
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
            # Choose best method based on quality scores
            methods = {
                'amplitude': (amp_signal, amp_score),
                'phase': (phase_signal, phase_score),
                'phase_diff': (diff_signal, diff_score),
                'weighted': None,  # Will compute
            }
            
            # Compute weighted as well
            total_score = amp_score + phase_score + diff_score + 1e-10
            w_amp = amp_score / total_score
            w_phase = phase_score / total_score
            w_diff = diff_score / total_score
            
            amp_aligned = self.align_to_reference(amp_signal, phase_signal)
            diff_aligned = self.align_to_reference(amp_signal, diff_signal)
            weighted = (w_amp * amp_aligned + w_phase * phase_signal + w_diff * diff_aligned)
            weighted_score, _, _ = self.compute_quality_score(weighted)
            
            methods['weighted'] = (weighted, weighted_score)
            
            # Find best
            best_method = max(methods.items(), key=lambda x: x[1][1])
            combined = best_method[1][0]
            combined_score = best_method[1][1]
            top_channels = {'best_method': best_method[0]}
            
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        # Normalize final signal
        if np.std(combined) > 0:
            combined = (combined - np.mean(combined)) / np.std(combined)
        
        return combined, combined_score, top_channels, self.quality_metrics
    
    def align_to_reference(self, ref_signal, signal_to_align):
        """Align signal to reference using cross-correlation"""
        if len(ref_signal) != len(signal_to_align):
            return signal_to_align
        
        corr = np.corrcoef(ref_signal, signal_to_align)[0, 1]
        if np.isfinite(corr) and corr < 0:
            return -signal_to_align
        return signal_to_align
    
    def estimate_heart_rate(self, signal_data):
        """Estimate heart rate from signal"""
        if len(signal_data) < 32:
            return 0, 0, 0
        
        freqs, psd = welch(
            signal_data - np.mean(signal_data),
            fs=self.fs,
            nperseg=min(256, len(signal_data)),
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
        
        # 3-point quadratic interpolation for peak refinement
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
        
        # Confidence score
        noise_floor = np.median(psd[(freqs >= 0.1) & (freqs <= 5.0) & ~hr_mask])
        confidence = peak_power / (noise_floor + 1e-10)
        
        return bpm, peak_freq, confidence


class PositionQualityAnalyzer:
    """
    Analyze signal quality for different positions.
    Helps determine optimal sitting position for captures.
    """
    
    def __init__(self, extractor, fs=100.0):
        self.extractor = extractor
        self.fs = fs
        self.position_results = []
    
    def evaluate_position(self, csi_data, position_name="unknown", gt_df=None):
        """
        Evaluate a single position/capture.
        
        Returns:
            dict with quality metrics
        """
        # Extract combined signal
        combined, score, channels, metrics = self.extractor.extract_combined_signal(
            csi_data, method='adaptive'
        )
        
        # Get per-method scores
        amp_score = metrics.get('amplitude', {}).get('score', 0)
        phase_score = metrics.get('phase', {}).get('score', 0)
        diff_score = metrics.get('phase_diff', {}).get('score', 0)
        
        # Estimate heart rate
        bpm, peak_freq, confidence = self.extractor.estimate_heart_rate(combined)
        
        # Signal characteristics
        signal_power = np.var(combined)
        signal_peaks, _ = find_peaks(
            combined,
            distance=int(0.3 * self.fs),
            prominence=np.std(combined) * 0.4
        )
        
        # Presence detection: If signal has strong periodic component in HR band
        hr_present = confidence > 2.0 and len(signal_peaks) > 5
        
        # If ground truth available, compute validation metrics
        validation_metrics = {}
        if gt_df is not None:
            validation_metrics = self._validate_with_gt(combined, gt_df)
        
        result = {
            'position': position_name,
            'combined_score': score,
            'amp_score': amp_score,
            'phase_score': phase_score,
            'phase_diff_score': diff_score,
            'bpm': bpm,
            'peak_freq': peak_freq,
            'confidence': confidence,
            'n_peaks': len(signal_peaks),
            'signal_power': signal_power,
            'hr_present': hr_present,
            **validation_metrics
        }
        
        self.position_results.append(result)
        return result
    
    def _validate_with_gt(self, signal_data, gt_df):
        """Validate against ground truth if available"""
        # Get ground truth timestamps
        if 'ts' in gt_df.columns:
            gt_times = gt_df['ts'].values
            gt_times = gt_times - gt_times[0]
        else:
            return {}
        
        # Detect peaks in CSI signal
        peaks, _ = find_peaks(
            signal_data,
            distance=int(0.3 * self.fs),
            prominence=np.std(signal_data) * 0.4
        )
        
        csi_times = np.arange(len(signal_data)) / self.fs
        csi_beats = csi_times[peaks]
        
        if len(csi_beats) < 3 or len(gt_times) < 3:
            return {'f1_score': 0, 'precision': 0, 'recall': 0, 'hr_mae': 999}
        
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
        
        # HR MAE
        if len(matched) > 2:
            csi_hr = 60.0 / (np.diff(peaks[:len(matched)]) / self.fs)
            gt_hr = 60.0 / np.diff(gt_times[:len(matched)])
            min_len = min(len(csi_hr), len(gt_hr))
            hr_mae = np.mean(np.abs(csi_hr[:min_len] - gt_hr[:min_len])) if min_len > 1 else 999
        else:
            hr_mae = 999
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'hr_mae': hr_mae,
            'gt_beats': len(gt_times),
            'matched_beats': tp
        }
    
    def get_position_ranking(self):
        """Get ranking of positions by quality"""
        if not self.position_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.position_results)
        
        # Create composite score
        if 'f1_score' in df.columns:
            # If ground truth available, use F1 score as primary
            df['composite_score'] = (
                0.5 * df['f1_score'].fillna(0) +
                0.3 * df['confidence'].fillna(0) / 10 +
                0.2 * df['combined_score'].fillna(0) / 10
            )
        else:
            # Without ground truth, use confidence and power
            df['composite_score'] = (
                0.5 * df['confidence'] / 10 +
                0.3 * df['combined_score'] / 10 +
                0.2 * df['signal_power']
            )
        
        df = df.sort_values('composite_score', ascending=False)
        return df


def validate_combined_methods(csi_path, gt_path, output_dir='combined_results'):
    """Run full combined method validation"""
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("COMBINED AMPLITUDE + PHASE HEARTBEAT VALIDATION")
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
    
    # Apply basic filters
    print("\nApplying basic filters...")
    filter_obj = CSIFilter(FilterConfig())
    filtered_csi = filter_obj.apply_all_filters(csi_data, verbose=False)
    
    # Initialize extractor
    extractor = CombinedAmplitudePhaseExtractor(fs, low_freq=1.2, high_freq=2.8, order=4)
    
    # Test different fusion methods
    methods = [
        'amplitude_only',
        'phase_only',
        'phase_diff_only',
        'weighted',
        'product',
        'complex',
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
        
        # Detect peaks - FIXED: Use scipy_signal.find_peaks from import
        peaks, _ = find_peaks(
            signal,
            distance=int(0.3 * fs),
            prominence=np.std(signal) * 0.4,
            height=np.mean(signal) + np.std(signal) * 0.2
        )
        
        signals[method] = signal
        
        result = {
            'method': method,
            'quality_score': score,
            'confidence': confidence,
            'bpm': bpm,
            'peak_freq': peak_freq,
            'n_peaks': len(peaks),
            'signal_power': np.var(signal),
            'channels': str(channels)
        }
        
        # If ground truth available, compute F1
        if gt_df is not None:
            # Quick validation with ground truth
            val_result = validate_with_gt(signal, gt_df, fs)
            result.update(val_result)
        
        results.append(result)
        
        print(f"  Score: {score:.2f}, Confidence: {confidence:.2f}, BPM: {bpm:.1f}")
        print(f"  Peaks: {len(peaks)}, Channels: {channels}")
        if gt_df is not None:
            print(f"  F1: {result.get('f1_score', 0):.3f}, HR_MAE: {result.get('hr_mae', 999):.1f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('quality_score', ascending=False)
    results_df.to_csv(f'{output_dir}/combined_methods_results.csv', index=False)
    
    # Create comparison plot
    create_combined_plot(results_df, signals, fs, gt_df, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - METHOD RANKING")
    print("="*70)
    for idx, row in results_df.iterrows():
        print(f"  {row['method']:15s}: Score={row['quality_score']:.2f}, "
              f"Confidence={row['confidence']:.2f}, BPM={row['bpm']:.1f}")
    
    best = results_df.iloc[0]
    print("\n" + "="*70)
    print(f"BEST METHOD: {best['method']}")
    print("="*70)
    print(f"  Quality Score: {best['quality_score']:.2f}")
    print(f"  Confidence: {best['confidence']:.2f}")
    print(f"  Heart Rate: {best['bpm']:.1f} BPM")
    if 'f1_score' in best:
        print(f"  F1 Score: {best['f1_score']:.3f}")
        print(f"  HR MAE: {best.get('hr_mae', 999):.1f} BPM")
    
    return results_df, signals


def validate_with_gt(signal_data, gt_df, fs):
    """Quick validation against ground truth"""
    if 'ts' in gt_df.columns:
        gt_times = gt_df['ts'].values
        gt_times = gt_times - gt_times[0]
    else:
        return {'f1_score': 0, 'precision': 0, 'recall': 0, 'hr_mae': 999}
    
    # Detect peaks
    peaks, _ = find_peaks(
        signal_data,
        distance=int(0.3 * fs),
        prominence=np.std(signal_data) * 0.4
    )
    
    csi_times = np.arange(len(signal_data)) / fs
    csi_beats = csi_times[peaks]
    
    if len(csi_beats) < 3 or len(gt_times) < 3:
        return {'f1_score': 0, 'precision': 0, 'recall': 0, 'hr_mae': 999}
    
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
    
    # HR MAE
    if len(matched) > 2:
        csi_hr = 60.0 / (np.diff(peaks[:len(matched)]) / fs)
        gt_hr = 60.0 / np.diff(gt_times[:len(matched)])
        min_len = min(len(csi_hr), len(gt_hr))
        hr_mae = np.mean(np.abs(csi_hr[:min_len] - gt_hr[:min_len])) if min_len > 1 else 999
    else:
        hr_mae = 999
    
    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'hr_mae': hr_mae,
        'matched_beats': tp,
        'total_gt_beats': len(gt_times)
    }


def create_combined_plot(results_df, signals, fs, gt_df, output_dir):
    """Create visualization for combined methods"""
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Method comparison bar chart
    ax1 = fig.add_subplot(gs[0, :2])
    methods = results_df['method'].values
    scores = results_df['quality_score'].values
    colors = ['green' if i == 0 else 'blue' for i in range(len(scores))]
    y_pos = np.arange(len(methods))
    ax1.barh(y_pos, scores, color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods)
    ax1.set_xlabel('Quality Score')
    ax1.set_title('Method Comparison by Quality Score')
    ax1.axvline(x=np.mean(scores), color='red', linestyle='--', alpha=0.5, label='Mean')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence vs BPM
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(results_df['bpm'], results_df['confidence'], s=50, alpha=0.7)
    for i, row in results_df.iterrows():
        ax2.annotate(row['method'], (row['bpm'], row['confidence']), fontsize=8, alpha=0.7)
    ax2.set_xlabel('Heart Rate (BPM)')
    ax2.set_ylabel('Confidence')
    ax2.set_title('Confidence vs Heart Rate')
    ax2.grid(True, alpha=0.3)
    
    # 3-5. Signal waveforms for top 3 methods
    top_methods = results_df.head(3)['method'].values
    for idx, method in enumerate(top_methods):
        ax = fig.add_subplot(gs[1, idx])
        signal_data = signals.get(method, np.array([]))
        if len(signal_data) > 0:
            t = np.arange(len(signal_data)) / fs
            # Show first 10 seconds
            n_show = min(len(signal_data), int(10 * fs))
            ax.plot(t[:n_show], signal_data[:n_show], linewidth=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{method} (first 10s)')
            ax.grid(True, alpha=0.3)
    
    # 6. Spectrogram for best method
    ax6 = fig.add_subplot(gs[2, 0])
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
        im = ax6.imshow(
            10 * np.log10(Sxx[freq_mask, :] + 1e-10),
            aspect='auto',
            origin='lower',
            extent=[times[0], times[-1], freqs[freq_mask][0], freqs[freq_mask][-1]],
            cmap='viridis'
        )
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Frequency (Hz)')
        ax6.set_title(f'Spectrogram: {best_method}')
        plt.colorbar(im, ax=ax6, label='Power (dB)')
    
    # 7. Quality metrics summary
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.axis('off')
    summary_text = "METHOD SUMMARY\n" + "="*30 + "\n\n"
    for idx, row in results_df.head(5).iterrows():
        summary_text += f"{row['method']:12s}: "
        summary_text += f"Q={row['quality_score']:.2f}, "
        summary_text += f"C={row['confidence']:.2f}, "
        summary_text += f"BPM={row['bpm']:.1f}"
        if 'f1_score' in row and pd.notna(row['f1_score']):
            summary_text += f", F1={row['f1_score']:.3f}"
        summary_text += "\n"
    
    summary_text += f"\nBest Method: {results_df.iloc[0]['method']}"
    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    # 8. F1 Score comparison (if ground truth available)
    ax8 = fig.add_subplot(gs[2, 2])
    if 'f1_score' in results_df.columns and results_df['f1_score'].notna().any():
        f1_scores = results_df['f1_score'].fillna(0).values
        ax8.barh(y_pos, f1_scores, color=colors)
        ax8.set_yticks(y_pos)
        ax8.set_yticklabels(methods)
        ax8.set_xlabel('F1 Score')
        ax8.set_title('F1 Score Comparison')
        ax8.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        ax8.grid(True, alpha=0.3)
        ax8.set_xlim(0, 1)
    else:
        # Show number of peaks instead
        n_peaks = results_df['n_peaks'].values
        ax8.barh(y_pos, n_peaks, color='purple')
        ax8.set_yticks(y_pos)
        ax8.set_yticklabels(methods)
        ax8.set_xlabel('Number of Peaks Detected')
        ax8.set_title('Peak Detection Count')
        ax8.grid(True, alpha=0.3)
    
    plt.suptitle('Combined Amplitude + Phase Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved combined analysis plot to '{output_dir}/combined_analysis.png'")
    plt.close()


def position_optimization(csi_paths, output_dir='position_optimization'):
    """
    Run position optimization test across multiple captures.
    
    Args:
        csi_paths: List of (position_name, csi_file_path) tuples
        output_dir: Output directory
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("POSITION OPTIMIZATION ANALYSIS")
    print("="*70)
    
    analyzer = PositionQualityAnalyzer(
        CombinedAmplitudePhaseExtractor(fs=100.0, low_freq=1.2, high_freq=2.8, order=4)
    )
    
    results = []
    
    for position_name, csi_path in csi_paths:
        print(f"\nEvaluating position: {position_name}")
        print(f"  File: {csi_path}")
        
        # Load CSI
        csi_path = Path(csi_path)
        if csi_path.suffix.lower() == '.csv':
            loader = CSIDataLoader(str(csi_path))
            csi_data, metadata = loader.load()
        else:
            loader = PCAPDataLoader(str(csi_path))
            csi_data, metadata = loader.load()
        
        analyzer.fs = metadata['rate']
        analyzer.extractor.fs = metadata['rate']
        
        # Check for ground truth
        gt_path = csi_path.parent / csi_path.name.replace('csi_capture', 'ground_truth').replace('.csv', '.csv')
        gt_df = None
        if gt_path.exists():
            gt_df = pd.read_csv(str(gt_path))
            print(f"  Found ground truth: {len(gt_df)} beats")
        
        # Evaluate position
        result = analyzer.evaluate_position(csi_data, position_name, gt_df)
        results.append(result)
        
        print(f"  Combined Score: {result['combined_score']:.2f}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Heart Rate: {result['bpm']:.1f} BPM")
        print(f"  HR Present: {result['hr_present']}")
        if 'f1_score' in result:
            print(f"  F1 Score: {result['f1_score']:.3f}")
    
    # Create ranking
    ranking = analyzer.get_position_ranking()
    ranking.to_csv(f'{output_dir}/position_ranking.csv', index=False)
    
    print("\n" + "="*70)
    print("POSITION RANKING")
    print("="*70)
    print(ranking[['position', 'composite_score', 'confidence', 'bpm', 'hr_present']].to_string(index=False))
    
    # Create position analysis plot
    create_position_plot(ranking, output_dir)
    
    # Best position recommendation
    if not ranking.empty:
        best = ranking.iloc[0]
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        print(f"Best position: {best['position']}")
        print(f"  Composite Score: {best['composite_score']:.3f}")
        print(f"  Confidence: {best['confidence']:.2f}")
        print(f"  Heart Rate: {best['bpm']:.1f} BPM")
        if 'f1_score' in best and pd.notna(best['f1_score']):
            print(f"  F1 Score: {best['f1_score']:.3f}")
    
    return ranking


def create_position_plot(ranking, output_dir):
    """Create position optimization visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Composite score
    ax1 = axes[0, 0]
    positions = ranking['position'].values
    scores = ranking['composite_score'].values
    colors = ['green' if i == 0 else 'blue' for i in range(len(scores))]
    y_pos = np.arange(len(positions))
    ax1.barh(y_pos, scores, color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(positions, fontsize=9)
    ax1.set_xlabel('Composite Score')
    ax1.set_title('Position Ranking by Composite Score')
    ax1.axvline(x=np.mean(scores), color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence vs Score
    ax2 = axes[0, 1]
    ax2.scatter(ranking['combined_score'], ranking['confidence'], s=50, alpha=0.7)
    for i, row in ranking.iterrows():
        ax2.annotate(row['position'], (row['combined_score'], row['confidence']), 
                    fontsize=8, alpha=0.7)
    ax2.set_xlabel('Combined Score')
    ax2.set_ylabel('Confidence')
    ax2.set_title('Confidence vs Quality Score')
    ax2.grid(True, alpha=0.3)
    
    # 3. HR Present status
    ax3 = axes[1, 0]
    hr_present = ranking['hr_present'].values
    colors = ['green' if p else 'red' for p in hr_present]
    ax3.barh(y_pos, [1 if p else 0 for p in hr_present], color=colors)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(positions, fontsize=9)
    ax3.set_xlabel('Heartbeat Present')
    ax3.set_title('Heartbeat Detection Status')
    ax3.set_xlim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)
    
    # 4. Heart rate consistency
    ax4 = axes[1, 1]
    bpm_values = ranking['bpm'].values
    valid_bpm = [b for b in bpm_values if 30 < b < 150]
    if valid_bpm:
        ax4.hist(valid_bpm, bins=10, alpha=0.7, color='purple')
        ax4.axvline(x=np.mean(valid_bpm), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(valid_bpm):.1f} BPM')
        ax4.set_xlabel('Heart Rate (BPM)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Heart Rate Distribution Across Positions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Position Optimization Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/position_optimization.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved position analysis plot to '{output_dir}/position_optimization.png'")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Combined Amplitude + Phase Heartbeat Detection Test',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--csi', help='CSI data file (.pcap or .csv)')
    parser.add_argument('--groundtruth', help='Ground truth CSV file')
    parser.add_argument('--output-dir', default='combined_results', help='Output directory')
    
    # Position optimization mode
    parser.add_argument('--position-optimization', action='store_true',
                       help='Run position optimization mode')
    parser.add_argument('--positions', nargs='+', 
                       help='Position files: "name1 file1.csv name2 file2.csv ..."')
    
    # Filter parameters
    parser.add_argument('--heart-low', type=float, default=1.2,
                       help='Lower cutoff for heartbeat filter (Hz)')
    parser.add_argument('--heart-high', type=float, default=2.8,
                       help='Upper cutoff for heartbeat filter (Hz)')
    parser.add_argument('--heart-order', type=int, default=4,
                       help='Butterworth filter order')
    
    args = parser.parse_args()
    
    if args.position_optimization:
        # Position optimization mode
        if args.positions:
            if len(args.positions) % 2 != 0:
                print("Error: --positions requires pairs of 'name file'")
                sys.exit(1)
            
            csi_paths = []
            for i in range(0, len(args.positions), 2):
                name = args.positions[i]
                path = args.positions[i+1]
                if not Path(path).exists():
                    print(f"Warning: File '{path}' not found, skipping")
                    continue
                csi_paths.append((name, path))
        else:
            # Auto-detect captures in current directory
            csi_paths = []
            for f in Path('.').glob('*csi_capture*.csv'):
                name = f.parent.name if f.parent.name != '.' else f.stem
                csi_paths.append((name, str(f)))
            for f in Path('.').glob('*capture*.pcap'):
                name = f.parent.name if f.parent.name != '.' else f.stem
                csi_paths.append((name, str(f)))
        
        if not csi_paths:
            print("Error: No CSI files found for position optimization")
            sys.exit(1)
        
        position_optimization(csi_paths, args.output_dir)
        
    else:
        # Standard validation mode
        if not args.csi:
            print("Error: --csi required for validation mode")
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