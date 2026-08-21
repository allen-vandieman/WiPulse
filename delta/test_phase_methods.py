# test_phase_methods.py
# Fair comparison: Amplitude vs Phase with identical processing pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import detrend
from pathlib import Path

from read_csv_csi import CSIDataLoader
from filters import CSIFilter, FilterConfig


class HeartbeatPipeline:
    """
    Unified pipeline for heartbeat extraction from CSI.
    Supports both amplitude and phase with identical processing steps.
    """
    
    def __init__(self, fs=100.0):
        self.fs = fs
        self.quality_metrics = None
        
    def extract_amplitude(self, csi_data):
        """Extract amplitude from complex CSI"""
        return np.abs(csi_data)
    
    def extract_phase(self, csi_data, unwrap=True):
        """Extract phase from complex CSI"""
        phase = np.angle(csi_data)
        if unwrap:
            for sc in range(phase.shape[1]):
                phase[:, sc] = np.unwrap(phase[:, sc])
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
    
    def apply_hampel(self, signal_data, window_size=11, threshold=3.0):
        """Apply Hampel filter for outlier removal"""
        if window_size % 2 == 0:
            window_size += 1
        
        half_window = window_size // 2
        filtered = signal_data.copy()
        
        for i in range(len(signal_data)):
            start = max(0, i - half_window)
            end = min(len(signal_data), i + half_window + 1)
            window = signal_data[start:end]
            
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            
            if mad > 0:
                z_score = np.abs(signal_data[i] - median) / (1.4826 * mad)
                if z_score > threshold:
                    filtered[i] = median
        
        return filtered
    
    def bandpass_filter(self, signal_data, low=1.2, high=2.8, order=4):
        """Apply bandpass filter"""
        nyquist = self.fs / 2
        sos = signal.butter(order, [low/nyquist, high/nyquist], btype='band', output='sos')
        return signal.sosfiltfilt(sos, signal_data - np.mean(signal_data))
    
    def compute_quality_score(self, signal_data, hr_range=(0.8, 3.5)):
        """
        Compute quality score using spectral peak metric.
        Score = peak_amplitude / (mean_noise_floor + epsilon)
        """
        n = len(signal_data)
        fft_vals = np.abs(np.fft.fft(signal_data - np.mean(signal_data)))
        freqs = np.fft.fftfreq(n, 1/self.fs)
        
        # Positive frequencies only
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        fft_pos = fft_vals[pos_mask]
        
        # Heart rate band
        hr_mask = (freqs_pos >= hr_range[0]) & (freqs_pos <= hr_range[1])
        
        if not np.any(hr_mask):
            return 0, 0, 0
        
        # Find peak in HR band
        hr_fft = fft_pos[hr_mask]
        hr_freqs = freqs_pos[hr_mask]
        peak_idx = np.argmax(hr_fft)
        peak_amp = hr_fft[peak_idx]
        peak_freq = hr_freqs[peak_idx]
        
        # Noise floor (outside HR band, but within 0-5 Hz)
        noise_mask = (freqs_pos >= 0.1) & (freqs_pos <= 5.0) & ~hr_mask
        if np.any(noise_mask):
            noise_floor = np.percentile(fft_pos[noise_mask], 50)
        else:
            noise_floor = np.median(fft_pos)
        
        # SNR-like quality score
        quality_score = peak_amp / (noise_floor + 1e-10)
        
        return quality_score, peak_freq, peak_amp
    
    def process_channel(self, signal_data, apply_hampel=True, 
                        detrend_signal=True, low=1.2, high=2.8, order=4):
        """
        Process a single channel (subcarrier or diff) through the full pipeline.
        """
        processed = signal_data.copy()
        
        # Step 1: Remove DC
        processed = processed - np.mean(processed)
        
        # Step 2: Hampel filter (optional)
        if apply_hampel:
            processed = self.apply_hampel(processed)
        
        # Step 3: Detrend (optional)
        if detrend_signal:
            processed = detrend(processed)
        
        # Step 4: Bandpass filter
        processed = self.bandpass_filter(processed, low, high, order)
        
        # Step 5: Normalize
        if np.std(processed) > 0:
            processed = processed / np.std(processed)
        
        return processed
    
    def process_all_channels(self, channels, channel_names=None, 
                            apply_hampel=True, detrend_signal=True,
                            low=1.2, high=2.8, order=4):
        """
        Process multiple channels through the pipeline.
        
        Args:
            channels: list of 1D signals
            channel_names: list of names for each channel
        
        Returns:
            processed_signals: dict mapping name -> processed signal
            quality_df: DataFrame with quality metrics
        """
        processed_signals = {}
        quality_data = []
        
        if channel_names is None:
            channel_names = [f"ch_{i}" for i in range(len(channels))]
        
        for name, signal_data in zip(channel_names, channels):
            try:
                processed = self.process_channel(
                    signal_data, apply_hampel, detrend_signal, low, high, order
                )
                processed_signals[name] = processed
                
                # Compute quality score
                score, peak_freq, peak_amp = self.compute_quality_score(processed)
                quality_data.append({
                    'name': name,
                    'quality_score': score,
                    'peak_freq_hz': peak_freq,
                    'peak_amp': peak_amp,
                    'std': np.std(processed),
                    'n_samples': len(processed)
                })
            except Exception as e:
                print(f"Warning: Failed to process {name}: {e}")
                continue
        
        self.quality_metrics = pd.DataFrame(quality_data)
        self.quality_metrics = self.quality_metrics.sort_values('quality_score', ascending=False)
        
        return processed_signals, self.quality_metrics
    
    def fuse_top_channels(self, processed_signals, quality_df, top_n=5, align_polarity=True):
        """
        Fuse top N channels by quality score.
        
        Args:
            processed_signals: dict mapping name -> processed signal
            quality_df: DataFrame with quality metrics (sorted by score)
            top_n: number of top channels to fuse
            align_polarity: if True, align polarity before averaging
        
        Returns:
            fused: averaged signal
            top_names: list of names used
        """
        top_names = quality_df.head(top_n)['name'].tolist()
        
        # Get signals for top names
        top_signals = []
        valid_names = []
        for name in top_names:
            if name in processed_signals:
                top_signals.append(processed_signals[name].copy())
                valid_names.append(name)
        
        if not top_signals:
            return np.array([]), valid_names
        
        # Polarity alignment
        if align_polarity and len(top_signals) > 1:
            reference = top_signals[0]
            aligned = [reference]
            
            for sig in top_signals[1:]:
                corr = np.corrcoef(reference, sig)[0, 1]
                if np.isfinite(corr) and corr < 0:
                    sig = -sig
                aligned.append(sig)
            
            top_signals = aligned
        
        # Average
        fused = np.mean(top_signals, axis=0)
        
        return fused, valid_names


def validate_signal(heartbeat_signal, gt_df, fs, csi_timestamps=None,
                    low=1.2, high=2.8, order=4):
    """
    Validate a heartbeat signal against ground truth.
    Uses peak detection and PSD-based BPM estimation.
    """
    from validate_heartbeat import HeartbeatValidator
    
    # Create validator instance
    validator = HeartbeatValidator()
    validator.csi_heartbeat = heartbeat_signal
    validator.metadata = {'rate': fs}
    validator.groundtruth_df = gt_df
    
    # Detect peaks
    from heartbeat_extraction import HeartbeatExtractor, HeartbeatConfig
    config = HeartbeatConfig()
    config.sampling_rate = fs
    config.bandpass_low = low
    config.bandpass_high = high
    config.filter_order = order
    
    extractor = HeartbeatExtractor(config)
    peak_result = extractor.detect_peaks(heartbeat_signal)
    
    validator.csi_peaks = peak_result['peaks']
    validator.csi_heart_rates = peak_result['heart_rates']
    
    # Use actual CSI timestamps if provided
    if csi_timestamps is not None:
        validator.csi_timestamps = csi_timestamps - csi_timestamps[0]
    else:
        validator.csi_timestamps = np.arange(len(heartbeat_signal)) / fs
    
    # Align and compute metrics
    validator.align_timestamps()
    metrics = validator.compute_validation_metrics()
    
    # Add PSD-based BPM estimation
    from scipy import signal as scipy_signal
    freqs, psd = scipy_signal.welch(
        heartbeat_signal - np.mean(heartbeat_signal),
        fs=fs,
        nperseg=min(256, len(heartbeat_signal))
    )
    
    hr_mask = (freqs >= 0.8) & (freqs <= 3.5)
    if np.any(hr_mask):
        peak_idx = np.argmax(psd[hr_mask])
        peak_freq = freqs[hr_mask][peak_idx]
        metrics['bpm_psd'] = peak_freq * 60.0
        metrics['psd_peak_amplitude'] = psd[hr_mask][peak_idx]
    else:
        metrics['bpm_psd'] = 0
        metrics['psd_peak_amplitude'] = 0
    
    return metrics


def create_comparison_plot(amp, phase, common, diff, gt_df, fs, summary, 
                           amp_top=None, phase_top=None, common_top=None, diff_top=None):
    """Create comparison plot of all methods - 4x2 layout"""
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    
    t = np.arange(len(amp)) / fs
    
    # Define the four methods
    methods = [
        ('Amplitude', amp, 'blue', amp_top),
        ('Phase', phase, 'green', phase_top),
        ('Phase + Common', common, 'orange', common_top),
        ('Phase Difference', diff, 'red', diff_top)
    ]
    
    for row, (name, waveform, color, top_list) in enumerate(methods):
        # Signal plot (left column)
        ax_signal = axes[row, 0]
        ax_signal.plot(t[:min(500, len(t))], waveform[:min(500, len(waveform))], 
                      color=color, alpha=0.7, linewidth=0.8)
        ax_signal.set_title(f'{name} Signal (first 5s)')
        ax_signal.set_ylabel('Normalized Amplitude')
        ax_signal.grid(True, alpha=0.3)
        if row == 3:
            ax_signal.set_xlabel('Time (s)')
        
        # Add top subcarriers info
        if top_list:
            top_str = ', '.join(top_list[:3])
            ax_signal.text(0.02, 0.95, f'Top: {top_str}', 
                          transform=ax_signal.transAxes, fontsize=8,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Spectrum plot (right column)
        ax_spec = axes[row, 1]
        freqs, psd = signal.welch(
            waveform - np.mean(waveform),
            fs=fs,
            nperseg=min(256, len(waveform))
        )
        mask = (freqs >= 0) & (freqs <= 5)
        ax_spec.plot(freqs[mask], psd[mask], color=color, linewidth=1.5)
        ax_spec.axvspan(0.8, 3.5, alpha=0.2, color='red', label='Heartbeat band')
        ax_spec.set_title(f'{name} Spectrum')
        ax_spec.set_xlabel('Frequency (Hz)')
        ax_spec.set_ylabel('PSD')
        ax_spec.grid(True, alpha=0.3)
        ax_spec.set_xlim(0, 5)
        if row == 0:
            ax_spec.legend()
    
    plt.suptitle('Amplitude vs Phase Comparison (4 Methods)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('phase_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved comparison plot to 'phase_comparison.png'")
    plt.close()
    
    # Also save summary as CSV
    summary.to_csv('phase_comparison_summary.csv', index=False)
    print("Saved summary to 'phase_comparison_summary.csv'")


def test_phase_methods(csi_path, gt_path, low=1.2, high=2.8, order=4):
    """
    Fair comparison of amplitude vs phase methods.
    """
    print("\n" + "="*70)
    print("PHASE vs AMPLITUDE: FAIR COMPARISON")
    print("="*70)
    
    # Load data
    print(f"\nLoading CSI from: {csi_path}")
    loader = CSIDataLoader(csi_path)
    csi_data, metadata = loader.load()
    fs = metadata['rate']
    print(f"  {metadata['n_packets']} packets at {fs:.1f} Hz")
    
    # Get actual timestamps
    csi_timestamps = None
    if hasattr(loader, 'df') and loader.df is not None:
        if 'ts' in loader.df.columns:
            csi_timestamps = loader.df['ts'].values
    
    # Load ground truth
    print(f"Loading ground truth: {gt_path}")
    gt_df = pd.read_csv(gt_path)
    print(f"  {len(gt_df)} beats")
    
    # DON'T pre-filter with amplitude-oriented filters
    print("\nUsing raw complex CSI (no pre-filtering)")
    
    # Initialize pipeline
    pipeline = HeartbeatPipeline(fs)
    
    results = []
    
    # --- METHOD 1: AMPLITUDE (Baseline) ---
    print("\n" + "-"*50)
    print("METHOD 1: AMPLITUDE (Baseline)")
    print("-"*50)
    
    amplitude = pipeline.extract_amplitude(csi_data)
    
    # Process each subcarrier
    channels = [amplitude[:, sc] for sc in range(amplitude.shape[1])]
    names = [f"amp_sc{sc}" for sc in range(amplitude.shape[1])]
    
    processed_amp, quality_amp = pipeline.process_all_channels(
        channels, names, apply_hampel=True, detrend_signal=True,
        low=low, high=high, order=order
    )
    
    # Fuse top 5 with polarity alignment
    fused_amp, top_amp = pipeline.fuse_top_channels(processed_amp, quality_amp, top_n=5, align_polarity=True)
    print(f"Top 5 amplitude subcarriers: {top_amp}")
    print(f"Best quality score: {quality_amp.iloc[0]['quality_score']:.3f}")
    
    # Validate with actual timestamps
    metrics_amp = validate_signal(fused_amp, gt_df, fs, csi_timestamps, low, high, order)
    print(f"  F1: {metrics_amp['f1_score']:.3f}")
    print(f"  HR MAE: {metrics_amp['heart_rate_mae']:.1f} BPM")
    print(f"  Correlation: {metrics_amp['heart_rate_correlation']:.3f}")
    print(f"  BPM (PSD): {metrics_amp['bpm_psd']:.1f}")
    results.append({'Method': 'Amplitude', **metrics_amp, 'Best_Quality': quality_amp.iloc[0]['quality_score']})
    
    # --- METHOD 2: PHASE (Unwrapped) ---
    print("\n" + "-"*50)
    print("METHOD 2: PHASE (Unwrapped)")
    print("-"*50)
    
    phase = pipeline.extract_phase(csi_data, unwrap=True)
    
    # Process each subcarrier
    channels_phase = [phase[:, sc] for sc in range(phase.shape[1])]
    names_phase = [f"phase_sc{sc}" for sc in range(phase.shape[1])]
    
    processed_phase, quality_phase = pipeline.process_all_channels(
        channels_phase, names_phase, apply_hampel=True, detrend_signal=True,
        low=low, high=high, order=order
    )
    
    fused_phase, top_phase = pipeline.fuse_top_channels(processed_phase, quality_phase, top_n=5, align_polarity=True)
    print(f"Top 5 phase subcarriers: {top_phase}")
    print(f"Best quality score: {quality_phase.iloc[0]['quality_score']:.3f}")
    
    metrics_phase = validate_signal(fused_phase, gt_df, fs, csi_timestamps, low, high, order)
    print(f"  F1: {metrics_phase['f1_score']:.3f}")
    print(f"  HR MAE: {metrics_phase['heart_rate_mae']:.1f} BPM")
    print(f"  Correlation: {metrics_phase['heart_rate_correlation']:.3f}")
    print(f"  BPM (PSD): {metrics_phase['bpm_psd']:.1f}")
    results.append({'Method': 'Phase', **metrics_phase, 'Best_Quality': quality_phase.iloc[0]['quality_score']})
    
    # --- METHOD 3: PHASE + COMMON PHASE REMOVAL ---
    print("\n" + "-"*50)
    print("METHOD 3: PHASE + Common Phase Removal")
    print("-"*50)
    
    phase_common_removed = pipeline.remove_common_phase(phase)
    
    channels_common = [phase_common_removed[:, sc] for sc in range(phase_common_removed.shape[1])]
    names_common = [f"common_sc{sc}" for sc in range(phase_common_removed.shape[1])]
    
    processed_common, quality_common = pipeline.process_all_channels(
        channels_common, names_common, apply_hampel=True, detrend_signal=True,
        low=low, high=high, order=order
    )
    
    fused_common, top_common = pipeline.fuse_top_channels(processed_common, quality_common, top_n=5, align_polarity=True)
    print(f"Top 5 common-removed subcarriers: {top_common}")
    print(f"Best quality score: {quality_common.iloc[0]['quality_score']:.3f}")
    
    metrics_common = validate_signal(fused_common, gt_df, fs, csi_timestamps, low, high, order)
    print(f"  F1: {metrics_common['f1_score']:.3f}")
    print(f"  HR MAE: {metrics_common['heart_rate_mae']:.1f} BPM")
    print(f"  Correlation: {metrics_common['heart_rate_correlation']:.3f}")
    print(f"  BPM (PSD): {metrics_common['bpm_psd']:.1f}")
    results.append({'Method': 'Phase + Common', **metrics_common, 'Best_Quality': quality_common.iloc[0]['quality_score']})
    
    # --- METHOD 4: PHASE DIFFERENCES ---
    print("\n" + "-"*50)
    print("METHOD 4: PHASE DIFFERENCES")
    print("-"*50)
    
    diffs, diff_indices = pipeline.compute_phase_differences(phase, max_lag=10)
    
    names_diff = [f"diff_{i}_{j}" for i, j in diff_indices]
    
    processed_diff, quality_diff = pipeline.process_all_channels(
        diffs, names_diff, apply_hampel=True, detrend_signal=True,
        low=low, high=high, order=order
    )
    
    fused_diff, top_diff = pipeline.fuse_top_channels(processed_diff, quality_diff, top_n=5, align_polarity=True)
    print(f"Top 5 phase differences: {top_diff}")
    print(f"Best quality score: {quality_diff.iloc[0]['quality_score']:.3f}")
    
    metrics_diff = validate_signal(fused_diff, gt_df, fs, csi_timestamps, low, high, order)
    print(f"  F1: {metrics_diff['f1_score']:.3f}")
    print(f"  HR MAE: {metrics_diff['heart_rate_mae']:.1f} BPM")
    print(f"  Correlation: {metrics_diff['heart_rate_correlation']:.3f}")
    print(f"  BPM (PSD): {metrics_diff['bpm_psd']:.1f}")
    results.append({'Method': 'Phase Differences', **metrics_diff, 'Best_Quality': quality_diff.iloc[0]['quality_score']})
    
    # --- SUMMARY ---
    print("\n" + "="*70)
    print("SUMMARY: AMPLITUDE vs PHASE")
    print("="*70)
    
    summary = pd.DataFrame(results)
    
    # Select and reorder columns for display
    display_cols = ['Method', 'f1_score', 'heart_rate_mae', 'heart_rate_correlation', 'bpm_psd', 'Best_Quality']
    print(summary[display_cols].to_string(index=False))
    
    # Best method
    best_idx = summary['f1_score'].idxmax()
    print(f"\nBEST METHOD: {summary.iloc[best_idx]['Method']}")
    print(f"  F1: {summary.iloc[best_idx]['f1_score']:.3f}")
    print(f"  HR_MAE: {summary.iloc[best_idx]['heart_rate_mae']:.1f}")
    print(f"  Correlation: {summary.iloc[best_idx]['heart_rate_correlation']:.3f}")
    print(f"  BPM (PSD): {summary.iloc[best_idx]['bpm_psd']:.1f}")
    
    # Create comparison plot
    create_comparison_plot(
        fused_amp, fused_phase, fused_common, fused_diff,
        gt_df, fs, summary,
        top_amp, top_phase, top_common, top_diff
    )
    
    return summary


if __name__ == "__main__":
    import sys
    
    csi_path = r"captures\Allen-S,0,1\csi_capture_20260820_135250.csv"
    gt_path = r"captures\Allen-S,0,1\ground_truth_20260820_135247.csv"
    
    # Use the optimal filter from your sweep
    test_phase_methods(csi_path, gt_path, low=1.2, high=2.8, order=4)