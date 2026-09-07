#!/usr/bin/env python3
# heartbeat_extraction.py - Heartbeat Detection Module
# Enhanced with occupancy detection, motion scoring, and signal quality assessment

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import deque
from typing import Optional, Tuple, Dict, List


class HeartbeatConfig:
    """Configuration for heartbeat extraction"""
    
    def __init__(self):
        # Bandpass filter settings (heart rate ~60-90 BPM = 1.0-1.5 Hz)
        self.bandpass_low = 0.8      # Lower cutoff frequency (Hz)
        self.bandpass_high = 2.5     # Upper cutoff frequency (Hz)
        self.filter_order = 3        # Butterworth filter order
        self.sampling_rate = 100.0   # Hz (estimated from CSI)
        
        # Heart rate estimation
        self.heart_rate_min = 40     # Minimum HR (BPM)
        self.heart_rate_max = 120    # Maximum HR (BPM)
        self.segment_duration = 10   # Seconds for sliding window analysis
        self.overlap = 0.5           # Overlap ratio for sliding windows
        
        # Visualization
        self.show_spectrogram = True
        self.show_heartbeat_waveform = True
        self.show_peak_detection = True
        
        # Subcarrier selection
        self.use_all_subcarriers = False  # If False, automatically select best subcarriers
        self.auto_select_percentile = 80  # Percentile for selecting best subcarriers
        self.max_subcarriers_to_plot = 10 # Max subcarriers to display in comparisons
        
        # ============ NEW: OCCUPANCY AND MOTION DETECTION ============
        # Motion detection
        self.motion_window_seconds = 3.0      # Window for motion analysis
        self.motion_threshold = 2.5           # Motion score threshold (higher = less sensitive)
        self.occupancy_energy_threshold = 0.05 # Minimum energy for occupancy
        self.min_snr = 1.5                    # Minimum SNR for valid BPM
        self.max_hr_change = 30.0             # Max BPM change between estimates
        self.min_occupancy_duration = 5.0     # Seconds of occupancy required before reporting HR
        
        # Quality thresholds
        self.min_peak_prominence = 2.0
        self.min_signal_quality = 0.2


class HeartbeatExtractor:
    """Extract heartbeat signal from CSI data with occupancy and motion detection"""
    
    def __init__(self, config=None):
        self.config = config if config else HeartbeatConfig()
        self.filtered_data = None
        self.heartbeat_signal = None
        self.heart_rate_history = []
        self.spectrogram = None
        self.frequencies = None
        self.times = None
        
        # ============ NEW: STATE FOR OCCUPANCY AND MOTION ============
        self.occupancy = False
        self.motion_level = 0.0
        self.signal_quality = 0.0
        self.confidence = 0.0
        self.occupancy_start_time = None
        self.is_occupied_stable = False
        self.amp_history = deque(maxlen=100)
        self.motion_history = deque(maxlen=100)
        self.hr_history = deque(maxlen=5)
        self.last_valid_bpm = None
        self.baseline_std = None
        self.baseline_mean = None
        self.baseline_update_time = 0
        
    def design_bandpass_filter(self):
        """Design a bandpass Butterworth filter for heartbeat extraction"""
        nyquist = self.config.sampling_rate / 2
        
        # Normalize frequencies
        low = self.config.bandpass_low / nyquist
        high = self.config.bandpass_high / nyquist
        
        # Ensure valid range
        low = max(0.001, low)
        high = min(0.999, high)
        
        # Design Butterworth bandpass filter
        sos = signal.butter(
            self.config.filter_order,
            [low, high],
            btype='band',
            output='sos'
        )
        
        return sos
    
    # ============ NEW: OCCUPANCY AND MOTION DETECTION METHODS ============
    
    def detect_occupancy_and_motion(self, csi_data) -> Tuple[bool, float, float]:
        """
        Detect if someone is present and estimate motion level.
        
        Returns:
            (is_occupied, motion_level, energy)
        """
        if csi_data is None or len(csi_data) < 10:
            return False, 0.0, 0.0
        
        # Compute amplitude statistics
        amps = np.abs(csi_data)
        mean_amp = np.mean(amps)
        
        # Energy: total signal power
        energy = np.mean(amps ** 2) / (np.max(amps) ** 2 + 1e-9)
        energy = np.clip(energy, 0, 1)
        
        # Motion: variance of amplitude over time
        time_variance = np.var(amps, axis=0)
        mean_variance = np.mean(time_variance)
        motion_level = mean_variance / (mean_amp ** 2 + 1e-9)
        motion_level = np.clip(motion_level, 0, 10)
        
        # Subcarrier diversity: how much subcarriers vary within a packet
        sc_variance = np.var(amps, axis=1)
        sc_diversity = np.mean(sc_variance) / (mean_amp ** 2 + 1e-9)
        sc_diversity = np.clip(sc_diversity, 0, 1)
        
        # Baseline comparison (adaptive)
        current_std = np.std(amps.flatten())
        
        # Update baseline
        if self.baseline_std is None:
            self.baseline_std = current_std
            self.baseline_mean = mean_amp
        else:
            # Slow adaptation
            self.baseline_std = 0.95 * self.baseline_std + 0.05 * current_std
            self.baseline_mean = 0.95 * self.baseline_mean + 0.05 * mean_amp
        
        # Motion ratio relative to baseline
        baseline_std = self.baseline_std or 0.001
        motion_ratio = current_std / baseline_std
        
        # Occupancy detection based on multiple factors:
        # 1. Energy above threshold (signal present)
        # 2. Subcarrier diversity (not just noise floor)
        # 3. Some temporal variation (not static)
        # 4. Motion ratio > threshold for dynamic signals
        
        energy_ok = energy > self.config.occupancy_energy_threshold
        diversity_ok = sc_diversity > 0.01
        motion_ok = motion_ratio > 1.2
        
        is_occupied = energy_ok and diversity_ok and motion_ok
        
        # Store history
        self.amp_history.append(mean_amp)
        self.motion_history.append(motion_ratio)
        
        return is_occupied, motion_ratio, energy
    
    def assess_signal_quality(self, signal_data) -> Dict[str, float]:
        """
        Assess the quality of the extracted heartbeat signal.
        
        Returns:
            Dict with quality metrics
        """
        if len(signal_data) < 100:
            return {"snr": 0.0, "prominence": 0.0, "quality": 0.0}
        
        # Compute PSD
        nperseg = min(len(signal_data), int(self.config.sampling_rate * 8))
        freqs, psd = signal.welch(
            signal_data,
            fs=self.config.sampling_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling='density'
        )
        
        # Heartbeat band
        hr_min_hz = self.config.heart_rate_min / 60.0
        hr_max_hz = self.config.heart_rate_max / 60.0
        band_mask = (freqs >= hr_min_hz) & (freqs <= hr_max_hz)
        
        if not np.any(band_mask):
            return {"snr": 0.0, "prominence": 0.0, "quality": 0.0}
        
        band_freqs = freqs[band_mask]
        band_psd = psd[band_mask]
        
        # Find peaks
        peaks, props = signal.find_peaks(
            band_psd,
            prominence=np.std(band_psd) * 0.3
        )
        
        if len(peaks) == 0:
            return {"snr": 0.0, "prominence": 0.0, "quality": 0.0}
        
        # Best peak by prominence
        best_idx = peaks[np.argmax(props["prominences"])]
        peak_power = band_psd[best_idx]
        peak_prominence = props["prominences"][best_idx]
        
        # SNR: peak power vs median in band
        median_power = np.median(band_psd)
        snr = peak_power / (median_power + 1e-12)
        snr = np.clip(snr, 0, 20)
        
        # Overall quality (0-1)
        quality = np.clip((snr - 1.0) / 9.0, 0, 1)
        
        return {
            "snr": snr,
            "prominence": peak_prominence,
            "quality": quality,
            "peak_freq": band_freqs[best_idx],
        }
    
    def get_motion_trend(self, window: int = 20) -> np.ndarray:
        """Get recent motion scores for trend analysis."""
        if len(self.motion_history) < window:
            return np.array(list(self.motion_history))
        return np.array(list(self.motion_history)[-window:])
    
    def get_motion_classification(self, motion_score: float) -> str:
        """Classify motion level."""
        if motion_score < 1.2:
            return "static"
        elif motion_score < 2.0:
            return "subtle"
        elif motion_score < 4.0:
            return "active"
        else:
            return "high"
    
    # ============ MODIFIED: EXISTING METHODS WITH OCCUPANCY INTEGRATION ============
    
    def apply_bandpass_filter(self, csi_data, subcarrier_indices=None):
        """Apply bandpass filter to selected subcarriers"""
        if subcarrier_indices is None:
            if self.config.use_all_subcarriers:
                subcarrier_indices = range(csi_data.shape[1])
            else:
                subcarrier_indices = self.select_best_subcarriers(csi_data)
        
        # Design filter
        sos = self.design_bandpass_filter()
        
        # Apply filter to each selected subcarrier
        filtered_data = np.zeros((csi_data.shape[0], len(subcarrier_indices)), dtype=complex)
        
        for idx, sc in enumerate(subcarrier_indices):
            # Use magnitude for heartbeat detection
            magnitude = np.abs(csi_data[:, sc])
            
            # Apply bandpass filter
            filtered_signal = signal.sosfiltfilt(sos, magnitude)
            
            # Store real-valued filtered signal
            filtered_data[:, idx] = filtered_signal
        
        # Average across selected subcarriers
        self.heartbeat_signal = np.mean(filtered_data, axis=1).real
        self.filtered_data = filtered_data.real
        
        return self.heartbeat_signal
    
    def select_best_subcarriers(self, csi_data):
        """Auto-select the best subcarriers for heartbeat detection"""
        magnitudes = np.abs(csi_data)
        n_packets, n_subcarriers = magnitudes.shape
        
        # Calculate metrics for each subcarrier
        metrics = []
        
        for sc in range(n_subcarriers):
            # Get magnitude time series
            mag_ts = magnitudes[:, sc]
            
            # Avoid DC subcarriers (very low mean)
            mean_mag = np.mean(mag_ts)
            if mean_mag < np.percentile(magnitudes, 10):
                metrics.append(-np.inf)
                continue
            
            # Calculate variance (higher is better for detecting changes)
            variance = np.var(mag_ts)
            
            # Calculate signal-to-noise ratio (relative to overall)
            noise_est = np.std(np.diff(mag_ts)) / np.sqrt(2)
            if noise_est > 0:
                snr = variance / (noise_est ** 2)
            else:
                snr = 0
            
            # Calculate peak-to-peak amplitude
            p2p = np.ptp(mag_ts)
            
            # Combined score
            score = snr * (p2p / mean_mag)
            metrics.append(score)
        
        # Sort subcarriers by score
        sorted_indices = np.argsort(metrics)[::-1]
        
        # Select top N percentile
        n_select = max(1, int(n_subcarriers * self.config.auto_select_percentile / 100))
        best_indices = sorted_indices[:n_select]
        
        print(f"Selected {len(best_indices)} subcarriers for heartbeat extraction")
        print(f"  Top 5 subcarriers: {best_indices[:5]}")
        
        return best_indices
    
    def estimate_heart_rate(self, signal_data, sampling_rate=None):
        """Estimate heart rate using frequency analysis"""
        if sampling_rate is None:
            sampling_rate = self.config.sampling_rate
        
        n_samples = len(signal_data)
        
        # Apply window to reduce spectral leakage
        window = np.hanning(n_samples)
        windowed_signal = signal_data * window
        
        # Compute FFT
        fft_data = fft(windowed_signal)
        freqs = fftfreq(n_samples, 1/sampling_rate)
        
        # Only positive frequencies
        positive_indices = (freqs > 0) & (freqs <= 5)
        freqs_positive = freqs[positive_indices]
        fft_positive = np.abs(fft_data[positive_indices])
        
        # Find dominant frequency in heartbeat range
        hr_min_hz = self.config.heart_rate_min / 60.0
        hr_max_hz = self.config.heart_rate_max / 60.0
        
        hr_range = (freqs_positive >= hr_min_hz) & (freqs_positive <= hr_max_hz)
        
        if np.any(hr_range):
            hr_freqs = freqs_positive[hr_range]
            hr_fft = fft_positive[hr_range]
            
            # Find peak
            peak_idx = np.argmax(hr_fft)
            dominant_freq = hr_freqs[peak_idx]
            heart_rate = dominant_freq * 60.0
            
            # Confidence score (peak prominence)
            if len(hr_fft) > 2:
                mean_hr_fft = np.mean(hr_fft)
                peak_value = hr_fft[peak_idx]
                confidence = peak_value / mean_hr_fft if mean_hr_fft > 0 else 0
                confidence = np.clip((confidence - 1) / 9 * 100, 0, 100)
            else:
                confidence = 0
            
            # Assess quality
            quality_metrics = self.assess_signal_quality(signal_data)
            
            # Combined confidence
            confidence = confidence * quality_metrics.get("quality", 0.5)
            
        else:
            dominant_freq = 0
            heart_rate = 0
            confidence = 0
            quality_metrics = {"quality": 0.0, "snr": 0.0}
        
        return {
            'heart_rate': heart_rate,
            'dominant_freq': dominant_freq,
            'confidence': confidence,
            'fft_frequencies': freqs_positive,
            'fft_magnitudes': fft_positive,
            'quality_metrics': quality_metrics
        }
    
    def compute_spectrogram(self, signal_data, nperseg=None):
        """Compute spectrogram for time-frequency analysis"""
        if nperseg is None:
            nperseg = int(self.config.segment_duration * self.config.sampling_rate)
            if nperseg % 2 == 1:
                nperseg += 1
        
        noverlap = int(nperseg * self.config.overlap)
        
        freqs, times, Sxx = signal.spectrogram(
            signal_data,
            fs=self.config.sampling_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density'
        )
        
        self.spectrogram = Sxx
        self.frequencies = freqs
        self.times = times
        
        return freqs, times, Sxx
    
    def detect_peaks(self, signal_data, min_distance=20):
        """Detect peaks in the heartbeat signal"""
        # Find peaks
        peaks, properties = signal.find_peaks(
            signal_data,
            distance=min_distance,
            prominence=np.std(signal_data) * 0.5,
            height=np.mean(signal_data) + np.std(signal_data) * 0.3
        )
        
        # Calculate inter-beat intervals
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.config.sampling_rate
            heart_rates = 60.0 / intervals
            avg_hr = np.mean(heart_rates)
        else:
            intervals = np.array([])
            heart_rates = np.array([])
            avg_hr = 0
        
        return {
            'peaks': peaks,
            'intervals': intervals,
            'heart_rates': heart_rates,
            'avg_heart_rate': avg_hr
        }
    
    # ============ NEW: UPDATED FULL ANALYSIS WITH OCCUPANCY ============
    
    def run_full_analysis(self, csi_data, metadata=None, output_dir='heartbeat_analysis'):
        """Run complete heartbeat extraction and analysis with occupancy detection"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Update sampling rate from metadata
        if metadata and 'rate' in metadata:
            self.config.sampling_rate = metadata['rate']
            print(f"Using sampling rate: {self.config.sampling_rate:.1f} Hz")
        
        print("\n" + "="*60)
        print("HEARTBEAT EXTRACTION AND ANALYSIS")
        print("="*60)
        
        # ============ NEW: Occupancy and motion detection ============
        print("\n0. Detecting occupancy and motion...")
        is_occupied, motion_level, energy = self.detect_occupancy_and_motion(csi_data)
        self.occupancy = is_occupied
        self.motion_level = motion_level
        
        print(f"   Occupancy: {is_occupied}")
        print(f"   Motion level: {motion_level:.3f}")
        print(f"   Energy: {energy:.3f}")
        
        if not is_occupied:
            print("\n⚠️  No occupancy detected. Skipping heartbeat analysis.")
            print("   (Check CSI signal quality or adjust occupancy threshold)")
            return {
                'heartbeat_signal': None,
                'heart_rate': 0,
                'confidence': 0,
                'peaks': [],
                'heart_rates': [],
                'occupancy': False,
                'motion_level': motion_level
            }
        
        # Step 1: Extract heartbeat signal
        print("\n1. Extracting heartbeat signal...")
        print(f"   Bandpass range: {self.config.bandpass_low}-{self.config.bandpass_high} Hz")
        print(f"   Filter order: {self.config.filter_order}")
        
        self.apply_bandpass_filter(csi_data)
        
        # Step 2: Assess signal quality
        print("\n2. Assessing signal quality...")
        quality_metrics = self.assess_signal_quality(self.heartbeat_signal)
        self.signal_quality = quality_metrics.get("quality", 0.0)
        print(f"   Signal quality: {self.signal_quality:.2f}")
        print(f"   SNR: {quality_metrics.get('snr', 0):.2f}")
        
        # Step 3: Estimate heart rate
        print("\n3. Estimating heart rate...")
        hr_result = self.estimate_heart_rate(self.heartbeat_signal)
        heart_rate = hr_result['heart_rate']
        confidence = hr_result['confidence']
        self.confidence = confidence
        
        # Check if HR is valid
        is_valid = (heart_rate > 0 and 
                   heart_rate >= self.config.heart_rate_min and 
                   heart_rate <= self.config.heart_rate_max and
                   self.signal_quality > self.config.min_signal_quality)
        
        if is_valid:
            print(f"   Heart rate: {heart_rate:.1f} BPM")
            print(f"   Confidence: {confidence:.1f}%")
            self.hr_history.append(heart_rate)
            self.last_valid_bpm = heart_rate
        else:
            print(f"   ❌ Invalid heart rate: {heart_rate:.1f} BPM")
            if self.signal_quality <= self.config.min_signal_quality:
                print("      Signal quality too low")
            if heart_rate <= 0:
                print("      No heartbeat detected")
        
        # Step 4: Compute spectrogram
        print("\n4. Computing spectrogram...")
        self.compute_spectrogram(self.heartbeat_signal)
        
        # Step 5: Detect peaks
        print("\n5. Detecting peaks...")
        peak_result = self.detect_peaks(self.heartbeat_signal)
        print(f"   Peaks detected: {len(peak_result['peaks'])}")
        if len(peak_result['peaks']) > 0:
            print(f"   Average HR from peaks: {peak_result['avg_heart_rate']:.1f} BPM")
        
        # Step 6: Generate plots
        print("\n6. Generating visualizations...")
        self.create_heartbeat_plots(output_dir, hr_result, peak_result)
        
        # Step 7: Save results
        self.save_results(output_dir, hr_result, peak_result)
        
        print("\n" + "="*60)
        print(f"Analysis complete! Results saved to '{output_dir}/'")
        print("="*60)
        
        return {
            'heartbeat_signal': self.heartbeat_signal,
            'heart_rate': heart_rate if is_valid else self.last_valid_bpm or 0,
            'confidence': confidence if is_valid else 0,
            'peaks': peak_result['peaks'],
            'heart_rates': peak_result['heart_rates'],
            'occupancy': is_occupied,
            'motion_level': motion_level,
            'signal_quality': self.signal_quality,
            'is_valid': is_valid
        }
    
    def create_heartbeat_plots(self, output_dir, hr_result, peak_result):
        """Create comprehensive heartbeat analysis plots with occupancy info"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Heartbeat waveform with peaks
        ax1 = plt.subplot(3, 2, 1)
        time_axis = np.arange(len(self.heartbeat_signal)) / self.config.sampling_rate
        ax1.plot(time_axis, self.heartbeat_signal, 'b-', linewidth=1, alpha=0.7)
        ax1.plot(time_axis[peak_result['peaks']], 
                self.heartbeat_signal[peak_result['peaks']], 
                'ro', markersize=4, label='Peaks')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Heartbeat Signal with Peak Detection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Frequency domain (FFT)
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(hr_result['fft_frequencies'], hr_result['fft_magnitudes'], 'b-', linewidth=1)
        if hr_result['heart_rate'] > 0:
            ax2.axvline(x=hr_result['dominant_freq'], color='r', linestyle='--', 
                       label=f"Peak: {hr_result['heart_rate']:.1f} BPM")
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Frequency Spectrum')
        ax2.set_xlim(0, 5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Spectrogram
        ax3 = plt.subplot(3, 2, 3)
        freq_range = (self.frequencies >= 0) & (self.frequencies <= 5)
        freqs_plot = self.frequencies[freq_range]
        Sxx_plot = self.spectrogram[freq_range, :]
        
        im = ax3.imshow(
            Sxx_plot, 
            aspect='auto',
            origin='lower',
            extent=[self.times[0], self.times[-1], freqs_plot[0], freqs_plot[-1]],
            cmap='viridis'
        )
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_title('Spectrogram (0-5 Hz)')
        plt.colorbar(im, ax=ax3, label='Power')
        
        # 4. Heart rate over time (sliding window)
        ax4 = plt.subplot(3, 2, 4)
        window_size = int(self.config.segment_duration * self.config.sampling_rate)
        hr_over_time = []
        time_points = []
        
        for i in range(0, len(self.heartbeat_signal) - window_size, window_size // 2):
            window_data = self.heartbeat_signal[i:i+window_size]
            hr_win = self.estimate_heart_rate(window_data)
            hr_over_time.append(hr_win['heart_rate'])
            time_points.append(i / self.config.sampling_rate)
        
        if time_points:
            ax4.plot(time_points, hr_over_time, 'r-', linewidth=2)
            if hr_result['heart_rate'] > 0:
                ax4.axhline(y=hr_result['heart_rate'], color='b', linestyle='--', 
                           label=f"Average: {hr_result['heart_rate']:.1f} BPM")
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Heart Rate (BPM)')
            ax4.set_title('Heart Rate Over Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Inter-beat interval distribution
        ax5 = plt.subplot(3, 2, 5)
        if len(peak_result['heart_rates']) > 1:
            ax5.hist(peak_result['heart_rates'], bins=20, alpha=0.7, color='green')
            ax5.axvline(x=peak_result['avg_heart_rate'], color='r', linestyle='--',
                       label=f"Mean: {peak_result['avg_heart_rate']:.1f} BPM")
            ax5.set_xlabel('Heart Rate (BPM)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Inter-beat Heart Rate Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics with occupancy info
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        
        quality_str = f"{self.signal_quality:.2f}" if self.signal_quality > 0 else "N/A"
        motion_class = self.get_motion_classification(self.motion_level)
        
        stats_text = f"""
        HEARTBEAT ANALYSIS SUMMARY
        
        Bandpass Range:  {self.config.bandpass_low:.1f}-{self.config.bandpass_high:.1f} Hz
        Filter Order:    {self.config.filter_order}
        Sampling Rate:   {self.config.sampling_rate:.1f} Hz
        
        OCCUPANCY:       {'✓' if self.occupancy else '✗'}
        Motion Level:    {self.motion_level:.3f} ({motion_class})
        Signal Quality:  {quality_str}
        
        Heart Rate:      {hr_result['heart_rate']:.1f} BPM
        Confidence:      {hr_result['confidence']:.1f}%
        Dominant Freq:   {hr_result['dominant_freq']:.2f} Hz
        
        Peaks Detected:  {len(peak_result['peaks'])}
        Avg HR (peaks):  {peak_result['avg_heart_rate']:.1f} BPM
        
        Signal Duration: {len(self.heartbeat_signal) / self.config.sampling_rate:.1f} s
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Highlight occupancy status
        if self.occupancy:
            plt.suptitle('Heartbeat Extraction Analysis - OCCUPIED', 
                        fontsize=14, fontweight='bold', color='green')
        else:
            plt.suptitle('Heartbeat Extraction Analysis - VACANT', 
                        fontsize=14, fontweight='bold', color='red')
        
        plt.tight_layout()
        
        # Save plot
        filename = f'{output_dir}/heartbeat_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to '{filename}'")
        plt.close()
        
        # Additional plot: Comparison of subcarriers
        if self.filtered_data is not None and self.filtered_data.shape[1] > 1:
            self.plot_subcarrier_comparison(output_dir)
    
    def plot_subcarrier_comparison(self, output_dir):
        """Plot comparison of filtered signals from different subcarriers"""
        n_subcarriers = self.filtered_data.shape[1]
        n_plot = min(n_subcarriers, self.config.max_subcarriers_to_plot)
        
        fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2 * n_plot), sharex=True)
        
        if n_plot == 1:
            axes = [axes]
        
        time_axis = np.arange(len(self.heartbeat_signal)) / self.config.sampling_rate
        
        for idx in range(n_plot):
            axes[idx].plot(time_axis, self.filtered_data[:, idx], linewidth=1)
            axes[idx].set_ylabel(f'SC {idx}')
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (seconds)')
        fig.suptitle('Heartbeat Signal by Subcarrier', fontsize=12)
        
        filename = f'{output_dir}/subcarrier_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved subcarrier comparison to '{filename}'")
        plt.close()
    
    def save_results(self, output_dir, hr_result, peak_result):
        """Save analysis results to files"""
        # Save heartbeat signal
        np.savetxt(
            f'{output_dir}/heartbeat_signal.txt',
            self.heartbeat_signal,
            header=f'Hearbeat Signal (rate={self.config.sampling_rate}Hz)'
        )
        
        # Save summary with occupancy info
        with open(f'{output_dir}/heartbeat_summary.txt', 'w') as f:
            f.write("HEARTBEAT EXTRACTION SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Bandpass Range:  {self.config.bandpass_low:.1f}-{self.config.bandpass_high:.1f} Hz\n")
            f.write(f"Filter Order:    {self.config.filter_order}\n")
            f.write(f"Sampling Rate:   {self.config.sampling_rate:.1f} Hz\n\n")
            f.write("OCCUPANCY AND MOTION\n")
            f.write("-"*30 + "\n")
            f.write(f"Occupancy:       {self.occupancy}\n")
            f.write(f"Motion Level:    {self.motion_level:.3f}\n")
            f.write(f"Signal Quality:  {self.signal_quality:.2f}\n\n")
            f.write("HEART RATE\n")
            f.write("-"*30 + "\n")
            f.write(f"Heart Rate:      {hr_result['heart_rate']:.1f} BPM\n")
            f.write(f"Dominant Freq:   {hr_result['dominant_freq']:.2f} Hz\n")
            f.write(f"Confidence:      {hr_result['confidence']:.1f}%\n\n")
            f.write(f"Peaks Detected:  {len(peak_result['peaks'])}\n")
            f.write(f"Avg HR (peaks):  {peak_result['avg_heart_rate']:.1f} BPM\n")
            f.write(f"Signal Duration: {len(self.heartbeat_signal) / self.config.sampling_rate:.1f} s\n")
        
        print(f"  Saved results to '{output_dir}/heartbeat_summary.txt'")
    
    # ============ NEW: REAL-TIME UPDATE METHOD ============
    
    def update_realtime(self, csi_data: np.ndarray, timestamp: float) -> Dict:
        """
        Real-time update for live monitoring.
        
        Args:
            csi_data: Single frame or batch of CSI data
            timestamp: Current timestamp
        
        Returns:
            Dict with current metrics
        """
        # Store frame for rolling analysis
        # (Implementation depends on how you want to handle rolling windows)
        # This is a simplified version
        
        # Detect occupancy and motion
        is_occupied, motion_level, energy = self.detect_occupancy_and_motion(
            csi_data if csi_data.ndim == 2 else csi_data.reshape(1, -1)
        )
        self.occupancy = is_occupied
        self.motion_level = motion_level
        
        # Only process HR if occupied
        if is_occupied and self.signal_quality > self.config.min_signal_quality:
            hr_result = self.estimate_heart_rate(self.heartbeat_signal)
            heart_rate = hr_result['heart_rate']
            confidence = hr_result['confidence']
            
            if heart_rate > 0:
                self.hr_history.append(heart_rate)
                self.last_valid_bpm = heart_rate
        else:
            heart_rate = self.last_valid_bpm or 0
            confidence = 0
        
        return {
            'heart_rate': heart_rate,
            'confidence': confidence,
            'occupancy': is_occupied,
            'motion_level': motion_level,
            'signal_quality': self.signal_quality,
            'energy': energy
        }


class HeartbeatFilter:
    """Wrapper for integrating heartbeat extraction with existing pipeline"""
    
    def __init__(self, low_freq=0.8, high_freq=2.5, order=3):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        self.sos = None
        self.sampling_rate = 100.0
        
    def design_filter(self, sampling_rate):
        """Design the bandpass filter"""
        self.sampling_rate = sampling_rate
        nyquist = sampling_rate / 2
        
        # Normalize frequencies
        low = max(0.001, self.low_freq / nyquist)
        high = min(0.999, self.high_freq / nyquist)
        
        # Design bandpass filter
        self.sos = signal.butter(
            self.order,
            [low, high],
            btype='band',
            output='sos'
        )
        
    def filter_csi_data(self, csi_data):
        """Apply bandpass filter to CSI magnitude data"""
        if self.sos is None:
            self.design_filter(100.0)
        
        # Use magnitude for heartbeat detection
        magnitudes = np.abs(csi_data)
        
        # Apply filter to each subcarrier
        filtered = np.zeros_like(csi_data)
        for sc in range(csi_data.shape[1]):
            filtered[:, sc] = signal.sosfiltfilt(self.sos, magnitudes[:, sc])
        
        return filtered


# Example usage
if __name__ == "__main__":
    # Test with synthetic data
    print("Testing heartbeat extraction with occupancy detection...")
    
    # Generate synthetic heartbeat signal
    fs = 100.0
    t = np.arange(0, 30, 1/fs)
    heart_rate = 72  # BPM
    heart_freq = heart_rate / 60.0
    
    # Create synthetic heartbeat with noise
    synthetic_heartbeat = np.sin(2 * np.pi * heart_freq * t) * 10
    synthetic_heartbeat += np.sin(4 * np.pi * heart_freq * t) * 3
    synthetic_heartbeat += np.random.randn(len(t)) * 2
    
    # Add respiratory component (0.2 Hz)
    synthetic_heartbeat += np.sin(2 * np.pi * 0.2 * t) * 5
    
    # Create synthetic CSI data (64 subcarriers)
    synthetic_csi = np.zeros((len(t), 64), dtype=complex)
    for sc in range(64):
        scale = 1 + 0.3 * np.sin(sc / 10)
        synthetic_csi[:, sc] = (100 + synthetic_heartbeat * scale) * np.exp(1j * sc * 0.1)
    
    # Configure and run with occupancy detection
    config = HeartbeatConfig()
    config.sampling_rate = fs
    config.bandpass_low = 0.5
    config.bandpass_high = 2.0
    config.occupancy_energy_threshold = 0.01
    
    extractor = HeartbeatExtractor(config)
    result = extractor.run_full_analysis(synthetic_csi, {'rate': fs}, 'test_heartbeat')
    
    print(f"\nResults:")
    print(f"  Occupancy: {result['occupancy']}")
    print(f"  Motion level: {result['motion_level']:.3f}")
    print(f"  Heart rate: {result['heart_rate']:.1f} BPM")
    print(f"  Confidence: {result['confidence']:.1f}%")