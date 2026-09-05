#!/usr/bin/env python3
# heartbeat_extraction.py - Heartbeat Detection Module

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


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


class HeartbeatExtractor:
    """Extract heartbeat signal from CSI data"""
    
    def __init__(self, config=None):
        self.config = config if config else HeartbeatConfig()
        self.filtered_data = None
        self.heartbeat_signal = None
        self.heart_rate_history = []
        self.spectrogram = None
        self.frequencies = None
        self.times = None
        
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
    
    def apply_bandpass_filter(self, csi_data, subcarrier_indices=None):
        """Apply bandpass filter to selected subcarriers"""
        if subcarrier_indices is None:
            # If no indices provided, use all or auto-select
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
            noise_est = np.std(np.diff(mag_ts)) / np.sqrt(2)  # Estimate noise
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
        positive_indices = (freqs > 0) & (freqs <= 5)  # Up to 5 Hz
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
            heart_rate = dominant_freq * 60.0  # Convert to BPM
            
            # Confidence score (peak prominence)
            if len(hr_fft) > 2:
                mean_hr_fft = np.mean(hr_fft)
                peak_value = hr_fft[peak_idx]
                confidence = peak_value / mean_hr_fft if mean_hr_fft > 0 else 0
            else:
                confidence = 0
        else:
            dominant_freq = 0
            heart_rate = 0
            confidence = 0
        
        return {
            'heart_rate': heart_rate,
            'dominant_freq': dominant_freq,
            'confidence': confidence,
            'fft_frequencies': freqs_positive,
            'fft_magnitudes': fft_positive
        }
    
    def compute_spectrogram(self, signal_data, nperseg=None):
        """Compute spectrogram for time-frequency analysis"""
        if nperseg is None:
            nperseg = int(self.config.segment_duration * self.config.sampling_rate)
            if nperseg % 2 == 1:
                nperseg += 1  # Make even
        
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
    
    def run_full_analysis(self, csi_data, metadata=None, output_dir='heartbeat_analysis'):
        """Run complete heartbeat extraction and analysis"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Update sampling rate from metadata
        if metadata and 'rate' in metadata:
            self.config.sampling_rate = metadata['rate']
            print(f"Using sampling rate: {self.config.sampling_rate:.1f} Hz")
        
        print("\n" + "="*60)
        print("HEARTBEAT EXTRACTION AND ANALYSIS")
        print("="*60)
        
        # Step 1: Extract heartbeat signal
        print("\n1. Extracting heartbeat signal...")
        print(f"   Bandpass range: {self.config.bandpass_low}-{self.config.bandpass_high} Hz")
        print(f"   Filter order: {self.config.filter_order}")
        
        self.apply_bandpass_filter(csi_data)
        
        # Step 2: Estimate heart rate
        print("\n2. Estimating heart rate...")
        hr_result = self.estimate_heart_rate(self.heartbeat_signal)
        print(f"   Heart rate: {hr_result['heart_rate']:.1f} BPM")
        print(f"   Confidence: {hr_result['confidence']:.2f}")
        
        # Step 3: Compute spectrogram
        print("\n3. Computing spectrogram...")
        self.compute_spectrogram(self.heartbeat_signal)
        
        # Step 4: Detect peaks
        print("\n4. Detecting peaks...")
        peak_result = self.detect_peaks(self.heartbeat_signal)
        print(f"   Peaks detected: {len(peak_result['peaks'])}")
        if len(peak_result['peaks']) > 0:
            print(f"   Average HR from peaks: {peak_result['avg_heart_rate']:.1f} BPM")
        
        # Step 5: Generate plots
        print("\n5. Generating visualizations...")
        self.create_heartbeat_plots(output_dir, hr_result, peak_result)
        
        # Step 6: Save results
        self.save_results(output_dir, hr_result, peak_result)
        
        print("\n" + "="*60)
        print(f"Analysis complete! Results saved to '{output_dir}/'")
        print("="*60)
        
        return {
            'heartbeat_signal': self.heartbeat_signal,
            'heart_rate': hr_result['heart_rate'],
            'confidence': hr_result['confidence'],
            'peaks': peak_result['peaks'],
            'heart_rates': peak_result['heart_rates']
        }
    
    def create_heartbeat_plots(self, output_dir, hr_result, peak_result):
        """Create comprehensive heartbeat analysis plots"""
        
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
        # Compute sliding window heart rates
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
        
        # 6. Summary statistics
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        
        stats_text = f"""
        HEARTBEAT ANALYSIS SUMMARY
        
        Bandpass Range:  {self.config.bandpass_low:.1f}-{self.config.bandpass_high:.1f} Hz
        Filter Order:    {self.config.filter_order}
        Sampling Rate:   {self.config.sampling_rate:.1f} Hz
        
        Heart Rate:      {hr_result['heart_rate']:.1f} BPM
        Dominant Freq:   {hr_result['dominant_freq']:.2f} Hz
        Confidence:      {hr_result['confidence']:.2f}
        
        Peaks Detected:  {len(peak_result['peaks'])}
        Avg HR (peaks):  {peak_result['avg_heart_rate']:.1f} BPM
        
        Signal Duration: {len(self.heartbeat_signal) / self.config.sampling_rate:.1f} s
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Heartbeat Extraction Analysis', fontsize=14, fontweight='bold')
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
        
        # Save summary
        with open(f'{output_dir}/heartbeat_summary.txt', 'w') as f:
            f.write("HEARTBEAT EXTRACTION SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Bandpass Range:  {self.config.bandpass_low:.1f}-{self.config.bandpass_high:.1f} Hz\n")
            f.write(f"Filter Order:    {self.config.filter_order}\n")
            f.write(f"Sampling Rate:   {self.config.sampling_rate:.1f} Hz\n\n")
            f.write(f"Heart Rate:      {hr_result['heart_rate']:.1f} BPM\n")
            f.write(f"Dominant Freq:   {hr_result['dominant_freq']:.2f} Hz\n")
            f.write(f"Confidence:      {hr_result['confidence']:.2f}\n\n")
            f.write(f"Peaks Detected:  {len(peak_result['peaks'])}\n")
            f.write(f"Avg HR (peaks):  {peak_result['avg_heart_rate']:.1f} BPM\n")
            f.write(f"Signal Duration: {len(self.heartbeat_signal) / self.config.sampling_rate:.1f} s\n")
        
        print(f"  Saved results to '{output_dir}/heartbeat_summary.txt'")


# Integration with Main Pipeline
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
            self.design_filter(100.0)  # Default
        
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
    print("Testing heartbeat extraction with synthetic data...")
    
    # Generate synthetic heartbeat signal
    fs = 100.0
    t = np.arange(0, 30, 1/fs)
    heart_rate = 72  # BPM
    heart_freq = heart_rate / 60.0
    
    # Create synthetic heartbeat with noise
    synthetic_heartbeat = np.sin(2 * np.pi * heart_freq * t) * 10
    synthetic_heartbeat += np.sin(4 * np.pi * heart_freq * t) * 3  # Harmonic
    synthetic_heartbeat += np.random.randn(len(t)) * 2  # Noise
    
    # Add respiratory component (0.2 Hz)
    synthetic_heartbeat += np.sin(2 * np.pi * 0.2 * t) * 5
    
    # Create synthetic CSI data (64 subcarriers, some with signal)
    synthetic_csi = np.zeros((len(t), 64), dtype=complex)
    for sc in range(64):
        # Add some variation across subcarriers
        scale = 1 + 0.3 * np.sin(sc / 10)
        synthetic_csi[:, sc] = (100 + synthetic_heartbeat * scale) * np.exp(1j * sc * 0.1)
    
    # Configure and run
    config = HeartbeatConfig()
    config.sampling_rate = fs
    config.bandpass_low = 0.5
    config.bandpass_high = 2.0
    
    extractor = HeartbeatExtractor(config)
    result = extractor.run_full_analysis(synthetic_csi, {'rate': fs}, 'test_heartbeat')
    
    print(f"\nDetected heart rate: {result['heart_rate']:.1f} BPM (expected: {heart_rate} BPM)")
    print(f"Confidence: {result['confidence']:.2f}")