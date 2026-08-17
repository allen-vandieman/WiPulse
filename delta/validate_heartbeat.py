#!/usr/bin/env python3
"""
CSI Heartbeat Validation Script
Aligns ground truth heartbeat data (from GPIO sensor) with CSI data
to validate heartbeat detection accuracy.

Usage: python validate_heartbeat.py --csi captures/csi_capture_*.csv --groundtruth output/ground_truth_*.csv

Supports:
- PCAP or CSV CSI data
- Ground truth CSV from heartbeat-giop.py
- Automatic timestamp alignment
- Comprehensive comparison plots
- Statistical validation metrics
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

# Try importing sklearn, fallback if not available
try:
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Some metrics will be unavailable.")
    print("Install with: pip install scikit-learn")
    
    # Fallback implementations
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    
    def mean_squared_error(y_true, y_pred):
        return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    
    def pearsonr(x, y):
        x = np.array(x)
        y = np.array(y)
        return np.corrcoef(x, y)[0, 1], 0

# Import CSI modules
from read_csv_csi import CSIDataLoader
from read_pcap import PCAPDataLoader
from filters import CSIFilter, FilterConfig
from heartbeat_extraction import HeartbeatExtractor, HeartbeatConfig


class HeartbeatValidator:
    """Align and compare CSI-detected heartbeat with ground truth"""
    
    def __init__(self, csi_data=None, groundtruth_df=None, metadata=None):
        self.csi_data = csi_data
        self.groundtruth_df = groundtruth_df
        self.metadata = metadata
        
        # Aligned data
        self.csi_heartbeat = None
        self.groundtruth_signal = None
        self.aligned_timestamps = None
        
        # Detection results
        self.csi_peaks = None
        self.groundtruth_peaks = None
        self.csi_heart_rates = None
        self.groundtruth_heart_rates = None
        
        # Validation metrics
        self.metrics = {}
        
    def load_csi_data(self, file_path):
        """Load CSI data from PCAP or CSV"""
        print(f"\nLoading CSI data from: {file_path}")
        
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            loader = CSIDataLoader(str(file_path))
            self.csi_data, self.metadata = loader.load()
        else:
            loader = PCAPDataLoader(str(file_path))
            self.csi_data, self.metadata = loader.load()
        
        print(f"  Loaded {self.metadata['n_packets']} packets at {self.metadata['rate']:.1f} Hz")
        return self.csi_data, self.metadata
    
    def load_groundtruth(self, csv_path):
        """Load ground truth heartbeat data from CSV"""
        print(f"\nLoading ground truth from: {csv_path}")
        self.groundtruth_df = pd.read_csv(csv_path)
        print(f"  Loaded {len(self.groundtruth_df)} beats")
        print(f"  Columns: {list(self.groundtruth_df.columns)}")
        
        # Convert timestamps if needed
        if 'ts' in self.groundtruth_df.columns:
            # Check if timestamps are Unix timestamps
            if self.groundtruth_df['ts'].iloc[0] > 1e9:
                # Already Unix timestamp
                self.groundtruth_df['timestamp'] = self.groundtruth_df['ts']
            else:
                # Might be relative timestamps, use as-is
                self.groundtruth_df['timestamp'] = self.groundtruth_df['ts']
        
        return self.groundtruth_df
    
    def extract_csi_heartbeat(self, low_freq=0.8, high_freq=2.5, order=3):
        """Extract heartbeat signal from CSI data"""
        print("\nExtracting heartbeat from CSI data...")
        
        # Configure heartbeat extraction
        config = HeartbeatConfig()
        config.bandpass_low = low_freq
        config.bandpass_high = high_freq
        config.filter_order = order
        config.sampling_rate = self.metadata['rate']
        config.use_all_subcarriers = False  # Auto-select best
        
        # Extract heartbeat
        extractor = HeartbeatExtractor(config)
        
        # Detect problematic subcarriers and filter
        filter_obj = CSIFilter(FilterConfig())
        filtered_csi = filter_obj.apply_all_filters(self.csi_data, verbose=False)
        
        # Extract heartbeat signal
        self.csi_heartbeat = extractor.apply_bandpass_filter(filtered_csi)
        self.csi_heart_rate_fft = extractor.estimate_heart_rate(self.csi_heartbeat)
        
        # Detect peaks
        peak_result = extractor.detect_peaks(self.csi_heartbeat)
        self.csi_peaks = peak_result['peaks']
        self.csi_heart_rates = peak_result['heart_rates']
        
        # Create timestamp array for CSI
        self.csi_timestamps = np.arange(len(self.csi_heartbeat)) / config.sampling_rate
        
        print(f"  Detected {len(self.csi_peaks)} peaks")
        print(f"  Heart rate (FFT): {self.csi_heart_rate_fft['heart_rate']:.1f} BPM")
        if len(self.csi_heart_rates) > 0:
            print(f"  Avg HR (peaks): {np.mean(self.csi_heart_rates):.1f} BPM")
        
        return self.csi_heartbeat, self.csi_peaks
    
    def align_timestamps(self, groundtruth_start_time=None, csi_start_time=None):
        """Align ground truth and CSI timestamps"""
        print("\nAligning timestamps...")
        
        # Get ground truth beat timestamps
        if 'timestamp' in self.groundtruth_df.columns:
            gt_timestamps = self.groundtruth_df['timestamp'].values
        elif 'ts' in self.groundtruth_df.columns:
            gt_timestamps = self.groundtruth_df['ts'].values
        else:
            # Use index as time
            gt_timestamps = np.arange(len(self.groundtruth_df))
        
        # Use relative timestamps
        gt_timestamps = gt_timestamps - gt_timestamps[0]
        
        # CSI timestamps
        if hasattr(self, 'csi_timestamps'):
            csi_timestamps = self.csi_timestamps
        else:
            csi_timestamps = np.arange(self.csi_data.shape[0]) / self.metadata['rate']
        
        # Align by matching first beat
        # Assume first ground truth beat corresponds to first CSI beat
        if len(self.csi_peaks) > 0 and len(gt_timestamps) > 0:
            # Find optimal offset by cross-correlating beat times
            csi_beat_times = csi_timestamps[self.csi_peaks]
            
            # Find best alignment offset
            best_offset = 0
            best_count = 0
            
            for offset in np.arange(-5, 5, 0.1):
                aligned_csi = csi_beat_times + offset
                # Count matches within tolerance
                matches = 0
                for gt_time in gt_timestamps:
                    if np.any(np.abs(aligned_csi - gt_time) < 0.2):  # 200ms tolerance
                        matches += 1
                if matches > best_count:
                    best_count = matches
                    best_offset = offset
            
            print(f"  Optimal alignment offset: {best_offset:.1f} seconds")
            print(f"  Matched beats: {best_count}/{len(gt_timestamps)}")
            
            # Apply alignment
            self.aligned_csi_timestamps = csi_timestamps + best_offset
            
            # Use aligned timestamps
            self.csi_timestamps = self.aligned_csi_timestamps
        
        self.gt_timestamps = gt_timestamps
        
        # Resample ground truth to match CSI sampling rate
        self._resample_groundtruth()
        
        return self.csi_timestamps, self.gt_timestamps
    
    def _resample_groundtruth(self):
        """Resample ground truth beat events to continuous signal"""
        print("\nResampling ground truth for comparison...")
        
        # Create ground truth signal at CSI sampling rate
        sampling_rate = self.metadata['rate']
        n_samples = len(self.csi_timestamps)
        
        # Initialize signal
        self.groundtruth_signal = np.zeros(n_samples)
        
        # Place impulses at beat times (aligned)
        for gt_time in self.gt_timestamps:
            # Find closest CSI sample
            idx = np.argmin(np.abs(self.csi_timestamps - gt_time))
            if idx < n_samples and np.abs(self.csi_timestamps[idx] - gt_time) < 0.1:
                self.groundtruth_signal[idx] = 1.0
        
        print(f"  Created ground truth signal with {np.sum(self.groundtruth_signal > 0)} beat markers")
        
        # Also create a filtered version for comparison
        # Convolve with a small Gaussian to create continuous signal
        kernel_width = int(0.05 * sampling_rate)  # 50ms
        if kernel_width > 0:
            kernel = np.exp(-np.linspace(-3, 3, kernel_width*2+1)**2 / 2)
            kernel = kernel / np.sum(kernel)
            self.groundtruth_signal_smooth = np.convolve(self.groundtruth_signal, kernel, mode='same')
        else:
            self.groundtruth_signal_smooth = self.groundtruth_signal
        
        # Detect peaks in ground truth for comparison
        self.groundtruth_peaks = find_peaks(
            self.groundtruth_signal_smooth,
            height=0.1,
            distance=int(0.3 * sampling_rate)  # Min 300ms between beats
        )[0]
        
        # Calculate ground truth heart rates
        if len(self.groundtruth_peaks) > 1:
            gt_intervals = np.diff(self.groundtruth_peaks) / sampling_rate
            self.groundtruth_heart_rates = 60.0 / gt_intervals
    
    def compute_validation_metrics(self):
        """Compute comprehensive validation metrics"""
        print("\nComputing validation metrics...")
        
        # 1. Beat detection accuracy
        if len(self.csi_peaks) > 0 and hasattr(self, 'groundtruth_peaks'):
            csi_beat_times = self.csi_timestamps[self.csi_peaks]
            gt_beat_times = self.csi_timestamps[self.groundtruth_peaks] if hasattr(self, 'groundtruth_peaks') else []
            
            # Match beats
            matched_pairs = []
            used_gt = set()
            used_csi = set()
            
            for i, csi_time in enumerate(csi_beat_times):
                # Find closest ground truth beat
                best_match = None
                best_dist = np.inf
                for j, gt_time in enumerate(gt_beat_times):
                    if j in used_gt:
                        continue
                    dist = abs(csi_time - gt_time)
                    if dist < best_dist and dist < 0.2:  # 200ms tolerance
                        best_dist = dist
                        best_match = j
                
                if best_match is not None:
                    matched_pairs.append({
                        'csi_idx': i,
                        'gt_idx': best_match,
                        'csi_time': csi_time,
                        'gt_time': gt_beat_times[best_match],
                        'error': best_dist
                    })
                    used_gt.add(best_match)
                    used_csi.add(i)
            
            # Calculate metrics
            true_positives = len(matched_pairs)
            false_positives = len(csi_beat_times) - true_positives
            false_negatives = len(gt_beat_times) - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            self.metrics['precision'] = precision
            self.metrics['recall'] = recall
            self.metrics['f1_score'] = f1_score
            self.metrics['true_positives'] = true_positives
            self.metrics['false_positives'] = false_positives
            self.metrics['false_negatives'] = false_negatives
            self.metrics['matched_pairs'] = matched_pairs
            
            # Timing errors
            if matched_pairs:
                errors = [p['error'] for p in matched_pairs]
                self.metrics['mean_abs_error'] = np.mean(errors)
                self.metrics['std_error'] = np.std(errors)
                self.metrics['max_error'] = np.max(errors)
        
        # 2. Heart rate accuracy
        if len(self.csi_heart_rates) > 0 and hasattr(self, 'groundtruth_heart_rates') and len(self.groundtruth_heart_rates) > 0:
            # Align heart rate sequences
            min_len = min(len(self.csi_heart_rates), len(self.groundtruth_heart_rates))
            if min_len > 0:
                csi_hr = self.csi_heart_rates[:min_len]
                gt_hr = self.groundtruth_heart_rates[:min_len]
                
                self.metrics['heart_rate_mae'] = mean_absolute_error(gt_hr, csi_hr)
                self.metrics['heart_rate_rmse'] = np.sqrt(mean_squared_error(gt_hr, csi_hr))
                corr, _ = pearsonr(csi_hr, gt_hr)
                self.metrics['heart_rate_correlation'] = corr
                self.metrics['heart_rate_mean_diff'] = np.mean(csi_hr - gt_hr)
                
                print(f"  HR MAE: {self.metrics['heart_rate_mae']:.2f} BPM")
                print(f"  HR RMSE: {self.metrics['heart_rate_rmse']:.2f} BPM")
                print(f"  HR Correlation: {self.metrics['heart_rate_correlation']:.3f}")
        
        # 3. Overall accuracy
        self.metrics['beat_detection_accuracy'] = (self.metrics.get('true_positives', 0) / 
                                                   (self.metrics.get('true_positives', 0) + 
                                                    self.metrics.get('false_positives', 0) + 
                                                    self.metrics.get('false_negatives', 0)) if (self.metrics.get('true_positives', 0) + self.metrics.get('false_positives', 0) + self.metrics.get('false_negatives', 0)) > 0 else 0)
        
        # Print summary
        self._print_metrics_summary()
        
        return self.metrics
    
    def _print_metrics_summary(self):
        """Print a summary of validation metrics"""
        print("\n" + "="*60)
        print("VALIDATION METRICS SUMMARY")
        print("="*60)
        
        print("\nBeat Detection Accuracy:")
        print(f"  True Positives:  {self.metrics.get('true_positives', 0)}")
        print(f"  False Positives: {self.metrics.get('false_positives', 0)}")
        print(f"  False Negatives: {self.metrics.get('false_negatives', 0)}")
        print(f"  Precision:       {self.metrics.get('precision', 0):.3f}")
        print(f"  Recall:          {self.metrics.get('recall', 0):.3f}")
        print(f"  F1-Score:        {self.metrics.get('f1_score', 0):.3f}")
        print(f"  Accuracy:        {self.metrics.get('beat_detection_accuracy', 0):.3f}")
        
        if 'mean_abs_error' in self.metrics:
            print(f"\nTiming Accuracy:")
            print(f"  Mean Abs Error:  {self.metrics['mean_abs_error']*1000:.1f} ms")
            print(f"  Std Error:       {self.metrics['std_error']*1000:.1f} ms")
            print(f"  Max Error:       {self.metrics['max_error']*1000:.1f} ms")
        
        if 'heart_rate_mae' in self.metrics:
            print(f"\nHeart Rate Accuracy:")
            print(f"  MAE:             {self.metrics['heart_rate_mae']:.2f} BPM")
            print(f"  RMSE:            {self.metrics['heart_rate_rmse']:.2f} BPM")
            print(f"  Correlation:     {self.metrics['heart_rate_correlation']:.3f}")
        
        # Confidence assessment
        print("\nConfidence Assessment:")
        f1 = self.metrics.get('f1_score', 0)
        if f1 >= 0.9:
            print("  ✅ EXCELLENT: CSI heartbeat detection strongly matches ground truth")
        elif f1 >= 0.7:
            print("  ✅ GOOD: CSI heartbeat detection matches ground truth well")
        elif f1 >= 0.5:
            print("  ⚠️ MODERATE: CSI heartbeat detection partially matches ground truth")
        else:
            print("  ❌ POOR: CSI heartbeat detection does not match ground truth")
    
    def create_comparison_plots(self, output_dir='validation'):
        """Create comprehensive comparison plots"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\nCreating comparison plots...")
        
        # Figure with multiple subplots
        fig = plt.figure(figsize=(16, 14))
        
        # 1. Overlaid heartbeat signals
        ax1 = plt.subplot(4, 2, 1)
        time_axis = self.csi_timestamps
        
        # Normalize CSI signal
        if np.std(self.csi_heartbeat) > 0:
            csi_normalized = (self.csi_heartbeat - np.mean(self.csi_heartbeat)) / np.std(self.csi_heartbeat)
        else:
            csi_normalized = self.csi_heartbeat - np.mean(self.csi_heartbeat)
        
        ax1.plot(time_axis, csi_normalized, 'b-', label='CSI Detected', alpha=0.7, linewidth=1)
        
        # Plot ground truth
        if hasattr(self, 'groundtruth_signal_smooth'):
            gt_normalized = self.groundtruth_signal_smooth * 2 - 1
            ax1.plot(time_axis, gt_normalized, 'r-', label='Ground Truth', alpha=0.7, linewidth=1)
            
            # Mark beats
            if hasattr(self, 'groundtruth_peaks'):
                ax1.scatter(time_axis[self.groundtruth_peaks], 
                           np.ones(len(self.groundtruth_peaks)), 
                           c='red', marker='^', s=50, label='GT Beats', zorder=5)
        
        ax1.scatter(time_axis[self.csi_peaks], 
                   np.ones(len(self.csi_peaks)), 
                   c='blue', marker='^', s=50, label='CSI Beats', zorder=5)
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Normalized Amplitude')
        ax1.set_title('CSI vs Ground Truth Heartbeat Signal')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Beat timing comparison
        ax2 = plt.subplot(4, 2, 2)
        if 'matched_pairs' in self.metrics:
            matched = self.metrics['matched_pairs']
            ax2.scatter([p['gt_time'] for p in matched], 
                       [p['csi_time'] for p in matched], 
                       c='green', alpha=0.6, s=20)
            
            # Plot diagonal
            if matched:
                min_t = min(min([p['gt_time'] for p in matched]), min([p['csi_time'] for p in matched]))
                max_t = max(max([p['gt_time'] for p in matched]), max([p['csi_time'] for p in matched]))
                ax2.plot([min_t, max_t], [min_t, max_t], 'k--', alpha=0.5)
            
            ax2.set_xlabel('Ground Truth Beat Time (s)')
            ax2.set_ylabel('CSI Detected Beat Time (s)')
            ax2.set_title('Beat Timing Comparison')
            ax2.grid(True, alpha=0.3)
        
        # 3. Heart rate comparison
        ax3 = plt.subplot(4, 2, 3)
        if hasattr(self, 'csi_heart_rates') and hasattr(self, 'groundtruth_heart_rates'):
            min_len = min(len(self.csi_heart_rates), len(self.groundtruth_heart_rates))
            if min_len > 0:
                csi_hr = self.csi_heart_rates[:min_len]
                gt_hr = self.groundtruth_heart_rates[:min_len]
                
                ax3.plot(csi_hr, 'b-', label='CSI HR', linewidth=1.5)
                ax3.plot(gt_hr, 'r-', label='Ground Truth HR', linewidth=1.5)
                ax3.set_xlabel('Beat Number')
                ax3.set_ylabel('Heart Rate (BPM)')
                ax3.set_title('Heart Rate Comparison')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # 4. Heart rate scatter
        ax4 = plt.subplot(4, 2, 4)
        if hasattr(self, 'csi_heart_rates') and hasattr(self, 'groundtruth_heart_rates'):
            min_len = min(len(self.csi_heart_rates), len(self.groundtruth_heart_rates))
            if min_len > 0:
                csi_hr = self.csi_heart_rates[:min_len]
                gt_hr = self.groundtruth_heart_rates[:min_len]
                
                ax4.scatter(gt_hr, csi_hr, alpha=0.6)
                
                # Plot diagonal
                min_hr = min(min(gt_hr), min(csi_hr)) - 5
                max_hr = max(max(gt_hr), max(csi_hr)) + 5
                ax4.plot([min_hr, max_hr], [min_hr, max_hr], 'k--', alpha=0.5)
                
                ax4.set_xlabel('Ground Truth HR (BPM)')
                ax4.set_ylabel('CSI HR (BPM)')
                ax4.set_title('Heart Rate Correlation')
                ax4.grid(True, alpha=0.3)
                
                # Add correlation text
                if 'heart_rate_correlation' in self.metrics:
                    ax4.text(0.05, 0.95, f"r = {self.metrics['heart_rate_correlation']:.3f}", 
                            transform=ax4.transAxes, fontsize=12, verticalalignment='top')
        
        # 5. Error distribution
        ax5 = plt.subplot(4, 2, 5)
        if 'matched_pairs' in self.metrics and self.metrics['matched_pairs']:
            errors = [p['error'] * 1000 for p in self.metrics['matched_pairs']]  # Convert to ms
            if errors:
                ax5.hist(errors, bins=min(30, len(errors)), alpha=0.7, color='purple')
                ax5.axvline(x=np.mean(errors), color='r', linestyle='--', 
                           label=f'Mean: {np.mean(errors):.1f} ms')
                ax5.set_xlabel('Timing Error (ms)')
                ax5.set_ylabel('Frequency')
                ax5.set_title('Beat Timing Error Distribution')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        # 6. Confusion matrix
        ax6 = plt.subplot(4, 2, 6)
        tn = 0  # No true negatives for beat detection
        fp = self.metrics.get('false_positives', 0)
        fn = self.metrics.get('false_negatives', 0)
        tp = self.metrics.get('true_positives', 0)
        
        cm = np.array([[tp, fp], [fn, tn]])
        im = ax6.imshow(cm, cmap='Blues', interpolation='nearest')
        ax6.set_xticks([0, 1])
        ax6.set_yticks([0, 1])
        ax6.set_xticklabels(['Beat', 'No Beat'])
        ax6.set_yticklabels(['Beat', 'No Beat'])
        ax6.set_xlabel('Predicted')
        ax6.set_ylabel('True')
        ax6.set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax6.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
        plt.colorbar(im, ax=ax6)
        
        # 7. Spectrogram comparison
        ax7 = plt.subplot(4, 2, 7)
        # Compute spectrogram of CSI signal
        fs = self.metadata['rate']
        nperseg = int(fs * 5)  # 5 second windows
        noverlap = int(nperseg * 0.8)
        
        if len(self.csi_heartbeat) > nperseg:
            f, t, Sxx = signal.spectrogram(
                self.csi_heartbeat - np.mean(self.csi_heartbeat),
                fs=fs,
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density'
            )
            
            # Plot only up to 5 Hz
            freq_mask = f <= 5
            if np.any(freq_mask):
                im = ax7.imshow(
                    10 * np.log10(Sxx[freq_mask, :] + 1e-10),
                    aspect='auto',
                    origin='lower',
                    extent=[t[0], t[-1], f[freq_mask][0], f[freq_mask][-1]],
                    cmap='viridis'
                )
                ax7.set_xlabel('Time (seconds)')
                ax7.set_ylabel('Frequency (Hz)')
                ax7.set_title('CSI Spectrogram (0-5 Hz)')
                plt.colorbar(im, ax=ax7, label='Power (dB)')
                
                # Overlay ground truth beats
                if hasattr(self, 'groundtruth_peaks'):
                    gt_times = self.csi_timestamps[self.groundtruth_peaks]
                    for gt_time in gt_times:
                        if gt_time < t[-1] and gt_time > t[0]:
                            ax7.axvline(x=gt_time, color='red', alpha=0.3, linewidth=0.5)
        
        # 8. Performance metrics text
        ax8 = plt.subplot(4, 2, 8)
        ax8.axis('off')
        
        metrics_text = f"""
        PERFORMANCE METRICS
        
        Beat Detection:
        • Precision:  {self.metrics.get('precision', 0):.3f}
        • Recall:     {self.metrics.get('recall', 0):.3f}
        • F1-Score:   {self.metrics.get('f1_score', 0):.3f}
        • Accuracy:   {self.metrics.get('beat_detection_accuracy', 0):.3f}
        
        Timing:
        • Mean Error: {self.metrics.get('mean_abs_error', 0)*1000:.1f} ms
        • Std Error:  {self.metrics.get('std_error', 0)*1000:.1f} ms
        
        Heart Rate:
        • MAE:        {self.metrics.get('heart_rate_mae', 0):.2f} BPM
        • RMSE:       {self.metrics.get('heart_rate_rmse', 0):.2f} BPM
        • Correlation:{self.metrics.get('heart_rate_correlation', 0):.3f}
        """
        
        ax8.text(0.1, 0.9, metrics_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle('CSI Heartbeat Validation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        filename = f'{output_dir}/validation_results.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to '{filename}'")
        plt.close()
        
        # Create additional detailed plots
        self.create_detailed_plots(output_dir)
    
    def create_detailed_plots(self, output_dir):
        """Create more detailed analysis plots"""
        # 1. Detailed waveform comparison (zoomed)
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # Zoom into a 30-second segment
        if len(self.csi_timestamps) > int(30 * self.metadata['rate']):
            start_idx = int(10 * self.metadata['rate'])  # Skip first 10s
            end_idx = start_idx + int(30 * self.metadata['rate'])
        else:
            start_idx = 0
            end_idx = len(self.csi_timestamps)
        
        if end_idx > start_idx:
            time_axis = self.csi_timestamps[start_idx:end_idx]
            csi_segment = self.csi_heartbeat[start_idx:end_idx]
            
            # Normalize
            if np.std(csi_segment) > 0:
                csi_segment = (csi_segment - np.mean(csi_segment)) / np.std(csi_segment)
            
            axes[0].plot(time_axis, csi_segment, 'b-', alpha=0.7, linewidth=1, label='CSI Signal')
            
            # Ground truth
            if hasattr(self, 'groundtruth_signal_smooth'):
                gt_segment = self.groundtruth_signal_smooth[start_idx:end_idx]
                gt_segment = gt_segment * 2 - 1
                axes[0].plot(time_axis, gt_segment, 'r-', alpha=0.7, linewidth=1, label='Ground Truth')
            
            # Mark beats
            csi_peaks_in_view = self.csi_peaks[(self.csi_peaks >= start_idx) & (self.csi_peaks < end_idx)]
            axes[0].scatter(self.csi_timestamps[csi_peaks_in_view], 
                           np.ones(len(csi_peaks_in_view)), 
                           c='blue', marker='^', s=80, label='CSI Beats', zorder=5)
            
            if hasattr(self, 'groundtruth_peaks'):
                gt_peaks_in_view = self.groundtruth_peaks[(self.groundtruth_peaks >= start_idx) & (self.groundtruth_peaks < end_idx)]
                axes[0].scatter(self.csi_timestamps[gt_peaks_in_view], 
                               -np.ones(len(gt_peaks_in_view)), 
                               c='red', marker='v', s=80, label='GT Beats', zorder=5)
            
            axes[0].set_xlabel('Time (seconds)')
            axes[0].set_ylabel('Normalized Amplitude')
            axes[0].set_title('Detailed Waveform Comparison (30s segment)')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(-2, 2)
        
        # 2. Error over time
        if 'matched_pairs' in self.metrics and self.metrics['matched_pairs']:
            matched = self.metrics['matched_pairs']
            times = [p['gt_time'] for p in matched]
            errors = [p['error'] * 1000 for p in matched]  # ms
            
            axes[1].scatter(times, errors, alpha=0.6, s=20)
            axes[1].plot(times, errors, 'b-', alpha=0.3)
            axes[1].axhline(y=np.mean(errors), color='r', linestyle='--', 
                           label=f'Mean: {np.mean(errors):.1f} ms')
            axes[1].axhline(y=np.mean(errors) + np.std(errors), color='orange', linestyle=':', 
                           label=f'±1σ: {np.std(errors):.1f} ms')
            axes[1].axhline(y=np.mean(errors) - np.std(errors), color='orange', linestyle=':')
            
            axes[1].set_xlabel('Time (seconds)')
            axes[1].set_ylabel('Beat Timing Error (ms)')
            axes[1].set_title('Beat Detection Error Over Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Detailed Validation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{output_dir}/validation_detailed.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved detailed plot to '{filename}'")
        plt.close()
    
    def save_validation_report(self, output_dir='validation'):
        """Save validation results to a text file"""
        Path(output_dir).mkdir(exist_ok=True)
        
        with open(f'{output_dir}/validation_report.txt', 'w') as f:
            f.write("CSI HEARTBEAT VALIDATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"CSI Data: {self.metadata.get('n_packets', 0)} packets at {self.metadata.get('rate', 0):.1f} Hz\n")
            f.write(f"Ground Truth: {len(self.groundtruth_df)} beats\n\n")
            
            f.write("BEAT DETECTION METRICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Precision:      {self.metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:         {self.metrics.get('recall', 0):.4f}\n")
            f.write(f"F1-Score:       {self.metrics.get('f1_score', 0):.4f}\n")
            f.write(f"Accuracy:       {self.metrics.get('beat_detection_accuracy', 0):.4f}\n")
            f.write(f"True Positives: {self.metrics.get('true_positives', 0)}\n")
            f.write(f"False Positives:{self.metrics.get('false_positives', 0)}\n")
            f.write(f"False Negatives:{self.metrics.get('false_negatives', 0)}\n\n")
            
            f.write("TIMING ACCURACY:\n")
            f.write("-"*40 + "\n")
            f.write(f"Mean Abs Error: {self.metrics.get('mean_abs_error', 0)*1000:.2f} ms\n")
            f.write(f"Std Deviation:  {self.metrics.get('std_error', 0)*1000:.2f} ms\n")
            f.write(f"Max Error:      {self.metrics.get('max_error', 0)*1000:.2f} ms\n\n")
            
            f.write("HEART RATE ACCURACY:\n")
            f.write("-"*40 + "\n")
            f.write(f"MAE:            {self.metrics.get('heart_rate_mae', 0):.2f} BPM\n")
            f.write(f"RMSE:           {self.metrics.get('heart_rate_rmse', 0):.2f} BPM\n")
            f.write(f"Correlation:    {self.metrics.get('heart_rate_correlation', 0):.4f}\n")
            f.write(f"Mean Difference:{self.metrics.get('heart_rate_mean_diff', 0):.2f} BPM\n\n")
            
            # Confidence assessment
            f.write("CONCLUSION:\n")
            f.write("-"*40 + "\n")
            f1 = self.metrics.get('f1_score', 0)
            if f1 >= 0.9:
                f.write("✅ EXCELLENT: CSI heartbeat detection strongly matches ground truth\n")
                f.write("The system can reliably detect heartbeats from CSI data.\n")
            elif f1 >= 0.7:
                f.write("✅ GOOD: CSI heartbeat detection matches ground truth well\n")
                f.write("The system shows good performance with minor improvements possible.\n")
            elif f1 >= 0.5:
                f.write("⚠️ MODERATE: CSI heartbeat detection partially matches ground truth\n")
                f.write("Some optimization of filter parameters may be needed.\n")
            else:
                f.write("❌ POOR: CSI heartbeat detection does not match ground truth\n")
                f.write("Please check sensor placement, filter settings, and alignment.\n")
            
            print(f"  Saved validation report to '{output_dir}/validation_report.txt'")

    def run_full_validation(self, csi_path, groundtruth_path, output_dir='validation',
                           low_freq=0.8, high_freq=2.5, order=3):
        """Run the complete validation pipeline"""
        print("\n" + "="*70)
        print("CSI HEARTBEAT VALIDATION PIPELINE")
        print("="*70)
        
        # Load data
        self.load_csi_data(csi_path)
        self.load_groundtruth(groundtruth_path)
        
        # Extract CSI heartbeat
        self.extract_csi_heartbeat(low_freq, high_freq, order)
        
        # Align timestamps
        self.align_timestamps()
        
        # Compute metrics
        self.compute_validation_metrics()
        
        # Create plots
        self.create_comparison_plots(output_dir)
        
        # Save report
        self.save_validation_report(output_dir)
        
        print("\n" + "="*70)
        print("VALIDATION COMPLETE!")
        print("="*70)
        print(f"Results saved to '{output_dir}/'")
        
        return self.metrics


def main():
    parser = argparse.ArgumentParser(
        description='Validate CSI heartbeat detection against ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--csi', required=True, 
                       help='CSI data file (.pcap or .csv)')
    parser.add_argument('--groundtruth', required=True,
                       help='Ground truth CSV file from heartbeat-giop.py')
    parser.add_argument('--output-dir', default='validation',
                       help='Output directory for results')
    parser.add_argument('--heart-low', type=float, default=0.8,
                       help='Lower cutoff for heartbeat filter (Hz)')
    parser.add_argument('--heart-high', type=float, default=2.5,
                       help='Upper cutoff for heartbeat filter (Hz)')
    parser.add_argument('--heart-order', type=int, default=3,
                       help='Butterworth filter order')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.csi).exists():
        print(f"Error: CSI file '{args.csi}' not found!")
        sys.exit(1)
    
    if not Path(args.groundtruth).exists():
        print(f"Error: Ground truth file '{args.groundtruth}' not found!")
        sys.exit(1)
    
    try:
        # Run validation
        validator = HeartbeatValidator()
        metrics = validator.run_full_validation(
            args.csi,
            args.groundtruth,
            args.output_dir,
            args.heart_low,
            args.heart_high,
            args.heart_order
        )
        
        # Print final summary
        print("\n" + "="*70)
        print("FINAL VALIDATION SUMMARY")
        print("="*70)
        print(f"F1-Score: {metrics.get('f1_score', 0):.3f}")
        print(f"HR MAE:   {metrics.get('heart_rate_mae', 0):.2f} BPM")
        print(f"HR Corr:  {metrics.get('heart_rate_correlation', 0):.3f}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()