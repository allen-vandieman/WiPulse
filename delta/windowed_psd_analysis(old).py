# windowed_psd_analysis.py
# Test if CSI can track heart rate changes over time
# Usage
# python windowed_psd_analysis.py
# uses pre set CSI and ground truth files in captures folder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

from read_csv_csi import CSIDataLoader
from filters import CSIFilter, FilterConfig


class WindowedPSDAnalyzer:
    """
    Windowed PSD analysis to track heart rate over time.
    Compares CSI-derived BPM against ground truth BPM per window.
    """
    
    def __init__(self, fs=100.0):
        self.fs = fs
        self.results = None
        
    def extract_heartbeat_signal(self, csi_data, method='phase_diff', 
                                  low=1.2, high=2.8, order=4):
        """Extract best heartbeat signal using specified method"""
        
        if method == 'amplitude':
            # Amplitude baseline
            signal_data = np.abs(csi_data)
            # Find best subcarriers by variance
            variances = np.var(signal_data, axis=0)
            top_n = np.argsort(variances)[-5:]
            heartbeat = np.mean(signal_data[:, top_n], axis=1)
            
        elif method == 'phase':
            # Phase method
            phase = np.angle(csi_data)
            for sc in range(phase.shape[1]):
                phase[:, sc] = np.unwrap(phase[:, sc])
            # Remove common phase
            common = np.median(phase, axis=1)
            phase_removed = phase - common[:, np.newaxis]
            # Detrend
            from scipy.signal import detrend
            phase_detrended = detrend(phase_removed, axis=0)
            # Bandpass
            nyquist = self.fs / 2
            sos = signal.butter(order, [low/nyquist, high/nyquist], btype='band', output='sos')
            phase_filtered = np.zeros_like(phase_detrended)
            for sc in range(phase_detrended.shape[1]):
                phase_filtered[:, sc] = signal.sosfiltfilt(sos, phase_detrended[:, sc])
            # Find best subcarrier by PSD peak
            best_sc = 0
            best_score = 0
            for sc in range(phase_filtered.shape[1]):
                freqs, psd = signal.welch(phase_filtered[:, sc], fs=self.fs, nperseg=min(128, len(phase_filtered)))
                hr_mask = (freqs >= 0.8) & (freqs <= 3.5)
                if np.any(hr_mask):
                    peak = np.max(psd[hr_mask])
                    if peak > best_score:
                        best_score = peak
                        best_sc = sc
            heartbeat = phase_filtered[:, best_sc]
            
        elif method == 'phase_diff':
            # Phase differences
            phase = np.angle(csi_data)
            for sc in range(phase.shape[1]):
                phase[:, sc] = np.unwrap(phase[:, sc])
            
            # Test a few phase differences
            best_diff = None
            best_score = 0
            for lag in [1, 2, 3, 5, 10]:
                for i in range(0, min(50, phase.shape[1] - lag), 10):
                    diff = phase[:, i] - phase[:, i + lag]
                    from scipy.signal import detrend
                    diff_detrended = detrend(diff)
                    nyquist = self.fs / 2
                    sos = signal.butter(order, [low/nyquist, high/nyquist], btype='band', output='sos')
                    diff_filtered = signal.sosfiltfilt(sos, diff_detrended)
                    freqs, psd = signal.welch(diff_filtered, fs=self.fs, nperseg=min(128, len(diff_filtered)))
                    hr_mask = (freqs >= 0.8) & (freqs <= 3.5)
                    if np.any(hr_mask):
                        peak = np.max(psd[hr_mask])
                        if peak > best_score:
                            best_score = peak
                            best_diff = diff_filtered
            heartbeat = best_diff if best_diff is not None else np.zeros(len(csi_data))
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize
        if np.std(heartbeat) > 0:
            heartbeat = (heartbeat - np.mean(heartbeat)) / np.std(heartbeat)
        
        return heartbeat
    
    def compute_bpm_from_psd(self, signal_data, fs=None):
        """Compute BPM from PSD peak in heartbeat band"""
        if fs is None:
            fs = self.fs
        
        freqs, psd = signal.welch(
            signal_data - np.mean(signal_data),
            fs=fs,
            nperseg=min(128, len(signal_data)),
            scaling='density'
        )
        
        # Heartbeat band: 0.8 - 3.5 Hz (48 - 210 BPM)
        hr_mask = (freqs >= 0.8) & (freqs <= 3.5)
        if not np.any(hr_mask):
            return 0, 0
        
        peak_idx = np.argmax(psd[hr_mask])
        peak_freq = freqs[hr_mask][peak_idx]
        bpm = peak_freq * 60.0
        peak_power = psd[hr_mask][peak_idx]
        
        return bpm, peak_power
    
    def compute_ground_truth_bpm(self, gt_df, window_start, window_end):
        """Compute ground truth BPM within a time window"""
        # Get ground truth beats in window
        if 'ts' in gt_df.columns:
            gt_times = gt_df['ts'].values - gt_df['ts'].values[0]
        else:
            gt_times = np.arange(len(gt_df))
        
        # Find beats in window
        mask = (gt_times >= window_start) & (gt_times <= window_end)
        beats_in_window = gt_times[mask]
        
        if len(beats_in_window) < 3:
            return 0, 0
        
        # Compute average BPM from intervals
        intervals = np.diff(beats_in_window)
        bpm = 60.0 / np.mean(intervals)
        
        return bpm, len(beats_in_window)
    
    def analyze_windows(self, heartbeat_signal, gt_df, csi_timestamps,
                        window_seconds=20, step_seconds=5,
                        min_beats=3):
        """
        Analyze heart rate in sliding windows.
        
        Args:
            heartbeat_signal: extracted heartbeat waveform
            gt_df: ground truth DataFrame
            csi_timestamps: actual CSI timestamps
            window_seconds: window duration in seconds
            step_seconds: step size between windows
            min_beats: minimum beats required for ground truth BPM
        
        Returns:
            DataFrame with per-window results
        """
        total_duration = len(heartbeat_signal) / self.fs
        
        results = []
        window_starts = np.arange(0, total_duration - window_seconds, step_seconds)
        
        for start in window_starts:
            end = start + window_seconds
            
            # Get CSI indices for this window
            start_idx = int(start * self.fs)
            end_idx = int(end * self.fs)
            
            if end_idx > len(heartbeat_signal):
                break
            
            window_signal = heartbeat_signal[start_idx:end_idx]
            
            # CSI BPM from PSD
            csi_bpm, csi_power = self.compute_bpm_from_psd(window_signal)
            
            # Ground truth BPM
            gt_bpm, n_beats = self.compute_ground_truth_bpm(gt_df, start, end)
            
            if n_beats >= min_beats and csi_bpm > 0:
                results.append({
                    'window_start': start,
                    'window_end': end,
                    'csi_bpm': csi_bpm,
                    'gt_bpm': gt_bpm,
                    'csi_power': csi_power,
                    'n_beats': n_beats,
                    'bpm_error': csi_bpm - gt_bpm,
                    'bpm_abs_error': abs(csi_bpm - gt_bpm)
                })
        
        self.results = pd.DataFrame(results)
        
        if len(self.results) > 0:
            self.results['bpm_error_pct'] = (self.results['bpm_error'] / self.results['gt_bpm']) * 100
        
        return self.results
    
    def print_metrics(self):
        """Print summary metrics"""
        if self.results is None or len(self.results) == 0:
            print("No results available")
            return
        
        metrics = {
            'MAE (BPM)': np.mean(self.results['bpm_abs_error']),
            'RMSE (BPM)': np.sqrt(np.mean(self.results['bpm_error']**2)),
            'Correlation': np.corrcoef(self.results['csi_bpm'], self.results['gt_bpm'])[0, 1],
            'Mean Error (BPM)': np.mean(self.results['bpm_error']),
            'Std Error (BPM)': np.std(self.results['bpm_error']),
            'Number of Windows': len(self.results)
        }
        
        print("\n" + "="*60)
        print("WINDOWED PSD METRICS")
        print("="*60)
        for key, value in metrics.items():
            print(f"  {key:20s}: {value:.3f}" if isinstance(value, float) else f"  {key:20s}: {value}")
        
        return metrics
    
    def plot_results(self, output_file='windowed_psd_results.png'):
        """Create visualization of windowed results"""
        if self.results is None or len(self.results) == 0:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. BPM tracking over time
        ax1 = axes[0]
        ax1.plot(self.results['window_start'], self.results['csi_bpm'], 
                'b-', linewidth=2, label='CSI BPM (PSD)')
        ax1.plot(self.results['window_start'], self.results['gt_bpm'], 
                'r-', linewidth=2, label='Ground Truth BPM')
        ax1.fill_between(self.results['window_start'], 
                        self.results['csi_bpm'] - self.results['bpm_abs_error'],
                        self.results['csi_bpm'] + self.results['bpm_abs_error'],
                        alpha=0.2, color='blue')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Heart Rate (BPM)')
        ax1.set_title('Heart Rate Tracking Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Error over time
        ax2 = axes[1]
        ax2.bar(self.results['window_start'], self.results['bpm_error'], 
                width=4, alpha=0.7, color='purple')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=np.mean(self.results['bpm_abs_error']), color='red', 
                   linestyle='--', label=f'Mean Error: {np.mean(self.results["bpm_abs_error"]):.1f} BPM')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('BPM Error (CSI - GT)')
        ax2.set_title('BPM Error Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Correlation scatter plot
        ax3 = axes[2]
        ax3.scatter(self.results['gt_bpm'], self.results['csi_bpm'], 
                   alpha=0.6, s=50, c=self.results['window_start'], cmap='viridis')
        
        # Add diagonal line
        min_bpm = min(self.results['gt_bpm'].min(), self.results['csi_bpm'].min()) - 5
        max_bpm = max(self.results['gt_bpm'].max(), self.results['csi_bpm'].max()) + 5
        ax3.plot([min_bpm, max_bpm], [min_bpm, max_bpm], 'k--', alpha=0.5, label='Perfect')
        
        # Add correlation text
        corr = np.corrcoef(self.results['csi_bpm'], self.results['gt_bpm'])[0, 1]
        ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax3.set_xlabel('Ground Truth BPM')
        ax3.set_ylabel('CSI BPM')
        ax3.set_title('CSI vs Ground Truth BPM')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Windowed PSD Heart Rate Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved plot to '{output_file}'")
        plt.close()


def main():
    """Run windowed PSD analysis"""
    csi_path = r"captures\Allen-S,0,1\csi_capture_20260820_135250.csv"
    gt_path = r"captures\Allen-S,0,1\ground_truth_20260820_135247.csv"
    
    print("\n" + "="*70)
    print("WINDOWED PSD HEART RATE ANALYSIS")
    print("="*70)
    
    # Load data
    print(f"\nLoading CSI from: {csi_path}")
    loader = CSIDataLoader(csi_path)
    csi_data, metadata = loader.load()
    fs = metadata['rate']
    print(f"  {metadata['n_packets']} packets at {fs:.1f} Hz")
    
    # Get actual timestamps
    if hasattr(loader, 'df') and loader.df is not None:
        if 'ts' in loader.df.columns:
            csi_timestamps = loader.df['ts'].values
        else:
            csi_timestamps = np.arange(len(csi_data)) / fs
    else:
        csi_timestamps = np.arange(len(csi_data)) / fs
    
    # Load ground truth
    print(f"Loading ground truth: {gt_path}")
    gt_df = pd.read_csv(gt_path)
    print(f"  {len(gt_df)} beats")
    
    # Initialize analyzer
    analyzer = WindowedPSDAnalyzer(fs)
    
    # Test different methods
    methods = [
        ('Amplitude', 'amplitude'),
        ('Phase', 'phase'),
        ('Phase Difference', 'phase_diff')
    ]
    
    all_results = {}
    
    for name, method in methods:
        print(f"\n" + "-"*50)
        print(f"Testing: {name}")
        print("-"*50)
        
        # Extract heartbeat
        heartbeat = analyzer.extract_heartbeat_signal(
            csi_data, method=method, low=1.2, high=2.8, order=4
        )
        
        # Analyze windows
        results = analyzer.analyze_windows(
            heartbeat, gt_df, csi_timestamps,
            window_seconds=20, step_seconds=5, min_beats=3
        )
        
        # Print metrics
        metrics = analyzer.print_metrics()
        all_results[name] = {'metrics': metrics, 'results': results}
    
    # Compare methods
    print("\n" + "="*70)
    print("METHOD COMPARISON")
    print("="*70)
    
    comparison = []
    for name, data in all_results.items():
        if data['metrics']:
            comparison.append({
                'Method': name,
                'MAE (BPM)': data['metrics']['MAE (BPM)'],
                'RMSE (BPM)': data['metrics']['RMSE (BPM)'],
                'Correlation': data['metrics']['Correlation'],
                'Windows': data['metrics']['Number of Windows']
            })
    
    if comparison:
        df_comp = pd.DataFrame(comparison)
        print(df_comp.to_string(index=False))
        
        best = df_comp.loc[df_comp['MAE (BPM)'].idxmin()]
        print(f"\nBEST METHOD: {best['Method']}")
        print(f"  MAE: {best['MAE (BPM)']:.1f} BPM")
        print(f"  Correlation: {best['Correlation']:.3f}")
    
    # Plot best method
    best_name = df_comp.loc[df_comp['MAE (BPM)'].idxmin()]['Method']
    best_results = all_results[best_name]['results']
    
    if best_results is not None and len(best_results) > 0:
        # Use the analyzer to plot
        analyzer.results = best_results
        analyzer.plot_results(f'windowed_psd_{best_name.lower().replace(" ", "_")}.png')


if __name__ == "__main__":
    main()