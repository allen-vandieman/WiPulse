# psd_subcarrier_analysis.py - Fixed path handling
# Usage: python psd_subcarrier_analysis.py --csi captures/Alicia/csi_capture_20260818_193158.csv --groundtruth captures/Alicia/ground_truth_20260818_193154.csv

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import CSI modules
from read_csv_csi import CSIDataLoader
from read_pcap import PCAPDataLoader
from filters import CSIFilter, FilterConfig
from heartbeat_extraction import HeartbeatExtractor, HeartbeatConfig


def resolve_path(path_str):
    """Resolve path - handles both Windows and Unix style paths"""
    path = Path(path_str)
    
    # If path doesn't exist, try with forward slashes
    if not path.exists():
        # Try the original string with forward slashes
        alt_path = Path(path_str.replace('\\', '/'))
        if alt_path.exists():
            return alt_path
        # Try with the current directory
        alt_path = Path(os.getcwd()) / path_str.replace('\\', '/')
        if alt_path.exists():
            return alt_path
        # Try removing any leading './' or '.\'
        clean_path = path_str.replace('./', '').replace('.\\', '')
        alt_path = Path(clean_path)
        if alt_path.exists():
            return alt_path
        alt_path = Path(os.getcwd()) / clean_path
        if alt_path.exists():
            return alt_path
    
    return path


class PSDSubcarrierAnalyzer:
    """PSD-based subcarrier analysis and selection"""
    
    def __init__(self, csi_data, metadata):
        self.csi_data = csi_data
        self.metadata = metadata
        self.fs = metadata['rate']
        self.n_packets, self.n_subcarriers = csi_data.shape
        
        # Storage for analysis results
        self.psd_results = None
        self.subcarrier_metrics = None
        self.selected_subcarriers = None
        self.weights = None
        
    def compute_psd(self, window_size=None, overlap=0.5):
        """Compute PSD for each subcarrier"""
        print("\n" + "="*70)
        print("PHASE 1: PSD SUBSCARRIER ANALYSIS")
        print("="*70)
        
        if window_size is None:
            window_size = int(self.fs * 5)
            if window_size > self.n_packets:
                window_size = self.n_packets // 2
            if window_size < 50:
                window_size = min(50, self.n_packets)
        
        print(f"Window size: {window_size} samples ({window_size/self.fs:.1f}s)")
        print(f"Overlap: {overlap*100:.0f}%")
        
        n_freq = window_size // 2 + 1
        psd_matrix = np.zeros((self.n_subcarriers, n_freq))
        
        magnitudes = np.abs(self.csi_data)
        
        for sc in range(self.n_subcarriers):
            try:
                freqs, psd = signal.welch(
                    magnitudes[:, sc],
                    fs=self.fs,
                    window='hann',
                    nperseg=min(window_size, len(magnitudes)),
                    noverlap=int(window_size * overlap),
                    scaling='density'
                )
                if len(psd) < n_freq:
                    psd_matrix[sc, :len(psd)] = psd
                else:
                    psd_matrix[sc, :] = psd[:n_freq]
            except:
                continue
        
        self.frequencies = freqs[:n_freq] if 'freqs' in locals() else np.arange(n_freq) / self.fs
        self.psd_matrix = psd_matrix
        
        self._compute_subcarrier_metrics(psd_matrix)
        self._plot_psd_analysis()
        
        return psd_matrix, self.frequencies, self.subcarrier_metrics
    
    def _compute_subcarrier_metrics(self, psd_matrix):
        """Compute quality metrics for each subcarrier"""
        metrics = []
        
        # Heartbeat frequency range
        hr_mask = (self.frequencies >= 0.5) & (self.frequencies <= 3.5)
        resp_mask = (self.frequencies >= 0.15) & (self.frequencies <= 0.4)
        total_mask = (self.frequencies >= 0) & (self.frequencies <= 5)
        
        for sc in range(self.n_subcarriers):
            psd = psd_matrix[sc, :]
            
            hr_power = np.sum(psd[hr_mask]) if np.any(hr_mask) else 0
            resp_power = np.sum(psd[resp_mask]) if np.any(resp_mask) else 0
            total_power = np.sum(psd[total_mask]) if np.any(total_mask) else 0
            
            if np.sum(psd[hr_mask]) > 0 and np.any(hr_mask):
                hr_peak_idx = np.argmax(psd[hr_mask])
                hr_peak_freq = self.frequencies[hr_mask][hr_peak_idx] if np.any(hr_mask) else 0
                hr_peak_power = psd[hr_mask][hr_peak_idx] if np.any(hr_mask) else 0
            else:
                hr_peak_freq = 0
                hr_peak_power = 0
            
            noise_floor = np.percentile(psd[total_mask], 25) if np.any(total_mask) else 1e-10
            snr = hr_power / (noise_floor + 1e-10)
            hr_resp_ratio = hr_power / (resp_power + 1e-10)
            
            metrics.append({
                'sc': sc,
                'hr_power': hr_power,
                'resp_power': resp_power,
                'total_power': total_power,
                'hr_peak_freq': hr_peak_freq,
                'hr_peak_power': hr_peak_power,
                'snr': snr,
                'hr_resp_ratio': hr_resp_ratio,
                'mean_psd': np.mean(psd[total_mask]) if np.any(total_mask) else 0,
                'std_psd': np.std(psd[total_mask]) if np.any(total_mask) else 0
            })
        
        self.subcarrier_metrics = pd.DataFrame(metrics)
        
        # Normalize and combine metrics
        max_hr_power = self.subcarrier_metrics['hr_power'].max() or 1
        max_snr = self.subcarrier_metrics['snr'].max() or 1
        max_hr_resp = self.subcarrier_metrics['hr_resp_ratio'].max() or 1
        
        self.subcarrier_metrics['norm_hr_power'] = self.subcarrier_metrics['hr_power'] / max_hr_power
        self.subcarrier_metrics['norm_snr'] = self.subcarrier_metrics['snr'] / max_snr
        self.subcarrier_metrics['norm_hr_resp'] = self.subcarrier_metrics['hr_resp_ratio'] / max_hr_resp
        
        self.subcarrier_metrics['quality_score'] = (
            0.4 * self.subcarrier_metrics['norm_hr_power'] +
            0.4 * self.subcarrier_metrics['norm_snr'] +
            0.2 * self.subcarrier_metrics['norm_hr_resp']
        )
        
        self.subcarrier_metrics = self.subcarrier_metrics.sort_values('quality_score', ascending=False)
        
        print(f"\nSubcarrier Metrics Summary:")
        print(f"  Top 10 subcarriers by quality score:")
        for idx, row in self.subcarrier_metrics.head(10).iterrows():
            print(f"    SC {int(row['sc']):3d}: score={row['quality_score']:.3f}, "
                  f"HR power={row['hr_power']:.2e}, SNR={row['snr']:.1f}")
        
        return self.subcarrier_metrics
    
    def select_subcarriers_psd(self, n_select=None, method='quality_score'):
        """Select best subcarriers using PSD metrics"""
        if n_select is None:
            n_select = min(10, self.n_subcarriers // 4)
        
        if method == 'quality_score':
            selected = self.subcarrier_metrics.head(n_select)['sc'].values.astype(int)
        elif method == 'snr':
            selected = self.subcarrier_metrics.nlargest(n_select, 'snr')['sc'].values.astype(int)
        elif method == 'hr_power':
            selected = self.subcarrier_metrics.nlargest(n_select, 'hr_power')['sc'].values.astype(int)
        elif method == 'hr_resp_ratio':
            selected = self.subcarrier_metrics.nlargest(n_select, 'hr_resp_ratio')['sc'].values.astype(int)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        self.selected_subcarriers = selected
        print(f"\nSelected {len(selected)} subcarriers using '{method}' method:")
        print(f"  {selected}")
        
        return selected
    
    def compute_weighted_average(self, csi_data, method='psd'):
        """Compute weighted average across subcarriers"""
        print("\n" + "="*70)
        print("PHASE 2: WEIGHTED SUBCARRIER AVERAGING")
        print("="*70)
        
        magnitudes = np.abs(csi_data)
        
        if method == 'psd':
            weights = self.subcarrier_metrics['hr_power'].values
            weights = weights / (np.sum(weights) + 1e-10)
            print("  Weighting by PSD power in heartbeat band")
        elif method == 'snr':
            weights = self.subcarrier_metrics['snr'].values
            weights = weights / (np.sum(weights) + 1e-10)
            print("  Weighting by SNR")
        elif method == 'quality_score':
            weights = self.subcarrier_metrics['quality_score'].values
            weights = weights / (np.sum(weights) + 1e-10)
            print("  Weighting by quality score")
        elif method == 'uniform':
            weights = np.ones(self.n_subcarriers) / self.n_subcarriers
            print("  Using uniform weights (simple average)")
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        self.weights = weights
        
        weighted_signal = np.zeros(self.n_packets)
        for sc in range(self.n_subcarriers):
            weighted_signal += weights[sc] * magnitudes[:, sc]
        
        print(f"  Top 5 weighted subcarriers:")
        top_weights = np.argsort(weights)[-5:][::-1]
        for sc in top_weights:
            print(f"    SC {sc:3d}: weight={weights[sc]:.4f}")
        
        return weighted_signal, weights
    
    def _plot_psd_analysis(self):
        """Create PSD analysis visualizations"""
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. PSD Heatmap
        ax1 = fig.add_subplot(gs[0, :])
        top_n = min(30, self.n_subcarriers)
        top_scs = self.subcarrier_metrics.head(top_n)['sc'].values.astype(int)
        psd_subset = self.psd_matrix[top_scs, :]
        
        freq_mask = self.frequencies <= 5
        if np.any(freq_mask):
            freq_plot = self.frequencies[freq_mask]
            psd_plot = psd_subset[:, freq_mask]
            im1 = ax1.imshow(psd_plot, aspect='auto', origin='lower',
                            extent=[freq_plot[0], freq_plot[-1], 0, len(top_scs)],
                            cmap='viridis')
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Subcarrier Rank')
            ax1.set_title(f'PSD Heatmap (Top {top_n} Subcarriers)')
            plt.colorbar(im1, ax=ax1, label='PSD')
        
        # 2. Quality score distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.subcarrier_metrics['quality_score'], bins=20, alpha=0.7, color='blue')
        ax2.axvline(x=self.subcarrier_metrics['quality_score'].quantile(0.8), 
                   color='red', linestyle='--', label='80th percentile')
        ax2.set_xlabel('Quality Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Subcarrier Quality Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. HR Power vs SNR
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(self.subcarrier_metrics['snr'].clip(0, 100),
                   self.subcarrier_metrics['hr_power'],
                   c=self.subcarrier_metrics['quality_score'],
                   cmap='viridis', alpha=0.6, s=15)
        top10 = self.subcarrier_metrics.head(10)
        ax3.scatter(top10['snr'].clip(0, 100), top10['hr_power'], 
                   c='red', s=50, marker='*', label='Top 10')
        ax3.set_xlabel('SNR')
        ax3.set_ylabel('HR Band Power')
        ax3.set_title('HR Power vs SNR')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Top subcarrier spectra
        ax5 = fig.add_subplot(gs[2, :])
        for idx, row in self.subcarrier_metrics.head(5).iterrows():
            sc = int(row['sc'])
            psd = self.psd_matrix[sc, :]
            freq_mask = self.frequencies <= 5
            if np.any(freq_mask):
                ax5.plot(self.frequencies[freq_mask], psd[freq_mask], 
                        label=f'SC {sc} (score={row["quality_score"]:.3f})', alpha=0.8)
        
        ax5.axvspan(0.8, 2.5, alpha=0.2, color='red', label='Heartbeat band')
        ax5.axvspan(0.15, 0.4, alpha=0.2, color='blue', label='Respiratory band')
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('PSD')
        ax5.set_title('Top 5 Subcarrier PSDs')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 5)
        
        # 5. Subcarrier weights
        ax6 = fig.add_subplot(gs[3, 0])
        if self.weights is not None:
            ax6.bar(range(self.n_subcarriers), self.weights, alpha=0.7)
            ax6.set_xlabel('Subcarrier Index')
            ax6.set_ylabel('Weight')
            ax6.set_title('Subcarrier Weights')
            ax6.grid(True, alpha=0.3)
        
        # 6. Cumulative weight distribution
        ax7 = fig.add_subplot(gs[3, 1])
        if self.weights is not None:
            sorted_weights = np.sort(self.weights)[::-1]
            cumulative = np.cumsum(sorted_weights)
            ax7.plot(range(1, len(cumulative)+1), cumulative, 'b-', linewidth=2)
            ax7.axhline(y=0.5, color='r', linestyle='--', label='50%')
            ax7.axhline(y=0.8, color='g', linestyle='--', label='80%')
            ax7.set_xlabel('Number of Subcarriers')
            ax7.set_ylabel('Cumulative Weight')
            ax7.set_title('Cumulative Weight Distribution')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 7. Summary
        ax8 = fig.add_subplot(gs[3, 2])
        ax8.axis('off')
        
        top5_str = self.subcarrier_metrics.head(5)[['sc', 'quality_score', 'snr']].to_string(
            index=False, float_format='%.3f', col_space=[10, 15, 10])
        
        summary_text = f"""PSD ANALYSIS SUMMARY

    Subcarriers:        {self.n_subcarriers}
    Packets:            {self.n_packets}
    Sampling Rate:      {self.fs:.1f} Hz

    Top 5 Subcarriers:
    {top5_str}

    Quality Score:
    Mean:  {self.subcarrier_metrics['quality_score'].mean():.3f}
    Std:   {self.subcarrier_metrics['quality_score'].std():.3f}
    Top 10%: {self.subcarrier_metrics['quality_score'].quantile(0.9):.3f}
    """
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('PSD Subcarrier Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig('psd_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSaved PSD analysis plot to 'psd_analysis.png'")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='PSD-based subcarrier analysis')
    parser.add_argument('--csi', required=True, help='CSI data file (.pcap or .csv)')
    parser.add_argument('--groundtruth', help='Ground truth CSV file (optional)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only run PSD analysis, no weighted averaging')
    parser.add_argument('--output-dir', default='psd_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Resolve paths
    csi_path = resolve_path(args.csi)
    if not csi_path.exists():
        print(f"Error: CSI file '{args.csi}' not found!")
        print(f"Tried: {csi_path}")
        sys.exit(1)
    
    gt_path = None
    if args.groundtruth:
        gt_path = resolve_path(args.groundtruth)
        if not gt_path.exists():
            print(f"Warning: Ground truth file '{args.groundtruth}' not found!")
            gt_path = None
    
    try:
        # Create output directory
        Path(args.output_dir).mkdir(exist_ok=True)
        
        # Load CSI data
        print(f"\nLoading CSI data from: {csi_path}")
        if csi_path.suffix.lower() == '.csv':
            loader = CSIDataLoader(str(csi_path))
            csi_data, metadata = loader.load()
        else:
            loader = PCAPDataLoader(str(csi_path))
            csi_data, metadata = loader.load()
        
        print(f"  Loaded {metadata['n_packets']} packets at {metadata['rate']:.1f} Hz")
        
        # Load ground truth if available
        gt_df = None
        if gt_path and gt_path.exists():
            gt_df = pd.read_csv(str(gt_path))
            print(f"  Loaded ground truth: {len(gt_df)} beats")
        
        # Initialize analyzer
        analyzer = PSDSubcarrierAnalyzer(csi_data, metadata)
        
        # Phase 1: PSD Analysis
        psd_matrix, freqs, metrics = analyzer.compute_psd()
        
        # Select subcarriers
        selected = analyzer.select_subcarriers_psd(n_select=10, method='quality_score')
        
        # Phase 2: Weighted averaging
        if not args.analyze_only:
            weighted_signals = {}
            methods = ['psd', 'snr', 'quality_score']
            
            for method in methods:
                signal, weights = analyzer.compute_weighted_average(csi_data, method)
                weighted_signals[method] = signal
            
            # Also compute uniform average
            uniform_signal = np.mean(np.abs(csi_data), axis=1)
            weighted_signals['uniform'] = uniform_signal
            
            print("\n" + "="*70)
            print("WEIGHTED AVERAGING COMPLETE")
            print("="*70)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"Results saved to '{args.output_dir}/'")
        print("  - psd_analysis.png: PSD analysis dashboard")
        
        if selected is not None:
            print(f"\nRecommended subcarriers: {selected}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()