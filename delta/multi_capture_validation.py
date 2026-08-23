# multi_capture_validation.py
# Test all pipelines across all captures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time

from read_csv_csi import CSIDataLoader
from windowed_psd_analysis import HeartbeatPipeline, WindowedPSDAnalyzer


class MultiCaptureValidator:
    """Test multiple pipelines across multiple CSI captures."""
    
    def __init__(self):
        self.results = []
        
    def get_heartbeat_signal(self, csi_data, method, fs, low=1.2, high=2.8, order=4):
        """Extract heartbeat using specified method."""
        pipeline = HeartbeatPipeline(fs)
        
        if method == 'amplitude':
            # Amplitude baseline
            data = pipeline.extract_amplitude(csi_data)
            channels = [data[:, sc] for sc in range(data.shape[1])]
            names = [f"amp_sc{sc}" for sc in range(data.shape[1])]
            
        elif method == 'phase':
            # Raw phase
            data = pipeline.extract_phase(csi_data, unwrap=True)
            channels = [data[:, sc] for sc in range(data.shape[1])]
            names = [f"phase_sc{sc}" for sc in range(data.shape[1])]
            
        elif method == 'phase_common':
            # Phase + common removal
            phase = pipeline.extract_phase(csi_data, unwrap=True)
            data = pipeline.remove_common_phase(phase)
            channels = [data[:, sc] for sc in range(data.shape[1])]
            names = [f"common_sc{sc}" for sc in range(data.shape[1])]
            
        elif method == 'phase_diff':
            # Phase differences
            phase = pipeline.extract_phase(csi_data, unwrap=True)
            channels = []
            names = []
            n_sc = phase.shape[1]
            # Test a subset of differences to save time
            for lag in [1, 2, 4, 8, 16, 32]:
                for i in range(0, min(50, n_sc - lag), 10):
                    j = i + lag
                    channels.append(phase[:, i] - phase[:, j])
                    names.append(f"diff_{i}_{j}")
                    
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Process all channels
        processed, quality = pipeline.process_all_channels(
            channels, names, low=low, high=high, order=order
        )
        
        # Fuse top 5 with polarity alignment
        fused, top_names = pipeline.fuse_top_channels(
            processed, quality, top_n=5, align_polarity=True
        )
        
        return fused, quality, top_names
    
    def validate_capture(self, csi_path, gt_path, method):
        """Validate a single capture with one method."""
        try:
            print(f"  Loading: {Path(csi_path).name}")
            
            # Load data
            loader = CSIDataLoader(csi_path)
            csi_data, metadata = loader.load()
            fs = metadata['rate']
            timestamps = loader.df['ts'].values if hasattr(loader, 'df') else None
            
            # Load ground truth
            gt_df = pd.read_csv(gt_path)
            
            # Get heart beat signal
            heartbeat, quality, top_names = self.get_heartbeat_signal(
                csi_data, method, fs
            )
            
            if len(heartbeat) < 100:
                print(f"    FAIL: Heartbeat signal too short")
                return None
            
            # Get BPM from PSD
            analyzer = WindowedPSDAnalyzer(fs)
            csi_bpm, power, peak_hz = analyzer.compute_bpm_from_psd(heartbeat)
            
            if csi_bpm <= 0 or csi_bpm > 200:
                print(f"    FAIL: Invalid BPM: {csi_bpm:.1f}")
                return None
            
            # Get ground truth BPM
            gt_times = gt_df['ts'].values - gt_df['ts'].values[0]
            if len(gt_times) > 3:
                gt_bpm = 60.0 / np.mean(np.diff(gt_times))
            else:
                print(f"    FAIL: Too few GT beats")
                return None
            
            # Calculate error
            error = csi_bpm - gt_bpm
            abs_error = abs(error)
            
            # Get quality metrics
            best_score = quality.iloc[0]['quality_score'] if len(quality) > 0 else 0
            top_5 = top_names[:5] if top_names else []
            
            result = {
                'capture': Path(csi_path).stem,
                'method': method,
                'fs': fs,
                'n_packets': len(csi_data),
                'n_beats': len(gt_times),
                'csi_bpm': csi_bpm,
                'gt_bpm': gt_bpm,
                'error': error,
                'abs_error': abs_error,
                'quality_score': best_score,
                'top_channels': '; '.join(top_5),
                'peak_hz': peak_hz,
                'power': power,
                'success': True
            }
            
            print(f"    OK: CSI={csi_bpm:.1f}, GT={gt_bpm:.1f}, Error={error:.1f}")
            return result
            
        except Exception as e:
            print(f"    ERROR: {e}")
            return None
    
    def run_full_test(self, captures, methods):
        """Run all methods on all captures."""
        print("\n" + "="*70)
        print("MULTI-CAPTURE PIPELINE VALIDATION")
        print("="*70)
        
        all_results = []
        
        for method in methods:
            print(f"\n--- METHOD: {method.upper()} ---")
            
            method_results = []
            
            for csi_path, gt_path in captures:
                result = self.validate_capture(csi_path, gt_path, method)
                if result:
                    method_results.append(result)
                    all_results.append(result)
            
            # Summary for this method
            if method_results:
                df = pd.DataFrame(method_results)
                print(f"\n  {method.upper()} Summary:")
                print(f"    Success rate: {len(df)}/{len(captures)}")
                print(f"    MAE: {df['abs_error'].mean():.2f} BPM")
                print(f"    RMSE: {np.sqrt((df['error']**2).mean()):.2f} BPM")
                print(f"    Correlation: {df['csi_bpm'].corr(df['gt_bpm']):.3f}")
        
        return pd.DataFrame(all_results)


def run_test():
    """Run the multi-capture validation test."""
    
    # Define your 6 captures
    captures = [
        (r"captures\Allen-N,0,0\csi_capture_20260820_133907.csv",
         r"captures\Allen-N,0,0\ground_truth_20260820_133904.csv"),
        (r"captures\Allen-N,0,1\csi_capture_20260820_133513.csv",
         r"captures\Allen-N,0,1\ground_truth_20260820_133509.csv"),
        (r"captures\Allen-N,0,-1\csi_capture_20260820_134305.csv",
         r"captures\Allen-N,0,-1\ground_truth_20260820_134301.csv"),
        (r"captures\Allen-S,0,0\csi_capture_20260820_134939.csv",
         r"captures\Allen-S,0,0\ground_truth_20260820_134936.csv"),
        (r"captures\Allen-S,0,1\csi_capture_20260820_135250.csv",
         r"captures\Allen-S,0,1\ground_truth_20260820_135247.csv"),
        (r"captures\Allen-S,0,-1\csi_capture_20260820_134631.csv",
         r"captures\Allen-S,0,-1\ground_truth_20260820_134627.csv"),
    ]
    
    # Test these 4 methods
    methods = [
        'amplitude',      # Baseline
        'phase',          # Raw phase
        'phase_common',   # Phase + common removal
        'phase_diff',     # Phase differences
    ]
    
    # Run the test
    validator = MultiCaptureValidator()
    results_df = validator.run_full_test(captures, methods)
    
    # Save results
    results_df.to_csv('multi_capture_results.csv', index=False)
    print("\nSaved: multi_capture_results.csv")
    
    # Create summary by method
    print("\n" + "="*70)
    print("FINAL SUMMARY BY METHOD")
    print("="*70)
    
    summary = results_df.groupby('method').agg({
        'abs_error': ['mean', 'std'],
        'error': lambda x: np.sqrt((x**2).mean()),  # RMSE
        'csi_bpm': lambda x: results_df[results_df['method'] == x.name]['csi_bpm'].corr(
                              results_df[results_df['method'] == x.name]['gt_bpm']),
        'capture': 'count'
    }).round(3)
    
    summary.columns = ['MAE', 'MAE_std', 'RMSE', 'Correlation', 'Count']
    summary = summary.sort_values('MAE')
    print(summary.to_string())
    
    # Best method
    best = summary.index[0]
    print(f"\nBEST METHOD: {best.upper()}")
    print(f"  MAE: {summary.loc[best, 'MAE']:.2f} BPM")
    print(f"  RMSE: {summary.loc[best, 'RMSE']:.2f} BPM")
    print(f"  Correlation: {summary.loc[best, 'Correlation']:.3f}")
    
    # Create comparison plot
    create_comparison_plot(results_df, summary)
    
    return results_df, summary


def create_comparison_plot(results_df, summary):
    """Create comparison plot across methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. MAE by method
    ax1 = axes[0, 0]
    methods = summary.index
    mae = summary['MAE'].values
    colors = ['green' if m == methods[0] else 'blue' for m in methods]
    ax1.bar(methods, mae, color=colors)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('MAE (BPM)')
    ax1.set_title('Mean Absolute Error by Method')
    ax1.grid(True, alpha=0.3)
    
    # 2. Correlation by method
    ax2 = axes[0, 1]
    corr = summary['Correlation'].values
    colors = ['green' if m == methods[0] else 'blue' for m in methods]
    ax2.bar(methods, corr, color=colors)
    ax2.axhline(y=0.5, color='red', linestyle='--', label='Target')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Correlation by Method')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot for each method
    ax3 = axes[1, 0]
    for method in methods:
        df = results_df[results_df['method'] == method]
        ax3.scatter(df['gt_bpm'], df['csi_bpm'], label=method, alpha=0.7, s=50)
    ax3.plot([40, 120], [40, 120], 'k--', alpha=0.5)
    ax3.set_xlabel('Ground Truth BPM')
    ax3.set_ylabel('CSI BPM')
    ax3.set_title('All Methods: CSI vs Ground Truth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution by method
    ax4 = axes[1, 1]
    for method in methods:
        df = results_df[results_df['method'] == method]
        ax4.hist(df['error'], bins=15, alpha=0.5, label=method)
    ax4.axvline(x=0, color='black', linestyle='-')
    ax4.set_xlabel('Error (BPM)')
    ax4.set_ylabel('Count')
    ax4.set_title('Error Distribution by Method')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Capture Pipeline Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('multi_capture_comparison.png', dpi=300)
    print("\nSaved: multi_capture_comparison.png")
    plt.close()


if __name__ == "__main__":
    # Run the full test
    results, summary = run_test()
    
    # Print individual results
    print("\n" + "="*70)
    print("INDIVIDUAL RESULTS")
    print("="*70)
    print(results_df[['capture', 'method', 'csi_bpm', 'gt_bpm', 'abs_error']].to_string(index=False))