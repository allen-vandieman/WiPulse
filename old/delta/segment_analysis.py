# segment_analysis.py
# Split CSI into continuous segments and analyze each separately

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from read_csv_csi import CSIDataLoader


def find_continuous_segments(timestamps, max_gap_ms=20):
    """
    Split timestamps into continuous segments where gaps are small.
    
    Args:
        timestamps: array of packet timestamps
        max_gap_ms: maximum allowed gap between packets (ms)
    
    Returns:
        segments: list of (start_idx, end_idx) tuples
        gap_indices: indices where gaps exceed threshold
    """
    dt = np.diff(timestamps) * 1000.0  # Convert to ms
    gap_indices = np.where(dt > max_gap_ms)[0]
    
    segments = []
    start_idx = 0
    
    for gap_idx in gap_indices:
        end_idx = gap_idx + 1
        if end_idx - start_idx > 10:  # Minimum segment length
            segments.append((start_idx, end_idx))
        start_idx = end_idx
    
    # Last segment
    if len(timestamps) - start_idx > 10:
        segments.append((start_idx, len(timestamps)))
    
    return segments, gap_indices


def analyze_segment(csi_segment, timestamps_segment, gt_df, fs_actual):
    """
    Analyze a single continuous segment.
    
    Returns:
        dict with segment metrics
    """
    from windowed_psd_analysis import HeartbeatPipeline, WindowedPSDAnalyzer
    
    # Use the same pipeline as before
    pipeline = HeartbeatPipeline(fs_actual)
    
    # Extract phase with common removal (best method)
    phase = pipeline.extract_phase(csi_segment, unwrap=True)
    phase_common = pipeline.remove_common_phase(phase)
    
    # Process channels
    channels = [phase_common[:, sc] for sc in range(phase_common.shape[1])]
    names = [f"sc{sc}" for sc in range(phase_common.shape[1])]
    
    processed, quality = pipeline.process_all_channels(
        channels, names, low=1.2, high=2.8, order=4
    )
    
    # Fuse top 5
    fused, top_names = pipeline.fuse_top_channels(processed, quality, top_n=5)
    
    # Get BPM from PSD
    analyzer = WindowedPSDAnalyzer(fs_actual)
    bpm, power, peak_hz = analyzer.compute_bpm_from_psd(fused)
    
    # Get ground truth BPM for this segment
    segment_start = timestamps_segment[0]
    segment_end = timestamps_segment[-1]
    
    gt_times = gt_df['ts'].values - gt_df['ts'].values[0]
    gt_beats = gt_times[(gt_times >= segment_start) & (gt_times <= segment_end)]
    
    if len(gt_beats) > 3:
        gt_bpm = 60.0 / np.mean(np.diff(gt_beats))
    else:
        gt_bpm = 0
    
    return {
        'segment_start': segment_start,
        'segment_end': segment_end,
        'duration': segment_end - segment_start,
        'n_packets': len(csi_segment),
        'n_beats': len(gt_beats),
        'csi_bpm': bpm,
        'gt_bpm': gt_bpm,
        'error': bpm - gt_bpm,
        'abs_error': abs(bpm - gt_bpm),
        'top_channels': top_names,
        'quality_score': quality.iloc[0]['quality_score'] if len(quality) > 0 else 0
    }


def segment_analysis(csi_path, gt_path, max_gap_ms=20):
    """
    Main segment-based analysis.
    """
    print("\n" + "="*70)
    print("SEGMENT-BASED HEART RATE ANALYSIS")
    print("="*70)
    
    # Load data
    print(f"\nLoading CSI: {csi_path}")
    loader = CSIDataLoader(csi_path)
    csi_data, metadata = loader.load()
    
    timestamps = loader.df['ts'].values if hasattr(loader, 'df') else None
    
    print(f"  {len(csi_data)} packets at {metadata['rate']:.1f} Hz")
    
    # Load ground truth
    print(f"Loading ground truth: {gt_path}")
    gt_df = pd.read_csv(gt_path)
    print(f"  {len(gt_df)} beats")
    
    # Align timestamps (use same origin)
    origin = timestamps[0]
    csi_time = timestamps - origin
    gt_time = gt_df['ts'].values - origin
    
    # Find continuous segments
    print(f"\nFinding continuous segments (max gap: {max_gap_ms} ms)...")
    segments, gap_indices = find_continuous_segments(csi_time, max_gap_ms)
    
    print(f"  Found {len(segments)} segments")
    print(f"  {len(gap_indices)} gaps detected")
    
    # Analyze each segment
    results = []
    
    for idx, (start, end) in enumerate(segments):
        print(f"\nSegment {idx+1}: packets {start}-{end} ({end-start} packets)")
        
        csi_segment = csi_data[start:end]
        timestamps_segment = csi_time[start:end]
        
        # Calculate actual sampling rate for this segment
        actual_fs = len(csi_segment) / (timestamps_segment[-1] - timestamps_segment[0])
        
        result = analyze_segment(csi_segment, timestamps_segment, gt_df, actual_fs)
        
        if result['gt_bpm'] > 0 and result['csi_bpm'] > 0:
            print(f"  CSI BPM: {result['csi_bpm']:.1f}, GT BPM: {result['gt_bpm']:.1f}")
            print(f"  Error: {result['error']:.1f} BPM")
            results.append(result)
        else:
            print(f"  Skipped (insufficient data)")
    
    # Summary
    if results:
        df = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("SEGMENT ANALYSIS SUMMARY")
        print("="*70)
        
        metrics = {
            'MAE (BPM)': df['abs_error'].mean(),
            'RMSE (BPM)': np.sqrt((df['error']**2).mean()),
            'Correlation': df['csi_bpm'].corr(df['gt_bpm']),
            'Mean Error (BPM)': df['error'].mean(),
            'Segments': len(df),
            'Total Duration (s)': df['duration'].sum()
        }
        
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.3f}")
            else:
                print(f"  {key:20s}: {value}")
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # BPM tracking
        ax = axes[0]
        ax.plot(df['segment_start'], df['csi_bpm'], 'bo-', label='CSI BPM')
        ax.plot(df['segment_start'], df['gt_bpm'], 'ro-', label='Ground Truth')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('BPM')
        ax.set_title('Heart Rate by Segment')
        ax.legend()
        ax.grid(True)
        
        # Error
        ax = axes[1]
        ax.bar(df['segment_start'], df['error'], width=df['duration']*0.8, alpha=0.7)
        ax.axhline(0, color='black', linestyle='-')
        ax.axhline(df['abs_error'].mean(), color='red', linestyle='--', 
                  label=f"MAE: {df['abs_error'].mean():.1f} BPM")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (BPM)')
        ax.set_title('BPM Error by Segment')
        ax.legend()
        ax.grid(True)
        
        # Correlation
        ax = axes[2]
        ax.scatter(df['gt_bpm'], df['csi_bpm'], s=100, alpha=0.7)
        
        # Perfect line
        min_bpm = min(df['gt_bpm'].min(), df['csi_bpm'].min()) - 5
        max_bpm = max(df['gt_bpm'].max(), df['csi_bpm'].max()) + 5
        ax.plot([min_bpm, max_bpm], [min_bpm, max_bpm], 'k--', label='Perfect')
        
        corr = df['csi_bpm'].corr(df['gt_bpm'])
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
               fontsize=12, verticalalignment='top')
        
        ax.set_xlabel('Ground Truth BPM')
        ax.set_ylabel('CSI BPM')
        ax.set_title('Segment BPM Correlation')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('segment_analysis_results.png', dpi=300)
        print("\nSaved plot: segment_analysis_results.png")
        
        # Save results
        df.to_csv('segment_analysis_results.csv', index=False)
        print("Saved: segment_analysis_results.csv")
        
        return df, metrics
    
    return None, None


if __name__ == "__main__":
    # Test on your data
    segment_analysis(
        r"captures\Allen-S,0,1\csi_capture_20260820_135250.csv",
        r"captures\Allen-S,0,1\ground_truth_20260820_135247.csv",
        max_gap_ms=20  # 20ms gap threshold (about 2x normal interval)
    )