# motion_timeline.py
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import sys

def analyze_motion_over_time(csv_file, window=50):
    df = pd.read_csv(csv_file)
    
    # Extract magnitudes
    complex_cols = [col for col in df.columns if col not in ['ts', 'seq_num', 'core', 'spatial_stream']]
    n_sc = len(complex_cols) // 2
    
    magnitudes = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        mags = []
        for sc in range(n_sc):
            i = row[f'sc{sc}_re']
            q = row[f'sc{sc}_im']
            mags.append(np.sqrt(i**2 + q**2))
        magnitudes.append(mags)
    
    magnitudes = np.array(magnitudes)
    
    # Calculate rolling motion score
    motion_scores = []
    for i in range(window, len(magnitudes)):
        window_data = magnitudes[i-window:i]
        per_sc_std = np.std(window_data, axis=0)
        mean_amp = np.mean(window_data)
        score = np.mean(per_sc_std) / mean_amp if mean_amp > 0 else 0
        motion_scores.append(score)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Motion score over time
    times = df['ts'].values[window:]
    ax1.plot(times - times[0], motion_scores, 'b-', linewidth=1)
    ax1.axhline(y=np.mean(motion_scores[:50]), color='g', linestyle='--', label='Baseline')
    ax1.axhline(y=np.mean(motion_scores[:50]) * 1.5, color='r', linestyle='--', label='Threshold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Motion Score')
    ax1.set_title('Motion Score Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mean magnitude over time
    mean_mags = np.mean(magnitudes, axis=1)
    ax2.plot(df['ts'].values - df['ts'].values[0], mean_mags, 'r-', linewidth=1)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Mean Magnitude')
    ax2.set_title('Average CSI Magnitude Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('motion_analysis.png', dpi=300)
    plt.show()
    
    print(f"Motion score range: {min(motion_scores):.5f} to {max(motion_scores):.5f}")
    print(f"Baseline (first 50): {np.mean(motion_scores[:50]):.5f}")
    print(f"Peak motion: {max(motion_scores):.5f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python motion_timeline.py csi_capture_*.csv")
        sys.exit(1)
    analyze_motion_over_time(sys.argv[1])