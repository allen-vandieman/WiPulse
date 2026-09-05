# diagnose_correlation.py
import numpy as np
import pandas as pd
from scipy import signal
from read_csv_csi import CSIDataLoader
from filters import CSIFilter, FilterConfig

def diagnose(csi_path, gt_path):
    # Load data
    loader = CSIDataLoader(csi_path)
    csi_data, metadata = loader.load()
    gt_df = pd.read_csv(gt_path)
    
    fs = metadata['rate']
    
    # Apply filters
    filter_obj = CSIFilter(FilterConfig())
    filtered_csi = filter_obj.apply_all_filters(csi_data, verbose=False)
    magnitudes = np.abs(filtered_csi)
    
    # Use best subcarriers
    best_sc = [55, 54, 51, 46, 50]
    signal_data = np.mean(magnitudes[:, best_sc], axis=1)
    
    # Bandpass filter (0.82-2.55 Hz, order 3)
    nyquist = fs / 2
    sos = signal.butter(3, [0.82/nyquist, 2.55/nyquist], btype='band', output='sos')
    filtered = signal.sosfiltfilt(sos, signal_data - np.mean(signal_data))
    
    # Detect peaks
    peaks, _ = signal.find_peaks(filtered, distance=int(0.3*fs), 
                                prominence=np.std(filtered)*0.4,
                                height=np.mean(filtered)+np.std(filtered)*0.2)
    
    # Get HR from CSI
    csi_times = np.arange(len(filtered)) / fs
    csi_beat_times = csi_times[peaks]
    csi_hr = 60.0 / np.diff(csi_beat_times)
    
    # Get HR from ground truth
    gt_times = gt_df['ts'].values - gt_df['ts'].values[0]
    gt_hr = 60.0 / np.diff(gt_times)
    
    # Align lengths
    min_len = min(len(csi_hr), len(gt_hr))
    csi_hr = csi_hr[:min_len]
    gt_hr = gt_hr[:min_len]
    
    # Print diagnostics
    print(f"CSI HR: mean={np.mean(csi_hr):.1f}, std={np.std(csi_hr):.1f}")
    print(f"GT HR: mean={np.mean(gt_hr):.1f}, std={np.std(gt_hr):.1f}")
    print(f"Correlation: {np.corrcoef(csi_hr, gt_hr)[0,1]:.3f}")
    print(f"HR diff: {np.mean(csi_hr - gt_hr):.1f} BPM")
    
    # Plot comparison
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(csi_hr, 'b-', label='CSI HR', alpha=0.7)
    axes[0].plot(gt_hr, 'r-', label='GT HR', alpha=0.7)
    axes[0].set_ylabel('Heart Rate (BPM)')
    axes[0].set_title('Heart Rate Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].scatter(gt_hr, csi_hr, alpha=0.6)
    axes[1].plot([40, 120], [40, 120], 'k--')
    axes[1].set_xlabel('GT HR (BPM)')
    axes[1].set_ylabel('CSI HR (BPM)')
    axes[1].set_title(f'Correlation: {np.corrcoef(csi_hr, gt_hr)[0,1]:.3f}')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('diagnostic_correlation.png', dpi=300)
    print("Saved diagnostic plot to diagnostic_correlation.png")

# Run it
diagnose(r"captures\Allen-S,0,1\csi_capture_20260820_135250.csv", 
         r"captures\Allen-S,0,1\ground_truth_20260820_135247.csv")