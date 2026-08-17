#!/usr/bin/env python3

# ----------------------------------------------------------------------
# Usage: python filter_csi.py csi_capture_*.csv
# Applies notch and Hampel filters to remove DC outliers and permanent zeros
# ----------------------------------------------------------------------

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

class FilterConfig:
    """Configuration for CSI filtering"""
    
    # DC Notch Filter Settings
    NOTCH_FREQ = 0.0  # DC component
    NOTCH_Q = 30.0    # Quality factor (higher = narrower notch)
    
    # Hampel Filter Settings
    HAMPEL_WINDOW_SIZE = 11  # Window size for median filter (must be odd)
    HAMPEL_THRESHOLD = 3.0   # Number of standard deviations for outlier detection
    
    # Zero Detection
    ZERO_THRESHOLD = 1e-6   # Threshold to consider a value as zero
    
    # Visualization
    PLOT_SUBCARRIERS = [0, 1, 50, 100, 150, 200]  # Subcarriers to plot for comparison


# ------------------------------------------------------------------
# Filter Functions
# ------------------------------------------------------------------

def detect_dc_subcarriers(csi_data, threshold=0.1):
    """
    Detect DC subcarriers (those with very low amplitude variation)
    Returns boolean mask of DC subcarriers
    """
    magnitudes = np.abs(csi_data)
    
    # Calculate mean magnitude per subcarrier
    mean_mag = np.mean(magnitudes, axis=0)
    
    # Calculate variance per subcarrier
    var_mag = np.var(magnitudes, axis=0)
    
    # Normalize variance by mean (coefficient of variation)
    cv = np.zeros_like(var_mag)
    mask = mean_mag > 0
    cv[mask] = var_mag[mask] / mean_mag[mask]
    
    # DC subcarriers have very low variation and low mean
    dc_mask = (mean_mag < np.percentile(mean_mag, 5)) | (cv < threshold)
    
    return dc_mask


def detect_permanent_zeros(csi_data, zero_threshold=1e-6):
    """
    Detect subcarriers that are permanently zero
    Returns boolean mask of permanent zero subcarriers
    """
    magnitudes = np.abs(csi_data)
    
    # Check if all values for a subcarrier are below threshold
    permanent_zero_mask = np.all(magnitudes < zero_threshold, axis=0)
    
    return permanent_zero_mask


def apply_notch_filter(csi_data, fs=100, freq=0.0, Q=30.0):
    """
    Apply notch filter to remove specific frequency components
    """
    # Calculate normalized frequency
    nyquist = fs / 2
    normalized_freq = freq / nyquist if nyquist > 0 else 0
    
    if normalized_freq <= 0 or normalized_freq >= 1:
        # If DC, use a high-pass filter instead
        b, a = signal.butter(1, 0.01, btype='high', fs=fs)
    else:
        # Create notch filter
        b, a = signal.iirnotch(normalized_freq, Q)
    
    # Apply filter to each subcarrier
    filtered_data = np.zeros_like(csi_data)
    for sc in range(csi_data.shape[1]):
        # Filter real and imaginary parts separately
        filtered_data[:, sc] = signal.filtfilt(
            b, a, csi_data[:, sc], axis=0
        )
    
    return filtered_data


def apply_hampel_filter(csi_data, window_size=11, threshold=3.0):
    """
    Apply Hampel filter to remove outliers
    """
    if window_size % 2 == 0:
        window_size += 1  # Make sure it's odd
    
    half_window = window_size // 2
    filtered_data = csi_data.copy()
    
    for sc in range(csi_data.shape[1]):
        # Process real and imaginary parts separately
        for component_idx in [0, 1]:  # 0 = real, 1 = imag
            component = csi_data[:, sc].real if component_idx == 0 else csi_data[:, sc].imag
            
            for i in range(len(component)):
                # Get window indices
                start = max(0, i - half_window)
                end = min(len(component), i + half_window + 1)
                window = component[start:end]
                
                # Compute median and MAD
                median_val = np.median(window)
                mad = np.median(np.abs(window - median_val))
                
                # Check if value is outlier
                if mad > 0:
                    z_score = np.abs(component[i] - median_val) / (1.4826 * mad)
                    if z_score > threshold:
                        # Replace with median
                        if component_idx == 0:
                            filtered_data[i, sc] = median_val + 1j * filtered_data[i, sc].imag
                        else:
                            filtered_data[i, sc] = filtered_data[i, sc].real + 1j * median_val
    
    return filtered_data


def apply_combined_filter(csi_data, config):
    """
    Apply combination of filters: notch + Hampel
    """
    # Step 1: Identify problematic subcarriers
    dc_mask = detect_dc_subcarriers(csi_data)
    zero_mask = detect_permanent_zeros(csi_data, config.ZERO_THRESHOLD)
    problematic_mask = dc_mask | zero_mask
    
    print(f"\nFiltering Statistics:")
    print(f"  DC subcarriers detected: {np.sum(dc_mask)}")
    print(f"  Permanent zero subcarriers: {np.sum(zero_mask)}")
    print(f"  Total problematic subcarriers: {np.sum(problematic_mask)}")
    
    # Step 2: Apply notch filter to remove DC component
    print("  Applying notch filter...")
    filtered_data = apply_notch_filter(
        csi_data, 
        freq=config.NOTCH_FREQ, 
        Q=config.NOTCH_Q
    )
    
    # Step 3: Apply Hampel filter to remove outliers
    print("  Applying Hampel filter...")
    filtered_data = apply_hampel_filter(
        filtered_data,
        window_size=config.HAMPEL_WINDOW_SIZE,
        threshold=config.HAMPEL_THRESHOLD
    )
    
    # Step 4: Replace problematic subcarriers with interpolated values
    print("  Interpolating problematic subcarriers...")
    for sc in range(csi_data.shape[1]):
        if problematic_mask[sc]:
            # Interpolate from neighboring subcarriers
            left_sc = sc - 1
            right_sc = sc + 1
            
            while left_sc >= 0 and problematic_mask[left_sc]:
                left_sc -= 1
            while right_sc < csi_data.shape[1] and problematic_mask[right_sc]:
                right_sc += 1
            
            if left_sc >= 0 and right_sc < csi_data.shape[1]:
                # Interpolate between neighbors
                weight_left = 1 - (sc - left_sc) / (right_sc - left_sc)
                weight_right = 1 - (right_sc - sc) / (right_sc - left_sc)
                
                # Interpolate both real and imaginary parts
                filtered_data[:, sc] = (
                    weight_left * filtered_data[:, left_sc] + 
                    weight_right * filtered_data[:, right_sc]
                )
            elif left_sc >= 0:
                filtered_data[:, sc] = filtered_data[:, left_sc]
            elif right_sc < csi_data.shape[1]:
                filtered_data[:, sc] = filtered_data[:, right_sc]
            else:
                # If no neighbors, use mean of all subcarriers
                filtered_data[:, sc] = np.mean(filtered_data, axis=1)
    
    return filtered_data, problematic_mask


# ------------------------------------------------------------------
# Visualization Functions
# ------------------------------------------------------------------

def plot_filter_comparison(csi_original, csi_filtered, metadata, 
                          subcarriers=None, output_dir='graphs'):
    """
    Plot comparison of original vs filtered CSI data
    """
    if subcarriers is None:
        subcarriers = [0, 1, 50, 100, 150, 200]
    
    # Ensure subcarriers are within range
    n_subcarriers = csi_original.shape[1]
    subcarriers = [s for s in subcarriers if s < n_subcarriers]
    if not subcarriers:
        subcarriers = [0, min(1, n_subcarriers-1), min(50, n_subcarriers-1)]
    
    magnitudes_orig = np.abs(csi_original)
    magnitudes_filt = np.abs(csi_filtered)
    
    fig, axes = plt.subplots(len(subcarriers), 2, figsize=(15, 4*len(subcarriers)))
    
    if len(subcarriers) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sc in enumerate(subcarriers):
        # Original magnitude
        axes[idx, 0].plot(magnitudes_orig[:, sc], 'b-', alpha=0.7, linewidth=1)
        axes[idx, 0].set_title(f'Original - Subcarrier {sc}')
        axes[idx, 0].set_xlabel('Packet Number')
        axes[idx, 0].set_ylabel('Magnitude')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Filtered magnitude
        axes[idx, 1].plot(magnitudes_filt[:, sc], 'r-', alpha=0.7, linewidth=1)
        axes[idx, 1].set_title(f'Filtered - Subcarrier {sc}')
        axes[idx, 1].set_xlabel('Packet Number')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.suptitle('CSI Magnitude Comparison: Original vs Filtered', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_dir).mkdir(exist_ok=True)
    plt.savefig(f'{output_dir}/filter_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison plot to 'graphs/filter_comparison.png'")
    plt.close()


def plot_filtered_analysis(csi_filtered, metadata, output_dir='graphs'):
    """
    Generate analysis plots for filtered data
    """
    n_packets, n_subcarriers = csi_filtered.shape
    
    magnitudes = np.abs(csi_filtered)
    phases = np.angle(csi_filtered)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Mean magnitude per subcarrier
    mean_mag = np.mean(magnitudes, axis=0)
    ax1.plot(mean_mag, 'b-', linewidth=1.5)
    ax1.set_xlabel('Subcarrier Index')
    ax1.set_ylabel('Mean Magnitude')
    ax1.set_title('Mean CSI Magnitude (Filtered)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Variance per subcarrier
    var_mag = np.var(magnitudes, axis=0)
    ax2.plot(var_mag, 'r-', linewidth=1.5)
    ax2.set_xlabel('Subcarrier Index')
    ax2.set_ylabel('Variance')
    ax2.set_title('CSI Variance (Filtered)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Amplitude heatmap
    max_packets = min(n_packets, 500)
    step = max(1, n_packets // max_packets)
    magnitudes_subset = magnitudes[::step, :]
    
    im1 = ax3.imshow(magnitudes_subset.T, aspect='auto', cmap='viridis',
                     extent=[0, len(magnitudes_subset), 0, n_subcarriers])
    ax3.set_xlabel('Packet Number')
    ax3.set_ylabel('Subcarrier Index')
    ax3.set_title(f'CSI Amplitude Heatmap (Filtered, showing {len(magnitudes_subset)} packets)')
    plt.colorbar(im1, ax=ax3, label='Magnitude')
    
    # 4. Phase heatmap
    phases_subset = phases[::step, :]
    im2 = ax4.imshow(phases_subset.T, aspect='auto', cmap='hsv',
                     extent=[0, len(phases_subset), 0, n_subcarriers],
                     vmin=-np.pi, vmax=np.pi)
    ax4.set_xlabel('Packet Number')
    ax4.set_ylabel('Subcarrier Index')
    ax4.set_title(f'CSI Phase Heatmap (Filtered, showing {len(phases_subset)} packets)')
    plt.colorbar(im2, ax=ax4, label='Phase (radians)')
    
    plt.suptitle(f'Filtered CSI Analysis - {n_packets} Packets, {n_subcarriers} Subcarriers',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_dir).mkdir(exist_ok=True)
    plt.savefig(f'{output_dir}/filtered_csi_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved filtered analysis to 'graphs/filtered_csi_analysis.png'")
    plt.close()


def plot_subcarrier_distribution(csi_original, csi_filtered, problematic_mask, output_dir='graphs'):
    """
    Plot distribution comparison of problematic subcarriers
    """
    if not np.any(problematic_mask):
        print("No problematic subcarriers to visualize")
        return
    
    # Find first few problematic subcarriers to show
    problematic_indices = np.where(problematic_mask)[0][:5]
    
    if len(problematic_indices) == 0:
        return
    
    fig, axes = plt.subplots(2, len(problematic_indices), figsize=(4*len(problematic_indices), 6))
    
    if len(problematic_indices) == 1:
        axes = axes.reshape(2, -1)
    
    for idx, sc in enumerate(problematic_indices):
        # Original distribution
        mag_orig = np.abs(csi_original[:, sc])
        axes[0, idx].hist(mag_orig, bins=50, alpha=0.7, color='blue')
        axes[0, idx].set_title(f'Original - Subcarrier {sc}')
        axes[0, idx].set_xlabel('Magnitude')
        axes[0, idx].set_ylabel('Frequency')
        
        # Filtered distribution
        mag_filt = np.abs(csi_filtered[:, sc])
        axes[1, idx].hist(mag_filt, bins=50, alpha=0.7, color='red')
        axes[1, idx].set_title(f'Filtered - Subcarrier {sc}')
        axes[1, idx].set_xlabel('Magnitude')
        axes[1, idx].set_ylabel('Frequency')
    
    plt.suptitle('Distribution Comparison: Problematic Subcarriers', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_dir).mkdir(exist_ok=True)
    plt.savefig(f'{output_dir}/problematic_subcarriers.png', dpi=300, bbox_inches='tight')
    print("Saved problematic subcarrier plot to 'graphs/problematic_subcarriers.png'")
    plt.close()


# ------------------------------------------------------------------
# Main Functions
# ------------------------------------------------------------------

def extract_csi_from_csv(csv_file):
    """Extract CSI data from CSV file"""
    print(f"Loading CSI data from {csv_file}...")
    
    df = pd.read_csv(csv_file)
    
    # Identify complex columns
    complex_cols = [col for col in df.columns if col not in ['ts', 'seq_num', 'core', 'spatial_stream']]
    n_sc = len(complex_cols) // 2
    
    # Extract complex data
    csi_data = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        i_vals = [row[f'sc{sc}_re'] for sc in range(n_sc)]
        q_vals = [row[f'sc{sc}_im'] for sc in range(n_sc)]
        complex_csi = np.array(i_vals, dtype=np.float64) + 1j * np.array(q_vals, dtype=np.float64)
        csi_data.append(complex_csi)
    
    csi_array = np.array(csi_data)
    
    metadata = {
        'n_packets': len(df),
        'n_subcarriers': n_sc,
        'duration': df['ts'].iloc[-1] - df['ts'].iloc[0],
        'rate': len(df) / (df['ts'].iloc[-1] - df['ts'].iloc[0]),
    }
    
    return csi_array, metadata


def run_original_plots(csi_data, metadata, output_dir='graphs'):
    """Run the original plot_csv.py analysis"""
    # This is a simplified version - you can import and use plot_csv.py functions directly
    from plot_csv import plot_csi_metrics, plot_csi_over_time, plot_statistical_summary
    
    print("\nGenerating original analysis plots...")
    
    # Generate original plots
    plot_csi_metrics(csi_data, metadata, output_dir)
    plot_csi_over_time(csi_data, output_dir=output_dir)
    plot_statistical_summary(csi_data, output_dir)
    
    # Rename to indicate original
    for plot_name in ['csv_csi_analysis.png', 'csv_csi_timeseries.png', 'csv_csi_statistics.png']:
        old_path = Path(output_dir) / plot_name
        if old_path.exists():
            new_path = Path(output_dir) / f'original_{plot_name}'
            old_path.rename(new_path)
            print(f"  Renamed {plot_name} to original_{plot_name}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} csi_capture_*.csv")
        print("\nThis script applies notch and Hampel filters to CSI data")
        print("to remove DC outliers and permanent zeros.")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not Path(csv_file).exists():
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    
    try:
        # Load data
        csi_data, metadata = extract_csi_from_csv(csv_file)
        print(f"Loaded {metadata['n_packets']} packets with {metadata['n_subcarriers']} subcarriers")
        
        # Create output directory
        output_dir = 'graphs'
        Path(output_dir).mkdir(exist_ok=True)
        
        # Run original analysis (before filtering)
        run_original_plots(csi_data, metadata, output_dir)
        
        # Apply filters
        config = FilterConfig()
        print("\n" + "="*60)
        print("Applying Filters...")
        print("="*60)
        
        # Detect problematic subcarriers before filtering
        dc_mask = detect_dc_subcarriers(csi_data)
        zero_mask = detect_permanent_zeros(csi_data)
        
        print(f"\nPre-filtering detection:")
        print(f"  DC subcarriers: {np.sum(dc_mask)}")
        print(f"  Permanent zeros: {np.sum(zero_mask)}")
        
        # Apply combined filter
        filtered_data, problematic_mask = apply_combined_filter(csi_data, config)
        
        # Show filtering statistics
        print(f"\nFiltering completed:")
        print(f"  Data shape: {filtered_data.shape}")
        print(f"  Modified packets: {np.sum(np.abs(filtered_data - csi_data) > 0.01)}")
        
        # Generate comparison visualizations
        print("\nGenerating comparison plots...")
        
        # Plot comparison for selected subcarriers
        plot_filter_comparison(csi_data, filtered_data, metadata, 
                              config.PLOT_SUBCARRIERS, output_dir)
        
        # Plot distribution of problematic subcarriers
        plot_subcarrier_distribution(csi_data, filtered_data, problematic_mask, output_dir)
        
        # Generate filtered analysis
        plot_filtered_analysis(filtered_data, metadata, output_dir)
        
        # Save filtered data (optional)
        filtered_csv = csv_file.replace('.csv', '_filtered.csv')
        print(f"\nSaving filtered data to {filtered_csv}...")
        
        # Create filtered CSV
        df_orig = pd.read_csv(csv_file)
        df_filtered = df_orig.copy()
        
        # Update complex columns with filtered data
        for sc in range(filtered_data.shape[1]):
            df_filtered[f'sc{sc}_re'] = filtered_data[:, sc].real
            df_filtered[f'sc{sc}_im'] = filtered_data[:, sc].imag
        
        df_filtered.to_csv(filtered_csv, index=False)
        print(f"  Filtered data saved to {filtered_csv}")
        
        print("\n" + "="*60)
        print("Filtering Complete!")
        print("="*60)
        print("\nGenerated files:")
        print(f"  - {output_dir}/filter_comparison.png")
        print(f"  - {output_dir}/filtered_csi_analysis.png")
        print(f"  - {output_dir}/problematic_subcarriers.png")
        print(f"  - {output_dir}/original_*.png (original analysis)")
        print(f"  - {filtered_csv} (filtered CSV data)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()