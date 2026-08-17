#!/usr/bin/env python3

# ----------------------------------------------------------------------
# Plotting functions for CSI data
# ----------------------------------------------------------------------

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CSIPlotter:
    """Generate various plots for CSI data"""
    
    def __init__(self, output_dir='graphs'):
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)
    
    def plot_csi_metrics(self, csi_data, metadata, prefix=''):
        """Create comprehensive CSI plots"""
        n_packets, n_subcarriers = csi_data.shape
        
        # Calculate metrics
        magnitudes = np.abs(csi_data)
        phases = np.angle(csi_data)
        
        mean_magnitude = np.mean(magnitudes, axis=0)
        var_magnitude = np.var(magnitudes, axis=0)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Mean CSI Magnitude per subcarrier
        ax1 = plt.subplot(2, 2, 1)
        subcarriers = np.arange(n_subcarriers)
        ax1.plot(subcarriers, mean_magnitude, 'b-', linewidth=1.5)
        ax1.set_xlabel('Subcarrier Index')
        ax1.set_ylabel('Mean Magnitude')
        ax1.set_title(f'Mean CSI Magnitude per Subcarrier (n={n_packets} packets)')
        ax1.grid(True, alpha=0.3)
        
        # 2. CSI Variance per subcarrier
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(subcarriers, var_magnitude, 'r-', linewidth=1.5)
        ax2.set_xlabel('Subcarrier Index')
        ax2.set_ylabel('Variance')
        ax2.set_title(f'CSI Variance per Subcarrier')
        ax2.grid(True, alpha=0.3)
        
        # 3. CSI Amplitude Heatmap
        ax3 = plt.subplot(2, 2, 3)
        max_packets_to_plot = min(n_packets, 500)
        step = max(1, n_packets // max_packets_to_plot)
        magnitudes_subset = magnitudes[::step, :]
        
        im = ax3.imshow(magnitudes_subset.T, aspect='auto', cmap='viridis', 
                        extent=[0, len(magnitudes_subset), 0, n_subcarriers])
        ax3.set_xlabel('Packet Number')
        ax3.set_ylabel('Subcarrier Index')
        ax3.set_title(f'CSI Amplitude Heatmap (showing {len(magnitudes_subset)} packets)')
        plt.colorbar(im, ax=ax3, label='Magnitude')
        
        # 4. CSI Phase Heatmap
        ax4 = plt.subplot(2, 2, 4)
        phases_subset = phases[::step, :]
        
        im2 = ax4.imshow(phases_subset.T, aspect='auto', cmap='hsv', 
                         extent=[0, len(phases_subset), 0, n_subcarriers],
                         vmin=-np.pi, vmax=np.pi)
        ax4.set_xlabel('Packet Number')
        ax4.set_ylabel('Subcarrier Index')
        ax4.set_title(f'CSI Phase Heatmap (showing {len(phases_subset)} packets)')
        plt.colorbar(im2, ax=ax4, label='Phase (radians)')
        
        # Add title
        title = f'CSI Analysis - {n_packets} Packets, {n_subcarriers} Subcarriers'
        if 'duration' in metadata:
            title += f'\nDuration: {metadata["duration"]:.1f}s, Rate: {metadata["rate"]:.1f} Hz'
        if 'core' in metadata:
            title += f', Core: {metadata["core"]}'
        if 'spatial_stream' in metadata:
            title += f', Stream: {metadata["spatial_stream"]}'
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        filename = f'{prefix}csi_analysis.png'
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        print(f"Saved plot to '{self.output_dir}/{filename}'")
        plt.close()
        
        return fig
    
    def plot_csi_timeseries(self, csi_data, subcarrier_indices=None, prefix=''):
        """Plot CSI magnitude and phase over time for specific subcarriers"""
        n_packets, n_subcarriers = csi_data.shape
        
        if subcarrier_indices is None:
            # Select 5 representative subcarriers
            step = max(1, n_subcarriers // 5)
            subcarrier_indices = np.arange(0, n_subcarriers, step)[:5]
            # Ensure we include subcarrier 0 and 1
            if 0 not in subcarrier_indices:
                subcarrier_indices[0] = 0
            if 1 not in subcarrier_indices and n_subcarriers > 1:
                subcarrier_indices[1] = 1
        
        magnitudes = np.abs(csi_data)
        phases = np.angle(csi_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot magnitude over time
        for idx in subcarrier_indices:
            ax1.plot(magnitudes[:, idx], 
                    label=f'Subcarrier {idx}', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Packet Number')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('CSI Magnitude Over Time for Selected Subcarriers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot phase over time
        for idx in subcarrier_indices:
            ax2.plot(phases[:, idx], 
                    label=f'Subcarrier {idx}', linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Packet Number')
        ax2.set_ylabel('Phase (radians)')
        ax2.set_title('CSI Phase Over Time for Selected Subcarriers')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'CSI Time Series Analysis - {len(subcarrier_indices)} Selected Subcarriers', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{prefix}csi_timeseries.png'
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        print(f"Saved plot to '{self.output_dir}/{filename}'")
        plt.close()
        
        return fig
    
    def plot_statistical_summary(self, csi_data, prefix=''):
        """Create statistical summary plots"""
        n_packets, n_subcarriers = csi_data.shape
        
        magnitudes = np.abs(csi_data)
        phases = np.angle(csi_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distribution of magnitudes
        ax1.hist(magnitudes.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Magnitude')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of All CSI Magnitudes')
        ax1.grid(True, alpha=0.3)
        
        # Distribution of phases
        ax2.hist(phases.flatten(), bins=100, alpha=0.7, color='red', edgecolor='black')
        ax2.set_xlabel('Phase (radians)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of All CSI Phases')
        ax2.grid(True, alpha=0.3)
        
        # Box plot of magnitudes per subcarrier
        if n_subcarriers > 50:
            step = max(1, n_subcarriers // 50)
            subcarriers_to_show = np.arange(0, n_subcarriers, step)
            data_to_plot = magnitudes[:, subcarriers_to_show]
            ax3.boxplot(data_to_plot, positions=subcarriers_to_show, widths=0.8, 
                        showfliers=False, patch_artist=True)
            ax3.set_xlabel('Subcarrier Index')
            ax3.set_ylabel('Magnitude')
            ax3.set_title(f'Magnitude Distribution per Subcarrier (showing {len(subcarriers_to_show)} subcarriers)')
        else:
            ax3.boxplot(magnitudes.T, patch_artist=True)
            ax3.set_xlabel('Subcarrier Index')
            ax3.set_ylabel('Magnitude')
            ax3.set_title('Magnitude Distribution per Subcarrier')
        ax3.grid(True, alpha=0.3)
        
        # Mean and std dev
        mean_mag = np.mean(magnitudes, axis=0)
        std_mag = np.std(magnitudes, axis=0)
        ax4.plot(mean_mag, 'b-', label='Mean', linewidth=2)
        ax4.fill_between(np.arange(n_subcarriers), 
                         mean_mag - std_mag, mean_mag + std_mag, 
                         alpha=0.3, color='blue', label='±1 Std Dev')
        ax4.set_xlabel('Subcarrier Index')
        ax4.set_ylabel('Magnitude')
        ax4.set_title('Mean Magnitude with Standard Deviation Bands')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Statistical Summary - {n_packets} Packets, {n_subcarriers} Subcarriers', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{prefix}csi_statistics.png'
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        print(f"Saved plot to '{self.output_dir}/{filename}'")
        plt.close()
        
        return fig
    
    def plot_motion_detection(self, csi_data, window=50, prefix=''):
        """Plot motion detection analysis"""
        n_packets, n_subcarriers = csi_data.shape
        
        magnitudes = np.abs(csi_data)
        
        # Calculate rolling motion score
        motion_scores = []
        for i in range(window, n_packets):
            window_data = magnitudes[i-window:i]
            per_sc_std = np.std(window_data, axis=0)
            mean_amp = np.mean(window_data)
            score = np.mean(per_sc_std) / mean_amp if mean_amp > 0 else 0
            motion_scores.append(score)
        
        motion_scores = np.array(motion_scores)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Motion score over time
        ax1.plot(motion_scores, 'b-', linewidth=1)
        if len(motion_scores) > 0:
            baseline = np.mean(motion_scores[:min(50, len(motion_scores))])
            ax1.axhline(y=baseline, color='g', linestyle='--', label='Baseline')
            ax1.axhline(y=baseline * 1.5, color='r', linestyle='--', label='Threshold (1.5x)')
        ax1.set_xlabel('Packet Number')
        ax1.set_ylabel('Motion Score')
        ax1.set_title(f'Motion Score Over Time (window={window} packets)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mean magnitude over time
        mean_mags = np.mean(magnitudes, axis=1)
        ax2.plot(mean_mags, 'r-', linewidth=1)
        ax2.set_xlabel('Packet Number')
        ax2.set_ylabel('Mean Magnitude')
        ax2.set_title('Average CSI Magnitude Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Motion Detection Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{prefix}motion_detection.png'
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        print(f"Saved plot to '{self.output_dir}/{filename}'")
        plt.close()
        
        return fig
    
    def plot_filter_comparison(self, csi_original, csi_filtered, 
                              subcarriers=None, prefix='filter_'):
        """Plot comparison of original vs filtered CSI data"""
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
        
        filename = f'{prefix}comparison.png'
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to '{self.output_dir}/{filename}'")
        plt.close()
    
    def plot_problematic_subcarriers(self, csi_original, csi_filtered, problematic_mask, prefix='filter_'):
        """Plot distribution comparison of problematic subcarriers"""
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
        
        filename = f'{prefix}problematic_subcarriers.png'
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        print(f"Saved problematic subcarrier plot to '{self.output_dir}/{filename}'")
        plt.close()


def create_all_plots(csi_data, metadata, output_dir='graphs', prefix=''):
    """Convenience function to create all standard plots"""
    plotter = CSIPlotter(output_dir)
    plotter.plot_csi_metrics(csi_data, metadata, prefix)
    plotter.plot_csi_timeseries(csi_data, prefix=prefix)
    plotter.plot_statistical_summary(csi_data, prefix)
    plotter.plot_motion_detection(csi_data, prefix=prefix)
    return plotter


if __name__ == "__main__":
    # Test the plotting functions
    print("Testing plotting functions...")
    test_data = np.random.randn(100, 64) + 1j * np.random.randn(100, 64)
    test_metadata = {'n_packets': 100, 'n_subcarriers': 64, 'duration': 1.0, 'rate': 100.0}
    
    plotter = CSIPlotter('test_graphs')
    plotter.plot_csi_metrics(test_data, test_metadata, 'test_')
    print("Test plots created in 'test_graphs/'")