#!/usr/bin/env python3

#------------------------------------------------------------------
# usage py plot_csi.py captures\csi_capture.pcap
#------------------------------------------------------------------

import sys
import numpy as np
from scapy.all import rdpcap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

HOFFSET = 15  # Found from debugging
BW = 80
NFFT = int(BW * 3.2)  # 256 for 80 MHz

# ------------------------------------------------------------------


def extract_all_csi(pcap_file):
    """Extract all CSI data from pcap file"""
    packets = rdpcap(pcap_file)
    
    all_complex = []
    metadata = {}
    
    for pkt in packets:
        raw = bytes(pkt)
        
        if len(raw) < 60:
            continue
            
        usable_len = (len(raw) // 4) * 4
        words = np.frombuffer(raw[:usable_len], dtype="<u4")
        
        if len(words) < (HOFFSET + NFFT):
            continue
            
        # Extract CSI
        H = words[HOFFSET:HOFFSET + NFFT]
        hout = H.view(np.int16)
        
        # Extract metadata from first packet
        if not metadata:
            payload8 = words.view(np.uint8)
            if len(payload8) >= 60:
                metadata['sequence'] = int.from_bytes(payload8[52:54], "little")
                core_ss = int.from_bytes(payload8[54:56], "little")
                metadata['core'] = core_ss & 0x7
                metadata['nss'] = (core_ss >> 3) & 0x7
                metadata['chanspec'] = int.from_bytes(payload8[56:58], "little")
                metadata['chipver'] = int.from_bytes(payload8[58:60], "little")
        
        # Convert to complex
        iq = hout.reshape(-1, 2)
        complex_csi = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
        all_complex.append(complex_csi)
    
    return np.array(all_complex), metadata


def plot_csi_metrics(csi_data, metadata):
    """Create comprehensive CSI plots"""
    
    n_packets, n_subcarriers = csi_data.shape
    
    print(f"Processing {n_packets} packets with {n_subcarriers} subcarriers each")
    
    # Calculate metrics
    magnitudes = np.abs(csi_data)  # Amplitude
    phases = np.angle(csi_data)    # Phase in radians
    
    mean_magnitude = np.mean(magnitudes, axis=0)
    var_magnitude = np.var(magnitudes, axis=0)
    std_magnitude = np.std(magnitudes, axis=0)
    
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
    
    # 3. CSI Amplitude (Magnitude) for all packets per subcarrier (heatmap)
    ax3 = plt.subplot(2, 2, 3)
    # Use a subset of packets if too many
    max_packets_to_plot = min(n_packets, 500)
    step = max(1, n_packets // max_packets_to_plot)
    magnitudes_subset = magnitudes[::step, :]
    
    im = ax3.imshow(magnitudes_subset.T, aspect='auto', cmap='viridis', 
                    extent=[0, len(magnitudes_subset), 0, n_subcarriers])
    ax3.set_xlabel('Packet Number')
    ax3.set_ylabel('Subcarrier Index')
    ax3.set_title(f'CSI Amplitude Heatmap (showing {len(magnitudes_subset)} packets)')
    plt.colorbar(im, ax=ax3, label='Magnitude')
    
    # 4. CSI Phase for all packets per subcarrier (heatmap)
    ax4 = plt.subplot(2, 2, 4)
    phases_subset = phases[::step, :]
    
    im2 = ax4.imshow(phases_subset.T, aspect='auto', cmap='hsv', 
                     extent=[0, len(phases_subset), 0, n_subcarriers],
                     vmin=-np.pi, vmax=np.pi)
    ax4.set_xlabel('Packet Number')
    ax4.set_ylabel('Subcarrier Index')
    ax4.set_title(f'CSI Phase Heatmap (showing {len(phases_subset)} packets)')
    plt.colorbar(im2, ax=ax4, label='Phase (radians)')
    
    plt.suptitle(f'CSI Analysis - {n_packets} Packets, {n_subcarriers} Subcarriers\n'
                 f'Chanspec: 0x{metadata.get("chanspec", 0):04X}, '
                 f'ChipVer: 0x{metadata.get("chipver", 0):04X}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('graphs/csi_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved plot to 'graphs/csi_analysis.png'")
    #plt.show()
    
    return fig


def plot_csi_over_time(csi_data, packet_indices=None, subcarrier_indices=None):
    """Plot CSI magnitude and phase over time for specific subcarriers"""
    
    n_packets, n_subcarriers = csi_data.shape
    
    if packet_indices is None:
        packet_indices = np.arange(n_packets)
    
    if subcarrier_indices is None:
        # Select 5 representative subcarriers
        step = max(1, n_subcarriers // 5)
        subcarrier_indices = np.arange(0, n_subcarriers, step)[:5]
    
    magnitudes = np.abs(csi_data)
    phases = np.angle(csi_data)
    
    # Create time series plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot magnitude over time
    for idx in subcarrier_indices:
        ax1.plot(packet_indices, magnitudes[packet_indices, idx], 
                label=f'Subcarrier {idx}', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Packet Number')
    ax1.set_ylabel('Magnitude')
    ax1.set_title('CSI Magnitude Over Time for Selected Subcarriers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot phase over time
    for idx in subcarrier_indices:
        ax2.plot(packet_indices, phases[packet_indices, idx], 
                label=f'Subcarrier {idx}', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Packet Number')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_title('CSI Phase Over Time for Selected Subcarriers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'CSI Time Series Analysis - {len(subcarrier_indices)} Selected Subcarriers', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs/csi_timeseries.png', dpi=300, bbox_inches='tight')
    print("Saved plot to 'graphs/csi_timeseries.png'")
    #plt.show()
    
    return fig


def plot_statistical_summary(csi_data):
    """Create statistical summary plots"""
    
    n_packets, n_subcarriers = csi_data.shape
    
    magnitudes = np.abs(csi_data)
    phases = np.angle(csi_data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distribution of magnitudes for all subcarriers
    ax1.hist(magnitudes.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of All CSI Magnitudes')
    ax1.grid(True, alpha=0.3)
    
    # Distribution of phases for all subcarriers
    ax2.hist(phases.flatten(), bins=100, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('Phase (radians)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of All CSI Phases')
    ax2.grid(True, alpha=0.3)
    
    # Box plot of magnitudes per subcarrier
    # Only show a subset if too many subcarriers
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
    
    # Mean and std dev of magnitudes across packets
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
    plt.savefig('graphs/csi_statistics.png', dpi=300, bbox_inches='tight')
    print("Saved plot to 'graphs/csi_statistics.png'")
    #plt.show()
    
    return fig


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} capture.pcap")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    
    print(f"Loading CSI data from {pcap_file}...")
    csi_data, metadata = extract_all_csi(pcap_file)
    
    if len(csi_data) == 0:
        print("No valid CSI data found!")
        sys.exit(1)
    
    n_packets, n_subcarriers = csi_data.shape
    print(f"Successfully extracted {n_packets} packets with {n_subcarriers} subcarriers each")
    
    # Print metadata
    print("\nMetadata:")
    print(f"  Sequence: {metadata.get('sequence', 'N/A')}")
    print(f"  Core: {metadata.get('core', 'N/A')}")
    print(f"  NSS: {metadata.get('nss', 'N/A')}")
    print(f"  Chanspec: 0x{metadata.get('chanspec', 0):04X}")
    print(f"  ChipVer: 0x{metadata.get('chipver', 0):04X}")
    
    # Create the plots
    print("\nGenerating plots...")
    
    # Plot 1: Main analysis plots
    plot_csi_metrics(csi_data, metadata)
    
    # Plot 2: Time series for selected subcarriers
    plot_csi_over_time(csi_data)
    
    # Plot 3: Statistical summary
    plot_statistical_summary(csi_data)
    
    print("\nAll plots generated successfully!")
    print("Files saved:")
    print("  - csi_analysis.png")
    print("  - csi_timeseries.png")
    print("  - csi_statistics.png")


if __name__ == "__main__":
    main()