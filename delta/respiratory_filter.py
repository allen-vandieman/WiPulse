# respiratory_filter.py
# Remove respiratory component from CSI signal - think this was an idea

import numpy as np
from scipy import signal

def remove_respiration(csi_data, fs, respiration_freq_range=(0.15, 0.4)):
    """
    Remove respiratory component from CSI data
    
    Args:
        csi_data: (n_packets, n_subcarriers) complex CSI
        fs: sampling rate (Hz)
        respiration_freq_range: (low, high) in Hz
    
    Returns:
        filtered_data: CSI with respiratory component removed
    """
    magnitudes = np.abs(csi_data)
    filtered = np.zeros_like(csi_data)
    
    # Design notch filter at respiratory frequency
    # Use adaptive filter that finds the dominant respiratory frequency
    
    for sc in range(csi_data.shape[1]):
        # Compute PSD to find dominant respiratory frequency
        freqs, psd = signal.welch(
            magnitudes[:, sc] - np.mean(magnitudes[:, sc]),
            fs=fs,
            nperseg=min(256, len(magnitudes)),
            scaling='density'
        )
        
        # Find peak in respiration range
        mask = (freqs >= respiration_freq_range[0]) & (freqs <= respiration_freq_range[1])
        if np.any(mask):
            resp_freq = freqs[mask][np.argmax(psd[mask])]
        else:
            resp_freq = 0.25  # Default
        
        if resp_freq > 0 and resp_freq < fs/2:
            # Design notch filter
            nyquist = fs / 2
            b, a = signal.iirnotch(resp_freq / nyquist, Q=10)
            filtered[:, sc] = signal.filtfilt(b, a, magnitudes[:, sc])
        else:
            filtered[:, sc] = magnitudes[:, sc]
    
    return filtered