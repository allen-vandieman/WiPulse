# improved_respiratory_filter.py
# Better respiratory removal using bandstop filter

import numpy as np
from scipy import signal

def remove_respiration_improved(signal_data, fs):
    """
    Remove respiratory component using a notch filter
    Higher Q means narrower notch, less impact on heartbeat
    """
    # Find dominant respiratory frequency
    freqs, psd = signal.welch(
        signal_data - np.mean(signal_data),
        fs=fs,
        nperseg=min(256, len(signal_data)),
        scaling='density'
    )
    
    # Find peak in 0.15-0.4 Hz
    resp_mask = (freqs >= 0.15) & (freqs <= 0.4)
    if np.any(resp_mask):
        resp_freq = freqs[resp_mask][np.argmax(psd[resp_mask])]
    else:
        resp_freq = 0.25
    
    # Use a high-Q notch filter (narrow bandwidth)
    nyquist = fs / 2
    if resp_freq > 0 and resp_freq < nyquist:
        # Q=30 gives a very narrow notch
        b, a = signal.iirnotch(resp_freq / nyquist, Q=30)
        filtered = signal.filtfilt(b, a, signal_data)
        return filtered, resp_freq
    else:
        return signal_data, resp_freq


def remove_respiration_with_reference(magnitudes, fs, reference_sc=None):
    """
    Remove respiration using a reference subcarrier
    The respiratory signal is similar across subcarriers, so we can use
    one subcarrier as a reference and subtract it from others
    """
    if reference_sc is None:
        # Use subcarrier 0 (usually has strongest respiratory signal)
        reference_sc = 0
    
    # Get the respiratory signal from reference subcarrier
    ref_signal = magnitudes[:, reference_sc]
    
    # Find respiratory frequency in reference
    freqs, psd = signal.welch(
        ref_signal - np.mean(ref_signal),
        fs=fs,
        nperseg=min(256, len(ref_signal)),
        scaling='density'
    )
    
    resp_mask = (freqs >= 0.15) & (freqs <= 0.4)
    if np.any(resp_mask):
        resp_freq = freqs[resp_mask][np.argmax(psd[resp_mask])]
    else:
        resp_freq = 0.25
    
    # Extract respiratory component from reference
    nyquist = fs / 2
    low = max(0.01, (resp_freq - 0.05) / nyquist)
    high = min(0.99, (resp_freq + 0.05) / nyquist)
    b, a = signal.butter(2, [low, high], btype='bandpass')
    
    resp_component = signal.filtfilt(b, a, ref_signal - np.mean(ref_signal))
    
    # Remove respiratory component from combined signal
    # Use linear regression to find the best scaling
    # (respiratory signal may have different amplitude across subcarriers)
    
    return resp_component, resp_freq