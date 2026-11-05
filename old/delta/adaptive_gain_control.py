# adaptive_gain_control.py
# Remove respiratory amplitude modulation

import numpy as np
from scipy import signal

def remove_respiratory_amplitude_modulation(signal_data, fs):
    """
    Remove respiratory amplitude modulation by normalizing
    the signal envelope over time.
    
    The respiratory signal causes amplitude modulation of the
    heartbeat signal. By estimating the envelope and dividing
    by it, we remove this effect.
    """
    # 1. Extract the envelope (amplitude modulation)
    # Use Hilbert transform or RMS in sliding window
    
    # Option A: Hilbert transform envelope
    analytic_signal = signal.hilbert(signal_data)
    envelope = np.abs(analytic_signal)
    
    # Smooth the envelope (remove high-frequency variations)
    window = int(0.5 * fs)  # 0.5 second smoothing
    kernel = np.ones(window) / window
    smooth_envelope = np.convolve(envelope, kernel, mode='same')
    
    # Avoid division by zero
    smooth_envelope = np.maximum(smooth_envelope, np.percentile(smooth_envelope, 10))
    
    # 2. Normalize the signal by the envelope
    normalized_signal = signal_data / smooth_envelope
    
    # 3. Remove any remaining DC offset
    normalized_signal = normalized_signal - np.mean(normalized_signal)
    
    return normalized_signal, smooth_envelope


def remove_baseline_wander(signal_data, fs):
    """
    Remove baseline wander using a high-pass filter
    """
    nyquist = fs / 2
    # Remove frequencies below 0.5 Hz (respiration and slower)
    sos = signal.butter(2, 0.5 / nyquist, btype='high', output='sos')
    filtered = signal.sosfiltfilt(sos, signal_data)
    return filtered