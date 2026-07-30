"""
filters.py
==========
Signal-cleaning filters applied along the time (packet) axis, per
subcarrier. Useful before FFT motion detection or plotting, since raw CSI
amplitude is often noisy with occasional large outliers from decode
glitches or RF interference.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def hampel_filter(x: np.ndarray, window_size: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """
    Hampel outlier filter along axis 0 (time). Works on a 1D or 2D array
    (2D: axis 0 = time/packets, axis 1 = subcarriers -- filtered independently
    per column).

    Replaces samples that deviate from the local median by more than
    n_sigmas * (1.4826 * local MAD) with the local median. Good for
    knocking out isolated spike artifacts without smearing real motion.
    """
    x = np.asarray(x, dtype=np.float64)
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]

    n, n_cols = x.shape
    out = x.copy()
    k = window_size // 2
    L = 1.4826  # scale factor so MAD approximates std for a Gaussian

    for col in range(n_cols):
        series = x[:, col]
        for i in range(n):
            lo = max(0, i - k)
            hi = min(n, i + k + 1)
            window = series[lo:hi]
            med = np.median(window)
            mad = np.median(np.abs(window - med))
            threshold = n_sigmas * L * mad
            if threshold == 0:
                continue
            if np.abs(series[i] - med) > threshold:
                out[i, col] = med

    return out[:, 0] if was_1d else out


def butterworth_lowpass(x: np.ndarray, cutoff_hz: float, fs_hz: float,
                         order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter along axis 0 (time).
    x: 1D (time,) or 2D (time, subcarriers) array.
    cutoff_hz: filter cutoff frequency in Hz.
    fs_hz: sampling rate of x in Hz (e.g. dataset.fs_est or your known TX rate).

    Raises ValueError if cutoff_hz >= fs_hz / 2 (Nyquist) or if there aren't
    enough samples for filtfilt's default padding.
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = fs_hz / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(f"cutoff_hz ({cutoff_hz}) must be in (0, {nyq}) given fs_hz={fs_hz}")

    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]

    min_len = 3 * (max(order, 1) * 2 + 1)  # filtfilt's default pad requirement, roughly
    if x.shape[0] <= min_len:
        raise ValueError(
            f"Not enough samples ({x.shape[0]}) for a stable order-{order} "
            f"filtfilt; need > ~{min_len}. Lower the order or capture more packets."
        )

    b, a = butter(order, cutoff_hz / nyq, btype="low")
    out = filtfilt(b, a, x, axis=0)
    return out[:, 0] if was_1d else out


def butterworth_bandpass(x: np.ndarray, low_hz: float, high_hz: float, fs_hz: float,
                          order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass filter along axis 0 (time). Handy for
    isolating a human-motion band (e.g. 0.1-2 Hz for walking/breathing)
    before an FFT motion check.
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = fs_hz / 2.0
    if not (0 < low_hz < high_hz < nyq):
        raise ValueError(f"Require 0 < low_hz < high_hz < nyquist ({nyq}); "
                          f"got low={low_hz}, high={high_hz}")

    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]

    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    out = filtfilt(b, a, x, axis=0)
    return out[:, 0] if was_1d else out


def remove_dc_drift(x: np.ndarray, window_size: int = 21) -> np.ndarray:
    """Subtract a rolling mean to remove slow drift from each subcarrier."""
    x = np.asarray(x, dtype=np.float64)
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]

    if x.shape[0] < 2:
        return x[:, 0] if was_1d else x

    if window_size < 2:
        window_size = 2

    pad = window_size // 2
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    moving_avg = np.apply_along_axis(lambda y: np.convolve(y, kernel, mode="same"), 0, x)
    detrended = x - moving_avg
    return detrended[:, 0] if was_1d else detrended


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Standardize each subcarrier to zero-mean, unit-variance."""
    x = np.asarray(x, dtype=np.float64)
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]

    std = np.std(x, axis=0, keepdims=True)
    std[std == 0] = 1.0
    out = (x - np.mean(x, axis=0, keepdims=True)) / std
    return out[:, 0] if was_1d else out


def preprocess_subcarrier_signal(
    x: np.ndarray,
    fs_hz: float,
    hampel_window: int = 51,
    hampel_sigmas: float = 3.0,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
    order: int = 4,
) -> np.ndarray:
    """Apply the full heartbeat-oriented filtering chain: Hampel → bandpass → drift removal → normalization."""
    if x.ndim != 1:
        raise ValueError("preprocess_subcarrier_signal expects a 1D signal")

    filtered = hampel_filter(x, window_size=hampel_window, n_sigmas=hampel_sigmas)
    filtered = butterworth_bandpass(filtered, low_hz=low_hz, high_hz=high_hz, fs_hz=fs_hz, order=order)
    filtered = remove_dc_drift(filtered)
    filtered = normalize_signal(filtered)
    return filtered
