"""
quality.py
==========
Subcarrier quality scoring for CSI heartbeat pipelines.

Each subcarrier is scored from a mix of temporal variability, spectral peak
strength, and, when available, correlation with a ground-truth signal.
The result is a ranking that can be used to keep only the most informative
carriers for later ML feature extraction.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _to_2d_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("signals must be a 1D or 2D array of shape (time, subcarriers)")
    return x


def _spectral_peak_strength(signal: np.ndarray, fs_hz: float, low_hz: float, high_hz: float) -> float:
    if fs_hz <= 0:
        return 0.0
    if len(signal) < 4:
        return 0.0

    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    n = len(signal)
    window = np.hanning(n)
    spec = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    mag = np.abs(spec)
    if len(mag) <= 1:
        return 0.0

    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0

    target = mag[mask]
    noise_floor = np.median(mag[~mask]) if np.any(~mask) else np.median(target)
    peak = float(np.max(target))
    if peak <= 0:
        return 0.0
    return float(peak / (noise_floor + 1e-8))


def evaluate_subcarriers(
    signals: np.ndarray,
    fs_hz: float,
    ground_truth: Optional[np.ndarray] = None,
    low_hz: float = 0.75,
    high_hz: float = 2.0,
    top_k: int = 10,
) -> dict:
    """
    Score each subcarrier on a weighted combination of:
      - variance (temporal variability)
      - peak SNR-like spectral strength in a target band
      - stability (correlation between first/second halves)
      - optional correlation with ground truth

    Returns a dictionary containing score vectors and the selected indices.
    """
    x = _to_2d_signal(signals)
    n_samples, n_subcarriers = x.shape
    if n_samples < 4:
        raise ValueError("Need at least 4 samples to score subcarriers")

    scores = np.zeros(n_subcarriers, dtype=np.float64)
    metrics = []

    for i in range(n_subcarriers):
        signal = x[:, i]
        variance = float(np.var(signal))
        peak_strength = _spectral_peak_strength(signal, fs_hz, low_hz, high_hz)

        mid = n_samples // 2
        if mid >= 2:
            first = signal[:mid]
            second = signal[mid:]
            stability = float(np.corrcoef(first, second)[0, 1]) if len(first) == len(second) else 0.0
            if not np.isfinite(stability):
                stability = 0.0
        else:
            stability = 0.0

        corr_gt = 0.0
        if ground_truth is not None:
            gt = np.asarray(ground_truth, dtype=np.float64)
            if gt.shape[0] != n_samples:
                raise ValueError("ground_truth must have the same length as the time axis")
            corr = float(np.corrcoef(signal, gt)[0, 1]) if len(signal) > 1 else 0.0
            corr_gt = corr if np.isfinite(corr) else 0.0

        variance_score = np.clip(variance / (variance + 1.0), 0.0, 1.0)
        peak_score = np.clip(np.log10(peak_strength + 1.0) / 2.0, 0.0, 1.0)
        stability_score = np.clip((stability + 1.0) / 2.0, 0.0, 1.0)
        corr_score = np.clip((corr_gt + 1.0) / 2.0, 0.0, 1.0)

        composite = 0.4 * corr_score + 0.3 * peak_score + 0.2 * variance_score + 0.1 * stability_score
        scores[i] = composite
        metrics.append({
            "variance": variance,
            "peak_strength": peak_strength,
            "stability": stability,
            "correlation": corr_gt,
            "score": float(composite),
        })

    ranked = np.argsort(scores)[::-1]
    selected = ranked[: max(1, min(int(top_k), len(ranked)))].tolist()

    return {
        "scores": scores,
        "metrics": metrics,
        "selected_indices": selected,
        "ranked_indices": ranked.tolist(),
    }
