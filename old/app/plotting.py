"""
plotting.py
===========
All figures use matplotlib's Agg backend by default so this runs fine
headless (e.g. over SSH on a Pi with no display). Every plot_* function
saves a PNG to out_path and, if show=True, also calls plt.show() (only
meaningful if a display is actually available).
"""

from __future__ import annotations

import logging
import os

import matplotlib
matplotlib.use("Agg")  # must happen before pyplot import; safe to no-op if already set
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("csi_analysis.plotting")


def _finish(fig, out_path: str, show: bool):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot: %s", out_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_amplitude_time(amp: np.ndarray, out_path: str, subcarrier_idx=None,
                         x_axis=None, title="CSI Amplitude over Time", show=False):
    """
    amp: (n_packets, nfft) amplitude matrix.
    subcarrier_idx: single index, list of indices, or None (mean over all).
    x_axis: optional array of length n_packets for the x-axis (e.g. timestamps).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    x = x_axis if x_axis is not None else np.arange(amp.shape[0])

    if subcarrier_idx is None:
        ax.plot(x, np.mean(amp, axis=1), label="mean over subcarriers")
    elif np.isscalar(subcarrier_idx):
        ax.plot(x, amp[:, subcarrier_idx], label=f"subcarrier {subcarrier_idx}")
    else:
        for idx in subcarrier_idx:
            ax.plot(x, amp[:, idx], label=f"subcarrier {idx}", alpha=0.8)

    ax.set_xlabel("Time (s)" if x_axis is not None else "Packet #")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _finish(fig, out_path, show)


def plot_phase_time(ph: np.ndarray, out_path: str, subcarrier_idx=0,
                     x_axis=None, title="CSI Phase over Time", show=False):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = x_axis if x_axis is not None else np.arange(ph.shape[0])
    if np.isscalar(subcarrier_idx):
        ax.plot(x, ph[:, subcarrier_idx], label=f"subcarrier {subcarrier_idx}")
    else:
        for idx in subcarrier_idx:
            ax.plot(x, ph[:, idx], label=f"subcarrier {idx}", alpha=0.8)
    ax.set_xlabel("Time (s)" if x_axis is not None else "Packet #")
    ax.set_ylabel("Phase (rad)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _finish(fig, out_path, show)


def plot_heatmap(amp: np.ndarray, out_path: str, sc_axis=None,
                  title="CSI Amplitude Heatmap (packet x subcarrier)", show=False):
    """
    amp: (n_packets, nfft) amplitude matrix, ideally already fftshifted so
    columns run low-freq -> DC -> high-freq (use analysis.fftshift_subcarriers
    and analysis.subcarrier_axis for sc_axis).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    n_pkts, nfft = amp.shape
    extent = None
    if sc_axis is not None:
        extent = [sc_axis[0], sc_axis[-1], n_pkts, 0]
    im = ax.imshow(amp, aspect="auto", extent=extent, cmap="viridis",
                    interpolation="nearest")
    ax.set_xlabel("Subcarrier")
    ax.set_ylabel("Packet #")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Amplitude")
    _finish(fig, out_path, show)


def plot_fft(freqs: np.ndarray, magnitude: np.ndarray, out_path: str,
             dominant_freq: float = None, title="FFT Motion Spectrum", show=False,
             max_freq: float = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    if max_freq is not None:
        mask = freqs <= max_freq
        freqs, magnitude = freqs[mask], magnitude[mask]
    ax.plot(freqs, magnitude)
    if dominant_freq is not None:
        ax.axvline(dominant_freq, color="r", linestyle="--",
                    label=f"dominant: {dominant_freq:.3f} Hz")
        ax.legend()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _finish(fig, out_path, show)


def plot_variance_bar(per_sc_std: np.ndarray, out_path: str, sc_axis=None,
                       title="Per-Subcarrier Amplitude Std Dev", show=False):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = sc_axis if sc_axis is not None else np.arange(len(per_sc_std))
    ax.bar(x, per_sc_std, width=1.0)
    ax.set_xlabel("Subcarrier")
    ax.set_ylabel("Std Dev of Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _finish(fig, out_path, show)


def plot_pipeline_stages(raw_signal: np.ndarray, hampel_signal: np.ndarray,
                          bandpass_signal: np.ndarray, detrended_signal: np.ndarray,
                          normalized_signal: np.ndarray, out_path: str,
                          x_axis=None, title="Pipeline stages for one subcarrier", show=False):
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    x = x_axis if x_axis is not None else np.arange(len(raw_signal))

    signals = [
        (raw_signal, "raw"),
        (hampel_signal, "Hampel"),
        (bandpass_signal, "Butterworth bandpass"),
        (detrended_signal, "DC drift removed"),
        (normalized_signal, "normalized"),
    ]

    for ax, (signal, label) in zip(axes, signals):
        ax.plot(x, signal, linewidth=1.0)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)" if x_axis is not None else "Packet #")
    axes[0].set_title(title)
    _finish(fig, out_path, show)
