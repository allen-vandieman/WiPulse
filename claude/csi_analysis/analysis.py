"""
analysis.py
===========
Turns a list of decoded CSIFrame objects into a packet x subcarrier complex
matrix, then derives the standard quantities you need to sanity-check a CSI
capture and detect motion:

    - amplitude / phase
    - fftshifted (DC-centered) subcarrier ordering, matching plotcsi.m
    - subcarrier selection / pilot & null-subcarrier masking
    - per-subcarrier variance over time (stability / motion indicator)
    - FFT of the amplitude time-series (periodic-motion / vibration detector)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from decoder import CSIFrame

logger = logging.getLogger("csi_analysis.analysis")


@dataclass
class CSIDataset:
    """Assembled CSI capture, ready for analysis/plotting."""
    csi: np.ndarray          # complex128, shape (n_packets, nfft)
    ts: np.ndarray           # float64, shape (n_packets,) capture timestamps
    seq_num: np.ndarray      # uint16, shape (n_packets,)
    nfft: int
    fs_est: float             # estimated sample rate (packets/sec), from ts

    @property
    def n_packets(self) -> int:
        return self.csi.shape[0]


def build_dataset(frames: List[CSIFrame]) -> CSIDataset:
    """Stack a list of CSIFrame into a single CSIDataset (packet x subcarrier)."""
    if not frames:
        raise ValueError("No CSI frames to build a dataset from (empty capture, "
                          "or everything failed to decode -- check chip/bw args).")

    nfft = frames[0].csi.shape[0]
    for f in frames:
        if f.csi.shape[0] != nfft:
            raise ValueError(
                f"Inconsistent nfft across frames: frame {f.index} has "
                f"{f.csi.shape[0]}, expected {nfft}. Capture may mix "
                f"bandwidths/chips."
            )

    csi = np.stack([f.csi for f in frames], axis=0)
    ts = np.array([f.ts for f in frames], dtype=np.float64)
    seq_num = np.array([f.seq_num for f in frames], dtype=np.uint32)

    duration = ts[-1] - ts[0] if len(ts) > 1 else 0.0
    fs_est = (len(ts) - 1) / duration if duration > 0 else 0.0

    logger.info("Built dataset: %d packets x %d subcarriers, est. rate %.2f pkt/s",
                csi.shape[0], nfft, fs_est)

    return CSIDataset(csi=csi, ts=ts, seq_num=seq_num, nfft=nfft, fs_est=fs_est)


def report_sequence_gaps(dataset: CSIDataset) -> dict:
    """
    Quick packet-loss check using the Wi-Fi frame sequence numbers embedded
    in the CSI header. Sequence numbers wrap at 4096 (12-bit 802.11 seqnum),
    so wraps are handled but not treated as loss.
    """
    seq = dataset.seq_num.astype(np.int64)
    diffs = np.diff(seq)
    # normalize wraparound (802.11 seq numbers are 12-bit, wrap at 4096)
    diffs = np.where(diffs < -2048, diffs + 4096, diffs)
    lost = int(np.sum(diffs[diffs > 1] - 1))
    duplicates_or_reorder = int(np.sum(diffs <= 0))
    return {
        "n_packets": dataset.n_packets,
        "estimated_lost": lost,
        "duplicate_or_reordered": duplicates_or_reorder,
    }


def fftshift_subcarriers(csi: np.ndarray) -> np.ndarray:
    """DC-center the subcarrier axis (axis=-1), matching plotcsi.m's fftshift."""
    return np.fft.fftshift(csi, axes=-1)


def subcarrier_axis(nfft: int) -> np.ndarray:
    """Centered subcarrier index axis, e.g. -128..127 for nfft=256."""
    return np.arange(-(nfft // 2), nfft - nfft // 2)


def amplitude(csi: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Magnitude of complex CSI. normalize scales each packet row to max=1."""
    amp = np.abs(csi)
    if normalize:
        row_max = np.max(amp, axis=-1, keepdims=True)
        row_max[row_max == 0] = 1.0
        amp = amp / row_max
    return amp


def phase(csi: np.ndarray, unwrap: bool = False) -> np.ndarray:
    """Phase (radians) of complex CSI. unwrap applies np.unwrap along subcarriers."""
    ph = np.angle(csi)
    if unwrap:
        ph = np.unwrap(ph, axis=-1)
    return ph


def select_subcarriers(csi: np.ndarray, indices: Optional[Sequence[int]] = None,
                        exclude_dc_guard: bool = False, nfft: Optional[int] = None
                        ) -> np.ndarray:
    """
    Select a subset of subcarriers (axis=-1).

    indices: explicit list/array of column indices into csi's *current*
        subcarrier axis (e.g. after fftshift, these are contiguous around
        DC). If None, all subcarriers are kept (subject to exclude_dc_guard).
    exclude_dc_guard: zero out a small band around DC and the two edges,
        matching common 802.11ac null-subcarrier conventions. Only applied
        when indices is None. Requires nfft.
    """
    if indices is not None:
        return csi[..., list(indices)]

    if exclude_dc_guard:
        if nfft is None:
            nfft = csi.shape[-1]
        mask = np.ones(nfft, dtype=bool)
        center = nfft // 2
        mask[center - 1:center + 2] = False   # DC + immediate neighbors
        mask[:2] = False                       # low guard band
        mask[-2:] = False                      # high guard band
        return csi[..., mask]

    return csi


def variance_analysis(amp: np.ndarray) -> dict:
    """
    Per-subcarrier variance/std over time (axis=0), plus a single scalar
    "motion score" (mean of per-subcarrier std, normalized by mean amplitude).
    A stationary/empty scene should give a low, stable score; movement or a
    new obstruction should visibly raise it.
    """
    per_sc_mean = np.mean(amp, axis=0)
    per_sc_std = np.std(amp, axis=0)
    per_sc_var = np.var(amp, axis=0)

    mean_amp = np.mean(amp)
    motion_score = float(np.mean(per_sc_std) / mean_amp) if mean_amp > 0 else 0.0

    # per-packet aggregate (mean amplitude across subcarriers, per packet) --
    # useful for a coarse "did anything change over time" time-series
    per_packet_mean = np.mean(amp, axis=1)

    return {
        "per_subcarrier_mean": per_sc_mean,
        "per_subcarrier_std": per_sc_std,
        "per_subcarrier_var": per_sc_var,
        "per_packet_mean_amplitude": per_packet_mean,
        "motion_score": motion_score,
    }


def fft_motion_detection(amp: np.ndarray, fs: float, subcarrier_idx: Optional[int] = None,
                          detrend: bool = True) -> dict:
    """
    FFT of an amplitude time-series to look for periodic motion (walking,
    breathing, swinging obstacles, etc).

    subcarrier_idx: which subcarrier's time-series to analyze. If None,
        uses the mean amplitude across all subcarriers per packet (more
        robust to picking an unlucky/nulled subcarrier).
    fs: sample rate in Hz (packets/sec) -- use dataset.fs_est or your
        known TX rate.

    Returns freqs (Hz, positive half only) and magnitude spectrum, plus the
    dominant non-DC frequency found (a strong dominant peak around
    human-motion range, ~0.1-2 Hz for walking/breathing, indicates movement
    is showing up in the CSI).
    """
    if fs <= 0:
        raise ValueError("fs must be > 0 to compute an FFT motion spectrum "
                          "(pass dataset.fs_est or your known capture rate)")

    if subcarrier_idx is None:
        sig = np.mean(amp, axis=1)
    else:
        sig = amp[:, subcarrier_idx]

    sig = sig.astype(np.float64)
    if detrend:
        sig = sig - np.mean(sig)

    n = len(sig)
    window = np.hanning(n)
    spec = np.fft.rfft(sig * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(spec)

    # dominant frequency, ignoring DC bin
    if len(mag) > 1:
        dom_idx = 1 + int(np.argmax(mag[1:]))
        dominant_freq = float(freqs[dom_idx])
        dominant_mag = float(mag[dom_idx])
    else:
        dominant_freq, dominant_mag = 0.0, 0.0

    return {
        "freqs": freqs,
        "magnitude": mag,
        "dominant_freq_hz": dominant_freq,
        "dominant_magnitude": dominant_mag,
    }
