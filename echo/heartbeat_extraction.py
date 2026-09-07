#!/usr/bin/env python3

"""
heartbeat_extraction.py
=======================

Heartbeat extraction for CSI-Heart.

This module deliberately contains TWO processing paths:

1. OFFLINE PATH
   run_full_analysis()
       - full analysis
       - plots
       - spectrogram
       - peak analysis
       - result files

2. LIVE PATH
   run_live_analysis()
       - no plotting
       - no disk I/O
       - lightweight rolling-window processing
       - suitable for a background worker
       - adaptive sampling-rate estimation
       - occupancy/motion/quality assessment
       - quality-weighted subcarrier fusion
       - Welch PSD
       - temporal HR stabilization

The live path must NEVER be called from the packet-reception
callback. It is designed to run in a separate worker thread.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

class HeartbeatConfig:
    """Configuration shared by offline and live heartbeat processing."""

    def __init__(self):

        # --------------------------------------------------------------
        # Heartbeat band
        # --------------------------------------------------------------

        self.bandpass_low = 0.7
        self.bandpass_high = 2.5

        self.filter_order = 4

        self.heart_rate_min = 42.0
        self.heart_rate_max = 150.0

        # --------------------------------------------------------------
        # Sampling
        # --------------------------------------------------------------

        self.sampling_rate = 100.0

        # Actual live sampling rate may be estimated from timestamps.
        self.min_sampling_rate = 50.0
        self.max_sampling_rate = 150.0

        # --------------------------------------------------------------
        # Live analysis
        # --------------------------------------------------------------

        self.live_window_seconds = 30.0
        self.live_update_seconds = 2.0

        # Minimum data before processing
        self.live_min_seconds = 15.0

        # --------------------------------------------------------------
        # Subcarrier selection
        # --------------------------------------------------------------

        self.use_all_subcarriers = False

        # Old implementation selected roughly 80% of all SCs.
        # That is too many for the live heartbeat signal.
        self.auto_select_percentile = 20.0

        # Preferred direct limit.
        self.live_max_subcarriers = 12

        # Minimum number if sufficient quality exists.
        self.live_min_subcarriers = 3

        # --------------------------------------------------------------
        # Occupancy
        # --------------------------------------------------------------

        self.occupancy_energy_threshold = 0.05
        self.occupancy_diversity_threshold = 0.005

        # IMPORTANT:
        # Motion is NOT required for occupancy.
        #
        # A stationary seated person must still count as occupied.
        self.occupancy_enter_count = 2
        self.occupancy_exit_count = 4

        # --------------------------------------------------------------
        # Motion
        # --------------------------------------------------------------

        self.motion_window_seconds = 3.0

        # Motion ratio relative to slowly adapting baseline.
        self.motion_active_ratio = 1.8
        self.motion_high_ratio = 3.0

        # --------------------------------------------------------------
        # Signal quality
        # --------------------------------------------------------------

        self.min_snr = 1.5
        self.min_signal_quality = 0.20

        self.min_peak_prominence = 0.0

        # --------------------------------------------------------------
        # HR stabilization
        # --------------------------------------------------------------

        self.max_hr_change = 25.0

        self.hr_history_length = 5

        # Maximum change allowed before a candidate is treated as
        # suspicious. It can still be accepted if it remains stable.
        self.hr_stability_tolerance = 8.0

        # --------------------------------------------------------------
        # PSD
        # --------------------------------------------------------------

        self.psd_segment_seconds = 8.0

        # --------------------------------------------------------------
        # Offline analysis
        # --------------------------------------------------------------

        self.segment_duration = 10.0
        self.overlap = 0.5

        # --------------------------------------------------------------
        # Visualization
        # --------------------------------------------------------------

        self.show_spectrogram = True
        self.show_heartbeat_waveform = True
        self.show_peak_detection = True
        self.max_subcarriers_to_plot = 10


# ======================================================================
# HEARTBEAT EXTRACTOR
# ======================================================================

class HeartbeatExtractor:
    """
    CSI heartbeat extractor.

    The object is intended to be owned by the heartbeat worker thread
    in live mode. The packet-reception thread should never call the
    expensive methods in this class.
    """

    def __init__(self, config: Optional[HeartbeatConfig] = None):

        self.config = config if config is not None else HeartbeatConfig()

        # --------------------------------------------------------------
        # Offline state
        # --------------------------------------------------------------

        self.filtered_data = None
        self.heartbeat_signal = None

        self.spectrogram = None
        self.frequencies = None
        self.times = None

        self.heart_rate_history = []

        # --------------------------------------------------------------
        # Live state
        # --------------------------------------------------------------

        self.occupancy = False
        self.motion_level = 0.0
        self.motion_class = "unknown"

        self.signal_quality = 0.0
        self.confidence = 0.0

        self.last_valid_bpm: Optional[float] = None
        self.last_candidate_bpm: Optional[float] = None

        self.last_live_result: Optional[Dict] = None

        self.hr_history = deque(
            maxlen=self.config.hr_history_length
        )

        self.motion_history = deque(maxlen=100)

        self.amp_history = deque(maxlen=100)

        self.baseline_mean = None
        self.baseline_std = None

        self.occupancy_enter_counter = 0
        self.occupancy_exit_counter = 0

        self.selected_subcarriers = np.array([], dtype=int)

    # ==================================================================
    # FILTER
    # ==================================================================

    def design_bandpass_filter(
        self,
        sampling_rate: Optional[float] = None,
    ):
        """Design Butterworth heartbeat bandpass filter."""

        fs = (
            sampling_rate
            if sampling_rate is not None
            else self.config.sampling_rate
        )

        nyquist = fs / 2.0

        low = max(
            0.001,
            self.config.bandpass_low / nyquist,
        )

        high = min(
            0.999,
            self.config.bandpass_high / nyquist,
        )

        if high <= low:
            raise ValueError(
                f"Invalid heartbeat band "
                f"{self.config.bandpass_low}-"
                f"{self.config.bandpass_high} Hz "
                f"for sampling rate {fs:.2f} Hz"
            )

        return signal.butter(
            self.config.filter_order,
            [low, high],
            btype="band",
            output="sos",
        )

    # ==================================================================
    # SAMPLING RATE
    # ==================================================================

    def estimate_sampling_rate(
        self,
        timestamps: Optional[Sequence[float]],
    ) -> float:
        """
        Estimate CSI sampling rate from packet timestamps.

        Robust against occasional packet gaps by using the median
        inter-packet interval.
        """

        if timestamps is None:
            return float(self.config.sampling_rate)

        ts = np.asarray(timestamps, dtype=np.float64)

        if len(ts) < 10:
            return float(self.config.sampling_rate)

        dt = np.diff(ts)

        # Remove invalid intervals.
        dt = dt[np.isfinite(dt) & (dt > 0)]

        if len(dt) < 5:
            return float(self.config.sampling_rate)

        # Reject extreme packet gaps.
        median_dt = np.median(dt)

        good = dt < median_dt * 5.0

        dt_good = dt[good]

        if len(dt_good) < 5:
            dt_good = dt

        robust_dt = np.median(dt_good)

        if robust_dt <= 0:
            return float(self.config.sampling_rate)

        fs = 1.0 / robust_dt

        fs = np.clip(
            fs,
            self.config.min_sampling_rate,
            self.config.max_sampling_rate,
        )

        return float(fs)

    # ==================================================================
    # OCCUPANCY + MOTION
    # ==================================================================

    def detect_occupancy_and_motion(
        self,
        csi_data,
    ) -> Tuple[bool, float, float]:
        """
        Detect occupancy and motion.

        Occupancy is intentionally NOT dependent on motion.

        Returns:
            occupied
            motion_ratio
            energy
        """

        if csi_data is None:
            return False, 0.0, 0.0

        if len(csi_data) < 10:
            return False, 0.0, 0.0

        amps = np.abs(csi_data)

        mean_amp = float(np.mean(amps))

        if mean_amp <= 0:
            return False, 0.0, 0.0

        # Robust power normalization.
        median_power = float(
            np.median(amps ** 2)
        )

        percentile_power = float(
            np.percentile(amps ** 2, 90)
        )

        energy = (
            median_power
            / (percentile_power + 1e-12)
        )

        energy = float(
            np.clip(energy, 0.0, 1.0)
        )

        # Subcarrier diversity.
        sc_variance = np.var(
            amps,
            axis=1,
        )

        sc_diversity = float(
            np.mean(sc_variance)
            / (mean_amp ** 2 + 1e-12)
        )

        sc_diversity = float(
            np.clip(sc_diversity, 0.0, 1.0)
        )

        # Temporal variation.
        current_std = float(
            np.std(amps)
        )

        # Initialize baseline.
        if self.baseline_std is None:

            self.baseline_std = max(
                current_std,
                1e-9,
            )

            self.baseline_mean = mean_amp

        else:

            # Slow adaptation.
            self.baseline_std = (
                0.98 * self.baseline_std
                + 0.02 * current_std
            )

            self.baseline_mean = (
                0.98 * self.baseline_mean
                + 0.02 * mean_amp
            )

        baseline_std = max(
            self.baseline_std,
            1e-9,
        )

        motion_ratio = (
            current_std
            / baseline_std
        )

        motion_ratio = float(
            np.clip(
                motion_ratio,
                0.0,
                10.0,
            )
        )

        # --------------------------------------------------------------
        # Occupancy
        # --------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # We only require signal energy/diversity.
        # Motion is a separate property.
        #

        energy_ok = (
            energy
            >= self.config.occupancy_energy_threshold
        )

        diversity_ok = (
            sc_diversity
            >= self.config.occupancy_diversity_threshold
        )

        occupancy_candidate = (
            energy_ok
            and diversity_ok
        )

        if occupancy_candidate:

            self.occupancy_enter_counter += 1
            self.occupancy_exit_counter = 0

            if (
                not self.occupancy
                and self.occupancy_enter_counter
                >= self.config.occupancy_enter_count
            ):
                self.occupancy = True

        else:

            self.occupancy_exit_counter += 1
            self.occupancy_enter_counter = 0

            if (
                self.occupancy
                and self.occupancy_exit_counter
                >= self.config.occupancy_exit_count
            ):
                self.occupancy = False

        self.motion_history.append(
            motion_ratio
        )

        self.amp_history.append(
            mean_amp
        )

        self.motion_level = motion_ratio
        self.motion_class = (
            self.get_motion_classification(
                motion_ratio
            )
        )

        return (
            self.occupancy,
            motion_ratio,
            energy,
        )

    # ==================================================================
    # MOTION CLASSIFICATION
    # ==================================================================

    def get_motion_classification(
        self,
        motion_score: float,
    ) -> str:

        if motion_score < 1.2:
            return "static"

        if motion_score < self.config.motion_active_ratio:
            return "subtle"

        if motion_score < self.config.motion_high_ratio:
            return "active"

        return "high"

    def get_motion_trend(
        self,
        window: int = 20,
    ) -> np.ndarray:

        if not self.motion_history:
            return np.array([])

        return np.asarray(
            list(self.motion_history)[-window:],
            dtype=float,
        )

    # ==================================================================
    # SUBCARRIER QUALITY
    # ==================================================================

    def score_subcarriers(
        self,
        csi_data,
        sampling_rate: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score subcarriers for heartbeat sensing.

        Score combines:
            - heartbeat-band power
            - relative variation
            - noise estimate

        Returns:
            indices
            normalized weights
        """

        if csi_data is None:
            return (
                np.array([], dtype=int),
                np.array([], dtype=float),
            )

        if len(csi_data) < 20:
            return (
                np.arange(
                    csi_data.shape[1],
                    dtype=int,
                ),
                np.ones(
                    csi_data.shape[1],
                    dtype=float,
                ),
            )

        fs = (
            sampling_rate
            if sampling_rate is not None
            else self.config.sampling_rate
        )

        amps = np.abs(csi_data)

        n_samples, n_sc = amps.shape

        # Heartbeat band.
        nperseg = min(
            n_samples,
            max(
                128,
                int(fs * 8.0),
            ),
        )

        scores = np.zeros(
            n_sc,
            dtype=float,
        )

        for sc in range(n_sc):

            x = amps[:, sc].astype(
                np.float64,
                copy=False,
            )

            mean_x = float(
                np.mean(x)
            )

            if mean_x <= 1e-12:
                scores[sc] = -np.inf
                continue

            # Remove DC.
            x = x - np.mean(x)

            variance = float(
                np.var(x)
            )

            if variance <= 1e-12:
                scores[sc] = 0.0
                continue

            # Difference-based noise estimate.
            diff_std = float(
                np.std(np.diff(x))
            )

            noise_power = (
                diff_std ** 2 / 2.0
            )

            total_power = variance

            rough_snr = (
                total_power
                / (noise_power + 1e-12)
            )

            # Heartbeat-band power.
            try:

                freqs, psd = signal.welch(
                    x,
                    fs=fs,
                    window="hann",
                    nperseg=nperseg,
                    noverlap=nperseg // 2,
                    scaling="density",
                )

                band = (
                    (freqs >= self.config.bandpass_low)
                    &
                    (freqs <= self.config.bandpass_high)
                )

                if np.any(band):

                    band_power = float(
                        np.trapezoid(
                            psd[band],
                            freqs[band],
                        )
                    )

                else:
                    band_power = 0.0

            except Exception:
                band_power = 0.0

            relative_variation = (
                np.std(x)
                / (mean_x + 1e-12)
            )

            score = (
                np.log1p(
                    max(rough_snr, 0.0)
                )
                * np.sqrt(
                    max(band_power, 0.0)
                    + 1e-12
                )
                * max(
                    relative_variation,
                    1e-6,
                )
            )

            scores[sc] = score

        valid = np.isfinite(scores)

        if not np.any(valid):

            indices = np.arange(
                min(
                    n_sc,
                    self.config.live_max_subcarriers,
                ),
                dtype=int,
            )

            return (
                indices,
                np.ones(
                    len(indices),
                    dtype=float,
                ),
            )

        valid_indices = np.where(valid)[0]

        sorted_indices = valid_indices[
            np.argsort(
                scores[valid],
            )[::-1]
        ]

        if self.config.use_all_subcarriers:

            selected = sorted_indices

        else:

            percentile_n = max(
                1,
                int(
                    n_sc
                    * self.config.auto_select_percentile
                    / 100.0
                ),
            )

            n_select = min(
                percentile_n,
                self.config.live_max_subcarriers,
            )

            n_select = max(
                n_select,
                min(
                    self.config.live_min_subcarriers,
                    len(sorted_indices),
                ),
            )

            selected = sorted_indices[
                :n_select
            ]

        selected_scores = np.maximum(
            scores[selected],
            0.0,
        )

        if np.sum(selected_scores) <= 0:

            weights = np.ones(
                len(selected),
                dtype=float,
            )

        else:

            weights = (
                selected_scores
                / np.sum(selected_scores)
            )

        return (
            selected.astype(int),
            weights.astype(float),
        )

    # ==================================================================
    # QUALITY ASSESSMENT
    # ==================================================================

    def assess_signal_quality(
        self,
        signal_data,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate PSD-based heartbeat quality.
        """

        if signal_data is None:
            return {
                "snr": 0.0,
                "prominence": 0.0,
                "quality": 0.0,
            }

        if len(signal_data) < 32:
            return {
                "snr": 0.0,
                "prominence": 0.0,
                "quality": 0.0,
            }

        fs = (
            sampling_rate
            if sampling_rate is not None
            else self.config.sampling_rate
        )

        nperseg = min(
            len(signal_data),
            max(
                128,
                int(
                    fs
                    * self.config.psd_segment_seconds
                ),
            ),
        )

        if nperseg < 32:
            return {
                "snr": 0.0,
                "prominence": 0.0,
                "quality": 0.0,
            }

        freqs, psd = signal.welch(
            signal_data,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling="density",
        )

        hr_min_hz = (
            self.config.heart_rate_min
            / 60.0
        )

        hr_max_hz = (
            self.config.heart_rate_max
            / 60.0
        )

        band = (
            (freqs >= hr_min_hz)
            &
            (freqs <= hr_max_hz)
        )

        if not np.any(band):
            return {
                "snr": 0.0,
                "prominence": 0.0,
                "quality": 0.0,
            }

        band_freqs = freqs[band]
        band_psd = psd[band]

        if len(band_psd) < 3:
            return {
                "snr": 0.0,
                "prominence": 0.0,
                "quality": 0.0,
            }

        peaks, properties = signal.find_peaks(
            band_psd,
            prominence=max(
                np.std(band_psd) * 0.15,
                1e-15,
            ),
        )

        if len(peaks) == 0:

            # Still find dominant frequency.
            best_idx = int(
                np.argmax(band_psd)
            )

            peak_power = float(
                band_psd[best_idx]
            )

            peak_prominence = 0.0

        else:

            prominence_values = properties[
                "prominences"
            ]

            best_peak = int(
                peaks[
                    np.argmax(
                        prominence_values
                    )
                ]
            )

            best_idx = best_peak

            peak_power = float(
                band_psd[best_idx]
            )

            peak_prominence = float(
                prominence_values[
                    np.argmax(
                        prominence_values
                    )
                ]
            )

        median_power = float(
            np.median(band_psd)
        )

        snr = (
            peak_power
            / (median_power + 1e-12)
        )

        snr = float(
            np.clip(
                snr,
                0.0,
                50.0,
            )
        )

        # Normalize 1..10+ approximately into 0..1.
        snr_quality = np.clip(
            (snr - 1.0) / 9.0,
            0.0,
            1.0,
        )

        # Prominence relative to PSD standard deviation.
        psd_std = float(
            np.std(band_psd)
        )

        if psd_std > 0:

            prominence_quality = np.clip(
                peak_prominence
                / (psd_std * 3.0),
                0.0,
                1.0,
            )

        else:
            prominence_quality = 0.0

        quality = (
            0.7 * snr_quality
            + 0.3 * prominence_quality
        )

        return {
            "snr": snr,
            "prominence": peak_prominence,
            "quality": float(
                np.clip(
                    quality,
                    0.0,
                    1.0,
                )
            ),
            "peak_freq": float(
                band_freqs[best_idx]
            ),
        }

    # ==================================================================
    # LIVE SIGNAL EXTRACTION
    # ==================================================================

    def extract_live_signal(
        self,
        csi_data,
        sampling_rate: Optional[float] = None,
    ):
        """
        Extract heartbeat waveform from live CSI.

        Steps:

            CSI magnitude
                ↓
            best subcarriers
                ↓
            detrend
                ↓
            Hampel-like median cleanup
                ↓
            bandpass
                ↓
            weighted subcarrier fusion
                ↓
            normalization
        """

        if csi_data is None:
            return (
                None,
                np.array([], dtype=int),
                np.array([], dtype=float),
            )

        if len(csi_data) < 32:
            return (
                None,
                np.array([], dtype=int),
                np.array([], dtype=float),
            )

        fs = (
            sampling_rate
            if sampling_rate is not None
            else self.config.sampling_rate
        )

        selected, weights = (
            self.score_subcarriers(
                csi_data,
                fs,
            )
        )

        if len(selected) == 0:
            return (
                None,
                selected,
                weights,
            )

        amplitudes = np.abs(
            csi_data[:, selected]
        ).astype(
            np.float64,
            copy=False,
        )

        # --------------------------------------------------------------
        # Per-subcarrier detrending
        # --------------------------------------------------------------

        cleaned = np.zeros_like(
            amplitudes,
            dtype=np.float64,
        )

        for i in range(
            amplitudes.shape[1]
        ):

            x = amplitudes[:, i]

            # Remove slow drift.
            x = signal.detrend(
                x,
                type="linear",
            )

            # Robust median outlier suppression.
            med = signal.medfilt(
                x,
                kernel_size=5,
            )

            residual = x - med

            mad = np.median(
                np.abs(
                    residual
                )
            )

            threshold = (
                6.0 * mad
            )

            if threshold > 0:

                x = np.where(
                    np.abs(residual)
                    > threshold,
                    med,
                    x,
                )

            cleaned[:, i] = x

        # --------------------------------------------------------------
        # Bandpass
        # --------------------------------------------------------------

        sos = (
            self.design_bandpass_filter(
                fs
            )
        )

        filtered = np.zeros_like(
            cleaned
        )

        for i in range(
            cleaned.shape[1]
        ):

            x = cleaned[:, i]

            try:

                filtered[:, i] = (
                    signal.sosfiltfilt(
                        sos,
                        x,
                    )
                )

            except ValueError:

                # Too little data for filtfilt.
                filtered[:, i] = x

        # --------------------------------------------------------------
        # Normalize each selected SC before fusion.
        # This prevents a large-amplitude SC from dominating.
        # --------------------------------------------------------------

        normalized = np.zeros_like(
            filtered
        )

        valid_weights = np.zeros(
            len(selected),
            dtype=float,
        )

        for i in range(
            filtered.shape[1]
        ):

            x = filtered[:, i]

            std = float(
                np.std(x)
            )

            if std <= 1e-12:
                continue

            normalized[:, i] = (
                x / std
            )

            valid_weights[i] = (
                weights[i]
            )

        if np.sum(valid_weights) <= 0:

            return (
                None,
                selected,
                weights,
            )

        valid_weights /= np.sum(
            valid_weights
        )

        # --------------------------------------------------------------
        # Weighted fusion
        # --------------------------------------------------------------

        heartbeat = np.sum(
            normalized
            * valid_weights[None, :],
            axis=1,
        )

        heartbeat = signal.detrend(
            heartbeat,
            type="constant",
        )

        std = float(
            np.std(heartbeat)
        )

        if std > 1e-12:
            heartbeat /= std

        self.filtered_data = (
            normalized
        )

        self.heartbeat_signal = (
            heartbeat
        )

        self.selected_subcarriers = (
            selected
        )

        return (
            heartbeat,
            selected,
            valid_weights,
        )

    # ==================================================================
    # LIVE HR ESTIMATION
    # ==================================================================

    def estimate_live_hr(
        self,
        heartbeat_signal,
        sampling_rate: Optional[float] = None,
    ) -> Dict:
        """
        Estimate HR using Welch PSD.

        Returns candidate BPM plus quality/confidence metrics.
        """

        if heartbeat_signal is None:

            return {
                "bpm": None,
                "candidate_bpm": None,
                "confidence": 0.0,
                "quality": 0.0,
                "snr": 0.0,
                "peak_prominence": 0.0,
                "peak_freq": None,
                "stable": False,
            }

        fs = (
            sampling_rate
            if sampling_rate is not None
            else self.config.sampling_rate
        )

        n = len(
            heartbeat_signal
        )

        if n < int(fs * 10):

            return {
                "bpm": None,
                "candidate_bpm": None,
                "confidence": 0.0,
                "quality": 0.0,
                "snr": 0.0,
                "peak_prominence": 0.0,
                "peak_freq": None,
                "stable": False,
            }

        nperseg = min(
            n,
            max(
                256,
                int(
                    fs
                    * self.config.psd_segment_seconds
                ),
            ),
        )

        noverlap = nperseg // 2

        freqs, psd = signal.welch(
            heartbeat_signal,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
        )

        hr_min_hz = (
            self.config.heart_rate_min
            / 60.0
        )

        hr_max_hz = (
            self.config.heart_rate_max
            / 60.0
        )

        band = (
            (freqs >= hr_min_hz)
            &
            (freqs <= hr_max_hz)
        )

        if not np.any(band):

            return {
                "bpm": None,
                "candidate_bpm": None,
                "confidence": 0.0,
                "quality": 0.0,
                "snr": 0.0,
                "peak_prominence": 0.0,
                "peak_freq": None,
                "stable": False,
            }

        band_freqs = freqs[band]
        band_psd = psd[band]

        # --------------------------------------------------------------
        # Find candidate peaks.
        # --------------------------------------------------------------

        prominence_threshold = max(
            np.std(band_psd) * 0.10,
            1e-15,
        )

        peaks, properties = signal.find_peaks(
            band_psd,
            prominence=prominence_threshold,
        )

        if len(peaks) == 0:

            best_idx = int(
                np.argmax(band_psd)
            )

            peak_prominence = 0.0

        else:

            best_prominence_idx = int(
                np.argmax(
                    properties["prominences"]
                )
            )

            best_idx = int(
                peaks[
                    best_prominence_idx
                ]
            )

            peak_prominence = float(
                properties[
                    "prominences"
                ][
                    best_prominence_idx
                ]
            )

        peak_freq = float(
            band_freqs[best_idx]
        )

        candidate_bpm = (
            peak_freq * 60.0
        )

        peak_power = float(
            band_psd[best_idx]
        )

        median_power = float(
            np.median(band_psd)
        )

        snr = (
            peak_power
            / (median_power + 1e-12)
        )

        snr = float(
            np.clip(
                snr,
                0.0,
                50.0,
            )
        )

        # --------------------------------------------------------------
        # Peak dominance.
        # --------------------------------------------------------------

        mean_power = float(
            np.mean(band_psd)
        )

        dominance = (
            peak_power
            / (mean_power + 1e-12)
        )

        dominance_score = np.clip(
            (dominance - 1.0) / 5.0,
            0.0,
            1.0,
        )

        snr_score = np.clip(
            (snr - 1.0) / 9.0,
            0.0,
            1.0,
        )

        # Quality of raw extracted waveform.
        waveform_std = float(
            np.std(
                heartbeat_signal
            )
        )

        waveform_quality = np.clip(
            waveform_std / 0.5,
            0.0,
            1.0,
        )

        quality = (
            0.55 * snr_score
            + 0.45 * dominance_score
        )

        quality = float(
            np.clip(
                quality,
                0.0,
                1.0,
            )
        )

        # --------------------------------------------------------------
        # Candidate stability against previous HR.
        # --------------------------------------------------------------

        stable = True

        if self.last_valid_bpm is not None:

            delta = abs(
                candidate_bpm
                - self.last_valid_bpm
            )

            if (
                delta
                > self.config.max_hr_change
            ):

                stable = False

        # --------------------------------------------------------------
        # Confidence.
        # --------------------------------------------------------------

        confidence = (
            100.0
            * (
                0.55 * snr_score
                + 0.30 * dominance_score
                + 0.15 * waveform_quality
            )
        )

        if not stable:
            confidence *= 0.55

        confidence = float(
            np.clip(
                confidence,
                0.0,
                100.0,
            )
        )

        # --------------------------------------------------------------
        # Accept candidate.
        # --------------------------------------------------------------

        accepted = (
            quality
            >= self.config.min_signal_quality
            and snr
            >= self.config.min_snr
            and confidence >= 25.0
        )

        if accepted:

            self.last_candidate_bpm = (
                candidate_bpm
            )

            self.hr_history.append(
                candidate_bpm
            )

            # Median stabilization.
            history = np.asarray(
                self.hr_history,
                dtype=float,
            )

            if len(history) >= 2:

                bpm = float(
                    np.median(history)
                )

            else:

                bpm = float(
                    candidate_bpm
                )

            self.last_valid_bpm = (
                bpm
            )

        else:

            bpm = (
                self.last_valid_bpm
            )

        return {
            "bpm": bpm,
            "candidate_bpm": candidate_bpm,
            "confidence": confidence,
            "quality": quality,
            "snr": snr,
            "peak_prominence": peak_prominence,
            "peak_freq": peak_freq,
            "stable": stable,
            "accepted": accepted,
            "psd_freqs": freqs,
            "psd": psd,
        }

    # ==================================================================
    # LIVE ENTRY POINT
    # ==================================================================

    def run_live_analysis(
        self,
        csi_data,
        timestamps: Optional[Sequence[float]] = None,
    ) -> Dict:
        """
        Lightweight live heartbeat analysis.

        This function deliberately performs NO plotting and NO disk I/O.

        It is intended to run in the heartbeat worker thread.
        """

        if csi_data is None:

            return {
                "status": "no_data",
                "bpm": self.last_valid_bpm,
                "candidate_bpm": None,
                "confidence": 0.0,
                "quality": 0.0,
                "snr": 0.0,
                "occupancy": False,
                "motion": 0.0,
                "motion_class": "unknown",
                "energy": 0.0,
                "sampling_rate": self.config.sampling_rate,
                "selected_subcarriers": [],
            }

        csi_data = np.asarray(
            csi_data
        )

        if csi_data.ndim != 2:

            raise ValueError(
                "CSI data must have shape "
                "(samples, subcarriers)"
            )

        # --------------------------------------------------------------
        # Actual sample rate.
        # --------------------------------------------------------------

        fs = (
            self.estimate_sampling_rate(
                timestamps
            )
        )

        self.config.sampling_rate = fs

        # --------------------------------------------------------------
        # Occupancy / motion.
        # --------------------------------------------------------------

        (
            occupied,
            motion,
            energy,
        ) = self.detect_occupancy_and_motion(
            csi_data
        )

        base_result = {
            "status": "warming_up",
            "bpm": self.last_valid_bpm,
            "candidate_bpm": None,
            "confidence": 0.0,
            "quality": 0.0,
            "snr": 0.0,
            "occupancy": occupied,
            "motion": motion,
            "motion_class": self.motion_class,
            "energy": energy,
            "sampling_rate": fs,
            "selected_subcarriers": [],
            "processing_samples": len(
                csi_data
            ),
        }

        min_samples = int(
            fs
            * self.config.live_min_seconds
        )

        if len(csi_data) < min_samples:

            self.signal_quality = 0.0
            self.confidence = 0.0

            self.last_live_result = (
                base_result
            )

            return base_result

        # --------------------------------------------------------------
        # Do not attempt heartbeat when no subject is present.
        # --------------------------------------------------------------

        if not occupied:

            self.signal_quality = 0.0
            self.confidence = 0.0

            result = {
                **base_result,
                "status": "no_occupancy",
                "bpm": None,
            }

            self.last_live_result = result

            return result

        # --------------------------------------------------------------
        # Extract heartbeat.
        # --------------------------------------------------------------

        (
            heartbeat,
            selected,
            weights,
        ) = self.extract_live_signal(
            csi_data,
            fs,
        )

        if heartbeat is None:

            self.signal_quality = 0.0
            self.confidence = 0.0

            result = {
                **base_result,
                "status": "extraction_failed",
                "selected_subcarriers": (
                    selected.tolist()
                ),
            }

            self.last_live_result = result

            return result

        # --------------------------------------------------------------
        # HR.
        # --------------------------------------------------------------

        hr_result = self.estimate_live_hr(
            heartbeat,
            fs,
        )

        self.signal_quality = float(
            hr_result["quality"]
        )

        self.confidence = float(
            hr_result["confidence"]
        )

        status = (
            "valid"
            if hr_result.get(
                "accepted",
                False,
            )
            else "low_quality"
        )

        result = {
            **base_result,
            "status": status,
            "bpm": hr_result["bpm"],
            "candidate_bpm": (
                hr_result[
                    "candidate_bpm"
                ]
            ),
            "confidence": (
                hr_result[
                    "confidence"
                ]
            ),
            "quality": (
                hr_result["quality"]
            ),
            "snr": hr_result["snr"],
            "peak_prominence": (
                hr_result[
                    "peak_prominence"
                ]
            ),
            "peak_freq": (
                hr_result["peak_freq"]
            ),
            "stable": (
                hr_result["stable"]
            ),
            "accepted": (
                hr_result["accepted"]
            ),
            "selected_subcarriers": (
                selected.tolist()
            ),
            "subcarrier_weights": (
                weights.tolist()
            ),
            "processing_samples": len(
                csi_data
            ),
        }

        self.last_live_result = result

        return result

    # ==================================================================
    # OFFLINE PATH
    # ==================================================================

    def apply_bandpass_filter(
        self,
        csi_data,
        subcarrier_indices=None,
    ):
        """
        Offline-compatible bandpass method.

        This preserves the existing API used by the offline pipeline.
        """

        if subcarrier_indices is None:

            if self.config.use_all_subcarriers:

                subcarrier_indices = range(
                    csi_data.shape[1]
                )

            else:

                subcarrier_indices, _ = (
                    self.score_subcarriers(
                        csi_data,
                        self.config.sampling_rate,
                    )
                )

        subcarrier_indices = list(
            subcarrier_indices
        )

        sos = (
            self.design_bandpass_filter()
        )

        filtered_data = np.zeros(
            (
                len(csi_data),
                len(subcarrier_indices),
            ),
            dtype=float,
        )

        for idx, sc in enumerate(
            subcarrier_indices
        ):

            magnitude = np.abs(
                csi_data[:, sc]
            )

            filtered_data[:, idx] = (
                signal.sosfiltfilt(
                    sos,
                    magnitude,
                )
            )

        if filtered_data.shape[1] > 0:

            self.heartbeat_signal = (
                np.mean(
                    filtered_data,
                    axis=1,
                )
            )

        else:

            self.heartbeat_signal = (
                np.zeros(
                    len(csi_data)
                )
            )

        self.filtered_data = (
            filtered_data
        )

        return self.heartbeat_signal

    def estimate_heart_rate(
        self,
        signal_data,
        sampling_rate=None,
    ):

        if sampling_rate is None:
            sampling_rate = (
                self.config.sampling_rate
            )

        n_samples = len(
            signal_data
        )

        if n_samples < 32:

            return {
                "heart_rate": 0.0,
                "dominant_freq": 0.0,
                "confidence": 0.0,
                "fft_frequencies": np.array([]),
                "fft_magnitudes": np.array([]),
                "quality_metrics": {},
            }

        window = np.hanning(
            n_samples
        )

        windowed_signal = (
            signal_data
            * window
        )

        fft_data = fft(
            windowed_signal
        )

        freqs = fftfreq(
            n_samples,
            1.0 / sampling_rate,
        )

        positive = (
            (freqs > 0)
            &
            (freqs <= 5)
        )

        freqs_positive = (
            freqs[positive]
        )

        fft_positive = np.abs(
            fft_data[positive]
        )

        hr_min_hz = (
            self.config.heart_rate_min
            / 60.0
        )

        hr_max_hz = (
            self.config.heart_rate_max
            / 60.0
        )

        hr_range = (
            (freqs_positive >= hr_min_hz)
            &
            (freqs_positive <= hr_max_hz)
        )

        if not np.any(hr_range):

            return {
                "heart_rate": 0.0,
                "dominant_freq": 0.0,
                "confidence": 0.0,
                "fft_frequencies": (
                    freqs_positive
                ),
                "fft_magnitudes": (
                    fft_positive
                ),
                "quality_metrics": {},
            }

        hr_freqs = (
            freqs_positive[hr_range]
        )

        hr_fft = (
            fft_positive[hr_range]
        )

        peak_idx = int(
            np.argmax(hr_fft)
        )

        dominant_freq = float(
            hr_freqs[peak_idx]
        )

        heart_rate = (
            dominant_freq * 60.0
        )

        quality = (
            self.assess_signal_quality(
                signal_data,
                sampling_rate,
            )
        )

        confidence = float(
            np.clip(
                quality.get(
                    "quality",
                    0.0,
                )
                * 100.0,
                0.0,
                100.0,
            )
        )

        return {
            "heart_rate": heart_rate,
            "dominant_freq": dominant_freq,
            "confidence": confidence,
            "fft_frequencies": (
                freqs_positive
            ),
            "fft_magnitudes": (
                fft_positive
            ),
            "quality_metrics": quality,
        }

    def compute_spectrogram(
        self,
        signal_data,
        nperseg=None,
    ):

        if nperseg is None:

            nperseg = int(
                self.config.segment_duration
                * self.config.sampling_rate
            )

        nperseg = max(
            32,
            min(
                nperseg,
                len(signal_data),
            ),
        )

        if nperseg % 2 == 1:
            nperseg -= 1

        noverlap = int(
            nperseg
            * self.config.overlap
        )

        noverlap = min(
            noverlap,
            nperseg - 1,
        )

        freqs, times, Sxx = (
            signal.spectrogram(
                signal_data,
                fs=self.config.sampling_rate,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="density",
            )
        )

        self.spectrogram = Sxx
        self.frequencies = freqs
        self.times = times

        return (
            freqs,
            times,
            Sxx,
        )

    def detect_peaks(
        self,
        signal_data,
        min_distance=None,
    ):

        if min_distance is None:

            min_distance = max(
                1,
                int(
                    self.config.sampling_rate
                    * 60.0
                    / self.config.heart_rate_max
                ),
            )

        peaks, properties = (
            signal.find_peaks(
                signal_data,
                distance=min_distance,
                prominence=max(
                    np.std(signal_data)
                    * 0.5,
                    1e-12,
                ),
                height=(
                    np.mean(signal_data)
                    + np.std(signal_data)
                    * 0.3
                ),
            )
        )

        if len(peaks) > 1:

            intervals = (
                np.diff(peaks)
                / self.config.sampling_rate
            )

            heart_rates = (
                60.0
                / intervals
            )

            # Remove physiologically impossible intervals.
            valid = (
                (heart_rates
                 >= self.config.heart_rate_min)
                &
                (heart_rates
                 <= self.config.heart_rate_max)
            )

            heart_rates = (
                heart_rates[valid]
            )

            intervals = (
                intervals[valid]
            )

            avg_hr = (
                float(
                    np.median(
                        heart_rates
                    )
                )
                if len(heart_rates)
                else 0.0
            )

        else:

            intervals = np.array([])
            heart_rates = np.array([])
            avg_hr = 0.0

        return {
            "peaks": peaks,
            "intervals": intervals,
            "heart_rates": heart_rates,
            "avg_heart_rate": avg_hr,
        }

    # ==================================================================
    # OFFLINE FULL ANALYSIS
    # ==================================================================

    def run_full_analysis(
        self,
        csi_data,
        metadata=None,
        output_dir="heartbeat_analysis",
    ):

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if metadata and "rate" in metadata:

            self.config.sampling_rate = (
                float(metadata["rate"])
            )

        # Occupancy is useful for offline diagnostics,
        # but we do not require motion.
        occupancy, motion, energy = (
            self.detect_occupancy_and_motion(
                csi_data
            )
        )

        if not occupancy:

            return {
                "heartbeat_signal": None,
                "heart_rate": 0.0,
                "confidence": 0.0,
                "peaks": np.array([]),
                "heart_rates": np.array([]),
                "occupancy": False,
                "motion_level": motion,
            }

        self.apply_bandpass_filter(
            csi_data
        )

        hr_result = (
            self.estimate_heart_rate(
                self.heartbeat_signal
            )
        )

        self.compute_spectrogram(
            self.heartbeat_signal
        )

        peak_result = (
            self.detect_peaks(
                self.heartbeat_signal
            )
        )

        self.create_heartbeat_plots(
            output_dir,
            hr_result,
            peak_result,
        )

        self.save_results(
            output_dir,
            hr_result,
            peak_result,
        )

        return {
            "heartbeat_signal": (
                self.heartbeat_signal
            ),
            "heart_rate": (
                hr_result[
                    "heart_rate"
                ]
            ),
            "confidence": (
                hr_result[
                    "confidence"
                ]
            ),
            "peaks": (
                peak_result["peaks"]
            ),
            "heart_rates": (
                peak_result["heart_rates"]
            ),
            "occupancy": occupancy,
            "motion_level": motion,
            "energy": energy,
        }

    # ==================================================================
    # OFFLINE PLOTS
    # ==================================================================

    def create_heartbeat_plots(
        self,
        output_dir,
        hr_result,
        peak_result,
    ):

        if self.heartbeat_signal is None:
            return

        output_dir = Path(
            output_dir
        )

        fig = plt.figure(
            figsize=(16, 12)
        )

        # --------------------------------------------------------------
        # Waveform
        # --------------------------------------------------------------

        ax1 = plt.subplot(
            3,
            2,
            1,
        )

        t = (
            np.arange(
                len(
                    self.heartbeat_signal
                )
            )
            / self.config.sampling_rate
        )

        ax1.plot(
            t,
            self.heartbeat_signal,
            linewidth=1,
        )

        peaks = peak_result[
            "peaks"
        ]

        if len(peaks):

            ax1.plot(
                t[peaks],
                self.heartbeat_signal[
                    peaks
                ],
                "o",
                markersize=4,
                label="Peaks",
            )

            ax1.legend()

        ax1.set_xlabel(
            "Time (seconds)"
        )

        ax1.set_ylabel(
            "Amplitude"
        )

        ax1.set_title(
            "Heartbeat Signal"
        )

        ax1.grid(True)

        # --------------------------------------------------------------
        # FFT
        # --------------------------------------------------------------

        ax2 = plt.subplot(
            3,
            2,
            2,
        )

        ax2.plot(
            hr_result[
                "fft_frequencies"
            ],
            hr_result[
                "fft_magnitudes"
            ],
        )

        ax2.axvline(
            hr_result[
                "dominant_freq"
            ],
            linestyle="--",
            label=(
                f"{hr_result['heart_rate']:.1f} BPM"
            ),
        )

        ax2.set_xlim(
            0,
            5,
        )

        ax2.set_xlabel(
            "Frequency (Hz)"
        )

        ax2.set_ylabel(
            "Magnitude"
        )

        ax2.set_title(
            "Frequency Spectrum"
        )

        ax2.legend()

        ax2.grid(True)

        # --------------------------------------------------------------
        # Spectrogram
        # --------------------------------------------------------------

        ax3 = plt.subplot(
            3,
            2,
            3,
        )

        if (
            self.spectrogram is not None
            and self.frequencies is not None
            and self.times is not None
        ):

            ax3.pcolormesh(
                self.times,
                self.frequencies,
                self.spectrogram,
                shading="auto",
            )

            ax3.set_ylim(
                0,
                4,
            )

        ax3.set_xlabel(
            "Time (seconds)"
        )

        ax3.set_ylabel(
            "Frequency (Hz)"
        )

        ax3.set_title(
            "Spectrogram"
        )

        # --------------------------------------------------------------
        # PSD
        # --------------------------------------------------------------

        ax4 = plt.subplot(
            3,
            2,
            4,
        )

        freqs, psd = signal.welch(
            self.heartbeat_signal,
            fs=self.config.sampling_rate,
            window="hann",
            nperseg=min(
                len(
                    self.heartbeat_signal
                ),
                int(
                    self.config.sampling_rate
                    * 8
                ),
            ),
        )

        ax4.plot(
            freqs,
            psd,
        )

        ax4.axvspan(
            self.config.bandpass_low,
            self.config.bandpass_high,
            alpha=0.2,
        )

        ax4.set_xlim(
            0,
            4,
        )

        ax4.set_xlabel(
            "Frequency (Hz)"
        )

        ax4.set_ylabel(
            "PSD"
        )

        ax4.set_title(
            "Welch PSD"
        )

        ax4.grid(True)

        # --------------------------------------------------------------
        # Peak HR
        # --------------------------------------------------------------

        ax5 = plt.subplot(
            3,
            2,
            5,
        )

        if len(
            peak_result["heart_rates"]
        ):

            ax5.plot(
                peak_result[
                    "heart_rates"
                ],
                marker="o",
            )

        ax5.set_xlabel(
            "Beat index"
        )

        ax5.set_ylabel(
            "BPM"
        )

        ax5.set_title(
            "Beat-to-Beat HR"
        )

        ax5.grid(True)

        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------

        ax6 = plt.subplot(
            3,
            2,
            6,
        )

        ax6.axis(
            "off"
        )

        stats_text = f"""
HEARTBEAT ANALYSIS SUMMARY

Bandpass:
  {self.config.bandpass_low:.2f}-
  {self.config.bandpass_high:.2f} Hz

Sampling rate:
  {self.config.sampling_rate:.2f} Hz

Heart rate:
  {hr_result['heart_rate']:.1f} BPM

Confidence:
  {hr_result['confidence']:.1f}%

Peaks:
  {len(peak_result['peaks'])}

Peak HR:
  {peak_result['avg_heart_rate']:.1f} BPM

Duration:
  {len(self.heartbeat_signal) /
    self.config.sampling_rate:.1f} s
"""

        ax6.text(
            0.05,
            0.95,
            stats_text,
            transform=ax6.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.suptitle(
            "Heartbeat Extraction Analysis",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        filename = (
            output_dir
            / "heartbeat_analysis.png"
        )

        plt.savefig(
            filename,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

    def save_results(
        self,
        output_dir,
        hr_result,
        peak_result,
    ):

        output_dir = Path(
            output_dir
        )

        if self.heartbeat_signal is not None:

            np.savetxt(
                output_dir
                / "heartbeat_signal.txt",
                self.heartbeat_signal,
                header=(
                    "Heartbeat Signal "
                    f"(rate={self.config.sampling_rate:.4f}Hz)"
                ),
            )

        with open(
            output_dir
            / "heartbeat_summary.txt",
            "w",
        ) as f:

            f.write(
                "HEARTBEAT EXTRACTION SUMMARY\n"
            )

            f.write(
                "=" * 50
                + "\n\n"
            )

            f.write(
                f"Bandpass Range: "
                f"{self.config.bandpass_low:.2f}-"
                f"{self.config.bandpass_high:.2f} Hz\n"
            )

            f.write(
                f"Filter Order: "
                f"{self.config.filter_order}\n"
            )

            f.write(
                f"Sampling Rate: "
                f"{self.config.sampling_rate:.4f} Hz\n\n"
            )

            f.write(
                f"Heart Rate: "
                f"{hr_result['heart_rate']:.2f} BPM\n"
            )

            f.write(
                f"Confidence: "
                f"{hr_result['confidence']:.2f}%\n"
            )

            f.write(
                f"Peaks Detected: "
                f"{len(peak_result['peaks'])}\n"
            )

            f.write(
                f"Average Peak HR: "
                f"{peak_result['avg_heart_rate']:.2f} BPM\n"
            )


# ======================================================================
# EXISTING PIPELINE COMPATIBILITY
# ======================================================================

class HeartbeatFilter:
    """
    Compatibility wrapper for the existing CSI processing pipeline.
    """

    def __init__(
        self,
        low_freq=0.8,
        high_freq=2.5,
        order=3,
    ):

        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order

        self.sos = None
        self.sampling_rate = 100.0

    def design_filter(
        self,
        sampling_rate,
    ):

        self.sampling_rate = (
            sampling_rate
        )

        nyquist = (
            sampling_rate / 2.0
        )

        low = max(
            0.001,
            self.low_freq / nyquist,
        )

        high = min(
            0.999,
            self.high_freq / nyquist,
        )

        self.sos = signal.butter(
            self.order,
            [low, high],
            btype="band",
            output="sos",
        )

    def filter_csi_data(
        self,
        csi_data,
    ):

        if self.sos is None:
            self.design_filter(
                self.sampling_rate
            )

        magnitudes = np.abs(
            csi_data
        )

        filtered = np.zeros_like(
            magnitudes,
            dtype=float,
        )

        for sc in range(
            csi_data.shape[1]
        ):

            try:

                filtered[:, sc] = (
                    signal.sosfiltfilt(
                        self.sos,
                        magnitudes[:, sc],
                    )
                )

            except ValueError:

                filtered[:, sc] = (
                    magnitudes[:, sc]
                )

        return filtered


# ======================================================================
# SELF TEST
# ======================================================================

if __name__ == "__main__":

    print(
        "Testing live heartbeat pipeline..."
    )

    fs = 97.3

    duration = 30.0

    t = np.arange(
        0,
        duration,
        1.0 / fs,
    )

    expected_hr = 72.0

    heart = np.sin(
        2
        * np.pi
        * expected_hr
        / 60.0
        * t
    )

    heart += (
        0.25
        * np.sin(
            4
            * np.pi
            * expected_hr
            / 60.0
            * t
        )
    )

    heart += (
        0.1
        * np.random.randn(
            len(t)
        )
    )

    n_sc = 64

    csi = np.zeros(
        (
            len(t),
            n_sc,
        ),
        dtype=complex,
    )

    rng = np.random.default_rng(
        1234
    )

    for sc in range(n_sc):

        sensitivity = (
            0.1
            + 0.9
            * np.exp(
                -(
                    (
                        sc
                        - 30
                    )
                    ** 2
                )
                / 300.0
            )
        )

        amplitude = (
            100.0
            + 8.0
            * sensitivity
            * heart
            + rng.normal(
                0,
                1.5,
                len(t),
            )
        )

        phase = (
            sc
            * 0.1
        )

        csi[:, sc] = (
            amplitude
            * np.exp(
                1j
                * phase
            )
        )

    config = HeartbeatConfig()

    config.heart_rate_min = 42
    config.heart_rate_max = 150

    extractor = HeartbeatExtractor(
        config
    )

    result = (
        extractor.run_live_analysis(
            csi,
            t,
        )
    )

    print()
    print(
        "Expected HR:",
        expected_hr,
    )

    print(
        "Detected HR:",
        result["bpm"],
    )

    print(
        "Candidate:",
        result["candidate_bpm"],
    )

    print(
        "Confidence:",
        result["confidence"],
    )

    print(
        "Quality:",
        result["quality"],
    )

    print(
        "Estimated FS:",
        result["sampling_rate"],
    )

    print(
        "Subcarriers:",
        result[
            "selected_subcarriers"
        ],
    )