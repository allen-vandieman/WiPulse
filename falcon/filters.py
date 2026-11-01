#!/usr/bin/env python3
"""
filters.py
==========

Unified CSI filtering and heartbeat signal-processing module.

This file combines:

1. CSI subcarrier quality detection
   - Detect low-variation / problematic subcarriers
   - Detect permanently-zero subcarriers

2. CSI repair
   - Interpolate problematic subcarriers from neighboring subcarriers

3. Outlier filtering
   - Hampel filter

4. Frequency-domain filtering
   - Butterworth low-pass
   - Butterworth band-pass
   - Optional notch filter

5. Heartbeat preprocessing
   - Hampel
   - Optional notch
   - Heartbeat band-pass
   - Slow drift removal
   - Normalization

The module supports both:
    - complex CSI data: shape (samples, subcarriers)
    - real-valued amplitude/phase signals

Typical CSI heartbeat pipeline:

    Raw complex CSI
          |
          v
    Detect bad subcarriers
          |
          v
    Interpolate bad subcarriers
          |
          v
    Extract amplitude / phase
          |
          v
    Hampel filter
          |
          v
    Optional notch
          |
          v
    Heartbeat band-pass
          |
          v
    Remove slow drift
          |
          v
    Normalize
          |
          v
    PSD / FFT / HR estimation
"""

from __future__ import annotations

import numpy as np
from scipy import signal


# ============================================================================
# CONFIGURATION
# ============================================================================

class FilterConfig:
    """Configuration for the unified CSI filtering pipeline."""

    def __init__(self):

        # ------------------------------------------------------------------
        # Sampling
        # ------------------------------------------------------------------

        self.sampling_rate = 100.0

        # ------------------------------------------------------------------
        # CSI subcarrier quality detection
        # ------------------------------------------------------------------

        # Values below this magnitude are considered zero.
        self.zero_threshold = 1e-6

        # Threshold used when detecting low-variation subcarriers.
        self.dc_threshold = 0.1

        # Lowest 2% of mean-magnitude subcarriers are also considered
        # potentially problematic.
        self.dc_percentile = 2.0

        # ------------------------------------------------------------------
        # Hampel filter
        # ------------------------------------------------------------------

        self.hampel_window_size = 51
        self.hampel_threshold = 3.0

        # ------------------------------------------------------------------
        # Heartbeat band-pass
        # ------------------------------------------------------------------

        # 0.7 - 2.5 Hz corresponds to approximately:
        #
        #       42 - 150 BPM
        #
        # These can be changed depending on the experiment.
        self.heartbeat_low_hz = 0.7
        self.heartbeat_high_hz = 2.5
        self.heartbeat_filter_order = 4

        # ------------------------------------------------------------------
        # Optional notch filter
        # ------------------------------------------------------------------

        self.apply_notch = False

        # Can be a single frequency or a list of frequencies.
        #
        # Example:
        # self.notch_frequencies = 2.0
        #
        # or:
        #
        # self.notch_frequencies = [1.9, 2.1]
        #
        self.notch_frequencies = []

        self.notch_Q = 30.0

        # ------------------------------------------------------------------
        # Drift removal
        # ------------------------------------------------------------------

        self.drift_window_size = 21

        # ------------------------------------------------------------------
        # Processing options
        # ------------------------------------------------------------------

        # Whether problematic subcarriers should be interpolated.
        self.interpolate_problematic = True

        # If True, Hampel/notch/etc. can be applied to all subcarriers
        # rather than only problematic ones where appropriate.
        self.aggressive_filtering = False


# ============================================================================
# BASIC HAMPEL FILTER
# ============================================================================

def hampel_filter(
    x: np.ndarray,
    window_size: int = 51,
    n_sigmas: float = 3.0
) -> np.ndarray:
    """
    Hampel outlier filter along axis 0 (time).

    Works with:

        1D:
            (samples,)

        2D:
            (samples, subcarriers)

    For 2D data, every subcarrier is filtered independently.

    Outliers are replaced by the local median.

    Parameters
    ----------
    x:
        Real-valued input signal.

    window_size:
        Number of samples in the local window.

    n_sigmas:
        Outlier threshold.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """

    x = np.asarray(x, dtype=np.float64)

    was_1d = x.ndim == 1

    if was_1d:
        x = x[:, None]

    if x.ndim != 2:
        raise ValueError(
            f"Hampel filter expects a 1D or 2D array, got shape {x.shape}"
        )

    if window_size < 3:
        window_size = 3

    if window_size % 2 == 0:
        window_size += 1

    n, n_cols = x.shape

    out = x.copy()

    half_window = window_size // 2

    # MAD-to-standard-deviation scaling factor.
    MAD_SCALE = 1.4826

    for col in range(n_cols):

        series = x[:, col]

        for i in range(n):

            lo = max(0, i - half_window)
            hi = min(n, i + half_window + 1)

            window = series[lo:hi]

            median_value = np.median(window)

            mad = np.median(
                np.abs(window - median_value)
            )

            threshold = n_sigmas * MAD_SCALE * mad

            if threshold == 0:
                continue

            if np.abs(series[i] - median_value) > threshold:
                out[i, col] = median_value

    return out[:, 0] if was_1d else out


# ============================================================================
# BUTTERWORTH LOW-PASS
# ============================================================================

def butterworth_lowpass(
    x: np.ndarray,
    cutoff_hz: float,
    fs_hz: float,
    order: int = 4
) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter.

    Filtering occurs along axis 0 (time).
    """

    x = np.asarray(x, dtype=np.float64)

    nyquist = fs_hz / 2.0

    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz ({cutoff_hz}) must be between "
            f"0 and Nyquist ({nyquist})"
        )

    was_1d = x.ndim == 1

    if was_1d:
        x = x[:, None]

    if x.ndim != 2:
        raise ValueError(
            "Input must be a 1D or 2D array."
        )

    b, a = signal.butter(
        order,
        cutoff_hz / nyquist,
        btype="low"
    )

    out = signal.filtfilt(
        b,
        a,
        x,
        axis=0
    )

    return out[:, 0] if was_1d else out


# ============================================================================
# BUTTERWORTH BAND-PASS
# ============================================================================

def butterworth_bandpass(
    x: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs_hz: float,
    order: int = 4
) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass filter.

    Keeps frequencies between low_hz and high_hz.

    Filtering occurs along axis 0 (time).
    """

    x = np.asarray(x, dtype=np.float64)

    nyquist = fs_hz / 2.0

    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Require 0 < low_hz < high_hz < Nyquist ({nyquist}). "
            f"Got low={low_hz}, high={high_hz}."
        )

    was_1d = x.ndim == 1

    if was_1d:
        x = x[:, None]

    if x.ndim != 2:
        raise ValueError(
            "Input must be a 1D or 2D array."
        )

    b, a = signal.butter(
        order,
        [low_hz / nyquist, high_hz / nyquist],
        btype="band"
    )

    out = signal.filtfilt(
        b,
        a,
        x,
        axis=0
    )

    return out[:, 0] if was_1d else out


# ============================================================================
# NOTCH FILTER
# ============================================================================

def notch_filter(
    x: np.ndarray,
    notch_hz,
    fs_hz: float,
    quality: float = 30.0
) -> np.ndarray:
    """
    Zero-phase IIR notch filter.

    Removes one or more narrow frequency components.

    notch_hz can be:

        2.0

    or:

        [1.9, 2.1]

    """

    x = np.asarray(x, dtype=np.float64)

    nyquist = fs_hz / 2.0

    if np.isscalar(notch_hz):
        frequencies = [float(notch_hz)]
    else:
        frequencies = list(notch_hz)

    for f in frequencies:

        if not (0 < f < nyquist):
            raise ValueError(
                f"notch frequency ({f}) must be between "
                f"0 and Nyquist ({nyquist})"
            )

    was_1d = x.ndim == 1

    if was_1d:
        x = x[:, None]

    if x.ndim != 2:
        raise ValueError(
            "Input must be a 1D or 2D array."
        )

    out = x.copy()

    for f in frequencies:

        b, a = signal.iirnotch(
            f,
            quality,
            fs=fs_hz
        )

        out = signal.filtfilt(
            b,
            a,
            out,
            axis=0
        )

    return out[:, 0] if was_1d else out


# ============================================================================
# DC / SLOW DRIFT REMOVAL
# ============================================================================

def remove_dc_drift(
    x: np.ndarray,
    window_size: int = 21
) -> np.ndarray:
    """
    Remove slow-moving baseline/DC drift.

    A moving average is calculated along the time axis
    and subtracted from the signal.
    """

    x = np.asarray(x, dtype=np.float64)

    was_1d = x.ndim == 1

    if was_1d:
        x = x[:, None]

    if x.ndim != 2:
        raise ValueError(
            "Input must be a 1D or 2D array."
        )

    if x.shape[0] < 2:
        return x[:, 0] if was_1d else x

    if window_size < 2:
        window_size = 2

    kernel = np.ones(
        window_size,
        dtype=np.float64
    ) / window_size

    moving_average = np.apply_along_axis(
        lambda y: np.convolve(
            y,
            kernel,
            mode="same"
        ),
        0,
        x
    )

    detrended = x - moving_average

    return detrended[:, 0] if was_1d else detrended


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_signal(
    x: np.ndarray
) -> np.ndarray:
    """
    Standardize every signal/subcarrier to:

        mean = 0
        std  = 1
    """

    x = np.asarray(x, dtype=np.float64)

    was_1d = x.ndim == 1

    if was_1d:
        x = x[:, None]

    if x.ndim != 2:
        raise ValueError(
            "Input must be a 1D or 2D array."
        )

    mean = np.mean(
        x,
        axis=0,
        keepdims=True
    )

    std = np.std(
        x,
        axis=0,
        keepdims=True
    )

    std[std == 0] = 1.0

    out = (x - mean) / std

    return out[:, 0] if was_1d else out


# ============================================================================
# CSI FILTER CLASS
# ============================================================================

class CSIFilter:
    """
    CSI-specific processing.

    Handles complex CSI data and identifies problematic
    subcarriers before signal processing.
    """

    def __init__(self, config=None):

        self.config = (
            config
            if config is not None
            else FilterConfig()
        )

        self.problematic_mask = None
        self.dc_mask = None
        self.zero_mask = None


    # ----------------------------------------------------------------------
    # Detect low-variation / problematic subcarriers
    # ----------------------------------------------------------------------

    def detect_dc_subcarriers(
        self,
        csi_data,
        threshold=None
    ):
        """
        Detect subcarriers with unusually low variation.

        NOTE:
        This identifies problematic CSI subcarriers.

        It does NOT mean that the time-domain signal contains
        a DC frequency component.
        """

        if threshold is None:
            threshold = self.config.dc_threshold

        csi_data = np.asarray(csi_data)

        if csi_data.ndim != 2:
            raise ValueError(
                "CSI data must have shape "
                "(samples, subcarriers)."
            )

        magnitudes = np.abs(csi_data)

        mean_mag = np.mean(
            magnitudes,
            axis=0
        )

        variance_mag = np.var(
            magnitudes,
            axis=0
        )

        # Original implementation used variance / mean.
        # Keep that behavior for compatibility.
        cv = np.zeros_like(
            variance_mag
        )

        valid = mean_mag > 0

        cv[valid] = (
            variance_mag[valid]
            / mean_mag[valid]
        )

        percentile_limit = np.percentile(
            mean_mag,
            self.config.dc_percentile
        )

        dc_mask = (
            (mean_mag < percentile_limit)
            |
            (cv < threshold)
        )

        self.dc_mask = dc_mask

        return dc_mask


    # ----------------------------------------------------------------------
    # Detect permanent zeros
    # ----------------------------------------------------------------------

    def detect_permanent_zeros(
        self,
        csi_data,
        zero_threshold=None
    ):
        """
        Detect subcarriers whose magnitude is effectively
        zero for the entire capture.
        """

        if zero_threshold is None:
            zero_threshold = (
                self.config.zero_threshold
            )

        csi_data = np.asarray(csi_data)

        magnitudes = np.abs(csi_data)

        zero_mask = np.all(
            magnitudes < zero_threshold,
            axis=0
        )

        self.zero_mask = zero_mask

        return zero_mask


    # ----------------------------------------------------------------------
    # Detect all problematic subcarriers
    # ----------------------------------------------------------------------

    def detect_problematic_subcarriers(
        self,
        csi_data
    ):
        """
        Detect all problematic subcarriers.

        Problematic =
            low-variation/DC-like
            OR
            permanently zero
        """

        dc_mask = self.detect_dc_subcarriers(
            csi_data
        )

        zero_mask = self.detect_permanent_zeros(
            csi_data
        )

        problematic_mask = (
            dc_mask | zero_mask
        )

        self.problematic_mask = (
            problematic_mask
        )

        return problematic_mask


    # ----------------------------------------------------------------------
    # Interpolate bad subcarriers
    # ----------------------------------------------------------------------

    def interpolate_problematic_subcarriers(
        self,
        csi_data,
        problematic_mask=None
    ):
        """
        Replace problematic subcarriers using neighboring
        valid subcarriers.

        Interpolation is performed directly on the complex CSI,
        meaning real and imaginary components are interpolated
        together.
        """

        if not self.config.interpolate_problematic:
            return csi_data

        if problematic_mask is None:
            problematic_mask = (
                self.problematic_mask
            )

        if (
            problematic_mask is None
            or not np.any(problematic_mask)
        ):
            return csi_data

        filtered_data = np.asarray(
            csi_data
        ).copy()

        n_subcarriers = (
            filtered_data.shape[1]
        )

        problematic_indices = np.where(
            problematic_mask
        )[0]

        for sc in problematic_indices:

            left_sc = sc - 1
            right_sc = sc + 1

            # Search left for valid subcarrier.
            while (
                left_sc >= 0
                and problematic_mask[left_sc]
            ):
                left_sc -= 1

            # Search right for valid subcarrier.
            while (
                right_sc < n_subcarriers
                and problematic_mask[right_sc]
            ):
                right_sc += 1

            if (
                left_sc >= 0
                and right_sc < n_subcarriers
            ):

                weight_left = (
                    1.0
                    - (sc - left_sc)
                    / (right_sc - left_sc)
                )

                weight_right = (
                    1.0
                    - (right_sc - sc)
                    / (right_sc - left_sc)
                )

                filtered_data[:, sc] = (
                    weight_left
                    * filtered_data[:, left_sc]
                    +
                    weight_right
                    * filtered_data[:, right_sc]
                )

            elif left_sc >= 0:

                filtered_data[:, sc] = (
                    filtered_data[:, left_sc]
                )

            elif right_sc < n_subcarriers:

                filtered_data[:, sc] = (
                    filtered_data[:, right_sc]
                )

            else:

                # Extremely unlikely for a normal CSI capture,
                # but provide a safe fallback.
                filtered_data[:, sc] = (
                    np.mean(
                        filtered_data,
                        axis=1
                    )
                )

        return filtered_data


    # ----------------------------------------------------------------------
    # Full CSI cleanup
    # ----------------------------------------------------------------------

    def clean_csi(
        self,
        csi_data,
        verbose=True
    ):
        """
        Perform CSI-specific cleaning.

        Pipeline:

            Detect bad subcarriers
                    |
                    v
            Interpolate bad subcarriers

        This operates on complex CSI data.

        Returns
        -------
        filtered_csi
        """

        csi_data = np.asarray(
            csi_data
        )

        if csi_data.ndim != 2:
            raise ValueError(
                "CSI data must have shape "
                "(samples, subcarriers)."
            )

        if verbose:
            print()
            print("CSI QUALITY FILTERING")
            print("=" * 60)

        # --------------------------------------------------------------
        # Detect problematic subcarriers
        # --------------------------------------------------------------

        problematic_mask = (
            self.detect_problematic_subcarriers(
                csi_data
            )
        )

        if verbose:

            print(
                f"    DC/low-variation: "
                f"{np.sum(self.dc_mask)}"
            )

            print(
                f"    Permanent zeros: "
                f"{np.sum(self.zero_mask)}"
            )

            print(
                f"    Total problematic: "
                f"{np.sum(problematic_mask)}"
            )

        # --------------------------------------------------------------
        # Repair them
        # --------------------------------------------------------------

        filtered = (
            self.interpolate_problematic_subcarriers(
                csi_data,
                problematic_mask
            )
        )

        if verbose:

            print(
                "    Problematic subcarriers "
                "interpolated."
            )

            print("=" * 60)

        return filtered


# ============================================================================
# AMPLITUDE EXTRACTION
# ============================================================================

def extract_amplitude(
    csi_data
):
    """
    Extract CSI amplitude/magnitude.

    Input:
        complex CSI

    Output:
        real-valued amplitude

    Shape is preserved:

        (samples, subcarriers)
    """

    return np.abs(csi_data)


# ============================================================================
# PHASE EXTRACTION
# ============================================================================

def extract_phase(
    csi_data,
    unwrap=True
):
    """
    Extract CSI phase.

    Parameters
    ----------
    unwrap:
        If True, unwrap phase along the time axis.
    """

    phase = np.angle(csi_data)

    if unwrap:
        phase = np.unwrap(
            phase,
            axis=0
        )

    return phase


# ============================================================================
# HEARTBEAT PREPROCESSING
# ============================================================================

def preprocess_subcarrier_signal(
    x: np.ndarray,
    fs_hz: float,
    hampel_window: int = 51,
    hampel_sigmas: float = 3.0,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
    order: int = 4,
    notch_hz=None,
    notch_quality: float = 30.0,
    drift_window: int = 21,
    normalize: bool = True
) -> np.ndarray:
    """
    Apply the complete heartbeat-oriented signal-processing chain.

    Pipeline:

        Hampel
          |
          v
        Optional notch
          |
          v
        Heartbeat band-pass
          |
          v
        Drift removal
          |
          v
        Normalization

    Input must be a 1D real-valued signal.

    Default heartbeat range:

        0.7 - 2.5 Hz

        42 - 150 BPM
    """

    x = np.asarray(
        x,
        dtype=np.float64
    )

    if x.ndim != 1:
        raise ValueError(
            "preprocess_subcarrier_signal "
            "expects a 1D signal."
        )

    if len(x) < 20:
        raise ValueError(
            "Signal is too short for heartbeat filtering."
        )

    # ------------------------------------------------------------------
    # 1. Hampel outlier removal
    # ------------------------------------------------------------------

    filtered = hampel_filter(
        x,
        window_size=hampel_window,
        n_sigmas=hampel_sigmas
    )

    # ------------------------------------------------------------------
    # 2. Optional notch
    # ------------------------------------------------------------------

    if notch_hz is not None:

        filtered = notch_filter(
            filtered,
            notch_hz=notch_hz,
            fs_hz=fs_hz,
            quality=notch_quality
        )

    # ------------------------------------------------------------------
    # 3. Heartbeat band-pass
    # ------------------------------------------------------------------

    filtered = butterworth_bandpass(
        filtered,
        low_hz=low_hz,
        high_hz=high_hz,
        fs_hz=fs_hz,
        order=order
    )

    # ------------------------------------------------------------------
    # 4. Remove slow drift
    # ------------------------------------------------------------------

    filtered = remove_dc_drift(
        filtered,
        window_size=drift_window
    )

    # ------------------------------------------------------------------
    # 5. Normalize
    # ------------------------------------------------------------------

    if normalize:

        filtered = normalize_signal(
            filtered
        )

    return filtered


# ============================================================================
# COMPLETE CSI HEARTBEAT PIPELINE
# ============================================================================

def preprocess_csi_for_heartbeat(
    csi_data,
    fs_hz,
    subcarrier=None,
    use_phase=False,
    hampel_window=51,
    hampel_sigmas=3.0,
    low_hz=0.7,
    high_hz=2.5,
    filter_order=4,
    notch_hz=None,
    notch_quality=30.0,
    drift_window=21,
    normalize=True,
    interpolate_bad=True,
    verbose=True
):
    """
    Complete CSI-to-heartbeat preprocessing pipeline.

    Pipeline:

        Complex CSI
             |
             v
        Detect problematic SCs
             |
             v
        Interpolate bad SCs
             |
             v
        Extract amplitude/phase
             |
             v
        Select subcarrier
             |
             v
        Hampel
             |
             v
        Optional notch
             |
             v
        Heartbeat band-pass
             |
             v
        Drift removal
             |
             v
        Normalize

    Parameters
    ----------
    csi_data:
        Complex CSI array:

            shape = (samples, subcarriers)

    fs_hz:
        CSI sampling rate.

    subcarrier:
        Specific subcarrier to process.

        Example:
            subcarrier=128

        If None:
            all subcarriers are processed.

    use_phase:
        False -> amplitude
        True  -> phase

    Returns
    -------
    filtered_signal
    """

    csi_data = np.asarray(
        csi_data
    )

    if csi_data.ndim != 2:
        raise ValueError(
            "csi_data must have shape "
            "(samples, subcarriers)."
        )

    # ------------------------------------------------------------------
    # CSI quality filtering
    # ------------------------------------------------------------------

    config = FilterConfig()

    config.sampling_rate = fs_hz
    config.interpolate_problematic = (
        interpolate_bad
    )

    csi_filter = CSIFilter(
        config
    )

    cleaned_csi = csi_filter.clean_csi(
        csi_data,
        verbose=verbose
    )

    # ------------------------------------------------------------------
    # Extract signal
    # ------------------------------------------------------------------

    if use_phase:

        signal_data = extract_phase(
            cleaned_csi
        )

    else:

        signal_data = extract_amplitude(
            cleaned_csi
        )

    # ------------------------------------------------------------------
    # Process selected subcarrier
    # ------------------------------------------------------------------

    if subcarrier is not None:

        if (
            subcarrier < 0
            or subcarrier >= signal_data.shape[1]
        ):
            raise IndexError(
                f"Subcarrier {subcarrier} is outside "
                f"the valid range 0-"
                f"{signal_data.shape[1] - 1}"
            )

        selected_signal = (
            signal_data[:, subcarrier]
        )

        return preprocess_subcarrier_signal(
            selected_signal,
            fs_hz=fs_hz,
            hampel_window=hampel_window,
            hampel_sigmas=hampel_sigmas,
            low_hz=low_hz,
            high_hz=high_hz,
            order=filter_order,
            notch_hz=notch_hz,
            notch_quality=notch_quality,
            drift_window=drift_window,
            normalize=normalize
        )

    # ------------------------------------------------------------------
    # Process every subcarrier
    # ------------------------------------------------------------------

    processed = np.zeros_like(
        signal_data,
        dtype=np.float64
    )

    for sc in range(
        signal_data.shape[1]
    ):

        processed[:, sc] = (
            preprocess_subcarrier_signal(
                signal_data[:, sc],
                fs_hz=fs_hz,
                hampel_window=hampel_window,
                hampel_sigmas=hampel_sigmas,
                low_hz=low_hz,
                high_hz=high_hz,
                order=filter_order,
                notch_hz=notch_hz,
                notch_quality=notch_quality,
                drift_window=drift_window,
                normalize=normalize
            )
        )

    return processed


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def filter_csi_data(
    csi_data,
    config=None,
    verbose=True
):
    """
    Convenience function for CSI quality filtering.

    This performs:

        detect problematic subcarriers
        +
        interpolate problematic subcarriers

    It does NOT perform heartbeat band-pass filtering.

    Use preprocess_csi_for_heartbeat() for the complete
    heartbeat pipeline.
    """

    filter_obj = CSIFilter(
        config
    )

    return filter_obj.clean_csi(
        csi_data,
        verbose=verbose
    )


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":

    print()
    print("Testing unified CSI filter module...")
    print("=" * 60)

    # --------------------------------------------------------------
    # Generate synthetic complex CSI
    # --------------------------------------------------------------

    fs = 100.0

    n_samples = 1000
    n_subcarriers = 64

    t = np.arange(
        n_samples
    ) / fs

    # Synthetic heartbeat at 1.2 Hz
    # = 72 BPM
    heartbeat = np.sin(
        2 * np.pi * 1.2 * t
    )

    test_data = (
        np.random.randn(
            n_samples,
            n_subcarriers
        )
        +
        1j * np.random.randn(
            n_samples,
            n_subcarriers
        )
    )

    # Add heartbeat-like variation
    # to a few subcarriers.
    for sc in [10, 20, 30]:

        test_data[:, sc] += (
            0.5 * heartbeat
        )

    # Make one subcarrier permanently zero.
    test_data[:, 5] = 0

    # --------------------------------------------------------------
    # Test CSI quality filtering
    # --------------------------------------------------------------

    config = FilterConfig()

    config.sampling_rate = fs

    csi_filter = CSIFilter(
        config
    )

    cleaned = csi_filter.clean_csi(
        test_data,
        verbose=True
    )

    print()
    print(
        f"Original CSI shape: "
        f"{test_data.shape}"
    )

    print(
        f"Cleaned CSI shape: "
        f"{cleaned.shape}"
    )

    # --------------------------------------------------------------
    # Test amplitude extraction
    # --------------------------------------------------------------

    amplitude = extract_amplitude(
        cleaned
    )

    print(
        f"Amplitude shape: "
        f"{amplitude.shape}"
    )

    # --------------------------------------------------------------
    # Test phase extraction
    # --------------------------------------------------------------

    phase = extract_phase(
        cleaned
    )

    print(
        f"Phase shape: "
        f"{phase.shape}"
    )

    # --------------------------------------------------------------
    # Test heartbeat processing
    # --------------------------------------------------------------

    heartbeat_signal = (
        preprocess_subcarrier_signal(
            amplitude[:, 20],
            fs_hz=fs,
            hampel_window=51,
            hampel_sigmas=3.0,
            low_hz=0.7,
            high_hz=2.5,
            order=4
        )
    )

    print(
        f"Heartbeat signal shape: "
        f"{heartbeat_signal.shape}"
    )

    print(
        f"Heartbeat signal mean: "
        f"{np.mean(heartbeat_signal):.6f}"
    )

    print(
        f"Heartbeat signal std: "
        f"{np.std(heartbeat_signal):.6f}"
    )

    # --------------------------------------------------------------
    # Test complete pipeline
    # --------------------------------------------------------------

    complete = (
        preprocess_csi_for_heartbeat(
            test_data,
            fs_hz=fs,
            subcarrier=20,
            use_phase=False,
            verbose=False
        )
    )

    print(
        f"Complete pipeline output shape: "
        f"{complete.shape}"
    )

    print()
    print("=" * 60)
    print("All tests completed successfully.")
    print("=" * 60)