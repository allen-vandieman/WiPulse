# windowed_psd_analysis.py
#
# Track heart rate over time from complex Wi-Fi CSI and compare it
# against timestamped ground-truth BPM samples.
#
# Ground-truth formats supported:
#
#   1. bpm_samples  <-- CURRENT WiPulse format
#      CSV:
#          ts,bpm
#          1788378695.929,100
#          1788378696.180,100
#          1788378696.431,100
#
#      The timestamp represents when the BPM value was received.
#      It is NOT an individual heartbeat timestamp.
#
#   2. beats       <-- legacy format
#      CSV containing individual heartbeat timestamps in "ts".
#
# For the current Arduino + two-sensor setup, use:
#
#      GT_TYPE = "bpm_samples"
#
# Pipeline:
#
#   Raw CSI
#      |
#      +--> amplitude
#      +--> phase
#      +--> phase + common removal
#      +--> phase differences
#                 |
#                 v
#        channel processing
#                 |
#        DC removal
#        Hampel filter
#        detrending
#        quality scoring
#        bandpass
#                 |
#                 v
#        top-N channel fusion
#                 |
#                 v
#          20-second PSD
#                 |
#                 v
#              CSI BPM
#
# Ground truth:
#
#   Arduino
#      |
#   two sensors
#      |
#   agreement / averaging
#      |
#   HR,<bpm>
#      |
#   Raspberry Pi
#      |
#   timestamped CSV
#      |
#   median BPM within each analysis window
#
# Run:
#
#   python windowed_psd_analysis.py
#
# ============================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import detrend

from read_csv_csi import CSIDataLoader


# ============================================================
# HEARTBEAT PIPELINE
# ============================================================


class HeartbeatPipeline:
    """Unified channel-processing pipeline used by all CSI methods."""

    def __init__(self, fs=100.0):
        self.fs = float(fs)

    # --------------------------------------------------------
    # CSI representations
    # --------------------------------------------------------

    def extract_amplitude(self, csi_data):
        """Return CSI magnitude/amplitude."""
        return np.abs(csi_data)

    def extract_phase(self, csi_data, unwrap=True):
        """Return CSI phase."""
        phase = np.angle(csi_data)

        if unwrap:
            phase = np.unwrap(phase, axis=0)

        return phase

    def remove_common_phase(self, phase_data):
        """Remove common phase variation across subcarriers."""
        common_phase = np.median(
            phase_data,
            axis=1,
        )

        return phase_data - common_phase[:, np.newaxis]

    # --------------------------------------------------------
    # Hampel filtering
    # --------------------------------------------------------

    def apply_hampel(
        self,
        x,
        window_size=11,
        threshold=3.0,
    ):
        """
        Simple Hampel outlier filter.
        """

        if window_size % 2 == 0:
            window_size += 1

        half = window_size // 2

        x = np.asarray(
            x,
            dtype=float,
        )

        y = x.copy()

        for i in range(len(x)):

            start = max(
                0,
                i - half,
            )

            end = min(
                len(x),
                i + half + 1,
            )

            w = x[start:end]

            med = np.median(w)

            mad = np.median(
                np.abs(w - med)
            )

            if mad > 0:

                z = (
                    abs(x[i] - med)
                    / (1.4826 * mad)
                )

                if z > threshold:
                    y[i] = med

        return y

    # --------------------------------------------------------
    # Bandpass filtering
    # --------------------------------------------------------

    def bandpass_filter(
        self,
        x,
        low,
        high,
        order,
    ):
        """Apply zero-phase Butterworth bandpass filter."""

        nyquist = self.fs / 2.0

        if not (
            0 < low
            < high
            < nyquist
        ):
            raise ValueError(
                f"Invalid bandpass "
                f"{low}-{high} Hz "
                f"for fs={self.fs:.3f} Hz"
            )

        sos = signal.butter(
            order,
            [
                low / nyquist,
                high / nyquist,
            ],
            btype="bandpass",
            output="sos",
        )

        return signal.sosfiltfilt(
            sos,
            x - np.mean(x),
        )

    # --------------------------------------------------------
    # Channel quality
    # --------------------------------------------------------

    def compute_quality_score(
        self,
        x,
        hr_range,
    ):
        """
        Score a cleaned, detrended signal BEFORE heartbeat
        bandpass filtering.

        Score =
            dominant heartbeat-band amplitude
            /
            median out-of-band noise

        Returns:
            score,
            peak_freq_hz,
            peak_amp
        """

        x = np.asarray(
            x,
            dtype=float,
        )

        if (
            len(x) < 16
            or np.std(x) == 0
        ):
            return (
                0.0,
                0.0,
                0.0,
            )

        spectrum = np.abs(
            np.fft.rfft(
                x - np.mean(x)
            )
        )

        freqs = np.fft.rfftfreq(
            len(x),
            d=1.0 / self.fs,
        )

        low, high = hr_range

        hr_mask = (
            (freqs >= low)
            & (freqs <= high)
        )

        if not np.any(hr_mask):
            return (
                0.0,
                0.0,
                0.0,
            )

        hr_spec = spectrum[hr_mask]
        hr_freqs = freqs[hr_mask]

        peak_idx = int(
            np.argmax(hr_spec)
        )

        peak_amp = float(
            hr_spec[peak_idx]
        )

        peak_freq = float(
            hr_freqs[peak_idx]
        )

        noise_mask = (
            (freqs >= 0.1)
            & (freqs <= 5.0)
            & ~hr_mask
        )

        if np.any(noise_mask):

            noise_floor = float(
                np.median(
                    spectrum[noise_mask]
                )
            )

        else:

            noise_floor = float(
                np.median(spectrum)
            )

        score = (
            peak_amp
            / (noise_floor + 1e-10)
        )

        return (
            float(score),
            peak_freq,
            peak_amp,
        )

    # --------------------------------------------------------
    # Process one channel
    # --------------------------------------------------------

    def process_channel(
        self,
        x,
        low,
        high,
        order=4,
        apply_hampel=True,
        detrend_signal=True,
    ):
        """
        Returns:

            processed_signal
                Heartbeat-band signal.

            quality_signal
                Cleaned/detrended signal BEFORE bandpass.
        """

        x = np.asarray(
            x,
            dtype=float,
        ).copy()

        # Remove DC
        x -= np.mean(x)

        # Hampel filtering
        if apply_hampel:

            x = self.apply_hampel(
                x,
                window_size=11,
                threshold=3.0,
            )

        # Linear detrending
        if detrend_signal:
            x = detrend(x)

        quality_signal = x.copy()

        # Heartbeat bandpass
        processed = self.bandpass_filter(
            x,
            low=low,
            high=high,
            order=order,
        )

        # Normalize
        std = np.std(processed)

        if std > 0:
            processed = (
                processed / std
            )

        return (
            processed,
            quality_signal,
        )

    # --------------------------------------------------------
    # Process every candidate channel
    # --------------------------------------------------------

    def process_all_channels(
        self,
        channels,
        channel_names,
        low,
        high,
        order=4,
        top_candidates=None,
    ):
        """
        Process and rank all supplied channels.

        Every supplied channel is currently processed so that
        channel selection is fair.
        """

        processed_signals = {}
        quality_rows = []

        for idx, (
            name,
            x,
        ) in enumerate(
            zip(
                channel_names,
                channels,
            ),
            start=1,
        ):

            try:

                (
                    processed,
                    quality_signal,
                ) = self.process_channel(
                    x,
                    low=low,
                    high=high,
                    order=order,
                    apply_hampel=True,
                    detrend_signal=True,
                )

                (
                    score,
                    peak_freq,
                    peak_amp,
                ) = self.compute_quality_score(
                    quality_signal,
                    hr_range=(
                        low,
                        high,
                    ),
                )

                processed_signals[
                    name
                ] = processed

                quality_rows.append(
                    {
                        "name": name,
                        "quality_score": score,
                        "peak_freq_hz": peak_freq,
                        "peak_bpm": (
                            peak_freq * 60.0
                        ),
                        "peak_amp": peak_amp,
                        "std": float(
                            np.std(processed)
                        ),
                    }
                )

            except Exception as exc:

                print(
                    f"Warning: failed to "
                    f"process {name}: {exc}"
                )

            if idx % 100 == 0:

                print(
                    f"  Processed "
                    f"{idx}/{len(channel_names)} "
                    f"channels"
                )

        quality_df = pd.DataFrame(
            quality_rows
        )

        if quality_df.empty:

            return (
                processed_signals,
                quality_df,
            )

        quality_df = (
            quality_df
            .sort_values(
                "quality_score",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        return (
            processed_signals,
            quality_df,
        )

    # --------------------------------------------------------
    # Fuse best channels
    # --------------------------------------------------------

    def fuse_top_channels(
        self,
        processed_signals,
        quality_df,
        top_n=5,
        align_polarity=True,
    ):
        """
        Select, polarity-align and average the best channels.
        """

        if quality_df.empty:

            return (
                np.array([]),
                [],
            )

        top_names = (
            quality_df
            .head(top_n)
            ["name"]
            .tolist()
        )

        top_signals = [
            processed_signals[name].copy()
            for name in top_names
            if name in processed_signals
        ]

        valid_names = [
            name
            for name in top_names
            if name in processed_signals
        ]

        if not top_signals:

            return (
                np.array([]),
                [],
            )

        # Align polarity before averaging
        if (
            align_polarity
            and len(top_signals) > 1
        ):

            reference = top_signals[0]

            aligned = [
                reference
            ]

            for x in top_signals[1:]:

                corr = np.corrcoef(
                    reference,
                    x,
                )[0, 1]

                if (
                    np.isfinite(corr)
                    and corr < 0
                ):
                    x = -x

                aligned.append(x)

            top_signals = aligned

        # Average channels
        fused = np.mean(
            top_signals,
            axis=0,
        )

        # Normalize fused signal
        std = np.std(fused)

        if std > 0:

            fused = (
                fused
                - np.mean(fused)
            ) / std

        return (
            fused,
            valid_names,
        )


# ============================================================
# WINDOWED PSD ANALYZER
# ============================================================


class WindowedPSDAnalyzer:
    """
    Extract CSI heartbeat signals and compare sliding-window
    BPM against ground truth.
    """

    def __init__(
        self,
        fs=100.0,
    ):

        self.fs = float(fs)

        self.pipeline = (
            HeartbeatPipeline(fs)
        )

        self.results = None

    # --------------------------------------------------------
    # Build candidate channels
    # --------------------------------------------------------

    def build_channels(
        self,
        csi_data,
        method,
        phase_lags=(
            1,
            2,
            4,
            8,
            16,
            32,
        ),
    ):
        """
        Create named 1-D candidate channels for one method.
        """

        # ----------------------------
        # Amplitude
        # ----------------------------

        if method == "amplitude":

            data = (
                self.pipeline
                .extract_amplitude(
                    csi_data
                )
            )

            names = [
                f"amp_sc{sc}"
                for sc in range(
                    data.shape[1]
                )
            ]

            channels = [
                data[:, sc]
                for sc in range(
                    data.shape[1]
                )
            ]

        # ----------------------------
        # Raw phase
        # ----------------------------

        elif method == "phase":

            data = (
                self.pipeline
                .extract_phase(
                    csi_data,
                    unwrap=True,
                )
            )

            names = [
                f"phase_sc{sc}"
                for sc in range(
                    data.shape[1]
                )
            ]

            channels = [
                data[:, sc]
                for sc in range(
                    data.shape[1]
                )
            ]

        # ----------------------------
        # Common phase removed
        # ----------------------------

        elif method == "phase_common":

            phase = (
                self.pipeline
                .extract_phase(
                    csi_data,
                    unwrap=True,
                )
            )

            data = (
                self.pipeline
                .remove_common_phase(
                    phase
                )
            )

            names = [
                f"common_sc{sc}"
                for sc in range(
                    data.shape[1]
                )
            ]

            channels = [
                data[:, sc]
                for sc in range(
                    data.shape[1]
                )
            ]

        # ----------------------------
        # Phase differences
        # ----------------------------

        elif method == "phase_diff":

            phase = (
                self.pipeline
                .extract_phase(
                    csi_data,
                    unwrap=True,
                )
            )

            channels = []
            names = []

            n_sc = phase.shape[1]

            for lag in phase_lags:

                if lag >= n_sc:
                    continue

                for i in range(
                    n_sc - lag
                ):

                    j = i + lag

                    channels.append(
                        phase[:, i]
                        - phase[:, j]
                    )

                    names.append(
                        f"diff_{i}_{j}"
                    )

        else:

            raise ValueError(
                f"Unknown method: {method}"
            )

        return (
            channels,
            names,
        )

    # --------------------------------------------------------
    # Extract heartbeat signal
    # --------------------------------------------------------

    def extract_heartbeat_signal(
        self,
        csi_data,
        method,
        low=1.2,
        high=2.8,
        order=4,
        top_n=5,
        phase_lags=(
            1,
            2,
            4,
            8,
            16,
            32,
        ),
    ):
        """
        Extract one fused heartbeat waveform.

        Pipeline:

            candidate channels
                ->
            DC removal
                ->
            Hampel
                ->
            detrend
                ->
            quality score
                ->
            rank channels
                ->
            bandpass
                ->
            polarity alignment
                ->
            top-N fusion
        """

        print(
            f"Building {method} candidates..."
        )

        (
            channels,
            names,
        ) = self.build_channels(
            csi_data,
            method=method,
            phase_lags=phase_lags,
        )

        print(
            f"  Candidates: {len(names)}"
        )

        (
            processed,
            quality,
        ) = (
            self.pipeline
            .process_all_channels(
                channels,
                names,
                low=low,
                high=high,
                order=order,
            )
        )

        (
            heartbeat,
            top_names,
        ) = (
            self.pipeline
            .fuse_top_channels(
                processed,
                quality,
                top_n=top_n,
                align_polarity=True,
            )
        )

        if heartbeat.size == 0:

            heartbeat = np.zeros(
                csi_data.shape[0]
            )

        return (
            heartbeat,
            quality,
            top_names,
        )

    # --------------------------------------------------------
    # Quadratic PSD peak interpolation
    # --------------------------------------------------------

    @staticmethod
    def _quadratic_peak(
        freqs,
        psd,
        idx,
    ):
        """
        Refine a PSD peak using
        3-point quadratic interpolation.
        """

        if (
            idx <= 0
            or idx >= len(psd) - 1
        ):

            return float(
                freqs[idx]
            )

        alpha = psd[idx - 1]
        beta = psd[idx]
        gamma = psd[idx + 1]

        denominator = (
            alpha
            - 2.0 * beta
            + gamma
        )

        if abs(
            denominator
        ) < 1e-20:

            return float(
                freqs[idx]
            )

        delta = (
            0.5
            * (alpha - gamma)
            / denominator
        )

        delta = float(
            np.clip(
                delta,
                -1.0,
                1.0,
            )
        )

        df = (
            freqs[1]
            - freqs[0]
        )

        return float(
            freqs[idx]
            + delta * df
        )

    # --------------------------------------------------------
    # PSD -> BPM
    # --------------------------------------------------------

    def compute_bpm_from_psd(
        self,
        x,
        low=1.2,
        high=2.8,
        nfft=4096,
    ):
        """
        Estimate BPM from the dominant PSD peak.

        The complete available window is used as the Welch
        segment. Zero-padding provides a denser frequency grid.
        """

        x = np.asarray(
            x,
            dtype=float,
        )

        if len(x) < max(
            16,
            int(self.fs * 3),
        ):

            return (
                0.0,
                0.0,
                0.0,
            )

        x = (
            x
            - np.mean(x)
        )

        nperseg = len(x)

        nfft = max(
            int(nfft),
            nperseg,
        )

        freqs, psd = signal.welch(
            x,
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=0,
            nfft=nfft,
            detrend="constant",
            scaling="density",
        )

        hr_mask = (
            (freqs >= low)
            & (freqs <= high)
        )

        if not np.any(
            hr_mask
        ):

            return (
                0.0,
                0.0,
                0.0,
            )

        hr_freqs = (
            freqs[hr_mask]
        )

        hr_psd = (
            psd[hr_mask]
        )

        peak_idx = int(
            np.argmax(hr_psd)
        )

        peak_freq = (
            self._quadratic_peak(
                hr_freqs,
                hr_psd,
                peak_idx,
            )
        )

        peak_power = float(
            hr_psd[peak_idx]
        )

        bpm = (
            peak_freq
            * 60.0
        )

        return (
            float(bpm),
            peak_power,
            float(peak_freq),
        )

    # ========================================================
    # TIMESTAMP ALIGNMENT
    # ========================================================

    @staticmethod
    def prepare_timebases(
        csi_timestamps,
        gt_df,
        gt_time_column="ts",
    ):
        """
        Put CSI and ground-truth timestamps on a common time axis.

        If the timestamps overlap on the same absolute clock,
        both are shifted to the beginning of their overlap.

        If they do not overlap, both are converted to relative
        time from their own starts.
        """

        csi_ts = np.asarray(
            csi_timestamps,
            dtype=float,
        )

        if (
            gt_time_column
            not in gt_df.columns
        ):

            raise ValueError(
                "Ground truth must contain "
                f"'{gt_time_column}' "
                "timestamps."
            )

        gt_ts = np.asarray(
            gt_df[
                gt_time_column
            ],
            dtype=float,
        )

        if (
            len(csi_ts) == 0
            or len(gt_ts) == 0
        ):

            raise ValueError(
                "Empty CSI or ground-truth "
                "timestamps."
            )

        # Ensure timestamps are sorted
        csi_order = np.argsort(
            csi_ts
        )

        gt_order = np.argsort(
            gt_ts
        )

        csi_ts = csi_ts[
            csi_order
        ]

        gt_ts = gt_ts[
            gt_order
        ]

        # Check absolute timestamp overlap
        absolute_overlap = (
            max(
                csi_ts[0],
                gt_ts[0],
            )
            <=
            min(
                csi_ts[-1],
                gt_ts[-1],
            )
        )

        if absolute_overlap:

            origin = max(
                csi_ts[0],
                gt_ts[0],
            )

            csi_time = (
                csi_ts
                - origin
            )

            gt_time = (
                gt_ts
                - origin
            )

            mode = (
                "common absolute timestamps"
            )

        else:

            csi_time = (
                csi_ts
                - csi_ts[0]
            )

            gt_time = (
                gt_ts
                - gt_ts[0]
            )

            mode = (
                "independent relative timestamps"
            )

            print(
                "WARNING: CSI and GT timestamp "
                "ranges do not overlap."
            )

            print(
                "Windows are aligned only by "
                "their individual starts."
            )

            print(
                "If the devices started at different "
                "times, provide a manual GT offset."
            )

        return (
            csi_time,
            gt_time,
            mode,
        )

    # ========================================================
    # LEGACY BEAT-BASED GROUND TRUTH
    # ========================================================

    @staticmethod
    def compute_ground_truth_bpm(
        gt_time,
        window_start,
        window_end,
        min_beats=3,
    ):
        """
        Legacy ground-truth calculation.

        Use only when gt_time contains actual heartbeat
        timestamps.

        Calculates BPM from RR intervals.
        """

        beats = np.asarray(
            gt_time,
            dtype=float,
        )

        if len(beats) < 2:

            return (
                0.0,
                0,
            )

        rr = np.diff(
            beats
        )

        rr_mid = (
            beats[:-1]
            + beats[1:]
        ) / 2.0

        mask = (
            (rr_mid >= window_start)
            & (rr_mid < window_end)
            & (rr > 0)
        )

        rr_window = (
            rr[mask]
        )

        if len(rr_window) < max(
            1,
            min_beats - 1,
        ):

            return (
                0.0,
                0,
            )

        bpm = (
            60.0
            / np.mean(
                rr_window
            )
        )

        n_beats = int(
            np.sum(
                (
                    beats
                    >= window_start
                )
                &
                (
                    beats
                    < window_end
                )
            )
        )

        return (
            float(bpm),
            n_beats,
        )

    # ========================================================
    # NEW BPM-SAMPLE GROUND TRUTH
    # ========================================================

    @staticmethod
    def compute_ground_truth_bpm_from_samples(
        gt_times,
        gt_bpms,
        window_start,
        window_end,
        min_samples=3,
        min_bpm=30.0,
        max_bpm=220.0,
    ):
        """
        Compute the ground-truth BPM for a window from
        timestamped BPM samples.

        This is the correct method for the current Arduino
        two-sensor ground-truth setup.

        Example:

            ts,bpm

            10.00,100
            10.25,100
            10.50,100
            10.75,101
            11.00,101

        These timestamps are BPM sample timestamps.
        They are NOT heartbeat timestamps.

        The median is used rather than the mean so that a
        small number of erroneous BPM estimates has less
        influence on the reference value.

        Returns:

            median_bpm, n_valid_samples

        or:

            0.0, 0

        if insufficient valid samples exist.
        """

        gt_times = np.asarray(
            gt_times,
            dtype=float,
        )

        gt_bpms = np.asarray(
            gt_bpms,
            dtype=float,
        )

        # ----------------------------------------------------
        # Validate arrays
        # ----------------------------------------------------

        if len(gt_times) != len(
            gt_bpms
        ):

            raise ValueError(
                "Ground-truth timestamp/BPM "
                "length mismatch: "
                f"{len(gt_times)} timestamps "
                f"vs "
                f"{len(gt_bpms)} BPM samples."
            )

        if len(gt_times) == 0:

            return (
                0.0,
                0,
            )

        # ----------------------------------------------------
        # Select samples in window
        # ----------------------------------------------------

        mask = (
            (gt_times >= window_start)
            &
            (gt_times < window_end)
            &
            np.isfinite(gt_times)
            &
            np.isfinite(gt_bpms)
            &
            (gt_bpms >= min_bpm)
            &
            (gt_bpms <= max_bpm)
        )

        bpms = gt_bpms[
            mask
        ]

        # ----------------------------------------------------
        # Minimum number of valid samples
        # ----------------------------------------------------

        if len(bpms) < max(
            1,
            min_samples,
        ):

            return (
                0.0,
                0,
            )

        # ----------------------------------------------------
        # Median reference BPM
        # ----------------------------------------------------

        gt_bpm = float(
            np.median(bpms)
        )

        return (
            gt_bpm,
            int(len(bpms)),
        )

    # ========================================================
    # WINDOWED ANALYSIS
    # ========================================================

    def analyze_windows(
        self,
        heartbeat_signal,
        csi_timestamps,
        gt_time,
        gt_bpm=None,
        window_seconds=20.0,
        step_seconds=5.0,
        low=1.2,
        high=2.8,
        min_beats=3,
        min_gt_samples=3,
        manual_gt_offset=0.0,
        gt_type="bpm_samples",
    ):
        """
        Track CSI BPM and ground-truth BPM in timestamp-based
        sliding windows.

        Ground truth modes:

            gt_type="bpm_samples"
                Current WiPulse format:
                    ts,bpm

            gt_type="beats"
                Legacy format:
                    individual heartbeat timestamps
        """

        # ----------------------------------------------------
        # Convert input arrays
        # ----------------------------------------------------

        heartbeat_signal = np.asarray(
            heartbeat_signal,
            dtype=float,
        )

        csi_time = np.asarray(
            csi_timestamps,
            dtype=float,
        )

        gt_time = np.asarray(
            gt_time,
            dtype=float,
        )

        # ----------------------------------------------------
        # Validate CSI
        # ----------------------------------------------------

        if len(csi_time) == 0:

            raise ValueError(
                "CSI timestamps are empty."
            )

        if len(gt_time) == 0:

            raise ValueError(
                "Ground-truth timestamps are empty."
            )

        if len(heartbeat_signal) != len(
            csi_time
        ):

            raise ValueError(
                "Heartbeat signal and CSI "
                "timestamp lengths differ: "
                f"{len(heartbeat_signal)} "
                f"vs "
                f"{len(csi_time)}."
            )

        # ----------------------------------------------------
        # Sort CSI and GT timestamps
        #
        # Normally they are already sorted, but doing this
        # here prevents subtle timestamp-order problems.
        # ----------------------------------------------------

        csi_order = np.argsort(
            csi_time
        )

        csi_time = csi_time[
            csi_order
        ]

        heartbeat_signal = (
            heartbeat_signal[
                csi_order
            ]
        )

        gt_order = np.argsort(
            gt_time
        )

        gt_time = gt_time[
            gt_order
        ]

        # ----------------------------------------------------
        # Apply manual GT offset
        #
        # Positive:
        #     move GT later
        #
        # Negative:
        #     move GT earlier
        # ----------------------------------------------------

        gt_time = (
            gt_time
            + manual_gt_offset
        )

        # ----------------------------------------------------
        # Validate BPM samples
        # ----------------------------------------------------

        if gt_type == "bpm_samples":

            if gt_bpm is None:

                raise ValueError(
                    "gt_bpm must be provided when "
                    "gt_type='bpm_samples'."
                )

            gt_bpm = np.asarray(
                gt_bpm,
                dtype=float,
            )

            if len(gt_bpm) != len(
                gt_time
            ):

                raise ValueError(
                    "Ground-truth timestamp/BPM "
                    "length mismatch: "
                    f"{len(gt_time)} timestamps "
                    f"vs "
                    f"{len(gt_bpm)} BPM samples."
                )

            # Apply the same timestamp ordering
            gt_bpm = gt_bpm[
                gt_order
            ]

        elif gt_type == "beats":

            # No BPM array required
            gt_bpm = None

        else:

            raise ValueError(
                "gt_type must be either "
                "'bpm_samples' or 'beats'."
            )

        # ----------------------------------------------------
        # Determine common analysis interval
        # ----------------------------------------------------

        analysis_start = max(
            csi_time[0],
            gt_time[0],
        )

        analysis_end = min(
            csi_time[-1],
            gt_time[-1],
        )

        if (
            analysis_end
            <= analysis_start
        ):

            print(
                "WARNING: No overlapping "
                "CSI/GT analysis interval."
            )

            self.results = pd.DataFrame()

            return self.results

        # ----------------------------------------------------
        # Generate sliding windows
        # ----------------------------------------------------

        results = []

        starts = np.arange(
            analysis_start,
            analysis_end
            - window_seconds
            + 1e-9,
            step_seconds,
        )

        # ----------------------------------------------------
        # Analyze each window
        # ----------------------------------------------------

        for start in starts:

            end = (
                start
                + window_seconds
            )

            # ----------------------------------------------
            # CSI samples in window
            # ----------------------------------------------

            csi_mask = (
                (csi_time >= start)
                &
                (csi_time < end)
            )

            window_signal = (
                heartbeat_signal[
                    csi_mask
                ]
            )

            window_times = (
                csi_time[
                    csi_mask
                ]
            )

            # Need at least 5 seconds of CSI data
            if len(window_signal) < int(
                self.fs * 5
            ):

                continue

            if len(window_times) < 2:

                continue

            actual_duration = (
                window_times[-1]
                - window_times[0]
            )

            if actual_duration <= 0:

                continue

            # ----------------------------------------------
            # CSI BPM
            # ----------------------------------------------

            (
                csi_bpm,
                csi_power,
                peak_hz,
            ) = self.compute_bpm_from_psd(
                window_signal,
                low=low,
                high=high,
            )

            if csi_bpm <= 0:

                continue

            # ----------------------------------------------
            # Ground truth
            # ----------------------------------------------

            if gt_type == "beats":

                (
                    gt_bpm_val,
                    n_gt,
                ) = self.compute_ground_truth_bpm(
                    gt_time,
                    start,
                    end,
                    min_beats=min_beats,
                )

            else:

                (
                    gt_bpm_val,
                    n_gt,
                ) = (
                    self
                    .compute_ground_truth_bpm_from_samples(
                        gt_time,
                        gt_bpm,
                        start,
                        end,
                        min_samples=min_gt_samples,
                    )
                )

            if gt_bpm_val <= 0:

                continue

            # ----------------------------------------------
            # Error
            # ----------------------------------------------

            bpm_error = (
                csi_bpm
                - gt_bpm_val
            )

            bpm_abs_error = abs(
                bpm_error
            )

            # ----------------------------------------------
            # Store result
            # ----------------------------------------------

            results.append(
                {
                    "window_start": float(
                        start
                    ),
                    "window_end": float(
                        end
                    ),
                    "window_center": float(
                        (start + end)
                        / 2.0
                    ),
                    "csi_samples": int(
                        len(window_signal)
                    ),
                    "actual_duration": float(
                        actual_duration
                    ),
                    "csi_bpm": float(
                        csi_bpm
                    ),
                    "gt_bpm": float(
                        gt_bpm_val
                    ),
                    "csi_peak_hz": float(
                        peak_hz
                    ),
                    "csi_power": float(
                        csi_power
                    ),
                    "n_gt_samples": int(
                        n_gt
                    ),
                    "bpm_error": float(
                        bpm_error
                    ),
                    "bpm_abs_error": float(
                        bpm_abs_error
                    ),
                }
            )

        # ----------------------------------------------------
        # Results DataFrame
        # ----------------------------------------------------

        self.results = pd.DataFrame(
            results
        )

        # ----------------------------------------------------
        # Percentage error
        # ----------------------------------------------------

        if not self.results.empty:

            self.results[
                "bpm_error_pct"
            ] = (
                self.results[
                    "bpm_error"
                ]
                / self.results[
                    "gt_bpm"
                ]
                * 100.0
            )

        return self.results

    # ========================================================
    # METRICS
    # ========================================================

    def print_metrics(self):

        if (
            self.results is None
            or self.results.empty
        ):

            print(
                "No valid windows available."
            )

            return None

        x = self.results[
            "csi_bpm"
        ].to_numpy()

        y = self.results[
            "gt_bpm"
        ].to_numpy()

        # Correlation only makes sense if both variables vary
        if (
            len(x) >= 2
            and np.std(x) > 0
            and np.std(y) > 0
        ):

            corr = float(
                np.corrcoef(
                    x,
                    y,
                )[0, 1]
            )

        else:

            corr = np.nan

        metrics = {
            "MAE (BPM)": float(
                np.mean(
                    self.results[
                        "bpm_abs_error"
                    ]
                )
            ),
            "RMSE (BPM)": float(
                np.sqrt(
                    np.mean(
                        self.results[
                            "bpm_error"
                        ] ** 2
                    )
                )
            ),
            "Correlation": corr,
            "Mean Error (BPM)": float(
                np.mean(
                    self.results[
                        "bpm_error"
                    ]
                )
            ),
            "Std Error (BPM)": float(
                np.std(
                    self.results[
                        "bpm_error"
                    ]
                )
            ),
            "Number of Windows": int(
                len(self.results)
            ),
        }

        print(
            "\n"
            + "=" * 60
        )

        print(
            "WINDOWED PSD METRICS"
        )

        print(
            "=" * 60
        )

        for key, value in metrics.items():

            if isinstance(
                value,
                float,
            ):

                print(
                    f"  {key:20s}: "
                    f"{value:.3f}"
                )

            else:

                print(
                    f"  {key:20s}: "
                    f"{value}"
                )

        return metrics

    # ========================================================
    # PLOTS
    # ========================================================

    def plot_results(
        self,
        output_file,
    ):

        if (
            self.results is None
            or self.results.empty
        ):

            print(
                "No results to plot."
            )

            return

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(14, 12),
        )

        t = self.results[
            "window_center"
        ]

        # ----------------------------------------------------
        # Plot 1: HR tracking
        # ----------------------------------------------------

        ax = axes[0]

        ax.plot(
            t,
            self.results[
                "csi_bpm"
            ],
            marker="o",
            label="CSI BPM",
        )

        ax.plot(
            t,
            self.results[
                "gt_bpm"
            ],
            marker="o",
            label="Ground Truth BPM",
        )

        ax.set_xlabel(
            "Time (s)"
        )

        ax.set_ylabel(
            "Heart Rate (BPM)"
        )

        ax.set_title(
            "Heart Rate Tracking Over Time"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3,
        )

        # ----------------------------------------------------
        # Plot 2: BPM error
        # ----------------------------------------------------

        ax = axes[1]

        ax.bar(
            self.results[
                "window_center"
            ],
            self.results[
                "bpm_error"
            ],
            width=3.5,
            alpha=0.7,
        )

        ax.axhline(
            0,
            linewidth=0.8,
        )

        mae = float(
            np.mean(
                self.results[
                    "bpm_abs_error"
                ]
            )
        )

        ax.axhline(
            mae,
            linestyle="--",
            label=(
                f"MAE magnitude = "
                f"{mae:.2f} BPM"
            ),
        )

        ax.axhline(
            -mae,
            linestyle="--",
        )

        ax.set_xlabel(
            "Time (s)"
        )

        ax.set_ylabel(
            "BPM Error "
            "(CSI - GT)"
        )

        ax.set_title(
            "BPM Error Over Time"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3,
        )

        # ----------------------------------------------------
        # Plot 3: CSI vs GT
        # ----------------------------------------------------

        ax = axes[2]

        ax.scatter(
            self.results[
                "gt_bpm"
            ],
            self.results[
                "csi_bpm"
            ],
            s=50,
            alpha=0.7,
        )

        min_bpm = min(
            self.results[
                "gt_bpm"
            ].min(),
            self.results[
                "csi_bpm"
            ].min(),
        ) - 5

        max_bpm = max(
            self.results[
                "gt_bpm"
            ].max(),
            self.results[
                "csi_bpm"
            ].max(),
        ) + 5

        # Perfect agreement line
        ax.plot(
            [
                min_bpm,
                max_bpm,
            ],
            [
                min_bpm,
                max_bpm,
            ],
            linestyle="--",
            label="Perfect agreement",
        )

        x = self.results[
            "csi_bpm"
        ].to_numpy()

        y = self.results[
            "gt_bpm"
        ].to_numpy()

        if (
            len(x) >= 2
            and np.std(x) > 0
            and np.std(y) > 0
        ):

            corr = float(
                np.corrcoef(
                    x,
                    y,
                )[0, 1]
            )

        else:

            corr = np.nan

        ax.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
        )

        ax.set_xlabel(
            "Ground Truth BPM"
        )

        ax.set_ylabel(
            "CSI BPM"
        )

        ax.set_title(
            "CSI vs Ground Truth BPM"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3,
        )

        plt.suptitle(
            "Windowed PSD Heart Rate Analysis",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Saved plot: "
            f"{output_file}"
        )


# ============================================================
# MAIN
# ============================================================


def main():

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    # Change these two paths for each experiment.
    #
    # Linux example:
    #
    # csi_path = (
    #     "captures/Allen-S,0,1/"
    #     "csi_capture_20260902_215135.csv"
    # )
    #
    # gt_path = (
    #     "captures/Allen-S,0,1/"
    #     "ground_truth_20260902_215135.csv"
    # )

    csi_path = (
        r"captures\Allen-S,0,1"
        r"\csi_capture_20260820_135250.csv"
    )

    gt_path = (
        r"captures\Allen-S,0,1"
        r"\ground_truth_20260820_135247.csv"
    )

    # --------------------------------------------------------
    # Heartbeat band
    # --------------------------------------------------------

    LOW = 1.2
    HIGH = 2.8

    # 1.2 Hz = 72 BPM
    # 2.8 Hz = 168 BPM
    #
    # If you need the full 30-220 BPM ground-truth range for
    # CSI estimation, change these values accordingly.
    #
    # Current values are retained from the original pipeline.

    ORDER = 4

    # Number of best channels to fuse
    TOP_N = 5

    # --------------------------------------------------------
    # Window configuration
    # --------------------------------------------------------

    WINDOW_SECONDS = 20.0
    STEP_SECONDS = 5.0

    # --------------------------------------------------------
    # Ground truth configuration
    # --------------------------------------------------------

    # CURRENT Arduino + two-sensor ground truth format
    GT_TYPE = "bpm_samples"

    # Minimum number of valid GT BPM samples per window
    MIN_GT_SAMPLES = 3

    # Legacy beat-based mode uses this value
    MIN_BEATS = 3

    # --------------------------------------------------------
    # Timestamp offset
    # --------------------------------------------------------

    # Set this only if CSI and GT timestamps require manual
    # alignment.
    #
    # Positive values:
    #     move GT later
    #
    # Negative values:
    #     move GT earlier

    MANUAL_GT_OFFSET_SECONDS = 0.0

    # --------------------------------------------------------
    # Phase differences
    # --------------------------------------------------------

    PHASE_LAGS = (
        1,
        2,
        4,
        8,
        16,
        32,
    )

    # --------------------------------------------------------
    # Methods
    # --------------------------------------------------------

    methods = [
        (
            "Amplitude",
            "amplitude",
        ),
        (
            "Phase",
            "phase",
        ),
        (
            "Phase + Common Removal",
            "phase_common",
        ),
        (
            "Phase Differences",
            "phase_diff",
        ),
    ]

    # ========================================================
    # START
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "WINDOWED PSD HEART RATE ANALYSIS"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # LOAD CSI
    # ========================================================

    print(
        f"\nLoading CSI: "
        f"{csi_path}"
    )

    loader = CSIDataLoader(
        csi_path
    )

    (
        csi_data,
        metadata,
    ) = loader.load()

    fs = float(
        metadata["rate"]
    )

    print(
        f"  {metadata['n_packets']} "
        f"packets, "
        f"{metadata['n_subcarriers']} "
        f"subcarriers, "
        f"{fs:.3f} Hz"
    )

    # --------------------------------------------------------
    # CSI timestamps
    # --------------------------------------------------------

    if (
        hasattr(loader, "df")
        and loader.df is not None
        and "ts" in loader.df
    ):

        csi_raw_timestamps = (
            loader.df[
                "ts"
            ].to_numpy(
                dtype=float
            )
        )

    else:

        print(
            "WARNING: No CSI timestamp "
            "column found; using "
            "synthetic timestamps."
        )

        csi_raw_timestamps = (
            np.arange(
                len(csi_data)
            )
            / fs
        )

    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================

    print(
        f"\nLoading ground truth: "
        f"{gt_path}"
    )

    gt_df = pd.read_csv(
        gt_path
    )

    print(
        f"  {len(gt_df)} "
        f"ground-truth samples"
    )

    # --------------------------------------------------------
    # Validate timestamp column
    # --------------------------------------------------------

    if "ts" not in gt_df.columns:

        raise ValueError(
            "Ground-truth CSV must "
            "contain a 'ts' column."
        )

    # --------------------------------------------------------
    # Validate current BPM-sample format
    # --------------------------------------------------------

    if GT_TYPE == "bpm_samples":

        if "bpm" not in gt_df.columns:

            raise ValueError(
                "Ground-truth CSV must "
                "contain a 'bpm' column "
                "when GT_TYPE='bpm_samples'."
            )

        gt_raw_bpm = gt_df[
            "bpm"
        ].to_numpy(
            dtype=float
        )

    else:

        gt_raw_bpm = None

    gt_raw_timestamps = (
        gt_df[
            "ts"
        ].to_numpy(
            dtype=float
        )
    )

    # ========================================================
    # TIMESTAMP ALIGNMENT
    # ========================================================

    (
        csi_common,
        gt_common,
        time_mode,
    ) = (
        WindowedPSDAnalyzer
        .prepare_timebases(
            csi_raw_timestamps,
            gt_df,
        )
    )

    # --------------------------------------------------------
    # Because prepare_timebases sorts the timestamps, we need
    # to keep BPM values paired with their sorted GT timestamps.
    # --------------------------------------------------------

    gt_order = np.argsort(
        gt_raw_timestamps
    )

    gt_raw_timestamps = (
        gt_raw_timestamps[
            gt_order
        ]
    )

    if (
        GT_TYPE
        == "bpm_samples"
    ):

        gt_common_bpm = (
            gt_raw_bpm[
                gt_order
            ]
        )

    else:

        gt_common_bpm = None

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # prepare_timebases() returns gt_common in the sorted order
    # of the GT timestamps. The BPM array above is therefore
    # explicitly sorted using the same order.
    # --------------------------------------------------------

    print(
        f"\nTimestamp alignment mode: "
        f"{time_mode}"
    )

    print(
        f"CSI duration on analysis axis: "
        f"{csi_common[-1] - csi_common[0]:.2f} s"
    )

    print(
        f"GT duration on analysis axis: "
        f"{gt_common[-1] - gt_common[0]:.2f} s"
    )

    if GT_TYPE == "bpm_samples":

        valid_gt = (
            np.isfinite(
                gt_common_bpm
            )
            & (
                gt_common_bpm >= 30
            )
            & (
                gt_common_bpm <= 220
            )
        )

        print(
            f"Valid GT BPM samples: "
            f"{np.sum(valid_gt)} / "
            f"{len(gt_common_bpm)}"
        )

        if np.any(valid_gt):

            print(
                f"GT BPM range: "
                f"{np.min(gt_common_bpm[valid_gt]):.1f} "
                f"to "
                f"{np.max(gt_common_bpm[valid_gt]):.1f} BPM"
            )

    # ========================================================
    # ANALYZE EACH METHOD
    # ========================================================

    all_results = {}
    summary_rows = []

    for (
        display_name,
        method,
    ) in methods:

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"METHOD: "
            f"{display_name}"
        )

        print(
            "-" * 70
        )

        # ----------------------------------------------------
        # Create analyzer
        # ----------------------------------------------------

        analyzer = (
            WindowedPSDAnalyzer(
                fs
            )
        )

        # ----------------------------------------------------
        # Extract heartbeat
        # ----------------------------------------------------

        (
            heartbeat,
            quality,
            top_names,
        ) = (
            analyzer
            .extract_heartbeat_signal(
                csi_data,
                method=method,
                low=LOW,
                high=HIGH,
                order=ORDER,
                top_n=TOP_N,
                phase_lags=PHASE_LAGS,
            )
        )

        # ----------------------------------------------------
        # Print selected channels
        # ----------------------------------------------------

        print(
            f"Top {len(top_names)} "
            f"channels:"
        )

        for name in top_names:

            print(
                f"  {name}"
            )

        # ----------------------------------------------------
        # Save channel quality
        # ----------------------------------------------------

        if not quality.empty:

            best = quality.iloc[0]

            print(
                f"Best quality candidate: "
                f"{best['name']} | "
                f"score="
                f"{best['quality_score']:.2f} | "
                f"peak="
                f"{best['peak_bpm']:.2f} BPM"
            )

            quality.to_csv(
                f"windowed_quality_"
                f"{method}.csv",
                index=False,
            )

        # ====================================================
        # WINDOWED ANALYSIS
        # ====================================================

        results = (
            analyzer.analyze_windows(
                heartbeat_signal=heartbeat,
                csi_timestamps=csi_common,
                gt_time=gt_common,
                gt_bpm=gt_common_bpm,
                window_seconds=WINDOW_SECONDS,
                step_seconds=STEP_SECONDS,
                low=LOW,
                high=HIGH,
                min_beats=MIN_BEATS,
                min_gt_samples=MIN_GT_SAMPLES,
                manual_gt_offset=(
                    MANUAL_GT_OFFSET_SECONDS
                ),
                gt_type=GT_TYPE,
            )
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = (
            analyzer.print_metrics()
        )

        # ----------------------------------------------------
        # Save window results
        # ----------------------------------------------------

        results_file = (
            f"windowed_results_"
            f"{method}.csv"
        )

        results.to_csv(
            results_file,
            index=False,
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plot_file = (
            f"windowed_psd_"
            f"{method}.png"
        )

        analyzer.plot_results(
            plot_file
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        all_results[
            display_name
        ] = {
            "method": method,
            "heartbeat": heartbeat,
            "quality": quality,
            "top_names": top_names,
            "results": results,
            "metrics": metrics,
        }

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        if metrics is not None:

            summary_rows.append(
                {
                    "Method": display_name,
                    "MAE_BPM": (
                        metrics[
                            "MAE (BPM)"
                        ]
                    ),
                    "RMSE_BPM": (
                        metrics[
                            "RMSE (BPM)"
                        ]
                    ),
                    "Correlation": (
                        metrics[
                            "Correlation"
                        ]
                    ),
                    "Mean_Error_BPM": (
                        metrics[
                            "Mean Error (BPM)"
                        ]
                    ),
                    "Windows": (
                        metrics[
                            "Number of Windows"
                        ]
                    ),
                    "Top_Channels": (
                        "; ".join(
                            top_names
                        )
                    ),
                }
            )

    # ========================================================
    # METHOD COMPARISON
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "METHOD COMPARISON"
    )

    print(
        "=" * 70
    )

    if summary.empty:

        print(
            "No valid method results."
        )

        return

    summary = (
        summary
        .sort_values(
            "MAE_BPM",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # BEST METHOD
    # ========================================================

    best = summary.iloc[0]

    print(
        f"\nBEST METHOD: "
        f"{best['Method']}"
    )

    print(
        f"  Windowed MAE: "
        f"{best['MAE_BPM']:.2f} BPM"
    )

    print(
        f"  Windowed RMSE: "
        f"{best['RMSE_BPM']:.2f} BPM"
    )

    print(
        f"  Correlation: "
        f"{best['Correlation']:.3f}"
    )

    print(
        f"  Windows: "
        f"{best['Windows']}"
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary.to_csv(
        "windowed_psd_method_comparison.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        "  windowed_psd_method_comparison.csv"
    )

    print(
        "  windowed_results_<method>.csv"
    )

    print(
        "  windowed_quality_<method>.csv"
    )

    print(
        "  windowed_psd_<method>.png"
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    main()