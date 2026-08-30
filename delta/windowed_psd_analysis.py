# windowed_psd_analysis.py
# Track heart rate over time from complex Wi-Fi CSI.
#
# Improvements over the previous version:
#   1. Uses the same quality-based channel selection for all four methods.
#   2. Scores_channels_before_heartbeat_bandpass_filtering.
#   3. Uses_actual_CSI_timestamps_to_define_sliding_windows.
#   4. Aligns_CSI_and_ground_truth_timestamps_on_a_common_time_reference.
#   5. Uses_a_larger_PSD_segment_and_zero_padding_for_finer_BPM_estimates.
#   6. Uses_quadratic_interpolation_around_the_PSD_peak.
#   7. Tests_amplitude,_raw_phase,_common_phase_removed_phase,_phase_difference.
#   8. Searches_phase_differences_over_the_full_CSI_matrix_using_several_useful_lags.
#   9. Uses_polarity_alignment_before_fusing_the_top_channels.
#
# Run:
#   python windowed_psd_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import detrend

from read_csv_csi import CSIDataLoader

class HeartbeatPipeline:
    """Unified_channel_processing_pipeline_used_by_all_CSI_methods."""

    def __init__(self, fs=100.0):
        self.fs = float(fs)

    def extract_amplitude(self, csi_data):
        return np.abs(csi_data)

    def extract_phase(self, csi_data, unwrap=True):
        phase = np.angle(csi_data)
        if unwrap:
            phase = np.unwrap(phase, axis=0)
        return phase

    def remove_common_phase(self, phase_data):
        common_phase = np.median(phase_data, axis=1)
        return phase_data - common_phase[:, np.newaxis]

    def apply_hampel(self, x, window_size=11, threshold=3.0):
        """Simple Hampel outlier filter."""
        if window_size % 2 == 0:
            window_size += 1

        half = window_size // 2
        y = np.asarray(x, dtype=float).copy()

        for i in range(len(x)):
            start = max(0, i - half)
            end = min(len(x), i + half + 1)
            w = x[start:end]

            med = np.median(w)
            mad = np.median(np.abs(w - med))

            if mad > 0:
                z = abs(x[i] - med) / (1.4826 * mad)
                if z > threshold:
                    y[i] = med

        return y

    def bandpass_filter(self, x, low, high, order):
        nyquist = self.fs / 2.0

        if not (0 < low < high < nyquist):
            raise ValueError(
                f"Invalid bandpass {low}-{high} Hz for fs={self.fs:.3f} Hz"
            )

        sos = signal.butter(
            order,
            [low / nyquist, high / nyquist],
            btype="bandpass",
            output="sos",
        )
        return signal.sosfiltfilt(sos, x - np.mean(x))

    def compute_quality_score(self, x, hr_range):
        """
        Score a cleaned, detrended signal BEFORE heartbeat bandpass filtering.

        Score = dominant heartbeat-band amplitude / median out-of-band noise.

        Returns:
            score, peak_freq_hz, peak_amp
        """
        x = np.asarray(x, dtype=float)

        if len(x) < 16 or np.std(x) == 0:
            return 0.0, 0.0, 0.0

        spectrum = np.abs(np.fft.rfft(x - np.mean(x)))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / self.fs)

        low, high = hr_range

        hr_mask = (freqs >= low) & (freqs <= high)

        if not np.any(hr_mask):
            return 0.0, 0.0, 0.0

        hr_spec = spectrum[hr_mask]
        hr_freqs = freqs[hr_mask]

        peak_idx = int(np.argmax(hr_spec))
        peak_amp = float(hr_spec[peak_idx])
        peak_freq = float(hr_freqs[peak_idx])

        noise_mask = (
            (freqs >= 0.1)
            & (freqs <= 5.0)
            & ~hr_mask
        )

        if np.any(noise_mask):
            noise_floor = float(np.median(spectrum[noise_mask]))
        else:
            noise_floor = float(np.median(spectrum))

        score = peak_amp / (noise_floor + 1e-10)
        return float(score), peak_freq, peak_amp

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
            processed_signal : heartbeat-band signal
            quality_signal   : cleaned/detrended signal BEFORE bandpass
        """
        x = np.asarray(x, dtype=float).copy()
        x -= np.mean(x)

        if apply_hampel:
            x = self.apply_hampel(x)

        if detrend_signal:
            x = detrend(x)

        quality_signal = x.copy()

        processed = self.bandpass_filter(
            x, low=low, high=high, order=order
        )

        std = np.std(processed)
        if std > 0:
            processed = processed / std

        return processed, quality_signal

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

        top_candidates can be used later if you want to reduce processing time,
        but is currently not used so every supplied channel is treated fairly.
        """
        processed_signals = {}
        quality_rows = []

        for idx, (name, x) in enumerate(zip(channel_names, channels), start=1):
            try:
                processed, quality_signal = self.process_channel(
                    x,
                    low=low,
                    high=high,
                    order=order,
                )

                score, peak_freq, peak_amp = self.compute_quality_score(
                    quality_signal,
                    hr_range=(low, high),
                )

                processed_signals[name] = processed
                quality_rows.append(
                    {
                        "name": name,
                        "quality_score": score,
                        "peak_freq_hz": peak_freq,
                        "peak_bpm": peak_freq * 60.0,
                        "peak_amp": peak_amp,
                        "std": float(np.std(processed)),
                    }
                )

            except Exception as exc:
                print(f"Warning: failed to process {name}: {exc}")

            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(channel_names)} channels")

        quality_df = pd.DataFrame(quality_rows)

        if quality_df.empty:
            return processed_signals, quality_df

        quality_df = quality_df.sort_values(
            "quality_score",
            ascending=False,
        ).reset_index(drop=True)

        return processed_signals, quality_df

    def fuse_top_channels(
        self,
        processed_signals,
        quality_df,
        top_n=5,
        align_polarity=True,
    ):
        """Select, polarity-align, and average the best channels."""
        if quality_df.empty:
            return np.array([]), []

        top_names = quality_df.head(top_n)["name"].tolist()
        top_signals = [
            processed_signals[name].copy()
            for name in top_names
            if name in processed_signals
        ]

        valid_names = [
            name for name in top_names
            if name in processed_signals
        ]

        if not top_signals:
            return np.array([]), []

        if align_polarity and len(top_signals) > 1:
            reference = top_signals[0]
            aligned = [reference]

            for x in top_signals[1:]:
                corr = np.corrcoef(reference, x)[0, 1]
                if np.isfinite(corr) and corr < 0:
                    x = -x
                aligned.append(x)

            top_signals = aligned

        fused = np.mean(top_signals, axis=0)

        std = np.std(fused)
        if std > 0:
            fused = (fused - np.mean(fused)) / std

        return fused, valid_names

# Note: The class above is slightly different structure than your file to maintain logic flow.
# I will provide the literal block update for the construction in your specific file below 
# so that both your 'HeartbeatPipeline' (in `windowed_psd_analysis.py`) 
# and your `get_candidates` (in `multi_capture_validation.py`) handle it.

class WindowedPSDAnalyzer:
    """Extract CSI heartbeat signals and compare sliding-window BPM to GT."""

    def __init__(self, fs=100.0):
        self.fs = float(fs)
        self.pipeline = HeartbeatPipeline(fs)
        self.results = None

    def build_channels(self, csi_data, method, phase_lags=(1, 2, 4, 8, 16, 32)):
        """Create named 1-D candidate channels for one processing method."""
        if method == "amplitude":
            data = self.pipeline.extract_amplitude(csi_data)
            names = [f"amp_sc{sc}" for sc in range(data.shape[1])]
            channels = [data[:, sc] for sc in range(data.shape[1])]

        elif method == "phase":
            data = self.pipeline.extract_phase(csi_data, unwrap=True)
            names = [f"phase_sc{sc}" for sc in range(data.shape[1])]
            channels = [data[:, sc] for sc in range(data.shape[1])]

        elif method == "phase_common":
            phase = self.pipeline.extract_phase(csi_data, unwrap=True)
            data = self.pipeline.remove_common_phase(phase)
            names = [f"common_sc{sc}" for sc in range(data.shape[1])]
            channels = [data[:, sc] for sc in range(data.shape[1])]

        elif method == "amplitude_phase":
            # --- NEW HYBRID METHOD ---
            # Combines 30% amplitude and 70% phase (normalized)
            amp = self.pipeline.extract_amplitude(csi_data)
            ph = self.pipeline.extract_phase(csi_data, unwrap=True)

            # Normalize both to ensure even weighting of the components
            amp_n = amp / (np.max(amp) + 1e-6)
            ph_n = ph / (np.max(ph) + 1e-6)

            mixed = (0.3 * amp_n) + (0.7 * ph_n)

            names = [f"hybrid_sc{sc}" for sc in range(mixed.shape[1])]
            channels = [mixed[:, sc] for sc in range(mixed.shape[1])]

        elif method == "phase_diff":
            phase = self.pipeline.extract_phase(csi_data, unwrap=True)

            channels = []
            names = []
            n_sc = phase.shape[1]

            for lag in phase_lags:
                if lag >= n_sc:
                    continue

                for i in range(n_sc - lag):
                    j = i + lag
                    channels.append(phase[:, i] - phase[:, j])
                    names.append(f"diff_{i}_{j}")

        else:
            raise ValueError(f"Unknown method: {method}")

        return channels, names

    def extract_heartbeat_signal(
        self,
        csi_data,
        method,
        low=1.2,
        high=2.8,
        order=4,
        top_n=5,
        phase_lags=(1, 2, 4, 8, 16, 32),
    ):
        """
        Extract one fused heartbeat waveform.

        The same pipeline is used for every method:
            candidate_channels
            ->_DC_removal
            ->_Hampel
            ->_detrend
            ->_quality_score_BEFORE_bandpass
            ->_rank_channels
            ->_bandpass
            ->_polarity_alignment
            ->_top-N_fusion
        """
        print(f"Building {method} candidates...")

        # Using the local builder to handle special cases like 'amplitude_phase'
        channels, names = self.build_channels(
            csi_data,
            method=method,
            phase_lags=phase_lags,
        )

        print(f"  Candidates: {len(names)}")

        processed, quality = self.pipeline.process_all_channels(
            channels,
            names,
            low=low,
            high=high,
            order=order,
        )

        heartbeat, top_names = self.pipeline.fuse_top_channels(
            processed,
            quality,
            top_n=top_n,
            align_polarity=True,
        )

        if heartbeat.size == 0:
            heartbeat = np.zeros(csi_data.shape[0])

        return heartbeat, quality, top_names

    @staticmethod
    def _quadratic_peak(freqs, psd, idx):
        """Refine a PSD peak using 3-point quadratic interpolation."""
        if idx <= 0 or idx >= len(psd) - 1:
            return float(freqs[idx])

        alpha = psd[idx - 1]
        beta = psd[idx]
        gamma = psd[idx + 1]

        denominator = alpha - 2.0 * beta + gamma

        if abs(denominator) < 1e-20:
            return float(freqs[idx])

        delta = 0.5 * (alpha - gamma) / denominator
        delta = float(np.clip(delta, -1.0, 1.0))

        df = freqs[1] - freqs[0]
        return float(freqs[idx] + delta * df)

    def compute_bpm_from_psd(
        self,
        x,
        low=1.2,
        high=2.8,
        nfft=4096,
    ):
        """
        Estimate BPM from the dominant PSD peak.

        The whole available window is used as nperseg when possible.
        nfft provides a denser frequency grid for peak interpolation.
        """
        x = np.asarray(x, dtype=float)

        if len(x) < max(16, int(self.fs * 3)):
            return 0.0, 0.0, 0.0

        x = x - np.mean(x)

        nperseg = len(x)
        nfft = max(int(nfft), nperseg)

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

        hr_mask = (freqs >= low) & (freqs <= high)

        if not np.any(hr_mask):
            return 0.0, 0.0, 0.0

        hr_freqs = freqs[hr_mask]
        hr_psd = psd[hr_mask]

        peak_idx = int(np.argmax(hr_psd))
        peak_freq = self._quadratic_peak(
            hr_freqs,
            hr_psd,
            peak_idx,
        )

        peak_power = float(hr_psd[peak_idx])
        bpm = peak_freq * 60.0

        return float(bpm), peak_power, float(peak_freq)

    @staticmethod
    def prepare_timebases(csi_timestamps, gt_df, gt_time_column="ts"):
        """
        Preserve the true timing relationship between CSI and ground truth.

        CSI and GT are captured on the same Pi and use the same system clock,
        so their absolute timestamps already share a common time reference.
        """

        csi_ts = np.asarray(csi_timestamps, dtype=float)

        if gt_time_column not in gt_df.columns:
            raise ValueError(
                f"Ground truth must contain '{gt_time_column}' timestamps."
            )

        gt_ts = np.asarray(
            gt_df[gt_time_column],
            dtype=float
        )

        if len(csi_ts) == 0 or len(gt_ts) == 0:
            raise ValueError(
                "Empty CSI or ground-truth timestamps."
            )

        # Check whether timestamps overlap
        overlap_start = max(csi_ts[0], gt_ts[0])
        overlap_end = min(csi_ts[-1], gt_ts[-1])

        if overlap_start >= overlap_end:
            raise ValueError("CSI and GT timestamps do not overlap.")

        # Shift BOTH datasets using exactly the same origin.
        # This preserves the real delay caused by manually pressing Start.
        origin = overlap_start

        csi_time = csi_ts - origin
        gt_time = gt_ts - origin

        return csi_time, gt_time, "shared system clock"

    @staticmethod
    def compute_ground_truth_bpm(
        gt_time,
        window_start,
        window_end,
        min_beats=3,
    ):
        """
        Compute mean GT BPM from RR intervals whose interval midpoints lie
        inside the current window.
        """
        beats = np.asarray(gt_time, dtype=float)

        if len(beats) < 2:
            return 0.0, 0

        rr = np.diff(beats)
        rr_mid = (beats[:-1] + beats[1:]) / 2.0

        mask = (
            (rr_mid >= window_start)
            & (rr_mid < window_end)
            & (rr > 0)
        )

        rr_window = rr[mask]

        if len(rr_window) < max(1, min_beats - 1):
            return 0.0, 0

        bpm = 60.0 / np.mean(rr_window)

        n_beats = int(np.sum(
            (beats >= window_start) & (beats < window_end)
        ))

        return float(bpm), n_beats

    def analyze_windows(
        self,
        heartbeat_signal,
        csi_timestamps,
        gt_time,
        window_seconds=20.0,
        step_seconds=5.0,
        low=1.2,
        high=2.8,
        min_beats=3,
        manual_gt_offset=0.0,
    ):
        """
        Track CSI BPM and GT BPM in actual timestamp-based sliding windows.
        """
        heartbeat_signal = np.asarray(heartbeat_signal)

        # CSI and GT already use the same common time origin.
        # Do NOT independently reset either timeline.
        csi_time = np.asarray(
            csi_timestamps,
            dtype=float
        )

        gt_time = np.asarray(
            gt_time,
            dtype=float
        ) + manual_gt_offset

        analysis_start = max(
            csi_time[0],
            gt_time[0]
        )

        analysis_end = min(
            csi_time[-1],
            gt_time[-1]
        )

        results = []

        starts = np.arange(
            analysis_start,
            analysis_end - window_seconds + 1e-9,
            step_seconds
        )

        for start in starts:
            end = start + window_seconds

            csi_mask = (
                (csi_time >= start)
                & (csi_time < end)
            )

            window_signal = heartbeat_signal[csi_mask]

            if len(window_signal) < int(self.fs * 5):
                continue

            actual_duration = (
                csi_time[csi_mask][-1]
                - csi_time[csi_mask][0]
            )

            if actual_duration <= 0:
                continue

            csi_bpm, csi_power, peak_hz = self.compute_bpm_from_psd(
                window_signal,
                low=low,
                high=high,
            )

            gt_bpm, n_beats = self.compute_ground_truth_bpm(
                gt_time,
                start,
                end,
                min_beats=min_beats,
            )

            if gt_bpm <= 0 or csi_bpm <= 0:
                continue

            results.append(
                {
                    "window_start": float(start),
                    "window_end": float(end),
                    "window_center": float((start + end) / 2.0),
                    "csi_samples": int(len(window_signal)),
                    "actual_duration": float(actual_duration),
                    "csi_bpm": float(csi_bpm),
                    "gt_bpm": float(gt_bpm),
                    "csi_peak_hz": float(peak_hz),
                    "csi_power": float(csi_power),
                    "n_gt_beats": int(n_beats),
                    "bpm_error": float(csi_bpm - gt_bpm),
                    "bpm_abs_error": float(abs(csi_bpm - gt_bpm)),
                }
            )

        self.results = pd.DataFrame(results)

        if not self.results.empty:
            self.results["bpm_error_pct"] = (
                self.results["bpm_error"]
                / self.results["gt_bpm"]
                * 100.0
            )

        return self.results

    def print_metrics(self):
        if self.results is None or self.results.empty:
            print("No valid windows available.")
            return None
        
        x = self.results["csi_bpm"].to_numpy()
        y = self.results["gt_bpm"].to_numpy()

        if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
        else:
            corr = np.nan

        metrics = {
            "MAE (BPM)": float(np.mean(self.results["bpm_abs_error"])),
            "RMSE (BPM)": float(
                np.sqrt(np.mean(self.results["bpm_error"] ** 2))
            ),
            "Correlation": corr,
            "Mean Error (BPM)": float(
                np.mean(self.results["bpm_error"])
            ),
            "Std Error (BPM)": float(
                np.std(self.results["bpm_error"])
            ),
            "Number of Windows": int(len(self.results)),
        }

        print("\n" + "=" * 60)
        print("WINDOWED PSD METRICS")
        print("=" * 60)

        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.3f}")
            else:
                print(f"  {key:20s}: {value}")

        return metrics

    def plot_results(self, output_file):
        if self.results is None or self.results.empty:
            print("No results to plot.")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        t = self.results["window_center"]

        ax = axes[0]
        ax.plot(t, self.results["csi_bpm"], marker="o", label="CSI BPM")
        ax.plot(t, self.results["gt_bpm"], marker="o", label="Ground Truth BPM")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heart Rate (BPM)")
        ax.set_title("Heart Rate Tracking Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.bar(
            self.results["window_center"],
            self.results["bpm_error"],
            width=3.5,
            alpha=0.7,
        )
        ax.axhline(0, linewidth=0.8)
        mae = np.mean(self.results["bpm_abs_error"])
        ax.axhline(
            mae,
            linestyle="--",
            label=f"MAE_magnitude_=_ {mae:.2f} BPM",
        )
        ax.axhline(-mae, linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("BPM Error (CSI - GT)")
        ax.set_title("BPM Error Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.scatter(
            self.results["gt_bpm"],
            self.results["csi_bpm"],
            s=50,
            alpha=0.7,
        )

        min_bpm = min(
            self.results["gt_bpm"].min(),
            self.results["csi_bpm"].min(),
        ) - 5
        max_bpm = max(
            self.results["gt_bpm"].max(),
            self.results["csi_bpm"].max(),
        ) + 5

        ax.plot(
            [min_bpm, max_bpm],
            [min_bpm, max_bpm],
            linestyle="--",
            label="Perfect agreement",
        )

        x = self.results["csi_bpm"].to_numpy()
        y = self.results["gt_bpm"].to_numpy()

        if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
            corr = np.corrcoef(x, y)[0, 1]
        else:
            corr = np.nan

        ax.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
        )

        ax.set_xlabel("Ground Truth BPM")
        ax.set_ylabel("CSI BPM")
        ax.set_title("CSI vs Ground Truth BPM")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(
            "Windowed PSD Heart Rate Analysis",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot: {output_file}")
