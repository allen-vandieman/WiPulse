# windowed_psd_analysis.py
# Track heart rate over time from complex Wi-Fi CSI.
#
# Improvements over the previous version:
#   1. Uses the same quality-based channel selection for all four methods.
#   2. Scores channels BEFORE heartbeat bandpass filtering.
#   3. Uses actual CSI timestamps to define sliding windows.
#   4. Aligns CSI and ground-truth timestamps on a common time reference.
#   5. Uses a larger PSD segment and zero-padding for finer BPM estimates.
#   6. Uses quadratic interpolation around the PSD peak.
#   7. Tests amplitude, raw phase, common-phase-removed phase, and phase differences.
#   8. Searches phase differences over the full CSI matrix using several useful lags.
#   9. Uses polarity alignment before fusing the top channels.
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
    """Unified channel-processing pipeline used by all CSI methods."""

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
        but is currently unused so every supplied channel is treated fairly.
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
                    apply_hampel=True,
                    detrend_signal=True,
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
            candidate channels
            -> DC removal
            -> Hampel
            -> detrend
            -> quality score BEFORE bandpass
            -> rank channels
            -> bandpass
            -> polarity alignment
            -> top-N fusion
        """
        print(f"Building {method} candidates...")

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
            raise ValueError(
                "CSI and GT timestamps do not overlap."
            )

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
            label=f"MAE magnitude = {mae:.2f} BPM",
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


def main():
    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------
    csi_path = r"captures\Allen-S,0,1\csi_capture_20260820_135250.csv"
    gt_path = r"captures\Allen-S,0,1\ground_truth_20260820_135247.csv"

    LOW = 1.2
    HIGH = 2.8
    ORDER = 4
    TOP_N = 5

    WINDOW_SECONDS = 20.0
    STEP_SECONDS = 5.0

    # Set this only if CSI and GT timestamps are not on the same clock.
    # Positive values move GT later; negative values move GT earlier.
    MANUAL_GT_OFFSET_SECONDS = 0.0

    # Wider separations are included because your earlier good candidates
    # included differences such as 128-130, 56-59, 160-163, and 184-191.
    PHASE_LAGS = (1, 2, 4, 8, 16, 32)

    methods = [
        ("Amplitude", "amplitude"),
        ("Phase", "phase"),
        ("Phase + Common Removal", "phase_common"),
        ("Phase Differences", "phase_diff"),
    ]

    print("\n" + "=" * 70)
    print("WINDOWED PSD HEART RATE ANALYSIS")
    print("=" * 70)

    print(f"\nLoading CSI: {csi_path}")
    loader = CSIDataLoader(csi_path)
    csi_data, metadata = loader.load()

    fs = float(metadata["rate"])

    print(
        f"  {metadata['n_packets']} packets, "
        f"{metadata['n_subcarriers']} subcarriers, "
        f"{fs:.3f} Hz"
    )

    if hasattr(loader, "df") and loader.df is not None and "ts" in loader.df:
        csi_raw_timestamps = loader.df["ts"].to_numpy(dtype=float)
    else:
        print(
            "WARNING: No CSI timestamp column found; "
            "using synthetic timestamps."
        )
        csi_raw_timestamps = np.arange(len(csi_data)) / fs

    print(f"\nLoading ground truth: {gt_path}")
    gt_df = pd.read_csv(gt_path)
    print(f"  {len(gt_df)} beat records")

    if "ts" not in gt_df.columns:
        raise ValueError(
            "Ground-truth CSV must contain a 'ts' column of beat timestamps."
        )

    csi_common, gt_common, time_mode = (
        WindowedPSDAnalyzer.prepare_timebases(
            csi_raw_timestamps,
            gt_df,
        )
    )

    print(f"\nTimestamp alignment mode: {time_mode}")
    print(
        f"CSI duration on analysis axis: "
        f"{csi_common[-1] - csi_common[0]:.2f} s"
    )
    print(
        f"GT duration on analysis axis: "
        f"{gt_common[-1] - gt_common[0]:.2f} s"
    )

    all_results = {}
    summary_rows = []

    for display_name, method in methods:
        print("\n" + "-" * 70)
        print(f"METHOD: {display_name}")
        print("-" * 70)

        analyzer = WindowedPSDAnalyzer(fs)

        heartbeat, quality, top_names = (
            analyzer.extract_heartbeat_signal(
                csi_data,
                method=method,
                low=LOW,
                high=HIGH,
                order=ORDER,
                top_n=TOP_N,
                phase_lags=PHASE_LAGS,
            )
        )

        print(f"Top {len(top_names)} channels:")
        for name in top_names:
            print(f"  {name}")

        if not quality.empty:
            best = quality.iloc[0]
            print(
                f"Best quality candidate: {best['name']} | "
                f"score={best['quality_score']:.2f} | "
                f"peak={best['peak_bpm']:.2f} BPM"
            )

            quality.to_csv(
                f"windowed_quality_{method}.csv",
                index=False,
            )

        results = analyzer.analyze_windows(
            heartbeat_signal=heartbeat,
            csi_timestamps=csi_common,
            gt_time=gt_common,
            window_seconds=WINDOW_SECONDS,
            step_seconds=STEP_SECONDS,
            low=LOW,
            high=HIGH,
            min_beats=3,
            manual_gt_offset=MANUAL_GT_OFFSET_SECONDS,
        )

        metrics = analyzer.print_metrics()

        results_file = f"windowed_results_{method}.csv"
        results.to_csv(results_file, index=False)

        plot_file = f"windowed_psd_{method}.png"
        analyzer.plot_results(plot_file)

        all_results[display_name] = {
            "method": method,
            "heartbeat": heartbeat,
            "quality": quality,
            "top_names": top_names,
            "results": results,
            "metrics": metrics,
        }

        if metrics is not None:
            summary_rows.append(
                {
                    "Method": display_name,
                    "MAE_BPM": metrics["MAE (BPM)"],
                    "RMSE_BPM": metrics["RMSE (BPM)"],
                    "Correlation": metrics["Correlation"],
                    "Mean_Error_BPM": metrics["Mean Error (BPM)"],
                    "Windows": metrics["Number of Windows"],
                    "Top_Channels": "; ".join(top_names),
                }
            )

    summary = pd.DataFrame(summary_rows)

    print("\n" + "=" * 70)
    print("METHOD COMPARISON")
    print("=" * 70)

    if summary.empty:
        print("No valid method results.")
        return

    summary = summary.sort_values(
        "MAE_BPM",
        ascending=True,
    ).reset_index(drop=True)

    print(summary.to_string(index=False))

    best = summary.iloc[0]

    print(f"\nBEST METHOD: {best['Method']}")
    print(f"  Windowed MAE: {best['MAE_BPM']:.2f} BPM")
    print(f"  Windowed RMSE: {best['RMSE_BPM']:.2f} BPM")
    print(f"  Correlation: {best['Correlation']:.3f}")
    print(f"  Windows: {best['Windows']}")

    summary.to_csv(
        "windowed_psd_method_comparison.csv",
        index=False,
    )

    print("\nSaved:")
    print("  windowed_psd_method_comparison.csv")
    print("  windowed_results_<method>.csv")
    print("  windowed_quality_<method>.csv")
    print("  windowed_psd_<method>.png")


if __name__ == "__main__":
    main()
