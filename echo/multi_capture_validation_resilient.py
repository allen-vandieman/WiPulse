# multi_capture_validation.py
#
# Multi-capture CSI heartbeat pipeline validation.
#
# Parallel and vectorized version.
#
# Pipeline:
#
#   Raw CSI
#       |
#       v
#   Read packet timestamps
#       |
#       v
#   Inspect packet timing
#       |
#       v
#   Split only at hard gaps
#       |
#       v
#   Resample/interpolate small gaps to 100 Hz
#       |
#       v
#   Extract amplitude / phase candidates
#       |
#       v
#   Quality evaluation + channel fusion
#       |
#       v
#   Sliding-window PSD heart-rate estimation
#       |
#       v
#   Match CSI and ground-truth BPM using timestamps
#       |
#       v
#   Per-window and per_capture validation
#
# Outputs:
#
#   multi_capture_results.csv
#   multi_capture_window_results.csv
#   multi_capture_summary.csv
#   multi_capture_comparison.png


import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from read_csv_csi import CSIDataLoader
from windowed_psd_analysis import (
    HeartbeatPipeline,
    WindowedPSDAnalyzer
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_FS = 100.0

SMALL_GAP_SEC = 0.05
MAX_INTERPOLATE_GAP_SEC = 0.20
HARD_GAP_SEC = 0.20

MIN_SEGMENT_SECONDS = 6.0
WINDOW_SECONDS = 10.0
RESCUE_WINDOW_SECONDS = 8.0
EMERGENCY_WINDOW_SECONDS = 6.0
STEP_SECONDS = 5.0

TARGET_FS = 100.0

MAX_INTERPOLATION_RATIO = 0.10
MAX_WINDOW_HARD_GAPS = 0

LOW_HZ = 0.8
HIGH_HZ = 2.5

FILTER_ORDER = 4

TOP_N_CHANNELS = 5

MIN_GT_SAMPLES = 3

MAX_BPM = 200.0

# Number of worker processes.  None → cpu_count() - 1.
MAX_WORKERS = 3  # Initial None , On laptop and desktop use 3 because 4 cores.


# ============================================================
# MULTI-CAPTURE VALIDATOR
# ============================================================

class MultiCaptureValidator:

    def __init__(
        self,
        expected_fs=EXPECTED_FS,
        gap_multiplier=None,
        min_segment_seconds=MIN_SEGMENT_SECONDS,
        window_seconds=WINDOW_SECONDS,
        step_seconds=STEP_SECONDS,
        target_fs=TARGET_FS,
        max_interpolate_gap=MAX_INTERPOLATE_GAP_SEC,
        hard_gap=HARD_GAP_SEC,
        max_interpolation_ratio=MAX_INTERPOLATION_RATIO,
        low_hz=LOW_HZ,
        high_hz=HIGH_HZ,
        order=FILTER_ORDER,
        top_n=TOP_N_CHANNELS,
    ):
        self.expected_fs = float(expected_fs)
        self.gap_multiplier = gap_multiplier
        self.min_segment_seconds = float(min_segment_seconds)

        self.window_seconds = float(window_seconds)
        self.rescue_window_seconds = float(RESCUE_WINDOW_SECONDS)
        self.emergency_window_seconds = float(EMERGENCY_WINDOW_SECONDS)
        self.step_seconds = float(step_seconds)

        self.target_fs = float(target_fs)
        self.max_interpolate_gap = float(max_interpolate_gap)
        self.hard_gap = float(hard_gap)
        self.max_interpolation_ratio = float(max_interpolation_ratio)

        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order
        self.top_n = top_n

        self.capture_results = []
        self.window_results = []

    # ========================================================
    # LOAD CSI WITH TIMESTAMPS
    # ========================================================

    def load_capture(self, csi_path):
        loader = CSIDataLoader(csi_path)
        csi_data, metadata = loader.load()

        if hasattr(loader, "df") and "ts" in loader.df.columns:
            timestamps = (
                pd.to_numeric(loader.df["ts"], errors="coerce")
                .to_numpy(dtype=float)
            )
        else:
            raise ValueError(
                "Could not find per-packet 'ts' timestamps in CSI CSV."
            )

        valid = np.isfinite(timestamps)
        timestamps = timestamps[valid]
        csi_data = csi_data[valid]

        if len(timestamps) != len(csi_data):
            raise ValueError(
                "Timestamp and CSI packet counts do not match."
            )

        return csi_data, timestamps, metadata

    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================

    def load_ground_truth(self, gt_path, gt_type="bpm_samples"):
        gt_df = pd.read_csv(gt_path)

        if "ts" not in gt_df.columns:
            raise ValueError("Ground truth CSV has no 'ts' column.")

        gt_times = (
            pd.to_numeric(gt_df["ts"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

        gt_bpm = None

        if gt_type == "bpm_samples":
            if "bpm" not in gt_df.columns:
                raise ValueError("BPM sample CSV has no 'bpm' column.")
            gt_bpm = (
                pd.to_numeric(gt_df["bpm"], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            min_len = min(len(gt_times), len(gt_bpm))
            gt_times = gt_times[:min_len]
            gt_bpm = gt_bpm[:min_len]
        elif gt_type == "beats":
            pass
        else:
            raise ValueError(f"Unknown gt_type: {gt_type}")

        if len(gt_times) < 2:
            raise ValueError("Too few valid ground-truth samples.")

        return gt_times, gt_bpm

    # ========================================================
    # DETECT CONTINUOUS SEGMENTS
    # ========================================================

    def split_continuous_segments(self, timestamps, fs):
        timestamps = np.asarray(timestamps, dtype=float)

        if len(timestamps) < 2:
            return [], None, self.hard_gap

        dt = np.diff(timestamps)
        positive_dt = dt[dt > 0]

        if len(positive_dt) == 0:
            normal_dt = 1.0 / max(float(fs), self.target_fs)
        else:
            normal_dt = float(np.median(positive_dt))

        hard_threshold = max(self.hard_gap, normal_dt * 10.0)

        hard_gap_indices = np.where(
            (dt <= 0) | (dt > hard_threshold)
        )[0]

        segments = []
        start = 0

        for gap_index in hard_gap_indices:
            end = gap_index + 1
            if end > start:
                segments.append((start, end))
            start = end

        if start < len(timestamps):
            segments.append((start, len(timestamps)))

        return segments, normal_dt, hard_threshold

    # ========================================================
    # RESAMPLE SEGMENT  (vectorized)
    # ========================================================

    def resample_segment(self, csi_segment, ts_segment):
        """
        Reconstruct a segment onto a uniform target-FS time grid.

        Vectorized interpolation across all subcarriers at once.
        """
        csi_segment = np.asarray(csi_segment)
        ts_segment = np.asarray(ts_segment, dtype=float)

        empty_diag = {
            "original_packets": int(len(ts_segment)),
            "resampled_packets": int(len(ts_segment)),
            "interpolated_samples": 0,
            "interpolation_ratio": 0.0,
            "n_repairable_gaps": 0,
            "max_gap_sec": 0.0,
            "hard_gaps": 0,
        }

        if len(ts_segment) < 2:
            return csi_segment, ts_segment, empty_diag

        valid = np.isfinite(ts_segment)
        csi_segment = csi_segment[valid]
        ts_segment = ts_segment[valid]

        if len(ts_segment) < 2:
            return csi_segment, ts_segment, empty_diag

        # Sort by timestamp.
        order = np.argsort(ts_segment)
        ts_segment = ts_segment[order]
        csi_segment = csi_segment[order]

        dt = np.diff(ts_segment)
        positive = dt > 0

        if not np.all(positive):
            keep = np.concatenate(([True], positive))
            ts_segment = ts_segment[keep]
            csi_segment = csi_segment[keep]
            dt = np.diff(ts_segment)

        if len(ts_segment) < 2:
            return csi_segment, ts_segment, empty_diag

        step = 1.0 / self.target_fs
        start = float(ts_segment[0])
        end = float(ts_segment[-1])

        uniform_ts = np.arange(
            start, end + 0.5 * step, step, dtype=float
        )

        if len(uniform_ts) < 2:
            empty_diag["max_gap_sec"] = (
                float(np.max(dt)) if len(dt) else 0.0
            )
            return csi_segment, ts_segment, empty_diag

        # ------------------------------------------------
        # Vectorized interpolation across all subcarriers.
        # ------------------------------------------------
        idx = np.searchsorted(ts_segment, uniform_ts, side="right")
        idx = np.clip(idx, 1, len(ts_segment) - 1)
        i0 = idx - 1
        i1 = idx

        t0 = ts_segment[i0]
        t1 = ts_segment[i1]
        dt_pair = t1 - t0
        dt_pair[dt_pair == 0] = 1.0
        w1 = (uniform_ts - t0) / dt_pair
        w0 = 1.0 - w1

        # Broadcast (n_uniform, 1) against (n_uniform, n_sub).
        repaired = (
            w0[:, None] * csi_segment[i0]
            + w1[:, None] * csi_segment[i1]
        )

        # ------------------------------------------------
        # Diagnostics
        # ------------------------------------------------
        positions = np.searchsorted(ts_segment, uniform_ts)
        positions = np.clip(positions, 0, len(ts_segment) - 1)
        left = np.maximum(positions - 1, 0)
        right = positions
        dist_left = np.abs(uniform_ts - ts_segment[left])
        dist_right = np.abs(uniform_ts - ts_segment[right])
        nearest_dist = np.minimum(dist_left, dist_right)

        interpolated_mask = nearest_dist > (step * 0.35)
        interpolated_samples = int(np.sum(interpolated_mask))

        repairable_gaps = int(
            np.sum(
                (dt > step * 1.5)
                & (dt <= self.max_interpolate_gap)
            )
        )

        diagnostics = {
            "original_packets": int(len(ts_segment)),
            "resampled_packets": int(len(repaired)),
            "interpolated_samples": interpolated_samples,
            "interpolation_ratio": (
                interpolated_samples / len(repaired)
                if len(repaired) else 0.0
            ),
            "n_repairable_gaps": repairable_gaps,
            "max_gap_sec": float(np.max(dt)) if len(dt) else 0.0,
            "hard_gaps": int(np.sum(dt > self.hard_gap)),
        }

        return repaired, uniform_ts, diagnostics

    # ========================================================
    # BUILD SIGNAL CANDIDATES
    # ========================================================

    def get_candidates(self, csi_data, method, fs):
        pipeline = HeartbeatPipeline(fs)
        channels = []
        names = []

        if method == "amplitude":
            data = pipeline.extract_amplitude(csi_data)
            for sc in range(data.shape[1]):
                channels.append(data[:, sc])
                names.append(f"amp_sc{sc}")

        elif method == "phase":
            data = pipeline.extract_phase(csi_data, unwrap=True)
            for sc in range(data.shape[1]):
                channels.append(data[:, sc])
                names.append(f"phase_sc{sc}")

        elif method == "phase_common":
            phase = pipeline.extract_phase(csi_data, unwrap=True)
            data = pipeline.remove_common_phase(phase)
            for sc in range(data.shape[1]):
                channels.append(data[:, sc])
                names.append(f"common_sc{sc}")

        elif method == "phase_diff":
            phase = pipeline.extract_phase(csi_data, unwrap=True)
            n_sc = phase.shape[1]

            lags = [1, 2, 4, 8, 16, 32]
            # Cap candidates per lag to keep runtime bounded.
            step_sub = max(1, n_sc // 32)

            for lag in lags:
                if lag >= n_sc:
                    continue
                for i in range(0, n_sc - lag, step_sub):
                    j = i + lag
                    channels.append(phase[:, i] - phase[:, j])
                    names.append(f"diff_{i}_{j}")

        elif method == "amplitude_phase":
            amp = pipeline.extract_amplitude(csi_data)
            ph = pipeline.extract_phase(csi_data, unwrap=True)

            amp_norm = amp / (np.max(amp) + 1e-6)
            ph_norm = ph / (np.max(ph) + 1e-6)
            mixed = (0.3 * amp_norm) + (0.7 * ph_norm)

            for sc in range(mixed.shape[1]):
                channels.append(mixed[:, sc])
                names.append(f"hybrid_sc{sc}")

        else:
            raise ValueError(f"Unknown method: {method}")

        return channels, names

    # ========================================================
    # PROCESS ONE SEGMENT
    # ========================================================

    def process_segment(self, csi_segment, fs, method):
        pipeline = HeartbeatPipeline(fs)
        channels, names = self.get_candidates(csi_segment, method, fs)

        if len(channels) == 0:
            raise ValueError("No signal candidates generated.")

        processed, quality = pipeline.process_all_channels(
            channels, names,
            low=self.low_hz, high=self.high_hz, order=self.order,
        )

        if quality is None or len(quality) == 0:
            raise ValueError("No valid quality results.")

        fused, top_names = pipeline.fuse_top_channels(
            processed, quality, top_n=self.top_n, align_polarity=True
        )

        if fused is None or len(fused) == 0:
            raise ValueError("Channel fusion failed.")

        return fused, quality, top_names

    # ========================================================
    # GROUND TRUTH WINDOW HELPERS
    # ========================================================

    def get_gt_bpm_for_window(self, gt_times, ws, we):
        # gt_times is assumed sorted.
        i0 = np.searchsorted(gt_times, ws, side="left")
        i1 = np.searchsorted(gt_times, we, side="right")
        beats = gt_times[i0:i1]

        if len(beats) < MIN_GT_SAMPLES:
            return None, 0

        intervals = np.diff(beats)
        intervals = intervals[intervals > 0]

        if len(intervals) == 0:
            return None, 0

        mean_rr = np.mean(intervals)
        if mean_rr <= 0:
            return None, 0

        return 60.0 / mean_rr, len(beats)

    def get_gt_bpm_from_samples(self, gt_times, gt_bpm, ws, we):
        # gt_times is assumed sorted.
        i0 = np.searchsorted(gt_times, ws, side="left")
        i1 = np.searchsorted(gt_times, we, side="right")
        bpms = gt_bpm[i0:i1]

        if len(bpms) < MIN_GT_SAMPLES:
            return None, 0

        return float(np.median(bpms)), len(bpms)

    # ========================================================
    # ANALYZE ONE SEGMENT
    # ========================================================

    def analyze_segment(
        self,
        csi_segment,
        ts_segment,
        gt_times,
        fs,
        method,
        capture_name,
        segment_id,
        gt_bpm=None,
        gt_type="beats",
        timing_diagnostics=None,
    ):
        results = []

        if len(csi_segment) < 2:
            return results

        heartbeat, quality, top_names = self.process_segment(
            csi_segment, fs, method
        )

        if len(heartbeat) < 2:
            return results

        n = min(len(heartbeat), len(ts_segment))
        heartbeat = np.asarray(heartbeat[:n], dtype=float)
        ts_segment = np.asarray(ts_segment[:n], dtype=float)

        analyzer = WindowedPSDAnalyzer(self.target_fs)

        start_time = float(ts_segment[0])
        end_time = float(ts_segment[-1])

        analysis_start = max(start_time, float(gt_times[0]))
        analysis_end = min(end_time, float(gt_times[-1]))

        if analysis_end <= analysis_start:
            return results

        best_quality = (
            float(quality.iloc[0]["quality_score"])
            if quality is not None and len(quality) > 0
            else np.nan
        )

        window_specs = [
            ("standard", self.window_seconds),
            ("rescue", self.rescue_window_seconds),
            ("emergency", self.emergency_window_seconds),
        ]

        evaluated_centers = set()

        for window_class, window_len in window_specs:
            window_start = analysis_start

            while window_start + window_len <= analysis_end + 1e-9:
                window_end = window_start + window_len
                center_key = round(
                    (window_start + window_end) / 2.0, 3
                )

                if center_key in evaluated_centers:
                    window_start += self.step_seconds
                    continue

                mask = (
                    (ts_segment >= window_start)
                    & (ts_segment < window_end)
                )
                signal_window = heartbeat[mask]
                ts_window = ts_segment[mask]

                if len(signal_window) < int(
                    self.target_fs * window_len * 0.75
                ):
                    window_start += self.step_seconds
                    continue

                actual_duration = (
                    ts_window[-1] - ts_window[0]
                    if len(ts_window) > 1 else 0.0
                )

                if actual_duration < window_len * 0.70:
                    window_start += self.step_seconds
                    continue

                local_dt = np.diff(ts_window)
                positive_dt = local_dt[local_dt > 0]

                if len(positive_dt) == 0:
                    window_start += self.step_seconds
                    continue

                local_median_dt = float(np.median(positive_dt))
                local_fs = (
                    1.0 / local_median_dt
                    if local_median_dt > 0 else np.nan
                )

                expected_samples = max(
                    1, int(round(window_len * self.target_fs))
                )
                sample_ratio = len(signal_window) / expected_samples
                missing_ratio = max(0.0, 1.0 - sample_ratio)

                local_hard_gaps = int(
                    np.sum(local_dt > self.hard_gap)
                )
                if local_hard_gaps > MAX_WINDOW_HARD_GAPS:
                    window_start += self.step_seconds
                    continue

                capture_interp_ratio = (
                    float(timing_diagnostics.get("interpolation_ratio", 0.0))
                    if timing_diagnostics else 0.0
                )

                if sample_ratio < 0.80:
                    window_start += self.step_seconds
                    continue

                try:
                    csi_bpm, power, peak_hz = (
                        analyzer.compute_bpm_from_psd(
                            signal_window,
                            low=self.low_hz,
                            high=self.high_hz,
                        )
                    )
                except Exception:
                    window_start += self.step_seconds
                    continue

                if (
                    not np.isfinite(csi_bpm)
                    or csi_bpm <= 0
                    or csi_bpm > MAX_BPM
                ):
                    window_start += self.step_seconds
                    continue

                if gt_type == "beats":
                    gt_bpm_val, n_gt = self.get_gt_bpm_for_window(
                        gt_times, window_start, window_end
                    )
                elif gt_type == "bpm_samples" and gt_bpm is not None:
                    gt_bpm_val, n_gt = self.get_gt_bpm_from_samples(
                        gt_times, gt_bpm, window_start, window_end
                    )
                else:
                    window_start += self.step_seconds
                    continue

                if gt_bpm_val is None:
                    window_start += self.step_seconds
                    continue

                if np.isfinite(best_quality) and best_quality < 0:
                    signal_quality = "weak"
                elif (
                    sample_ratio < 0.90
                    or capture_interp_ratio > self.max_interpolation_ratio
                ):
                    signal_quality = "degraded"
                else:
                    signal_quality = "good"

                error = float(csi_bpm - gt_bpm_val)

                result = {
                    "capture": capture_name,
                    "method": method,
                    "segment": segment_id,
                    "window_class": window_class,
                    "window_start": float(window_start),
                    "window_end": float(window_end),
                    "window_center": float(
                        (window_start + window_end) / 2.0
                    ),
                    "window_seconds": float(window_len),
                    "csi_samples": int(len(signal_window)),
                    "actual_duration": float(actual_duration),
                    "local_median_fs": float(local_fs),
                    "sample_ratio": float(sample_ratio),
                    "missing_ratio": float(missing_ratio),
                    "capture_interpolation_ratio": float(
                        capture_interp_ratio
                    ),
                    "local_hard_gaps": local_hard_gaps,
                    "csi_bpm": float(csi_bpm),
                    "gt_bpm": float(gt_bpm_val),
                    "error": error,
                    "abs_error": abs(error),
                    "peak_hz": float(peak_hz),
                    "power": float(power),
                    "quality_score": best_quality,
                    "signal_quality": signal_quality,
                    "top_channels": "; ".join(top_names),
                    "n_gt_samples": int(n_gt),
                }

                results.append(result)
                evaluated_centers.add(center_key)
                window_start += self.step_seconds

        return results

    # ========================================================
    # VALIDATE ONE CAPTURE
    # ========================================================

    def validate_capture(
        self,
        csi_path,
        gt_path,
        method,
        gt_type="bpm_samples",
    ):
        capture_name = Path(csi_path).stem
        print(f"\n  Loading: {capture_name}")

        start_time = time.time()

        try:
            csi_data, timestamps, metadata = self.load_capture(csi_path)
            fs_loader = float(metadata.get("rate", self.expected_fs))

            gt_times, gt_bpm = self.load_ground_truth(
                gt_path, gt_type=gt_type
            )

            # Ensure GT is sorted for searchsorted-based lookup.
            gt_order = np.argsort(gt_times)
            gt_times = gt_times[gt_order]
            if gt_bpm is not None:
                gt_bpm = gt_bpm[gt_order]

            segments, normal_dt, gap_threshold = (
                self.split_continuous_segments(timestamps, fs_loader)
            )

            normal_fs = (
                1.0 / normal_dt
                if normal_dt and normal_dt > 0
                else self.target_fs
            )

            capture_duration = (
                timestamps[-1] - timestamps[0]
                if len(timestamps) > 1 else 0.0
            )
            effective_fs = (
                len(timestamps) / capture_duration
                if capture_duration > 0 else np.nan
            )

            timing_dt = np.diff(timestamps)
            positive_dt = timing_dt[timing_dt > 0]
            p95_dt = (
                float(np.percentile(positive_dt, 95))
                if len(positive_dt) else np.nan
            )
            max_dt = (
                float(np.max(positive_dt))
                if len(positive_dt) else np.nan
            )

            expected_packets = (
                capture_duration * self.target_fs
                if capture_duration > 0 else np.nan
            )
            estimated_loss_ratio = (
                max(0.0, 1.0 - (len(timestamps) / expected_packets))
                if np.isfinite(expected_packets) and expected_packets > 0
                else np.nan
            )

            print(f"    Packets: {len(csi_data)}")
            print(f"    Loader/effective FS: {fs_loader:.3f} Hz")
            print(f"    Effective received FS: {effective_fs:.3f} Hz")
            print(f"    Median timing FS: {normal_fs:.3f} Hz")
            print(f"    Target processing FS: {self.target_fs:.3f} Hz")

            if np.isfinite(estimated_loss_ratio):
                print(
                    f"    Estimated packet loss: "
                    f"{estimated_loss_ratio * 100:.2f}%"
                )

            if np.isfinite(p95_dt) and np.isfinite(max_dt):
                print(
                    f"    Timing p95/max gap: "
                    f"{p95_dt * 1000:.1f}/{max_dt * 1000:.1f} ms"
                )

            print(f"    Continuous segments: {len(segments)}")
            print(f"    GT samples: {len(gt_times)}")

            capture_windows = []
            valid_segments = 0

            for segment_id, (start_idx, end_idx) in enumerate(segments):
                csi_segment = csi_data[start_idx:end_idx]
                ts_segment = timestamps[start_idx:end_idx]

                if len(ts_segment) < 2:
                    continue

                duration = ts_segment[-1] - ts_segment[0]

                if duration < self.min_segment_seconds:
                    continue

                repaired_csi, repaired_ts, timing_diag = (
                    self.resample_segment(csi_segment, ts_segment)
                )

                valid_segments += 1

                print(
                    f"    Segment {segment_id}: "
                    f"{duration:.2f}s "
                    f"({len(csi_segment)} packets -> "
                    f"{len(repaired_csi)} samples), "
                    f"interpolation="
                    f"{timing_diag['interpolation_ratio'] * 100:.2f}%"
                )

                segment_results = self.analyze_segment(
                    repaired_csi,
                    repaired_ts,
                    gt_times,
                    self.target_fs,
                    method,
                    capture_name,
                    segment_id,
                    gt_bpm=gt_bpm,
                    gt_type=gt_type,
                    timing_diagnostics=timing_diag,
                )

                capture_windows.extend(segment_results)

            if len(capture_windows) == 0:
                elapsed = time.time() - start_time
                print("    FAIL: No valid windows.")
                return (
                    {
                        "capture": capture_name,
                        "method": method,
                        "success": False,
                        "failure_reason": "no_valid_windows",
                        "n_packets": len(csi_data),
                        "n_segments": len(segments),
                        "n_valid_segments": valid_segments,
                        "n_windows": 0,
                        "effective_fs": effective_fs,
                        "target_fs": self.target_fs,
                        "estimated_packet_loss_pct": (
                            estimated_loss_ratio * 100.0
                            if np.isfinite(estimated_loss_ratio)
                            else np.nan
                        ),
                        "p95_gap_ms": (
                            p95_dt * 1000.0
                            if np.isfinite(p95_dt) else np.nan
                        ),
                        "max_gap_ms": (
                            max_dt * 1000.0
                            if np.isfinite(max_dt) else np.nan
                        ),
                        "processing_seconds": elapsed,
                        "gt_type": gt_type,
                    },
                    [],
                )

            window_df = pd.DataFrame(capture_windows)
            errors = window_df["error"].to_numpy()
            abs_errors = window_df["abs_error"].to_numpy()

            correlation = np.nan
            if len(window_df) >= 2:
                correlation = (
                    window_df["csi_bpm"].corr(window_df["gt_bpm"])
                )

            elapsed = time.time() - start_time

            capture_result = {
                "capture": capture_name,
                "method": method,
                "success": True,
                "failure_reason": "",
                "n_packets": len(csi_data),
                "n_segments": len(segments),
                "n_valid_segments": valid_segments,
                "n_windows": len(window_df),
                "mean_mae": np.mean(abs_errors),
                "median_mae": np.median(abs_errors),
                "std_mae": np.std(abs_errors),
                "rmse": np.sqrt(np.mean(errors ** 2)),
                "mean_error": np.mean(errors),
                "worst_mae": np.max(abs_errors),
                "correlation": correlation,
                "normal_fs": normal_fs,
                "effective_fs": effective_fs,
                "target_fs": self.target_fs,
                "estimated_packet_loss_pct": (
                    estimated_loss_ratio * 100.0
                    if np.isfinite(estimated_loss_ratio)
                    else np.nan
                ),
                "p95_gap_ms": (
                    p95_dt * 1000.0
                    if np.isfinite(p95_dt) else np.nan
                ),
                "max_gap_ms": (
                    max_dt * 1000.0
                    if np.isfinite(max_dt) else np.nan
                ),
                "gap_threshold_ms": gap_threshold * 1000.0,
                "processing_seconds": elapsed,
                "gt_type": gt_type,
            }

            print(
                f"    OK: MAE={capture_result['mean_mae']:.2f} BPM, "
                f"Median={capture_result['median_mae']:.2f}, "
                f"Worst={capture_result['worst_mae']:.2f}, "
                f"Windows={capture_result['n_windows']}"
            )

            return capture_result, capture_windows

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"    ERROR: {e}")
            return (
                {
                    "capture": capture_name,
                    "method": method,
                    "success": False,
                    "failure_reason": str(e),
                    "processing_seconds": elapsed,
                    "gt_type": gt_type,
                },
                [],
            )

    # ========================================================
    # SEQUENTIAL TEST
    # ========================================================

    def run_full_test(self, captures, methods, gt_type="bpm_samples"):
        print()
        print("=" * 70)
        print("MULTI-CAPTURE PIPELINE VALIDATION (sequential)")
        print("=" * 70)
        print(
            f"Target processing rate: {self.target_fs:.1f} Hz | "
            f"Hard gap: {self.hard_gap:.3f} s | "
            f"Max interpolation gap: {self.max_interpolate_gap:.3f} s"
        )

        for method in methods:
            print()
            print(f"METHOD: {method.upper()}")
            print("-" * 70)

            method_results = []

            for csi_path, gt_path in captures:
                result, windows = self.validate_capture(
                    csi_path, gt_path, method, gt_type=gt_type
                )
                self.capture_results.append(result)
                method_results.append(result)
                self.window_results.extend(windows)

            method_df = pd.DataFrame(method_results)
            success_df = method_df[method_df["success"] == True]

            print()
            print(f"{method.upper()} SUMMARY")
            print(f"  Success rate: {len(success_df)}/{len(captures)}")

            if len(success_df) > 0:
                print(
                    f"  Mean_capture MAE: "
                    f"{success_df['mean_mae'].mean():.2f}"
                )
                print(
                    f"  Median_capture MAE: "
                    f"{success_df['median_mae'].median():.2f}"
                )
                print(
                    f"  Worst_capture MAE: "
                    f"{success_df['worst_mae'].max():.2f}"
                )

        return (
            pd.DataFrame(self.capture_results),
            pd.DataFrame(self.window_results),
        )

    # ========================================================
    # PARALLEL TEST
    # ========================================================

    def run_full_test_parallel(
        self,
        captures,
        methods,
        gt_type="bpm_samples",
        max_workers=None,
    ):
        """
        Same as run_full_test but each (capture, method) task runs in a
        separate worker process.  This is usually 4-8x faster on a modern
        machine, since each capture is fully independent.
        """
        if max_workers is None:
            max_workers = MAX_WORKERS

        if max_workers is None:
            import multiprocessing as mp
            max_workers = max(1, mp.cpu_count() - 1)

        print()
        print("=" * 70)
        print(f"MULTI-CAPTURE PIPELINE VALIDATION (parallel x{max_workers})")
        print("=" * 70)
        print(
            f"Target processing rate: {self.target_fs:.1f} Hz | "
            f"Hard gap: {self.hard_gap:.3f} s | "
            f"Max interpolation gap: {self.max_interpolate_gap:.3f} s"
        )

        kwargs = dict(
            expected_fs=self.expected_fs,
            min_segment_seconds=self.min_segment_seconds,
            window_seconds=self.window_seconds,
            step_seconds=self.step_seconds,
            target_fs=self.target_fs,
            max_interpolate_gap=self.max_interpolate_gap,
            hard_gap=self.hard_gap,
            max_interpolation_ratio=self.max_interpolation_ratio,
            low_hz=self.low_hz,
            high_hz=self.high_hz,
            order=self.order,
            top_n=self.top_n,
        )

        tasks = []
        for method in methods:
            for csi_path, gt_path in captures:
                tasks.append(
                    (csi_path, gt_path, method, gt_type, kwargs)
                )

        capture_results = []
        window_results = []

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_validate_one_task, t): t for t in tasks
            }

            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    method, result, windows = fut.result()
                except Exception as e:
                    print(f"  [{i}/{len(tasks)}] WORKER ERROR: {e}")
                    continue

                capture_results.append(result)
                window_results.extend(windows)

                ok = "OK" if result.get("success") else "FAIL"
                print(
                    f"  [{i}/{len(tasks)}] "
                    f"{result.get('capture', '?')} "
                    f"({method}) - {ok}"
                )

        return (
            pd.DataFrame(capture_results),
            pd.DataFrame(window_results),
        )


# ============================================================
# MODULE-LEVEL WORKER (required for ProcessPool on Windows)
# ============================================================

def _validate_one_task(task):
    (csi_path, gt_path, method, gt_type, kwargs) = task
    validator = MultiCaptureValidator(**kwargs)
    result, windows = validator.validate_capture(
        csi_path, gt_path, method, gt_type=gt_type
    )
    return method, result, windows


# ============================================================
# AUTO-DETECT GT TYPE
# ============================================================

def detect_gt_type(gt_path):
    df = pd.read_csv(gt_path, nrows=1)
    if "bpm" in df.columns:
        return "bpm_samples"
    return "beats"


# ============================================================
# METHOD SUMMARY
# ============================================================

def create_method_summary(results_df):
    rows = []
    methods = sorted(results_df["method"].unique())

    for method in methods:
        method_df = results_df[results_df["method"] == method]
        successful = method_df[method_df["success"] == True]

        total = len(method_df)
        success_count = len(successful)

        row = {
            "method": method,
            "captures_total": total,
            "captures_success": success_count,
            "success_rate": (
                success_count / total if total > 0 else np.nan
            ),
        }

        if success_count > 0:
            row.update({
                "mean_mae": successful["mean_mae"].mean(),
                "median_mae": successful["median_mae"].median(),
                "mae_std": successful["mean_mae"].std(),
                "mean_rmse": successful["rmse"].mean(),
                "worst_capture_mae": successful["worst_mae"].max(),
                "mean_correlation": successful["correlation"].mean(),
                "mean_windows": successful["n_windows"].mean(),
                "mean_processing_seconds": (
                    successful["processing_seconds"].mean()
                ),
            })

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        by=["median_mae", "mean_mae"], ascending=True
    )
    return summary


# ============================================================
# COMPARISON PLOTS
# ============================================================

def create_comparison_plot(results_df, summary, window_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    methods = summary["method"].tolist()

    ax = axes[0, 0]
    ax.bar(methods, summary["mean_mae"])
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean_capture MAE (BPM)")
    ax.set_title("Mean MAE Across Captures")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.bar(methods, summary["worst_capture_mae"])
    ax.set_xlabel("Method")
    ax.set_ylabel("Worst Capture MAE (BPM)")
    ax.set_title("Worst-Case Performance")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.bar(methods, summary["success_rate"] * 100.0)
    ax.set_xlabel("Method")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Capture Success Rate")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(window_df) > 0:
        for method in methods:
            df = window_df[window_df["method"] == method]
            if len(df) == 0:
                continue
            ax.scatter(
                df["gt_bpm"], df["csi_bpm"],
                alpha=0.5, s=20, label=method,
            )

        min_bpm = min(
            window_df["gt_bpm"].min(), window_df["csi_bpm"].min()
        )
        max_bpm = max(
            window_df["gt_bpm"].max(), window_df["csi_bpm"].max()
        )
        ax.plot([min_bpm, max_bpm], [min_bpm, max_bpm], "k--", alpha=0.5)

    ax.set_xlabel("Ground Truth BPM")
    ax.set_ylabel("CSI BPM")
    ax.set_title("Window-Level CSI vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Multi-Capture Pipeline Comparison",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("multi_capture_comparison.png", dpi=300)
    plt.close()

    print("\nSaved: multi_capture_comparison.png")


# ============================================================
# RUN TEST
# ============================================================

def run_test(gt_type="bpm_samples", parallel=True):
    captures = [
        (
            r"captures\07-09-2026\Allen-N,0,0-T1\csi_capture_20260907_211939.csv",
            r"captures\07-09-2026\Allen-N,0,0-T1\ground_truth_20260907_211942.csv"
        ),
        (
            r"captures\07-09-2026\Allen-N,0,1-T1\csi_capture_20260907_204311.csv",
            r"captures\07-09-2026\Allen-N,0,1-T1\ground_truth_20260907_204314.csv"
        ),
        (
            r"captures\07-09-2026\Allen-N,0,-1-T1\csi_capture_20260907_203329.csv",
            r"captures\07-09-2026\Allen-N,0,-1-T1\ground_truth_20260907_203332.csv"
        ),
        (
            r"captures\07-09-2026\Allen-S,0,0-T1\csi_capture_20260907_211154.csv",
            r"captures\07-09-2026\Allen-S,0,0-T1\ground_truth_20260907_211157.csv"
        ),
        (
            r"captures\07-09-2026\Allen-S,0,1-T1\csi_capture_20260907_205349.csv",
            r"captures\07-09-2026\Allen-S,0,1-T1\ground_truth_20260907_205352.csv"
        ),
        (
            r"captures\07-09-2026\Allen-S,0,-1-T1\csi_capture_20260907_210505.csv",
            r"captures\07-09-2026\Allen-S,0,-1-T1\ground_truth_20260907_210508.csv"
        ),
        (
            r"captures\06-09-2026\Allen-N,0,-1-T1\csi_capture_20260906_234541.csv",
            r"captures\06-09-2026\Allen-N,0,-1-T1\ground_truth_20260906_234544.csv"
        ),
        (
            r"captures\06-09-2026\Allen-S,0,1-T1\csi_capture_20260906_131236.csv",
            r"captures\06-09-2026\Allen-S,0,1-T1\ground_truth_20260906_131258.csv"
        ),
    ]

    methods = [
        "amplitude",
        "phase",
        "phase_common",
        "phase_diff",
        "amplitude_phase",
    ]

    validator = MultiCaptureValidator()

    if parallel:
        results_df, window_df = validator.run_full_test_parallel(
            captures, methods, gt_type=gt_type
        )
    else:
        results_df, window_df = validator.run_full_test(
            captures, methods, gt_type=gt_type
        )

    results_df.to_csv("multi_capture_results.csv", index=False)
    window_df.to_csv("multi_capture_window_results.csv", index=False)
    print("\nSaved: multi_capture_results.csv")
    print("Saved: multi_capture_window_results.csv")

    summary = create_method_summary(results_df)
    summary.to_csv("multi_capture_summary.csv", index=False)
    print("Saved: multi_capture_summary.csv")

    print()
    print("=" * 70)
    print("FINAL SUMMARY BY METHOD")
    print("=" * 70)
    print(summary.to_string(index=False))

    valid_summary = summary.dropna(subset=["median_mae"])
    if len(valid_summary) > 0:
        best = valid_summary.iloc[0]
        print()
        print("=" * 70)
        print(f"BEST METHOD: {best['method'].upper()}")
        print("=" * 70)
        print(f"Median MAE: {best['median_mae']:.2f} BPM")
        print(f"Mean MAE: {best['mean_mae']:.2f} BPM")
        print(f"Worst Capture MAE: {best['worst_capture_mae']:.2f} BPM")
        print(f"Success Rate: {best['success_rate'] * 100:.1f}%")

    create_comparison_plot(results_df, summary, window_df)

    print()
    print("=" * 70)
    print("INDIVIDUAL CAPTURE RESULTS")
    print("=" * 70)

    display_columns = [
        "capture", "method", "success",
        "n_segments", "n_valid_segments", "n_windows",
        "effective_fs", "estimated_packet_loss_pct",
        "mean_mae", "median_mae", "worst_mae",
        "correlation", "gt_type", "failure_reason",
    ]
    existing_columns = [
        c for c in display_columns if c in results_df.columns
    ]
    print(results_df[existing_columns].to_string(index=False))

    return results_df, window_df, summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results, windows, summary = run_test(
        gt_type="bpm_samples",
        parallel=True,
    )