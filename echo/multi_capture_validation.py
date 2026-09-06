# multi_capture_validation.py
#
# Multi-capture CSI heartbeat pipeline validation.
#
# Goal:
#   Find the most reliable processing pipeline across many captures.
#
# Pipeline:
#
#   Raw CSI
#       |
#       v
#   Read packet timestamps
#       |
#       v
#   Detect large packet gaps
#       |
#       v
#   Split into continuous segments
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from read_csv_csi import CSIDataLoader
from windowed_psd_analysis import (
    HeartbeatPipeline,
    WindowedPSDAnalyzer
)


# ============================================================
# CONFIGURATION
# ============================================================

# Expected packet interval.
# Your packet timing test showed a median interval of ~10 ms,
# which corresponds to approximately 100 Hz.
EXPECTED_FS = 100.0

# A gap larger than this multiple of the normal interval
# splits the recording into separate continuous segments.
GAP_MULTIPLIER = 1.5

# Ignore segments shorter than this.
MIN_SEGMENT_SECONDS = 10.0

# Sliding-window heart-rate analysis.
WINDOW_SECONDS = 10.0
STEP_SECONDS = 5.0

# Heart-rate search range.
LOW_HZ = 0.8
HIGH_HZ = 2.5

FILTER_ORDER = 4

# Number of quality-selected channels to fuse.
TOP_N_CHANNELS = 5

# Minimum GT samples/beats required inside a window.
MIN_GT_SAMPLES = 3

# Maximum acceptable CSI BPM.
MAX_BPM = 200.0


# ============================================================
# MULTI-CAPTURE VALIDATOR
# ============================================================

class MultiCaptureValidator:
    """
    Test multiple CSI heartbeat pipelines across multiple captures.

    The validator treats each capture independently:

        capture
            |
            +-- detect packet gaps
            |
            +-- split into continuous segments
            |
            +-- process each segment
            |
            +-- estimate HR in sliding windows
            |
            +-- compare with GT
    """

    def __init__(
        self,
        expected_fs=EXPECTED_FS,
        gap_multiplier=GAP_MULTIPLIER,
        min_segment_seconds=MIN_SEGMENT_SECONDS,
        window_seconds=WINDOW_SECONDS,
        step_seconds=STEP_SECONDS,
        low_hz=LOW_HZ,
        high_hz=HIGH_HZ,
        order=FILTER_ORDER,
        top_n=TOP_N_CHANNELS
    ):

        self.expected_fs = expected_fs
        self.gap_multiplier = gap_multiplier
        self.min_segment_seconds = min_segment_seconds

        self.window_seconds = window_seconds
        self.step_seconds = step_seconds

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
        """
        Load CSI data and recover the actual per-packet timestamps.
        """

        loader = CSIDataLoader(csi_path)

        csi_data, metadata = loader.load()

        if hasattr(loader, "df") and "ts" in loader.df.columns:

            timestamps = (
                pd.to_numeric(
                    loader.df["ts"],
                    errors="coerce"
                )
                .to_numpy(dtype=float)
            )

        else:

            raise ValueError(
                "Could not find per-packet 'ts' timestamps "
                "in CSI CSV."
            )

        # Remove invalid timestamp rows.
        valid = np.isfinite(timestamps)

        timestamps = timestamps[valid]
        csi_data = csi_data[valid]

        if len(timestamps) != len(csi_data):

            raise ValueError(
                "Timestamp and CSI packet counts do not match."
            )

        return csi_data, timestamps, metadata


    # ========================================================
    # LOAD GROUND TRUTH (SUPPORTS BOTH FORMATS)
    # ========================================================

    def load_ground_truth(self, gt_path, gt_type="bpm_samples"):
        """
        Load ground truth data.

        Supports two formats:
            - "beats": CSV with 'ts' column (beat timestamps)
            - "bpm_samples": CSV with 'ts' and 'bpm' columns (BPM samples)
        """
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
            # Ensure same length
            min_len = min(len(gt_times), len(gt_bpm))
            gt_times = gt_times[:min_len]
            gt_bpm = gt_bpm[:min_len]
        elif gt_type == "beats":
            # Just timestamps, BPM will be calculated from RR intervals
            pass
        else:
            raise ValueError(f"Unknown gt_type: {gt_type}")

        if len(gt_times) < 2:
            raise ValueError("Too few valid ground-truth samples.")

        return gt_times, gt_bpm


    # ========================================================
    # DETECT CONTINUOUS SEGMENTS
    # ========================================================

    def split_continuous_segments(
        self,
        timestamps,
        fs
    ):
        """
        Split a recording when a large packet timing gap occurs.

        Example:

            packet
              |
              | 10 ms
              |
            packet
              |
              | 2.56 seconds  <-- large gap
              |
            packet

        The two sides of the gap are processed as independent
        continuous segments.

        We deliberately do NOT interpolate across these gaps.
        """

        if len(timestamps) < 2:
            return [], None, 0

        dt = np.diff(timestamps)

        # Use the median interval as the normal packet interval.
        normal_dt = np.median(
            dt[dt > 0]
        )

        if not np.isfinite(normal_dt) or normal_dt <= 0:

            normal_dt = 1.0 / fs

        gap_threshold = (
            normal_dt * self.gap_multiplier
        )

        gap_indices = np.where(
            (dt <= 0) |
            (dt > gap_threshold)
        )[0]

        segments = []

        start = 0

        for gap_index in gap_indices:

            end = gap_index + 1

            if end > start:

                segments.append(
                    (start, end)
                )

            start = end

        # Final segment.
        if start < len(timestamps):

            segments.append(
                (start, len(timestamps))
            )

        return segments, normal_dt, gap_threshold


    # ========================================================
    # BUILD SIGNAL CANDIDATES
    # ========================================================

    def get_candidates(
        self,
        csi_data,
        method,
        fs
    ):
        """
        Build candidate heartbeat signals.

        Each capture performs its own automatic candidate
        quality evaluation.

        No fixed subcarrier is used across captures.
        """

        pipeline = HeartbeatPipeline(fs)

        channels = []
        names = []

        # ----------------------------------------------------
        # AMPLITUDE
        # ----------------------------------------------------

        if method == "amplitude":

            data = pipeline.extract_amplitude(
                csi_data
            )

            for sc in range(data.shape[1]):

                channels.append(
                    data[:, sc]
                )

                names.append(
                    f"amp_sc{sc}"
                )

        # ----------------------------------------------------
        # PHASE
        # ----------------------------------------------------

        elif method == "phase":

            data = pipeline.extract_phase(
                csi_data,
                unwrap=True
            )

            for sc in range(data.shape[1]):

                channels.append(
                    data[:, sc]
                )

                names.append(
                    f"phase_sc{sc}"
                )

        # ----------------------------------------------------
        # PHASE + COMMON PHASE REMOVAL
        # ----------------------------------------------------

        elif method == "phase_common":

            phase = pipeline.extract_phase(
                csi_data,
                unwrap=True
            )

            data = pipeline.remove_common_phase(
                phase
            )

            for sc in range(data.shape[1]):

                channels.append(
                    data[:, sc]
                )

                names.append(
                    f"common_sc{sc}"
                )

        # ----------------------------------------------------
        # PHASE DIFFERENCES
        # ----------------------------------------------------

        elif method == "phase_diff":

            phase = pipeline.extract_phase(
                csi_data,
                unwrap=True
            )

            n_sc = phase.shape[1]

            # Test the full subcarrier range.
            #
            # We test nearby subcarrier differences because
            # these are the most computationally practical
            # and preserve local frequency structure.
            #
            # The earlier limited search could miss candidates
            # such as diff_114_115 or diff_251_252.

            lags = [
                1,
                2,
                3,
                4,
                8,
                16,
                32
            ]

            for lag in lags:

                if lag >= n_sc:
                    continue

                for i in range(
                    n_sc - lag
                ):

                    j = i + lag

                    channels.append(
                        phase[:, i]
                        -
                        phase[:, j]
                    )

                    names.append(
                        f"diff_{i}_{j}"
                    )

        # ----------------------------------------------------
        # HYBRID (30% Amplitude, 70% Phase)
        # ----------------------------------------------------

        elif method == "amplitude_phase":
            amp = pipeline.extract_amplitude(csi_data)
            ph = pipeline.extract_phase(csi_data, unwrap=True)

            # Normalize both components so they share the same range (0 to 1)
            # before applying the weighted mix. This prevents one signal from
            # dominating just due to its raw amplitude values.
            amp_norm = amp / (np.max(amp) + 1e-6)
            ph_norm = ph / (np.max(ph) + 1e-6)

            mixed = (0.3 * amp_norm) + (0.7 * ph_norm)

            for sc in range(mixed.shape[1]):
                channels.append(mixed[:, sc])
                names.append(f"hybrid_sc{sc}")

        else:

            raise ValueError(
                f"Unknown method: {method}"
            )

        return channels, names


    # ========================================================
    # PROCESS ONE SEGMENT
    # ========================================================

    def process_segment(
        self,
        csi_segment,
        fs,
        method
    ):
        """
        Process one continuous CSI segment.

        Steps:

            candidates
                |
                v
            filtering
                |
                v
            quality scoring
                |
                v
            top-N fusion
        """

        pipeline = HeartbeatPipeline(fs)

        channels, names = self.get_candidates(
            csi_segment,
            method,
            fs
        )

        if len(channels) == 0:

            raise ValueError(
                "No signal candidates generated."
            )

        processed, quality = (
            pipeline.process_all_channels(
                channels,
                names,
                low=self.low_hz,
                high=self.high_hz,
                order=self.order,
            )
        )

        if (
            quality is None
            or len(quality) == 0
        ):

            raise ValueError(
                "No valid quality results."
            )

        fused, top_names = (
            pipeline.fuse_top_channels(
                processed,
                quality,
                top_n=self.top_n,
                align_polarity=True
            )
        )

        if (
            fused is None
            or len(fused) == 0
        ):

            raise ValueError(
                "Channel fusion failed."
            )

        return (
            fused,
            quality,
            top_names
        )


    # ========================================================
    # GROUND TRUTH BPM (OLD FORMAT: BEAT TIMESTAMPS)
    # ========================================================

    def get_gt_bpm_for_window(
        self,
        gt_times,
        window_start,
        window_end
    ):
        """
        Calculate average GT BPM using beats that fall
        inside the CSI time window.

        The timestamps use the same system clock, so CSI and
        GT are compared on their original absolute time axis.
        """

        mask = (
            (gt_times >= window_start)
            &
            (gt_times <= window_end)
        )

        beats = gt_times[mask]

        if len(beats) < MIN_GT_SAMPLES:

            return None, 0

        intervals = np.diff(beats)

        intervals = intervals[
            intervals > 0
        ]

        if len(intervals) == 0:

            return None, 0

        mean_rr = np.mean(intervals)

        if mean_rr <= 0:

            return None, 0

        return 60.0 / mean_rr, len(beats)


    # ========================================================
    # GROUND TRUTH BPM (NEW FORMAT: BPM SAMPLES)
    # ========================================================

    def get_gt_bpm_from_samples(
        self,
        gt_times,
        gt_bpm,
        window_start,
        window_end
    ):
        """
        Calculate median GT BPM using BPM samples that fall
        inside the CSI time window.
        """
        mask = (
            (gt_times >= window_start)
            &
            (gt_times <= window_end)
        )

        bpms = gt_bpm[mask]

        if len(bpms) < MIN_GT_SAMPLES:
            return None, 0

        return float(np.median(bpms)), len(bpms)


    # ========================================================
    # ANALYZE ONE CONTINUOUS SEGMENT
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
    ):
        """
        Run sliding-window HR analysis on one continuous
        CSI segment.
        """

        results = []

        # Process the entire continuous segment once.
        #
        # Channel selection is capture/segment adaptive.

        heartbeat, quality, top_names = (
            self.process_segment(
                csi_segment,
                fs,
                method
            )
        )

        if len(heartbeat) < 2:

            return results

        # Segment timestamps should match heartbeat samples.

        n = min(
            len(heartbeat),
            len(ts_segment)
        )

        heartbeat = heartbeat[:n]
        ts_segment = ts_segment[:n]

        analyzer = WindowedPSDAnalyzer(fs)

        start_time = ts_segment[0]
        end_time = ts_segment[-1]

        window_start = start_time

        while (
            window_start
            +
            self.window_seconds
            <= end_time
        ):

            window_end = (
                window_start
                +
                self.window_seconds
            )

            mask = (
                (ts_segment >= window_start)
                &
                (ts_segment < window_end)
            )

            signal_window = heartbeat[mask]

            if len(signal_window) < 10:

                window_start += (
                    self.step_seconds
                )

                continue

            try:

                csi_bpm, power, peak_hz = (
                    analyzer.compute_bpm_from_psd(
                        signal_window
                    )
                )

            except Exception:

                window_start += (
                    self.step_seconds
                )

                continue

            # Validate BPM.

            if (
                not np.isfinite(csi_bpm)
                or csi_bpm <= 0
                or csi_bpm > MAX_BPM
            ):

                window_start += (
                    self.step_seconds
                )

                continue

            # Ground truth for same absolute time window.
            if gt_type == "beats":
                gt_bpm_val, n_samples = self.get_gt_bpm_for_window(
                    gt_times,
                    window_start,
                    window_end
                )
            elif gt_type == "bpm_samples" and gt_bpm is not None:
                gt_bpm_val, n_samples = self.get_gt_bpm_from_samples(
                    gt_times,
                    gt_bpm,
                    window_start,
                    window_end
                )
            else:
                window_start += self.step_seconds
                continue

            if gt_bpm_val is None:

                window_start += (
                    self.step_seconds
                )

                continue

            error = csi_bpm - gt_bpm_val

            best_quality = (
                quality.iloc[0]["quality_score"]
                if len(quality) > 0
                else np.nan
            )

            result = {

                "capture":
                    capture_name,

                "method":
                    method,

                "segment":
                    segment_id,

                "window_start":
                    window_start,

                "window_end":
                    window_end,

                "csi_bpm":
                    csi_bpm,

                "gt_bpm":
                    gt_bpm_val,

                "error":
                    error,

                "abs_error":
                    abs(error),

                "peak_hz":
                    peak_hz,

                "power":
                    power,

                "quality_score":
                    best_quality,

                "top_channels":
                    "; ".join(top_names),

                "n_gt_samples":
                    n_samples,
            }

            results.append(
                result
            )

            window_start += (
                self.step_seconds
            )

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
        """
        Validate one capture using one pipeline method.
        Returns (result_dict, list_of_windows) or (error_dict, []) on failure.
        
        Args:
            gt_type: "beats" (RR intervals from beat timestamps) or 
                    "bpm_samples" (direct BPM values from sensor)
        """

        capture_name = Path(
            csi_path
        ).stem

        print(
            f"\n  Loading: {capture_name}"
        )

        start_time = time.time()

        try:

            # ------------------------------------------------
            # Load CSI
            # ------------------------------------------------

            csi_data, timestamps, metadata = (
                self.load_capture(
                    csi_path
                )
            )

            fs_loader = metadata.get(
                "rate",
                self.expected_fs
            )

            # ------------------------------------------------
            # Load GT (supports both formats)
            # ------------------------------------------------

            gt_times, gt_bpm = self.load_ground_truth(
                gt_path,
                gt_type=gt_type
            )

            # ------------------------------------------------
            # Detect packet gaps
            # ------------------------------------------------

            (
                segments,
                normal_dt,
                gap_threshold
            ) = self.split_continuous_segments(
                timestamps,
                fs_loader
            )

            normal_fs = (
                1.0 / normal_dt
            )

            print(
                f"    Packets: {len(csi_data)}"
            )

            print(
                f"    Loader FS: "
                f"{fs_loader:.3f} Hz"
            )

            print(
                f"    Median timing FS: "
                f"{normal_fs:.3f} Hz"
            )

            print(
                f"    Continuous segments: "
                f"{len(segments)}"
            )

            print(
                f"    GT samples: {len(gt_times)}"
            )

            # ------------------------------------------------
            # Analyze each valid segment
            # ------------------------------------------------

            capture_windows = []

            valid_segments = 0

            for segment_id, (
                start_idx,
                end_idx
            ) in enumerate(segments):

                csi_segment = (
                    csi_data[
                        start_idx:end_idx
                    ]
                )

                ts_segment = (
                    timestamps[
                        start_idx:end_idx
                    ]
                )

                if len(ts_segment) < 2:

                    continue

                duration = (
                    ts_segment[-1]
                    -
                    ts_segment[0]
                )

                if (
                    duration
                    <
                    self.min_segment_seconds
                ):

                    continue

                valid_segments += 1

                print(
                    f"    Segment {segment_id}: "
                    f"{duration:.2f}s "
                    f"({len(csi_segment)} packets)"
                )

                segment_results = (
                    self.analyze_segment(
                        csi_segment,
                        ts_segment,
                        gt_times,
                        normal_fs,
                        method,
                        capture_name,
                        segment_id,
                        gt_bpm=gt_bpm,
                        gt_type=gt_type,
                    )
                )

                capture_windows.extend(
                    segment_results
                )

            # ------------------------------------------------
            # Failure: no valid windows
            # ------------------------------------------------

            if len(capture_windows) == 0:

                elapsed = (
                    time.time()
                    -
                    start_time
                )

                print(
                    "    FAIL: No valid windows."
                )

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
                        "processing_seconds": elapsed
                    },
                    []
                )

            # ------------------------------------------------
            # Per-capture statistics
            # ------------------------------------------------

            window_df = pd.DataFrame(
                capture_windows
            )

            errors = (
                window_df["error"]
                .to_numpy()
            )

            abs_errors = (
                window_df["abs_error"]
                .to_numpy()
            )

            correlation = np.nan

            if len(window_df) >= 2:

                correlation = (
                    window_df["csi_bpm"]
                    .corr(
                        window_df["gt_bpm"]
                    )
                )

            elapsed = (
                time.time()
                -
                start_time
            )

            capture_result = {

                "capture":
                    capture_name,

                "method":
                    method,

                "success":
                    True,

                "failure_reason":
                    "",

                "n_packets":
                    len(csi_data),

                "n_segments":
                    len(segments),

                "n_valid_segments":
                    valid_segments,

                "n_windows":
                    len(window_df),

                "mean_mae":
                    np.mean(abs_errors),

                "median_mae":
                    np.median(abs_errors),

                "std_mae":
                    np.std(abs_errors),

                "rmse":
                    np.sqrt(
                        np.mean(
                            errors ** 2
                        )
                    ),

                "mean_error":
                    np.mean(errors),

                "worst_mae":
                    np.max(abs_errors),

                "correlation":
                    correlation,

                "normal_fs":
                    normal_fs,

                "gap_threshold_ms":
                    gap_threshold * 1000.0,

                "processing_seconds":
                    elapsed,

                "gt_type":
                    gt_type,
            }

            print(
                f"    OK: "
                f"MAE={capture_result['mean_mae']:.2f} BPM, "
                f"Median={capture_result['median_mae']:.2f}, "
                f"Worst={capture_result['worst_mae']:.2f}, "
                f"Windows={capture_result['n_windows']}"
            )

            return (
                capture_result,
                capture_windows
            )

        except Exception as e:

            elapsed = (
                time.time()
                -
                start_time
            )

            print(
                f"    ERROR: {e}"
            )

            return (
                {
                    "capture": capture_name,
                    "method": method,
                    "success": False,
                    "failure_reason": str(e),
                    "processing_seconds": elapsed,
                    "gt_type": gt_type,
                },
                []
            )


    # ========================================================
    # RUN ALL METHODS / CAPTURES
    # ========================================================

    def run_full_test(
        self,
        captures,
        methods,
        gt_type="bpm_samples",  # Default to new format
    ):
        """
        Test every method against every capture.
        
        Args:
            gt_type: "beats" or "bpm_samples" (can be overridden per capture)
        """

        print()
        print("=" * 70)
        print("MULTI-CAPTURE PIPELINE VALIDATION")
        print("=" * 70)

        for method in methods:

            print()
            print(
                f"METHOD: {method.upper()}"
            )

            print(
                "-" * 70
            )

            method_results = []

            for csi_path, gt_path in captures:

                result, windows = (
                    self.validate_capture(
                        csi_path,
                        gt_path,
                        method,
                        gt_type=gt_type
                    )
                )

                self.capture_results.append(
                    result
                )

                method_results.append(
                    result
                )

                self.window_results.extend(
                    windows
                )

            # Method summary.

            method_df = pd.DataFrame(
                method_results
            )

            success_df = method_df[
                method_df["success"] == True
            ]

            print()
            print(
                f"{method.upper()} SUMMARY"
            )

            print(
                f"  Success rate: "
                f"{len(success_df)}/{len(captures)}"
            )

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
            pd.DataFrame(
                self.capture_results
            ),
            pd.DataFrame(
                self.window_results
            )
        )


# ============================================================
# AUTO-DETECT GT TYPE
# ============================================================

def detect_gt_type(gt_path):
    """
    Auto-detect whether ground truth CSV uses beat timestamps or BPM samples.
    
    Returns:
        "beats" if only 'ts' column exists
        "bpm_samples" if both 'ts' and 'bpm' columns exist
    """
    df = pd.read_csv(gt_path, nrows=1)
    if "bpm" in df.columns:
        return "bpm_samples"
    return "beats"


# ============================================================
# CREATE METHOD SUMMARY
# ============================================================

def create_method_summary(
    results_df
):
    """
    Create robust method-level statistics.

    Only successful captures contribute to accuracy statistics,
    but failures are still included in the success rate.
    """

    rows = []

    methods = sorted(
        results_df["method"].unique()
    )

    for method in methods:

        method_df = results_df[
            results_df["method"] == method
        ]

        successful = method_df[
            method_df["success"] == True
        ]

        total = len(method_df)
        success_count = len(successful)

        row = {

            "method":
                method,

            "captures_total":
                total,

            "captures_success":
                success_count,

            "success_rate":
                (
                    success_count / total
                    if total > 0
                    else np.nan
                )
        }

        if success_count > 0:

            row.update({

                "mean_mae":
                    successful["mean_mae"].mean(),

                "median_mae":
                    successful["median_mae"].median(),

                "mae_std":
                    successful["mean_mae"].std(),

                "mean_rmse":
                    successful["rmse"].mean(),

                "worst_capture_mae":
                    successful["worst_mae"].max(),

                "mean_correlation":
                    successful["correlation"].mean(),

                "mean_windows":
                    successful["n_windows"].mean(),

                "mean_processing_seconds":
                    successful[
                        "processing_seconds"
                    ].mean()
            })

        rows.append(row)

    summary = pd.DataFrame(rows)

    # Sort primarily by median MAE.
    #
    # Median is more robust to one very bad capture.

    summary = summary.sort_values(
        by=[
            "median_mae",
            "mean_mae"
        ],
        ascending=True
    )

    return summary


# ============================================================
# CREATE COMPARISON PLOTS
# ============================================================

def create_comparison_plot(
    results_df,
    summary,
    window_df
):
    """
    Create comparison plots.
    """

    successful = results_df[
        results_df["success"] == True
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    methods = summary["method"].tolist()

    # --------------------------------------------------------
    # Mean MAE
    # --------------------------------------------------------

    ax = axes[0, 0]

    ax.bar(
        methods,
        summary["mean_mae"]
    )

    ax.set_xlabel(
        "Method"
    )

    ax.set_ylabel(
        "Mean_capture MAE (BPM)"
    )

    ax.set_title(
        "Mean MAE Across Captures"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Worst capture MAE
    # --------------------------------------------------------

    ax = axes[0, 1]

    ax.bar(
        methods,
        summary["worst_capture_mae"]
    )

    ax.set_xlabel(
        "Method"
    )

    ax.set_ylabel(
        "Worst Capture MAE (BPM)"
    )

    ax.set_title(
        "Worst-Case Performance"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Success rate
    # --------------------------------------------------------

    ax = axes[1, 0]

    ax.bar(
        methods,
        summary["success_rate"] * 100.0
    )

    ax.set_xlabel(
        "Method"
    )

    ax.set_ylabel(
        "Success Rate (%)"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_title(
        "Capture Success Rate"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Window-level CSI vs GT
    # --------------------------------------------------------

    ax = axes[1, 1]

    if len(window_df) > 0:

        for method in methods:

            df = window_df[
                window_df["method"] == method
            ]

            if len(df) == 0:
                continue

            ax.scatter(
                df["gt_bpm"],
                df["csi_bpm"],
                alpha=0.5,
                s=20,
                label=method
            )

        min_bpm = min(
            window_df["gt_bpm"].min(),
            window_df["csi_bpm"].min()
        )

        max_bpm = max(
            window_df["gt_bpm"].max(),
            window_df["csi_bpm"].max()
        )

        ax.plot(
            [min_bpm, max_bpm],
            [min_bpm, max_bpm],
            "k--",
            alpha=0.5
        )

    ax.set_xlabel(
        "Ground Truth BPM"
    )

    ax.set_ylabel(
        "CSI BPM"
    )

    ax.set_title(
        "Window-Level CSI vs Ground Truth"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    plt.suptitle(
        "Multi-Capture Pipeline Comparison",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        "multi_capture_comparison.png",
        dpi=300
    )

    plt.close()

    print(
        "\nSaved: multi_capture_comparison.png"
    )


# ============================================================
# RUN TEST
# ============================================================

def run_test(gt_type="bpm_samples"):
    """
    Run the multi-capture validation test.
    
    Args:
        gt_type: "beats" (old format) or "bpm_samples" (new format)
                 Set to "beats" for backward compatibility with old CSV files.
    """

    # --------------------------------------------------------
    # CAPTURES
    # --------------------------------------------------------

    captures = [
        #(
        #    r"captures\Allen-N,0,1"
        #    r"\csi_capture_20260820_133513.csv",
        #    r"captures\Allen-N,0,1"
        #    r"\ground_truth_20260820_133509.csv"
        #),
        #(
        #    r"captures\Allen-N,0,-1"
        #    r"\csi_capture_20260820_134305.csv",
        #    r"captures\Allen-N,0,-1"
        #    r"\ground_truth_20260820_134301.csv"
        #),
        #(
        #    r"captures\Allen-S,0,0"
        #    r"\csi_capture_20260820_134939.csv",
        #    r"captures\Allen-S,0,0"
        #    r"\ground_truth_20260820_134936.csv"
        #),
        #(
        #    r"captures\Allen-S,0,1"
        #    r"\csi_capture_20260820_135250.csv",
        #    r"captures\Allen-S,0,1"
        #    r"\ground_truth_20260820_135247.csv"
        #),
        #(
        #    r"captures\Allen-S,0,-1"
        #    r"\csi_capture_20260820_134631.csv",
        #    r"captures\Allen-S,0,-1"
        #    r"\ground_truth_20260820_134627.csv"
        #),
        #(
        #    r"captures\Anusha1\csi_capture_20260818_190333.csv",
        #    r"captures\Anusha1\ground_truth_20260818_190327.csv"
        #),
        # (
        #     r"captures\04-09-2026\Allen-S,0,1\csi_capture_20260904_232607.csv",
        #     r"captures\04-09-2026\Allen-S,0,1\ground_truth_20260904_232615.csv"
        # ),
        # (
        #     r"captures\05-09-2026\Allen-S,0,1\csi_capture_20260905_004623.csv",
        #     r"captures\05-09-2026\Allen-S,0,1\ground_truth_20260905_004619.csv"
        # ),
        # (
        #     r"captures\06-09-2026\Allen-S,0,1-T1\csi_capture_20260906_131236.csv",
        #     r"captures\06-09-2026\Allen-S,0,1-T1\ground_truth_20260906_131258.csv"
        # ),
        (
            r"captures\06-09-2026\Allen-N,0,-1-T1\csi_capture_20260906_234541.csv",
            r"captures\06-09-2026\Allen-N,0,-1-T1\ground_truth_20260906_234544.csv"
        ),
    ]

    # --------------------------------------------------------
    # METHODS
    # --------------------------------------------------------

    methods = [
        "amplitude",
        "phase",
        "phase_common",
        "phase_diff",
        "amplitude_phase",
    ]

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    validator = MultiCaptureValidator()

    results_df, window_df = (
        validator.run_full_test(
            captures,
            methods,
            gt_type=gt_type
        )
    )

    # --------------------------------------------------------
    # SAVE RAW RESULTS
    # --------------------------------------------------------

    results_df.to_csv(
        "multi_capture_results.csv",
        index=False
    )

    window_df.to_csv(
        "multi_capture_window_results.csv",
        index=False
    )

    print(
        "\nSaved: multi_capture_results.csv"
    )

    print(
        "Saved: multi_capture_window_results.csv"
    )

    # --------------------------------------------------------
    # CREATE SUMMARY
    # --------------------------------------------------------

    summary = create_method_summary(
        results_df
    )

    summary.to_csv(
        "multi_capture_summary.csv",
        index=False
    )

    print(
        "Saved: multi_capture_summary.csv"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SUMMARY BY METHOD")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BEST METHOD
    # --------------------------------------------------------

    valid_summary = summary.dropna(
        subset=["median_mae"]
    )

    if len(valid_summary) > 0:

        best = (
            valid_summary.iloc[0]
        )

        print()
        print("=" * 70)
        print(
            f"BEST METHOD: "
            f"{best['method'].upper()}"
        )
        print("=" * 70)

        print(
            f"Median MAE: "
            f"{best['median_mae']:.2f} BPM"
        )

        print(
            f"Mean MAE: "
            f"{best['mean_mae']:.2f} BPM"
        )

        print(
            f"Worst Capture MAE: "
            f"{best['worst_capture_mae']:.2f} BPM"
        )

        print(
            f"Success Rate: "
            f"{best['success_rate'] * 100:.1f}%"
        )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    create_comparison_plot(
        results_df,
        summary,
        window_df
    )

    # --------------------------------------------------------
    # INDIVIDUAL RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INDIVIDUAL CAPTURE RESULTS")
    print("=" * 70)

    display_columns = [
        "capture",
        "method",
        "success",
        "n_segments",
        "n_valid_segments",
        "n_windows",
        "mean_mae",
        "median_mae",
        "worst_mae",
        "correlation",
        "gt_type",
        "failure_reason"
    ]

    existing_columns = [
        col
        for col in display_columns
        if col in results_df.columns
    ]

    print(
        results_df[
            existing_columns
        ].to_string(
            index=False
        )
    )

    return (
        results_df,
        window_df,
        summary
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # By default, use the new BPM samples format.
    # To use old format (beat timestamps), pass gt_type="beats"
    results, windows, summary = run_test(gt_type="bpm_samples")