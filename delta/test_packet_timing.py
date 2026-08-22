# test_packet_timing.py
#
# CSI Packet Timing Uniformity Test
#
# Tests whether CSI packets are sampled uniformly enough for
# fixed-rate filtering and PSD analysis.
#
# Output:
#   packet_timing_over_time.png
#   packet_timing_histogram.png
#   packet_timing_jitter.png
#   packet_timing_statistics.csv


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from read_csv_csi import CSIDataLoader


# ============================================================
# CONFIGURATION
# ============================================================

#CSI_PATH = r"captures\Allen-S,0,1\csi_capture_20260820_135250.csv"
CSI_PATH = r"captures\Allen-N,0,1\csi_capture_20260820_133513.csv"

PLOT = True

# A packet interval larger than this multiple of the normal
# interval is considered a significant gap.
GAP_MULTIPLIER = 1.5

# Number of largest gaps to print
TOP_GAPS = 10


# ============================================================
# FIND TIMESTAMP COLUMN
# ============================================================

def find_timestamp_column(df):
    """
    Automatically find the per-packet timestamp column.
    """

    candidates = [
        "timestamp",
        "timestamps",
        "ts",
        "time",
        "Timestamp",
        "TIMESTAMP",
        "TS",
        "Time",
    ]

    # First try exact matches
    for column in candidates:
        if column in df.columns:
            return column

    # Then try case-insensitive matching
    lower_columns = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        candidate_lower = candidate.lower()

        if candidate_lower in lower_columns:
            return lower_columns[candidate_lower]

    raise ValueError(
        "\nCould not automatically find a timestamp column.\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_packet_timing(csi_path, plot=True):

    print("=" * 70)
    print("CSI PACKET TIMING UNIFORMITY TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load CSI using normal loader
    # --------------------------------------------------------

    print(f"\nLoading CSI: {csi_path}")

    loader = CSIDataLoader(csi_path)
    csi_data, metadata = loader.load()

    print(f"\nCSI packets loaded     : {csi_data.shape[0]}")
    print(f"CSI subcarriers        : {csi_data.shape[1]}")

    if "rate" in metadata:
        print(f"Loader sample rate     : {metadata['rate']:.6f} Hz")

    if "first_ts" in metadata:
        print(f"Loader first timestamp : {metadata['first_ts']}")

    if "last_ts" in metadata:
        print(f"Loader last timestamp  : {metadata['last_ts']}")

    # --------------------------------------------------------
    # Load raw CSV to get PER-PACKET timestamps
    # --------------------------------------------------------

    print("\nLoading raw CSV for per-packet timestamps...")

    df = pd.read_csv(csi_path)

    print(f"CSV rows: {len(df)}")
    print(f"CSV columns: {len(df.columns)}")

    timestamp_column = find_timestamp_column(df)

    print(f"\nUsing timestamp column: {timestamp_column}")

    timestamps_raw = pd.to_numeric(
        df[timestamp_column],
        errors="coerce"
    ).to_numpy(dtype=float)

    # --------------------------------------------------------
    # Remove invalid timestamps but preserve packet order
    # --------------------------------------------------------

    valid = np.isfinite(timestamps_raw)

    invalid_count = np.sum(~valid)

    if invalid_count > 0:
        print(
            f"\nWARNING: Removing {invalid_count} "
            "invalid timestamp values."
        )

    timestamps = timestamps_raw[valid]

    if len(timestamps) < 2:
        raise ValueError(
            "Not enough valid timestamps for timing analysis."
        )

    print(f"Valid timestamps: {len(timestamps)}")

    # --------------------------------------------------------
    # Timestamp ordering check
    # --------------------------------------------------------

    raw_dt = np.diff(timestamps)

    negative_indices = np.where(raw_dt < 0)[0]
    zero_indices = np.where(raw_dt == 0)[0]

    negative_count = len(negative_indices)
    zero_count = len(zero_indices)

    print("\n" + "=" * 70)
    print("TIMESTAMP ORDER CHECK")
    print("=" * 70)

    print(f"Negative intervals : {negative_count}")
    print(f"Zero intervals     : {zero_count}")

    if negative_count > 0:

        print("\nWARNING: Timestamps go backwards.")

        print("First few negative intervals:")

        for i in negative_indices[:10]:

            print(
                f"  Packet {i} -> {i + 1}: "
                f"{raw_dt[i] * 1000:.6f} ms"
            )

    if zero_count > 0:

        print(
            "\nWARNING: Duplicate/zero timestamps detected."
        )

    # --------------------------------------------------------
    # Keep only positive intervals for timing statistics
    # --------------------------------------------------------

    positive_mask = raw_dt > 0

    dt = raw_dt[positive_mask]

    if len(dt) == 0:
        raise ValueError(
            "No positive packet intervals available."
        )

    dt_ms = dt * 1000.0

    # --------------------------------------------------------
    # Basic timing statistics
    # --------------------------------------------------------

    mean_dt = np.mean(dt)
    median_dt = np.median(dt)
    std_dt = np.std(dt)

    min_dt = np.min(dt)
    max_dt = np.max(dt)

    p1_dt = np.percentile(dt, 1)
    p5_dt = np.percentile(dt, 5)
    p95_dt = np.percentile(dt, 95)
    p99_dt = np.percentile(dt, 99)

    effective_fs = 1.0 / mean_dt
    median_fs = 1.0 / median_dt

    cv_percent = (
        std_dt / mean_dt
    ) * 100.0

    print("\n" + "=" * 70)
    print("PACKET INTERVAL STATISTICS")
    print("=" * 70)

    print(f"Mean interval       : {mean_dt * 1000:.6f} ms")
    print(f"Median interval     : {median_dt * 1000:.6f} ms")
    print(f"Std deviation       : {std_dt * 1000:.6f} ms")

    print()

    print(f"Minimum interval    : {min_dt * 1000:.6f} ms")
    print(f"1st percentile      : {p1_dt * 1000:.6f} ms")
    print(f"5th percentile      : {p5_dt * 1000:.6f} ms")

    print()

    print(f"95th percentile     : {p95_dt * 1000:.6f} ms")
    print(f"99th percentile     : {p99_dt * 1000:.6f} ms")
    print(f"Maximum interval    : {max_dt * 1000:.6f} ms")

    print()

    print(f"Mean sample rate    : {effective_fs:.6f} Hz")
    print(f"Median sample rate  : {median_fs:.6f} Hz")

    print()

    print(f"Timing variation CV : {cv_percent:.6f}%")

    # --------------------------------------------------------
    # Timing jitter
    # --------------------------------------------------------

    # Compare each interval to the median interval.
    # Median is more robust against occasional large packet gaps.

    expected_dt = median_dt

    jitter = dt - expected_dt
    jitter_ms = jitter * 1000.0

    mean_abs_jitter = np.mean(np.abs(jitter_ms))

    rms_jitter = np.sqrt(
        np.mean(jitter_ms ** 2)
    )

    max_abs_jitter = np.max(
        np.abs(jitter_ms)
    )

    jitter_std = np.std(
        jitter_ms
    )

    print("\n" + "=" * 70)
    print("TIMING JITTER")
    print("=" * 70)

    print(
        f"Expected interval   : "
        f"{expected_dt * 1000:.6f} ms"
    )

    print(
        f"Mean absolute jitter: "
        f"{mean_abs_jitter:.6f} ms"
    )

    print(
        f"RMS jitter          : "
        f"{rms_jitter:.6f} ms"
    )

    print(
        f"Jitter std          : "
        f"{jitter_std:.6f} ms"
    )

    print(
        f"Maximum jitter      : "
        f"{max_abs_jitter:.6f} ms"
    )

    # --------------------------------------------------------
    # Packet gap analysis
    # --------------------------------------------------------

    gap_threshold = (
        median_dt * GAP_MULTIPLIER
    )

    gap_indices_positive = np.where(
        dt > gap_threshold
    )[0]

    gap_count = len(
        gap_indices_positive
    )

    gap_percentage = (
        gap_count / len(dt)
    ) * 100.0

    # Estimate how many packets may be missing.
    #
    # Example:
    #
    # Normal interval = 10 ms
    # Observed interval = 40 ms
    #
    # Expected ~4 intervals
    # Therefore ~3 packets may be missing.

    estimated_missing_packets = 0

    for gap in dt[gap_indices_positive]:

        estimated_intervals = int(
            np.round(gap / median_dt)
        )

        missing = max(
            estimated_intervals - 1,
            0
        )

        estimated_missing_packets += missing

    print("\n" + "=" * 70)
    print("PACKET GAP ANALYSIS")
    print("=" * 70)

    print(
        f"Normal interval     : "
        f"{median_dt * 1000:.6f} ms"
    )

    print(
        f"Gap threshold       : "
        f"{gap_threshold * 1000:.6f} ms"
    )

    print(
        f"Large gaps detected : "
        f"{gap_count}"
    )

    print(
        f"Percentage affected : "
        f"{gap_percentage:.6f}%"
    )

    print(
        f"Estimated missing packets from gaps: "
        f"{estimated_missing_packets}"
    )

    # --------------------------------------------------------
    # Print largest gaps
    # --------------------------------------------------------

    if gap_count > 0:

        print(f"\nLargest {TOP_GAPS} packet gaps:")

        largest_indices = np.argsort(
            dt
        )[-TOP_GAPS:][::-1]

        # Need mapping back to original packet indices
        original_positive_indices = np.where(
            positive_mask
        )[0]

        for rank, dt_index in enumerate(
            largest_indices,
            start=1
        ):

            original_index = (
                original_positive_indices[
                    dt_index
                ]
            )

            interval_ms = (
                dt[dt_index] * 1000.0
            )

            ratio = (
                dt[dt_index] / median_dt
            )

            estimated_intervals = int(
                np.round(ratio)
            )

            estimated_missing = max(
                estimated_intervals - 1,
                0
            )

            print(
                f"\n{rank}. Packet "
                f"{original_index} -> "
                f"{original_index + 1}"
            )

            print(
                f"   Gap: {interval_ms:.6f} ms"
            )

            print(
                f"   Ratio to normal: "
                f"{ratio:.2f}x"
            )

            print(
                f"   Estimated missing packets: "
                f"{estimated_missing}"
            )

            print(
                f"   Timestamp: "
                f"{timestamps[original_index]:.6f}"
            )

    # --------------------------------------------------------
    # Check sample-rate consistency
    # --------------------------------------------------------

    duration_from_timestamps = (
        timestamps[-1] - timestamps[0]
    )

    calculated_rate = (
        (len(timestamps) - 1)
        / duration_from_timestamps
    )

    print("\n" + "=" * 70)
    print("SAMPLE RATE CONSISTENCY")
    print("=" * 70)

    print(
        f"Timestamp duration  : "
        f"{duration_from_timestamps:.6f} s"
    )

    print(
        f"Calculated rate     : "
        f"{calculated_rate:.6f} Hz"
    )

    if "rate" in metadata:

        loader_rate = float(
            metadata["rate"]
        )

        difference = (
            calculated_rate
            - loader_rate
        )

        print(
            f"Loader rate         : "
            f"{loader_rate:.6f} Hz"
        )

        print(
            f"Difference          : "
            f"{difference:.6f} Hz"
        )

    # --------------------------------------------------------
    # Uniformity assessment
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("UNIFORM SAMPLING ASSESSMENT")
    print("=" * 70)

    # Assessment considers both general timing variation
    # and the percentage of large gaps.

    if (
        cv_percent < 1.0
        and gap_percentage < 0.1
    ):

        assessment = "EXCELLENT"

        recommendation = (
            "Packet timing is highly uniform. "
            "Using a fixed sampling rate is appropriate."
        )

    elif (
        cv_percent < 5.0
        and gap_percentage < 1.0
    ):

        assessment = "GOOD"

        recommendation = (
            "Packet timing has minor variation. "
            "Your current fixed-rate filtering is likely acceptable. "
            "Resampling can still be tested as an experiment."
        )

    elif (
        cv_percent < 10.0
        and gap_percentage < 5.0
    ):

        assessment = "MODERATE"

        recommendation = (
            "Packet timing has noticeable variation. "
            "Testing interpolation/resampling to a uniform time grid "
            "before filtering is recommended."
        )

    else:

        assessment = "POOR"

        recommendation = (
            "Packet timing is significantly irregular or contains "
            "frequent packet gaps. Timestamp-based resampling should "
            "be strongly considered before filtering and PSD analysis."
        )

    print(
        f"Assessment          : {assessment}"
    )

    print("\nRecommendation:")

    print(
        f"  {recommendation}"
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    statistics = {
        "timestamp_column": timestamp_column,

        "n_packets": len(timestamps),

        "invalid_timestamps": int(
            invalid_count
        ),

        "negative_intervals": int(
            negative_count
        ),

        "zero_intervals": int(
            zero_count
        ),

        "duration_seconds":
            duration_from_timestamps,

        "mean_interval_ms":
            mean_dt * 1000.0,

        "median_interval_ms":
            median_dt * 1000.0,

        "std_interval_ms":
            std_dt * 1000.0,

        "min_interval_ms":
            min_dt * 1000.0,

        "max_interval_ms":
            max_dt * 1000.0,

        "p1_interval_ms":
            p1_dt * 1000.0,

        "p5_interval_ms":
            p5_dt * 1000.0,

        "p95_interval_ms":
            p95_dt * 1000.0,

        "p99_interval_ms":
            p99_dt * 1000.0,

        "mean_sample_rate_hz":
            effective_fs,

        "median_sample_rate_hz":
            median_fs,

        "calculated_sample_rate_hz":
            calculated_rate,

        "cv_percent":
            cv_percent,

        "mean_absolute_jitter_ms":
            mean_abs_jitter,

        "rms_jitter_ms":
            rms_jitter,

        "jitter_std_ms":
            jitter_std,

        "max_jitter_ms":
            max_abs_jitter,

        "gap_threshold_ms":
            gap_threshold * 1000.0,

        "large_gap_count":
            gap_count,

        "large_gap_percentage":
            gap_percentage,

        "estimated_missing_packets":
            estimated_missing_packets,

        "assessment":
            assessment,
    }

    stats_df = pd.DataFrame(
        [statistics]
    )

    stats_filename = (
        "packet_timing_statistics.csv"
    )

    stats_df.to_csv(
        stats_filename,
        index=False
    )

    print(
        f"\nSaved: {stats_filename}"
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    if plot:

        # Relative time for each interval.
        # Use original timestamps associated with the interval.

        interval_times = (
            timestamps[:-1][positive_mask]
            - timestamps[0]
        )

        # ====================================================
        # PLOT 1
        # Packet interval over time
        # ====================================================

        plt.figure(figsize=(12, 5))

        plt.plot(
            interval_times,
            dt_ms,
            linewidth=0.5
        )

        plt.axhline(
            mean_dt * 1000.0,
            linestyle="--",
            label=(
                f"Mean = "
                f"{mean_dt * 1000:.3f} ms"
            )
        )

        plt.axhline(
            gap_threshold * 1000.0,
            linestyle="--",
            label=(
                f"Gap threshold = "
                f"{gap_threshold * 1000:.3f} ms"
            )
        )

        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Packet Interval (ms)"
        )

        plt.title(
            "CSI Packet Interval Over Time"
        )

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        filename = (
            "packet_timing_over_time.png"
        )

        plt.savefig(
            filename,
            dpi=150
        )

        print(
            f"Saved: {filename}"
        )

        # ====================================================
        # PLOT 2
        # Histogram
        # ====================================================

        plt.figure(figsize=(10, 5))

        plt.hist(
            dt_ms,
            bins=100
        )

        plt.axvline(
            mean_dt * 1000.0,
            linestyle="--",
            label=(
                f"Mean = "
                f"{mean_dt * 1000:.3f} ms"
            )
        )

        plt.axvline(
            median_dt * 1000.0,
            linestyle="--",
            label=(
                f"Median = "
                f"{median_dt * 1000:.3f} ms"
            )
        )

        plt.xlabel(
            "Packet Interval (ms)"
        )

        plt.ylabel(
            "Count"
        )

        plt.title(
            "CSI Packet Interval Distribution"
        )

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        filename = (
            "packet_timing_histogram.png"
        )

        plt.savefig(
            filename,
            dpi=150
        )

        print(
            f"Saved: {filename}"
        )

        # ====================================================
        # PLOT 3
        # Timing jitter
        # ====================================================

        plt.figure(figsize=(12, 5))

        plt.plot(
            interval_times,
            jitter_ms,
            linewidth=0.5
        )

        plt.axhline(
            0,
            linestyle="--"
        )

        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Timing Jitter (ms)"
        )

        plt.title(
            "CSI Packet Timing Jitter "
            "Relative to Median Interval"
        )

        plt.grid(True)

        plt.tight_layout()

        filename = (
            "packet_timing_jitter.png"
        )

        plt.savefig(
            filename,
            dpi=150
        )

        print(
            f"Saved: {filename}"
        )

        plt.show()

    print("\n" + "=" * 70)
    print("TIMING TEST COMPLETE")
    print("=" * 70)

    return statistics


# ============================================================
# RUN TEST
# ============================================================

if __name__ == "__main__":

    analyze_packet_timing(
        CSI_PATH,
        plot=PLOT
    )