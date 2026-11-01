
#!/usr/bin/env python3
"""
diagnose_peaks.py
=================

Time-domain heartbeat diagnostic for CSI-Heart.

Analyzes one window of a CSI capture using:

    CSI amplitude
        -> HeartbeatPipeline processing
        -> Top-N subcarrier fusion
        -> Hilbert envelope
        -> Adaptive peak detection
        -> Inter-beat interval (IBI) analysis
        -> Median BPM estimate

Plots:
    1. Fused CSI waveform with detected peaks
    2. Hilbert envelope with all and accepted peaks
    3. Inter-beat intervals and instantaneous BPM
    4. Subcarrier quality ranking

Input formats:
    .csv
    .pcap
    .pcapng

Examples:

    python diagnose_peaks.py ^
        --csi captures\\csi_capture.csv ^
        --gt captures\\ground_truth.csv ^
        --window-start 20 ^
        --window-seconds 20

    python diagnose_peaks.py ^
        --csi captures\\csi_capture.csv ^
        --gt captures\\ground_truth.csv ^
        --window-start 20 ^
        --window-seconds 20 ^
        --min-distance 0.33 ^
        --prominence-factor 0.3

    python diagnose_peaks.py ^
        --csi captures\\csi_capture.csv ^
        --no-gt ^
        --window-start 20 ^
        --window-seconds 20

    python diagnose_peaks.py ^
        --csi captures\\csi_capture.csv ^
        --true-hr 90 ^
        --window-start 20 ^
        --window-seconds 20

Linux/macOS:

    python diagnose_peaks.py \
        --csi captures/csi_capture.csv \
        --gt captures/ground_truth.csv \
        --window-start 20 \
        --window-seconds 20

Notes:
    - The heartbeat bandpass is performed by HeartbeatPipeline.
    - The Hilbert transform is applied to the fused, bandpassed signal.
    - Peak distance is specified in seconds and converted to samples.
    - The default minimum distance corresponds to 180 BPM maximum.
    - The BPM estimate uses the median of accepted instantaneous BPM.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import find_peaks, hilbert

# ----------------------------------------------------------------------
# Project imports
# ----------------------------------------------------------------------

from windowed_psd_analysis import HeartbeatPipeline

from read_pcap import check_clipping


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

TARGET_FS = 100.0

BANDPASS_LOW = 0.8
BANDPASS_HIGH = 2.5
FILTER_ORDER = 4

TOP_N = 5

# Default peak detection settings
DEFAULT_MIN_DISTANCE_SECONDS = 60.0 / 180.0
DEFAULT_PROMINENCE_FACTOR = 0.30

# Physiological limits used for rejecting implausible intervals.
# These are analysis defaults, not medical diagnostic limits.
DEFAULT_MIN_HR = 40.0
DEFAULT_MAX_HR = 180.0


# ----------------------------------------------------------------------
# Load CSI from CSV
# ----------------------------------------------------------------------

def _load_csv_csi(csi_path: Path):
    """
    Load CSI data from a CSV capture written by live_monitor.py.

    Returns:
        csi_data : complex ndarray, shape (samples, subcarriers)
        ts       : timestamp ndarray
        meta     : loader metadata
    """

    from read_csv_csi import CSIDataLoader

    loader = CSIDataLoader(str(csi_path))
    csi_data, meta = loader.load()

    if hasattr(loader, "df") and "ts" in loader.df.columns:

        ts = pd.to_numeric(
            loader.df["ts"],
            errors="coerce",
        ).to_numpy(dtype=float)

    else:
        raise ValueError(
            "CSI CSV has no 'ts' column."
        )

    valid = np.isfinite(ts)

    ts = ts[valid]
    csi_data = csi_data[valid]

    if len(ts) != len(csi_data):

        raise ValueError(
            "Timestamp / CSI length mismatch in CSV."
        )

    return csi_data, ts, meta


# ----------------------------------------------------------------------
# Load CSI from PCAP
# ----------------------------------------------------------------------

def _load_pcap_csi(csi_path: Path):
    """
    Load CSI data from a PCAP/PCAPNG capture.

    Uses PCAPDataLoader from the existing project.
    """

    from read_pcap import PCAPDataLoader

    loader = PCAPDataLoader(str(csi_path))

    csi_data, meta = loader.load()

    ts = None

    if (
        hasattr(loader, "csi_timestamps")
        and loader.csi_timestamps is not None
    ):

        ts = np.asarray(
            loader.csi_timestamps,
            dtype=float,
        )

    elif "timestamps" in meta:

        ts = np.asarray(
            meta["timestamps"],
            dtype=float,
        )

    # Fallback if packet timestamps are unavailable.
    if ts is None or len(ts) != len(csi_data):

        rate = float(meta.get("rate", 0.0)) or 93.5

        print(
            "Warning: no per-packet PCAP timestamps — "
            f"using synthetic {rate:.2f} Hz grid"
        )

        ts = np.arange(
            len(csi_data),
            dtype=float,
        ) / rate

    return csi_data, ts, meta


# ----------------------------------------------------------------------
# Load CSI — automatic format selection
# ----------------------------------------------------------------------

def load_csi(csi_path: Path):
    """
    Load CSI from CSV, PCAP, or PCAPNG.
    """

    suffix = csi_path.suffix.lower()

    if suffix == ".csv":

        return _load_csv_csi(csi_path)

    if suffix in (".pcap", ".pcapng"):

        return _load_pcap_csi(csi_path)

    raise ValueError(
        f"Unsupported capture format: {suffix}. "
        "Expected .csv, .pcap, or .pcapng"
    )


# ----------------------------------------------------------------------
# Uniform resampling
# ----------------------------------------------------------------------

def resample_to_uniform(
    csi,
    ts,
    target_fs=TARGET_FS,
):
    """
    Resample CSI onto a uniform time grid.

    The existing CSI captures may have slightly irregular
    packet timestamps. Peak detection should operate on a
    uniform sampling grid.
    """

    ts = np.asarray(ts, dtype=float)

    csi = np.asarray(csi)

    order = np.argsort(ts)

    ts = ts[order]
    csi = csi[order]

    if len(ts) < 2:

        raise ValueError(
            "Need at least two CSI timestamps for resampling."
        )

    step = 1.0 / target_fs

    uniform = np.arange(
        ts[0],
        ts[-1] + 0.5 * step,
        step,
    )

    idx = np.searchsorted(
        ts,
        uniform,
        side="right",
    )

    idx = np.clip(
        idx,
        1,
        len(ts) - 1,
    )

    i0 = idx - 1
    i1 = idx

    t0 = ts[i0]
    t1 = ts[i1]

    dt = t1 - t0

    dt[dt == 0] = 1.0

    w1 = (uniform - t0) / dt
    w0 = 1.0 - w1

    out = (
        w0[:, None] * csi[i0]
        + w1[:, None] * csi[i1]
    )

    return out, uniform


# ----------------------------------------------------------------------
# Peak detection helpers
# ----------------------------------------------------------------------

def detect_envelope_peaks(
    envelope,
    fs=TARGET_FS,
    min_distance_seconds=DEFAULT_MIN_DISTANCE_SECONDS,
    prominence_factor=DEFAULT_PROMINENCE_FACTOR,
    min_hr=DEFAULT_MIN_HR,
    max_hr=DEFAULT_MAX_HR,
):
    """
    Detect peaks on the Hilbert envelope.

    Returns:
        peaks
        properties
        prominence_threshold
        accepted_interval_mask
        intervals_seconds
        instantaneous_bpm
        accepted_bpm
        median_bpm
    """

    envelope = np.asarray(
        envelope,
        dtype=float,
    )

    if len(envelope) < 3:

        return {
            "peaks": np.array([], dtype=int),
            "properties": {},
            "prominence_threshold": 0.0,
            "accepted_interval_mask": np.array([], dtype=bool),
            "intervals_seconds": np.array([], dtype=float),
            "instantaneous_bpm": np.array([], dtype=float),
            "accepted_bpm": np.array([], dtype=float),
            "median_bpm": 0.0,
        }

    # Remove non-finite values.
    envelope = np.nan_to_num(
        envelope,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Center the envelope for a more stable prominence threshold.
    envelope_centered = envelope - np.median(envelope)

    envelope_std = float(
        np.std(envelope_centered)
    )

    prominence_threshold = (
        prominence_factor * envelope_std
    )

    min_distance_samples = max(
        1,
        int(round(
            min_distance_seconds * fs
        )),
    )

    # --------------------------------------------------------------
    # Detect peaks
    # --------------------------------------------------------------

    peaks, properties = find_peaks(
        envelope_centered,
        distance=min_distance_samples,
        prominence=prominence_threshold,
    )

    # --------------------------------------------------------------
    # Calculate IBI
    # --------------------------------------------------------------

    if len(peaks) > 1:

        intervals_seconds = (
            np.diff(peaks) / fs
        )

        instantaneous_bpm = (
            60.0 / intervals_seconds
        )

    else:

        intervals_seconds = np.array(
            [],
            dtype=float,
        )

        instantaneous_bpm = np.array(
            [],
            dtype=float,
        )

    # --------------------------------------------------------------
    # Reject implausible intervals
    # --------------------------------------------------------------

    min_interval = 60.0 / max_hr
    max_interval = 60.0 / min_hr

    accepted_interval_mask = (
        (intervals_seconds >= min_interval)
        & (intervals_seconds <= max_interval)
    )

    accepted_bpm = instantaneous_bpm[
        accepted_interval_mask
    ]

    # --------------------------------------------------------------
    # Robust median BPM
    # --------------------------------------------------------------

    if len(accepted_bpm) > 0:

        median_bpm = float(
            np.median(accepted_bpm)
        )

    else:

        median_bpm = 0.0

    return {
        "peaks": peaks,
        "properties": properties,
        "prominence_threshold": prominence_threshold,
        "accepted_interval_mask": accepted_interval_mask,
        "intervals_seconds": intervals_seconds,
        "instantaneous_bpm": instantaneous_bpm,
        "accepted_bpm": accepted_bpm,
        "median_bpm": median_bpm,
    }


# ----------------------------------------------------------------------
# Analyze one window
# ----------------------------------------------------------------------

def analyze_window_peaks(
    csi,
    ts,
    gt_df,
    ws,
    we,
    min_distance_seconds=DEFAULT_MIN_DISTANCE_SECONDS,
    prominence_factor=DEFAULT_PROMINENCE_FACTOR,
    min_hr=DEFAULT_MIN_HR,
    max_hr=DEFAULT_MAX_HR,
):
    """
    Analyze one CSI window using Hilbert-envelope peaks.

    Pipeline:

        CSI
        -> amplitude
        -> HeartbeatPipeline processing
        -> top-N fusion
        -> Hilbert transform
        -> envelope
        -> find_peaks
        -> IBI
        -> median BPM
    """

    # --------------------------------------------------------------
    # Extract window
    # --------------------------------------------------------------

    mask = (
        (ts >= ws)
        & (ts < we)
    )

    seg = csi[mask]
    seg_ts = ts[mask]

    if len(seg) < 20:

        raise RuntimeError(
            f"Only {len(seg)} samples in window — too few."
        )

    # --------------------------------------------------------------
    # Existing CSI heartbeat pipeline
    # --------------------------------------------------------------

    pipeline = HeartbeatPipeline(TARGET_FS)

    # Extract amplitude.
    amp = pipeline.extract_amplitude(seg)

    channels = [
        amp[:, sc]
        for sc in range(amp.shape[1])
    ]

    names = [
        f"amp_sc{sc}"
        for sc in range(amp.shape[1])
    ]

    # Existing processing and quality ranking.
    processed, quality = (
        pipeline.process_all_channels(
            channels,
            names,
            low=BANDPASS_LOW,
            high=BANDPASS_HIGH,
            order=FILTER_ORDER,
        )
    )

    # Fuse top-N processed channels.
    fused, top_names = (
        pipeline.fuse_top_channels(
            processed,
            quality,
            top_n=TOP_N,
            align_polarity=True,
        )
    )

    # --------------------------------------------------------------
    # 1. Fused heartbeat waveform
    # --------------------------------------------------------------

    fused = np.asarray(
        fused,
        dtype=float,
    )

    fused = np.nan_to_num(
        fused,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # --------------------------------------------------------------
    # 2. Direct fused waveform
    # --------------------------------------------------------------

    # The fused signal has already been bandpassed by
    # HeartbeatPipeline.process_all_channels().
    #
    # Do not apply Hilbert envelope yet.
    # Test whether the actual oscillating waveform contains
    # heartbeat peaks.

    waveform = np.asarray(
        fused,
        dtype=float,
    )

    waveform = np.nan_to_num(
        waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Remove DC offset.
    waveform = waveform - np.median(waveform)

    # Use the direct waveform for peak detection.
    envelope = waveform

    envelope_centered = waveform

    # --------------------------------------------------------------
    # 3. Peak detection
    # --------------------------------------------------------------

    peak_result = detect_envelope_peaks(
        envelope,
        fs=TARGET_FS,
        min_distance_seconds=min_distance_seconds,
        prominence_factor=prominence_factor,
        min_hr=min_hr,
        max_hr=max_hr,
    )

    peaks = peak_result["peaks"]

    intervals_seconds = (
        peak_result["intervals_seconds"]
    )

    instantaneous_bpm = (
        peak_result["instantaneous_bpm"]
    )

    accepted_interval_mask = (
        peak_result["accepted_interval_mask"]
    )

    accepted_bpm = (
        peak_result["accepted_bpm"]
    )

    median_bpm = (
        peak_result["median_bpm"]
    )

    # --------------------------------------------------------------
    # 4. Ground truth BPM
    # --------------------------------------------------------------

    gt_mask = (
        (gt_df["ts"] >= ws)
        & (gt_df["ts"] < we)
    )

    gt_vals = (
        gt_df.loc[
            gt_mask,
            "bpm",
        ]
        .to_numpy(dtype=float)
    )

    gt_vals = gt_vals[
        np.isfinite(gt_vals)
    ]

    if len(gt_vals) > 0:

        gt_bpm_median = float(
            np.median(gt_vals)
        )

        gt_freq = (
            gt_bpm_median / 60.0
        )

    else:

        gt_bpm_median = np.nan
        gt_freq = np.nan

    # --------------------------------------------------------------
    # 5. Accepted peak locations
    # --------------------------------------------------------------

    # An accepted interval connects peaks[i] -> peaks[i+1].
    # We display the second peak of each accepted interval
    # in green, while all detected peaks are red.

    if len(peaks) > 1:

        accepted_peak_indices = (
            peaks[1:][accepted_interval_mask]
        )

    else:

        accepted_peak_indices = np.array(
            [],
            dtype=int,
        )

    # --------------------------------------------------------------
    # Return diagnostic results
    # --------------------------------------------------------------

    return {
        "seg": seg,
        "seg_ts": seg_ts,

        "fused": fused,
        "envelope": envelope,
        "envelope_centered": envelope_centered,

        "quality": quality,
        "top_names": top_names,

        "peaks": peaks,
        "accepted_peaks": accepted_peak_indices,

        "peak_properties": peak_result["properties"],
        "prominence_threshold": peak_result[
            "prominence_threshold"
        ],

        "intervals_seconds": intervals_seconds,
        "instantaneous_bpm": instantaneous_bpm,
        "accepted_interval_mask": accepted_interval_mask,
        "accepted_bpm": accepted_bpm,

        "peak_bpm": median_bpm,

        "gt_bpm": gt_bpm_median,
        "gt_freq": gt_freq,
        "gt_bpm_vals": gt_vals,
    }


# ----------------------------------------------------------------------
# Safe savefig
# ----------------------------------------------------------------------

def _safe_savefig(
    fig,
    out_file,
    dpi=150,
):
    """
    Save figure.

    If the output file is locked, save to a timestamped
    fallback filename instead.
    """

    out_path = Path(out_file)

    try:

        fig.savefig(
            out_path,
            dpi=dpi,
        )

        return out_path

    except OSError as e:

        ts = datetime.now().strftime(
            "%H%M%S"
        )

        alt = out_path.with_name(
            f"{out_path.stem}_{ts}{out_path.suffix}"
        )

        print(
            f"Warning: could not write '{out_path}' "
            f"({e}). Falling back to '{alt}'"
        )

        try:

            fig.savefig(
                alt,
                dpi=dpi,
            )

            return alt

        except OSError as e2:

            print(
                f"Warning: fallback also failed "
                f"({e2}). Not saving plot."
            )

            return None


# ----------------------------------------------------------------------
# Plot diagnostic
# ----------------------------------------------------------------------

def plot_diagnostic(
    res,
    ws,
    we,
    csi_path,
    gt_path,
    out_file,
    mode="gt_csv",
):
    """
    Create time-domain peak diagnostic figure.
    """

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 16),
    )

    # --------------------------------------------------------------
    # Panel 1 — Fused waveform
    # --------------------------------------------------------------

    ax = axes[0]

    t_rel = (
        res["seg_ts"] - res["seg_ts"][0]
    )

    ax.plot(
        t_rel,
        res["fused"],
        linewidth=1.0,
        color="steelblue",
        label="Fused CSI",
    )

    peaks = res["peaks"]

    if len(peaks) > 0:

        ax.plot(
            t_rel[peaks],
            res["fused"][peaks],
            "ro",
            markersize=5,
            label=f"Detected peaks ({len(peaks)})",
        )

    accepted_peaks = res["accepted_peaks"]

    if len(accepted_peaks) > 0:

        ax.plot(
            t_rel[accepted_peaks],
            res["fused"][accepted_peaks],
            "go",
            markersize=7,
            markerfacecolor="none",
            markeredgewidth=1.5,
            label="Accepted interval peaks",
        )

    ax.set_xlabel(
        "Time within window (s)"
    )

    ax.set_ylabel(
        "Fused CSI amplitude"
    )

    ax.set_title(
        f"Fused heartbeat waveform "
        f"(top {TOP_N}: {', '.join(res['top_names'])})"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="upper right",
        fontsize=9,
    )

    # --------------------------------------------------------------
    # Panel 2 — Hilbert envelope + peaks
    # --------------------------------------------------------------

    ax = axes[1]

    ax.plot(
        t_rel,
        res["envelope_centered"],
        linewidth=1.2,
        color="darkorange",
        label="Centered Hilbert envelope",
    )

    if len(peaks) > 0:

        ax.plot(
            t_rel[peaks],
            res["envelope_centered"][peaks],
            "ro",
            markersize=5,
            label="All detected peaks",
        )

    if len(accepted_peaks) > 0:

        ax.plot(
            t_rel[accepted_peaks],
            res["envelope_centered"][accepted_peaks],
            "go",
            markersize=7,
            markerfacecolor="none",
            markeredgewidth=1.5,
            label="Accepted peaks",
        )

    prominence_threshold = (
        res["prominence_threshold"]
    )

    ax.axhline(
        0.0,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )

    ax.set_xlabel(
        "Time within window (s)"
    )

    ax.set_ylabel(
        "Envelope amplitude"
    )

    ax.set_title(
        "Hilbert envelope with peak detection"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="upper right",
        fontsize=9,
    )

    # --------------------------------------------------------------
    # Panel 3 — IBI and instantaneous BPM
    # --------------------------------------------------------------

    ax = axes[2]

    intervals = res["intervals_seconds"]

    bpm = res["instantaneous_bpm"]

    accepted_mask = (
        res["accepted_interval_mask"]
    )

    if len(intervals) > 0:

        beat_numbers = np.arange(
            1,
            len(intervals) + 1,
        )

        ax2 = ax.twinx()

        # All IBI values.
        ax.plot(
            beat_numbers,
            intervals,
            "o-",
            color="steelblue",
            markersize=5,
            linewidth=1.0,
            label="IBI",
        )

        # Accepted intervals.
        if np.any(accepted_mask):

            ax.plot(
                beat_numbers[accepted_mask],
                intervals[accepted_mask],
                "go",
                markersize=7,
                markerfacecolor="none",
                markeredgewidth=1.5,
                label="Accepted IBI",
            )

        # Instantaneous BPM.
        ax2.plot(
            beat_numbers,
            bpm,
            "s--",
            color="darkorange",
            markersize=4,
            linewidth=1.0,
            label="Instantaneous BPM",
        )

        if np.any(accepted_mask):

            ax2.plot(
                beat_numbers[accepted_mask],
                bpm[accepted_mask],
                "go",
                markersize=6,
                markerfacecolor="none",
                markeredgewidth=1.5,
                label="Accepted BPM",
            )

        # GT reference.
        if np.isfinite(res["gt_bpm"]):

            gt_interval = (
                60.0 / res["gt_bpm"]
            )

            ax.axhline(
                gt_interval,
                color="green",
                linestyle="--",
                alpha=0.7,
                label=(
                    f"GT IBI = {gt_interval:.3f} s"
                ),
            )

            ax2.axhline(
                res["gt_bpm"],
                color="green",
                linestyle=":",
                alpha=0.7,
                label=(
                    f"GT = {res['gt_bpm']:.1f} BPM"
                ),
            )

        ax.set_xlabel(
            "Beat interval number"
        )

        ax.set_ylabel(
            "IBI (seconds)",
            color="steelblue",
        )

        ax2.set_ylabel(
            "Instantaneous BPM",
            color="darkorange",
        )

        ax.set_title(
            "Inter-beat intervals and instantaneous BPM"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        # Combine legends.
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
            fontsize=8,
        )

    else:

        ax.text(
            0.5,
            0.5,
            "No valid peak intervals detected",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

        ax.set_title(
            "Inter-beat intervals — no intervals"
        )

    # --------------------------------------------------------------
    # Panel 4 — Subcarrier quality ranking
    # --------------------------------------------------------------

    ax = axes[3]

    q = res["quality"].head(20)

    y = np.arange(len(q))

    ax.barh(
        y,
        q["quality_score"].to_numpy(),
        color="steelblue",
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        q["name"].tolist(),
        fontsize=8,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Quality score"
    )

    ax.set_ylabel(
        "Subcarrier"
    )

    ax.set_title(
        "Top 20 subcarriers by quality score"
    )

    ax.grid(
        True,
        alpha=0.3,
        axis="x",
    )

    # --------------------------------------------------------------
    # Summary box
    # --------------------------------------------------------------

    gt_file_display = (
        gt_path
        if gt_path
        else "<none>"
    )

    if mode == "baseline":

        gt_line = (
            "GT BPM      : — (baseline mode)"
        )

        err_line = (
            "error       : —"
        )

    else:

        gt_line = (
            f"GT BPM      : {res['gt_bpm']:.2f} "
            f"({len(res['gt_bpm_vals'])} samples"
            + (
                f", σ={np.std(res['gt_bpm_vals']):.2f})"
                if len(res["gt_bpm_vals"]) > 1
                else ")"
            )
        )

        if np.isfinite(res["gt_bpm"]):

            err_line = (
                f"error       : "
                f"{res['peak_bpm'] - res['gt_bpm']:+.2f} BPM"
            )

        else:

            err_line = (
                "error       : —"
            )

    # Instantaneous BPM statistics.
    bpm = res["instantaneous_bpm"]

    accepted_bpm = res["accepted_bpm"]

    if len(accepted_bpm) > 0:

        bpm_median = np.median(
            accepted_bpm
        )

        bpm_std = np.std(
            accepted_bpm
        )

        bpm_line = (
            f"IBI BPM     : {bpm_median:.2f} "
            f"(σ={bpm_std:.2f})"
        )

    else:

        bpm_line = (
            "IBI BPM     : — (no valid intervals)"
        )

    summary = (
        f"mode        : {mode}\n"
        f"window      : {ws:.2f} → {we:.2f} "
        f"({we - ws:.1f} s)\n"
        f"window samples: {len(res['seg'])}\n"
        f"peaks       : {len(res['peaks'])}\n"
        f"accepted IBIs: {len(accepted_bpm)}\n"
        f"CSI BPM     : {res['peak_bpm']:.2f}\n"
        f"{bpm_line}\n"
        f"min distance: {DEFAULT_MIN_DISTANCE_SECONDS:.3f} s\n"
        f"prominence  : {res['prominence_threshold']:.5f}\n"
        f"{gt_line}\n"
        f"{err_line}\n"
        f"CSI file    : {Path(csi_path).name}\n"
        f"GT  file    : "
        f"{Path(gt_file_display).name if gt_file_display and gt_file_display != '<none>' else '<none>'}"
    )

    fig.text(
        0.02,
        0.015,
        summary,
        fontsize=9,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            facecolor="lightyellow",
            alpha=0.9,
        ),
    )

    plt.suptitle(
        "Time-domain peak / IBI diagnostic",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(
        rect=(0, 0.12, 1, 0.97)
    )

    saved_path = _safe_savefig(
        fig,
        out_file,
        dpi=150,
    )

    plt.close()

    if saved_path is not None:

        print(
            f"\nSaved: {saved_path}"
        )

    print(summary)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser(
        description=(
            "Time-domain CSI heartbeat peak diagnostic. "
            "Accepts .csv, .pcap, or .pcapng captures."
        )
    )

    ap.add_argument(
        "--csi",
        required=True,
        help=(
            "CSI capture file "
            "(.csv, .pcap, or .pcapng)"
        ),
    )

    # --------------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------------

    gt_group = ap.add_mutually_exclusive_group()

    gt_group.add_argument(
        "--gt",
        default=None,
        help=(
            "Ground truth CSV with columns ts,bpm. "
            "If omitted with --true-hr and --no-gt, "
            "runs in baseline-only mode."
        ),
    )

    gt_group.add_argument(
        "--true-hr",
        type=float,
        default=None,
        help=(
            "Known heart rate in BPM "
            "(synthetic ground truth)."
        ),
    )

    gt_group.add_argument(
        "--no-gt",
        action="store_true",
        help=(
            "Force baseline-only mode "
            "(no GT overlay)."
        ),
    )

    # --------------------------------------------------------------
    # Window selection
    # --------------------------------------------------------------

    win_group = ap.add_mutually_exclusive_group()

    win_group.add_argument(
        "--window-start",
        type=float,
        default=None,
        help=(
            "Window start in seconds from capture start "
            "(default: 20)"
        ),
    )

    win_group.add_argument(
        "--ts-start",
        type=float,
        default=None,
        help=(
            "Absolute timestamp for window start."
        ),
    )

    ap.add_argument(
        "--window-seconds",
        type=float,
        default=10.0,
        help=(
            "Window length in seconds "
            "(default: 10)"
        ),
    )

    # --------------------------------------------------------------
    # Peak parameters
    # --------------------------------------------------------------

    ap.add_argument(
        "--min-distance",
        type=float,
        default=DEFAULT_MIN_DISTANCE_SECONDS,
        help=(
            "Minimum distance between peaks in seconds. "
            f"Default: {DEFAULT_MIN_DISTANCE_SECONDS:.3f} "
            "(approximately 180 BPM maximum)."
        ),
    )

    ap.add_argument(
        "--prominence-factor",
        type=float,
        default=DEFAULT_PROMINENCE_FACTOR,
        help=(
            "Peak prominence threshold as a fraction "
            "of envelope standard deviation. "
            "Default: 0.30."
        ),
    )

    ap.add_argument(
        "--min-hr",
        type=float,
        default=DEFAULT_MIN_HR,
        help=(
            "Minimum allowed HR for interval rejection. "
            "Default: 40 BPM."
        ),
    )

    ap.add_argument(
        "--max-hr",
        type=float,
        default=DEFAULT_MAX_HR,
        help=(
            "Maximum allowed HR for interval rejection. "
            "Default: 180 BPM."
        ),
    )

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Output PNG path. "
            "If omitted, a unique name is derived "
            "from the CSI file stem and window offset."
        ),
    )

    ap.add_argument(
        "--drop-guard",
        type=int,
        default=0,
        help=(
            "Drop this many subcarriers from each edge "
            "before analysis. Recommended: 2."
        ),
    )

    args = ap.parse_args()

    # --------------------------------------------------------------
    # Validate peak parameters
    # --------------------------------------------------------------

    if args.min_distance <= 0:

        raise ValueError(
            "--min-distance must be greater than zero."
        )

    if args.prominence_factor < 0:

        raise ValueError(
            "--prominence-factor must be >= 0."
        )

    if args.min_hr <= 0:

        raise ValueError(
            "--min-hr must be greater than zero."
        )

    if args.max_hr <= args.min_hr:

        raise ValueError(
            "--max-hr must be greater than --min-hr."
        )

    # --------------------------------------------------------------
    # Load CSI
    # --------------------------------------------------------------

    csi_path = Path(args.csi)

    print(
        f"Loading CSI: {csi_path}"
    )

    csi, ts, meta = load_csi(
        csi_path
    )

    print(
        f"  {len(csi)} packets, "
        f"{csi.shape[1]} subcarriers"
    )

    print(
        f"  rate: {meta.get('rate', 0):.2f} Hz"
    )

    if len(ts) > 1:

        print(
            f"  time span: "
            f"{ts[-1] - ts[0]:.2f} s"
        )

    # --------------------------------------------------------------
    # Drop guard subcarriers
    # --------------------------------------------------------------

    if args.drop_guard > 0:

        g = args.drop_guard

        if csi.shape[1] <= 2 * g:

            raise ValueError(
                f"--drop-guard {g} would leave no subcarriers "
                f"(only {csi.shape[1]} present)."
            )

        csi = csi[:, g:-g]

        print(
            f"  dropped {g} guard SCs on each side "
            f"-> {csi.shape[1]} subcarriers remain"
        )

    # --------------------------------------------------------------
    # Clipping check
    # --------------------------------------------------------------

    check_clipping(
        csi,
        verbose=True,
    )

    # --------------------------------------------------------------
    # Uniform resampling
    # --------------------------------------------------------------

    csi_u, ts_u = resample_to_uniform(
        csi,
        ts,
        TARGET_FS,
    )

    print(
        f"  resampled to {len(csi_u)} samples "
        f"at {TARGET_FS:.1f} Hz"
    )

    # --------------------------------------------------------------
    # Load / construct GT
    # --------------------------------------------------------------

    if args.no_gt:

        mode = "baseline"

        print(
            "Loading GT: "
            "<baseline mode — no ground truth>"
        )

        gt_df = pd.DataFrame({
            "ts": [
                ts_u[0] - 10.0,
                ts_u[-1] + 10.0,
            ],
            "bpm": [
                np.nan,
                np.nan,
            ],
        })

        gt_path_display = ""

    elif args.true_hr is not None:

        mode = "true_hr"

        print(
            f"Loading GT: "
            f"<synthetic {args.true_hr:.2f} BPM>"
        )

        gt_df = pd.DataFrame({
            "ts": [
                ts_u[0] - 10.0,
                ts_u[-1] + 10.0,
            ],
            "bpm": [
                args.true_hr,
                args.true_hr,
            ],
        })

        gt_path_display = (
            f"<true_hr={args.true_hr:.2f}>"
        )

    elif args.gt is not None:

        mode = "gt_csv"

        print(
            f"Loading GT: {args.gt}"
        )

        gt_df = pd.read_csv(
            args.gt
        )

        # Support your current GT format.
        if (
            "ts" not in gt_df.columns
            or "bpm" not in gt_df.columns
        ):

            raise ValueError(
                "Ground truth CSV must contain "
                "'ts' and 'bpm' columns."
            )

        gt_df["ts"] = pd.to_numeric(
            gt_df["ts"],
            errors="coerce",
        )

        gt_df["bpm"] = pd.to_numeric(
            gt_df["bpm"],
            errors="coerce",
        )

        gt_df = (
            gt_df
            .dropna()
            .sort_values("ts")
            .reset_index(drop=True)
        )

        print(
            f"  {len(gt_df)} BPM samples"
        )

        gt_path_display = args.gt

    else:

        mode = "baseline"

        print(
            "Loading GT: "
            "<no --gt provided — baseline mode>"
        )

        gt_df = pd.DataFrame({
            "ts": [
                ts_u[0] - 10.0,
                ts_u[-1] + 10.0,
            ],
            "bpm": [
                np.nan,
                np.nan,
            ],
        })

        gt_path_display = ""

    # --------------------------------------------------------------
    # Determine window
    # --------------------------------------------------------------

    if args.ts_start is not None:

        ws = args.ts_start

        we = (
            ws + args.window_seconds
        )

        win_tag = (
            f"ts{int(args.ts_start)}"
        )

    else:

        offset = (
            args.window_start
            if args.window_start is not None
            else 20.0
        )

        ws = (
            ts_u[0] + offset
        )

        we = (
            ws + args.window_seconds
        )

        win_tag = (
            f"w{int(offset)}"
        )

    print(
        f"\nAnalyzing window: "
        f"{ws:.3f} → {we:.3f}"
    )

    # --------------------------------------------------------------
    # Output path
    # --------------------------------------------------------------

    if args.out is None:

        out_file = Path(
            f"diagnose_peaks_"
            f"{csi_path.stem}_"
            f"{win_tag}.png"
        )

    else:

        out_file = Path(
            args.out
        )

    # --------------------------------------------------------------
    # Analyze + plot
    # --------------------------------------------------------------

    res = analyze_window_peaks(
        csi_u,
        ts_u,
        gt_df,
        ws,
        we,
        min_distance_seconds=args.min_distance,
        prominence_factor=args.prominence_factor,
        min_hr=args.min_hr,
        max_hr=args.max_hr,
    )

    plot_diagnostic(
        res,
        ws,
        we,
        args.csi,
        gt_path_display,
        out_file,
        mode=mode,
    )


if __name__ == "__main__":

    main()