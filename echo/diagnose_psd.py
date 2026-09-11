#!/usr/bin/env python3
"""
diagnose_psd.py
===============

Zoom into a single window of one capture and show:
    1. CSI heartbeat waveform (top subcarriers)
    2. Welch PSD with detected peak and (optionally) GT marker
    3. Subcarrier quality ranking

Input formats:
    .csv           - CSI captures written by live_monitor.py
    .pcap / .pcapng - Nexmon CSI PCAP captures

Modes:
    (a) With ground truth CSV (default):
        python diagnose_psd.py --csi <csv|pcap> --gt <csv> \\
            --window-start 20 --window-seconds 10

    (b) With a known HR (synthetic testing):
        python diagnose_psd.py --csi <csv|pcap> --true-hr 90 \\
            --window-start 20 --window-seconds 10

    (c) Baseline-only, no GT:
        python diagnose_psd.py --csi <csv|pcap> --no-gt \\
            --window-start 20 --window-seconds 10

    Window can also be selected by absolute timestamp:
        --ts-start 1788806029.0 --window-seconds 10

Output:
    If --out is not given, a unique default name is derived from the
    input file stem and the window offset:
        diagnose_psd_<stem>_w<offset>.png
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

# Pipeline for processing.
from windowed_psd_analysis import HeartbeatPipeline

# Shared clipping check.
from read_pcap import check_clipping


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

TARGET_FS = 100.0
BANDPASS_LOW = 0.8
BANDPASS_HIGH = 2.5
FILTER_ORDER = 4
TOP_N = 5


# ----------------------------------------------------------------------
# Load CSI — dispatches by extension
# ----------------------------------------------------------------------

def _load_csv_csi(csi_path: Path):
    """Load a CSV capture written by live_monitor.py."""
    from read_csv_csi import CSIDataLoader

    loader = CSIDataLoader(str(csi_path))
    csi_data, meta = loader.load()

    if hasattr(loader, "df") and "ts" in loader.df.columns:
        ts = pd.to_numeric(
            loader.df["ts"], errors="coerce"
        ).to_numpy(dtype=float)
    else:
        raise ValueError("CSI CSV has no 'ts' column")

    valid = np.isfinite(ts)
    ts = ts[valid]
    csi_data = csi_data[valid]

    if len(ts) != len(csi_data):
        raise ValueError("Timestamp / CSI length mismatch in CSV")

    return csi_data, ts, meta


def _load_pcap_csi(csi_path: Path):
    """Load a PCAP/PCAPNG capture written by live_monitor.py."""
    from read_pcap import PCAPDataLoader

    loader = PCAPDataLoader(str(csi_path))
    csi_data, meta = loader.load()

    ts = None

    if hasattr(loader, "csi_timestamps") and loader.csi_timestamps is not None:
        ts = np.asarray(loader.csi_timestamps, dtype=float)
    elif "timestamps" in meta:
        ts = np.asarray(meta["timestamps"], dtype=float)

    if ts is None or len(ts) != len(csi_data):
        rate = float(meta.get("rate", 0.0)) or 93.5
        print(
            f"Warning: no per-packet PCAP timestamps — "
            f"using synthetic {rate:.2f} Hz grid"
        )
        ts = np.arange(len(csi_data), dtype=float) / rate

    return csi_data, ts, meta


def load_csi(csi_path: Path):
    """Load CSI data from CSV or PCAP; returns (csi, timestamps, meta)."""
    suffix = csi_path.suffix.lower()

    if suffix == ".csv":
        return _load_csv_csi(csi_path)
    if suffix in (".pcap", ".pcapng"):
        return _load_pcap_csi(csi_path)

    raise ValueError(
        f"Unsupported capture format: {suffix}. "
        f"Expected .csv, .pcap, or .pcapng"
    )


# ----------------------------------------------------------------------
# Uniform resample
# ----------------------------------------------------------------------

def resample_to_uniform(csi, ts, target_fs=TARGET_FS):
    ts = np.asarray(ts, dtype=float)
    order = np.argsort(ts)
    ts = ts[order]
    csi = csi[order]

    step = 1.0 / target_fs
    uniform = np.arange(ts[0], ts[-1] + 0.5 * step, step)

    idx = np.searchsorted(ts, uniform, side="right")
    idx = np.clip(idx, 1, len(ts) - 1)
    i0 = idx - 1
    i1 = idx

    t0 = ts[i0]
    t1 = ts[i1]
    dt = t1 - t0
    dt[dt == 0] = 1.0
    w1 = (uniform - t0) / dt
    w0 = 1.0 - w1

    out = w0[:, None] * csi[i0] + w1[:, None] * csi[i1]
    return out, uniform


# ----------------------------------------------------------------------
# Analyze one window
# ----------------------------------------------------------------------

def analyze_window(csi, ts, gt_df, ws, we):
    mask = (ts >= ws) & (ts < we)
    seg = csi[mask]
    seg_ts = ts[mask]

    if len(seg) < 20:
        raise RuntimeError(f"Only {len(seg)} samples in window — too few")

    pipeline = HeartbeatPipeline(TARGET_FS)

    amp = pipeline.extract_amplitude(seg)
    channels = [amp[:, sc] for sc in range(amp.shape[1])]
    names = [f"amp_sc{sc}" for sc in range(amp.shape[1])]

    processed, quality = pipeline.process_all_channels(
        channels, names,
        low=BANDPASS_LOW, high=BANDPASS_HIGH, order=FILTER_ORDER,
    )

    fused, top_names = pipeline.fuse_top_channels(
        processed, quality, top_n=TOP_N, align_polarity=True,
    )

    x = fused - np.mean(fused)
    freqs, psd = signal.welch(
        x,
        fs=TARGET_FS,
        window="hann",
        nperseg=len(x),
        noverlap=0,
        nfft=max(4096, len(x)),
        detrend="constant",
        scaling="density",
    )

    band = (freqs >= BANDPASS_LOW) & (freqs <= BANDPASS_HIGH)
    band_freqs = freqs[band]
    band_psd = psd[band]

    peak_idx = int(np.argmax(band_psd))
    peak_freq = float(band_freqs[peak_idx])
    peak_bpm = peak_freq * 60.0

    gt_mask = (gt_df["ts"] >= ws) & (gt_df["ts"] < we)
    gt_vals = gt_df.loc[gt_mask, "bpm"].to_numpy(dtype=float)
    gt_vals = gt_vals[np.isfinite(gt_vals)]

    if len(gt_vals) > 0:
        gt_bpm_median = float(np.median(gt_vals))
        gt_freq = gt_bpm_median / 60.0
    else:
        gt_bpm_median = np.nan
        gt_freq = np.nan

    return {
        "seg": seg,
        "seg_ts": seg_ts,
        "fused": fused,
        "quality": quality,
        "top_names": top_names,
        "freqs": freqs,
        "psd": psd,
        "band_freqs": band_freqs,
        "band_psd": band_psd,
        "peak_freq": peak_freq,
        "peak_bpm": peak_bpm,
        "gt_bpm": gt_bpm_median,
        "gt_freq": gt_freq,
        "gt_bpm_vals": gt_vals,
    }


# ----------------------------------------------------------------------
# Safe savefig with fallback name
# ----------------------------------------------------------------------

def _safe_savefig(fig, out_file, dpi=150):
    """
    Save a figure, falling back to a timestamped name if the target
    file is locked (common on Windows when an image viewer keeps a
    handle open).
    """
    out_path = Path(out_file)

    try:
        fig.savefig(out_path, dpi=dpi)
        return out_path
    except OSError as e:
        ts = datetime.now().strftime("%H%M%S")
        alt = out_path.with_name(
            f"{out_path.stem}_{ts}{out_path.suffix}"
        )
        print(
            f"Warning: could not write '{out_path}' ({e}). "
            f"Falling back to '{alt}'"
        )
        try:
            fig.savefig(alt, dpi=dpi)
            return alt
        except OSError as e2:
            print(f"Warning: fallback also failed ({e2}). Not saving plot.")
            return None


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def plot_diagnostic(res, ws, we, csi_path, gt_path, out_file, mode="gt_csv"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # ---------------- Panel 1: Waveform ----------------
    ax = axes[0]
    t_rel = res["seg_ts"] - res["seg_ts"][0]
    ax.plot(t_rel, res["fused"], linewidth=1.0, color="steelblue")
    ax.set_xlabel("Time within window (s)")
    ax.set_ylabel("Fused CSI amplitude (normalized)")
    ax.set_title(
        f"Fused heartbeat waveform "
        f"(top {TOP_N}: {', '.join(res['top_names'])})"
    )
    ax.grid(True, alpha=0.3)

    # ---------------- Panel 2: PSD ----------------
    ax = axes[1]

    full = (res["freqs"] >= 0.05) & (res["freqs"] <= 5.0)
    ax.semilogy(
        res["freqs"][full],
        res["psd"][full],
        color="gray", linewidth=1.0, label="full PSD",
    )

    ax.semilogy(
        res["band_freqs"], res["band_psd"],
        color="steelblue", linewidth=1.5, label="HR band",
    )

    if np.isfinite(res["gt_freq"]):
        ax.axvline(
            res["gt_freq"], color="green", linestyle="--", alpha=0.8,
            label=f"GT = {res['gt_bpm']:.1f} BPM ({res['gt_freq']:.3f} Hz)",
        )

    ax.axvline(
        res["peak_freq"], color="red", linestyle=":", alpha=0.9,
        label=f"CSI peak = {res['peak_bpm']:.1f} BPM ({res['peak_freq']:.3f} Hz)",
    )

    ax.axvspan(BANDPASS_LOW, BANDPASS_HIGH, alpha=0.08, color="blue")
    ax.set_xlim(0, 4.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Welch PSD of the fused CSI signal")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)

    if np.isfinite(res["gt_bpm"]):
        err = res["peak_bpm"] - res["gt_bpm"]
        box_text = f"|CSI - GT| = {abs(err):.1f} BPM"
    else:
        box_text = f"CSI peak = {res['peak_bpm']:.1f} BPM"

    ax.text(
        0.02, 0.95, box_text,
        transform=ax.transAxes, fontsize=10,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    # ---------------- Panel 3: Subcarrier quality ----------------
    ax = axes[2]
    q = res["quality"].head(20)
    y = np.arange(len(q))
    ax.barh(y, q["quality_score"].to_numpy(), color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels(q["name"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Quality score (peak HR-band / median noise)")
    ax.set_ylabel("Subcarrier")
    ax.set_title("Top 20 subcarriers by quality score")
    ax.grid(True, alpha=0.3, axis="x")

    # ---------------- Summary box ----------------
    gt_file_display = gt_path if gt_path else "<none>"

    if mode == "baseline":
        gt_line = "GT BPM      : — (baseline mode, no GT)"
        err_line = "error       : —"
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
        err_line = f"error       : {res['peak_bpm'] - res['gt_bpm']:+.2f} BPM"

    summary = (
        f"mode        : {mode}\n"
        f"window      : {ws:.2f} → {we:.2f}  ({we - ws:.1f} s)\n"
        f"window samples: {len(res['seg'])}\n"
        f"CSI peak    : {res['peak_bpm']:.2f} BPM "
        f"({res['peak_freq']:.4f} Hz)\n"
        f"{gt_line}\n"
        f"{err_line}\n"
        f"CSI file    : {Path(csi_path).name}\n"
        f"GT  file    : "
        f"{Path(gt_file_display).name if gt_file_display and gt_file_display != '<none>' else '<none>'}"
    )

    fig.text(
        0.02, 0.02, summary,
        fontsize=9, family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    plt.suptitle(
        "PSD diagnostic — one clean window",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0.10, 1, 0.97))

    saved_path = _safe_savefig(fig, out_file, dpi=150)
    plt.close()

    if saved_path is not None:
        print(f"\nSaved: {saved_path}")
    print(summary)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "PSD diagnostic for CSI heartbeat analysis. "
            "Accepts .csv, .pcap, or .pcapng captures."
        )
    )

    ap.add_argument(
        "--csi", required=True,
        help="CSI capture file (.csv, .pcap, or .pcapng)"
    )

    # GT source flags
    gt_group = ap.add_mutually_exclusive_group()
    gt_group.add_argument(
        "--gt", default=None,
        help="Ground truth CSV (ts,bpm). "
             "If omitted along with --true-hr and --no-gt, "
             "runs in baseline-only mode."
    )
    gt_group.add_argument(
        "--true-hr", type=float, default=None,
        help="Known heart rate in BPM (synthetic GT). "
             "Intended for synthetic / self-test captures."
    )
    gt_group.add_argument(
        "--no-gt", action="store_true",
        help="Force baseline-only mode (no GT overlay)."
    )

    # Window selection
    win_group = ap.add_mutually_exclusive_group()
    win_group.add_argument(
        "--window-start", type=float, default=None,
        help="Window start in seconds from capture start (default: 20)"
    )
    win_group.add_argument(
        "--ts-start", type=float, default=None,
        help="Absolute timestamp for window start"
    )

    ap.add_argument(
        "--window-seconds", type=float, default=10.0,
        help="Window length in seconds"
    )
    ap.add_argument(
        "--out", default=None,
        help=(
            "Output PNG path. If not provided, a unique name is derived "
            "from the CSI file stem and the window offset."
        )
    )
    ap.add_argument(
        "--drop-guard", type=int, default=0,
        help=(
            "Drop this many subcarriers from each edge before analysis. "
            "Nexmon's lowest/highest subcarriers are guard tones and can "
            "carry extreme values that trigger false clipping. "
            "Recommended: 2."
        )
    )

    args = ap.parse_args()

    # ----------------------------------------------------------------
    # Load CSI (auto-detect format)
    # ----------------------------------------------------------------
    csi_path = Path(args.csi)
    print(f"Loading CSI: {csi_path}")

    csi, ts, meta = load_csi(csi_path)

    print(f"  {len(csi)} packets, {csi.shape[1]} subcarriers")
    print(f"  rate: {meta.get('rate', 0):.2f} Hz")
    if len(ts) > 1:
        print(f"  time span: {ts[-1] - ts[0]:.2f} s")

    # Optional: drop guard subcarriers from each edge.
    if args.drop_guard > 0:
        g = args.drop_guard
        if csi.shape[1] <= 2 * g:
            raise ValueError(
                f"--drop-guard {g} would leave no subcarriers "
                f"(only {csi.shape[1]} present)"
            )
        csi = csi[:, g:-g]
        print(
            f"  dropped {g} guard SCs on each side "
            f"-> {csi.shape[1]} subcarriers remain"
        )

    # Clip check AFTER guard removal, so the banner reflects
    # what will actually be analyzed.
    check_clipping(csi, verbose=True)

    # Resample to uniform grid
    csi_u, ts_u = resample_to_uniform(csi, ts, TARGET_FS)
    print(f"  resampled to {len(csi_u)} samples at {TARGET_FS:.1f} Hz")

    # ----------------------------------------------------------------
    # Load / construct GT
    # ----------------------------------------------------------------
    if args.no_gt:
        mode = "baseline"
        print("Loading GT: <baseline mode — no ground truth>")
        gt_df = pd.DataFrame({
            "ts":  [ts_u[0] - 10.0, ts_u[-1] + 10.0],
            "bpm": [np.nan, np.nan],
        })
        gt_path_display = ""

    elif args.true_hr is not None:
        mode = "true_hr"
        print(f"Loading GT: <synthetic GT = {args.true_hr} BPM>")
        gt_df = pd.DataFrame({
            "ts":  [ts_u[0] - 10.0, ts_u[-1] + 10.0],
            "bpm": [args.true_hr, args.true_hr],
        })
        gt_path_display = ""

    elif args.gt is not None:
        mode = "gt_csv"
        print(f"Loading GT: {args.gt}")
        gt_df = pd.read_csv(args.gt)
        if "ts" not in gt_df.columns or "bpm" not in gt_df.columns:
            raise ValueError(
                "Ground truth CSV must contain 'ts' and 'bpm' columns."
            )
        gt_df["ts"] = pd.to_numeric(gt_df["ts"], errors="coerce")
        gt_df["bpm"] = pd.to_numeric(gt_df["bpm"], errors="coerce")
        gt_df = gt_df.dropna().sort_values("ts").reset_index(drop=True)
        print(f"  {len(gt_df)} BPM samples")
        gt_path_display = args.gt

    else:
        mode = "baseline"
        print("Loading GT: <no --gt provided — baseline mode>")
        gt_df = pd.DataFrame({
            "ts":  [ts_u[0] - 10.0, ts_u[-1] + 10.0],
            "bpm": [np.nan, np.nan],
        })
        gt_path_display = ""

    # ----------------------------------------------------------------
    # Determine window
    # ----------------------------------------------------------------
    if args.ts_start is not None:
        ws = args.ts_start
        we = ws + args.window_seconds
        win_tag = f"ts{int(args.ts_start)}"
    else:
        offset = args.window_start if args.window_start is not None else 20.0
        ws = ts_u[0] + offset
        we = ws + args.window_seconds
        win_tag = f"w{int(offset)}"

    print(f"\nAnalyzing window: {ws:.3f} → {we:.3f}")

    # ----------------------------------------------------------------
    # Resolve output path
    # ----------------------------------------------------------------
    if args.out is None:
        csi_stem = csi_path.stem
        out_file = Path(f"diagnose_psd_{csi_stem}_{win_tag}.png")
    else:
        out_file = Path(args.out)

    # ----------------------------------------------------------------
    # Analyze + plot
    # ----------------------------------------------------------------
    res = analyze_window(csi_u, ts_u, gt_df, ws, we)

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