#!/usr/bin/env python3
"""
diagnose_psd.py
===============

Zoom into a single window of one capture and show:
    1. CSI heartbeat waveform (top subcarriers)
    2. Welch PSD with detected peak and GT marker
    3. GT BPM over the same window
    4. Subcarrier quality ranking

Usage:
    python diagnose_psd.py \\
        --csi <csi_capture.csv> \\
        --gt  <ground_truth.csv> \\
        --window-start 20 --window-seconds 10

    # or by absolute timestamp
    python diagnose_psd.py \\
        --csi <csv> --gt <csv> \\
        --ts-start 1788378695.0 --ts-end 1788378705.0

Output:
    diagnose_psd.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

# Reuse the existing pipeline classes.
from read_csv_csi import CSIDataLoader
from windowed_psd_analysis import HeartbeatPipeline


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

TARGET_FS = 100.0
BANDPASS_LOW = 0.8
BANDPASS_HIGH = 2.5
FILTER_ORDER = 4
TOP_N = 5


# ----------------------------------------------------------------------
# Load CSI
# ----------------------------------------------------------------------

def load_csi(csi_path: Path):
    loader = CSIDataLoader(str(csi_path))
    csi_data, meta = loader.load()

    if hasattr(loader, "df") and "ts" in loader.df.columns:
        ts = pd.to_numeric(loader.df["ts"], errors="coerce").to_numpy(float)
    else:
        raise ValueError("CSI CSV has no 'ts' column")

    valid = np.isfinite(ts)
    ts = ts[valid]
    csi_data = csi_data[valid]

    return csi_data, ts, meta


# ----------------------------------------------------------------------
# Simple uniform resample (matches what multi_capture_validation does)
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
# Analyze ONE window
# ----------------------------------------------------------------------

def analyze_window(csi, ts, gt_df, ws, we):
    """
    csi, ts : uniform CSI and timestamps
    gt_df   : DataFrame with ts,bpm
    ws, we  : window start/end in the CSI time base
    """
    mask = (ts >= ws) & (ts < we)
    seg = csi[mask]
    seg_ts = ts[mask]

    if len(seg) < 20:
        raise RuntimeError(
            f"Only {len(seg)} samples in window — too few"
        )

    pipeline = HeartbeatPipeline(TARGET_FS)

    # Build amplitude channels only (simpler, and it's the best method)
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

    # Welch PSD on the fused signal
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

    # GT BPM in this window
    gt_mask = (gt_df["ts"] >= ws) & (gt_df["ts"] < we)
    gt_bpm_vals = gt_df.loc[gt_mask, "bpm"].to_numpy()
    gt_bpm_median = float(np.median(gt_bpm_vals)) if len(gt_bpm_vals) else np.nan
    gt_freq = gt_bpm_median / 60.0 if np.isfinite(gt_bpm_median) else np.nan

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
        "gt_bpm_vals": gt_bpm_vals,
    }


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def plot_diagnostic(res, ws, we, csi_path, gt_path, out_file):
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

    # Show the whole useful band 0-5 Hz
    full = (res["freqs"] >= 0.05) & (res["freqs"] <= 5.0)
    ax.semilogy(
        res["freqs"][full],
        res["psd"][full],
        color="gray", linewidth=1.0, label="full PSD",
    )

    # Highlight the HR band
    ax.semilogy(
        res["band_freqs"], res["band_psd"],
        color="steelblue", linewidth=1.5, label="HR band",
    )

    # GT frequency marker
    if np.isfinite(res["gt_freq"]):
        ax.axvline(
            res["gt_freq"], color="green", linestyle="--", alpha=0.8,
            label=f"GT = {res['gt_bpm']:.1f} BPM ({res['gt_freq']:.3f} Hz)",
        )

    # Detected peak
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

    # Annotate frequency error
    if np.isfinite(res["gt_bpm"]):
        err = res["peak_bpm"] - res["gt_bpm"]
        ax.text(
            0.02, 0.95,
            f"|CSI - GT| = {abs(err):.1f} BPM",
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

    # ---------------- Summary ----------------
    summary = (
        f"window      : {ws:.2f} → {we:.2f}  "
        f"({we - ws:.1f} s)\n"
        f"window samples: {len(res['seg'])}\n"
        f"CSI peak    : {res['peak_bpm']:.2f} BPM "
        f"({res['peak_freq']:.4f} Hz)\n"
        f"GT BPM      : {res['gt_bpm']:.2f} "
        f"({len(res['gt_bpm_vals'])} samples, "
        f"σ={np.std(res['gt_bpm_vals']):.2f})\n"
        f"error       : {res['peak_bpm'] - res['gt_bpm']:+.2f} BPM\n"
        f"CSI file    : {Path(csi_path).name}\n"
        f"GT  file    : {Path(gt_path).name}"
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
    plt.savefig(out_file, dpi=150)
    plt.close()

    print(f"\nSaved: {out_file}")
    print(summary)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csi", required=True, help="CSI capture CSV")
    ap.add_argument("--gt", required=True, help="Ground truth CSV")

    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--window-start", type=float, default=None,
        help="Window start in seconds from CSI start (default: 20)"
    )
    g.add_argument(
        "--ts-start", type=float, default=None,
        help="Absolute timestamp for window start"
    )

    ap.add_argument(
        "--window-seconds", type=float, default=10.0,
        help="Window length in seconds"
    )
    ap.add_argument(
        "--out", default="diagnose_psd.png",
        help="Output PNG"
    )
    args = ap.parse_args()

    # ---- Load CSI ----
    print(f"Loading CSI: {args.csi}")
    csi, ts, meta = load_csi(Path(args.csi))
    print(f"  {len(csi)} packets, {csi.shape[1]} subcarriers")
    print(f"  rate: {meta.get('rate', 0):.2f} Hz")

    # ---- Resample to uniform 100 Hz ----
    csi_u, ts_u = resample_to_uniform(csi, ts, TARGET_FS)
    print(f"  resampled to {len(csi_u)} samples at {TARGET_FS:.1f} Hz")

    # ---- Load GT ----
    print(f"Loading GT: {args.gt}")
    gt_df = pd.read_csv(args.gt)
    gt_df["ts"] = pd.to_numeric(gt_df["ts"], errors="coerce")
    gt_df["bpm"] = pd.to_numeric(gt_df["bpm"], errors="coerce")
    gt_df = gt_df.dropna().sort_values("ts").reset_index(drop=True)
    print(f"  {len(gt_df)} BPM samples")

    # ---- Determine window ----
    if args.ts_start is not None:
        ws = args.ts_start
        we = ws + args.window_seconds
    else:
        offset = args.window_start if args.window_start is not None else 20.0
        ws = ts_u[0] + offset
        we = ws + args.window_seconds

    print(f"\nAnalyzing window: {ws:.3f} → {we:.3f}")

    res = analyze_window(csi_u, ts_u, gt_df, ws, we)

    plot_diagnostic(res, ws, we, args.csi, args.gt, args.out)


if __name__ == "__main__":
    main()