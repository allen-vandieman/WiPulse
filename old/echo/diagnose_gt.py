#!/usr/bin/env python3
"""
diagnose_gt.py
==============

Plot ground-truth BPM data from one or more WiPulse ground truth CSVs.

Purpose:
    Determine whether the ground truth itself is stable enough to
    validate CSI heartbeat extraction against.

Usage:
    python diagnose_gt.py gt1.csv [gt2.csv ...]

    or with explicit labels:

    python diagnose_gt.py --labels Allen-N,0,0 Allen-S,0,1 \\
        gt1.csv gt2.csv

Output:
    diagnose_gt_<stem>.png  for each CSV
    diagnose_gt_all.png     if more than one CSV is supplied
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Load one ground truth CSV
# ----------------------------------------------------------------------

def load_gt(gt_path: Path):
    df = pd.read_csv(gt_path)

    if "ts" not in df.columns:
        raise ValueError(f"{gt_path}: no 'ts' column")

    if "bpm" not in df.columns:
        raise ValueError(f"{gt_path}: no 'bpm' column")

    ts = pd.to_numeric(df["ts"], errors="coerce").to_numpy(dtype=float)
    bpm = pd.to_numeric(df["bpm"], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(ts) & np.isfinite(bpm)
    ts = ts[mask]
    bpm = bpm[mask]

    order = np.argsort(ts)
    ts = ts[order]
    bpm = bpm[order]

    if len(ts) < 2:
        raise ValueError(f"{gt_path}: too few samples")

    # Relative time in seconds from first sample.
    t_rel = ts - ts[0]

    return t_rel, bpm, ts


# ----------------------------------------------------------------------
# Plot one GT capture
# ----------------------------------------------------------------------

def plot_gt(t_rel, bpm, title, out_file, sample_interval=None):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # ---------------- Panel 1: BPM vs time ----------------
    ax = axes[0]
    ax.plot(t_rel, bpm, linewidth=0.8, alpha=0.85)
    ax.axhline(
        np.median(bpm), color="red", linestyle="--", alpha=0.6,
        label=f"median = {np.median(bpm):.1f} BPM",
    )
    ax.axhline(
        np.mean(bpm), color="orange", linestyle=":", alpha=0.6,
        label=f"mean = {np.mean(bpm):.1f} BPM",
    )
    ax.fill_between(
        t_rel,
        np.median(bpm) - np.std(bpm),
        np.median(bpm) + np.std(bpm),
        alpha=0.15, color="red",
        label=f"±1σ (σ={np.std(bpm):.1f})",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("BPM")
    ax.set_title(f"{title} — GT BPM over time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # ---------------- Panel 2: Distribution ----------------
    ax = axes[1]
    ax.hist(bpm, bins=40, alpha=0.7, edgecolor="black")
    ax.axvline(
        np.median(bpm), color="red", linestyle="--",
        label=f"median = {np.median(bpm):.1f}",
    )
    ax.set_xlabel("BPM")
    ax.set_ylabel("Count")
    ax.set_title("GT BPM distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---------------- Panel 3: Sliding window std ----------------
    # How much does the median GT BPM vary over a typical 10 s window?
    win = 10.0
    centers = []
    medians = []
    stds = []
    i = 0
    while i < len(t_rel):
        t0 = t_rel[i]
        j = i
        while j < len(t_rel) and t_rel[j] < t0 + win:
            j += 1
        if j - i >= 3:
            centers.append((t_rel[i] + t_rel[j - 1]) / 2)
            medians.append(np.median(bpm[i:j]))
            stds.append(np.std(bpm[i:j]))
        i += 1

    centers = np.asarray(centers)
    medians = np.asarray(medians)
    stds = np.asarray(stds)

    ax = axes[2]
    ax.plot(centers, medians, marker="o", markersize=3, label="window median")
    ax.fill_between(
        centers, medians - stds, medians + stds,
        alpha=0.3, label="±1σ within window",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("BPM")
    ax.set_title(f"GT stability over {win:.0f} s sliding windows")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---------------- Summary text ----------------
    if sample_interval is not None:
        sample_rate = 1.0 / sample_interval if sample_interval > 0 else np.nan
    else:
        sample_rate = np.nan

    summary = (
        f"samples     = {len(bpm)}\n"
        f"duration    = {t_rel[-1]:.1f} s\n"
        f"median rate = {sample_rate:.2f} Hz\n"
        f"BPM  median = {np.median(bpm):.1f}\n"
        f"BPM  range  = {bpm.min():.1f} – {bpm.max():.1f}\n"
        f"BPM  σ      = {np.std(bpm):.2f}\n"
        f"window σ    = {np.median(stds):.2f} (median over windows)"
    )

    fig.text(
        0.02, 0.02, summary,
        fontsize=9, family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0.06, 1, 0.97))
    plt.savefig(out_file, dpi=150)
    plt.close()

    print(f"Saved: {out_file}")
    print(summary)
    print()


# ----------------------------------------------------------------------
# Plot many GT files on one figure for cross-capture comparison
# ----------------------------------------------------------------------

def plot_gt_combined(gt_list, out_file):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    ax = axes[0]
    for label, (t_rel, bpm, _) in gt_list:
        ax.plot(t_rel, bpm, linewidth=0.8, alpha=0.7, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("BPM")
    ax.set_title("GT BPM over time — all captures")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    ax = axes[1]
    for label, (_, bpm, _) in gt_list:
        ax.hist(bpm, bins=30, alpha=0.35, label=label)
    ax.set_xlabel("BPM")
    ax.set_ylabel("Count")
    ax.set_title("GT BPM distribution — all captures")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    plt.suptitle("Ground truth cross-capture overview", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_file, dpi=150)
    plt.close()

    print(f"Saved: {out_file}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", help="Ground truth CSV file(s)")
    parser.add_argument(
        "--labels", nargs="*", default=None,
        help="Optional labels matching each CSV"
    )
    parser.add_argument(
        "--out-dir", default=".", help="Output directory"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(p) for p in args.csv]

    if args.labels is not None and len(args.labels) != len(paths):
        print("Warning: labels count != CSV count; ignoring labels")
        args.labels = None

    loaded = []

    for i, p in enumerate(paths):
        label = args.labels[i] if args.labels else p.stem

        try:
            t_rel, bpm, ts = load_gt(p)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue

        intervals = np.diff(ts)
        intervals = intervals[intervals > 0]
        sample_interval = float(np.median(intervals)) if len(intervals) else None

        out_file = out_dir / f"diagnose_gt_{p.stem}.png"
        plot_gt(t_rel, bpm, label, out_file, sample_interval)

        loaded.append((label, (t_rel, bpm, ts)))

    if len(loaded) > 1:
        plot_gt_combined(loaded, out_dir / "diagnose_gt_all.png")


if __name__ == "__main__":
    main()