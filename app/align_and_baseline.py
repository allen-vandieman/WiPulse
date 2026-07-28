#!/usr/bin/env python3
"""
align_and_baseline.py
======================
Baseline check: does the CSI amplitude signal actually contain a usable
heart-rate signature, before spending any effort on an ML model?

Takes the two CSVs produced by this project --

    output/csi_capture_<timestamp>.csv     (from live_monitor.py, 'c' to record)
    output/ground_truth.csv                (from heartbeat-giop.py)

-- and:

  1. loads + aligns them on their shared time.time() timestamp column
  2. resamples the (possibly irregularly-timed) CSI amplitude onto a
     uniform time grid at its estimated sample rate
  3. band-pass filters that signal to the human heart-rate band
     (default 0.5-3.7 Hz == 30-220 BPM, matching heartbeat-giop.py's
     MIN_BPM/MAX_BPM) to knock out breathing and slow motion artifacts
  4. slides a window (default 8s, 50% overlap) over the filtered signal,
     FFTs each window (Hanning-windowed, like analysis.fft_motion_detection),
     and takes the dominant in-band frequency as the CSI-estimated BPM
  5. matches each window against the ground-truth beats that fall inside
     it (or the nearest beat, if none fall inside) and reports MAE/RMSE,
     plus a plot of estimated vs. true BPM over time and a scatter/parity
     comparison

This is intentionally *not* the ML model itself -- it's the sanity check
that tells you whether the raw signal-processing approach is anywhere
close, which is much easier to debug than a model that silently learns
nothing useful. The windowed results are also saved as a CSV so they can
be reused directly as (feature, label) pairs for a later ML step.

Usage:
    python align_and_baseline.py \\
        --csi-csv output/csi_capture_20260728_101500.csv \\
        --gt-csv output/ground_truth.csv \\
        --out-dir results/baseline_ml
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe, same convention as plotting.py
import matplotlib.pyplot as plt

from filters import butterworth_bandpass

logger = logging.getLogger("csi_analysis.align_and_baseline")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csi_csv(path: str, subcarrier: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a CSI capture CSV written by live_monitor.py (CSICapture) or
    utils.save_csv(). Returns (ts, amp_signal) where amp_signal is either
    the mean amplitude across all subcarriers, or a single subcarrier's
    amplitude if `subcarrier` is given (index into the CSV's own subcarrier
    columns, i.e. raw decode order, not fftshifted).

    Supports CSVs written in "amplitude" mode (sc{i}_amp columns) or
    "complex" mode (sc{i}_re / sc{i}_im columns, magnitude computed here).
    "phase"-mode CSVs aren't supported for this heart-rate check.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows in {path}")

    ts = np.array([float(r[0]) for r in rows], dtype=np.float64)

    amp_idx = [i for i, h in enumerate(header) if h.endswith("_amp")]
    re_idx = [i for i, h in enumerate(header) if h.endswith("_re")]
    im_idx = [i for i, h in enumerate(header) if h.endswith("_im")]

    if amp_idx:
        amp = np.array([[float(r[i]) for i in amp_idx] for r in rows], dtype=np.float64)
    elif re_idx and im_idx:
        re = np.array([[float(r[i]) for i in re_idx] for r in rows], dtype=np.float64)
        im = np.array([[float(r[i]) for i in im_idx] for r in rows], dtype=np.float64)
        amp = np.sqrt(re ** 2 + im ** 2)
    else:
        raise ValueError(
            f"Couldn't find sc*_amp or sc*_re/sc*_im columns in {path} -- "
            f"is this a 'phase'-mode capture? Header: {header[:8]}..."
        )

    if subcarrier is not None:
        if not (0 <= subcarrier < amp.shape[1]):
            raise ValueError(f"--subcarrier {subcarrier} out of range (0..{amp.shape[1] - 1})")
        signal = amp[:, subcarrier]
    else:
        signal = np.mean(amp, axis=1)

    return ts, signal


def load_ground_truth_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load heartbeat-giop.py's ground_truth.csv: timestamp, ibi_s, instant_bpm, avg_bpm."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows in {path}")

    ts = np.array([float(r[0]) for r in rows], dtype=np.float64)
    instant_bpm = np.array([float(r[2]) for r in rows], dtype=np.float64)
    avg_bpm = np.array([float(r[3]) for r in rows], dtype=np.float64)
    return ts, instant_bpm, avg_bpm


# ---------------------------------------------------------------------------
# Resampling / windowing
# ---------------------------------------------------------------------------

def resample_uniform(ts: np.ndarray, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate an irregularly-timed series onto a uniform grid at fs Hz."""
    t0, t1 = ts[0], ts[-1]
    n = int((t1 - t0) * fs) + 1
    t_uniform = t0 + np.arange(n) / fs
    x_uniform = np.interp(t_uniform, ts, x)
    return t_uniform, x_uniform


def windowed_fft_bpm(t_uniform: np.ndarray, x_filtered: np.ndarray, fs: float,
                      window_s: float, overlap: float,
                      low_hz: float, high_hz: float) -> List[dict]:
    """
    Slide a Hanning-windowed FFT over x_filtered and, per window, return the
    dominant frequency (converted to BPM) within [low_hz, high_hz].
    """
    win_samples = int(round(window_s * fs))
    if win_samples < 8:
        raise ValueError(
            f"--window-s {window_s} is too short at fs={fs:.2f} Hz "
            f"(only {win_samples} samples/window); use a longer window."
        )
    step = max(1, int(round(win_samples * (1.0 - overlap))))

    results = []
    for start in range(0, len(x_filtered) - win_samples + 1, step):
        seg = x_filtered[start:start + win_samples]
        seg = seg - np.mean(seg)
        seg = seg * np.hanning(len(seg))

        spec = np.fft.rfft(seg)
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
        mag = np.abs(spec)

        band_mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(band_mask):
            continue

        band_freqs = freqs[band_mask]
        band_mag = mag[band_mask]
        dom_idx = int(np.argmax(band_mag))

        t_start = t_uniform[start]
        t_end = t_uniform[start + win_samples - 1]
        results.append({
            "t_start": t_start,
            "t_end": t_end,
            "t_center": (t_start + t_end) / 2.0,
            "est_bpm": float(band_freqs[dom_idx] * 60.0),
            "est_magnitude": float(band_mag[dom_idx]),
        })

    return results


def attach_ground_truth(windows: List[dict], gt_ts: np.ndarray,
                         gt_instant_bpm: np.ndarray, gt_avg_bpm: np.ndarray) -> None:
    """
    For each window, average the ground-truth instant_bpm readings whose
    timestamp falls inside [t_start, t_end]. If none fall inside (ground
    truth sampled less often than the window length), fall back to the
    nearest ground-truth avg_bpm sample to the window center.
    """
    for w in windows:
        mask = (gt_ts >= w["t_start"]) & (gt_ts <= w["t_end"])
        if np.any(mask):
            w["true_bpm"] = float(np.mean(gt_instant_bpm[mask]))
            w["n_gt_beats"] = int(np.sum(mask))
        elif len(gt_ts) > 0:
            nearest = int(np.argmin(np.abs(gt_ts - w["t_center"])))
            w["true_bpm"] = float(gt_avg_bpm[nearest])
            w["n_gt_beats"] = 0
        else:
            w["true_bpm"] = float("nan")
            w["n_gt_beats"] = 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_results_csv(windows: List[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = ["t_start", "t_end", "t_center", "est_bpm", "est_magnitude", "true_bpm", "n_gt_beats"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for w in windows:
            writer.writerow({k: w[k] for k in fields})
    logger.info("Saved windowed results: %s", out_path)


def plot_comparison(windows: List[dict], out_path: str):
    t0 = windows[0]["t_start"]
    t_center = np.array([w["t_center"] - t0 for w in windows])
    est_bpm = np.array([w["est_bpm"] for w in windows])
    true_bpm = np.array([w["true_bpm"] for w in windows])
    has_gt = np.array([w["n_gt_beats"] > 0 for w in windows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(t_center, est_bpm, label="CSI-estimated BPM", color="tab:blue")
    ax1.plot(t_center, true_bpm, label="Ground-truth BPM", color="tab:red", alpha=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("BPM")
    ax1.set_title("Estimated vs. ground-truth BPM over time")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    valid = ~np.isnan(true_bpm)
    ax2.scatter(true_bpm[valid & has_gt], est_bpm[valid & has_gt],
                s=14, alpha=0.7, label="window w/ ground-truth beat(s)")
    ax2.scatter(true_bpm[valid & ~has_gt], est_bpm[valid & ~has_gt],
                s=14, alpha=0.4, color="gray", label="window w/ nearest-match only")
    if np.any(valid):
        lo = min(true_bpm[valid].min(), est_bpm[valid].min())
        hi = max(true_bpm[valid].max(), est_bpm[valid].max())
        ax2.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x (perfect)")
    ax2.set_xlabel("Ground-truth BPM")
    ax2.set_ylabel("CSI-estimated BPM")
    ax2.set_title("Parity plot")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=7)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", out_path)


def print_error_metrics(windows: List[dict]):
    est = np.array([w["est_bpm"] for w in windows])
    true = np.array([w["true_bpm"] for w in windows])
    has_gt = np.array([w["n_gt_beats"] > 0 for w in windows])

    valid = ~np.isnan(true)
    n_valid = int(np.sum(valid))
    n_direct = int(np.sum(valid & has_gt))

    print("\n=== Baseline results ===")
    print(f"windows total: {len(windows)}")
    print(f"windows with ground truth: {n_valid} ({n_direct} with a beat inside the window, "
          f"{n_valid - n_direct} matched to nearest sample)")

    if n_valid == 0:
        print("No overlapping ground truth -- can't compute error metrics. "
              "Check that both CSVs cover the same time range.")
        return

    err = est[valid] - true[valid]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(est[valid], true[valid])[0, 1]) if n_valid > 1 else float("nan")

    print(f"MAE:  {mae:.2f} BPM")
    print(f"RMSE: {rmse:.2f} BPM")
    print(f"Bias (est - true, mean): {bias:+.2f} BPM")
    print(f"Correlation (est vs true): {corr:.3f}")
    print(
        "\nRule of thumb: MAE/RMSE well under ~5-10 BPM with a correlation trending toward\n"
        "1.0 suggests the raw FFT approach is already picking up a real cardiac signal --\n"
        "worth moving on to an ML model for refinement. Errors comparable to the BPM range\n"
        "itself, or near-zero correlation, mean the signal isn't there yet at this\n"
        "chip/bw/placement/window setting before any model will help."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Align CSI + ground-truth heart-rate CSVs and FFT-estimate BPM per window "
                    "as a baseline check before building an ML model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csi-csv", required=True, help="Path to a csi_capture_*.csv from live_monitor.py")
    p.add_argument("--gt-csv", default="output/ground_truth.csv",
                   help="Path to heartbeat-giop.py's ground_truth.csv")
    p.add_argument("--out-dir", default="results/baseline_ml", help="Directory for outputs")
    p.add_argument("--subcarrier", type=int, default=None,
                   help="Use a single subcarrier's amplitude (index into the CSV's own "
                        "columns) instead of the mean across all subcarriers")

    p.add_argument("--window-s", type=float, default=8.0, help="FFT window length (seconds)")
    p.add_argument("--overlap", type=float, default=0.5, help="Fractional overlap between windows (0-1)")

    p.add_argument("--low-hz", type=float, default=0.5,
                   help="Low edge of the heart-rate band-pass (Hz); default matches 30 BPM")
    p.add_argument("--high-hz", type=float, default=3.7,
                   help="High edge of the heart-rate band-pass (Hz); default matches ~220 BPM")

    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.path.exists(args.csi_csv):
        logger.error("CSI CSV not found: %s", args.csi_csv)
        return 1
    if not os.path.exists(args.gt_csv):
        logger.error("Ground-truth CSV not found: %s", args.gt_csv)
        return 1

    logger.info("Loading CSI CSV: %s", args.csi_csv)
    csi_ts, csi_signal = load_csi_csv(args.csi_csv, subcarrier=args.subcarrier)

    logger.info("Loading ground-truth CSV: %s", args.gt_csv)
    gt_ts, gt_instant_bpm, gt_avg_bpm = load_ground_truth_csv(args.gt_csv)

    # --- overlap sanity check ---
    overlap_start = max(csi_ts[0], gt_ts[0])
    overlap_end = min(csi_ts[-1], gt_ts[-1])
    if overlap_end <= overlap_start:
        logger.error(
            "CSI capture (%.1f - %.1f) and ground truth (%.1f - %.1f) don't overlap in time -- "
            "make sure both were recorded during the same session.",
            csi_ts[0], csi_ts[-1], gt_ts[0], gt_ts[-1],
        )
        return 1
    logger.info("Time overlap: %.1fs", overlap_end - overlap_start)

    # --- resample CSI to a uniform grid ---
    fs_est = 1.0 / np.median(np.diff(csi_ts))
    logger.info("Estimated CSI sample rate: %.2f Hz", fs_est)
    t_uniform, x_uniform = resample_uniform(csi_ts, csi_signal, fs_est)

    # --- band-pass to the heart-rate band ---
    nyq = fs_est / 2.0
    if args.high_hz >= nyq:
        logger.warning(
            "--high-hz (%.2f) >= Nyquist (%.2f) at this sample rate -- clamping to %.2f",
            args.high_hz, nyq, nyq * 0.95,
        )
        args.high_hz = nyq * 0.95
    logger.info("Band-pass filtering to %.2f-%.2f Hz (%.0f-%.0f BPM)...",
                args.low_hz, args.high_hz, args.low_hz * 60, args.high_hz * 60)
    x_filtered = butterworth_bandpass(x_uniform, args.low_hz, args.high_hz, fs_est)

    # --- windowed FFT -> BPM estimate ---
    windows = windowed_fft_bpm(
        t_uniform, x_filtered, fs_est,
        window_s=args.window_s, overlap=args.overlap,
        low_hz=args.low_hz, high_hz=args.high_hz,
    )
    if not windows:
        logger.error("No windows produced -- capture may be too short for --window-s %.1f", args.window_s)
        return 1
    logger.info("Computed %d windows", len(windows))

    # --- attach ground truth + report ---
    attach_ground_truth(windows, gt_ts, gt_instant_bpm, gt_avg_bpm)
    print_error_metrics(windows)

    results_path = os.path.join(args.out_dir, "baseline_results.csv")
    plot_path = os.path.join(args.out_dir, "baseline_bpm_comparison.png")
    save_results_csv(windows, results_path)
    plot_comparison(windows, plot_path)

    logger.info("Done. Outputs in: %s", os.path.abspath(args.out_dir))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
