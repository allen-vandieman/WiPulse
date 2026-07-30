#!/usr/bin/env python3
"""
align_and_baseline.py
======================
Baseline check: does the CSI signal actually contain a usable heart-rate
signature, before spending any effort on an ML model?

Takes the two CSVs produced by this project --

    output/csi_capture_<timestamp>.csv     (from live_monitor.py, 'c' to record)
    output/ground_truth.csv                (from heartbeat-giop.py)

-- and:

  1. loads + aligns them on their shared time.time() timestamp column
  2. resamples the (possibly irregularly-timed) CSI signal onto a
     uniform time grid at its estimated sample rate
  3. band-pass filters that signal to the human heart-rate band
     (default 0.5-3.7 Hz == 30-220 BPM, matching heartbeat-giop.py's
     MIN_BPM/MAX_BPM) to knock out breathing and slow motion artifacts
  4. slides a window (default 8s, 50% overlap) over the filtered signal,
     FFTs each window (Hanning-windowed, like analysis.fft_motion_detection),
     and takes the dominant in-band frequency -- refined with parabolic
     (3-point) interpolation around the peak bin, so the BPM estimate
     isn't quantized to the raw FFT bin spacing (60/window_s BPM steps,
     e.g. 7.5 BPM at the default 8s window) -- as the CSI-estimated BPM
  5. matches each window against the ground-truth beats that fall inside
     it (or the nearest beat, if none fall inside) and reports MAE/RMSE,
     plus a plot of estimated vs. true BPM over time and a scatter/parity
     comparison

Two signal types are supported via --signal-type:
  - "amplitude" (default): mean/selected subcarrier amplitude, as before.
  - "phase": per-packet phase, sanitized to remove the linear phase ramp
    across subcarriers caused by SFO/CFO (a per-packet least-squares fit
    vs. subcarrier index, subtracted out), then treated as the signal.
    Requires a --capture-mode complex CSV from live_monitor.py (needs
    sc*_re/sc*_im columns -- raw sc*_pha columns alone aren't usable here
    since they're neither unwrapped nor sanitized). Heartbeat-scale chest
    displacement is sub-millimeter, which shows up far more reliably in
    phase than in amplitude, so this is worth trying if the amplitude
    path doesn't correlate with ground truth.

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

    # phase-based path (needs a --capture-mode complex capture)
    python align_and_baseline.py \\
        --csi-csv output/csi_capture_20260728_101500.csv \\
        --gt-csv output/ground_truth.csv \\
        --signal-type phase \\
        --out-dir results/baseline_ml_phase
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe, same convention as plotting.py
import matplotlib.pyplot as plt

from filters import butterworth_bandpass

logger = logging.getLogger("csi_analysis.align_and_baseline")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _find_header_index(header: Sequence[str], candidates: Sequence[str]) -> int:
    normalized = [h.strip().lower() for h in header]
    for candidate in candidates:
        for idx, name in enumerate(normalized):
            if name == candidate.strip().lower():
                return idx
    return -1


def _load_numeric_column(rows: Sequence[Sequence[str]], idx: int) -> np.ndarray:
    values = []
    for row in rows:
        if idx >= len(row):
            values.append(np.nan)
            continue
        try:
            values.append(float(row[idx]))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.array(values, dtype=np.float64)


def load_csi_csv(path: str, subcarrier: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a CSI capture CSV written by live_monitor.py (CSICapture) or
    utils.save_csv(). Returns (ts, amp_signal) where amp_signal is either
    the mean amplitude across all subcarriers, or a single subcarrier's
    amplitude if `subcarrier` is given.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows in {path}")

    ts_idx = _find_header_index(header, ("ts", "timestamp", "time", "time_s", "sample_time", "t"))
    ts = _load_numeric_column(rows, ts_idx if ts_idx >= 0 else 0)

    amp_idx = [i for i, h in enumerate(header) if h.strip().lower().endswith("_amp")]
    re_idx = [i for i, h in enumerate(header) if h.strip().lower().endswith("_re")]
    im_idx = [i for i, h in enumerate(header) if h.strip().lower().endswith("_im")]

    if amp_idx:
        amp = np.column_stack([_load_numeric_column(rows, i) for i in amp_idx])
    elif re_idx and im_idx:
        re = np.column_stack([_load_numeric_column(rows, i) for i in re_idx])
        im = np.column_stack([_load_numeric_column(rows, i) for i in im_idx])
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


def load_csi_amp_matrix(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all CSI subcarrier amplitudes as a 2D matrix with shape (time, subcarriers)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows in {path}")

    ts_idx = _find_header_index(header, ("ts", "timestamp", "time", "time_s", "sample_time", "t"))
    ts = _load_numeric_column(rows, ts_idx if ts_idx >= 0 else 0)

    amp_idx = [i for i, h in enumerate(header) if h.strip().lower().endswith("_amp")]
    re_idx = [i for i, h in enumerate(header) if h.strip().lower().endswith("_re")]
    im_idx = [i for i, h in enumerate(header) if h.strip().lower().endswith("_im")]

    if amp_idx:
        amp = np.column_stack([_load_numeric_column(rows, i) for i in amp_idx])
    elif re_idx and im_idx:
        re = np.column_stack([_load_numeric_column(rows, i) for i in re_idx])
        im = np.column_stack([_load_numeric_column(rows, i) for i in im_idx])
        amp = np.sqrt(re ** 2 + im ** 2)
    else:
        raise ValueError(
            f"Couldn't find sc*_amp or sc*_re/sc*_im columns in {path} -- "
            f"is this a 'phase'-mode capture? Header: {header[:8]}..."
        )

    return ts, amp


def load_csi_complex_matrix(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the full complex CSI matrix (time, subcarriers) from a
    --capture-mode complex CSV (sc*_re / sc*_im columns). Needed for the
    phase-sanitization path, since amplitude-only or raw-phase-only CSVs
    don't carry enough information to remove the SFO/CFO phase ramp.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows in {path}")

    ts_idx = _find_header_index(header, ("ts", "timestamp", "time", "time_s", "sample_time", "t"))
    ts = _load_numeric_column(rows, ts_idx if ts_idx >= 0 else 0)

    re_idx = [i for i, h in enumerate(header) if h.strip().lower().endswith("_re")]
    im_idx = [i for i, h in enumerate(header) if h.strip().lower().endswith("_im")]

    if not (re_idx and im_idx):
        raise ValueError(
            f"Couldn't find sc*_re/sc*_im columns in {path} -- phase-based analysis needs "
            f"a capture recorded with live_monitor.py's --capture-mode complex (full I/Q). "
            f"Header: {header[:8]}..."
        )

    re = np.column_stack([_load_numeric_column(rows, i) for i in re_idx])
    im = np.column_stack([_load_numeric_column(rows, i) for i in im_idx])
    csi = re + 1j * im
    return ts, csi


def sanitize_phase(csi: np.ndarray) -> np.ndarray:
    """
    Per-packet phase sanitization: unwrap phase across the subcarrier axis,
    then remove the linear trend (least-squares fit vs. subcarrier index)
    that SFO/CFO impose on every packet. What's left is (approximately) the
    true channel phase, which is where sub-millimeter chest-wall motion
    from a heartbeat actually shows up -- amplitude is dominated by larger
    motion (breathing, body sway) and mostly buries it.

    csi: (n_packets, nfft) complex matrix.
    Returns: (n_packets, nfft) sanitized phase, radians.
    """
    csi = np.asarray(csi)
    n_packets, nfft = csi.shape
    unwrapped = np.unwrap(np.angle(csi), axis=1)

    x = np.arange(nfft, dtype=np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = np.sum(x_centered ** 2)
    if denom == 0:
        return unwrapped - unwrapped.mean(axis=1, keepdims=True)

    y_mean = unwrapped.mean(axis=1, keepdims=True)
    y_centered = unwrapped - y_mean
    slope = (y_centered @ x_centered) / denom          # shape (n_packets,)
    intercept = y_mean[:, 0] - slope * x_mean

    fitted = slope[:, None] * x[None, :] + intercept[:, None]
    return unwrapped - fitted


def load_ground_truth_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load heartbeat-giop.py's ground_truth.csv: timestamp, ibi_s, instant_bpm, avg_bpm."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows in {path}")

    ts_idx = _find_header_index(header, ("ts", "timestamp", "time", "time_s", "sample_time", "t"))
    instant_idx = _find_header_index(header, ("instant_bpm", "instant", "bpm", "heart_rate_bpm"))
    avg_idx = _find_header_index(header, ("avg_bpm", "avg", "average_bpm", "rolling_avg_bpm"))

    ts = _load_numeric_column(rows, ts_idx if ts_idx >= 0 else 0)
    instant_bpm = _load_numeric_column(rows, instant_idx if instant_idx >= 0 else 2)
    avg_bpm = _load_numeric_column(rows, avg_idx if avg_idx >= 0 else 3)
    return ts, instant_bpm, avg_bpm


# ---------------------------------------------------------------------------
# Resampling / windowing
# ---------------------------------------------------------------------------

def resample_uniform(ts: np.ndarray, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate an irregularly-timed series onto a uniform grid at fs Hz."""
    ts = np.asarray(ts, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    valid = np.isfinite(ts) & np.isfinite(x)
    ts = ts[valid]
    x = x[valid]
    if ts.size == 0:
        raise ValueError("No finite timestamp/value pairs available for resampling")

    order = np.argsort(ts)
    ts = ts[order]
    x = x[order]

    if ts.size > 1:
        uniq_ts, inverse = np.unique(ts, return_inverse=True)
        if uniq_ts.size != ts.size:
            summed = np.zeros(uniq_ts.size, dtype=np.float64)
            counts = np.zeros(uniq_ts.size, dtype=np.int64)
            for idx, group in enumerate(inverse):
                summed[group] += x[idx]
                counts[group] += 1
            x = summed / counts
            ts = uniq_ts

    if ts.size < 2:
        t_uniform = np.array([ts[0]], dtype=np.float64)
        x_uniform = np.array([x[0]], dtype=np.float64)
        return t_uniform, x_uniform

    t0, t1 = ts[0], ts[-1]
    if t1 <= t0:
        raise ValueError("Resampling requires at least two distinct timestamps")

    n = int((t1 - t0) * fs) + 1
    n = max(1, n)
    t_uniform = t0 + np.arange(n) / fs
    x_uniform = np.interp(t_uniform, ts, x)
    return t_uniform, x_uniform


def windowed_fft_bpm(t_uniform: np.ndarray, x_filtered: np.ndarray, fs: float,
                      window_s: float, overlap: float,
                      low_hz: float, high_hz: float,
                      interpolate: bool = True) -> List[dict]:
    """
    Slide a Hanning-windowed FFT over x_filtered and, per window, return the
    dominant frequency (converted to BPM) within [low_hz, high_hz].

    interpolate: if True (default), refine the dominant bin with 3-point
        parabolic interpolation over the log-magnitude spectrum around the
        peak. A raw FFT bin pick is quantized to fs/window_s Hz steps (e.g.
        7.5 BPM at the default 100 Hz / 8s window) regardless of how clean
        the signal is; interpolation gives a sub-bin estimate without
        needing a longer window (which would cost temporal resolution) or
        zero-padding (more compute for the same info). Falls back to the
        raw bin at either edge of the spectrum, where there's no neighbor
        on one side to interpolate against.
    """
    win_samples = int(round(window_s * fs))
    if win_samples < 8:
        raise ValueError(
            f"--window-s {window_s} is too short at fs={fs:.2f} Hz "
            f"(only {win_samples} samples/window); use a longer window."
        )
    step = max(1, int(round(win_samples * (1.0 - overlap))))
    freq_res = fs / win_samples

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

        band_indices = np.where(band_mask)[0]
        band_mag = mag[band_indices]
        dom_local = int(np.argmax(band_mag))
        dom_idx = int(band_indices[dom_local])

        est_freq = freqs[dom_idx]
        est_mag = mag[dom_idx]

        if interpolate and 0 < dom_idx < len(mag) - 1:
            alpha, beta, gamma = mag[dom_idx - 1], mag[dom_idx], mag[dom_idx + 1]
            denom = alpha - 2.0 * beta + gamma
            if denom != 0:
                p = 0.5 * (alpha - gamma) / denom
                p = float(np.clip(p, -1.0, 1.0))  # guard against a pathological ratio
                est_freq = freqs[dom_idx] + p * freq_res
                # parabolic estimate of the interpolated peak magnitude
                est_mag = beta - 0.25 * (alpha - gamma) * p
            # clamp back into the requested band -- interpolation can, at a
            # band edge, nudge the estimate slightly outside [low_hz, high_hz]
            est_freq = float(np.clip(est_freq, low_hz, high_hz))

        t_start = t_uniform[start]
        t_end = t_uniform[start + win_samples - 1]
        results.append({
            "t_start": t_start,
            "t_end": t_end,
            "t_center": (t_start + t_end) / 2.0,
            "est_bpm": float(est_freq * 60.0),
            "est_magnitude": float(est_mag),
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


def summarize_metrics(windows: List[dict]) -> Dict[str, float]:
    est = np.array([w["est_bpm"] for w in windows])
    true = np.array([w["true_bpm"] for w in windows])
    has_gt = np.array([w["n_gt_beats"] > 0 for w in windows])

    valid = ~np.isnan(true)
    n_valid = int(np.sum(valid))
    n_direct = int(np.sum(valid & has_gt))

    if n_valid == 0:
        return {
            "windows_total": float(len(windows)),
            "windows_with_ground_truth": 0.0,
            "n_direct": float(n_direct),
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "corr": float("nan"),
            "score": float("nan"),
        }

    err = est[valid] - true[valid]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    if n_valid > 1 and np.std(est[valid]) > 0 and np.std(true[valid]) > 0:
        corr = float(np.corrcoef(est[valid], true[valid])[0, 1])
    else:
        corr = float("nan")
    score = float(corr / (1.0 + mae)) if np.isfinite(corr) and np.isfinite(mae) else float("nan")

    return {
        "windows_total": float(len(windows)),
        "windows_with_ground_truth": float(n_valid),
        "n_direct": float(n_direct),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "corr": corr,
        "score": score,
    }


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


def save_metrics_csv(rows: List[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = ["rank", "subcarrier", "corr", "mae", "rmse", "bias", "score"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    logger.info("Saved subcarrier metrics: %s", out_path)


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


def plot_metric_vs_subcarrier(values: np.ndarray, out_path: str, metric: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(values)), values, marker="o", linestyle="-", color="tab:green")
    ax.set_xlabel("Subcarrier")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs. subcarrier")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved metric plot: %s", out_path)


def print_error_metrics(windows: List[dict], label: str = "Baseline results"):
    metrics = summarize_metrics(windows)
    print(f"\n=== {label} ===")
    print(f"windows total: {int(metrics['windows_total'])}")
    print(f"windows with ground truth: {int(metrics['windows_with_ground_truth'])} "
          f"({int(metrics['n_direct'])} with a beat inside the window)")

    if not np.isfinite(metrics["mae"]):
        print("No overlapping ground truth -- can't compute error metrics. "
              "Check that both CSVs cover the same time range.")
        return metrics

    print(f"MAE:  {metrics['mae']:.2f} BPM")
    print(f"RMSE: {metrics['rmse']:.2f} BPM")
    print(f"Bias (est - true, mean): {metrics['bias']:+.2f} BPM")
    print(f"Correlation (est vs true): {metrics['corr']:.3f}")
    print(f"Score: {metrics['score']:.3f}")
    print(
        "\nRule of thumb: MAE/RMSE well under ~5-10 BPM with a correlation trending toward\n"
        "1.0 suggests the raw FFT approach is already picking up a real cardiac signal --\n"
        "worth moving on to an ML model for refinement. Errors comparable to the BPM range\n"
        "itself, or near-zero correlation, mean the signal isn't there yet at this\n"
        "chip/bw/placement/window setting before any model will help."
    )
    return metrics


def evaluate_signal(csi_ts: np.ndarray, signal: np.ndarray, gt_ts: np.ndarray,
                     gt_instant_bpm: np.ndarray, gt_avg_bpm: np.ndarray,
                     args, fs_est: float, low_hz: float, high_hz: float) -> Tuple[List[dict], Dict[str, float]]:
    t_uniform, x_uniform = resample_uniform(csi_ts, signal, fs_est)
    x_filtered = butterworth_bandpass(x_uniform, low_hz, high_hz, fs_est)
    windows = windowed_fft_bpm(
        t_uniform,
        x_filtered,
        fs_est,
        window_s=args.window_s,
        overlap=args.overlap,
        low_hz=low_hz,
        high_hz=high_hz,
        interpolate=not args.no_fft_interp,
    )
    if not windows:
        raise ValueError("No windows produced -- capture may be too short for the current window length")
    attach_ground_truth(windows, gt_ts, gt_instant_bpm, gt_avg_bpm)
    metrics = summarize_metrics(windows)
    return windows, metrics


def write_summary(out_dir: str, subcarrier_rows: List[dict], topn_results: List[dict], recommendation: Dict[str, object]):
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("====================================================\n")
        f.write("Subcarrier Search Complete\n")
        f.write("====================================================\n\n")
        f.write(f"{len(subcarrier_rows)} subcarriers tested\n\n")
        f.write("Top 10\n\n")
        f.write("Rank   SC   Corr   MAE   RMSE   Score\n")
        f.write("-----------------------------------------\n")
        for row in subcarrier_rows[:10]:
            f.write(
                f"{row['rank']:<6} {row['subcarrier']:<5} {row['corr']:.3f}   {row['mae']:.2f}   "
                f"{row['rmse']:.2f}   {row['score']:.3f}\n"
            )
        f.write("\nBest averaged signals\n\n")
        for row in topn_results:
            f.write(
                f"Top {row['n']} : Corr={row['corr']:.3f} MAE={row['mae']:.2f} RMSE={row['rmse']:.2f}\n"
            )
        f.write("\nRecommendation\n--------------\n\n")
        f.write(f"Use the average of the top {recommendation['n']} subcarriers.\n\n")
        f.write("Subcarriers:\n")
        f.write(
            ", ".join(str(sc) for sc in recommendation["subcarriers"]) + "\n\n"
        )
        f.write("This configuration achieved:\n\n")
        f.write(f"Correlation : {recommendation['corr']:.3f}\n")
        f.write(f"MAE : {recommendation['mae']:.2f} BPM\n")
        f.write(f"RMSE : {recommendation['rmse']:.2f} BPM\n")
    logger.info("Saved summary: %s", summary_path)


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
                   help="Use a single subcarrier's signal (index into the CSV's own "
                        "columns) instead of the mean across all subcarriers")
    p.add_argument("--search-subcarriers", action="store_true",
                   help="Evaluate every subcarrier and the average of the top-ranked subcarrier groups")

    p.add_argument("--signal-type", choices=["amplitude", "phase"], default="amplitude",
                   help="'amplitude' (default) uses sc*_amp or sc*_re/sc*_im magnitude. "
                        "'phase' uses sanitized phase (linear SFO/CFO ramp removed per packet) "
                        "and requires a --capture-mode complex CSV from live_monitor.py.")

    p.add_argument("--window-s", type=float, default=8.0, help="FFT window length (seconds)")
    p.add_argument("--overlap", type=float, default=0.5, help="Fractional overlap between windows (0-1)")
    p.add_argument("--no-fft-interp", action="store_true",
                   help="Disable parabolic (3-point) interpolation of the FFT peak bin and use "
                        "the raw bin frequency instead. Interpolation is on by default since the "
                        "raw bin pick quantizes BPM estimates to fs/window_s Hz steps "
                        "(7.5 BPM at the default 100 Hz / 8s window).")

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
    if args.signal_type == "phase":
        csi_ts, csi_complex = load_csi_complex_matrix(args.csi_csv)
        logger.info("Sanitizing phase (removing per-packet linear SFO/CFO ramp across "
                     "%d subcarriers)...", csi_complex.shape[1])
        signal_matrix = sanitize_phase(csi_complex)
    else:
        csi_ts, signal_matrix = load_csi_amp_matrix(args.csi_csv)

    logger.info("Loading ground-truth CSV: %s", args.gt_csv)
    gt_ts, gt_instant_bpm, gt_avg_bpm = load_ground_truth_csv(args.gt_csv)

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

    fs_est = 1.0 / np.median(np.diff(csi_ts))
    logger.info("Estimated CSI sample rate: %.2f Hz", fs_est)

    nyq = fs_est / 2.0
    if args.high_hz >= nyq:
        logger.warning(
            "--high-hz (%.2f) >= Nyquist (%.2f) at this sample rate -- clamping to %.2f",
            args.high_hz, nyq, nyq * 0.95,
        )
        args.high_hz = nyq * 0.95
    logger.info("Signal type: %s   FFT peak interpolation: %s",
                args.signal_type, "off" if args.no_fft_interp else "on")
    logger.info("Band-pass filtering to %.2f-%.2f Hz (%.0f-%.0f BPM)...",
                args.low_hz, args.high_hz, args.low_hz * 60, args.high_hz * 60)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.search_subcarriers:
        subcarrier_rows = []
        for subcarrier_idx in range(signal_matrix.shape[1]):
            signal = signal_matrix[:, subcarrier_idx]
            windows, metrics = evaluate_signal(
                csi_ts,
                signal,
                gt_ts,
                gt_instant_bpm,
                gt_avg_bpm,
                args,
                fs_est,
                args.low_hz,
                args.high_hz,
            )
            subcarrier_rows.append({
                "subcarrier": subcarrier_idx,
                "corr": metrics["corr"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "score": metrics["score"],
            })

        ranked_rows = sorted(subcarrier_rows, key=lambda row: row["score"] if np.isfinite(row["score"]) else -np.inf, reverse=True)
        for idx, row in enumerate(ranked_rows, start=1):
            row["rank"] = idx
        save_metrics_csv(ranked_rows, os.path.join(args.out_dir, "subcarrier_metrics.csv"))

        top10 = [row for row in ranked_rows[:10]]
        top10_path = os.path.join(args.out_dir, "top10_subcarriers.csv")
        save_metrics_csv(top10, top10_path)

        corr_vals = np.array([row["corr"] for row in ranked_rows], dtype=np.float64)
        mae_vals = np.array([row["mae"] for row in ranked_rows], dtype=np.float64)
        plot_metric_vs_subcarrier(corr_vals, os.path.join(args.out_dir, "correlation_vs_subcarrier.png"), "Correlation")
        plot_metric_vs_subcarrier(mae_vals, os.path.join(args.out_dir, "mae_vs_subcarrier.png"), "MAE")

        topn_results = []
        for n in [1, 3, 5, 10, 20]:
            if n > len(ranked_rows):
                continue
            selected = [row["subcarrier"] for row in ranked_rows[:n]]
            signal = np.mean(signal_matrix[:, selected], axis=1)
            windows, metrics = evaluate_signal(
                csi_ts,
                signal,
                gt_ts,
                gt_instant_bpm,
                gt_avg_bpm,
                args,
                fs_est,
                args.low_hz,
                args.high_hz,
            )
            topn_results.append({
                "n": n,
                "subcarriers": selected,
                "corr": metrics["corr"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "score": metrics["score"],
                "windows": windows,
            })

        if topn_results:
            for row in topn_results:
                plot_comparison(row["windows"], os.path.join(args.out_dir, f"best{row['n']}_comparison.png"))

        primary_windows, primary_metrics = evaluate_signal(
            csi_ts,
            signal_matrix[:, ranked_rows[0]["subcarrier"]],
            gt_ts,
            gt_instant_bpm,
            gt_avg_bpm,
            args,
            fs_est,
            args.low_hz,
            args.high_hz,
        )
        save_results_csv(primary_windows, os.path.join(args.out_dir, "baseline_results.csv"))
        plot_comparison(primary_windows, os.path.join(args.out_dir, "baseline_bpm_comparison.png"))
        print_error_metrics(primary_windows, label=f"Best single subcarrier ({args.signal_type})")

        best_avg = max(topn_results, key=lambda row: row["score"] if np.isfinite(row["score"]) else -np.inf)
        recommendation = {
            "n": best_avg["n"],
            "subcarriers": best_avg["subcarriers"],
            "corr": best_avg["corr"],
            "mae": best_avg["mae"],
            "rmse": best_avg["rmse"],
        }
        print("\n====================================================")
        print("Subcarrier Search Complete")
        print("====================================================")
        print(f"{len(ranked_rows)} subcarriers tested\n")
        print("Top 10")
        print("Rank   SC   Corr   MAE   RMSE   Score")
        print("-----------------------------------------")
        for row in ranked_rows[:10]:
            print(f"{row['rank']:<6} {row['subcarrier']:<5} {row['corr']:.3f}   {row['mae']:.2f}   {row['rmse']:.2f}   {row['score']:.3f}")
        print("\nBest averaged signals")
        for row in topn_results:
            print(f"Top {row['n']} : Corr={row['corr']:.3f} MAE={row['mae']:.2f} RMSE={row['rmse']:.2f}")
        print("\nRecommendation")
        print("--------------")
        print(f"\nUse the average of the top {recommendation['n']} subcarriers.")
        print("\nSubcarriers:")
        print(", ".join(str(sc) for sc in recommendation["subcarriers"]))
        print("\nThis configuration achieved:")
        print(f"Correlation : {recommendation['corr']:.3f}")
        print(f"MAE : {recommendation['mae']:.2f} BPM")
        print(f"RMSE : {recommendation['rmse']:.2f} BPM")
        write_summary(args.out_dir, ranked_rows, topn_results, recommendation)
        logger.info("Done. Outputs in: %s", os.path.abspath(args.out_dir))
        return 0

    if args.subcarrier is not None:
        signal = signal_matrix[:, args.subcarrier]
    else:
        signal = np.mean(signal_matrix, axis=1)

    windows, metrics = evaluate_signal(
        csi_ts,
        signal,
        gt_ts,
        gt_instant_bpm,
        gt_avg_bpm,
        args,
        fs_est,
        args.low_hz,
        args.high_hz,
    )
    print_error_metrics(windows, label=f"Baseline results ({args.signal_type})")

    save_results_csv(windows, os.path.join(args.out_dir, "baseline_results.csv"))
    plot_comparison(windows, os.path.join(args.out_dir, "baseline_bpm_comparison.png"))

    logger.info("Done. Outputs in: %s", os.path.abspath(args.out_dir))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
