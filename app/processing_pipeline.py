from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from quality import evaluate_subcarriers


DEFAULT_HEART_RATE_LOW_HZ = 0.8
DEFAULT_HEART_RATE_HIGH_HZ = 2.17


def extract_amplitude_signal(csi: np.ndarray) -> np.ndarray:
    """Return the complex magnitude per packet/subcarrier."""
    csi = np.asarray(csi, dtype=np.complex128)
    if csi.ndim == 1:
        return np.abs(csi)
    return np.abs(csi)


def amplitude_from_complex(csi: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for per-packet complex CSI arrays."""
    return extract_amplitude_signal(csi)


def remove_dc_component(signal: np.ndarray, window_size: int = 101) -> np.ndarray:
    """Subtract a moving-average baseline to remove DC drift and center the result at zero."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 2:
        if x.shape[1] == 0:
            return x.copy()
        out = np.empty_like(x, dtype=np.float64)
        for i in range(x.shape[1]):
            out[:, i] = remove_dc_component(x[:, i], window_size=window_size)
        return out
    if x.size < 3:
        return x.copy()
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    padded = np.pad(x, (pad, pad), mode="edge")
    moving = np.convolve(padded, kernel, mode="valid")
    detrended = x - moving
    return detrended - np.mean(detrended)


def bandpass_heartbeat(signal: np.ndarray, fs_hz: float, low_hz: float = DEFAULT_HEART_RATE_LOW_HZ,
                       high_hz: float = DEFAULT_HEART_RATE_HIGH_HZ, order: int = 3) -> np.ndarray:
    """Apply a zero-phase third-order Butterworth band-pass to heart-rate frequencies."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 2:
        return np.column_stack([
            bandpass_heartbeat(x[:, i], fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz, order=order)
            for i in range(x.shape[1])
        ])
    nyq = fs_hz / 2.0
    if not (0 < low_hz < high_hz < nyq):
        raise ValueError(f"Require 0 < low_hz < high_hz < Nyquist (fs_hz={fs_hz}, nyq={nyq})")
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def savgol_smooth(signal: np.ndarray, window_length: int = 9, polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing while preserving signal shape and peak structure."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 2:
        return np.column_stack([
            savgol_smooth(x[:, i], window_length=window_length, polyorder=polyorder)
            for i in range(x.shape[1])
        ])
    if x.size < window_length:
        window_length = min(len(x) if len(x) % 2 == 1 else len(x) - 1, 7)
    if window_length < polyorder + 2:
        window_length = polyorder + 3
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= polyorder:
        return x.copy()
    return savgol_filter(x, window_length=window_length, polyorder=polyorder, mode="interp")


def segment_overlapping_windows(signal: np.ndarray, window_size: int, hop_size: int) -> List[np.ndarray]:
    """Split a signal into overlapping windows with a given hop size."""
    x = np.asarray(signal, dtype=np.float64)
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size and hop_size must be positive")
    if x.size < window_size:
        return [x.copy()]
    windows: List[np.ndarray] = []
    for start in range(0, x.size - window_size + 1, hop_size):
        windows.append(x[start:start + window_size])
    if windows and windows[-1].shape[0] != window_size:
        windows.append(x[-window_size:])
    return windows


def load_csi_csv_matrix(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a CSI CSV from live_monitor.py and return (ts, complex_matrix)."""
    path = Path(path)
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError(f"No rows found in {path}")

    ts_index = None
    for i, name in enumerate(header):
        if name.strip().lower() in {"ts", "timestamp", "time", "time_s", "sample_time", "t"}:
            ts_index = i
            break
    ts = np.array([float(row[ts_index]) if ts_index is not None and ts_index < len(row) else np.nan for row in rows], dtype=np.float64)

    re_cols = [i for i, name in enumerate(header) if name.strip().lower().endswith("_re")]
    im_cols = [i for i, name in enumerate(header) if name.strip().lower().endswith("_im")]
    if not re_cols or not im_cols:
        raise ValueError(f"CSV at {path} does not contain complex CSI columns (sc*_re/sc*_im)")
    if len(re_cols) != len(im_cols):
        raise ValueError(f"Unequal re/im column counts in {path}")

    csi = np.empty((len(rows), len(re_cols)), dtype=np.complex128)
    for r, row in enumerate(rows):
        for c, (re_idx, im_idx) in enumerate(zip(re_cols, im_cols)):
            re_val = float(row[re_idx]) if re_idx < len(row) else 0.0
            im_val = float(row[im_idx]) if im_idx < len(row) else 0.0
            csi[r, c] = re_val + 1j * im_val
    return ts, csi


def load_ground_truth_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ts, bpm) from a heartbeat-giop CSV."""
    path = Path(path)
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"No GT rows in {path}")
    ts_idx = next((i for i, name in enumerate(header) if name.strip().lower() == "ts"), 0)
    bpm_idx = next((i for i, name in enumerate(header) if name.strip().lower() == "avg_bpm"), 3)
    ts = np.array([float(row[ts_idx]) for row in rows], dtype=np.float64)
    bpm = np.array([float(row[bpm_idx]) for row in rows], dtype=np.float64)
    return ts, bpm


def estimate_heart_rate_from_complex(
    csi: np.ndarray,
    fs_hz: float,
    low_hz: float = DEFAULT_HEART_RATE_LOW_HZ,
    high_hz: float = DEFAULT_HEART_RATE_HIGH_HZ,
    window_size: int = 30,
    hop_size: int = 10,
) -> List[Tuple[float, float, float]]:
    """Convenience pipeline: amplitude -> remove DC -> bandpass -> Savitzky-Golay -> segment -> return (start, end, mean_bpm_estimate)."""
    amp = extract_amplitude_signal(csi)
    if amp.ndim == 2:
        mean_amp = np.mean(amp, axis=1)
    else:
        mean_amp = amp
    mean_amp = remove_dc_component(mean_amp)
    filtered = bandpass_heartbeat(mean_amp, fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz)
    smoothed = savgol_smooth(filtered, window_length=9, polyorder=3)
    segments = segment_overlapping_windows(smoothed, window_size=window_size, hop_size=hop_size)
    results: List[Tuple[float, float, float]] = []
    for idx, seg in enumerate(segments):
        if len(seg) < 8:
            continue
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs_hz)
        spectrum = np.abs(np.fft.rfft(seg))
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(mask):
            continue
        dominant_hz = freqs[mask][np.argmax(spectrum[mask])]
        bpm = dominant_hz * 60.0
        start = idx * hop_size
        end = start + len(seg)
        results.append((float(start) / fs_hz, float(end) / fs_hz, float(bpm)))
    return results


def stage_signal_for_pipeline(signal: np.ndarray, stage: str, fs_hz: float,
                             low_hz: float = DEFAULT_HEART_RATE_LOW_HZ,
                             high_hz: float = DEFAULT_HEART_RATE_HIGH_HZ) -> np.ndarray:
    """Return the specified stage of the pipeline: raw, dc_removed, bandpassed, smoothed."""
    if stage == "raw":
        return np.asarray(signal, dtype=np.float64)
    if stage == "dc_removed":
        return remove_dc_component(signal)
    if stage == "bandpassed":
        return bandpass_heartbeat(remove_dc_component(signal), fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz)
    if stage == "smoothed":
        stage_signal = bandpass_heartbeat(remove_dc_component(signal), fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz)
        return savgol_smooth(stage_signal, window_length=9, polyorder=3)
    raise ValueError(f"Unknown stage: {stage}")


def benchmark_pipeline_stages(
    csi: np.ndarray,
    gt_ts: Sequence[float],
    gt_bpm: Sequence[float],
    fs_hz: float,
    low_hz: float = DEFAULT_HEART_RATE_LOW_HZ,
    high_hz: float = DEFAULT_HEART_RATE_HIGH_HZ,
    window_size: int = 30,
    hop_size: int = 10,
    subcarrier_indices: Sequence[int] | None = None,
) -> dict:
    """Evaluate each processing stage against the same ground-truth reference."""
    amp = extract_amplitude_signal(csi)
    if amp.ndim == 1:
        signal_matrix = amp[:, None]
    else:
        if subcarrier_indices is None:
            signal_matrix = amp
        else:
            signal_matrix = amp[:, list(subcarrier_indices)]

    if signal_matrix.shape[1] == 0:
        raise ValueError("No subcarrier signal selected for benchmark")

    base = np.mean(signal_matrix, axis=1) if signal_matrix.shape[1] > 1 else signal_matrix[:, 0]

    report: dict[str, dict] = {}
    for stage in ("raw", "dc_removed", "bandpassed", "smoothed"):
        signal = stage_signal_for_pipeline(base, stage, fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz)
        segments = segment_overlapping_windows(signal, window_size=window_size, hop_size=hop_size)
        estimates = []
        for idx, seg in enumerate(segments):
            if len(seg) < 8:
                continue
            freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs_hz)
            spectrum = np.abs(np.fft.rfft(seg))
            mask = (freqs >= low_hz) & (freqs <= high_hz)
            if not np.any(mask):
                continue
            dominant_hz = freqs[mask][np.argmax(spectrum[mask])]
            estimates.append((idx * hop_size / fs_hz, (idx * hop_size + len(seg)) / fs_hz, dominant_hz * 60.0))
        report[stage] = compare_estimated_bpm_to_ground_truth(estimates, gt_ts, gt_bpm)
    return report


def signal_to_bpm_estimates(
    signal: np.ndarray,
    fs_hz: float,
    low_hz: float = DEFAULT_HEART_RATE_LOW_HZ,
    high_hz: float = DEFAULT_HEART_RATE_HIGH_HZ,
    window_size: int = 30,
    hop_size: int = 10,
) -> List[Tuple[float, float, float]]:
    """Convert a 1D processed signal into per-window BPM estimates."""
    signal = np.asarray(signal, dtype=np.float64)
    segments = segment_overlapping_windows(signal, window_size=window_size, hop_size=hop_size)
    estimates: List[Tuple[float, float, float]] = []
    for idx, seg in enumerate(segments):
        if len(seg) < 8:
            continue
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs_hz)
        spectrum = np.abs(np.fft.rfft(seg))
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(mask):
            continue
        dominant_hz = freqs[mask][np.argmax(spectrum[mask])]
        start_s = idx * hop_size / fs_hz
        end_s = (idx * hop_size + len(seg)) / fs_hz
        estimates.append((start_s, end_s, dominant_hz * 60.0))
    return estimates


def select_best_subcarriers_from_csi(
    csi: np.ndarray,
    fs_hz: float,
    low_hz: float = DEFAULT_HEART_RATE_LOW_HZ,
    high_hz: float = DEFAULT_HEART_RATE_HIGH_HZ,
    top_k: int = 5,
) -> tuple[list[int], dict]:
    """Return the most informative subcarrier indices after the stage-wise preprocessing chain."""
    amp = extract_amplitude_signal(csi)
    if amp.ndim == 1:
        amp = amp[:, None]

    processed = []
    for idx in range(amp.shape[1]):
        sig = remove_dc_component(amp[:, idx])
        sig = bandpass_heartbeat(sig, fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz)
        sig = savgol_smooth(sig, window_length=9, polyorder=3)
        processed.append(sig)
    processed = np.column_stack(processed)
    scores = evaluate_subcarriers(processed, fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz, top_k=top_k)
    return scores["selected_indices"], scores


def benchmark_best_k_subcarriers(
    csi: np.ndarray,
    gt_ts: Sequence[float],
    gt_bpm: Sequence[float],
    fs_hz: float,
    candidate_ks: Sequence[int] = (1, 3, 5, 10, 20),
    low_hz: float = DEFAULT_HEART_RATE_LOW_HZ,
    high_hz: float = DEFAULT_HEART_RATE_HIGH_HZ,
    window_size: int = 30,
    hop_size: int = 10,
) -> dict:
    """Benchmark best-K subcarrier combinations against GT, then return the winner by MAE."""
    amp = extract_amplitude_signal(csi)
    if amp.ndim == 1:
        amp = amp[:, None]

    processed: list[np.ndarray] = []
    for idx in range(amp.shape[1]):
        sig = remove_dc_component(amp[:, idx])
        sig = bandpass_heartbeat(sig, fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz)
        sig = savgol_smooth(sig, window_length=9, polyorder=3)
        processed.append(sig)
    processed = np.column_stack(processed)

    scores = evaluate_subcarriers(processed, fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz, top_k=max(candidate_ks))
    ranked = scores["ranked_indices"]

    results = []
    for k in candidate_ks:
        selected = ranked[:k]
        combined = np.mean(processed[:, selected], axis=1) if k > 1 else processed[:, selected[0]]
        estimates = signal_to_bpm_estimates(
            combined,
            fs_hz=fs_hz,
            low_hz=low_hz,
            high_hz=high_hz,
            window_size=window_size,
            hop_size=hop_size,
        )
        metrics = compare_estimated_bpm_to_ground_truth(estimates, gt_ts, gt_bpm)

        gt_match = []
        for start_s, end_s, _ in estimates:
            center_s = 0.5 * (start_s + end_s)
            nearest_idx = int(np.argmin(np.abs(np.asarray(gt_ts, dtype=np.float64) - center_s)))
            gt_match.append(float(gt_bpm[nearest_idx]))
        est_bpm = np.asarray([float(e[2]) for e in estimates], dtype=np.float64)
        gt_match_arr = np.asarray(gt_match, dtype=np.float64)
        corr = float(np.corrcoef(est_bpm, gt_match_arr)[0, 1]) if len(est_bpm) > 1 and np.std(est_bpm) > 0 and np.std(gt_match_arr) > 0 else 0.0

        result = {
            "k": int(k),
            "selected_indices": [int(i) for i in selected],
            "window_count": int(metrics["window_count"]),
            "mae_bpm": float(metrics["mae_bpm"]),
            "rmse_bpm": float(metrics["rmse_bpm"]),
            "median_error_bpm": float(metrics["median_error_bpm"]),
            "correlation": corr,
        }
        results.append(result)

    winner = min(results, key=lambda r: (float(r["mae_bpm"]), float(r["rmse_bpm"])))
    return {"results": results, "winner": winner}


def compare_estimated_bpm_to_ground_truth(
    estimates: Sequence[Tuple[float, float, float]],
    gt_ts: Sequence[float],
    gt_bpm: Sequence[float],
) -> dict:
    """Compare per-window BPM estimates to the nearest ground-truth BPM sample."""
    if not estimates:
        return {"window_count": 0, "mae_bpm": float("nan"), "rmse_bpm": float("nan"), "median_error_bpm": float("nan")}

    gt_ts = np.asarray(gt_ts, dtype=np.float64)
    gt_bpm = np.asarray(gt_bpm, dtype=np.float64)
    errors: List[float] = []
    for start_s, end_s, est_bpm in estimates:
        center_s = 0.5 * (start_s + end_s)
        nearest_idx = int(np.argmin(np.abs(gt_ts - center_s)))
        errors.append(abs(float(est_bpm) - float(gt_bpm[nearest_idx])))

    errs = np.asarray(errors, dtype=np.float64)
    return {
        "window_count": int(len(estimates)),
        "mae_bpm": float(np.mean(errs)),
        "rmse_bpm": float(np.sqrt(np.mean(errs ** 2))),
        "median_error_bpm": float(np.median(errs)),
        "max_error_bpm": float(np.max(errs)),
    }


def run_stage_benchmark_cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark each stage of the CSI heartbeat pipeline against ground-truth BPM.")
    parser.add_argument("--csi-csv", required=True, help="Complex CSI CSV from live_monitor.py")
    parser.add_argument("--gt-csv", required=True, help="Heartbeat ground-truth CSV")
    parser.add_argument("--fs-hz", type=float, default=None, help="Override sample rate (Hz); auto-estimated if omitted")
    parser.add_argument("--low-hz", type=float, default=DEFAULT_HEART_RATE_LOW_HZ)
    parser.add_argument("--high-hz", type=float, default=DEFAULT_HEART_RATE_HIGH_HZ)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--hop-size", type=int, default=10)
    args = parser.parse_args(argv)

    ts, csi = load_csi_csv_matrix(args.csi_csv)
    gt_ts, gt_bpm = load_ground_truth_csv(args.gt_csv)
    fs_hz = args.fs_hz if args.fs_hz is not None else (1.0 / np.median(np.diff(ts[np.isfinite(ts)])) if len(ts) > 1 else 0.0)
    if fs_hz <= 0:
        raise ValueError("Need a positive sampling rate to benchmark the pipeline")

    report = benchmark_pipeline_stages(
        csi,
        gt_ts=gt_ts,
        gt_bpm=gt_bpm,
        fs_hz=fs_hz,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        window_size=args.window_size,
        hop_size=args.hop_size,
    )

    print(f"CSI samples={len(ts)}  fs_hz={fs_hz:.3f}  gt_samples={len(gt_ts)}")
    for stage, metrics in report.items():
        print(
            f"{stage:>11}  windows={metrics['window_count']:>4}  "
            f"mae={metrics['mae_bpm']:>7.2f} bpm  "
            f"rmse={metrics['rmse_bpm']:>7.2f} bpm  "
            f"median={metrics['median_error_bpm']:>7.2f} bpm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_stage_benchmark_cli())
