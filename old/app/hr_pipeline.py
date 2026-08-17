#!/usr/bin/env python3
"""HR extraction from CSI captures.

This module provides a minimal contactless heart-rate estimation pipeline for
Nexmon CSI captures. It supports offline pcap decoding or post-hoc CSI CSV
inputs, selects or ranks informative subcarriers, and estimates a BPM time
series with a windowed FFT over a filtered heartbeat band.

Usage examples:
    python hr_pipeline.py --pcap csi_capture.pcap --chip 43455c0 --bw 80 \
        --out-dir results/hr --top-k 5 --window-s 8.0 --overlap 0.5

    python hr_pipeline.py --csi-csv output/csi_capture_20260728_101500.csv \
        --gt-csv output/ground_truth.csv --out-dir results/hr \
        --selected-subcarriers 4,12,18 --window-s 10.0
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from analysis import (
    amplitude,
    build_dataset,
    fftshift_subcarriers,
    select_stream,
    select_subcarriers,
)
from align_and_baseline import (
    attach_ground_truth,
    load_csi_amp_matrix,
    load_csi_csv,
    load_ground_truth_csv,
    resample_uniform,
    save_results_csv,
    summarize_metrics,
    windowed_fft_bpm,
)
from decoder import BW_TO_NFFT, CHIP_TABLE, iter_pcap_csi_frames
from filters import (
    butterworth_bandpass,
    hampel_filter,
    normalize_signal,
    remove_dc_drift,
)
from quality import evaluate_subcarriers

logger = logging.getLogger("csi_analysis.hr_pipeline")


def load_pcap_signal_matrix(
    pcap_path: str,
    chip: str,
    bw: int,
    udp_port: int = 5500,
    header_format: str = "auto",
    src_ip: Optional[str] = "10.10.10.10",
    select_core: Optional[int] = None,
    select_stream_idx: Optional[int] = None,
    exclude_dc_guard: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    frames = list(iter_pcap_csi_frames(
        pcap_path,
        chip=chip,
        bw=bw,
        udp_port=udp_port,
        header_format=header_format,
        src_ip_filter=src_ip,
    ))
    if not frames:
        raise ValueError("No CSI frames decoded from pcap")
    dataset = build_dataset(frames)
    if select_core is not None or select_stream_idx is not None:
        if select_core is None or select_stream_idx is None:
            raise ValueError("--select-core and --select-stream must be given together")
        dataset = select_stream(dataset, select_core, select_stream_idx)
    csi_shifted = fftshift_subcarriers(dataset.csi)
    matrix = amplitude(csi_shifted)
    if exclude_dc_guard:
        matrix = select_subcarriers(matrix, exclude_dc_guard=True, nfft=dataset.nfft)
    return dataset.ts, matrix, dataset.fs_est


def preprocess_signal_matrix(
    matrix: np.ndarray,
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    hampel_window: int = 51,
    hampel_sigmas: float = 3.0,
    order: int = 4,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    filtered = hampel_filter(matrix, window_size=hampel_window, n_sigmas=hampel_sigmas)
    filtered = butterworth_bandpass(filtered, low_hz=low_hz, high_hz=high_hz, fs_hz=fs_hz, order=order)
    filtered = remove_dc_drift(filtered)
    return filtered


def combine_subcarriers(matrix: np.ndarray, indices: Optional[List[int]] = None, method: str = "mean") -> np.ndarray:
    if matrix.ndim == 1:
        return matrix
    if indices is None:
        return np.mean(matrix, axis=1)
    selected = matrix[:, indices]
    if method == "mean":
        return np.mean(selected, axis=1)
    if method == "median":
        return np.median(selected, axis=1)
    if method == "sum":
        return np.sum(selected, axis=1)
    raise ValueError(f"Unknown combine method: {method}")


def estimate_heart_rate(
    ts: np.ndarray,
    signal: np.ndarray,
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    window_s: float,
    overlap: float,
    interpolate: bool = True,
) -> List[dict]:
    t_uniform, x_uniform = resample_uniform(ts, signal, fs_hz)
    if len(x_uniform) < 2:
        raise ValueError("Need at least two samples to estimate heart rate")
    x_filtered = butterworth_bandpass(x_uniform, low_hz=low_hz, high_hz=high_hz, fs_hz=fs_hz)
    return windowed_fft_bpm(
        t_uniform,
        x_filtered,
        fs_hz,
        window_s=window_s,
        overlap=overlap,
        low_hz=low_hz,
        high_hz=high_hz,
        interpolate=interpolate,
    )


def select_best_subcarriers(
    preprocessed_matrix: np.ndarray,
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    top_k: int = 5,
) -> List[int]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    quality = evaluate_subcarriers(
        preprocessed_matrix,
        fs_hz=fs_hz,
        low_hz=low_hz,
        high_hz=high_hz,
        top_k=top_k,
    )
    return quality["selected_indices"]


def parse_subcarrier_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == "":
        return None
    return [int(x) for x in value.split(",") if x.strip() != ""]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Contactless heart-rate estimation from Nexmon CSI captures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--pcap", help="Path to a recorded CSI pcap file")
    source.add_argument("--csi-csv", help="Path to an amplitude CSI CSV from live_monitor.py")

    p.add_argument("--chip", default="43455c0", choices=list(CHIP_TABLE), help="Broadcom chip used for decoding pcap captures")
    p.add_argument("--bw", type=int, default=80, choices=list(BW_TO_NFFT), help="Channel bandwidth used in the capture")
    p.add_argument("--nfft", type=int, default=None, help="Override nfft instead of deriving from bw")
    p.add_argument("--udp-port", type=int, default=5500, help="UDP port CSI was sent to")
    p.add_argument("--header-format", default="auto", choices=["auto", "legacy", "rssi"], help="Nexmon CSI header format for pcap decoding")
    p.add_argument("--src-ip", default="10.10.10.10", help="Expected CSI source IP; use '' to disable filtering")
    p.add_argument("--select-core", type=int, default=None, help="Only use packets from this core index")
    p.add_argument("--select-stream", type=int, default=None, help="Only use packets from this spatial stream index")
    p.add_argument("--exclude-dc-guard", action="store_true", help="Remove DC and guard subcarriers before processing")

    p.add_argument("--selected-subcarriers", default=None, help="Comma-separated list of subcarriers to use instead of automatic top-k selection")
    p.add_argument("--top-k", type=int, default=5, help="Number of top-ranked subcarriers to average for HR estimation")
    p.add_argument("--combine-method", choices=["mean", "median", "sum"], default="mean", help="How to combine selected subcarriers into a single signal")

    p.add_argument("--low-hz", type=float, default=0.7, help="Lower edge of the heartbeat band-pass")
    p.add_argument("--high-hz", type=float, default=3.7, help="Upper edge of the heartbeat band-pass")
    p.add_argument("--hampel-window", type=int, default=51, help="Hampel filter window size in packets")
    p.add_argument("--hampel-sigmas", type=float, default=3.0, help="Hampel filter sigma threshold")
    p.add_argument("--window-s", type=float, default=8.0, help="FFT window length in seconds")
    p.add_argument("--overlap", type=float, default=0.5, help="Fractional overlap between windows")
    p.add_argument("--no-fft-interp", action="store_true", help="Disable parabolic interpolation of the FFT peak")

    p.add_argument("--gt-csv", default=None, help="Optional heartbeat-giop ground-truth CSV for metric reporting")
    p.add_argument("--out-dir", default="results/hr", help="Directory to save HR estimates and logs")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p.parse_args(argv)


def run(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.src_ip == "":
        args.src_ip = None
    if (args.select_core is None) != (args.select_stream is None):
        raise ValueError("--select-core and --select-stream must be given together")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pcap:
        ts, matrix, fs_hz = load_pcap_signal_matrix(
            args.pcap,
            chip=args.chip,
            bw=args.bw,
            udp_port=args.udp_port,
            header_format=args.header_format,
            src_ip=args.src_ip,
            select_core=args.select_core,
            select_stream_idx=args.select_stream,
            exclude_dc_guard=args.exclude_dc_guard,
        )
    else:
        ts, matrix = load_csi_amp_matrix(args.csi_csv)
        fs_hz = 1.0 / np.median(np.diff(ts)) if len(ts) > 1 else 0.0
        if args.exclude_dc_guard:
            raise ValueError("--exclude-dc-guard is only supported for pcap input")

    if fs_hz <= 0:
        raise ValueError("Estimated sampling rate must be > 0")
    if args.high_hz >= fs_hz / 2.0:
        raise ValueError("--high-hz must be below Nyquist frequency fs/2")

    logger.info("Loaded CSI signal: %d packets, estimated fs=%.2f Hz", len(ts), fs_hz)

    preprocessed = preprocess_signal_matrix(
        matrix,
        fs_hz=fs_hz,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        hampel_window=args.hampel_window,
        hampel_sigmas=args.hampel_sigmas,
    )

    selected = parse_subcarrier_list(args.selected_subcarriers)
    if selected is None:
        selected = select_best_subcarriers(preprocessed, fs_hz, args.low_hz, args.high_hz, top_k=args.top_k)
        logger.info("Automatically selected top-%d subcarriers: %s", args.top_k, selected)
    else:
        logger.info("Using explicit selected subcarriers: %s", selected)

    combined = combine_subcarriers(preprocessed, selected, method=args.combine_method)
    combined = normalize_signal(combined)

    windows = estimate_heart_rate(
        ts,
        combined,
        fs_hz=fs_hz,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        window_s=args.window_s,
        overlap=args.overlap,
        interpolate=(not args.no_fft_interp),
    )

    if not windows:
        raise ValueError("No HR windows produced; try a longer capture or a larger window")

    save_results_csv(windows, str(out_dir / "hr_estimates.csv"))

    if args.gt_csv:
        gt_ts, gt_instant_bpm, gt_avg_bpm = load_ground_truth_csv(args.gt_csv)
        attach_ground_truth(windows, gt_ts, gt_instant_bpm, gt_avg_bpm)
        metrics = summarize_metrics(windows)
        logger.info("Ground-truth metrics: MAE=%.2f BPM RMSE=%.2f BPM Corr=%.3f", metrics["mae"], metrics["rmse"], metrics["corr"])
        print(f"MAE: {metrics['mae']:.2f} BPM")
        print(f"RMSE: {metrics['rmse']:.2f} BPM")
        print(f"Correlation: {metrics['corr']:.3f}")
    else:
        bps = [w["est_bpm"] for w in windows]
        print(f"Estimated HR windows: {len(bps)}")
        print(f"Median BPM: {np.median(bps):.1f}")
        print(f"Mean BPM:   {np.mean(bps):.1f}")
        print(f"Min BPM:    {np.min(bps):.1f}")
        print(f"Max BPM:    {np.max(bps):.1f}")

    selected_path = out_dir / "selected_subcarriers.txt"
    selected_path.write_text(
        "\n".join(str(idx) for idx in selected) + "\n"
    )
    logger.info("Saved selected subcarriers: %s", selected_path)
    logger.info("Saved HR estimate windows: %s", out_dir / "hr_estimates.csv")
    return 0


def main(argv=None):
    try:
        return run(argv)
    except Exception as exc:
        logger.exception("HR pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
