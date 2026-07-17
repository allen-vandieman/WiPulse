#!/usr/bin/env python3
"""
main.py
=======
CLI entry point for the Nexmon CSI analysis toolkit.

Typical usage
-------------
Decode a capture, dump validity stats, save heatmap + FFT motion plot,
and cache the decoded dataset for later ML use:

    python main.py --pcap csi_capture.pcap --chip 43455c0 --bw 80 \\
        --out-dir results --heatmap --fft --save-npy --save-csv amplitude

Validate a capture is well-formed (no plots, just stats):

    python main.py --pcap csi_capture.pcap --chip 43455c0 --bw 80 --stats-only

Compare a "stationary" baseline capture against a "movement" capture (see
README for the two-capture test procedure):

    python main.py --pcap baseline.pcap --chip 43455c0 --bw 80 \\
        --out-dir results/baseline --heatmap --fft
    python main.py --pcap movement.pcap --chip 43455c0 --bw 80 \\
        --out-dir results/movement --heatmap --fft
    # then compare motion_score and the FFT dominant frequency between the two
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

from decoder import iter_pcap_csi_frames, CHIP_TABLE, BW_TO_NFFT
from analysis import (
    build_dataset, report_sequence_gaps, fftshift_subcarriers, subcarrier_axis,
    amplitude, phase, select_subcarriers, variance_analysis, fft_motion_detection,
)
from filters import hampel_filter, butterworth_lowpass
from plotting import plot_amplitude_time, plot_phase_time, plot_heatmap, plot_fft, plot_variance_bar
from utils import setup_logging, ensure_dir, save_npy, save_csv

logger = logging.getLogger("csi_analysis.main")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Decode and analyze Nexmon CSI captures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pcap", required=True, help="Path to the captured .pcap file")
    p.add_argument("--chip", default="43455c0", choices=list(CHIP_TABLE),
                   help="Broadcom chip used for extraction")
    p.add_argument("--bw", type=int, default=80, choices=list(BW_TO_NFFT),
                   help="Channel bandwidth (MHz) used during capture")
    p.add_argument("--nfft", type=int, default=None,
                   help="Override nfft directly instead of deriving from --bw")
    p.add_argument("--udp-port", type=int, default=5500, help="UDP port CSI was sent to")
    p.add_argument("--header-format", default="auto", choices=["auto", "legacy", "rssi"],
                   help="Nexmon CSI header layout: 'legacy' = 6-byte magic (main README), "
                        "'rssi' = 2-byte magic + RSSI + FrameControl (newer Pi forks). "
                        "'auto' detects per-packet.")
    p.add_argument("--src-ip", default="10.10.10.10",
                   help="Expected CSI source IP; pass '' to disable this filter")
    p.add_argument("--out-dir", default="results", help="Directory for outputs (plots, exports)")

    p.add_argument("--stats-only", action="store_true",
                   help="Only decode + print validity/loss stats; skip plots and exports")

    # analysis options
    p.add_argument("--subcarriers", default=None,
                   help="Comma-separated subcarrier indices to focus on for time-series "
                        "plots, e.g. '10,20,30' (indices are into the fftshifted axis). "
                        "Default: mean over all subcarriers.")
    p.add_argument("--exclude-dc-guard", action="store_true",
                   help="Zero out DC and edge guard subcarriers before analysis")
    p.add_argument("--normalize-amp", action="store_true", help="Normalize each packet's amplitude to max=1")

    # filtering
    p.add_argument("--hampel", action="store_true", help="Apply a Hampel filter to amplitude before analysis/plots")
    p.add_argument("--hampel-window", type=int, default=7)
    p.add_argument("--hampel-sigmas", type=float, default=3.0)
    p.add_argument("--lowpass-hz", type=float, default=None,
                   help="Apply a Butterworth low-pass filter at this cutoff (Hz) before FFT/plots")

    # outputs
    p.add_argument("--heatmap", action="store_true", help="Save a packet x subcarrier amplitude heatmap")
    p.add_argument("--time-plot", action="store_true", help="Save an amplitude-over-time plot")
    p.add_argument("--phase-plot", action="store_true", help="Save a phase-over-time plot")
    p.add_argument("--fft", action="store_true", help="Run FFT-based motion detection and save the spectrum plot")
    p.add_argument("--fft-max-freq", type=float, default=None, help="Clip the FFT plot x-axis to this frequency (Hz)")
    p.add_argument("--variance-plot", action="store_true", help="Save a per-subcarrier variance bar chart")
    p.add_argument("--save-npy", action="store_true", help="Save the decoded dataset as .npz")
    p.add_argument("--save-csv", choices=["complex", "amplitude", "phase"], default=None,
                   help="Save the decoded dataset as CSV in the given representation")

    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.verbose)

    if not os.path.exists(args.pcap):
        logger.error("pcap file not found: %s", args.pcap)
        return 1

    ensure_dir(args.out_dir)

    subcarrier_idx = None
    if args.subcarriers:
        try:
            subcarrier_idx = [int(x) for x in args.subcarriers.split(",")]
        except ValueError:
            logger.error("--subcarriers must be a comma-separated list of ints, got: %s", args.subcarriers)
            return 1

    # --- decode ---
    logger.info("Decoding %s (chip=%s, bw=%dMHz)...", args.pcap, args.chip, args.bw)
    frames = list(iter_pcap_csi_frames(
        args.pcap, chip=args.chip, bw=args.bw, udp_port=args.udp_port,
        nfft_override=args.nfft, src_ip_filter=(args.src_ip or None),
        header_format=args.header_format,
    ))

    if not frames:
        logger.error(
            "No CSI frames decoded. Common causes: wrong --chip/--bw for this "
            "capture, wrong --udp-port, or --src-ip filter not matching (try --src-ip '')."
        )
        return 1

    dataset = build_dataset(frames)
    gaps = report_sequence_gaps(dataset)

    print("\n=== Capture validity ===")
    print(json.dumps(gaps, indent=2))
    print(f"nfft (subcarriers): {dataset.nfft}")
    print(f"capture span: {dataset.ts[-1] - dataset.ts[0]:.2f} s" if dataset.n_packets > 1 else "capture span: n/a")
    print(f"estimated packet rate: {dataset.fs_est:.2f} pkt/s")

    if args.stats_only:
        return 0

    # --- amplitude / phase, fftshifted to DC-centered order ---
    csi_shifted = fftshift_subcarriers(dataset.csi)
    sc_axis = subcarrier_axis(dataset.nfft)

    if args.exclude_dc_guard:
        csi_shifted = select_subcarriers(csi_shifted, exclude_dc_guard=True, nfft=dataset.nfft)
        mask = np.ones(dataset.nfft, dtype=bool)
        center = dataset.nfft // 2
        mask[center - 1:center + 2] = False
        mask[:2] = False
        mask[-2:] = False
        sc_axis = sc_axis[mask]

    amp = amplitude(csi_shifted, normalize=args.normalize_amp)
    ph = phase(csi_shifted, unwrap=False)

    if args.hampel:
        logger.info("Applying Hampel filter (window=%d, sigmas=%.1f)...",
                    args.hampel_window, args.hampel_sigmas)
        amp = hampel_filter(amp, window_size=args.hampel_window, n_sigmas=args.hampel_sigmas)

    if args.lowpass_hz is not None:
        fs = dataset.fs_est
        if fs <= 0:
            logger.warning("Cannot low-pass filter: no valid estimated sample rate from timestamps")
        else:
            logger.info("Applying Butterworth low-pass at %.3f Hz (fs=%.2f Hz)...", args.lowpass_hz, fs)
            amp = butterworth_lowpass(amp, cutoff_hz=args.lowpass_hz, fs_hz=fs)

    # --- variance / motion score ---
    var_stats = variance_analysis(amp)
    print("\n=== Variance / motion score ===")
    print(f"motion_score (higher = more amplitude fluctuation): {var_stats['motion_score']:.5f}")

    # --- FFT motion detection ---
    fft_result = None
    if args.fft:
        fs = dataset.fs_est
        if fs <= 0:
            logger.warning("Skipping FFT: no valid estimated sample rate from capture timestamps")
        else:
            sc_for_fft = subcarrier_idx[0] if subcarrier_idx else None
            fft_result = fft_motion_detection(amp, fs=fs, subcarrier_idx=sc_for_fft)
            print("\n=== FFT motion detection ===")
            print(f"dominant frequency: {fft_result['dominant_freq_hz']:.4f} Hz")
            print(f"dominant magnitude: {fft_result['dominant_magnitude']:.4f}")

    # --- plots ---
    if args.time_plot:
        plot_amplitude_time(amp, os.path.join(args.out_dir, "amplitude_time.png"),
                             subcarrier_idx=subcarrier_idx, x_axis=dataset.ts - dataset.ts[0])
    if args.phase_plot:
        plot_phase_time(ph, os.path.join(args.out_dir, "phase_time.png"),
                         subcarrier_idx=subcarrier_idx or 0, x_axis=dataset.ts - dataset.ts[0])
    if args.heatmap:
        plot_heatmap(amp, os.path.join(args.out_dir, "heatmap.png"), sc_axis=sc_axis)
    if args.variance_plot:
        plot_variance_bar(var_stats["per_subcarrier_std"], os.path.join(args.out_dir, "variance.png"),
                           sc_axis=sc_axis)
    if args.fft and fft_result is not None:
        plot_fft(fft_result["freqs"], fft_result["magnitude"],
                 os.path.join(args.out_dir, "fft_motion.png"),
                 dominant_freq=fft_result["dominant_freq_hz"], max_freq=args.fft_max_freq)

    # --- exports for later ML use ---
    if args.save_npy:
        save_npy(dataset, os.path.join(args.out_dir, "csi_dataset.npz"))
    if args.save_csv:
        save_csv(dataset, os.path.join(args.out_dir, f"csi_dataset_{args.save_csv}.csv"), mode=args.save_csv)

    logger.info("Done. Outputs in: %s", os.path.abspath(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
