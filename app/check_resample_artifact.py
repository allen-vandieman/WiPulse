#!/usr/bin/env python3
"""
check_resample_artifact.py
===========================
Diagnoses whether a dominant frequency reported by main.py's --fft is a real
RF-driven signal or an artifact of treating unevenly-spaced packets as if
they were evenly spaced at fs_est.

Background: fft_motion_detection() (analysis.py) runs np.fft.rfft() on the
amplitude array by *index*, assuming uniform spacing at dataset.fs_est (the
*average* packet rate). If packets were actually lost in bursts rather than
spread evenly, the real time axis is irregular, and that irregularity can
alias into a spurious "dominant frequency" that has nothing to do with the
physical environment.

This script decodes a pcap two ways and compares the FFT result:
  1. "naive"     -- exactly what main.py does: index-based FFT at fs_est,
                     ignoring real timestamps.
  2. "resampled" -- amplitude is linearly interpolated onto a *true* uniform
                     time grid (align_and_baseline.resample_uniform) before
                     the FFT, so gaps/bursts are handled honestly.

If the two dominant frequencies agree, the naive result was trustworthy for
this capture. If they diverge, packet loss is corrupting the naive FFT and
the resampled number is the one to trust.

Usage:
    python check_resample_artifact.py --pcap test/baseline-router-off.pcap \\
        --chip 43455c0 --bw 80 --bandpass-low 0.7 --bandpass-high 2.5
"""

from __future__ import annotations

import argparse

import numpy as np

from decoder import iter_pcap_csi_frames, CHIP_TABLE, BW_TO_NFFT
from analysis import fftshift_subcarriers, amplitude, select_subcarriers, fft_motion_detection
from filters import butterworth_bandpass
from align_and_baseline import resample_uniform


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pcap", required=True)
    p.add_argument("--chip", default="43455c0", choices=list(CHIP_TABLE))
    p.add_argument("--bw", type=int, default=80, choices=list(BW_TO_NFFT))
    p.add_argument("--udp-port", type=int, default=5500)
    p.add_argument("--src-ip", default="10.10.10.10", help="pass '' to disable")
    p.add_argument("--exclude-dc-guard", action="store_true")
    p.add_argument("--bandpass-low", type=float, default=0.7)
    p.add_argument("--bandpass-high", type=float, default=2.5)
    return p.parse_args()


def main():
    args = parse_args()

    frames = list(iter_pcap_csi_frames(
        args.pcap, chip=args.chip, bw=args.bw, udp_port=args.udp_port,
        src_ip_filter=(args.src_ip or None),
    ))
    if not frames:
        print("No frames decoded -- check --chip/--bw/--udp-port.")
        return 1

    nfft = frames[0].csi.shape[0]
    csi = np.stack([f.csi for f in frames], axis=0)
    ts = np.array([f.ts for f in frames], dtype=np.float64)

    csi_shifted = fftshift_subcarriers(csi)
    if args.exclude_dc_guard:
        csi_shifted = select_subcarriers(csi_shifted, exclude_dc_guard=True, nfft=nfft)

    amp = amplitude(csi_shifted)
    mean_amp = np.mean(amp, axis=1)  # mean across subcarriers -> one signal for the diagnosis

    duration = ts[-1] - ts[0]
    n_packets = len(ts)
    fs_est = (n_packets - 1) / duration if duration > 0 else 0.0
    loss_pct = 100.0 * (1.0 - (n_packets / (fs_est * duration))) if fs_est > 0 else float("nan")

    print(f"packets decoded: {n_packets}   span: {duration:.2f}s   fs_est: {fs_est:.2f} Hz")
    print(f"(rough) missing-sample fraction implied by fs_est vs actual count: {loss_pct:.1f}%")

    if fs_est <= 0:
        print("fs_est <= 0, can't proceed.")
        return 1

    # --- 1. naive path: exactly what main.py does ---
    naive_filtered = butterworth_bandpass(mean_amp, low_hz=args.bandpass_low,
                                           high_hz=args.bandpass_high, fs_hz=fs_est)
    naive_result = fft_motion_detection(naive_filtered[:, None], fs=fs_est, subcarrier_idx=0)

    # --- 2. resampled path: honest uniform time grid ---
    t_uniform, amp_uniform = resample_uniform(ts, mean_amp, fs_est)
    resampled_filtered = butterworth_bandpass(amp_uniform, low_hz=args.bandpass_low,
                                                high_hz=args.bandpass_high, fs_hz=fs_est)
    resampled_result = fft_motion_detection(resampled_filtered[:, None], fs=fs_est, subcarrier_idx=0)

    print("\n=== Naive (index-based, ignores real gaps -- what main.py --fft does) ===")
    print(f"dominant frequency: {naive_result['dominant_freq_hz']:.4f} Hz")
    print(f"dominant magnitude: {naive_result['dominant_magnitude']:.4f}")

    print("\n=== Resampled onto true uniform time grid ===")
    print(f"dominant frequency: {resampled_result['dominant_freq_hz']:.4f} Hz")
    print(f"dominant magnitude: {resampled_result['dominant_magnitude']:.4f}")

    diff = abs(naive_result["dominant_freq_hz"] - resampled_result["dominant_freq_hz"])
    print(f"\nfrequency difference: {diff:.4f} Hz")
    if diff < 0.05:
        print("-> Naive and resampled agree closely. This capture's loss pattern "
              "probably isn't the source of the artifact you're chasing.")
    else:
        print("-> Naive and resampled diverge. Packet loss/gap handling is likely "
              "distorting the naive FFT for this capture -- treat the resampled "
              "number as the more trustworthy one, and prioritize getting a "
              "lower-loss capture before drawing RF conclusions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
