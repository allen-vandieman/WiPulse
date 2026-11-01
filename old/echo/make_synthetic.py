#!/usr/bin/env python3
"""
make_synthetic.py
=================

Generate a synthetic CSI capture with a KNOWN heart rate, plus a
matching ground-truth CSV, so that diagnose_psd.py can be run on it.

Usage:
    python make_synthetic.py --hr 90 --duration 120 --out-prefix synthetic_90bpm

Outputs:
    <out-prefix>_csi.csv
    <out-prefix>_gt.csv

Then run:
    python diagnose_psd.py ^
        --csi synthetic_90bpm_csi.csv ^
        --gt  synthetic_90bpm_gt.csv ^
        --window-start 60 --window-seconds 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic(
    hr_bpm: float,
    duration_s: float,
    fs: float,
    n_sc: int,
    out_prefix: Path,
    breath_bpm: float = 18.0,
    noise_level: float = 1.5,
    hr_amplitude: float = 5.0,
    breath_amplitude: float = 20.0,
    packet_loss: float = 0.0,
    seed: int = 1234,
):
    rng = np.random.default_rng(seed)

    n = int(duration_s * fs)
    t = np.arange(n) / fs

    hr_hz = hr_bpm / 60.0
    breath_hz = breath_bpm / 60.0

    # Simulated "packet loss" — drop a fraction of samples and let the
    # remaining ones sit at irregular timestamps.
    if packet_loss > 0:
        keep = rng.random(n) > packet_loss
        t = t[keep]
        n = len(t)

    # Start from a base timestamp that looks like a real capture.
    ts0 = 1_788_000_000.0
    ts = ts0 + t

    # Build the CSV rows.
    rows = []
    for i, ti in enumerate(t):
        row = {
            "ts": ts[i],
            "seq_num": i,
            "core": 0,
            "spatial_stream": 0,
        }
        for sc in range(n_sc):
            # Give different subcarriers different sensitivity so the
            # pipeline has a real "quality" ranking to work with.
            sens = 0.2 + 0.8 * np.exp(-((sc - 30) ** 2) / 300.0)

            amp = (
                100.0
                + hr_amplitude * sens * np.sin(2 * np.pi * hr_hz * ti)
                + breath_amplitude * sens * np.sin(2 * np.pi * breath_hz * ti)
                + rng.normal(0.0, noise_level)
            )

            phase = sc * 0.1
            row[f"sc{sc}_re"] = amp * np.cos(phase)
            row[f"sc{sc}_im"] = amp * np.sin(phase)

        rows.append(row)

    csi_df = pd.DataFrame(rows)
    csi_path = out_prefix.parent / f"{out_prefix.name}_csi.csv"
    csi_df.to_csv(csi_path, index=False)
    print(f"Wrote CSI    : {csi_path}  ({len(csi_df)} rows)")

    # ------------------------------------------------------------
    # Ground-truth CSV
    # ------------------------------------------------------------
    # The GT format expects one BPM sample per second.  We insert a
    # tiny amount of realistic jitter (±0.5 BPM) to simulate sensor
    # noise, but the median is exactly hr_bpm.
    gt_rate = 1.0
    n_gt = int(duration_s * gt_rate)
    gt_t = ts0 + np.arange(n_gt) / gt_rate

    # Smooth drift so it's not a flat line — mimics real physiology.
    drift = 3.0 * np.sin(2 * np.pi * 0.02 * np.arange(n_gt))
    sensor_noise = rng.normal(0.0, 0.5, size=n_gt)
    gt_bpm = hr_bpm + drift + sensor_noise

    gt_df = pd.DataFrame({"ts": gt_t, "bpm": gt_bpm})
    gt_path = out_prefix.parent / f"{out_prefix.name}_gt.csv"
    gt_df.to_csv(gt_path, index=False)
    print(f"Wrote GT     : {gt_path}  ({len(gt_df)} rows)")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print()
    print("Synthetic capture summary")
    print("-------------------------")
    print(f"  True HR       : {hr_bpm:.2f} BPM ({hr_hz:.4f} Hz)")
    print(f"  Duration      : {duration_s:.1f} s")
    print(f"  Sample rate   : {fs:.2f} Hz")
    print(f"  Subcarriers   : {n_sc}")
    print(f"  Packets kept  : {len(csi_df)}")
    print(f"  GT median     : {np.median(gt_bpm):.2f} BPM")
    print(f"  GT σ          : {np.std(gt_bpm):.2f} BPM")
    print()
    print("Now run:")
    print()
    print(
        f"  python diagnose_psd.py "
        f"--csi \"{csi_path}\" "
        f"--gt  \"{gt_path}\" "
        f"--window-start 60 --window-seconds 10"
    )

    return csi_path, gt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", type=float, default=90.0,
                    help="True heart rate in BPM")
    ap.add_argument("--duration", type=float, default=120.0,
                    help="Capture duration in seconds")
    ap.add_argument("--fs", type=float, default=100.0,
                    help="Target sample rate")
    ap.add_argument("--n-sc", type=int, default=256,
                    help="Number of subcarriers")
    ap.add_argument("--breath", type=float, default=18.0,
                    help="Respiration rate in BPM")
    ap.add_argument("--noise", type=float, default=1.5,
                    help="ADC noise sigma")
    ap.add_argument("--packet-loss", type=float, default=0.0,
                    help="Fraction of packets to drop (0-1)")
    ap.add_argument("--out-prefix", type=str,
                    default="synthetic",
                    help="Output file prefix")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    make_synthetic(
        hr_bpm=args.hr,
        duration_s=args.duration,
        fs=args.fs,
        n_sc=args.n_sc,
        out_prefix=out_prefix,
        breath_bpm=args.breath,
        noise_level=args.noise,
        packet_loss=args.packet_loss,
    )


if __name__ == "__main__":
    main()