"""
utils.py
========
Saving/loading helpers and small shared utilities. Keeping decoded CSI
around as .npy (fast, exact complex128 round-trip) or .csv (portable, human
readable, one row per packet) means later ML experiments can load straight
from disk instead of re-parsing the pcap every time.
"""

from __future__ import annotations

import csv
import logging
import os
import sys

import numpy as np

from analysis import CSIDataset


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_npy(dataset: CSIDataset, out_path: str):
    """
    Saves the full dataset (complex CSI + timestamps + sequence numbers +
    core/spatial-stream indices) as a single .npz (still a NumPy binary
    format; using .npz lets us store the related arrays together under one
    file without hand-rolling a container format).
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    if not out_path.endswith(".npz"):
        out_path = os.path.splitext(out_path)[0] + ".npz"
    np.savez(out_path, csi=dataset.csi, ts=dataset.ts, seq_num=dataset.seq_num,
              core=dataset.core, spatial_stream=dataset.spatial_stream,
              nfft=dataset.nfft, fs_est=dataset.fs_est)
    logging.getLogger("csi_analysis.utils").info("Saved dataset: %s", out_path)
    return out_path


def load_npy(path: str) -> CSIDataset:
    data = np.load(path)
    return CSIDataset(csi=data["csi"], ts=data["ts"], seq_num=data["seq_num"],
                       core=data["core"], spatial_stream=data["spatial_stream"],
                       nfft=int(data["nfft"]), fs_est=float(data["fs_est"]))


def save_csv(dataset: CSIDataset, out_path: str, mode: str = "complex"):
    """
    Save as CSV, one row per packet. Columns: ts, seq_num, then per-subcarrier
    values.

    mode:
      "complex"   -> two columns per subcarrier: sc{i}_re, sc{i}_im
      "amplitude" -> one column per subcarrier: sc{i}_amp
      "phase"     -> one column per subcarrier: sc{i}_phase
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    nfft = dataset.nfft

    if mode == "complex":
        header = ["ts", "seq_num", "core", "spatial_stream"]
        for i in range(nfft):
            header += [f"sc{i}_re", f"sc{i}_im"]
    elif mode in ("amplitude", "phase"):
        header = ["ts", "seq_num", "core", "spatial_stream"] + [f"sc{i}_{mode[:3]}" for i in range(nfft)]
    else:
        raise ValueError(f"unknown mode '{mode}', expected complex/amplitude/phase")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(dataset.n_packets):
            row = [dataset.ts[i], int(dataset.seq_num[i]), int(dataset.core[i]), int(dataset.spatial_stream[i])]
            if mode == "complex":
                for v in dataset.csi[i]:
                    row += [v.real, v.imag]
            elif mode == "amplitude":
                row += list(np.abs(dataset.csi[i]))
            else:
                row += list(np.angle(dataset.csi[i]))
            writer.writerow(row)

    logging.getLogger("csi_analysis.utils").info("Saved CSV (%s): %s", mode, out_path)
    return out_path
