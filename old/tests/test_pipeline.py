import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from align_and_baseline import prepare_ml_dataset
from filters import remove_dc_drift, normalize_signal
from quality import evaluate_subcarriers


def test_remove_dc_and_normalize_preserve_shape_and_centering():
    x = np.column_stack([
        np.linspace(0.0, 5.0, 120) + np.sin(np.linspace(0, 4 * np.pi, 120)) * 0.2,
        np.linspace(2.0, -1.0, 120) + np.cos(np.linspace(0, 3 * np.pi, 120)) * 0.1,
    ])

    detrended = remove_dc_drift(x, window_size=11)
    normalized = normalize_signal(detrended)

    assert detrended.shape == x.shape
    assert normalized.shape == x.shape
    assert np.isclose(np.mean(normalized[:, 0]), 0.0, atol=0.2)
    assert np.isclose(np.std(normalized[:, 0]), 1.0, atol=0.2)


def test_subcarrier_quality_ranking_is_stable():
    t = np.linspace(0, 10, 400)
    good = np.sin(2 * np.pi * 1.2 * t) + 0.01 * np.sin(2 * np.pi * 0.3 * t)
    bad = np.sin(2 * np.pi * 0.3 * t) + 0.01 * np.cos(2 * np.pi * 0.1 * t)
    signals = np.column_stack([good, bad])

    result = evaluate_subcarriers(signals, fs_hz=20.0, top_k=1)

    assert result["selected_indices"] == [0]
    assert result["scores"][0] >= result["scores"][1]


def test_prepare_ml_dataset_builds_windows_and_labels():
    t = np.arange(0, 12.0, 0.1)
    amp = np.column_stack([
        np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.sin(2 * np.pi * 0.2 * t),
        np.sin(2 * np.pi * 0.3 * t) + 0.05 * np.cos(2 * np.pi * 0.1 * t),
        np.sin(2 * np.pi * 1.1 * t) + 0.02 * np.sin(2 * np.pi * 0.4 * t),
    ])
    gt_ts = t
    gt_bpm = np.full_like(t, 72.0, dtype=np.float64)

    dataset = prepare_ml_dataset(
        csi_ts=t,
        signal_matrix=amp,
        gt_ts=gt_ts,
        gt_bpm=gt_bpm,
        fs_hz=10.0,
        window_size_packets=20,
        step_packets=10,
        selected_subcarriers=[0, 2],
        low_hz=0.7,
        high_hz=2.5,
    )

    assert dataset["processed_matrix"].shape[0] >= 1
    assert dataset["processed_matrix"].shape[1] == 2
    assert len(dataset["labels"]) == dataset["windows"].shape[0]
    assert dataset["windows"].shape[1] == 20
    assert dataset["windows"].shape[2] == 2
