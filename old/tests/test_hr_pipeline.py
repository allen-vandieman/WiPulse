import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from hr_pipeline import (
    combine_subcarriers,
    estimate_heart_rate,
    parse_subcarrier_list,
    preprocess_signal_matrix,
    select_best_subcarriers,
)


def test_parse_subcarrier_list_none():
    assert parse_subcarrier_list(None) is None
    assert parse_subcarrier_list("") is None


def test_combine_subcarriers_mean_median_sum():
    matrix = np.array([
        [1.0, 2.0, 5.0],
        [3.0, 6.0, 1.0],
        [7.0, 8.0, 4.0],
    ])
    assert np.allclose(combine_subcarriers(matrix, [0, 2], method="mean"), np.array([3.0, 2.0, 5.5]))
    assert np.allclose(combine_subcarriers(matrix, [0, 2], method="median"), np.array([3.0, 2.0, 5.5]))
    assert np.allclose(combine_subcarriers(matrix, [0, 2], method="sum"), np.array([6.0, 4.0, 11.0]))


def test_preprocess_signal_matrix_preserves_shape():
    t = np.linspace(0, 20, 201)
    matrix = np.column_stack([
        np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t)),
        np.sin(2 * np.pi * 1.4 * t) + 0.05 * np.random.randn(len(t)),
    ])
    processed = preprocess_signal_matrix(matrix, fs_hz=10.0, low_hz=0.7, high_hz=3.5)
    assert processed.shape == matrix.shape
    assert np.all(np.isfinite(processed))


def test_select_best_subcarriers_returns_indices():
    t = np.linspace(0, 20, 201)
    signals = np.column_stack([
        np.sin(2 * np.pi * 1.2 * t),
        np.sin(2 * np.pi * 0.2 * t),
        np.sin(2 * np.pi * 1.5 * t),
    ])
    selected = select_best_subcarriers(signals, fs_hz=10.0, low_hz=0.7, high_hz=3.5, top_k=2)
    assert isinstance(selected, list)
    assert len(selected) == 2
    assert 0 in selected or 2 in selected


def test_estimate_heart_rate_outputs_windows():
    fs_hz = 25.0
    t = np.arange(0, 30.0, 1.0 / fs_hz)
    signal = np.sin(2 * np.pi * (1.33 / 60.0) * t * 60.0)
    windows = estimate_heart_rate(
        ts=t,
        signal=signal,
        fs_hz=fs_hz,
        low_hz=0.7,
        high_hz=3.7,
        window_s=8.0,
        overlap=0.5,
        interpolate=True,
    )
    assert len(windows) >= 1
    assert all("est_bpm" in w for w in windows)
    assert 30 <= np.median([w["est_bpm"] for w in windows]) <= 120
