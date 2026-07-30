import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

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
