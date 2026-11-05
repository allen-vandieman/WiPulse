import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from processing_pipeline import (
    amplitude_from_complex,
    bandpass_heartbeat,
    benchmark_best_k_subcarriers,
    benchmark_pipeline_stages,
    compare_estimated_bpm_to_ground_truth,
    extract_amplitude_signal,
    extract_phase_signal,
    remove_dc_component,
    sanitize_phase_per_packet,
    savgol_smooth,
    segment_overlapping_windows,
    select_best_subcarriers_from_csi,
)


def test_extract_amplitude_signal_uses_complex_magnitude():
    csi = np.array([
        [1.0 + 2.0j, 3.0 + 4.0j],
        [0.0 + 0.0j, 5.0 + 12.0j],
    ], dtype=np.complex128)
    expected = np.array([np.hypot(1.0, 2.0), np.hypot(3.0, 4.0)])
    amp = extract_amplitude_signal(csi)
    assert amp.shape == (2, 2)
    assert np.allclose(amp[0], np.array([np.hypot(1.0, 2.0), np.hypot(3.0, 4.0)]))
    assert np.allclose(amp[1], np.array([0.0, np.hypot(5.0, 12.0)]))


def test_remove_dc_component_removes_drift():
    t = np.linspace(0, 10, 200)
    signal = 2.0 + np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 0.08 * t)
    detrended = remove_dc_component(signal)
    assert np.isclose(np.mean(detrended), 0.0, atol=1e-6)
    assert np.std(detrended) < np.std(signal)


def test_bandpass_heartbeat_keeps_heart_band():
    fs_hz = 20.0
    t = np.arange(0, 60.0, 1.0 / fs_hz)
    heart = 0.6 * np.sin(2 * np.pi * 1.2 * t)
    signal = heart + 0.05 * np.sin(2 * np.pi * 0.3 * t)
    filtered = bandpass_heartbeat(signal, fs_hz=fs_hz, low_hz=0.8, high_hz=2.17)
    fft = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs_hz)
    dominant = freqs[np.argmax(fft)]
    assert 0.8 <= dominant <= 2.17


def test_savgol_smooth_preserves_signal_shape():
    x = np.linspace(0, 10, 400)
    signal = np.sin(2 * np.pi * 1.2 * x) + 0.1 * np.sin(2 * np.pi * 8.0 * x)
    smooth = savgol_smooth(signal, window_length=31, polyorder=3)
    assert smooth.shape == signal.shape
    assert np.all(np.isfinite(smooth))
    assert np.corrcoef(signal, smooth)[0, 1] > 0.9


def test_segment_overlapping_windows_creates_expected_segments():
    signal = np.arange(0, 100, dtype=float)
    windows = segment_overlapping_windows(signal, window_size=10, hop_size=5)
    assert len(windows) == 19
    assert windows[0].shape == (10,)
    assert np.allclose(windows[-1], signal[90:100])


def test_amplitude_from_complex_handles_numpy_array():
    csi = np.array([
        [3.0 + 4.0j, 0.0 + 0.0j],
        [5.0 + 12.0j, 1.0 + 1.0j],
    ], dtype=np.complex128)
    out = amplitude_from_complex(csi)
    assert np.allclose(out, np.array([
        [5.0, 0.0],
        [13.0, np.sqrt(2.0)],
    ]))


def test_compare_estimated_bpm_to_ground_truth_metrics():
    estimates = [(0.0, 10.0, 72.0), (10.0, 20.0, 79.0)]
    gt_ts = np.array([0.0, 10.0, 20.0])
    gt_bpm = np.array([70.0, 80.0, 78.0])
    metrics = compare_estimated_bpm_to_ground_truth(estimates, gt_ts, gt_bpm)
    assert metrics["window_count"] == 2
    assert metrics["mae_bpm"] > 0.0
    assert metrics["rmse_bpm"] >= metrics["mae_bpm"]


def test_benchmark_pipeline_stages_returns_stage_report():
    fs_hz = 25.0
    t = np.arange(0, 60.0, 1.0 / fs_hz)
    heart_hz = 1.2 / 60.0
    signal = np.sin(2 * np.pi * heart_hz * t * 60.0)
    csi = np.column_stack([signal, signal * 1.1 + 0.2])
    gt_ts = np.linspace(0.0, 60.0, 10)
    gt_bpm = np.full_like(gt_ts, 72.0, dtype=float)
    report = benchmark_pipeline_stages(csi, gt_ts, gt_bpm, fs_hz=fs_hz, window_size=30, hop_size=10)
    assert set(report) == {"raw", "dc_removed", "bandpassed", "smoothed"}
    assert all("mae_bpm" in stage for stage in report.values())


def test_select_best_subcarriers_from_csi_prefers_heartbeat_signal():
    fs_hz = 25.0
    t = np.arange(0, 30.0, 1.0 / fs_hz)
    heartbeat = 2.0 * np.sin(2 * np.pi * (1.1 / 60.0) * t * 60.0)
    noise = np.random.default_rng(0).normal(0.0, 0.3, size=t.shape)
    csi = np.column_stack([
        noise,
        heartbeat + 0.1 * noise,
        0.2 * np.sin(2 * np.pi * 0.1 * t),
    ])
    selected, _ = select_best_subcarriers_from_csi(csi, fs_hz=fs_hz, low_hz=0.8, high_hz=2.17, top_k=1)
    assert selected == [1]


def test_benchmark_best_k_subcarriers_selects_best_candidate():
    fs_hz = 25.0
    t = np.arange(0, 30.0, 1.0 / fs_hz)
    heartbeat = 2.5 * np.sin(2 * np.pi * (1.1 / 60.0) * t * 60.0)
    csi = np.column_stack([
        0.1 * np.sin(2 * np.pi * 0.1 * t),
        heartbeat,
        0.2 * np.sin(2 * np.pi * 0.2 * t),
    ])
    gt_ts = np.linspace(0.0, 30.0, 6)
    gt_bpm = np.full_like(gt_ts, 66.0, dtype=float)
    benchmark = benchmark_best_k_subcarriers(csi, gt_ts, gt_bpm, fs_hz=fs_hz, candidate_ks=(1, 2), low_hz=0.8, high_hz=2.17, window_size=12, hop_size=6, signal_type="amplitude")
    assert len(benchmark["results"]) == 2
    assert benchmark["winner"]["k"] == 1
    assert benchmark["winner"]["selected_indices"] == [1]


def test_extract_phase_signal_returns_angle():
    csi = np.array([
        [1.0 + 1.0j, 0.0 + 2.0j],
        [3.0 + 4.0j, -1.0 + 1.0j],
    ], dtype=np.complex128)
    phase = extract_phase_signal(csi, unwrap=False)
    assert phase.shape == csi.shape
    assert np.isfinite(phase).all()
    assert np.allclose(phase[0, 0], np.arctan2(1.0, 1.0))


def test_sanitize_phase_per_packet_removes_linear_trend():
    csi = np.array([
        np.exp(1j * (0.1 * np.arange(8) + 0.5)),
        np.exp(1j * (0.2 * np.arange(8) - 0.3)),
    ], dtype=np.complex128)
    sanitized = sanitize_phase_per_packet(csi)
    assert sanitized.shape == csi.shape
    assert np.isfinite(sanitized).all()
