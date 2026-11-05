"""Debug phase vs amplitude signal quality on real captures."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from processing_pipeline import (
    extract_amplitude_signal,
    sanitize_phase_per_packet,
    remove_dc_component,
    bandpass_heartbeat,
    savgol_smooth,
    segment_overlapping_windows,
    load_csi_csv_matrix,
    load_ground_truth_csv,
)
from quality import evaluate_subcarriers


def debug_signal_quality(csi_csv, gt_csv):
    """Debug: show signal quality after best-K subcarrier selection."""
    ts, csi = load_csi_csv_matrix(csi_csv)
    gt_ts, gt_bpm = load_ground_truth_csv(gt_csv)
    fs_hz = 100.0

    low_hz, high_hz = 0.8, 2.17
    window_size, hop_size = 30, 10

    print(f"\nCSI shape: {csi.shape}  (n_packets={csi.shape[0]}, nfft={csi.shape[1]})")
    print(f"GT samples: {len(gt_ts)}\n")

    # === AMPLITUDE ===
    print("="*80)
    print("AMPLITUDE-BASED SIGNAL")
    print("="*80)
    
    amp = extract_amplitude_signal(csi)
    print(f"Amplitude shape: {amp.shape}")
    print(f"Amplitude value range: [{np.min(amp):.4f}, {np.max(amp):.4f}]")
    print(f"Amplitude mean: {np.mean(amp):.6f}")

    # Process each subcarrier, then score
    processed_amp = []
    for idx in range(amp.shape[1]):
        sig = remove_dc_component(amp[:, idx])
        sig = bandpass_heartbeat(sig, fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz)
        sig = savgol_smooth(sig, window_length=9, polyorder=3)
        processed_amp.append(sig)
    processed_amp = np.column_stack(processed_amp)

    scores = evaluate_subcarriers(processed_amp, fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz, top_k=10)
    ranked_amp = scores["ranked_indices"]
    print(f"Top 3 amplitude subcarriers by score: {ranked_amp[:3]}")
    
    # Show signal quality for best 3
    for k, idx in enumerate(ranked_amp[:3]):
        sel_signal = processed_amp[:, idx]
        print(f"  Subcarrier {idx}: min={np.min(sel_signal):.6f}, max={np.max(sel_signal):.6f}, std={np.std(sel_signal):.6f}")
        # Check FFT
        seg = sel_signal[:window_size]
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs_hz)
        spectrum = np.abs(np.fft.rfft(seg))
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if np.any(mask):
            peak_hz = freqs[mask][np.argmax(spectrum[mask])]
            peak_power = np.max(spectrum[mask])
            print(f"    First window FFT peak: {peak_hz:.3f} Hz ({peak_hz*60:.1f} BPM), power={peak_power:.4f}")
        else:
            print(f"    First window FFT: no energy in heart band")

    # Average best-1 and best-3
    best_1_amp = processed_amp[:, ranked_amp[0]]
    best_3_amp = np.mean(processed_amp[:, ranked_amp[:3]], axis=1)
    print(f"Best-1 amplitude avg: min={np.min(best_1_amp):.6f}, max={np.max(best_1_amp):.6f}, std={np.std(best_1_amp):.6f}")
    print(f"Best-3 amplitude avg: min={np.min(best_3_amp):.6f}, max={np.max(best_3_amp):.6f}, std={np.std(best_3_amp):.6f}")

    seg = best_1_amp[:window_size]
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs_hz)
    spectrum = np.abs(np.fft.rfft(seg))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if np.any(mask):
        peak_hz = freqs[mask][np.argmax(spectrum[mask])]
        print(f"Best-1 first window FFT peak: {peak_hz*60:.1f} BPM")

    print()

    # === PHASE ===
    print("="*80)
    print("PHASE-BASED SIGNAL")
    print("="*80)
    
    phase = sanitize_phase_per_packet(csi)
    print(f"Phase shape: {phase.shape}")
    print(f"Phase value range: [{np.min(phase):.4f}, {np.max(phase):.4f}]")
    print(f"Phase mean: {np.mean(phase):.6f}")

    # Process each subcarrier, then score
    processed_phase = []
    for idx in range(phase.shape[1]):
        sig = remove_dc_component(phase[:, idx])
        sig = bandpass_heartbeat(sig, fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz)
        sig = savgol_smooth(sig, window_length=9, polyorder=3)
        processed_phase.append(sig)
    processed_phase = np.column_stack(processed_phase)

    scores = evaluate_subcarriers(processed_phase, fs_hz=fs_hz, low_hz=low_hz, high_hz=high_hz, top_k=10)
    ranked_phase = scores["ranked_indices"]
    print(f"Top 3 phase subcarriers by score: {ranked_phase[:3]}")
    
    # Show signal quality for best 3
    for k, idx in enumerate(ranked_phase[:3]):
        sel_signal = processed_phase[:, idx]
        print(f"  Subcarrier {idx}: min={np.min(sel_signal):.6f}, max={np.max(sel_signal):.6f}, std={np.std(sel_signal):.6f}")
        # Check FFT
        seg = sel_signal[:window_size]
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs_hz)
        spectrum = np.abs(np.fft.rfft(seg))
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if np.any(mask):
            peak_hz = freqs[mask][np.argmax(spectrum[mask])]
            peak_power = np.max(spectrum[mask])
            print(f"    First window FFT peak: {peak_hz:.3f} Hz ({peak_hz*60:.1f} BPM), power={peak_power:.4f}")
        else:
            print(f"    First window FFT: no energy in heart band")

    # Average best-1 and best-3
    best_1_phase = processed_phase[:, ranked_phase[0]]
    best_3_phase = np.mean(processed_phase[:, ranked_phase[:3]], axis=1)
    print(f"Best-1 phase avg: min={np.min(best_1_phase):.6f}, max={np.max(best_1_phase):.6f}, std={np.std(best_1_phase):.6f}")
    print(f"Best-3 phase avg: min={np.min(best_3_phase):.6f}, max={np.max(best_3_phase):.6f}, std={np.std(best_3_phase):.6f}")

    seg = best_1_phase[:window_size]
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs_hz)
    spectrum = np.abs(np.fft.rfft(seg))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if np.any(mask):
        peak_hz = freqs[mask][np.argmax(spectrum[mask])]
        print(f"Best-1 first window FFT peak: {peak_hz*60:.1f} BPM")

    print()


if __name__ == "__main__":
    debug_signal_quality(
        "app/output/csi_capture_20260801_192024.csv",
        "app/output/ground_truth_20260801_192034.csv"
    )
