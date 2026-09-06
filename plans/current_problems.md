# Plan

## Current live pipeline

```
add_frame()          rolling deque, kept by timestamp not packet count (window_seconds, default 30s)
    |
update()             triggered every update_seconds (2s), requires >=100 frames and >=60% of window filled
    |
_estimate_sample_rate()   median(dt) over all timestamps in window
    |
_clean_csi()          CSIFilter.clean_csi() -> detect DC/zero subcarriers, interpolate them
    |
_extract_best_signal() amplitude, keep subcarriers with finite mean>1e-9,
                        normalize each by its own mean, take MEDIAN across subcarriers
    |
_process_signal()      Hampel -> (optional notch) -> Butterworth bandpass -> remove_dc_drift -> normalize
    |
_estimate_bpm()        Welch PSD (~8s segments), argmax in [low_hz, high_hz], bpm = peak_hz*60
                        confidence = clip((peak_power/median_in-band_power - 1)/9 * 100, 0, 100)
```

## Improvements needed

### 1. No quality-based channel selection — this is the biggest gap.

`_extract_best_signal` throws every "valid" subcarrier into one median, unweighted. Your offline pipeline (`HeartbeatPipeline.compute_quality_score` + `fuse_top_channels`) instead scores each candidate channel by in-band peak amplitude vs. out-of-band noise floor, then fuses only the top-N with polarity alignment. The live version has no equivalent — a handful of noisy/DC-adjacent subcarriers that slipped past the coarse `mean > 1e-9` check can drag the median around. Porting a lightweight version of `compute_quality_score` + top-N fusion into the live path would likely be your single highest-leverage fix.

### 2. No polarity alignment before combining.

Even with the median (which is somewhat robust to sign flips vs. a mean), subcarriers whose heartbeat component is phase-inverted relative to others still partially cancel signal rather than reinforcing it. Your offline fusion explicitly checks correlation sign and flips as needed.

### 3. Confidence metric is a coarse heuristic.

`peak_power / median_power` within the narrow heartbeat band isn't the same comparison your offline `compute_quality_score` makes (peak vs. noise floor outside the band, over 0.1–5 Hz). Since the in-band median can already be inflated by respiration harmonics or filter passband shape, the live confidence number probably doesn't track true signal quality as well as it could. I'd reuse the out-of-band noise floor approach for consistency between live and offline results.

### 4. No quadratic peak interpolation.

`windowed_psd_analysis.py` refines the PSD peak with `_quadratic_peak` (3-point parabolic interpolation). The live `_estimate_bpm` just takes `argmax` on the raw Welch bin — with ~8s segments your frequency resolution is roughly 0.125 Hz ≈ 7.5 BPM per bin, so your live BPM is effectively quantized to ~7.5 BPM steps. Adding the same interpolation would meaningfully smooth the output.

### 5. Motion isn't gating the estimate.

You already compute a `motion_score` for the dashboard, but it's fully decoupled from `HeartRateEstimator`. During a motion event the amplitude signal is dominated by movement, not perfusion — the current code will still confidently report a (likely wrong) BPM. Wiring the motion score in to suppress/flag the BPM output during high-motion windows would cut a class of bad readings.

### 6. Minor: remove_dc_drift edge effects.

It uses `np.convolve(..., mode="same")`, which implicitly zero-pads — the first/last `window_size/2` samples get biased toward zero rather than a true local average. Not likely dominant, but worth a `mode='reflect'`/`uniform_filter1d(..., mode='nearest')` swap if you're chasing small errors.

### 7. No temporal smoothing across updates.

Each `update()` call is a fresh, independent PSD estimate — no continuity constraint from the previous BPM. A light EMA or median-of-last-3 on the reported BPM (not the underlying signal) would reduce jitter in the dashboard number without adding real lag, since your window already overlaps heavily between updates.

### 8. Single method only.

Live only ever uses amplitude. Your validation framework tests phase, phase_common, phase_diff, and the amplitude/phase hybrid too — if one of those came out ahead in your `multi_capture_validation.py` runs, it'd be worth matching the live extraction method to whichever won, rather than defaulting to amplitude.

If you want, I can draft the concrete patch for #1 (quality-scored top-N fusion) since that's the change most likely to actually move your accuracy numbers — it's a fairly direct port of `HeartbeatPipeline` into the live class.