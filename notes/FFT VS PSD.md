# PSD-Based Subcarrier Selection vs Weighted Subcarrier Averaging

## Background

A concern with the current auto-selection process is that it may be choosing subcarriers that contain significant noise or interference rather than the most reliable signal content.

Two potential improvements were identified:

1. PSD-guided subcarrier selection
2. Weighted subcarrier averaging

---

## FFT vs PSD

These approaches are not replacements for one another.

### FFT (Fast Fourier Transform)

The FFT provides the complex frequency-domain representation of a signal.

It allows us to:

- Convert a time-domain signal into the frequency domain.
- Observe amplitude and phase information for each frequency bin.
- Extract individual subcarriers.

### PSD (Power Spectral Density)

PSD is typically derived from the FFT and indicates how signal power is distributed across frequencies.

It tells us:

- How much energy exists at each frequency.
- Which frequency regions are dominant.
- Which frequencies may contain excessive noise or interference.

PSD is useful for:

- Identifying noisy frequency bins.
- Detecting persistent interference.
- Estimating subcarrier quality.
- Supporting automatic subcarrier ranking.

---

## Recommended Signal Processing Flow

Instead of:

FFT → Auto-select strongest subcarrier

Use:

FFT → Compute PSD → Rank subcarriers → Select best candidates

This provides a more informed selection process and reduces the chance of choosing a noisy subcarrier.

---

## PSD-Based Subcarrier Selection

For each subcarrier k:

PSD(k) = |X(k)|² / N

where:

- X(k) is the FFT value for subcarrier k.
- N is the FFT size.

### Proposed Method

1. Compute PSD for all subcarriers.
2. Average PSD across multiple packets or windows.
3. Calculate PSD stability over time.
4. Reject frequencies showing excessive variance.
5. Select subcarriers with:
   - High average power
   - Low noise
   - Stable behaviour over time

### Benefits

- Reduces sensitivity to interference.
- Avoids selecting unstable frequency bins.
- Provides a measurable quality metric for automatic selection.

---

## Weighted Subcarrier Averaging

Rather than selecting a single 'best' subcarrier, combine information from multiple subcarriers.

### General Form

S(t) = Σ wi · SCi(t)

where:

- SCi(t) is subcarrier i.
- wi is the weighting factor.
- S(t) is the combined signal.

---

## Weighting Strategies

### 1. Power-Based Weighting

wi = PSDi / ΣPSDi

Characteristics:

- Stronger subcarriers contribute more.
- Simple to implement.
- Uses PSD directly.

---

### 2. SNR-Based Weighting

wi = SNRi / ΣSNRi

Characteristics:

- Favors cleaner channels.
- Often performs better than power-only weighting.
- More robust against interference.

---

### 3. Variance-Based Weighting

wi = 1 / σi²

Characteristics:

- Rewards stable subcarriers.
- Suppresses noisy measurements.
- Useful when noise statistics are known.

---

## Advantages of Weighted Averaging

Compared with selecting a single subcarrier:

- No single bad subcarrier dominates the result.
- Better robustness to channel variation.
- Lower sensitivity to auto-selection errors.
- Improved signal stability.
- Better utilisation of available information.

---

## Proposed Evaluation Plan

### Phase 1: PSD Analysis

Use PSD as a diagnostic tool.

Generate:

- PSD per subcarrier
- Average PSD over time
- PSD variance over time

Objective:

Determine whether the current auto-selection process is choosing noisy or unstable subcarriers.

---

### Phase 2: Weighted Averaging Test

Implement:

wi = PSDi / ΣPSDi

Compare against:

- Current auto-selected subcarrier
- Best single subcarrier
- Average of all subcarriers
- PSD-weighted average

Objective:

Evaluate improvements in robustness and signal quality.

---

### Phase 3: PSD-Guided Auto-Selection

If PSD-based weighting shows positive results:

1. Use PSD metrics to score subcarriers.
2. Select the highest-quality candidates automatically.
3. Optionally combine selection and weighting approaches.

---

## Recommendation

Do not replace FFT with PSD.

Use FFT to obtain frequency-domain information, then use PSD as the quality metric for evaluating subcarriers.

Recommended priority:

1. Weighted subcarrier averaging ⭐
2. PSD-guided subcarrier selection
3. Current auto-selection method

### Expected Outcome

Weighted averaging is likely to provide the fastest performance improvement because it leverages information from multiple subcarriers rather than relying on a single chosen candidate. This generally leads to greater robustness, improved stability, and reduced sensitivity to poor auto-selection decisions.
