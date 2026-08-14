# WiPulse Project Plan

## Objective
Build a contactless heart-rate monitor using Wi-Fi CSI on Raspberry Pi, with real-time signal processing and validation against ground-truth pulse measurements.

## Scope
- Reuse existing Nexmon CSI capture and analysis tools in `app/`
- Add a dedicated HR estimation pipeline in `app/hr_pipeline.py`
- Provide documentation and usage examples in `README.md` and `app/README.md`
- Add regression tests that verify HR extraction and subcarrier selection

## Approach
1. Decode recorded CSI or load live-monitor CSVs.
2. Preprocess CSI amplitude:
   - Hampel outlier removal
   - Heartbeat bandpass filtering (`0.7-3.7 Hz`)
   - DC drift removal
3. Score and select informative subcarriers.
4. Combine selected subcarriers and estimate heart rate using windowed FFT.
5. Save per-window BPM estimates and optional ground-truth comparison.

## Verification
- Synthetic unit tests in `tests/test_hr_pipeline.py`
- Ground-truth validation via `align_and_baseline.py` output and `app/ground_truth.csv`
- Documentation examples for offline and live use

## Files Added
- `app/hr_pipeline.py`
- `plan.md`
- `tests/test_hr_pipeline.py`

## Usage
- Offline pcap: `python app/hr_pipeline.py --pcap csi_capture.pcap --chip 43455c0 --bw 80 --out-dir results/hr`
- Live CSV: `python app/hr_pipeline.py --csi-csv output/csi_capture_20260728_101500.csv --gt-csv output/ground_truth.csv --out-dir results/hr`

## Phase 2: Advanced Research & Implementation

Building on the successful CSI extraction pipeline (Phase 1), Phase 2 focuses on integrating advanced signal processing methods derived from recent academic research found in the `research/` folder. The goal is to improve HR estimation accuracy and robustify performance against challenging environmental conditions.

**Next Steps:**
1. **Research Integration:** Systematically review papers and models stored in the `research/` directory (e.g., advanced feature extraction, deep learning approaches).
2. **Method Implementation:** Select promising techniques (e.g., non-linear regression, specific spectral analysis) and implement wrappers or modules within `app/utils.py` or a new dedicated module to test their efficacy against current data.
3. **Module Enhancement:** Update core modules like `hr_pipeline.py` and potentially add a new module for complex signal modeling.
4. **Testing & Benchmarking:** Develop specific benchmarks in the `test/` directory that compare the standard FFT approach (Phase 1) against newly implemented research-based methods, using datasets from `app/output/`.

**Target Output:** A validated, high-accuracy HR estimation pipeline ready for field deployment considerations.
