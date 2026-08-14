# Filtering Pipeline

The research papers clearly point toward moving beyond simple bandpass filtering towards more sophisticated techniques like Variational Mode Decomposition (VMD), which is crucial for separating overlapping physiological signals (like heart rate, breathing, and motion) present in the raw CSI data.

I have structured three distinct pipeline options for you—from a highly foundational approach to a state-of-the-art decomposition model—all designed to process your captured CSI data from output or test data structures.

## 🔬 Pipeline Option 1: Advanced Filtering and Feature Selection (Baseline Improvement)

This option improves the existing "filtering" approach by systematically selecting subcarriers based on their signal characteristics in relation to known physiological bands, minimizing noise contamination.

Goal: Isolate clean HR signals while mitigating baseline drift and common artifacts.

Key Technique Focus: Selective frequency domain analysis and advanced windowing.

Steps:

1. **Data Ingestion & Preprocessing**: Load CSI magnitude/phase data (from app/output/*.csv or .pcap). Normalize the signal across all subcarriers to mitigate channel attenuation variations.

2. **Drift Mitigation**: Implement a robust DC-drift removal technique, perhaps using a rolling average or a polynomial fitting model over longer time windows (>5 seconds).

3. **Subcarrier Weighting** (The Core Improvement): Instead of fixed bandwidths, use Band Power Analysis. Calculate the power spectral density (PSD) for each subcarrier's signal across adjacent bands (e.g., 0.7-1.2 Hz, 1.2-2.5 Hz). Subcarriers whose power distribution aligns best with known cardiac and respiratory frequency signatures receive a higher weighting factor (𝑤<sub>𝑖</sub>).

4. **Signal Aggregation**: Compute the combined signal 𝑆<sub>𝐻𝑅</sub>(𝑡) by taking a weighted average of the time series from all high-𝑤 subcarriers:

<h3><center>𝑆<sub>𝐻𝑅</sub>(𝑡)=∑𝑤<sub>𝑖</sub>⋅𝐶𝑆𝐼<sub>𝑖</sub>(𝑡)</center></h3>

5. **Heart Rate Estimation**: Apply FFT/Wavelet Transform on 𝑆<sub>𝐻𝑅</sub>(𝑡) with short, overlapping windows (𝑁=128). Use the peak detection method (e.g., auto-correlation function or adaptive filtering) to estimate BPM.

*Papers Guided By*: Mining Wi-Fi Channel State Information for breathing and heart rate, Parameter tuning for accurate heart rate measurement using Wi-Fi signals.

## 🌀 Pipeline Option 2: Decomposition-Based Source Separation (VMD Approach - Recommended Overhaul)

This is the most robust approach suggested by your research papers, which leverage Variational Mode Decomposition (VMD). VMD decomposes a complex signal into multiple intrinsic components (or "modes") that are nearly orthogonal and represent underlying physical sources of the signal. This is superior because it doesn't assume a single fixed frequency band; it extracts them.

Goal: Decompose mixed signals into source-separated physiological modes (e.g., breathing mode, HR mode, motion artifacts).
Key Technique Focus: Variational Mode Decomposition (VMD) or Empirical Mode Decomposition (EMD).

Steps:

1. **Signal Vectorization**: Collect a sufficiently long time window of CSI magnitude data across all relevant subcarriers into a single signal vector **𝑋(𝑡)**.

2. Mode Decomposition (The Core Step): Apply the VMD algorithm to **𝑋(𝑡)** using an initial estimate for the number of modes, 𝐾 (e.g., 𝐾=4 for heart/breath/motion/noise). This yields 𝐾 mode signals:
{Mode<sub>1</sub>(𝑡),Mode<sub>2</sub>(𝑡),...,Mode<sub>𝐾</sub>(𝑡)}.

3. **Signal Labeling & Filtering**: Analyze the frequency content (spectrum) of each Mode<sub>𝑘</sub>(𝑡). Manually or semi-automatically assign modes to sources:

- Mode with peaks near 0.7–2.5 Hz → Heart Rate ( 𝐻𝑅_𝑚𝑜𝑑𝑒).

- Mode with peaks near 0.1–0.4 Hz → Breathing Rate (𝐵𝑅_𝑚𝑜𝑑𝑒).

- Other lower-frequency/high-amplitude modes → Body Movement (𝑀𝑜𝑡𝑖𝑜𝑛_𝑚𝑜𝑑𝑒).

4. Heart Rate Extraction: Perform a final, targeted analysis (e.g., simple peak detection or filtering) only on the extracted 𝐻𝑅_𝑚𝑜𝑑𝑒(𝑡) to get the clearest BPM estimate.

*Papers Guided By*: Biometric Information Acquisition System Using VMD in Wi-Fi Channel Status Information, Heart Rate Variability and Body-Movement Extractions using Wi-Fi Channel State Information.

## 🧠 Pipeline Option 3: Machine Learning Feature Engineering (Model-Driven)

If the signal characteristics are highly non-linear or subject to environmental noise not covered by pure signal processing, a machine learning approach is needed. This treats the pipeline as an end-to-end feature extractor for classification/regression.

Goal: Train a model on manually engineered features derived from CSI data to predict BPM.

**Key Technique Focus**: Feature extraction (statistical moments, frequency domain representations) followed by ML modeling (Random Forest, LSTM).

Steps:

1. **Feature Calculation Block**: For every fixed time window 𝑊:

- Calculate basic statistics: Mean, Std Dev, Skewness, Kurtosis of CSI magnitude/phase for all subcarriers.

- Apply Frequency Domain Features: Calculate the spectral centroid, spectral width, and total power within specific frequency bands (e.g., [0.7,3.7] Hz).

- Derive Time-Domain features: Short-Time Fourier Transform (STFT) coefficients over a fixed window length.

2. **Feature Matrix Assembly**: Compile the calculated values into a feature vector 𝐹<sub>𝑊</sub> for that window 𝑊.

3. **Training/Inference**:

- *Training*: Use labeled data (from ground_truth_*.csv) to train a classifier/regressor (𝑀𝑜𝑑𝑒𝑙) on [𝐹<sub>𝑊</sub>] → {𝐵𝑃𝑀}.

- Live Inference: For new captured CSI, calculate the feature vector 𝐹<sub>𝑛𝑒𝑤</sub> and pass it through the trained 𝑀𝑜𝑑𝑒𝑙 to predict BPM.

*Papers Guided By*: LSTM-CNN network for human activity recognition using, Human Biometric Signals Monitoring based on.