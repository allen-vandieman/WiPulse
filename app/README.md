# WiPulse CSI analysis toolkit

This repository now contains a full contactless vital-sign workflow for
[Nexmon CSI](https://github.com/seemoo-lab/nexmon_csi) captures. The app-side
pipeline can decode CSI from pcap files or a live interface, filter and score
subcarriers, run motion/heartbeat diagnostics, and export ML-ready datasets.

Recent updates in this repo include:
- a real-time terminal dashboard in `live_monitor.py` for live CSI capture
- a ground-truth alignment pipeline in `align_and_baseline.py` for BPM sanity checks
- subcarrier quality scoring and top-K selection for heartbeat-focused analysis
- phase-based analysis support for complex I/Q captures
- export options for `.npz` and CSV datasets for downstream ML workflows
- `check_resample_artifact.py`, a diagnostic for telling a real RF-driven
  frequency apart from an artifact of treating unevenly-spaced packets as
  evenly spaced
- a `--notch-hz` / `--notch-q` notch filter in `main.py` for suppressing a
  known narrowband interferer before the heartbeat bandpass

## Content

1. [Layout](#layout)
1. [Setup](#setup)
1. [Chip](#chip--bandwidth)
1. [Header format](#header-format)
1. [Validate the capture](#step-1--validate-the-capture-is-well-formed)
1. [Look at it](#step-2--look-at-it)
1. [Movement test](#step-3--the-movement-test)
1. [Diagnosing sample-rate/timing artifacts](#diagnosing-sample-ratetiming-artifacts)
1. [Cleaning noisy data](#cleaning-noisy-data)
1. [Suppressing a known interferer (notch filter)](#suppressing-a-known-interferer-notch-filter)
1. [Ground-truth-first quality testing](#ground-truth-first-quality-testing)
1. [Exporting for ML](#exporting-for-ml)
1. [Flags reference](#useful-flags-reference)
1. [Loss/stream diagnostics](#reading-the-lossstream-diagnostics)
1. [Live monitoring](#live-monitoring)
1. [Recording live CSI (for ML)](#recording-live-csi-for-ml)
1. [Ground Truth](#ground-truth-data)
1. [Baseline heart-rate check](#baseline-heart-rate-check)
1. [Amplitude vs. phase](#amplitude-vs-phase---signal-type)
1. [Searching all subcarriers](#searching-all-subcarriers)

## Layout

```
WiPulse/
├── app/
│   ├── main.py                 # Offline pcap analysis CLI
│   ├── live_monitor.py         # Real-time terminal dashboard and live capture
│   ├── decoder.py              # PCAP parsing + Nexmon header/CSI decoding
│   ├── analysis.py             # Amplitude/phase, subcarrier selection, variance, FFT motion detection
│   ├── plotting.py             # Headless-safe plots for time series, heatmaps, FFT, and pipeline stages
│   ├── filters.py              # Hampel, Butterworth, drift removal, and normalization
│   ├── quality.py              # Subcarrier quality scoring with optional ground-truth correlation
│   ├── utils.py                # Save/load helpers, logging, and dataset utilities
│   ├── align_and_baseline.py   # CSI + ground-truth alignment and BPM sanity-check workflow
│   ├── check_resample_artifact.py  # Diagnoses naive-FFT-vs-real-timing artifacts from packet loss
│   ├── heartbeat-giop.py       # Ground-truth heartbeat reference capture helper
│   ├── requirements.txt
│   └── scripts/
│       ├── optimized-fast-sender.py
│       └── rssi_data-extensive.py
├── tests/
└── notes/
```

## Setup

```bash
cd app
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Recommended workflow

1. Validate a capture with `main.py --stats-only`.
2. Inspect motion/heartbeat structure with plots and variance/FFT output.
3. Record live CSI and ground-truth data together for a sanity check.
4. Use `align_and_baseline.py` to confirm that the CSI signal contains a usable heart-rate signature before moving to ML.

## Chip / bandwidth

Set `--chip` to whatever your nexmon_csi build targets:

| chip       | CSI format                          |
|------------|--------------------------------------|
| `4339`     | int16 I/Q                            |
| `43455c0`  | int16 I/Q  (Raspberry Pi 3B+/4 typical) |
| `4358`     | packed float, 9 mantissa / 5 exponent bits |
| `4366c0`   | packed float, 12 mantissa / 6 exponent bits |

`--bw` (20/40/80 MHz) sets `nfft` (64/128/256 subcarriers) — this must match
what you configured in `makecsiparams` / your nexmon_csi extraction, or
decoding will silently misalign. If you're not sure, run with `--stats-only`
first: if `estimated_lost`/`duplicate_or_reordered` look insane or frames
mostly fail to decode, try the other bandwidth or chip first.

## Header format

There are two Nexmon CSI UDP header layouts in the wild, depending on which
nexmon_csi fork/version produced the capture:

- **`legacy`** (main seemoo-lab README): 6-byte magic, then MAC/seq/etc.
- **`rssi`** (some newer Pi forks, e.g. zeroby0/nexmonster): 2-byte magic +
  1-byte RSSI + 1-byte Frame Control, *then* MAC/seq/etc. (18-byte header
  instead of 20).

`--header-format auto` (default) detects this per-packet and just works for
either. If you know which one your build uses you can force it with
`--header-format legacy` or `--header-format rssi` — mostly useful for
debugging, since a mismatch fails fast with a clear "bad magic bytes"
message rather than silently misparsing.

## Step 1 — Validate the capture is well-formed

```bash
python main.py --pcap csi_capture.pcap --chip 43455c0 --bw 80 --stats-only
```

This decodes every frame, checks the Nexmon magic bytes / header, and
reports:
- how many packets were seen vs. matched the UDP/CSI filter vs. decoded OK
- estimated lost packets (from gaps in the embedded 802.11 sequence number)
- estimated packet rate

If frames are being skipped, `-v` will log the specific `DecodeError` for
each one. "bad magic bytes" almost always means the wrong `--header-format`
was forced (try `auto`, the default) or, less commonly, wrong chip/bw, or
that you captured something other than the CSI UDP stream on port 5500.

## Step 2 — Look at it

```bash
python main.py --pcap csi_capture.pcap --chip 43455c0 --bw 80 \
    --out-dir results --heatmap --time-plot --variance-plot
```

- `heatmap.png` — packet x subcarrier amplitude. Real CSI should show smooth
  structure across neighboring subcarriers (multipath response of the
  channel), not white noise. A handful of null/pilot subcarriers reading
  near-zero is normal and expected.
- `variance.png` — per-subcarrier std dev. Very uniform, very low variance
  across all subcarriers on a completely static link with nothing moving
  is expected; a single wildly-off subcarrier suggests a decode/alignment
  bug rather than real RF.

## Step 3 — The movement test

This is the actual "does CSI reflect the physical environment" check.

1. **Baseline capture**: room empty/static, nobody moving, capture ~20-30s
   at your normal TX rate (e.g. the 100 Hz `optimized-fast-sender.py`
   pattern in this project).
   ```bash
   sudo tcpdump -i wlan0 udp port 5500 -w baseline.pcap
   ```
2. **Movement capture**: same link, same duration, but someone walks through
   the Tx-Rx path (or you introduce/remove an obstacle) partway through.
   ```bash
   sudo tcpdump -i wlan0 udp port 5500 -w movement.pcap
   ```
3. Run both through the analyzer:
   ```bash
   python main.py --pcap baseline.pcap  --chip 43455c0 --bw 80 --out-dir results/baseline --heatmap --fft --variance-plot
   python main.py --pcap movement.pcap  --chip 43455c0 --bw 80 --out-dir results/movement --heatmap --fft --variance-plot
   ```
4. Compare:
   - **`motion_score`** (printed to console, also in `variance_analysis`):
     should be noticeably higher in `movement` than `baseline`.
   - **Heatmaps**: `movement.png` should show visible time-varying banding/
     ripple where the walk happened; `baseline.png` should look comparatively
     flat and static.
   - **FFT (`--fft`)**: human walking typically shows energy in the
     ~0.5-3 Hz band; breathing (if you try a "sit still and breathe" test
     instead) is closer to ~0.2-0.4 Hz. The `dominant_freq_hz` printed
     should land in a plausible range during movement and be comparatively
     featureless/low-amplitude in the static baseline. Use `--fft-max-freq 5`
     to zoom the plot into that range.

If baseline and movement look statistically indistinguishable, that's a
sign to check antenna placement/orientation, TX rate (needs to be fast
enough relative to the motion you're trying to catch — Nyquist), or whether
you're actually capturing on the right interface/channel.

## Diagnosing sample-rate/timing artifacts

`main.py --fft` (and `fft_motion_detection` under the hood) runs its FFT
*by packet index*, assuming packets arrived evenly spaced at
`dataset.fs_est` (the *average* rate over the whole capture). If a capture
actually has meaningful packet loss, especially bursty loss rather than
loss spread evenly through the capture, the real time axis is irregular
and that irregularity can alias into a spurious "dominant frequency" that
has nothing to do with anything physical in the room.

`check_resample_artifact.py` isolates that question on a single pcap by
computing the dominant frequency two ways and comparing them:

```bash
python check_resample_artifact.py --pcap test/baseline.pcap \
    --chip 43455c0 --bw 80 --bandpass-low 0.7 --bandpass-high 2.5
```

- **"naive"** — exactly what `main.py --fft` does: index-based FFT at
  `fs_est`, ignoring real timestamps.
- **"resampled"** — the same signal linearly interpolated onto a *true*
  uniform time grid first (`align_and_baseline.resample_uniform`), so gaps
  are handled honestly, then FFT'd.

If the two dominant frequencies agree closely (within roughly one FFT bin,
`1/capture_duration`), the naive result was trustworthy for that capture —
whatever frequency you're seeing is real, not a loss artifact. If they
diverge substantially, packet loss is distorting the naive FFT, and you
should either trust the resampled number or get a lower-loss capture
before drawing conclusions from `main.py --fft`'s output.

This is worth running any time you're chasing an unexplained frequency
(like a mystery peak that shows up regardless of what's actually happening
in the room) before spending time hunting for a physical/RF cause — rule
out "it's just how the tool handles packet loss" first. It's also useful
sanity-checking for a good, low-loss capture: agreement there confirms the
frequency reading is meaningful before you rely on it for a notch (see
below) or for interpreting `--fft` results in general.

Note this script duplicates a slice of `main.py`'s amplitude pipeline
(decode → fftshift → amplitude → bandpass) rather than reusing it directly,
and averages across subcarriers *before* filtering rather than after —
so don't expect its frequency numbers to match `main.py --fft` exactly
even on the same pcap; it's meant for the naive-vs-resampled comparison,
not as a drop-in replacement for `main.py`.

## Cleaning noisy data

```bash
python main.py --pcap movement.pcap --chip 43455c0 --bw 80 \
    --out-dir results/movement --hampel --bandpass-low 0.7 --bandpass-high 2.5 \
    --plot-pipeline --quality-top-k 10 --fft --heatmap
```

This is the new recommended heartbeat-focused preprocessing pipeline:

- `--hampel` removes isolated spike artifacts per-subcarrier before other
  filtering.
- `--bandpass-low` / `--bandpass-high` apply a zero-phase Butterworth
  band-pass tuned for heartbeat-like motion (default `0.7-2.5 Hz`, which is
  a good first pass for the 45-120 BPM range).
- The pipeline then removes slow DC drift and normalizes each subcarrier so
  high-power carriers do not dominate downstream analysis.
- `--plot-pipeline` saves a stage-by-stage plot (`raw → Hampel → bandpass →
  detrended → normalized`) for one selected subcarrier, which is extremely
  useful for debugging when the signal looks weak or unstable.
- `--quality-top-k` and `--quality-threshold` rank carriers with the new
  subcarrier-quality scorer and keep only the most informative ones.

## Suppressing a known interferer (notch filter)

If `--fft` (or `check_resample_artifact.py`, once you've ruled out a
sampling-rate artifact) turns up a stable, narrowband frequency that shows
up identically across captures regardless of what's actually happening in
the room — survives router-off, fan-off, and relocating the Pis — that's a
real but unwanted narrowband signal sitting in your capture, and it can be
suppressed with a notch filter before the heartbeat bandpass:

```bash
python main.py --pcap movement.pcap --chip 43455c0 --bw 80 \
    --out-dir results/movement --hampel --notch-hz 2.0 --notch-q 15 \
    --bandpass-low 0.7 --bandpass-high 2.5 --fft --heatmap
```

This applies `filters.notch_filter` (a zero-phase IIR notch,
`scipy.signal.iirnotch` + `filtfilt`) in the pipeline right after `--hampel`
and before the bandpass. Unlike the bandpass, which keeps only a range and
discards everything else, the notch keeps everything and removes only a
thin slice around the target frequency — use it to clean an interferer out
before the bandpass does its normal job, not as a substitute for the
bandpass.

### How `--notch-hz` works

- Pass a single frequency in Hz, e.g. `--notch-hz 2.0`, to suppress a peak
  at 2 Hz.
- Pass a comma-separated list, e.g. `--notch-hz 2.0,4.0`, to suppress
  several frequencies (each notch is applied in sequence) — useful if the
  interferer has a harmonic.
- Each frequency must be strictly between 0 and the Nyquist frequency
  (`fs_est / 2`) for the capture; the CLI will error out with the valid
  range if not.
- **Default: off.** If `--notch-hz` isn't passed, no notch filter runs at
  all — the pipeline behaves exactly as it did before this option existed.
  There's no separate flag needed to disable it; simply omit `--notch-hz`.

### How `--notch-q` works

`--notch-q` sets the filter's Q factor (`center_frequency / bandwidth`),
which controls how narrow the cut is:

- **Higher Q → narrower notch.** Removes less of your real signal on
  either side of the target frequency, but only cuts a very tight band —
  needs the interferer to be quite frequency-stable across captures to be
  effective.
- **Lower Q → wider notch.** More tolerant of the interferer drifting a
  bit between captures (this project's own investigation found an
  unexplained ~2 Hz peak that moved by as much as ~0.2 Hz between two
  back-to-back captures under identical conditions), at the cost of also
  suppressing more of any real signal that happens to fall near that
  frequency.
- **Default: `30`**, which gives roughly a `center_freq / 30` Hz-wide notch
  (e.g. ~0.07 Hz wide at a 2 Hz notch — quite tight). If the interferer
  you're targeting drifts noticeably between runs (check by comparing
  `dominant_freq_hz` across a few captures first), lower this — something
  in the `10-15` range is a reasonable starting point for a drifting ~2 Hz
  peak specifically.
- If in doubt, run with `--fft` before and after adding the notch and
  compare `fft_motion.png` — the target peak should visibly shrink or
  disappear. If it doesn't move much, lower `--notch-q` and try again
  before assuming the notch isn't working.

### How to bypass/disable it

The notch filter is opt-in and off by default — just don't pass
`--notch-hz` and nothing changes versus the standard pipeline. If you've
been running with a notch and want to go back to unfiltered output for a
specific run (e.g. to re-check whether the interferer is still present
before deciding whether to keep notching it out), drop `--notch-hz` from
the command entirely rather than passing an empty value:

```bash
# with notch
python main.py --pcap movement.pcap --chip 43455c0 --bw 80 \
    --out-dir results/movement --notch-hz 2.0 --fft

# without -- just omit --notch-hz, everything else identical
python main.py --pcap movement.pcap --chip 43455c0 --bw 80 \
    --out-dir results/movement --fft
```

`--notch-q` has no effect on its own — it's only read when `--notch-hz` is
also set, so there's nothing to unset separately.

## Ground-truth-first quality testing

Before relying on the new subcarrier-quality ranking on its own, the safest
first test is to validate it against a ground-truth heartbeat reference.
That is the best way to answer: "are these selected carriers actually
carrying the heartbeat signal, or do they just look smooth/noisy?"

Recommended workflow:

1. Record CSI and ground truth at the same time (for example, use
   `live_monitor.py` for CSI and `heartbeat-giop.py` for the reference).
2. Run the new preprocessing pipeline with the heart-rate band-pass and
   subcarrier scoring enabled.
3. Compare the selected subcarriers against the ground-truth signal using the
   quality scorer in `quality.py` (it accepts an optional ground-truth array
   for correlation-based scoring) and the baseline heartbeat check in
   `align_and_baseline.py`.
4. Only after the ranking is stable and correlated with the ground truth
   should you trust the selected carriers for ML feature extraction.

A good first command is:

```bash
python main.py --pcap movement.pcap --chip 43455c0 --bw 80 \
    --out-dir results/movement --hampel --bandpass-low 0.7 --bandpass-high 2.5 \
    --plot-pipeline --quality-top-k 10 --fft --heatmap
```

If the selected carriers do not line up with the heartbeat reference, reduce
confidence in the ranking and return to the plot/debugging flow before moving
on to ML.

## Exporting for ML

```bash
python main.py --pcap movement.pcap --chip 43455c0 --bw 80 \
    --out-dir results/movement --save-npy --save-csv amplitude
```

- `--save-npy` writes `csi_dataset.npz` with the full complex CSI matrix,
  timestamps, and sequence numbers (`np.load(...)`, keys: `csi`, `ts`,
  `seq_num`, `nfft`, `fs_est`) — round-trips exactly, use this for anything
  serious.
- `--save-csv {complex,amplitude,phase}` writes a flat CSV, one row per
  packet, for quick inspection in a spreadsheet or for pipelines that don't
  want a NumPy dependency.

## Useful flags reference

Run `python main.py --help` for the full list; the ones worth knowing about:

- `--subcarriers 10,20,30` — focus time-series/FFT plots on specific
  subcarrier indices (indexed into the DC-centered/fftshifted axis) instead
  of the mean over all of them. Good once you've found a subcarrier that's
  particularly sensitive to your test motion.
- `--exclude-dc-guard` — zero out DC and edge guard subcarriers before
  analysis (these carry no real channel information and can skew variance).
- `--src-ip ''` — disable the `10.10.10.10` source-IP filter if your setup
  doesn't follow that nexmon_csi convention.
- `--notch-hz 2.0` (with `--notch-q`) — suppress a known narrowband
  interferer before the heartbeat bandpass; off by default, just omit it
  to bypass. See [Suppressing a known interferer](#suppressing-a-known-interferer-notch-filter).
- `check_resample_artifact.py` — separate script (not a `main.py` flag) for
  telling a real frequency apart from a packet-loss/resampling artifact.
  See [Diagnosing sample-rate/timing artifacts](#diagnosing-sample-ratetiming-artifacts).

## Reading the loss/stream diagnostics

`estimated_lost` is computed from the pure 12-bit 802.11 sequence number
(the raw Sequence Control field has a 4-bit fragment number packed into the
low bits, which the decoder strips out). If this number looks enormous
relative to `n_packets`, double check you're not on an old copy of this
tool -- an earlier version of this decoder had that units bug and reported
~15x inflated loss.

The stats output also prints a `core`/`spatial_stream` breakdown. If your
nexmon_csi config reports more than one core or spatial stream, you'll get
multiple interleaved combinations in one capture -- each with its own
baseline amplitude. Averaging across all of them (the default) produces a
distinctive bimodal/banded pattern in time-series plots that has nothing to
do with real motion. Use `--select-core N --select-stream M` to isolate one
combination before doing time-series/FFT/motion analysis. When multiple
streams share a capture, `duplicate_or_reordered` will also look inflated,
since packets from different streams triggered by the same underlying
Wi-Fi frame legitimately carry the same sequence number.

## Live monitoring

For a real-time terminal dashboard instead of capture-then-analyze:

```bash
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80
```

Live view with filter

```bash
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80 --hampel
```

(Needs root/`CAP_NET_RAW`, same as `tcpdump` -- it's a live raw capture.)

Shows, updated live:
- packets seen/decoded/errors and live packet rate
- the latest packet's sequence number, core/stream, RSSI, chanspec
- current mean amplitude and phase
- a scrolling amplitude sparkline (auto-scaled trend over the last N packets)
- a per-subcarrier sparkline of the *current* packet (a quick "shape" check --
  should look similar in shape to your MATLAB `plotcsi` traces)
- a rolling motion score, compared against an early baseline, with a
  **`** MOTION / CHANGE DETECTED **`** flag once it exceeds
  `--motion-threshold` times the baseline

Press `q` to quit (or Ctrl+C).

Useful flags:
- `--select-core N --select-stream M` -- if your capture interleaves
  multiple streams (see the "Core / spatial-stream breakdown" note above),
  restrict the live view to just one for a clean signal.
- `--window` -- how many recent packets the motion score and sparkline
  are computed over (default 100).
- `--motion-threshold` -- how many times over baseline the motion score
  needs to be to trigger the alert (default 1.5x).
- `--src-ip ''` -- disable the `10.10.10.10` source filter if needed.

This reuses the exact same `decoder.decode_frame_payload()` as `main.py`,
so any chip/header-format/bandwidth settings that work for your pcap
captures will work here too.

### Recording live CSI (for ML)

Press `c` in the dashboard to start recording every decoded CSI packet to a
timestamped CSV; press `c` again to stop. This lets you capture data
live -- no separate tcpdump-then-decode step -- and, by default, writes to
the same `output/` folder that `heartbeat-giop.py` writes
`ground_truth.csv` to, so both data streams land side by side, ready to be
paired up for the ML step:

```bash
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80
# press 'c' to start recording, 'c' again to stop, 'q' to quit
```

While active, the dashboard shows a bolded status line with the current
filename and frame count (`** CAPTURING ** -> csi_capture_20260728_101500.csv
  frames written: 842`); while idle it shows where the next capture will go.

File format, one row per packet:

```
ts, seq_num, core, spatial_stream, sc0_amp, sc1_amp, ..., sc{nfft-1}_amp
```

- **`ts`** is a raw `time.time()` epoch timestamp formatted to 3 decimals --
  the exact same convention `heartbeat-giop.py` uses for `ground_truth.csv`
  -- so the two streams can be aligned/interpolated by timestamp afterward.
- Subcarriers are written in **raw decode order (not fftshifted)**, matching
  `utils.save_csv()` / `main.py --save-csv`, so a live capture lines up
  column-for-column with an offline pcap export.
- Files are named `csi_capture_<YYYYMMDD_HHMMSS>.csv`; each `c`-start opens
  a new file rather than appending, so separate recording sessions in the
  same run don't get mixed together.

Useful flags:
- `--output-dir` -- where captures are written (default: `output/` next to
  `live_monitor.py`, matching `heartbeat-giop.py`).
- `--capture-mode {amplitude,phase,complex}` -- representation to record per
  subcarrier (default `amplitude`; `complex` keeps full I/Q if you need
  phase for a later step).

## Ground truth data

```bash
python3 heartbeat-giop.py
```

Use this to record timed heart-rate data to output directory. 

## Baseline heart-rate check

Once you have both a CSI capture (`output/csi_capture_*.csv` from
`live_monitor.py`) and a ground-truth recording (`output/ground_truth.csv`
from `heartbeat-giop.py`) covering the same session, `align_and_baseline.py`
answers the key question before you build anything fancier: **does the CSI
signal actually contain a usable heart-rate signature?**

```bash
python align_and_baseline.py \
    --csi-csv output/csi_capture_20260728_101500.csv \
    --gt-csv output/ground_truth.csv \
    --out-dir results/baseline_ml
```

What it does:

1. Loads both CSVs and aligns them on their shared `time.time()` timestamp
   column (checks they actually overlap in time first).
2. Resamples the (possibly slightly irregular) CSI signal onto a uniform
   time grid at its estimated sample rate.
3. Band-pass filters that signal to the human heart-rate band (default
   0.5-3.7 Hz == 30-220 BPM, matching `heartbeat-giop.py`'s
   `MIN_BPM`/`MAX_BPM`) using `filters.butterworth_bandpass`, to knock out
   breathing and slow motion artifacts.
4. Slides a window (default 8s, 50% overlap) over the filtered signal, FFTs
   each one (Hanning-windowed, same idea as `analysis.fft_motion_detection`),
   and takes the dominant in-band frequency -- refined with 3-point
   parabolic interpolation around the peak bin -- as the CSI-estimated BPM.
5. Matches each window against the ground-truth beats that fall inside it
   (or the nearest beat, if none fall inside) and reports **MAE, RMSE, bias,
   and correlation** against the true BPM.

This is intentionally *not* the ML model itself -- it's the sanity check
that tells you whether the raw signal-processing approach is anywhere
close, which is much easier to debug than a model that silently learns
nothing useful.

Outputs, written to `--out-dir` (default `results/baseline_ml`):

- **`baseline_bpm_comparison.png`** -- estimated vs. ground-truth BPM over
  time, plus a parity (scatter) plot against the `y = x` line.
- **`baseline_results.csv`** -- one row per window
  (`t_start, t_end, t_center, est_bpm, est_magnitude, true_bpm, n_gt_beats`).
  These are directly reusable as `(feature, label)` pairs for a later ML
  step, so you don't need to redo the alignment/windowing work.

Rule of thumb printed alongside the metrics: MAE/RMSE well under ~5-10 BPM
with correlation trending toward 1.0 means the raw FFT approach is already
picking up a real cardiac signal, and it's worth moving on to an ML model
for refinement. Errors comparable to the BPM range itself, or near-zero
correlation, mean the signal isn't there yet at this chip/bw/placement/
window setting -- worth checking those before any model will help.

### Amplitude vs. phase (`--signal-type`)

By default the script uses **amplitude** (`sc*_amp` columns, or the
magnitude of `sc*_re`/`sc*_im`). Heartbeat-scale chest-wall displacement is
sub-millimeter, though, which is usually swamped in amplitude by breathing
and body sway. If your amplitude-based correlation is low (and especially
if `correlation_vs_subcarrier.png` shows a smooth, near-monotonic trend
across the whole subcarrier range rather than a noisy scatter -- a sign
you're picking up systematic drift, not real per-tone channel modulation),
it's worth trying **phase** instead:

```bash
# 1. record a fresh capture with full I/Q (phase needs this -- an
#    amplitude-only or raw sc*_pha capture doesn't carry enough
#    information to remove the SFO/CFO phase ramp)
sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80 \
    --capture-mode complex
# press 'c' to start/stop recording, 'q' to quit

# 2. run the baseline check on phase instead of amplitude
python align_and_baseline.py \
    --csi-csv output/csi_capture_<new_timestamp>.csv \
    --gt-csv output/ground_truth.csv \
    --signal-type phase \
    --search-subcarriers \
    --out-dir results/baseline_ml_phase
```

`--signal-type phase` unwraps phase across the subcarrier axis per packet,
then removes the linear SFO/CFO ramp with a per-packet least-squares fit
against subcarrier index (`align_and_baseline.sanitize_phase`) -- what's
left is (approximately) the true channel phase, everything downstream
(resampling, band-pass, windowed FFT, ground-truth matching, plots) is
identical to the amplitude path.

Useful flags:
- `--signal-type {amplitude,phase}` -- amplitude (default) or sanitized
  phase; see above. `phase` requires a `--capture-mode complex` CSV.
- `--subcarrier N` -- use a single subcarrier's signal (index into the
  CSV's own columns) instead of the mean across all of them; useful once
  you've found a subcarrier with better SNR.
- `--search-subcarriers` -- evaluate every subcarrier plus the average of
  the top-ranked groups (top 1/3/5/10/20), and write out a ranked
  `subcarrier_metrics.csv`, `correlation_vs_subcarrier.png`, and
  `mae_vs_subcarrier.png` alongside a `summary.txt` recommendation.
- `--window-s` / `--overlap` -- FFT window length and overlap (default 8s,
  50%).
- `--no-fft-interp` -- disable the parabolic peak interpolation and use
  the raw FFT bin instead. Interpolation is on by default since a raw bin
  pick quantizes BPM estimates to `60/window_s` Hz steps (7.5 BPM at the
  default 100 Hz / 8s window) regardless of how clean the signal is.
- `--low-hz` / `--high-hz` -- band-pass edges in Hz (default 0.5/3.7, i.e.
  30-222 BPM).

### Searching all subcarriers

`--search-subcarriers` evaluates every subcarrier individually (ranked by
a `corr / (1 + mae)` score) and also the average of the top 1/3/5/10/20
performers, which is usually a better-conditioned signal than any single
subcarrier alone:

```bash
python align_and_baseline.py \
  --csi-csv output/csi_capture.csv \
  --gt-csv output/ground_truth.csv \
  --search-subcarriers \
  --out-dir results/baseline_ml
```

Same flag works with `--signal-type phase` -- run both and compare the two
`correlation_vs_subcarrier.png` plots. A noisier, less monotonic-looking
curve with a higher peak correlation for phase is the signature that
phase is picking up real per-tone heartbeat modulation that amplitude was
missing, rather than the same systematic drift artifact.
```