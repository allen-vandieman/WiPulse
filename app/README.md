# csi_analysis

Toolkit for decoding and analyzing [Nexmon CSI](https://github.com/seemoo-lab/nexmon_csi)
captures (`sudo tcpdump -i wlan0 udp port 5500 -w csi_capture.pcap`) and
validating that the extracted CSI actually reflects movement/obstacles in
the environment.

## Content

1. [Layout](#layout)
1. [Setup](#setup)
1. [Chip](#chip--bandwidth)
1. [Header format](#header-format)
1. [Validate the capture](#step-1--validate-the-capture-is-well-formed)
1. [Look at it](#step-2--look-at-it)
1. [Movement test](#step-3--the-movement-test)
1. [Cleaning noisy data](#cleaning-noisy-data)
1. [Exporting for ML](#exporting-for-ml)
1. [Flags reference](#useful-flags-reference)
1. [Loss/stream diagnostics](#reading-the-lossstream-diagnostics)
1. [Live monitoring](#live-monitoring)
1. [Recording live CSI (for ML)](#recording-live-csi-for-ml)
1. [Ground Truth](#ground-truth-data)

## Layout

```
csi_analysis/
├── main.py          # CLI entry point
├── decoder.py        # PCAP parsing + Nexmon header/CSI decoding (int16 & float formats)
├── analysis.py        # Amplitude/phase, subcarrier selection, variance, FFT motion detection
├── plotting.py        # Time plots, heatmaps, FFT plots (headless-safe)
├── filters.py          # Hampel filter, Butterworth low-pass/band-pass
├── utils.py             # Save/load to .npz / .csv, logging setup
├── requirements.txt
└── example_pcap/        # put test captures here
```

## Setup

```bash
# Create an enviroment
python3 -m venv csi_env

# Activate the enviroment
source test/bin/activate

# Install pre-requirements to enviroment
pip install -r requirements.txt
```

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

## Cleaning noisy data

```bash
python main.py --pcap movement.pcap --chip 43455c0 --bw 80 \
    --out-dir results/movement --hampel --lowpass-hz 10 --fft --heatmap
```

- `--hampel` knocks out isolated spike artifacts per-subcarrier before
  anything else runs.
- `--lowpass-hz` applies a zero-phase Butterworth low-pass (needs a valid
  estimated packet rate from your capture timestamps, printed in Step 1).

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