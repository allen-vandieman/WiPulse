#!/usr/bin/env python3

"""
live_monitor.py
===============

Real-time terminal dashboard for Nexmon CSI streams.

In addition to packet statistics, amplitude/phase, motion score, and
key-triggered CSI capture, this version estimates heart rate (BPM) from
the live CSI stream.

Live heartbeat pipeline
-----------------------

    Raw complex CSI
        |
        v
    Detect problematic subcarriers
        |
        v
    Interpolate problematic subcarriers
        |
        v
    Extract amplitude
        |
        v
    Hampel filter
        |
        v
    Optional notch filter
        |
        v
    Heartbeat bandpass
        |
        v
    Drift removal
        |
        v
    Normalization
        |
        v
    Welch PSD
        |
        v
    Dominant heartbeat frequency
        |
        v
    Estimated BPM

The BPM estimator uses a rolling time window rather than trying to estimate
heart rate from individual packets.

Usage
-----

    sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80

Useful options:

    --hr-window 30
        Number of seconds of CSI used for each BPM estimate.

    --hr-update 2
        Recalculate BPM every N seconds.

    --hr-low 0.7
        Lower heartbeat frequency limit in Hz (42 BPM).

    --hr-high 2.5
        Upper heartbeat frequency limit in Hz (150 BPM).

    --hr-hampel-window 51
        Hampel window used by the heartbeat pipeline.

    --hr-hampel-sigmas 3.0
        Hampel threshold.

    --hr-notch 0
        Optional notch frequency in Hz. 0 disables notch filtering.

Press 'q' to quit.
Press 'c' to start/stop recording decoded CSI to a timestamped CSV.
"""

from __future__ import annotations

import argparse
import csv
import curses
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from decoder import (
    decode_frame_payload,
    DecodeError,
    CHIP_TABLE,
    BW_TO_NFFT,
)

from filters import (
    hampel_filter,
    butterworth_bandpass,
    notch_filter,
    remove_dc_drift,
    normalize_signal,
    FilterConfig,
    CSIFilter,
)

try:
    from scipy.signal import welch
except ImportError:
    welch = None


logger = logging.getLogger("csi_analysis.live_monitor")

SPARK_CHARS = "▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------------------
# CSI CAPTURE
# ---------------------------------------------------------------------------

class CSICapture:
    """
    Handles writing live CSI frames to CSV, toggled on/off with the 'c' key.
    """

    def __init__(
        self,
        out_dir: Path,
        nfft: int,
        mode: str = "amplitude",
        flush_every: int = 20,
    ):
        self.out_dir = out_dir
        self.nfft = nfft
        self.mode = mode
        self.flush_every = max(1, flush_every)

        self.lock = threading.Lock()
        self.file = None
        self.writer = None
        self.path: Optional[Path] = None
        self.n_written = 0
        self.active = False

    def _header(self):
        header = [
            "ts",
            "seq_num",
            "core",
            "spatial_stream",
        ]

        if self.mode == "complex":
            for i in range(self.nfft):
                header += [
                    f"sc{i}_re",
                    f"sc{i}_im",
                ]

        else:
            suffix = "amp" if self.mode == "amplitude" else "pha"
            header += [
                f"sc{i}_{suffix}"
                for i in range(self.nfft)
            ]

        return header

    def start(self) -> Path:
        """Open a new timestamped CSV file and start writing."""
        with self.lock:
            if self.active:
                return self.path

            self.out_dir.mkdir(exist_ok=True)

            ts_tag = time.strftime("%Y%m%d_%H%M%S")

            self.path = (
                self.out_dir /
                f"csi_capture_{ts_tag}.csv"
            )

            self.file = open(
                self.path,
                "w",
                newline="",
            )

            self.writer = csv.writer(self.file)
            self.writer.writerow(self._header())

            self.n_written = 0
            self.active = True

            return self.path

    def stop(self):
        """Flush and close the current file."""
        with self.lock:
            if not self.active:
                return

            self.active = False

            if self.file is not None:
                self.file.flush()
                self.file.close()

            self.file = None
            self.writer = None

    def toggle(self):
        if self.active:
            self.stop()
            return False

        self.start()
        return True

    def write_frame(self, frame):
        """Write one decoded CSI frame as a CSV row."""
        with self.lock:
            if not self.active or self.writer is None:
                return

            row = [
                f"{frame.ts:.3f}",
                int(frame.seq_num),
                int(frame.core),
                int(frame.spatial_stream),
            ]

            if self.mode == "complex":

                for v in frame.csi:
                    row += [
                        v.real,
                        v.imag,
                    ]

            elif self.mode == "amplitude":

                row += list(
                    np.abs(frame.csi)
                )

            else:

                row += list(
                    np.angle(frame.csi)
                )

            self.writer.writerow(row)

            self.n_written += 1

            if self.n_written % self.flush_every == 0:
                self.file.flush()

    def close(self):
        self.stop()


# ---------------------------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------------------------

def sparkline(values, width: int = 40) -> str:
    """ASCII/unicode sparkline, auto-scaled to min/max."""
    values = list(values)[-width:]

    if not values:
        return " " * width

    lo = min(values)
    hi = max(values)

    span = hi - lo if hi > lo else 1.0

    chars = [
        SPARK_CHARS[
            int(
                (v - lo)
                / span
                * (len(SPARK_CHARS) - 1)
            )
        ]
        for v in values
    ]

    return "".join(chars).rjust(width)


# ---------------------------------------------------------------------------
# HEART RATE ESTIMATOR
# ---------------------------------------------------------------------------

class HeartRateEstimator:
    """
    Rolling live CSI heart-rate estimator.

    CSI is stored as complex data so that the CSI quality-cleaning stage
    can operate before amplitude extraction.

    The estimator deliberately uses a relatively long rolling window because
    heartbeat is a low-frequency signal and a short window produces unstable
    PSD estimates.
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        update_seconds: float = 2.0,
        low_hz: float = 0.7,
        high_hz: float = 2.5,
        hampel_window: int = 51,
        hampel_sigmas: float = 3.0,
        notch_hz: float = 0.0,
        drift_window: int = 21,
    ):
        self.window_seconds = float(window_seconds)
        self.update_seconds = float(update_seconds)

        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)

        self.hampel_window = int(hampel_window)
        self.hampel_sigmas = float(hampel_sigmas)

        self.notch_hz = float(notch_hz)
        self.drift_window = int(drift_window)

        self.frames = deque()
        self.timestamps = deque()

        self.lock = threading.Lock()

        self.last_update = 0.0

        self.bpm: Optional[float] = None
        self.confidence: Optional[float] = None
        self.peak_hz: Optional[float] = None

        self.status = "warming up"

        self.last_sample_rate: Optional[float] = None

        # We create the CSI cleaner once rather than constructing it for
        # every heartbeat calculation.
        try:
            config = FilterConfig(
                sampling_rate=100.0,
                interpolate_problematic=True,
                apply_notch=False,
                apply_hampel=False,
            )

            self.csi_filter = CSIFilter(config)

        except Exception as exc:
            logger.warning(
                "Could not initialize CSIFilter: %s",
                exc,
            )

            self.csi_filter = None

    # ------------------------------------------------------------------
    # FRAME INPUT
    # ------------------------------------------------------------------

    def add_frame(self, frame):
        """
        Add one CSI frame.

        Frames are retained according to timestamp rather than simply
        packet count so packet-rate variations do not change the intended
        time window.
        """

        now = float(frame.ts)

        with self.lock:

            self.frames.append(
                np.asarray(frame.csi, dtype=np.complex128)
            )

            self.timestamps.append(now)

            cutoff = now - self.window_seconds

            while (
                self.timestamps
                and self.timestamps[0] < cutoff
            ):
                self.timestamps.popleft()
                self.frames.popleft()

    # ------------------------------------------------------------------
    # SAMPLE RATE
    # ------------------------------------------------------------------

    def _estimate_sample_rate(self, timestamps):
        """
        Estimate actual CSI sample rate from packet timestamps.
        """

        if len(timestamps) < 2:
            return None

        ts = np.asarray(
            timestamps,
            dtype=np.float64,
        )

        dt = np.diff(ts)

        # Remove clearly broken intervals caused by packet gaps.
        dt = dt[
            np.isfinite(dt)
            & (dt > 0)
        ]

        if len(dt) < 10:
            return None

        # Median is more robust than mean when packet loss occurs.
        median_dt = float(
            np.median(dt)
        )

        if median_dt <= 0:
            return None

        return 1.0 / median_dt

    # ------------------------------------------------------------------
    # CSI QUALITY CLEANING
    # ------------------------------------------------------------------

    def _clean_csi(self, csi):
        """
        Apply the CSI-level cleaning stage.

        The unified filters.py contains the CSI-specific detection and
        interpolation logic. If that API is unavailable for some reason,
        the raw CSI is retained rather than crashing the live dashboard.
        """

        if self.csi_filter is None:
            return csi

        try:
            cleaned = self.csi_filter.clean_csi(csi)

            if cleaned is not None:
                return np.asarray(
                    cleaned,
                    dtype=np.complex128,
                )

        except Exception as exc:
            logger.debug(
                "CSI cleaning failed: %s",
                exc,
            )

        return csi

    # ------------------------------------------------------------------
    # SIGNAL EXTRACTION
    # ------------------------------------------------------------------

    def _extract_best_signal(self, csi):
        """
        Extract a heartbeat candidate from the CSI matrix.

        For the live dashboard we first clean the CSI and then calculate
        amplitude.

        We use the median across subcarriers rather than blindly using
        every subcarrier equally. This reduces the influence of extreme
        subcarriers while retaining the common heartbeat variation.
        """

        amp = np.abs(csi)

        if amp.ndim != 2:
            return None

        # Remove subcarriers that contain almost no useful variation.
        sc_std = np.std(
            amp,
            axis=0,
        )

        sc_mean = np.mean(
            amp,
            axis=0,
        )

        valid = (
            np.isfinite(sc_std)
            & np.isfinite(sc_mean)
            & (sc_mean > 1e-9)
        )

        if np.count_nonzero(valid) < 3:
            return None

        valid_amp = amp[:, valid]

        # Normalize each subcarrier by its own average amplitude.
        #
        # This prevents large-amplitude subcarriers from dominating the
        # combined heartbeat signal.
        valid_mean = np.mean(
            valid_amp,
            axis=0,
        )

        normalized = (
            valid_amp
            / np.maximum(valid_mean, 1e-12)
        )

        # Median across valid subcarriers gives a robust common-mode
        # heartbeat signal.
        signal = np.median(
            normalized,
            axis=1,
        )

        return np.asarray(
            signal,
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # HEARTBEAT PIPELINE
    # ------------------------------------------------------------------

    def _process_signal(
        self,
        signal,
        fs,
    ):
        """
        Apply the recommended heartbeat signal-processing pipeline.
        """

        signal = np.asarray(
            signal,
            dtype=np.float64,
        )

        if len(signal) < 100:
            return None

        if not np.all(
            np.isfinite(signal)
        ):
            signal = np.nan_to_num(
                signal,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        # --------------------------------------------------------------
        # 1. Hampel filter
        # --------------------------------------------------------------

        try:
            signal = hampel_filter(
                signal,
                window_size=self.hampel_window,
                n_sigmas=self.hampel_sigmas,
            )
        except Exception:
            pass

        # --------------------------------------------------------------
        # 2. Optional notch
        # --------------------------------------------------------------

        if self.notch_hz > 0:

            try:
                signal = notch_filter(
                    signal,
                    fs=fs,
                    frequencies=self.notch_hz,
                    q=30.0,
                )
            except TypeError:

                # Compatibility with possible alternate parameter name.
                try:
                    signal = notch_filter(
                        signal,
                        fs=fs,
                        frequencies=[
                            self.notch_hz
                        ],
                        Q=30.0,
                    )
                except Exception:
                    pass

            except Exception:
                pass

        # --------------------------------------------------------------
        # 3. Heartbeat bandpass
        # --------------------------------------------------------------

        try:
            signal = butterworth_bandpass(
                signal,
                fs=fs,
                low_hz=self.low_hz,
                high_hz=self.high_hz,
                order=4,
            )

        except TypeError:

            # Compatibility fallback for alternate argument names.
            try:
                signal = butterworth_bandpass(
                    signal,
                    fs=fs,
                    low=self.low_hz,
                    high=self.high_hz,
                    order=4,
                )
            except Exception:
                return None

        except Exception:
            return None

        # --------------------------------------------------------------
        # 4. Remove slow drift
        # --------------------------------------------------------------

        try:
            signal = remove_dc_drift(
                signal,
                window_size=self.drift_window,
            )

        except TypeError:

            try:
                signal = remove_dc_drift(
                    signal,
                    window=self.drift_window,
                )
            except Exception:
                pass

        except Exception:
            pass

        # --------------------------------------------------------------
        # 5. Normalize
        # --------------------------------------------------------------

        try:
            signal = normalize_signal(
                signal
            )
        except Exception:

            std = np.std(signal)

            if std > 1e-12:
                signal = (
                    signal - np.mean(signal)
                ) / std
            else:
                return None

        if not np.all(
            np.isfinite(signal)
        ):
            return None

        return signal

    # ------------------------------------------------------------------
    # PSD / BPM
    # ------------------------------------------------------------------

    def _estimate_bpm(self, signal, fs):
        """
        Estimate BPM from the dominant heartbeat-band PSD peak.
        """

        if welch is None:
            self.status = "scipy.signal.welch unavailable"
            return None, None, None

        n = len(signal)

        if n < max(
            200,
            int(fs * 10),
        ):
            return None, None, None

        # Use roughly 8-second Welch segments.
        nperseg = min(
            n,
            max(
                256,
                int(fs * 8),
            ),
        )

        if nperseg >= n:
            nperseg = n

        try:
            freqs, psd = welch(
                signal,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=nperseg // 2,
                detrend="constant",
                scaling="density",
            )
        except Exception as exc:
            logger.debug(
                "Welch PSD failed: %s",
                exc,
            )
            return None, None, None

        band = (
            (freqs >= self.low_hz)
            & (freqs <= self.high_hz)
            & np.isfinite(psd)
        )

        if not np.any(band):
            return None, None, None

        band_freqs = freqs[band]
        band_psd = psd[band]

        if len(band_psd) == 0:
            return None, None, None

        peak_idx = int(
            np.argmax(band_psd)
        )

        peak_power = float(
            band_psd[peak_idx]
        )

        peak_hz = float(
            band_freqs[peak_idx]
        )

        bpm = peak_hz * 60.0

        # --------------------------------------------------------------
        # Confidence estimate
        # --------------------------------------------------------------
        #
        # Compare the strongest PSD peak with the median power in the
        # heartbeat band. A clean heartbeat should produce a relatively
        # strong narrow peak.
        #

        median_power = float(
            np.median(band_psd)
        )

        if median_power > 1e-12:
            peak_ratio = (
                peak_power
                / median_power
            )
        else:
            peak_ratio = 0.0

        # Convert the ratio into a simple 0-100 confidence scale.
        confidence = np.clip(
            (peak_ratio - 1.0) / 9.0 * 100.0,
            0.0,
            100.0,
        )

        return (
            float(bpm),
            float(confidence),
            peak_hz,
        )

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def update(self, force=False):
        """
        Recalculate BPM if enough time has passed.

        Returns the current BPM.
        """

        now = time.time()

        if (
            not force
            and now - self.last_update
            < self.update_seconds
        ):
            return self.bpm

        with self.lock:

            if len(self.frames) < 100:
                self.status = "warming up"
                return self.bpm

            frames = np.stack(
                list(self.frames),
                axis=0,
            )

            timestamps = np.asarray(
                self.timestamps,
                dtype=np.float64,
            )

        duration = (
            timestamps[-1]
            - timestamps[0]
        )

        if duration < self.window_seconds * 0.6:
            self.status = (
                f"warming up "
                f"({duration:.1f}/{self.window_seconds:.0f}s)"
            )
            return self.bpm

        fs = self._estimate_sample_rate(
            timestamps
        )

        if fs is None:
            self.status = "estimating sample rate"
            return self.bpm

        self.last_sample_rate = fs

        # --------------------------------------------------------------
        # CSI cleaning
        # --------------------------------------------------------------

        cleaned_csi = self._clean_csi(
            frames
        )

        # --------------------------------------------------------------
        # Extract common amplitude heartbeat signal
        # --------------------------------------------------------------

        signal = self._extract_best_signal(
            cleaned_csi
        )

        if signal is None:
            self.status = "no usable CSI signal"
            return self.bpm

        # --------------------------------------------------------------
        # Signal-processing pipeline
        # --------------------------------------------------------------

        processed = self._process_signal(
            signal,
            fs,
        )

        if processed is None:
            self.status = "signal processing failed"
            return self.bpm

        # --------------------------------------------------------------
        # PSD / BPM
        # --------------------------------------------------------------

        bpm, confidence, peak_hz = (
            self._estimate_bpm(
                processed,
                fs,
            )
        )

        if bpm is None:
            self.status = "no heartbeat peak"
            return self.bpm

        self.bpm = bpm
        self.confidence = confidence
        self.peak_hz = peak_hz

        self.last_update = now

        self.status = "OK"

        return self.bpm


# ---------------------------------------------------------------------------
# LIVE STATE
# ---------------------------------------------------------------------------

class LiveState:
    """Rolling buffers + counters updated from the sniff callback thread."""

    def __init__(
        self,
        window: int,
        capture: Optional[CSICapture] = None,
        hr_estimator: Optional[HeartRateEstimator] = None,
    ):
        self.window = window

        self.mean_amp_hist = deque(
            maxlen=window
        )

        self.per_sc_amp_hist = deque(
            maxlen=window
        )

        self.n_seen = 0
        self.n_decoded = 0
        self.n_errors = 0

        self.last_frame = None

        self.start_time = time.time()

        self.pkt_times = deque(
            maxlen=50
        )

        self.lock = threading.Lock()

        self.capture = capture

        self.hr_estimator = hr_estimator

    def update(self, frame):

        amp = np.abs(
            frame.csi
        )

        with self.lock:

            self.n_decoded += 1

            self.last_frame = frame

            now = time.time()

            self.pkt_times.append(
                now
            )

            self.mean_amp_hist.append(
                float(
                    np.mean(amp)
                )
            )

            self.per_sc_amp_hist.append(
                amp
            )

        # Heart-rate buffer.
        if self.hr_estimator is not None:
            self.hr_estimator.add_frame(
                frame
            )

        # Outside the state lock so a slow disk write never blocks
        # dashboard bookkeeping.
        if self.capture is not None:
            self.capture.write_frame(
                frame
            )

    @property
    def live_rate(self) -> float:

        with self.lock:

            if len(self.pkt_times) < 2:
                return 0.0

            dt = (
                self.pkt_times[-1]
                - self.pkt_times[0]
            )

            return (
                (len(self.pkt_times) - 1)
                / dt
                if dt > 0
                else 0.0
            )

    def motion_score(self) -> Optional[float]:

        with self.lock:

            if len(
                self.per_sc_amp_hist
            ) < max(
                5,
                self.window // 4,
            ):
                return None

            stack = np.stack(
                self.per_sc_amp_hist,
                axis=0,
            )

        per_sc_std = np.std(
            stack,
            axis=0,
        )

        mean_amp = np.mean(
            stack
        )

        return (
            float(
                np.mean(per_sc_std)
                / mean_amp
            )
            if mean_amp > 0
            else 0.0
        )


# ---------------------------------------------------------------------------
# SCAPY CALLBACK
# ---------------------------------------------------------------------------

def make_sniff_callback(
    state: LiveState,
    args,
):
    """Build the scapy sniff() callback."""

    from scapy.all import UDP, IP

    def handle_pkt(pkt):

        state.n_seen += 1

        if not pkt.haslayer(UDP):
            return

        udp = pkt[UDP]

        if udp.dport != args.udp_port:
            return

        if (
            args.src_ip
            and pkt.haslayer(IP)
            and pkt[IP].src != args.src_ip
        ):
            return

        payload = bytes(
            udp.payload
        )

        if not payload:
            return

        try:

            frame = decode_frame_payload(
                payload,
                index=state.n_decoded,
                ts=time.time(),
                chip=args.chip,
                bw=args.bw,
                nfft_override=args.nfft,
                header_format=args.header_format,
            )

        except DecodeError:

            state.n_errors += 1
            return

        if (
            args.select_core is not None
            and frame.core != args.select_core
        ):
            return

        if (
            args.select_stream is not None
            and frame.spatial_stream
            != args.select_stream
        ):
            return

        state.update(
            frame
        )

    return handle_pkt


# ---------------------------------------------------------------------------
# CURSES DASHBOARD
# ---------------------------------------------------------------------------

def run_curses(
    stdscr,
    args,
):

    curses.curs_set(0)

    stdscr.nodelay(True)

    nfft = (
        args.nfft
        or BW_TO_NFFT[args.bw]
    )

    capture = CSICapture(
        out_dir=args.output_dir,
        nfft=nfft,
        mode=args.capture_mode,
    )

    hr_estimator = HeartRateEstimator(
        window_seconds=args.hr_window,
        update_seconds=args.hr_update,
        low_hz=args.hr_low,
        high_hz=args.hr_high,
        hampel_window=args.hr_hampel_window,
        hampel_sigmas=args.hr_hampel_sigmas,
        notch_hz=args.hr_notch,
        drift_window=args.hr_drift_window,
    )

    state = LiveState(
        window=args.window,
        capture=capture,
        hr_estimator=hr_estimator,
    )

    baseline_motion = None

    baseline_needed = (
        args.baseline_window
        or args.window
    )

    from scapy.all import sniff

    stop_flag = threading.Event()

    handle_pkt = make_sniff_callback(
        state,
        args,
    )

    def sniff_thread():

        try:

            sniff(
                iface=args.iface,
                filter=f"udp port {args.udp_port}",
                prn=handle_pkt,
                store=False,
                stop_filter=lambda p:
                    stop_flag.is_set(),
            )

        except PermissionError:

            state.n_errors = -1

        except Exception as exc:

            logger.exception(
                "Sniffing failed: %s",
                exc,
            )

            state.n_errors = -1

    t = threading.Thread(
        target=sniff_thread,
        daemon=True,
    )

    t.start()

    try:

        while True:

            try:

                ch = stdscr.getch()

                if ch in (
                    ord("q"),
                    ord("Q"),
                ):
                    break

                if ch in (
                    ord("c"),
                    ord("C"),
                ):
                    capture.toggle()

                if state.n_errors == -1:

                    stdscr.erase()

                    stdscr.addstr(
                        0,
                        0,
                        "Permission denied/opening interface failed.",
                    )

                    stdscr.addstr(
                        1,
                        0,
                        "Try running with sudo.",
                    )

                    stdscr.addstr(
                        3,
                        0,
                        "Press 'q' to quit.",
                    )

                    stdscr.refresh()

                    time.sleep(0.2)

                    continue

                # ------------------------------------------------------
                # Update BPM
                # ------------------------------------------------------

                hr_estimator.update()

                # ------------------------------------------------------
                # Dashboard
                # ------------------------------------------------------

                stdscr.erase()

                h, w = stdscr.getmaxyx()

                w = max(
                    w,
                    20,
                )

                def put(row, text):

                    if 0 <= row < h - 1:

                        try:
                            stdscr.addstr(
                                row,
                                0,
                                text[: w - 1],
                            )
                        except curses.error:
                            pass

                elapsed = (
                    time.time()
                    - state.start_time
                )

                put(
                    0,
                    f"csi_analysis live monitor -- "
                    f"iface={args.iface} "
                    f"chip={args.chip} "
                    f"bw={args.bw}MHz "
                    f"port={args.udp_port} "
                    f"[q quit, c capture]",
                )

                put(
                    1,
                    "-" * min(
                        w - 1,
                        78,
                    ),
                )

                put(
                    2,
                    f"elapsed: {elapsed:6.1f}s   "
                    f"seen: {state.n_seen:6d}   "
                    f"decoded: {state.n_decoded:6d}   "
                    f"errors: {max(state.n_errors, 0):4d}   "
                    f"live rate: {state.live_rate:5.1f} pkt/s",
                )

                # ------------------------------------------------------
                # Capture status
                # ------------------------------------------------------

                if capture.active:

                    stdscr.attron(
                        curses.A_BOLD
                    )

                    put(
                        3,
                        f"** CAPTURING ** -> "
                        f"{capture.path.name}   "
                        f"frames written: "
                        f"{capture.n_written}",
                    )

                    stdscr.attroff(
                        curses.A_BOLD
                    )

                else:

                    prev = (
                        f" (last: {capture.path.name}, "
                        f"{capture.n_written} frames)"
                        if capture.path
                        else ""
                    )

                    put(
                        3,
                        f"capture: OFF -- press 'c' "
                        f"to start recording to "
                        f"{capture.out_dir}/{prev}",
                    )

                # ------------------------------------------------------
                # Heart rate
                # ------------------------------------------------------

                if hr_estimator.bpm is not None:

                    bpm = (
                        hr_estimator.bpm
                    )

                    confidence = (
                        hr_estimator.confidence
                        if hr_estimator.confidence
                        is not None
                        else 0.0
                    )

                    put(
                        5,
                        f"ESTIMATED BPM: "
                        f"{bpm:6.1f} BPM    "
                        f"confidence: "
                        f"{confidence:5.1f}%",
                    )

                    put(
                        6,
                        f"  heartbeat band: "
                        f"{args.hr_low:.2f}-"
                        f"{args.hr_high:.2f} Hz    "
                        f"PSD peak: "
                        f"{hr_estimator.peak_hz:.3f} Hz",
                    )

                    if (
                        hr_estimator.last_sample_rate
                        is not None
                    ):

                        put(
                            7,
                            f"  HR window: "
                            f"{args.hr_window:.0f}s    "
                            f"sample rate: "
                            f"{hr_estimator.last_sample_rate:.2f} Hz    "
                            f"status: "
                            f"{hr_estimator.status}",
                        )

                else:

                    put(
                        5,
                        f"ESTIMATED BPM: "
                        f"--.- BPM    "
                        f"status: "
                        f"{hr_estimator.status}",
                    )

                    put(
                        6,
                        f"  heartbeat band: "
                        f"{args.hr_low:.2f}-"
                        f"{args.hr_high:.2f} Hz    "
                        f"HR window: "
                        f"{args.hr_window:.0f}s",
                    )

                    put(
                        7,
                        "  waiting for sufficient CSI data...",
                    )

                # ------------------------------------------------------
                # Last packet
                # ------------------------------------------------------

                frame = state.last_frame

                if frame is not None:

                    rssi_str = (
                        f"{frame.rssi} dBm"
                        if frame.rssi is not None
                        else "n/a"
                    )

                    put(
                        9,
                        f"last packet -- "
                        f"seq={frame.seq_num:5d}  "
                        f"core={frame.core}  "
                        f"stream={frame.spatial_stream}  "
                        f"rssi={rssi_str}  "
                        f"chanspec=0x{frame.chanspec:04x}",
                    )

                    amp = np.abs(
                        frame.csi
                    )

                    ph = np.angle(
                        frame.csi,
                        deg=True,
                    )

                    put(
                        11,
                        f"current mean amplitude: "
                        f"{float(np.mean(amp)):10.2f}",
                    )

                    put(
                        12,
                        f"current mean phase:     "
                        f"{float(np.mean(ph)):10.2f} deg",
                    )

                    # --------------------------------------------------
                    # Raw amplitude trend
                    # --------------------------------------------------

                    put(
                        14,
                        f"amplitude trend "
                        f"(last {len(state.mean_amp_hist)} pkts, raw):",
                    )

                    put(
                        15,
                        sparkline(
                            state.mean_amp_hist,
                            width=min(
                                60,
                                w - 2,
                            ),
                        ),
                    )

                    # --------------------------------------------------
                    # Optional Hampel trend
                    # --------------------------------------------------

                    if args.hampel:

                        put(
                            16,
                            f"amplitude trend "
                            f"(Hampel window="
                            f"{args.hampel_window}, "
                            f"sigmas="
                            f"{args.hampel_sigmas}):",
                        )

                        if len(
                            state.mean_amp_hist
                        ) >= 5:

                            filtered_hist = (
                                hampel_filter(
                                    np.array(
                                        state.mean_amp_hist,
                                        dtype=np.float64,
                                    ),
                                    window_size=(
                                        args.hampel_window
                                    ),
                                    n_sigmas=(
                                        args.hampel_sigmas
                                    ),
                                )
                            )

                            put(
                                17,
                                sparkline(
                                    filtered_hist,
                                    width=min(
                                        60,
                                        w - 2,
                                    ),
                                ),
                            )

                        else:

                            put(
                                17,
                                "(warming up...)",
                            )

                    # --------------------------------------------------
                    # Subcarrier profile
                    # --------------------------------------------------

                    put(
                        19,
                        "current subcarrier profile "
                        "(this packet):",
                    )

                    put(
                        20,
                        sparkline(
                            np.fft.fftshift(amp),
                            width=min(
                                60,
                                w - 2,
                            ),
                        ),
                    )

                    # --------------------------------------------------
                    # Motion score
                    # --------------------------------------------------

                    ms = state.motion_score()

                    if ms is not None:

                        if (
                            baseline_motion is None
                            and len(
                                state.mean_amp_hist
                            ) >= baseline_needed
                        ):
                            baseline_motion = ms

                        put(
                            22,
                            f"motion score "
                            f"(rolling {args.window}-pkt "
                            f"window): {ms:.5f}",
                        )

                        if baseline_motion:

                            ratio = (
                                ms
                                / baseline_motion
                                if baseline_motion > 0
                                else 0
                            )

                            put(
                                23,
                                f"  vs baseline: "
                                f"{ratio:5.2f}x",
                            )

                            if (
                                ratio
                                > args.motion_threshold
                            ):

                                stdscr.attron(
                                    curses.A_BOLD
                                )

                                put(
                                    24,
                                    "** MOTION / CHANGE "
                                    "DETECTED **",
                                )

                                stdscr.attroff(
                                    curses.A_BOLD
                                )

                    else:

                        put(
                            22,
                            f"motion score: "
                            f"(warming up, need "
                            f"{baseline_needed} packets)",
                        )

                else:

                    put(
                        9,
                        "waiting for first CSI packet... "
                        "(check --iface / --src-ip / "
                        "that the sender is actually running)",
                    )

                stdscr.refresh()

                time.sleep(
                    max(
                        0.0,
                        1.0 / args.refresh_hz,
                    )
                )

            except KeyboardInterrupt:
                break

    finally:

        stop_flag.set()

        capture.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(argv=None):

    p = argparse.ArgumentParser(
        description=(
            "Live terminal dashboard for Nexmon CSI "
            "streams with real-time BPM estimation."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    # --------------------------------------------------------------
    # CSI / network options
    # --------------------------------------------------------------

    p.add_argument(
        "--iface",
        required=True,
        help="Interface to sniff on (e.g. wlan0)",
    )

    p.add_argument(
        "--chip",
        default="43455c0",
        choices=list(CHIP_TABLE),
    )

    p.add_argument(
        "--bw",
        type=int,
        default=80,
        choices=list(BW_TO_NFFT),
    )

    p.add_argument(
        "--nfft",
        type=int,
        default=None,
        help=(
            "Override nfft instead of deriving "
            "from --bw"
        ),
    )

    p.add_argument(
        "--udp-port",
        type=int,
        default=5500,
    )

    p.add_argument(
        "--src-ip",
        default="10.10.10.10",
        help=(
            "Expected CSI source IP; "
            "pass '' to disable"
        ),
    )

    p.add_argument(
        "--header-format",
        default="auto",
        choices=[
            "auto",
            "legacy",
            "rssi",
        ],
    )

    p.add_argument(
        "--select-core",
        type=int,
        default=None,
        help=(
            "Only show packets from this core"
        ),
    )

    p.add_argument(
        "--select-stream",
        type=int,
        default=None,
        help=(
            "Only show packets from this spatial stream"
        ),
    )

    # --------------------------------------------------------------
    # Motion monitoring
    # --------------------------------------------------------------

    p.add_argument(
        "--window",
        type=int,
        default=100,
        help=(
            "Rolling window size in packets "
            "for motion score"
        ),
    )

    p.add_argument(
        "--hampel",
        action="store_true",
        help=(
            "Show a Hampel-filtered amplitude "
            "trend alongside the raw one"
        ),
    )

    p.add_argument(
        "--hampel-window",
        type=int,
        default=7,
        help=(
            "Hampel filter window for dashboard "
            "trend"
        ),
    )

    p.add_argument(
        "--hampel-sigmas",
        type=float,
        default=3.0,
        help=(
            "Hampel outlier threshold for "
            "dashboard trend"
        ),
    )

    p.add_argument(
        "--baseline-window",
        type=int,
        default=None,
        help=(
            "Packets to average for motion-score "
            "baseline"
        ),
    )

    p.add_argument(
        "--motion-threshold",
        type=float,
        default=1.5,
        help=(
            "Motion-score ratio that triggers "
            "the alert"
        ),
    )

    # --------------------------------------------------------------
    # HEART RATE OPTIONS
    # --------------------------------------------------------------

    p.add_argument(
        "--hr-window",
        type=float,
        default=30.0,
        help=(
            "Seconds of CSI used for each "
            "heart-rate estimate"
        ),
    )

    p.add_argument(
        "--hr-update",
        type=float,
        default=2.0,
        help=(
            "Seconds between heart-rate "
            "recalculations"
        ),
    )

    p.add_argument(
        "--hr-low",
        type=float,
        default=0.7,
        help=(
            "Lower heartbeat frequency limit "
            "in Hz (42 BPM)"
        ),
    )

    p.add_argument(
        "--hr-high",
        type=float,
        default=2.5,
        help=(
            "Upper heartbeat frequency limit "
            "in Hz (150 BPM)"
        ),
    )

    p.add_argument(
        "--hr-hampel-window",
        type=int,
        default=51,
        help=(
            "Hampel window used by heartbeat "
            "pipeline"
        ),
    )

    p.add_argument(
        "--hr-hampel-sigmas",
        type=float,
        default=3.0,
        help=(
            "Hampel threshold used by heartbeat "
            "pipeline"
        ),
    )

    p.add_argument(
        "--hr-notch",
        type=float,
        default=0.0,
        help=(
            "Optional narrowband interference "
            "frequency in Hz; 0 disables notch"
        ),
    )

    p.add_argument(
        "--hr-drift-window",
        type=int,
        default=21,
        help=(
            "Moving-average drift-removal window"
        ),
    )

    # --------------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------------

    p.add_argument(
        "--refresh-hz",
        type=float,
        default=10.0,
        help="Dashboard redraw rate",
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for keypress-triggered "
            "CSI captures"
        ),
    )

    p.add_argument(
        "--capture-mode",
        choices=[
            "complex",
            "amplitude",
            "phase",
        ],
        default="amplitude",
        help=(
            "Representation written per "
            "subcarrier when capturing"
        ),
    )

    args = p.parse_args(argv)

    if not args.src_ip:
        args.src_ip = None

    if (
        args.select_core is None
    ) != (
        args.select_stream is None
    ):
        p.error(
            "--select-core and "
            "--select-stream must be "
            "given together"
        )

    if args.hr_low <= 0:
        p.error(
            "--hr-low must be greater than 0"
        )

    if args.hr_high <= args.hr_low:
        p.error(
            "--hr-high must be greater than "
            "--hr-low"
        )

    if args.hr_window < 10:
        p.error(
            "--hr-window should be at least "
            "10 seconds"
        )

    if args.hr_update <= 0:
        p.error(
            "--hr-update must be greater than 0"
        )

    args.output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            Path(__file__).resolve().parent
            / "output"
        )
    )

    curses.wrapper(
        run_curses,
        args,
    )


if __name__ == "__main__":
    main()