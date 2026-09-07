#!/usr/bin/env python3

"""
live_monitor.py
===============

Real-time terminal dashboard for Nexmon CSI streams.

In addition to packet statistics, amplitude/phase, motion score, and
key-triggered CSI capture, this version estimates heart rate (BPM) from
the live CSI stream with occupancy detection and motion scoring.

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

OCCUPANCY AND MOTION DETECTION
------------------------------

    Energy-based occupancy detection
    Motion scoring with baseline adaptation
    Signal quality assessment
    Confidence-based BPM validation

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

    --motion-threshold 2.5
        Motion score threshold (higher = less sensitive to motion).

    --occupancy-threshold 0.05
        Energy threshold for occupancy detection.

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

# Use the enhanced heartbeat extraction
from heartbeat_extraction import HeartbeatExtractor, HeartbeatConfig

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
# LIVE STATE
# ---------------------------------------------------------------------------

class LiveState:
    """Rolling buffers + counters updated from the sniff callback thread."""

    def __init__(
        self,
        window: int,
        capture: Optional[CSICapture] = None,
        heartbeat_extractor: Optional[HeartbeatExtractor] = None,
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
        self.heartbeat_extractor = heartbeat_extractor
        
        # Buffer for rolling CSI data for heartbeat extraction
        self.csi_buffer = []
        self.ts_buffer = []

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
            
            # Store CSI data for heartbeat extraction
            self.csi_buffer.append(frame.csi)
            self.ts_buffer.append(frame.ts)
            
            # Keep buffer within reasonable size
            max_buffer = int(self.heartbeat_extractor.config.sampling_rate * 60) if self.heartbeat_extractor else 6000
            if len(self.csi_buffer) > max_buffer:
                self.csi_buffer = self.csi_buffer[-max_buffer:]
                self.ts_buffer = self.ts_buffer[-max_buffer:]

        # Heartbeat extraction is done separately
        if self.heartbeat_extractor is not None:
            # We'll process in batches to avoid overhead
            pass

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

    # ============ Create heartbeat extractor with occupancy detection ============
    hr_config = HeartbeatConfig()
    hr_config.sampling_rate = 100.0
    hr_config.bandpass_low = args.hr_low
    hr_config.bandpass_high = args.hr_high
    hr_config.filter_order = 4
    hr_config.heart_rate_min = args.hr_low * 60
    hr_config.heart_rate_max = args.hr_high * 60
    hr_config.occupancy_energy_threshold = args.occupancy_threshold
    hr_config.min_snr = args.min_snr
    hr_config.max_hr_change = args.max_hr_change
    hr_config.segment_duration = args.hr_window
    
    heartbeat_extractor = HeartbeatExtractor(hr_config)

    state = LiveState(
        window=args.window,
        capture=capture,
        heartbeat_extractor=heartbeat_extractor,
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

    # Variables for periodic HR update
    last_hr_update = 0
    hr_update_interval = args.hr_update

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

                # ============ Update heartbeat extraction ============
                now = time.time()
                if now - last_hr_update >= hr_update_interval:
                    last_hr_update = now
                    
                    with state.lock:
                        if len(state.csi_buffer) > 100:
                            # Convert buffer to numpy array
                            csi_batch = np.stack(state.csi_buffer[-int(hr_config.sampling_rate * args.hr_window):], axis=0)
                            
                            # Run extraction
                            try:
                                hr_result = heartbeat_extractor.run_full_analysis(
                                    csi_batch,
                                    {'rate': hr_config.sampling_rate},
                                    output_dir=args.output_dir / 'heartbeat_analysis'
                                )
                            except Exception as e:
                                logger.debug(f"HR extraction error: {e}")

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

                # ============ Heart rate with occupancy ============
                
                # Get metrics from heartbeat extractor
                occupancy = heartbeat_extractor.occupancy if hasattr(heartbeat_extractor, 'occupancy') else False
                motion_level = heartbeat_extractor.motion_level if hasattr(heartbeat_extractor, 'motion_level') else 0
                signal_quality = heartbeat_extractor.signal_quality if hasattr(heartbeat_extractor, 'signal_quality') else 0
                hr = heartbeat_extractor.last_valid_bpm or 0
                confidence = heartbeat_extractor.confidence if hasattr(heartbeat_extractor, 'confidence') else 0
                
                # Display occupancy status
                occupancy_str = "OCCUPIED" if occupancy else "VACANT"
                occupancy_color = curses.A_BOLD if occupancy else curses.A_NORMAL
                stdscr.attron(occupancy_color)
                put(5, f"OCCUPANCY: {occupancy_str}   motion: {motion_level:.3f}   quality: {signal_quality:.2f}")
                stdscr.attroff(occupancy_color)

                # Display BPM
                if occupancy and hr > 0 and signal_quality > args.min_snr:
                    put(6, f"ESTIMATED BPM: {hr:6.1f} BPM    confidence: {confidence:5.1f}%")
                    put(7, f"  heartbeat band: {args.hr_low:.2f}-{args.hr_high:.2f} Hz")
                elif occupancy:
                    put(6, f"ESTIMATED BPM: ---.- BPM  (signal quality too low)")
                    put(7, f"  quality: {signal_quality:.2f}, need > {args.min_snr}")
                else:
                    put(6, f"ESTIMATED BPM: ---.- BPM  (no occupancy)")
                    put(7, f"  waiting for subject to enter detection zone")

                # Motion trend
                if hasattr(heartbeat_extractor, 'get_motion_trend'):
                    motion_trend = heartbeat_extractor.get_motion_trend(20)
                    if len(motion_trend) > 5:
                        put(8, f"motion trend: {sparkline(motion_trend, width=min(40, w-15))}")

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

                    put(10,
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

                    put(12,
                        f"current mean amplitude: "
                        f"{float(np.mean(amp)):10.2f}",
                    )

                    put(13,
                        f"current mean phase:     "
                        f"{float(np.mean(ph)):10.2f} deg",
                    )

                    # --------------------------------------------------
                    # Raw amplitude trend
                    # --------------------------------------------------

                    put(15,
                        f"amplitude trend "
                        f"(last {len(state.mean_amp_hist)} pkts, raw):",
                    )

                    put(16,
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

                        put(17,
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

                            put(18,
                                sparkline(
                                    filtered_hist,
                                    width=min(
                                        60,
                                        w - 2,
                                    ),
                                ),
                            )

                        else:

                            put(18,
                                "(warming up...)",
                            )

                    # --------------------------------------------------
                    # Subcarrier profile
                    # --------------------------------------------------

                    put(20,
                        "current subcarrier profile "
                        "(this packet):",
                    )

                    put(21,
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

                        put(23,
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

                            put(24,
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

                                put(25,
                                    "** MOTION / CHANGE "
                                    "DETECTED **",
                                )

                                stdscr.attroff(
                                    curses.A_BOLD
                                )

                    else:

                        put(23,
                            f"motion score: "
                            f"(warming up, need "
                            f"{baseline_needed} packets)",
                        )

                else:

                    put(9,
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

    # ============ NEW: OCCUPANCY AND MOTION OPTIONS ============
    
    p.add_argument(
        "--occupancy-threshold",
        type=float,
        default=0.05,
        help=(
            "Energy threshold for occupancy detection "
            "(lower = more sensitive)"
        ),
    )

    p.add_argument(
        "--min-snr",
        type=float,
        default=1.5,
        help=(
            "Minimum SNR for valid BPM estimate "
            "(higher = more conservative)"
        ),
    )

    p.add_argument(
        "--max-hr-change",
        type=float,
        default=30.0,
        help=(
            "Maximum BPM change between estimates "
            "(higher = more responsive)"
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