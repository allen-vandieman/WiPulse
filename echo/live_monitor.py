#!/usr/bin/env python3

"""
live_monitor.py
===============

Real-time Nexmon CSI monitor with a separate heartbeat-processing path.

Architecture
------------

                 ┌──────────────→ CSI CSV capture
                 │
UDP CSI
   │
   ↓
decode
   │
   ↓
LiveState / rolling buffer
   │
   └──────────────→ HeartbeatWorker
                           │
                           ↓
                    run_live_analysis()
                           │
                           ↓
                      HR result
                           │
                           ↓
                       Dashboard

IMPORTANT:

The Scapy packet callback never performs heartbeat analysis.

The heartbeat worker copies a rolling CSI window while holding the
state lock and then releases the lock BEFORE doing any filtering,
FFT/PSD, subcarrier selection, or other expensive processing.

This prevents heartbeat processing from blocking packet reception.
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
)

from heartbeat_extraction import (
    HeartbeatConfig,
    HeartbeatExtractor,
)


# ======================================================================
# LOGGING
# ======================================================================

logger = logging.getLogger(
    "csi_analysis.live_monitor"
)

SPARK_CHARS = (
    "▁▂▃▄▅▆▇█"
)


# ======================================================================
# CSI CAPTURE
# ======================================================================

class CSICapture:
    """
    Writes decoded CSI frames to CSV.

    Disk I/O is completely independent from heartbeat processing.
    """

    def __init__(
        self,
        out_dir: Path,
        nfft: int,
        mode: str = "amplitude",
        flush_every: int = 20,
    ):

        self.out_dir = Path(
            out_dir
        )

        self.nfft = int(
            nfft
        )

        self.mode = mode

        self.flush_every = max(
            1,
            int(flush_every),
        )

        self.lock = threading.Lock()

        self.file = None
        self.writer = None

        self.path: Optional[
            Path
        ] = None

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

            for i in range(
                self.nfft
            ):

                header += [
                    f"sc{i}_re",
                    f"sc{i}_im",
                ]

        else:

            suffix = (
                "amp"
                if self.mode
                == "amplitude"
                else "pha"
            )

            header += [
                f"sc{i}_{suffix}"
                for i in range(
                    self.nfft
                )
            ]

        return header

    def start(self) -> Path:

        with self.lock:

            if self.active:
                return self.path

            self.out_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            ts_tag = time.strftime(
                "%Y%m%d_%H%M%S"
            )

            self.path = (
                self.out_dir
                / f"csi_capture_{ts_tag}.csv"
            )

            self.file = open(
                self.path,
                "w",
                newline="",
                buffering=1,
            )

            self.writer = csv.writer(
                self.file
            )

            self.writer.writerow(
                self._header()
            )

            self.n_written = 0
            self.active = True

            return self.path

    def stop(self):

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

    def write_frame(
        self,
        frame,
    ):
        """
        Write one CSI frame.

        This function only performs disk I/O if capture is active.
        """

        with self.lock:

            if (
                not self.active
                or self.writer is None
            ):
                return

            row = [
                f"{frame.ts:.6f}",
                int(frame.seq_num),
                int(frame.core),
                int(frame.spatial_stream),
            ]

            if self.mode == "complex":

                for value in frame.csi:

                    row += [
                        float(
                            np.real(value)
                        ),
                        float(
                            np.imag(value)
                        ),
                    ]

            elif self.mode == "amplitude":

                row += list(
                    np.abs(
                        frame.csi
                    )
                )

            else:

                row += list(
                    np.angle(
                        frame.csi
                    )
                )

            self.writer.writerow(
                row
            )

            self.n_written += 1

            if (
                self.n_written
                % self.flush_every
                == 0
            ):

                self.file.flush()

    def close(self):
        self.stop()


# ======================================================================
# DISPLAY
# ======================================================================

def sparkline(
    values,
    width: int = 40,
) -> str:

    values = list(values)[
        -width:
    ]

    if not values:

        return " " * width

    values = [
        float(v)
        for v in values
        if np.isfinite(v)
    ]

    if not values:

        return " " * width

    lo = min(values)
    hi = max(values)

    span = (
        hi - lo
        if hi > lo
        else 1.0
    )

    chars = []

    for value in values:

        idx = int(
            (
                (value - lo)
                / span
            )
            * (
                len(
                    SPARK_CHARS
                )
                - 1
            )
        )

        idx = max(
            0,
            min(
                idx,
                len(
                    SPARK_CHARS
                )
                - 1,
            ),
        )

        chars.append(
            SPARK_CHARS[idx]
        )

    return "".join(
        chars
    ).rjust(width)


# ======================================================================
# LIVE STATE
# ======================================================================

class LiveState:
    """
    Thread-safe live CSI state.

    Packet callback:
        update()

    Dashboard:
        snapshot_*()

    Heartbeat worker:
        get_heartbeat_window()
    """

    def __init__(
        self,
        window: int,
        buffer_seconds: float = 60.0,
        capture: Optional[
            CSICapture
        ] = None,
    ):

        self.window = max(
            20,
            int(window),
        )

        self.buffer_seconds = (
            float(buffer_seconds)
        )

        # --------------------------------------------------------------
        # Dashboard history
        # --------------------------------------------------------------

        self.mean_amp_hist = deque(
            maxlen=self.window
        )

        self.per_sc_amp_hist = deque(
            maxlen=self.window
        )

        # --------------------------------------------------------------
        # Heartbeat rolling buffer
        # --------------------------------------------------------------

        self.csi_buffer = deque()

        self.ts_buffer = deque()

        # --------------------------------------------------------------
        # Counters
        # --------------------------------------------------------------

        self.n_seen = 0
        self.n_decoded = 0
        self.n_errors = 0

        self.start_time = (
            time.time()
        )

        self.pkt_times = deque(
            maxlen=100
        )

        self.last_frame = None

        # --------------------------------------------------------------
        # Lock
        # --------------------------------------------------------------

        self.lock = threading.Lock()

        self.capture = capture

        # --------------------------------------------------------------
        # Heartbeat result
        # --------------------------------------------------------------

        self.hr_result = {
            "status": "waiting",
            "bpm": None,
            "candidate_bpm": None,
            "confidence": 0.0,
            "quality": 0.0,
            "snr": 0.0,
            "occupancy": False,
            "motion": 0.0,
            "motion_class": "unknown",
            "sampling_rate": 0.0,
            "selected_subcarriers": [],
            "processing_samples": 0,
        }

        self.hr_result_time = 0.0

        self.hr_processing = False

        self.hr_processing_count = 0

        self.hr_processing_time = 0.0

    # ==================================================================
    # PACKET UPDATE
    # ==================================================================

    def update(
        self,
        frame,
    ):
        """
        Extremely lightweight packet callback.

        NO heartbeat processing occurs here.
        """

        amp = np.abs(
            frame.csi
        )

        now = time.time()

        # --------------------------------------------------------------
        # State update
        # --------------------------------------------------------------

        with self.lock:

            self.n_decoded += 1

            self.last_frame = frame

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

            self.csi_buffer.append(
                np.asarray(
                    frame.csi,
                    dtype=np.complex128,
                ).copy()
            )

            self.ts_buffer.append(
                float(frame.ts)
            )

            # Keep approximately buffer_seconds.
            while (
                len(self.ts_buffer) > 2
                and (
                    self.ts_buffer[-1]
                    - self.ts_buffer[0]
                    > self.buffer_seconds
                )
            ):

                self.ts_buffer.popleft()
                self.csi_buffer.popleft()

        # --------------------------------------------------------------
        # Capture is outside the state lock.
        # --------------------------------------------------------------

        if self.capture is not None:

            self.capture.write_frame(
                frame
            )

    # ==================================================================
    # HEARTBEAT WINDOW
    # ==================================================================

    def get_heartbeat_window(
        self,
        seconds: float,
    ):
        """
        Copy a rolling CSI window.

        IMPORTANT:

        Only the COPY happens while the lock is held.

        Processing occurs after the lock is released.
        """

        required_seconds = (
            float(seconds)
        )

        with self.lock:

            if len(
                self.csi_buffer
            ) < 10:

                return (
                    None,
                    None,
                )

            timestamps = np.asarray(
                self.ts_buffer,
                dtype=np.float64,
            )

            csi = np.asarray(
                self.csi_buffer,
                dtype=np.complex128,
            )

        if len(timestamps) < 2:

            return (
                None,
                None,
            )

        end_ts = (
            timestamps[-1]
        )

        start_ts = (
            end_ts
            - required_seconds
        )

        indices = np.where(
            timestamps
            >= start_ts
        )[0]

        if len(indices) < 2:

            return (
                None,
                None,
            )

        first = int(
            indices[0]
        )

        return (
            csi[first:].copy(),
            timestamps[first:].copy(),
        )

    # ==================================================================
    # SNAPSHOTS
    # ==================================================================

    @property
    def live_rate(self):

        with self.lock:

            if len(
                self.pkt_times
            ) < 2:

                return 0.0

            dt = (
                self.pkt_times[-1]
                - self.pkt_times[0]
            )

            if dt <= 0:
                return 0.0

            return (
                len(
                    self.pkt_times
                )
                - 1
            ) / dt

    def motion_score(
        self,
    ) -> Optional[float]:

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

        mean_amp = float(
            np.mean(stack)
        )

        if mean_amp <= 0:
            return 0.0

        return float(
            np.mean(
                per_sc_std
            )
            / mean_amp
        )

    def get_dashboard_snapshot(
        self,
    ):

        with self.lock:

            result = dict(
                self.hr_result
            )

            frame = (
                self.last_frame
            )

            mean_hist = list(
                self.mean_amp_hist
            )

            per_sc_hist = list(
                self.per_sc_amp_hist
            )

            return {
                "n_seen": self.n_seen,
                "n_decoded": self.n_decoded,
                "n_errors": self.n_errors,
                "live_rate": self.live_rate,
                "last_frame": frame,
                "mean_amp_hist": mean_hist,
                "per_sc_amp_hist": per_sc_hist,
                "hr_result": result,
                "hr_result_time": (
                    self.hr_result_time
                ),
                "hr_processing": (
                    self.hr_processing
                ),
                "hr_processing_count": (
                    self.hr_processing_count
                ),
                "hr_processing_time": (
                    self.hr_processing_time
                ),
            }


# ======================================================================
# HEARTBEAT WORKER
# ======================================================================

class HeartbeatWorker:
    """
    Dedicated heartbeat-processing thread.

    The worker owns its HeartbeatExtractor.

    The packet thread never touches this extractor.
    """

    def __init__(
        self,
        state: LiveState,
        extractor: HeartbeatExtractor,
        window_seconds: float,
        update_seconds: float,
    ):

        self.state = state

        self.extractor = (
            extractor
        )

        self.window_seconds = (
            float(window_seconds)
        )

        self.update_seconds = max(
            0.25,
            float(update_seconds),
        )

        self.stop_event = (
            threading.Event()
        )

        self.wake_event = (
            threading.Event()
        )

        self.thread = threading.Thread(
            target=self._run,
            name="heartbeat-worker",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def stop(self):

        self.stop_event.set()

        self.wake_event.set()

        if self.thread.is_alive():

            self.thread.join(
                timeout=3.0
            )

    def wake(self):

        self.wake_event.set()

    def _run(self):

        next_run = time.monotonic()

        while not self.stop_event.is_set():

            now = time.monotonic()

            wait_time = max(
                0.0,
                next_run - now,
            )

            self.wake_event.wait(
                timeout=wait_time
            )

            self.wake_event.clear()

            if self.stop_event.is_set():
                break

            next_run = (
                time.monotonic()
                + self.update_seconds
            )

            self.process_once()

    def process_once(self):

        (
            csi,
            timestamps,
        ) = (
            self.state
            .get_heartbeat_window(
                self.window_seconds
            )
        )

        if csi is None:

            return

        # --------------------------------------------------------------
        # Mark processing state.
        # --------------------------------------------------------------

        with self.state.lock:

            self.state.hr_processing = True
            self.state.hr_processing_count += 1

        start = time.monotonic()

        try:

            result = (
                self.extractor
                .run_live_analysis(
                    csi,
                    timestamps,
                )
            )

            elapsed = (
                time.monotonic()
                - start
            )

            with self.state.lock:

                self.state.hr_result = (
                    dict(result)
                )

                self.state.hr_result_time = (
                    time.time()
                )

                self.state.hr_processing_time = (
                    elapsed
                )

        except Exception as exc:

            logger.exception(
                "Heartbeat worker error: %s",
                exc,
            )

            elapsed = (
                time.monotonic()
                - start
            )

            with self.state.lock:

                previous = dict(
                    self.state.hr_result
                )

                previous[
                    "status"
                ] = "processing_error"

                previous[
                    "error"
                ] = str(exc)

                self.state.hr_result = (
                    previous
                )

                self.state.hr_result_time = (
                    time.time()
                )

                self.state.hr_processing_time = (
                    elapsed
                )

        finally:

            with self.state.lock:

                self.state.hr_processing = False


# ======================================================================
# SCAPY CALLBACK
# ======================================================================

def make_sniff_callback(
    state: LiveState,
    args,
):

    from scapy.all import (
        UDP,
        IP,
    )

    def handle_pkt(
        pkt,
    ):

        # --------------------------------------------------------------
        # Packet count.
        # --------------------------------------------------------------

        state.n_seen += 1

        if not pkt.haslayer(
            UDP
        ):
            return

        udp = pkt[
            UDP
        ]

        if (
            udp.dport
            != args.udp_port
        ):
            return

        if (
            args.src_ip
            and pkt.haslayer(IP)
            and pkt[IP].src
            != args.src_ip
        ):

            return

        payload = bytes(
            udp.payload
        )

        if not payload:
            return

        # --------------------------------------------------------------
        # Decode.
        # --------------------------------------------------------------

        try:

            frame = (
                decode_frame_payload(
                    payload,
                    index=state.n_decoded,
                    ts=time.time(),
                    chip=args.chip,
                    bw=args.bw,
                    nfft_override=args.nfft,
                    header_format=args.header_format,
                )
            )

        except DecodeError:

            with state.lock:
                state.n_errors += 1

            return

        # --------------------------------------------------------------
        # Core / stream filters.
        # --------------------------------------------------------------

        if (
            args.select_core
            is not None
            and frame.core
            != args.select_core
        ):

            return

        if (
            args.select_stream
            is not None
            and frame.spatial_stream
            != args.select_stream
        ):

            return

        # --------------------------------------------------------------
        # Update.
        #
        # This must stay cheap.
        # --------------------------------------------------------------

        state.update(
            frame
        )

    return handle_pkt


# ======================================================================
# DASHBOARD
# ======================================================================

def run_curses(
    stdscr,
    args,
):

    curses.curs_set(
        0
    )

    stdscr.nodelay(
        True
    )

    # --------------------------------------------------------------
    # NFFT
    # --------------------------------------------------------------

    nfft = (
        args.nfft
        or BW_TO_NFFT[
            args.bw
        ]
    )

    # --------------------------------------------------------------
    # Capture
    # --------------------------------------------------------------

    capture = CSICapture(
        out_dir=args.output_dir,
        nfft=nfft,
        mode=args.capture_mode,
    )

    # --------------------------------------------------------------
    # Heartbeat configuration
    # --------------------------------------------------------------

    hr_config = (
        HeartbeatConfig()
    )

    hr_config.bandpass_low = (
        args.hr_low
    )

    hr_config.bandpass_high = (
        args.hr_high
    )

    hr_config.heart_rate_min = (
        args.hr_low * 60.0
    )

    hr_config.heart_rate_max = (
        args.hr_high * 60.0
    )

    hr_config.filter_order = (
        args.hr_filter_order
    )

    hr_config.live_window_seconds = (
        args.hr_window
    )

    hr_config.live_update_seconds = (
        args.hr_update
    )

    hr_config.live_max_subcarriers = (
        args.hr_subcarriers
    )

    hr_config.auto_select_percentile = (
        args.hr_percentile
    )

    hr_config.occupancy_energy_threshold = (
        args.occupancy_threshold
    )

    hr_config.occupancy_diversity_threshold = (
        args.occupancy_diversity
    )

    hr_config.min_snr = (
        args.min_snr
    )

    hr_config.max_hr_change = (
        args.max_hr_change
    )

    # --------------------------------------------------------------
    # State
    # --------------------------------------------------------------

    state = LiveState(
        window=args.window,
        buffer_seconds=max(
            60.0,
            args.hr_window
            + 10.0,
        ),
        capture=capture,
    )

    # --------------------------------------------------------------
    # Dedicated heartbeat extractor.
    # --------------------------------------------------------------

    heartbeat_extractor = (
        HeartbeatExtractor(
            hr_config
        )
    )

    heartbeat_worker = (
        HeartbeatWorker(
            state=state,
            extractor=heartbeat_extractor,
            window_seconds=args.hr_window,
            update_seconds=args.hr_update,
        )
    )

    # --------------------------------------------------------------
    # Sniffer.
    # --------------------------------------------------------------

    from scapy.all import sniff

    stop_flag = (
        threading.Event()
    )

    handle_pkt = (
        make_sniff_callback(
            state,
            args,
        )
    )

    def sniff_thread():

        try:

            sniff(
                iface=args.iface,
                filter=(
                    f"udp port "
                    f"{args.udp_port}"
                ),
                prn=handle_pkt,
                store=False,
                stop_filter=lambda p:
                    stop_flag.is_set(),
            )

        except PermissionError:

            with state.lock:
                state.n_errors = -1

        except Exception as exc:

            logger.exception(
                "Sniffing failed: %s",
                exc,
            )

            with state.lock:
                state.n_errors = -1

    sniffer = threading.Thread(
        target=sniff_thread,
        name="csi-sniffer",
        daemon=True,
    )

    sniffer.start()

    # --------------------------------------------------------------
    # Heartbeat worker.
    # --------------------------------------------------------------

    heartbeat_worker.start()

    # --------------------------------------------------------------
    # Dashboard.
    # --------------------------------------------------------------

    baseline_motion = None

    baseline_needed = (
        args.baseline_window
        or args.window
    )

    try:

        while True:

            try:

                ch = (
                    stdscr.getch()
                )

                # --------------------------------------------------
                # Quit
                # --------------------------------------------------

                if ch in (
                    ord("q"),
                    ord("Q"),
                ):

                    break

                # --------------------------------------------------
                # Capture
                # --------------------------------------------------

                if ch in (
                    ord("c"),
                    ord("C"),
                ):

                    capture.toggle()

                # --------------------------------------------------
                # Fatal sniffer error
                # --------------------------------------------------

                with state.lock:

                    fatal_error = (
                        state.n_errors
                        == -1
                    )

                if fatal_error:

                    stdscr.erase()

                    stdscr.addstr(
                        0,
                        0,
                        "CSI SNIFFER ERROR",
                    )

                    stdscr.addstr(
                        2,
                        0,
                        "Permission denied "
                        "or interface could "
                        "not be opened.",
                    )

                    stdscr.addstr(
                        3,
                        0,
                        "Try running with sudo.",
                    )

                    stdscr.addstr(
                        5,
                        0,
                        "Press q to quit.",
                    )

                    stdscr.refresh()

                    time.sleep(
                        0.2
                    )

                    continue

                # --------------------------------------------------
                # Snapshot state.
                # --------------------------------------------------

                snapshot = (
                    state
                    .get_dashboard_snapshot()
                )

                elapsed = (
                    time.time()
                    - state.start_time
                )

                # --------------------------------------------------
                # Draw.
                # --------------------------------------------------

                stdscr.erase()

                h, w = (
                    stdscr.getmaxyx()
                )

                w = max(
                    w,
                    20,
                )

                def put(
                    row,
                    text,
                ):

                    if (
                        0
                        <= row
                        < h - 1
                    ):

                        try:

                            stdscr.addstr(
                                row,
                                0,
                                str(
                                    text
                                )[
                                    :w - 1
                                ],
                            )

                        except curses.error:

                            pass

                # --------------------------------------------------
                # Header
                # --------------------------------------------------

                put(
                    0,
                    "CSI-Heart LIVE MONITOR "
                    f"-- iface={args.iface} "
                    f"chip={args.chip} "
                    f"bw={args.bw}MHz "
                    f"port={args.udp_port}",
                )

                put(
                    1,
                    "-" * min(
                        w - 1,
                        90,
                    ),
                )

                put(
                    2,
                    f"elapsed: {elapsed:6.1f}s   "
                    f"seen: {snapshot['n_seen']:7d}   "
                    f"decoded: {snapshot['n_decoded']:7d}   "
                    f"errors: {max(snapshot['n_errors'], 0):4d}   "
                    f"rate: {snapshot['live_rate']:6.2f} pkt/s",
                )

                # --------------------------------------------------
                # Capture
                # --------------------------------------------------

                if capture.active:

                    put(
                        3,
                        f"CAPTURING: "
                        f"{capture.path.name}   "
                        f"frames={capture.n_written}",
                    )

                else:

                    put(
                        3,
                        "capture: OFF "
                        "-- press 'c' to start",
                    )

                # --------------------------------------------------
                # HR result
                # --------------------------------------------------

                hr = snapshot[
                    "hr_result"
                ]

                occupancy = bool(
                    hr.get(
                        "occupancy",
                        False,
                    )
                )

                motion = float(
                    hr.get(
                        "motion",
                        0.0,
                    )
                )

                motion_class = (
                    hr.get(
                        "motion_class",
                        "unknown",
                    )
                )

                quality = float(
                    hr.get(
                        "quality",
                        0.0,
                    )
                )

                confidence = float(
                    hr.get(
                        "confidence",
                        0.0,
                    )
                )

                bpm = hr.get(
                    "bpm"
                )

                candidate = hr.get(
                    "candidate_bpm"
                )

                sampling_rate = float(
                    hr.get(
                        "sampling_rate",
                        0.0,
                    )
                )

                status = hr.get(
                    "status",
                    "waiting",
                )

                selected = hr.get(
                    "selected_subcarriers",
                    [],
                )

                # --------------------------------------------------
                # Occupancy
                # --------------------------------------------------

                occupancy_text = (
                    "OCCUPIED"
                    if occupancy
                    else "VACANT"
                )

                put(
                    5,
                    f"OCCUPANCY: "
                    f"{occupancy_text:<9} "
                    f"motion={motion:6.3f} "
                    f"({motion_class}) "
                    f"quality={quality:5.2f}",
                )

                # --------------------------------------------------
                # BPM
                # --------------------------------------------------

                if status == "valid" and bpm is not None:

                    put(
                        6,
                        f"HEART RATE: "
                        f"{float(bpm):6.1f} BPM   "
                        f"confidence={confidence:5.1f}%   "
                        f"status=VALID",
                    )

                elif (
                    occupancy
                    and bpm is not None
                ):

                    put(
                        6,
                        f"HEART RATE: "
                        f"{float(bpm):6.1f} BPM   "
                        f"confidence={confidence:5.1f}%   "
                        f"status={status.upper()}",
                    )

                elif occupancy:

                    put(
                        6,
                        "HEART RATE: "
                        "---.- BPM   "
                        f"status={status.upper()}",
                    )

                else:

                    put(
                        6,
                        "HEART RATE: "
                        "---.- BPM   "
                        "status=NO OCCUPANCY",
                    )

                # --------------------------------------------------
                # Candidate
                # --------------------------------------------------

                if candidate is not None:

                    put(
                        7,
                        f"candidate: "
                        f"{float(candidate):6.1f} BPM   "
                        f"snr={float(hr.get('snr', 0.0)):5.2f}   "
                        f"peak={float(hr.get('peak_freq', 0.0) or 0.0):.3f} Hz",
                    )

                else:

                    put(
                        7,
                        f"heartbeat band: "
                        f"{args.hr_low:.2f}-"
                        f"{args.hr_high:.2f} Hz",
                    )

                # --------------------------------------------------
                # Processing
                # --------------------------------------------------

                processing = snapshot[
                    "hr_processing"
                ]

                process_time = snapshot[
                    "hr_processing_time"
                ]

                if processing:

                    put(
                        8,
                        "heartbeat worker: "
                        "PROCESSING...",
                    )

                else:

                    put(
                        8,
                        f"heartbeat worker: "
                        f"idle   "
                        f"last processing="
                        f"{process_time:.3f}s   "
                        f"fs={sampling_rate:.2f} Hz",
                    )

                # --------------------------------------------------
                # Subcarriers
                # --------------------------------------------------

                if selected:

                    selected_text = (
                        ", ".join(
                            str(x)
                            for x in selected[:12]
                        )
                    )

                    if len(selected) > 12:

                        selected_text += (
                            ", ..."
                        )

                    put(
                        9,
                        "selected SCs: "
                        f"{selected_text}",
                    )

                else:

                    put(
                        9,
                        "selected SCs: "
                        "warming up...",
                    )

                # --------------------------------------------------
                # Result age
                # --------------------------------------------------

                result_time = snapshot[
                    "hr_result_time"
                ]

                if result_time > 0:

                    age = (
                        time.time()
                        - result_time
                    )

                    put(
                        10,
                        f"HR result age: "
                        f"{age:5.1f}s   "
                        f"window={args.hr_window:.1f}s   "
                        f"update={args.hr_update:.1f}s",
                    )

                else:

                    put(
                        10,
                        "HR result age: "
                        "waiting for first analysis...",
                    )

                # --------------------------------------------------
                # Last packet
                # --------------------------------------------------

                frame = snapshot[
                    "last_frame"
                ]

                if frame is not None:

                    rssi = (
                        f"{frame.rssi} dBm"
                        if frame.rssi
                        is not None
                        else "n/a"
                    )

                    put(
                        12,
                        f"last packet: "
                        f"seq={frame.seq_num:6d} "
                        f"core={frame.core} "
                        f"stream={frame.spatial_stream} "
                        f"RSSI={rssi} "
                        f"chanspec=0x{frame.chanspec:04x}",
                    )

                    amp = np.abs(
                        frame.csi
                    )

                    phase = np.angle(
                        frame.csi,
                        deg=True,
                    )

                    put(
                        14,
                        f"mean amplitude: "
                        f"{float(np.mean(amp)):10.3f}",
                    )

                    put(
                        15,
                        f"mean phase:     "
                        f"{float(np.mean(phase)):10.3f} deg",
                    )

                    # --------------------------------------------------
                    # Amplitude trend
                    # --------------------------------------------------

                    put(
                        17,
                        "amplitude trend:",
                    )

                    put(
                        18,
                        sparkline(
                            snapshot[
                                "mean_amp_hist"
                            ],
                            width=min(
                                70,
                                w - 2,
                            ),
                        ),
                    )

                    # --------------------------------------------------
                    # Optional Hampel trend
                    # --------------------------------------------------

                    if args.hampel:

                        put(
                            20,
                            f"Hampel amplitude "
                            f"trend "
                            f"(window={args.hampel_window}, "
                            f"sigmas={args.hampel_sigmas}):",
                        )

                        hist = snapshot[
                            "mean_amp_hist"
                        ]

                        if len(hist) >= 5:

                            filtered = (
                                hampel_filter(
                                    np.asarray(
                                        hist,
                                        dtype=float,
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
                                21,
                                sparkline(
                                    filtered,
                                    width=min(
                                        70,
                                        w - 2,
                                    ),
                                ),
                            )

                        else:

                            put(
                                21,
                                "warming up...",
                            )

                    # --------------------------------------------------
                    # Current subcarrier profile
                    # --------------------------------------------------

                    put(
                        23,
                        "current subcarrier "
                        "amplitude profile:",
                    )

                    put(
                        24,
                        sparkline(
                            np.fft.fftshift(
                                amp
                            ),
                            width=min(
                                70,
                                w - 2,
                            ),
                        ),
                    )

                    # --------------------------------------------------
                    # Legacy motion score
                    # --------------------------------------------------

                    ms = (
                        state.motion_score()
                    )

                    if ms is not None:

                        if (
                            baseline_motion
                            is None
                            and len(
                                snapshot[
                                    "mean_amp_hist"
                                ]
                            )
                            >= baseline_needed
                        ):

                            baseline_motion = ms

                        put(
                            26,
                            f"legacy motion score: "
                            f"{ms:.5f}",
                        )

                        if baseline_motion:

                            ratio = (
                                ms
                                / baseline_motion
                                if baseline_motion
                                > 0
                                else 0
                            )

                            put(
                                27,
                                f"baseline ratio: "
                                f"{ratio:.2f}x",
                            )

                            if (
                                ratio
                                > args.motion_threshold
                            ):

                                put(
                                    28,
                                    "MOTION / CHANGE "
                                    "DETECTED",
                                )

                    else:

                        put(
                            26,
                            "legacy motion score: "
                            "warming up...",
                        )

                else:

                    put(
                        12,
                        "waiting for first CSI "
                        "packet...",
                    )

                # --------------------------------------------------
                # Footer
                # --------------------------------------------------

                put(
                    h - 2,
                    "[q] quit   "
                    "[c] capture   "
                    "CSI reception and heartbeat "
                    "processing are independent",
                )

                stdscr.refresh()

                time.sleep(
                    max(
                        0.01,
                        1.0
                        / args.refresh_hz,
                    )
                )

            except KeyboardInterrupt:

                break

    finally:

        stop_flag.set()

        heartbeat_worker.stop()

        capture.close()


# ======================================================================
# MAIN
# ======================================================================

def main(
    argv=None,
):

    parser = argparse.ArgumentParser(
        description=(
            "CSI-Heart live Nexmon CSI "
            "monitor with isolated heartbeat "
            "processing."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    # ==================================================================
    # CSI
    # ==================================================================

    parser.add_argument(
        "--iface",
        required=True,
        help=(
            "Interface to sniff on "
            "(e.g. wlan0)"
        ),
    )

    parser.add_argument(
        "--chip",
        default="43455c0",
        choices=list(
            CHIP_TABLE
        ),
    )

    parser.add_argument(
        "--bw",
        type=int,
        default=80,
        choices=list(
            BW_TO_NFFT
        ),
    )

    parser.add_argument(
        "--nfft",
        type=int,
        default=None,
        help=(
            "Override NFFT instead of "
            "deriving it from bandwidth."
        ),
    )

    parser.add_argument(
        "--udp-port",
        type=int,
        default=5500,
    )

    parser.add_argument(
        "--src-ip",
        default="10.10.10.10",
        help=(
            "Expected CSI source IP. "
            "Use '' to disable."
        ),
    )

    parser.add_argument(
        "--header-format",
        default="auto",
        choices=[
            "auto",
            "legacy",
            "rssi",
        ],
    )

    parser.add_argument(
        "--select-core",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--select-stream",
        type=int,
        default=None,
    )

    # ==================================================================
    # GENERAL MONITORING
    # ==================================================================

    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help=(
            "Rolling packet window used "
            "by dashboard motion monitoring."
        ),
    )

    parser.add_argument(
        "--refresh-hz",
        type=float,
        default=10.0,
        help=(
            "Dashboard redraw rate."
        ),
    )

    parser.add_argument(
        "--hampel",
        action="store_true",
        help=(
            "Show Hampel-filtered amplitude "
            "trend."
        ),
    )

    parser.add_argument(
        "--hampel-window",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--hampel-sigmas",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--baseline-window",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=1.5,
    )

    # ==================================================================
    # HEARTBEAT
    # ==================================================================

    parser.add_argument(
        "--hr-window",
        type=float,
        default=30.0,
        help=(
            "Seconds of CSI used for each "
            "live HR estimate."
        ),
    )

    parser.add_argument(
        "--hr-update",
        type=float,
        default=2.0,
        help=(
            "Seconds between live HR "
            "estimates."
        ),
    )

    parser.add_argument(
        "--hr-low",
        type=float,
        default=0.7,
        help=(
            "Lower heartbeat frequency "
            "limit in Hz."
        ),
    )

    parser.add_argument(
        "--hr-high",
        type=float,
        default=2.5,
        help=(
            "Upper heartbeat frequency "
            "limit in Hz."
        ),
    )

    parser.add_argument(
        "--hr-filter-order",
        type=int,
        default=4,
        help=(
            "Butterworth heartbeat "
            "filter order."
        ),
    )

    parser.add_argument(
        "--hr-subcarriers",
        type=int,
        default=12,
        help=(
            "Maximum number of subcarriers "
            "used for live heartbeat fusion."
        ),
    )

    parser.add_argument(
        "--hr-percentile",
        type=float,
        default=20.0,
        help=(
            "Percentage of subcarriers "
            "eligible for live selection "
            "before the hard maximum."
        ),
    )

    parser.add_argument(
        "--occupancy-threshold",
        type=float,
        default=0.05,
        help=(
            "Energy threshold for "
            "occupancy detection."
        ),
    )

    parser.add_argument(
        "--occupancy-diversity",
        type=float,
        default=0.005,
        help=(
            "Subcarrier diversity threshold "
            "for occupancy detection."
        ),
    )

    parser.add_argument(
        "--min-snr",
        type=float,
        default=1.5,
        help=(
            "Minimum PSD SNR for accepting "
            "a heartbeat candidate."
        ),
    )

    parser.add_argument(
        "--max-hr-change",
        type=float,
        default=25.0,
        help=(
            "Maximum BPM jump before a "
            "candidate is treated as unstable."
        ),
    )

    # ==================================================================
    # OUTPUT
    # ==================================================================

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for CSI captures."
        ),
    )

    parser.add_argument(
        "--capture-mode",
        choices=[
            "complex",
            "amplitude",
            "phase",
        ],
        default="complex",
        help=(
            "Representation stored by CSI "
            "capture. Complex is recommended "
            "for later heartbeat analysis."
        ),
    )

    args = parser.parse_args(
        argv
    )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    if args.src_ip == "":
        args.src_ip = None

    if (
        args.select_core is None
    ) != (
        args.select_stream is None
    ):

        parser.error(
            "--select-core and "
            "--select-stream must be "
            "specified together."
        )

    if args.hr_low <= 0:

        parser.error(
            "--hr-low must be > 0."
        )

    if (
        args.hr_high
        <= args.hr_low
    ):

        parser.error(
            "--hr-high must be greater "
            "than --hr-low."
        )

    if args.hr_window < 10:

        parser.error(
            "--hr-window should be at "
            "least 10 seconds."
        )

    if args.hr_update <= 0:

        parser.error(
            "--hr-update must be > 0."
        )

    if args.hr_subcarriers < 1:

        parser.error(
            "--hr-subcarriers must be >= 1."
        )

    if not (
        0 < args.hr_percentile <= 100
    ):

        parser.error(
            "--hr-percentile must be "
            "between 0 and 100."
        )

    args.output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else (
            Path(__file__)
            .resolve()
            .parent
            / "output"
        )
    )

    # ==================================================================
    # RUN
    # ==================================================================

    curses.wrapper(
        run_curses,
        args,
    )


if __name__ == "__main__":

    main()