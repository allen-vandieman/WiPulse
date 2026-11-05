#!/usr/bin/env python3
"""
live_monitor.py
================
Real-time terminal dashboard for Nexmon CSI streams. Sniffs the same UDP
CSI broadcast that main.py decodes from a pcap file, but does it live off a
network interface, so you can watch packets arrive, see current
amplitude/phase, and get a rolling motion-score indicator that flags when
something changes -- without having to capture-then-analyze in two steps.

Reuses decoder.decode_frame_payload() directly, so it always stays in sync
with whatever header-format/chip fixes land in the offline pipeline.

Usage:
    sudo python live_monitor.py --iface wlan0 --chip 43455c0 --bw 80

Needs root (or CAP_NET_RAW), same as tcpdump/scapy sniffing normally does.
Press 'q' to quit (or Ctrl+C). Press 'c' to start/stop recording decoded CSI
to a timestamped CSV in --output-dir (default: an 'output/' folder next to
this script -- the same one heartbeat-giop.py writes ground_truth.csv to).
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

from decoder import decode_frame_payload, DecodeError, CHIP_TABLE, BW_TO_NFFT
from filters import hampel_filter

logger = logging.getLogger("csi_analysis.live_monitor")

SPARK_CHARS = "▁▂▃▄▅▆▇█"


class CSICapture:
    """
    Handles writing live CSI frames to CSV, toggled on/off with the 'c' key.

    Deliberately mirrors heartbeat-giop.py's conventions so the two data
    streams can be fed into the same ML pipeline afterwards:
      - writes into a sibling "output/" directory next to this script (the
        same folder heartbeat-giop.py uses when run from the same repo),
        so both land side-by-side without extra setup
      - timestamps are raw time.time() epoch seconds formatted to 3
        decimals ("%.3f"), same as heartbeat-giop.py's ground_truth.csv --
        that's what lets you align/interpolate the two time series later
      - one row per packet, flushed periodically (not on every single row --
        CSI arrives far faster than heartbeats, so flushing every row would
        add real I/O overhead at 100+ Hz)

    CSV layout (mode="amplitude", the default):
        ts, seq_num, core, spatial_stream, sc0_amp, sc1_amp, ..., sc{nfft-1}_amp
    Subcarriers are written in raw decode order (NOT fftshifted), matching
    utils.save_csv()/main.py's --save-csv export, so a live capture and an
    offline pcap export are directly comparable column-for-column.
    """

    def __init__(self, out_dir: Path, nfft: int, mode: str = "amplitude", flush_every: int = 20):
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
        header = ["ts", "seq_num", "core", "spatial_stream"]
        if self.mode == "complex":
            for i in range(self.nfft):
                header += [f"sc{i}_re", f"sc{i}_im"]
        else:
            suffix = "amp" if self.mode == "amplitude" else "pha"
            header += [f"sc{i}_{suffix}" for i in range(self.nfft)]
        return header

    def start(self) -> Path:
        """Open a new timestamped CSV file and start writing. Idempotent while active."""
        with self.lock:
            if self.active:
                return self.path
            self.out_dir.mkdir(exist_ok=True)
            ts_tag = time.strftime("%Y%m%d_%H%M%S")
            self.path = self.out_dir / f"csi_capture_{ts_tag}.csv"
            self.file = open(self.path, "w", newline="")
            self.writer = csv.writer(self.file)
            self.writer.writerow(self._header())
            self.n_written = 0
            self.active = True
            return self.path

    def stop(self):
        """Flush and close the current file, if any."""
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
        else:
            self.start()
            return True

    def write_frame(self, frame):
        """Write one decoded CSI frame as a CSV row, if capture is active."""
        with self.lock:
            if not self.active or self.writer is None:
                return
            row = [f"{frame.ts:.3f}", int(frame.seq_num), int(frame.core), int(frame.spatial_stream)]
            if self.mode == "complex":
                for v in frame.csi:
                    row += [v.real, v.imag]
            elif self.mode == "amplitude":
                row += list(np.abs(frame.csi))
            else:
                row += list(np.angle(frame.csi))
            self.writer.writerow(row)
            self.n_written += 1
            if self.n_written % self.flush_every == 0:
                self.file.flush()

    def close(self):
        self.stop()


def sparkline(values, width: int = 40) -> str:
    """ASCII/unicode sparkline, auto-scaled to the min/max of the given values."""
    values = list(values)[-width:]
    if not values:
        return " " * width
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1.0
    chars = [SPARK_CHARS[int((v - lo) / span * (len(SPARK_CHARS) - 1))] for v in values]
    return "".join(chars).rjust(width)


class LiveState:
    """Rolling buffers + counters updated from the sniff callback thread."""

    def __init__(self, window: int, capture: Optional["CSICapture"] = None):
        self.window = window
        self.mean_amp_hist = deque(maxlen=window)
        self.per_sc_amp_hist = deque(maxlen=window)
        self.n_seen = 0
        self.n_decoded = 0
        self.n_errors = 0
        self.last_frame = None
        self.start_time = time.time()
        self.pkt_times = deque(maxlen=50)
        self.lock = threading.Lock()
        self.capture = capture  # CSICapture instance; writes are a no-op while inactive

    def update(self, frame):
        amp = np.abs(frame.csi)
        with self.lock:
            self.n_decoded += 1
            self.last_frame = frame
            now = time.time()
            self.pkt_times.append(now)
            self.mean_amp_hist.append(float(np.mean(amp)))
            self.per_sc_amp_hist.append(amp)
        # Outside the state lock (CSICapture has its own lock) so a slow
        # disk write never blocks the sparkline/motion-score bookkeeping.
        if self.capture is not None:
            self.capture.write_frame(frame)

    @property
    def live_rate(self) -> float:
        with self.lock:
            if len(self.pkt_times) < 2:
                return 0.0
            dt = self.pkt_times[-1] - self.pkt_times[0]
            return (len(self.pkt_times) - 1) / dt if dt > 0 else 0.0

    def motion_score(self) -> Optional[float]:
        with self.lock:
            if len(self.per_sc_amp_hist) < max(5, self.window // 4):
                return None
            stack = np.stack(self.per_sc_amp_hist, axis=0)
        per_sc_std = np.std(stack, axis=0)
        mean_amp = np.mean(stack)
        return float(np.mean(per_sc_std) / mean_amp) if mean_amp > 0 else 0.0


def make_sniff_callback(state: LiveState, args):
    """Builds the scapy sniff() prn callback bound to this run's args/state."""
    from scapy.all import UDP, IP

    def handle_pkt(pkt):
        state.n_seen += 1
        if not pkt.haslayer(UDP):
            return
        udp = pkt[UDP]
        if udp.dport != args.udp_port:
            return
        if args.src_ip and pkt.haslayer(IP) and pkt[IP].src != args.src_ip:
            return
        payload = bytes(udp.payload)
        if not payload:
            return
        try:
            frame = decode_frame_payload(
                payload, index=state.n_decoded, ts=time.time(), chip=args.chip,
                bw=args.bw, nfft_override=args.nfft, header_format=args.header_format,
            )
        except DecodeError:
            state.n_errors += 1
            return
        if args.select_core is not None and frame.core != args.select_core:
            return
        if args.select_stream is not None and frame.spatial_stream != args.select_stream:
            return
        state.update(frame)

    return handle_pkt


def run_curses(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)

    nfft = args.nfft or BW_TO_NFFT[args.bw]
    capture = CSICapture(out_dir=args.output_dir, nfft=nfft, mode=args.capture_mode)

    state = LiveState(window=args.window, capture=capture)
    baseline_motion = None
    baseline_needed = args.baseline_window or args.window

    from scapy.all import sniff

    stop_flag = threading.Event()
    handle_pkt = make_sniff_callback(state, args)

    def sniff_thread():
        try:
            sniff(iface=args.iface, filter=f"udp port {args.udp_port}", prn=handle_pkt,
                  store=False, stop_filter=lambda p: stop_flag.is_set())
        except PermissionError:
            state.n_errors = -1  # signal permission failure to the UI loop

    t = threading.Thread(target=sniff_thread, daemon=True)
    t.start()

    try:
        while True:
            try:
                ch = stdscr.getch()
                if ch in (ord("q"), ord("Q")):
                    break
                if ch in (ord("c"), ord("C")):
                    capture.toggle()

                if state.n_errors == -1:
                    stdscr.erase()
                    stdscr.addstr(0, 0, "Permission denied opening the interface for sniffing.")
                    stdscr.addstr(1, 0, "Try running with sudo (raw capture needs CAP_NET_RAW).")
                    stdscr.addstr(3, 0, "Press 'q' to quit.")
                    stdscr.refresh()
                    time.sleep(0.2)
                    continue

                stdscr.erase()
                h, w = stdscr.getmaxyx()
                w = max(w, 20)

                def put(row, text):
                    if 0 <= row < h - 1:
                        stdscr.addstr(row, 0, text[: w - 1])

                elapsed = time.time() - state.start_time
                put(0, f"csi_analysis live monitor -- iface={args.iface} chip={args.chip} "
                       f"bw={args.bw}MHz port={args.udp_port}  [q to quit, c to start/stop capture]")
                put(1, "-" * min(w - 1, 78))
                put(2, f"elapsed: {elapsed:6.1f}s   seen: {state.n_seen:6d}   decoded: {state.n_decoded:6d}   "
                       f"errors: {max(state.n_errors,0):4d}   live rate: {state.live_rate:5.1f} pkt/s")

                if capture.active:
                    stdscr.attron(curses.A_BOLD)
                    put(3, f"** CAPTURING ** -> {capture.path.name}   frames written: {capture.n_written}")
                    stdscr.attroff(curses.A_BOLD)
                else:
                    prev = f" (last: {capture.path.name}, {capture.n_written} frames)" if capture.path else ""
                    put(3, f"capture: OFF -- press 'c' to start recording to {capture.out_dir}/{prev}")

                frame = state.last_frame
                if frame is not None:
                    rssi_str = f"{frame.rssi} dBm" if frame.rssi is not None else "n/a"
                    put(4, f"last packet -- seq={frame.seq_num:5d}  core={frame.core}  "
                           f"stream={frame.spatial_stream}  rssi={rssi_str}  "
                           f"chanspec=0x{frame.chanspec:04x}")

                    amp = np.abs(frame.csi)
                    ph = np.angle(frame.csi, deg=True)
                    put(6, f"current mean amplitude: {float(np.mean(amp)):10.2f}")
                    put(7, f"current mean phase:     {float(np.mean(ph)):10.2f} deg")

                    put(9, f"amplitude trend (last {len(state.mean_amp_hist)} pkts, raw):")
                    put(10, sparkline(state.mean_amp_hist, width=min(60, w - 2)))

                    if args.hampel:
                        put(11, f"amplitude trend (last {len(state.mean_amp_hist)} pkts, "
                                f"hampel-filtered window={args.hampel_window} sigmas={args.hampel_sigmas}):")
                        if len(state.mean_amp_hist) >= 5:
                            filtered_hist = hampel_filter(
                                np.array(state.mean_amp_hist, dtype=np.float64),
                                window_size=args.hampel_window,
                                n_sigmas=args.hampel_sigmas,
                            )
                            put(12, sparkline(filtered_hist, width=min(60, w - 2)))
                        else:
                            put(12, "(warming up...)")

                    put(14, "current subcarrier profile (this packet):")
                    put(15, sparkline(np.fft.fftshift(amp), width=min(60, w - 2)))

                    ms = state.motion_score()
                    if ms is not None:
                        if baseline_motion is None and len(state.mean_amp_hist) >= baseline_needed:
                            baseline_motion = ms
                        put(17, f"motion score (rolling {args.window}-pkt window): {ms:.5f}")
                        if baseline_motion:
                            ratio = ms / baseline_motion if baseline_motion > 0 else 0
                            put(18, f"  vs baseline: {ratio:5.2f}x")
                            if ratio > args.motion_threshold:
                                stdscr.attron(curses.A_BOLD)
                                put(19, "  ** MOTION / CHANGE DETECTED **")
                                stdscr.attroff(curses.A_BOLD)
                    else:
                        put(17, f"motion score: (warming up, need {baseline_needed} packets)")
                else:
                    put(4, "waiting for first CSI packet... (check --iface / --src-ip / that "
                           "the sender is actually running)")

                stdscr.refresh()
                time.sleep(max(0.0, 1.0 / args.refresh_hz))
            except KeyboardInterrupt:
                break
    finally:
        stop_flag.set()
        capture.close()


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Live terminal dashboard for Nexmon CSI streams (real-time, off a NIC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iface", required=True, help="Interface to sniff on (e.g. wlan0)")
    p.add_argument("--chip", default="43455c0", choices=list(CHIP_TABLE))
    p.add_argument("--bw", type=int, default=80, choices=list(BW_TO_NFFT))
    p.add_argument("--nfft", type=int, default=None, help="Override nfft instead of deriving from --bw")
    p.add_argument("--udp-port", type=int, default=5500)
    p.add_argument("--src-ip", default="10.10.10.10", help="Expected CSI source IP; pass '' to disable")
    p.add_argument("--header-format", default="auto", choices=["auto", "legacy", "rssi"])
    p.add_argument("--select-core", type=int, default=None,
                    help="Only show packets from this core (use with --select-stream if your "
                         "capture interleaves multiple streams)")
    p.add_argument("--select-stream", type=int, default=None, help="Only show packets from this spatial stream")
    p.add_argument("--window", type=int, default=100, help="Rolling window size (packets) for motion score")
    p.add_argument("--hampel", action="store_true",
                    help="Show a second, Hampel-filtered amplitude trend line alongside the raw one "
                         "(knocks out isolated spike artifacts; same filter as main.py --hampel)")
    p.add_argument("--hampel-window", type=int, default=7, help="Hampel filter window size (packets)")
    p.add_argument("--hampel-sigmas", type=float, default=3.0, help="Hampel filter outlier threshold (sigmas)")
    p.add_argument("--baseline-window", type=int, default=None,
                    help="Packets to average for the motion-score baseline (default: same as --window)")
    p.add_argument("--motion-threshold", type=float, default=1.5,
                    help="Ratio of current motion score over baseline that triggers the alert")
    p.add_argument("--refresh-hz", type=float, default=10.0, help="Dashboard redraw rate")

    p.add_argument("--output-dir", type=str, default=None,
                    help="Directory for keypress-triggered CSI captures (press 'c' in the "
                         "dashboard to start/stop). Default: an 'output/' folder next to this "
                         "script -- the same folder heartbeat-giop.py writes ground_truth.csv "
                         "to, so both land together for the ML step.")
    p.add_argument("--capture-mode", choices=["complex", "amplitude", "phase"], default="amplitude",
                    help="Representation to write per subcarrier when capturing (see utils.save_csv "
                         "for the same convention used by main.py's --save-csv)")
    args = p.parse_args(argv)

    if not args.src_ip:
        args.src_ip = None
    if (args.select_core is None) != (args.select_stream is None):
        p.error("--select-core and --select-stream must be given together")

    args.output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).resolve().parent / "output"
    )

    curses.wrapper(run_curses, args)


if __name__ == "__main__":
    main()
