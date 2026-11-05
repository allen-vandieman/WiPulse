#!/usr/bin/env python3
"""CSI-Heart | Live servo-motion frequency demonstrator.

Uses the project's decoder.py and the same Nexmon CSI UDP format as live_monitor.py.
Receives/scans packets in one thread, estimates the dominant frequency from an
independent rolling CSI window in a second thread, and displays a curses UI plus a real-time HTML dashboard on the LAN.

Real hardware (Nexmon 20 MHz, Raspberry Pi):
  sudo python3 servo_frequency_monitor.py --iface wlan0 --bw 20 --expected-hz 1.0
  python servo_frequency_monitor.py --iface Ethernet --expected-hz 1.0

Nexmon injected UDP can be visible to tcpdump while Scapy's 'udp port 5500'
capture filter drops every frame. By default this monitor receives all frames
from the selected interface and checks UDP port/source IP in Python.
Use --bpf-filter only if verified to work on your interface. --debug-capture
prints additional capture counters without changing this default.

Demo rehearsal (clearly labelled synthetic input):
  python servo_frequency_monitor.py --simulate --expected-hz 1.2
  python servo_frequency_monitor.py --self-test

Optional prerecorded Nexmon UDP PCAP playback (clearly labelled REPLAY):
  python servo_frequency_monitor.py --pcap captures/example.pcap --expected-hz 1.0

Web: http://<Pi-LAN-IP>:8080 (servo_dashboard.html must be beside this script).
Keys: c start/stop CSV evidence capture, +/- expected frequency by 0.1 Hz,
      0 hide expected frequency, q quit.

Terminology: Hz measures CSI oscillations, NOT a validated human heart rate.
A servo may produce two movement-related CSI peaks per commanded servo cycle;
observation of a second harmonic is not automatically an error.
"""
from __future__ import annotations

import argparse
import csv
import curses
import logging
import math
import os
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal

from decoder import BW_TO_NFFT, CHIP_TABLE, DecodeError, decode_frame_payload

LOGGER = logging.getLogger("servo_frequency_monitor")
SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values, width=65):
    y = np.asarray(values, dtype=float)[-width:]
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return " " * width
    lo, hi = np.percentile(y, [5, 95]) if len(y) >= 10 else (y.min(), y.max())
    span = max(float(hi - lo), 1e-12)
    index = np.clip(((y - lo) / span * 7).astype(int), 0, 7)
    return "".join(SPARK[x] for x in index).rjust(width)


def new_result(status="WAITING FOR CSI"):
    return dict(status=status, frequency=None, ratio=0.0, coverage=0.0,
                n_used=0, supported=0, n_carriers=0, strongest_sc=None,
                waveform=[], spectrum=[], spectrum_f=[], peaks=[],
                sampling_rate=0.0, duration=0.0, gap=0.0,
                resolution=0.0, updated=time.time(), processing_ms=0.0)


class Recorder:
    """Output compatible with read_csv_csi.py: complex CSI columns and timestamps."""
    def __init__(self, out_dir, nfft, capture_mode, source="live"):
        self.out_dir = Path(out_dir)
        self.nfft = nfft
        self.capture_mode = capture_mode
        self.source = source
        self.lock = threading.Lock()
        self.raw_file = self.results_file = None
        self.raw_writer = self.results_writer = None
        self.raw_path = self.results_path = None
        self.rows = 0
        self.dropped = 0

    def start(self):
        with self.lock:
            if self.raw_file:
                return
            self.out_dir.mkdir(parents=True, exist_ok=True)
            tag = time.strftime("%Y%m%d_%H%M%S")
            self.raw_path = self.out_dir / f"{self.source}_servo_csi_{tag}.csv"
            self.results_path = self.out_dir / f"{self.source}_servo_estimates_{tag}.csv"
            self.raw_file = self.raw_path.open("w", newline="", buffering=1)
            self.results_file = self.results_path.open("w", newline="", buffering=1)
            self.raw_writer = csv.writer(self.raw_file)
            self.results_writer = csv.writer(self.results_file)
            header = ["ts", "seq_num", "core", "spatial_stream"]
            if self.capture_mode == "complex":
                for k in range(self.nfft):
                    header.extend((f"sc{k}_re", f"sc{k}_im"))
            else:
                ext = "amp" if self.capture_mode == "amplitude" else "pha"
                header += [f"sc{k}_{ext}" for k in range(self.nfft)]
            self.raw_writer.writerow(header)
            self.results_writer.writerow(["ts", "source", "status", "measured_hz", "expected_hz",
                                         "abs_error_hz", "peak_to_median", "support_count",
                                         "total_carriers", "strongest_sc", "sampling_rate_hz",
                                         "window_seconds", "packet_coverage", "max_gap_s"])
            self.rows = 0

    def stop(self):
        with self.lock:
            for f in (self.raw_file, self.results_file):
                if f:
                    f.flush()
                    f.close()
            self.raw_file = self.results_file = None
            self.raw_writer = self.results_writer = None

    def toggle(self):
        if self.active:
            self.stop()
        else:
            self.start()

    @property
    def active(self):
        with self.lock:
            return self.raw_file is not None

    def write_frame(self, frame):
        with self.lock:
            if self.raw_writer is None:
                return
            row = [f"{frame.ts:.6f}", int(frame.seq_num),
                   int(frame.core), int(frame.spatial_stream)]
            if self.capture_mode == "complex":
                csi = np.asarray(frame.csi)
                iq = np.column_stack((csi.real, csi.imag)).ravel()
                row.extend(iq)
            elif self.capture_mode == "amplitude":
                row.extend(np.abs(frame.csi))
            else:
                row.extend(np.angle(frame.csi))
            self.raw_writer.writerow(row)
            self.rows += 1
            if self.rows % 100 == 0:
                self.raw_file.flush()

    def write_result(self, result, expected_hz):
        with self.lock:
            if self.results_writer is None:
                return
            f = result.get("frequency")
            diff = abs(f - expected_hz) if f is not None and expected_hz is not None else ""
            self.results_writer.writerow([
                f"{time.time():.6f}", self.source, result["status"], f if f is not None else "",
                expected_hz if expected_hz is not None else "", diff,
                result["ratio"], result["supported"], result["n_carriers"],
                result["strongest_sc"], result["sampling_rate"],
                result["duration"], result["coverage"], result["gap"]])
            self.results_file.flush()


class State:
    def __init__(self, seconds, recorder):
        self.lock = threading.Lock()
        self.ts = deque()
        self.csi = deque()
        self.keep_seconds = max(25.0, seconds + 4.0)
        self.last_frame = None
        self.total_seen = self.total_decoded = self.errors = self.ignored = 0
        # Staged counters allow diagnosing traffic even when decoding is 0.
        self.udp_seen = self.port_seen = self.source_seen = 0
        self.last_udp = self.last_decode_error = self.last_non_udp = ""
        self.last_payload_bytes = 0
        self.packet_times = deque(maxlen=200)
        self.last_rx_mono = 0.0
        self.start_wall = time.time()
        self.result = new_result()
        self.last_error = ""
        self.frequency_history = deque(maxlen=60)
        self.finished = False
        self.finished_at = 0.0
        self.recorder = recorder
        self.selected_core_stream = None
        self.expected = None

    def handle_frame(self, frame, args):
        with self.lock:
            # Never combine multiple antennas/streams in a single time series.
            cs = (frame.core, frame.spatial_stream)
            if args.select_core is not None:
                if cs != (args.select_core, args.select_stream):
                    self.ignored += 1
                    return
                self.selected_core_stream = cs
            elif self.selected_core_stream is None:
                self.selected_core_stream = cs
            elif cs != self.selected_core_stream:
                self.ignored += 1
                return
            if self.ts and frame.ts <= self.ts[-1]:
                self.ignored += 1
                return
            if self.csi and len(frame.csi) != len(self.csi[-1]):
                self.errors += 1
                return
            self.ts.append(float(frame.ts))
            self.csi.append(np.asarray(frame.csi, dtype=np.complex64).copy())
            self.total_decoded += 1
            self.last_frame = frame
            self.last_rx_mono = time.monotonic()
            self.packet_times.append(self.last_rx_mono)
            while len(self.ts) > 2 and self.ts[-1] - self.ts[0] > self.keep_seconds:
                self.ts.popleft()
                self.csi.popleft()
        # Keep reception/decoding independent of signal processing and UI.
        self.recorder.write_frame(frame)

    def window(self, seconds):
        with self.lock:
            if len(self.ts) < 3:
                return None, None
            # Copy only requested samples, with a small allowance for uneven arrival.
            t = np.asarray(self.ts, dtype=np.float64)
            k = int(np.searchsorted(t, t[-1] - seconds, side="left"))
            if len(t) - k < 3:
                return None, None
            x = np.stack(list(self.csi)[k:], axis=0)
            return t[k:], x

    def snapshot(self):
        with self.lock:
            if len(self.packet_times) > 1:
                span = self.packet_times[-1] - self.packet_times[0]
                rate = (len(self.packet_times) - 1) / span if span > 0 else 0
            else:
                rate = 0
            return dict(seen=self.total_seen, decoded=self.total_decoded,
                        errors=self.errors, ignored=self.ignored, rate=rate,
                        result=dict(self.result), core_stream=self.selected_core_stream,
                        frame=self.last_frame, err=self.last_error,
                        expected=self.expected, frequency_history=list(self.frequency_history),
                        finished=self.finished,
                        finished_at=self.finished_at,
                        udp_seen=self.udp_seen, port_seen=self.port_seen,
                        source_seen=self.source_seen, last_udp=self.last_udp,
                        last_decode_error=self.last_decode_error,
                        last_non_udp=self.last_non_udp,
                        last_payload_bytes=self.last_payload_bytes)



# ============================================================================
# READ-ONLY BROWSER DASHBOARD (Python standard library; no Flask required)
# ============================================================================

def lan_ip_hint():
    """Best-effort LAN address for display only; HTTP binds independently."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # No data is sent. UDP connect chooses the outbound local IP.
            sock.connect(("192.0.2.1", 1))
            return sock.getsockname()[0]
    except OSError:
        try:
            ip = socket.gethostbyname(socket.gethostname())
            return ip if not ip.startswith('127.') else 'PI_IP_ADDRESS'
        except OSError:
            return 'PI_IP_ADDRESS'


def web_payload(state, args):
    """Serialize only explicitly approved fields, never a CSI packet or raw IQ."""
    snap = state.snapshot()
    result = snap['result']
    freq = result['frequency']
    expected = snap['expected']
    status = result['status']
    valid = freq is not None and status in ('LIVE', 'MEASURING')
    difference = abs(freq-expected) if valid and expected is not None else None
    if difference is None:
        comparison = 'NOT VERIFIED'
    elif difference <= args.tolerance_hz:
        comparison = 'MATCH'
    elif abs(freq-2*expected) <= args.tolerance_hz:
        comparison = 'POSSIBLE 2ND HARMONIC'
    else:
        comparison = 'DIFFERENT'
    return {
        'server_time': time.time(),
        'mode': 'SYNTHETIC SIMULATION' if args.simulate else 'PCAP REPLAY' if args.pcap else 'LIVE NEXMON CSI',
        'source': args.iface or str(args.pcap) if not args.simulate else 'synthetic',
        'status': status, 'comparison': comparison,
        'frequency_hz': freq, 'rate_equivalent_cpm': freq*60 if freq is not None else None,
        'expected_hz': expected, 'error_hz': difference,
        'tolerance_hz': args.tolerance_hz,
        'peak_to_median': result['ratio'],
        'peak_to_median_db': 10*math.log10(result['ratio']) if result['ratio'] > 0 else None,
        'supported_subcarriers': result['supported'],
        'usable_subcarriers': result['n_carriers'],
        'best_subcarrier': result['strongest_sc'],
        'bandwidth_mhz': args.bw, 'chip': args.chip,
        'udp_port': args.udp_port, 'core_stream': snap['core_stream'],
        'packets_decoded': snap['decoded'], 'packets_seen': snap['seen'],
        'decode_errors': snap['errors'], 'packets_ignored': snap['ignored'],
        'packet_rate': snap['rate'], 'timestamp_fs': result['sampling_rate'],
        'coverage_pct': result['coverage']*100,
        'max_gap_ms': result['gap']*1000,
        'window_seconds': args.window, 'window_filled_seconds': result['duration'],
        'resolution_hz': result['resolution'],
        'processing_ms': result['processing_ms'],
        'update_seconds': args.update_seconds,
        'result_timestamp': result['updated'],
        'waveform': result['waveform'],
        'spectrum_hz': result['spectrum_f'],
        'spectrum_power': result['spectrum'],
        'top_peaks_hz': result['peaks'],
        'frequency_history_hz': snap['frequency_history'],
        'recording': state.recorder.active,
        'recorded_frames': state.recorder.rows,
        'error_message': snap['err'],
        'finished': snap['finished'],
        'notes': 'Cycles/min is a mathematical equivalent, not a validated human heart-rate reading.'
    }


def start_web_dashboard(state, args):
    """Start a read-only HTTP server sharing the same State as curses UI."""
    html_path = Path(__file__).resolve().with_name('servo_dashboard.html')
    if not html_path.is_file():
        raise FileNotFoundError(f'Place servo_dashboard.html next to {Path(__file__).name}')
    html_bytes = html_path.read_bytes()

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            from urllib.parse import urlsplit
            path = urlsplit(self.path).path
            if path in ('/', '/index.html'):
                body = html_bytes
                content_type = 'text/html; charset=utf-8'
                code = 200
            elif path == '/api/status':
                try:
                    body = json.dumps(web_payload(state, args), allow_nan=False,
                                      separators=(',', ':')).encode('utf-8')
                    code = 200
                except Exception as exc:
                    LOGGER.exception('web API serialization error')
                    body = json.dumps({'error': str(exc)}).encode('utf-8')
                    code = 500
                content_type = 'application/json; charset=utf-8'
            elif path == '/health':
                body = b'ok'
                content_type = 'text/plain; charset=utf-8'
                code = 200
            else:
                body = b'Not found'
                content_type = 'text/plain; charset=utf-8'
                code = 404
            try:
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy',
                                 "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                 "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                                 "connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, fmt, *values):
            # Avoid mixing HTTP request logs into the curses dashboard.
            LOGGER.debug('web %s', fmt % values)

    # Bind only to the requested interface. By default available on the LAN.
    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Server((args.web_host, args.web_port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever,
                              name='servo-web-dashboard', daemon=True)
    thread.start()
    return server, thread


def analyze_csi(timestamps, csi, args):
    """Blind spectral analysis: expected servo Hz is never an input to this routine.

    Steps: validate window, resample irregular timestamps to uniform 100 Hz,
    robustly suppress spikes, detrend+bandpass amplitude, compute PSD for each
    usable subcarrier, average *normalized* spectra, find global band peak.
    """
    tstart = time.monotonic()
    result = new_result("WARMING UP")
    if csi is None or len(timestamps) < 30:
        return result
    t = np.asarray(timestamps, dtype=np.float64)
    x = np.asarray(csi)
    duration = float(t[-1] - t[0])
    result["duration"] = duration
    if duration < args.min_seconds:
        return result
    dt = np.diff(t)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        result["status"] = "BAD TIMESTAMPS"
        return result
    result["sampling_rate"] = float(1.0 / np.median(dt))
    result["gap"] = float(np.max(dt))
    # Record measured sample coverage; use target resampling rate as reference.
    result["coverage"] = min(1.0, float(len(t) / (duration * args.sample_rate)))
    if result["gap"] > args.max_gap:
        result["status"] = "PACKET GAP"
        return result
    if result["coverage"] < args.min_coverage:
        result["status"] = "LOW PACKET RATE"
        return result

    fs = args.sample_rate
    regular_t = np.arange(t[0], t[-1], 1.0 / fs)
    if len(regular_t) < 128:
        return result
    # np.interp is 1D: vectorized interpolation along packet/time axis.
    # Interpolate each amplitude channel, not complex phase, for this demo.
    from scipy.interpolate import interp1d
    amps = np.abs(x).astype(np.float32)
    y = interp1d(t - t[0], amps, axis=0, bounds_error=False,
                 fill_value="extrapolate", assume_sorted=True)(regular_t - t[0])
    valid = np.all(np.isfinite(y), axis=0)
    mean_amp = np.mean(y, axis=0)
    std_amp = np.std(y, axis=0)
    # Reject null/unused subcarriers and effectively constant subcarriers.
    valid &= (mean_amp > 1e-5) & (std_amp / (mean_amp + 1e-12) > 1e-4)
    if not np.any(valid):
        result["status"] = "NO USABLE SUBCARRIERS"
        return result
    indices = np.flatnonzero(valid)
    y = y[:, valid].astype(np.float64, copy=False)

    # Robust clipping reduces sporadic amplitude spikes without the expensive
    # per-sample Hampel filter. MAD scale is only used to cap *outliers*.
    mid = np.median(y, axis=0)
    mad = np.median(np.abs(y - mid), axis=0)
    cap = 8.0 * 1.4826 * np.maximum(mad, 1e-6)
    y = np.clip(y, mid - cap, mid + cap)
    y = signal.detrend(y, axis=0, type="linear")
    sos = signal.butter(3, [args.low_hz, args.high_hz],
                        fs=fs, btype="bandpass", output="sos")
    y = signal.sosfiltfilt(sos, y, axis=0)
    standard_deviation = np.std(y, axis=0)
    use = standard_deviation > 1e-8
    if not np.any(use):
        result["status"] = "NO BAND ENERGY"
        return result
    y = y[:, use]
    indices = indices[use]
    y /= standard_deviation[use]
    n = len(y)
    # Entire window gives true frequency resolution ≈ 1 / window seconds.
    # Padding smooths peak display but does not increase true resolution.
    nfft = int(2 ** math.ceil(math.log2(n * 2)))
    freq, psd = signal.periodogram(y, fs=fs, window="hann", nfft=nfft,
                                   axis=0, scaling="density", detrend=False)
    inside = (freq >= args.low_hz) & (freq <= args.high_hz)
    fb, pb = freq[inside], psd[inside]
    if len(fb) < 4:
        result["status"] = "BAND TOO NARROW"
        return result
    # Every subcarrier contributes equal *relative* band power. No expected
    # frequency is used to guide the selection.
    total_power = np.sum(pb, axis=0)
    use = total_power > 1e-14
    if not np.any(use):
        result["status"] = "NO BAND ENERGY"
        return result
    pb = pb[:, use]
    y = y[:, use]
    indices = indices[use]
    normalized = pb / np.maximum(np.sum(pb, axis=0, keepdims=True), 1e-20)
    # A few motion-sensitive carriers can be drowned out by hundreds of weak
    # carriers. Give spectral concentration some weight, but preserve a robust
    # unweighted component; no target/servo setting is involved in ranking.
    sc_medians = np.median(pb, axis=0)
    concentrations = np.max(pb, axis=0) / np.maximum(sc_medians, 1e-20)
    weights = np.clip(np.maximum(concentrations - 3.0, 0.0)**2, 0.0, 400.0)
    average_psd = np.mean(normalized, axis=1)
    if np.sum(weights) > 1e-12:
        weighted_psd = (normalized @ weights) / np.sum(weights)
        mean_psd = 0.35 * average_psd + 0.65 * weighted_psd
    else:
        mean_psd = average_psd
    peak_index = int(np.argmax(mean_psd))
    # Sub-bin parabolic interpolation on log PSD improves displayed Hz.
    peak_hz = float(fb[peak_index])
    if 0 < peak_index < len(mean_psd) - 1:
        l, c, r = np.log(np.maximum(mean_psd[peak_index-1:peak_index+2], 1e-20))
        den = l - 2*c + r
        if abs(den) > 1e-12:
            offset = float(np.clip(0.5 * (l - r) / den, -0.5, 0.5))
            peak_hz = float(fb[peak_index] + offset * (fb[1] - fb[0]))

    # Count carriers whose own strongest band peak agrees with global peak.
    local_top = fb[np.argmax(pb, axis=0)]
    tolerance = max(1.5 / duration, args.vote_tolerance)
    agree = np.abs(local_top - peak_hz) <= tolerance
    n_agree = int(np.count_nonzero(agree))
    ratio = float(mean_psd[peak_index] / (np.median(mean_psd) + 1e-12))
    # Select one actual SC for illustrative time trace, avoiding phase
    # cancellation from averaging out-of-phase motion-sensitive subcarriers.
    candidates = np.flatnonzero(agree)
    if len(candidates) == 0:
        candidates = np.arange(len(indices))
    best_local = int(candidates[np.argmax(normalized[peak_index, candidates])])
    wave = y[:, best_local]
    # Last ~8 seconds of an actual measured subcarrier, never invented waveform.
    last_n = int(min(len(wave), fs * 8))
    step = max(1, last_n // 70)
    trace = wave[-last_n::step].tolist()

    # Significant spectral local maxima, useful for spotting 2x harmonics.
    pk, _ = signal.find_peaks(mean_psd)
    pk = sorted(pk, key=lambda p: mean_psd[p], reverse=True)
    peak_list = [float(fb[p]) for p in pk[:3]]
    if not pk or pk[0] != peak_index:
        peak_list.insert(0, peak_hz)
    spectrum_i = np.linspace(0, len(fb) - 1, 65).astype(int)
    result.update(frequency=peak_hz, ratio=ratio, supported=n_agree,
                  n_carriers=len(indices), strongest_sc=int(indices[best_local]),
                  waveform=trace, spectrum=mean_psd[spectrum_i].tolist(),
                  spectrum_f=fb[spectrum_i].tolist(), peaks=peak_list[:3],
                  n_used=len(t), resolution=1.0/duration,
                  status="MEASURING" if duration < args.window - 1.0 else "LIVE",
                  processing_ms=(time.monotonic() - tstart)*1000, updated=time.time())
    if ratio < args.min_ratio or n_agree < args.min_support:
        result["status"] = "WEAK/UNVERIFIED PEAK"
    return result


def frequency_worker(state, args, stop):
    while not stop.is_set():
        start = time.monotonic()
        with state.lock:
            last_rx = state.last_rx_mono
            finished = state.finished
        t, x = state.window(args.window)
        if t is not None or (last_rx and not finished):
            try:
                if not finished and last_rx and time.monotonic() - last_rx > max(3.0, 2 * args.update_seconds):
                    result = new_result("NO RECENT CSI PACKETS")
                else:
                    result = analyze_csi(t, x, args)
            except Exception as exc:
                LOGGER.exception("analysis failure")
                result = new_result("ANALYSIS ERROR")
                with state.lock:
                    state.last_error = str(exc)
            with state.lock:
                state.result = result
                if result['frequency'] is not None and result['status'] in ('MEASURING', 'LIVE'):
                    state.frequency_history.append(result['frequency'])
                expected = state.expected
            state.recorder.write_result(result, expected)
        stop.wait(max(0.1, args.update_seconds - (time.monotonic()-start)))


def sniff_live(state, args, stop):
    """Receive Nexmon CSI; software-filter UDP/5500 by default.

    Packet arrival -> UDP -> dest port -> Nexmon src IP -> decode -> core/stream
    are intentionally independent counters. In particular, `seen` always
    counts packets reaching Scapy (not only successfully decoded CSI).
    """
    try:
        from scapy.all import AsyncSniffer, UDP, IP
    except ImportError:
        with state.lock:
            state.last_error = "Scapy missing; install with: python -m pip install scapy"
        stop.set()
        return

    def handle(pkt):
        with state.lock:
            state.total_seen += 1
            sequence_index = state.total_seen
            if not state.last_non_udp and UDP not in pkt:
                state.last_non_udp = str(pkt.summary())[:120]

        if UDP not in pkt:
            return
        udp = pkt[UDP]
        with state.lock:
            state.udp_seen += 1
            if IP in pkt:
                state.last_udp = (f"{pkt[IP].src}:{udp.sport} -> "
                                  f"{pkt[IP].dst}:{udp.dport}")
            else:
                state.last_udp = f"non-IPv4 UDP dport={udp.dport}"

        if udp.dport != args.udp_port:
            return
        with state.lock:
            state.port_seen += 1

        if args.src_ip and (IP not in pkt or pkt[IP].src != args.src_ip):
            return
        payload = bytes(udp.payload)
        with state.lock:
            state.source_seen += 1
            state.last_payload_bytes = len(payload)

        try:
            frame = decode_frame_payload(
                payload, index=sequence_index, ts=time.time(),
                chip=args.chip, bw=args.bw, nfft_override=args.nfft,
                header_format=args.header_format,
            )
        except DecodeError as exc:
            with state.lock:
                state.errors += 1
                state.last_decode_error = str(exc)[:160]
            return
        state.handle_frame(frame, args)

    sniff = None
    try:
        # DO NOT enable the Scapy BPF filter by default. The user's Pi 5
        # Nexmon export produces valid UDP/5500 that this filter did not pass.
        # Software filtering above is sufficient at ~100 CSI frames/sec.
        capture_filter = f"udp port {args.udp_port}" if args.bpf_filter else None
        sniff = AsyncSniffer(iface=args.iface, filter=capture_filter,
                             prn=handle, store=False)
        sniff.start()
        while not stop.wait(0.25):
            if not sniff.running:
                raise RuntimeError("packet sniffer terminated")
    except Exception as exc:
        LOGGER.exception("sniffing failure")
        with state.lock:
            state.last_error = f"Sniffer: {exc} (check interface and sudo/Npcap)"
        stop.set()
    finally:
        if sniff is not None and sniff.running:
            try:
                sniff.stop()
            except Exception:
                pass


def play_pcap(state, args, stop):
    try:
        from scapy.all import PcapReader, UDP, IP
        with PcapReader(str(args.pcap)) as reader:
            first_capture_ts, first_wall = None, None
            for pkt in reader:
                if stop.is_set():
                    break
                if UDP not in pkt or pkt[UDP].dport != args.udp_port:
                    continue
                if args.src_ip and (IP not in pkt or pkt[IP].src != args.src_ip):
                    continue
                pkt_ts = float(pkt.time)
                if first_capture_ts is None:
                    first_capture_ts, first_wall = pkt_ts, time.time()
                target = first_wall + (pkt_ts - first_capture_ts) / args.playback_speed
                delay = target - time.time()
                if delay > 0 and stop.wait(delay):
                    break
                with state.lock:
                    state.total_seen += 1
                try:
                    frame = decode_frame_payload(bytes(pkt[UDP].payload),
                                                 index=state.total_seen,
                                                 ts=first_wall + (pkt_ts-first_capture_ts),
                                                 chip=args.chip, bw=args.bw, nfft_override=args.nfft,
                                                 header_format=args.header_format)
                except DecodeError:
                    with state.lock:
                        state.errors += 1
                    continue
                state.handle_frame(frame, args)
        with state.lock:
            state.finished = True
            state.finished_at = time.time()
    except Exception as exc:
        LOGGER.exception("PCAP replay failure")
        with state.lock:
            state.last_error = f"Replay: {exc}"
        stop.set()


def synthetic_csi(t, hz, nfft=256, seed=28):
    """Deterministic test pattern, NOT actual CSI sensing data."""
    rng = np.random.default_rng(seed)
    sc = np.arange(nfft)
    baseline = 110 + rng.normal(0, 12, nfft)
    # Some carriers respond strongly, most weakly. A second harmonic exists.
    modulation = rng.uniform(0.5, 1.3, nfft)
    responsive = rng.choice(nfft, 45, replace=False)
    modulation[responsive] = rng.uniform(5, 12, len(responsive))
    phase = rng.uniform(-0.6, 0.6, nfft)
    oscillation = (np.sin(2*np.pi*hz*t[:, None] + phase) +
                   0.27*np.sin(2*np.pi*2*hz*t[:, None] + 2*phase))
    amp = baseline + modulation*oscillation + rng.normal(0, 2.5, (len(t), nfft))
    carrier_phase = sc / 15.0 + 0.04*np.sin(2*np.pi*hz*t[:, None]+phase)
    return (amp * np.exp(1j*carrier_phase)).astype(np.complex64)


def simulate_feed(state, args, stop):
    rng = np.random.default_rng(2026)
    start = time.time()
    n = 0
    # This mode is visibly flagged as synthetic on screen and in CLI.
    while not stop.is_set():
        elapsed = time.time() - start
        n += 1
        if rng.random() < 0.005:
            stop.wait(1.0/args.sample_rate)
            continue
        hz = args.simulate_hz
        data = synthetic_frame(elapsed, hz, args.nfft or BW_TO_NFFT[args.bw], rng)
        from types import SimpleNamespace
        frame = SimpleNamespace(ts=time.time(), seq_num=n % 4096, core=0,
                                spatial_stream=0, csi=data)
        with state.lock:
            state.total_seen += 1
        state.handle_frame(frame, args)
        stop.wait(max(0, start + (n + 1) / args.sample_rate - time.time()))


def synthetic_frame(elapsed, hz, nfft, rng):
    # Reproducible fixed 'channel' response, variable Gaussian noise.
    sc = np.arange(nfft)
    phase = 0.2 + sc / 47.0
    sensitivity = np.where((sc % 7) < 2, 7.0, 0.7)
    amp = (120 + sensitivity * np.sin(2*np.pi*hz*elapsed + phase)
           + 0.25 * sensitivity * np.sin(2*np.pi*2*hz*elapsed + phase*2)
           + rng.normal(0, 2.0, nfft))
    return (amp * np.exp(1j*sc/20.0)).astype(np.complex64)


def safe_put(screen, row, txt, style=0):
    h, w = screen.getmaxyx()
    if 0 <= row < h-1 and w >= 2:
        try:
            screen.addnstr(row, 0, str(txt), w-1, style)
        except curses.error:
            pass


def display(stdscr, state, args, stop):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    try:
        curses.start_color()
        if curses.has_colors():
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.init_pair(4, curses.COLOR_CYAN, -1)
    except curses.error:
        pass
    while not stop.is_set():
        key = stdscr.getch()
        if key in (ord('q'), ord('Q')):
            stop.set()
            break
        if key in (ord('c'), ord('C')):
            state.recorder.toggle()
        if key in (ord('+'), ord('='), ord('-'), ord('_'), ord('0')):
            with state.lock:
                if key == ord('0'):
                    state.expected = None
                else:
                    base = state.expected if state.expected is not None else 1.0
                    state.expected = round(max(0.1, base + (0.1 if key in (ord('+'), ord('=')) else -0.1)), 2)

        if args.duration and time.time() - state.start_wall >= args.duration:
            stop.set()
            break
        snap = state.snapshot()
        result = snap['result']
        f = result.get('frequency')
        exp = snap['expected']
        valid = f is not None and result['status'] in ('LIVE', 'MEASURING')
        if exp is not None and valid:
            error = abs(f-exp)
            comparison = ('MATCH' if error <= args.tolerance_hz else
                          '2nd harmonic?' if abs(f-2*exp) <= args.tolerance_hz else 'DIFFERENT')
        else:
            error = None
            comparison = 'NOT VERIFIED'
        stdscr.erase()
        kind = 'SYNTHETIC SIMULATION - NOT REAL CSI' if args.simulate else 'PCAP REPLAY' if args.pcap else 'LIVE NEXMON CSI'
        safe_put(stdscr, 0, f'CSI-HEART | SERVO MOVEMENT FREQUENCY MONITOR  [{kind}]', curses.A_BOLD)
        safe_put(stdscr, 1, '-' * 90)
        safe_put(stdscr, 2, f"Packets: decoded {snap['decoded']:,} | seen {snap['seen']:,} | errors {snap['errors']} | other streams {snap['ignored']} | rate {snap['rate']:.1f}/s")
        sc = snap['core_stream']
        safe_put(stdscr, 3, f"Input: {args.iface or args.pcap or 'SYNTHETIC'} | core/stream {sc or 'auto'} | chip {args.chip} | bandwidth {args.bw} MHz | UDP {args.udp_port}")
        safe_put(stdscr, 4, f"Recorder: {'ON ' + str(state.recorder.rows) + ' frames' if state.recorder.active else 'OFF'} | [C] capture CSI+frequency CSV | [Q] quit")
        mode = 'BPF udp filter' if args.bpf_filter else 'software UDP filter (Nexmon-compatible)'
        safe_put(stdscr, 5, f"Capture: {mode} | raw {snap['seen']:,} | UDP {snap['udp_seen']:,} | port {args.udp_port} {snap['port_seen']:,} | source OK {snap['source_seen']:,}")
        safe_put(stdscr, 6, 'MEASURED DOMINANT CSI FREQUENCY', curses.A_BOLD)
        safe_put(stdscr, 7, f"     {f:6.3f} Hz   ({f*60:6.1f} cycles/min)" if f is not None else '       ---.--- Hz   (waiting for sufficient CSI)')
        safe_put(stdscr, 8, f"Expected servo frequency: {exp:.2f} Hz" if exp is not None else 'Expected servo frequency: not set')
        safe_put(stdscr, 9, f"Comparison: {comparison}" + (f" | absolute error {error:.3f} Hz" if error is not None else '') + '    [+/-] adjust, [0] clear')
        status = result['status']
        if args.debug_capture or (snap['decoded'] == 0 and time.time() - state.start_wall >= 3):
            diagnostic = (f"last UDP: {snap['last_udp']} | payload {snap['last_payload_bytes']} bytes"
                          if snap['last_udp'] else
                          f"no UDP captured; sample other packet: {snap['last_non_udp'] or 'none'}")
            if snap['last_decode_error']:
                diagnostic += f" | decode: {snap['last_decode_error']}"
            safe_put(stdscr, 10, diagnostic)
        safe_put(stdscr, 11, f"Signal status: {status} | spectral peak/median {result['ratio']:.1f}x | supporting SCs {result['supported']}/{result['n_carriers']}")
        safe_put(stdscr, 12, f"Window {result['duration']:.1f}/{args.window:.0f}s | resolution ~{result['resolution']:.3f} Hz | timing fs {result['sampling_rate']:.1f} Hz | coverage {result['coverage']*100:.0f}% | largest gap {result['gap']:.3f}s")
        safe_put(stdscr, 13, f"Waveform uses subcarrier {result['strongest_sc']} | update {args.update_seconds:.1f}s | computation {result['processing_ms']:.0f} ms")
        safe_put(stdscr, 15, 'AMPLITUDE OSCILLATION (last ~8 seconds, band-passed; one real subcarrier)', curses.A_BOLD)
        safe_put(stdscr, 16, sparkline(result['waveform'], width=min(70, max(15, stdscr.getmaxyx()[1]-2))))
        safe_put(stdscr, 18, f"NORMALIZED FREQUENCY SPECTRUM ({args.low_hz:.2f} to {args.high_hz:.2f} Hz)", curses.A_BOLD)
        safe_put(stdscr, 19, sparkline(result['spectrum'], width=min(65, max(15, stdscr.getmaxyx()[1]-2))))
        if f is not None:
            safe_put(stdscr, 20, f"Top peaks: {', '.join(f'{p:.3f} Hz' for p in result['peaks'])} | shown frequency is independently detected")
        past = snap['frequency_history']
        if past:
            safe_put(stdscr, 21, "FREQUENCY TREND (recent analyses, 1 point/update): " +
                     sparkline(past, width=min(35, max(10, stdscr.getmaxyx()[1]-56))))
        safe_put(stdscr, 22, "DEMO CHECK: servo OFF -> ON -> change servo period -> OFF; verify frequency changes and disappears.")
        safe_put(stdscr, 23, "Note: strong peaks can be harmonics, Wi-Fi/network periodicity or ambient motion.")
        if not args.no_web:
            safe_put(stdscr, 24, f'WEB DASHBOARD: http://{args.web_display_ip}:{args.web_port}  (Windows browser, same LAN)')
        if snap['err']:
            safe_put(stdscr, 25, f"ERROR: {snap['err']}")
        if snap['finished']:
            safe_put(stdscr, 26, "PCAP replay finished; analysis result remains visible. Press Q to quit.")
        stdscr.refresh()
        stop.wait(1.0 / args.refresh_hz)


def console_display(state, args, stop):
    """CI/headless testing, no terminal needed. Also usable on Windows terminals."""
    last_update = None
    last_diag_time = 0.0
    while not stop.wait(0.25):
        snap = state.snapshot()
        res = snap['result']
        if res['updated'] != last_update and res['updated']:
            f = res['frequency']
            print(f"[{time.strftime('%H:%M:%S')}] mode={'SYNTHETIC' if args.simulate else 'PCAP REPLAY' if args.pcap else 'LIVE'} "
                  f"frequency={f'{f:.3f}' if f is not None else '--'}Hz "
                  f"status={res['status']} power_ratio={res['ratio']:.2f} "
                  f"support={res['supported']}/{res['n_carriers']} "
                  f"packets={snap['decoded']}", flush=True)
            last_update = res['updated']
        # Show packet progression if requested, or automatically on capture failure.
        if ((args.debug_capture or
             (snap['decoded'] == 0 and time.time() - state.start_wall >= 3))
                and time.monotonic() - last_diag_time >= 2.0):
            print(f"  CAPTURE raw={snap['seen']} UDP={snap['udp_seen']} "
                  f"port{args.udp_port}={snap['port_seen']} "
                  f"sourceOK={snap['source_seen']} decoded={snap['decoded']} "
                  f"errors={snap['errors']} ignored={snap['ignored']}", flush=True)
            print(f"          last_udp={snap['last_udp'] or '-'} "
                  f"payload_bytes={snap['last_payload_bytes']} "
                  f"decode_error={snap['last_decode_error'] or '-'}", flush=True)
            last_diag_time = time.monotonic()
        if snap['err']:
            print('ERROR:', snap['err'], flush=True)
            stop.set()
        if args.duration and time.time()-state.start_wall >= args.duration:
            stop.set()
        if snap['finished'] and res['updated'] >= snap['finished_at']:
            stop.set()


def run_self_test(args):
    """Deterministic synthetic analysis sanity check, not a claim of hardware validation."""
    print('SELF-TEST: SYNTHETIC CSI ONLY (not live hardware)')
    fs = args.sample_rate
    t = np.arange(0, args.window, 1/fs)
    ok = True
    for freq in (0.6, 1.0, 1.6, 2.25):
        csi = synthetic_csi(t, freq, args.nfft or BW_TO_NFFT[args.bw], seed=13)
        result = analyze_csi(t, csi, args)
        estimated = result['frequency']
        err = abs(estimated-freq) if estimated is not None else float('inf')
        passed = err <= max(0.09, 2.0/args.window)
        ok &= passed
        print(f"  injected={freq:.2f} Hz  detected={estimated!s} Hz  "
              f"error={err:.3f} Hz  {result['status']}  {'PASS' if passed else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--iface', help='Live sniffing interface, e.g. wlan0 or Ethernet')
    source.add_argument('--pcap', type=Path, help='Replay a real Nexmon UDP PCAP/PCAPNG')
    source.add_argument('--simulate', action='store_true', help='Synthetic demo, visually labelled on-screen')
    source.add_argument('--self-test', action='store_true', help='Run offline synthetic detection tests')
    p.add_argument('--chip', choices=list(CHIP_TABLE), default='43455c0')
    p.add_argument('--bw', type=int, choices=list(BW_TO_NFFT), default=80)
    p.add_argument('--nfft', type=int, help='Override FFT length from bandwidth')
    p.add_argument('--udp-port', type=int, default=5500)
    p.add_argument('--src-ip', default='10.10.10.10', help="Use '' to disable source-IP check")
    p.add_argument('--header-format', choices=['auto', 'legacy', 'rssi'], default='auto')
    p.add_argument('--select-core', type=int, help='Force one CSI core, instead of auto-locking')
    p.add_argument('--select-stream', type=int, help='Force one spatial stream, instead of auto-locking')
    p.add_argument('--window', type=float, default=16.0, help='Rolling frequency estimation window (seconds)')
    p.add_argument('--min-seconds', type=float, default=10.0, help='Minimum elapsed data before estimating')
    p.add_argument('--update-seconds', type=float, default=1.0)
    p.add_argument('--refresh-hz', type=float, default=5.0)
    p.add_argument('--low-hz', type=float, default=0.3, help='Lower motion frequency boundary (Hz)')
    p.add_argument('--high-hz', type=float, default=3.5, help='Upper motion frequency boundary (Hz)')
    p.add_argument('--sample-rate', type=float, default=100.0, help='Uniform resampling frequency (Hz)')
    p.add_argument('--max-gap', type=float, default=0.6, help='Reject windows containing larger packet gaps')
    p.add_argument('--min-coverage', type=float, default=0.55, help='Minimum packets relative to target rate')
    p.add_argument('--min-ratio', type=float, default=1.8, help='Minimum averaged PSD peak-to-median ratio')
    p.add_argument('--min-support', type=int, default=4, help='Minimum carriers supporting the same peak')
    p.add_argument('--vote-tolerance', type=float, default=0.12, help='Minimum carrier agreement width (Hz)')
    p.add_argument('--expected-hz', type=float, help='Known servo full-cycle oscillation frequency for comparison only')
    p.add_argument('--tolerance-hz', type=float, default=0.15, help='Acceptable deviation from expected Hz')
    p.add_argument('--output-dir', type=Path, default=Path('output'))
    p.add_argument('--capture-mode', choices=['complex', 'amplitude', 'phase'], default='complex')
    p.add_argument('--record-on-start', action='store_true', help='Start CSI/estimate CSV capture automatically')
    p.add_argument('--playback-speed', type=float, default=1.0, help='PCAP playback factor; 1.0 is realtime')
    p.add_argument('--simulate-hz', type=float, default=1.2, help='Injected synthetic frequency, not measured hardware')
    p.add_argument('--no-ui', action='store_true', help='Simple headless text output instead of curses')
    p.add_argument('--no-web', action='store_true', help='Disable browser dashboard (enabled by default)')
    p.add_argument('--web-host', default='0.0.0.0', help='Web server bind address; 0.0.0.0 allows LAN access')
    p.add_argument('--web-port', type=int, default=8080, help='Web dashboard TCP port')
    p.add_argument('--debug-capture', action='store_true',
                   help='Show staged packet counters/last UDP and decoding errors')
    p.add_argument('--bpf-filter', action='store_true',
                   help='Opt IN to kernel UDP BPF capture filter (may drop Nexmon packets on Pi)')
    p.add_argument('--duration', type=float, help='Optional runtime cap in seconds (useful for --no-ui testing)')
    args = p.parse_args(argv)
    if (args.select_core is None) != (args.select_stream is None):
        p.error('--select-core and --select-stream must be supplied together')
    if not (0 < args.low_hz < args.high_hz < args.sample_rate/2):
        p.error('Require 0 < low-hz < high-hz < sample-rate/2')
    if args.window <= args.min_seconds or args.min_seconds < 2:
        p.error('--window must be greater than --min-seconds, and --min-seconds >= 2')
    if args.update_seconds <= 0 or args.refresh_hz <= 0 or args.playback_speed <= 0:
        p.error('Update/refresh/playback rates must be positive')
    if args.min_coverage <= 0 or args.min_coverage > 1:
        p.error('--min-coverage must be in (0,1]')
    if not 1 <= args.web_port <= 65535:
        p.error('--web-port must be in 1..65535')
    if args.expected_hz is not None and args.expected_hz <= 0:
        p.error('--expected-hz must be positive')
    if args.pcap and not args.pcap.exists():
        p.error(f'PCAP does not exist: {args.pcap}')
    if args.src_ip == '':
        args.src_ip = None
    if args.self_test:
        return run_self_test(args)
    nfft = args.nfft or BW_TO_NFFT[args.bw]
    source_label = "synthetic" if args.simulate else "replay" if args.pcap else "live"
    recorder = Recorder(args.output_dir, nfft, args.capture_mode, source=source_label)
    state = State(args.window, recorder)
    state.expected = args.expected_hz
    stop = threading.Event()
    web_server = None
    web_thread = None
    args.web_display_ip = (lan_ip_hint() if args.web_host in ('0.0.0.0', '::')
                           else args.web_host)
    if not args.no_web:
        try:
            web_server, web_thread = start_web_dashboard(state, args)
            print(f'WiPulse web dashboard: http://{args.web_display_ip}:{args.web_port}', flush=True)
            if args.web_display_ip == 'PI_IP_ADDRESS':
                print('Tip: run hostname -I on the Pi to find its LAN IP address.', flush=True)
        except (OSError, FileNotFoundError) as exc:
            args.no_web = True
            print(f'Web dashboard unavailable: {exc}', flush=True)
    if args.record_on_start:
        recorder.start()
    worker = threading.Thread(target=frequency_worker, args=(state, args, stop), daemon=True)
    worker.start()
    if args.simulate:
        target = simulate_feed
    elif args.pcap:
        target = play_pcap
    else:
        target = sniff_live
    receive = threading.Thread(target=target, args=(state, args, stop), daemon=True)
    receive.start()
    try:
        if args.no_ui:
            console_display(state, args, stop)
        else:
            curses.wrapper(display, state, args, stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()
            if web_thread is not None:
                web_thread.join(timeout=2)
        receive.join(timeout=3)
        worker.join(timeout=3)
        recorder.stop()
        if recorder.raw_path:
            print(f'Raw CSI CSV:       {recorder.raw_path}')
            print(f'Frequency log CSV: {recorder.results_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
