"""
decoder.py
==========
Decoding of Nexmon CSI UDP frames captured via tcpdump/Wireshark into pcap
files, e.g.:

    sudo tcpdump -i wlan0 udp port 5500 -w csi_capture.pcap

There are two header layouts in the wild, depending on which nexmon_csi
fork/version produced the capture. Both start at byte 0 of the UDP
*payload* (i.e. after Ethernet/IP/UDP headers have been stripped):

Legacy layout (main seemoo-lab README, older builds), HEADER_LEN=20:
    offset  size  field
    ------  ----  -----------------------------------------------------
    0       6     magic bytes, 0x111111111111 (4-byte + 2-byte magic
                   concatenated back to back)
    6       6     source MAC address of the triggering frame
    12      2     sequence number of the triggering Wi-Fi frame (uint16)
    14      2     core (low 3 bits) / spatial stream (next 3 bits)
    16      2     chanspec used during extraction
    18      2     chip version
    20      -     CSI payload

Newer layout (e.g. zeroby0/nexmonster Pi forks), HEADER_LEN=18:
    offset  size  field
    ------  ----  -----------------------------------------------------
    0       2     magic bytes, 0x1111
    2       1     RSSI (signed int8, dBm)
    3       1     Frame Control byte of the triggering frame
    4       6     source MAC address of the triggering frame
    10      2     sequence number of the triggering Wi-Fi frame (uint16)
    12      2     core (low 3 bits) / spatial stream (next 3 bits)
    14      2     chanspec used during extraction
    16      2     chip version
    18      -     CSI payload

In both layouts, the CSI payload itself is:
    - bcm4339 / bcm43455c0 : interleaved int16 I/Q, nfft*4 bytes
    - bcm4358 / bcm4366c0  : packed float format (9/5 or 12/6
      mantissa/exponent bits, see unpack_float_acphy), nfft*4 bytes

By default this module auto-detects which layout a capture uses (see
`detect_header_format`). Pass --header-format explicitly on the CLI if
auto-detection picks the wrong one.

This module uses scapy only to parse the pcap container / Ethernet-IP-UDP
framing (already a dependency in your capture scripts), and does all
Nexmon-specific decoding itself in pure Python/NumPy.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger("csi_analysis.decoder")

# Two known Nexmon CSI UDP header layouts. "legacy" = main seemoo-lab README
# (6-byte magic). "rssi" = newer forks with 2-byte magic + RSSI + Frame
# Control byte ahead of the MAC (seen in the wild on some Pi builds).
HEADER_FORMATS = {
    "legacy": {"header_len": 20, "magic": b"\x11\x11\x11\x11\x11\x11", "has_rssi_fc": False},
    "rssi":   {"header_len": 18, "magic": b"\x11\x11", "has_rssi_fc": True},
}

# Chip -> ("format", params). "int16" chips give raw I/Q. "float" chips need
# the packed-float unpacking algorithm ported from unpack_float.c.
CHIP_TABLE = {
    "4339":     {"format": "int16"},
    "43455c0":  {"format": "int16"},
    "4358":     {"format": "float", "nbits": 10, "nman": 9,  "nexp": 5},
    "4366c0":   {"format": "float", "nbits": 10, "nman": 12, "nexp": 6},
}

BW_TO_NFFT = {20: 64, 40: 128, 80: 256}


class DecodeError(ValueError):
    """Raised for a malformed / non-CSI frame."""


@dataclass
class CSIFrame:
    """A single decoded CSI frame."""
    index: int                 # position of this frame within the capture
    ts: float                  # capture timestamp (seconds, from pcap header)
    src_mac: str
    seq_num: int
    core: int
    spatial_stream: int
    chanspec: int
    chip_version: int
    csi: np.ndarray            # complex128, length nfft, as-stored subcarrier
                                # order (NOT fftshifted -- see analysis.py)
    rssi: Optional[int] = None         # dBm, only present in "rssi" header format
    frame_control: Optional[int] = None  # only present in "rssi" header format
    fragment_num: int = 0
    raw_len: int = field(repr=False, default=0)


def _mac_str(b: bytes) -> str:
    return ":".join(f"{x:02x}" for x in b)


def detect_header_format(payload: bytes) -> str:
    """
    Best-effort auto-detect of which header layout a UDP payload uses.

    Both layouts start with 0x11 0x11. We disambiguate by checking whether
    bytes 2-5 also look like magic (0x11 0x11 0x11 0x11 -> legacy) versus
    looking like a plausible RSSI/FrameControl pair (-> rssi format). RSSI
    as a signed int8 for real Wi-Fi is virtually always in [-100, 0]; that's
    used as the tiebreaker.
    """
    if len(payload) < 6 or payload[0:2] != b"\x11\x11":
        raise DecodeError(f"bad magic bytes: {payload[:6].hex()}")

    if payload[2:6] == b"\x11\x11\x11\x11":
        return "legacy"

    rssi_signed = payload[2] - 256 if payload[2] >= 128 else payload[2]
    if -100 <= rssi_signed <= 0:
        return "rssi"

    # Fall back to legacy if the RSSI heuristic doesn't look plausible --
    # better to try the more common/documented layout than guess wrong twice.
    return "legacy"




def unpack_float_acphy(nbits: int, nman: int, nexp: int, nfft: int,
                        H: np.ndarray, autoscale: bool = False) -> np.ndarray:
    """
    Pure-Python/NumPy port of unpack_float_acphy() from unpack_float.c.

    Decodes the Broadcom "packed float" CSI format used by bcm4358 /
    bcm4366c0 into a complex128 array of length nfft.

    H: array-like of uint32, length nfft (one word per subcarrier).
    Returns: complex128 ndarray, length nfft.
    """
    H = np.asarray(H, dtype=np.uint32)
    if H.shape[0] != nfft:
        raise DecodeError(f"unpack_float_acphy: expected {nfft} words, got {H.shape[0]}")

    # The C reference does all of this in 32-bit int32_T arithmetic, including
    # (well-defined-in-practice) wraparound on left shifts. We match that
    # exactly with int64 intermediate math + explicit mask back to signed-32
    # range after every shift, rather than letting NumPy's native integer
    # promotion silently change edge-case behavior.
    H64 = H.astype(np.int64)

    SGN_MASK = 1 << 31
    iq_mask = (1 << (nman - 1)) - 1
    e_mask = (1 << nexp) - 1
    e_p = 1 << (nexp - 1)
    sgnr_mask = 1 << (nexp + 2 * nman - 1)
    sgni_mask = sgnr_mask >> nman
    e_zero = -nman

    def to_int32(v):
        """Emulate C's int32_T truncation/wraparound."""
        return ((v + 0x80000000) & 0xFFFFFFFF) - 0x80000000

    vi = (H64 >> (nexp + nman)) & iq_mask
    vq = (H64 >> nexp) & iq_mask
    e = H64 & e_mask
    e = np.where(e >= e_p, e - (e_p << 1), e)
    He = e.copy()

    maxbit = -e_p
    if autoscale:
        # Per-element leading-bit scan, matching the C reference exactly.
        for i in range(nfft):
            x = int(vi[i] | vq[i])
            ei = int(He[i])
            if x:
                m, b, s = 0xffff0000, 0xffff, 16
                while s > 0:
                    if x & m:
                        ei += s
                        x >>= s
                    s >>= 1
                    m = (m >> s) & b
                    b >>= s
                if ei > maxbit:
                    maxbit = ei
            He[i] = ei

    sgn_i_flag = (H64 & sgnr_mask) != 0
    sgn_q_flag = (H64 & sgni_mask) != 0
    vi = np.where(sgn_i_flag, vi | SGN_MASK, vi)
    vq = np.where(sgn_q_flag, vq | SGN_MASK, vq)

    shft = nbits - maxbit

    def rescale(v: np.ndarray, e_word: np.ndarray) -> np.ndarray:
        sgn = np.where((v & SGN_MASK) != 0, -1, 1)
        mag = np.where((v & SGN_MASK) != 0, v & ~SGN_MASK, v)
        ee = e_word + shft
        out = np.zeros_like(mag)
        m_zero = ee < e_zero
        m_rshift = (ee >= e_zero) & (ee < 0)
        m_lshift = ee >= 0
        out[m_rshift] = mag[m_rshift] >> (-ee[m_rshift])
        # left shift in C wraps at 32 bits (int32_T); replicate that exactly
        out[m_lshift] = to_int32(mag[m_lshift] << ee[m_lshift])
        out[m_zero] = 0
        return to_int32(sgn * out)

    real = rescale(vi, He)
    imag = rescale(vq, He)
    return real.astype(np.float64) + 1j * imag.astype(np.float64)


def decode_int16_csi(csi_bytes: bytes, nfft: int) -> np.ndarray:
    """Decode bcm4339 / bcm43455c0 style interleaved little-endian int16 I/Q."""
    expected = nfft * 4
    if len(csi_bytes) < expected:
        raise DecodeError(f"int16 CSI payload too short: {len(csi_bytes)} < {expected}")
    vals = np.frombuffer(csi_bytes[:expected], dtype="<i2")  # little-endian int16
    vals = vals.reshape(nfft, 2)
    return vals[:, 0].astype(np.float64) + 1j * vals[:, 1].astype(np.float64)


def decode_frame_payload(payload: bytes, index: int, ts: float, chip: str,
                          bw: int, nfft_override: Optional[int] = None,
                          header_format: str = "auto") -> CSIFrame:
    """
    Decode a single UDP payload (post IP/UDP stripping) into a CSIFrame.
    Raises DecodeError if the payload isn't a valid Nexmon CSI frame.

    header_format: "auto" (default, per-packet detection), "legacy", or "rssi".
    """
    if header_format == "auto":
        fmt_name = detect_header_format(payload)
    elif header_format in HEADER_FORMATS:
        fmt_name = header_format
    else:
        raise DecodeError(f"unknown header_format '{header_format}'")

    fmt = HEADER_FORMATS[fmt_name]
    header_len = fmt["header_len"]

    if len(payload) < header_len:
        raise DecodeError(f"payload too short for '{fmt_name}' header "
                           f"({len(payload)} < {header_len})")

    if payload[:len(fmt["magic"])] != fmt["magic"]:
        raise DecodeError(f"bad magic bytes for '{fmt_name}' format: "
                           f"{payload[:6].hex()}")

    rssi = None
    frame_control = None

    if fmt["has_rssi_fc"]:
        rssi_byte = payload[2]
        rssi = rssi_byte - 256 if rssi_byte >= 128 else rssi_byte
        frame_control = payload[3]
        off = 4
    else:
        off = 6

    src_mac = _mac_str(payload[off:off + 6])
    off += 6
    seq_ctrl = struct.unpack("<H", payload[off:off + 2])[0]
    off += 2
    # The 802.11 Sequence Control field packs a 4-bit Fragment Number into
    # the low bits and the 12-bit Sequence Number into the high bits
    # (SeqCtrl = SeqNum<<4 | FragNum). For loss/gap analysis we need the
    # pure 12-bit sequence number, not the raw combined field -- otherwise
    # every unfragmented frame (the overwhelming majority) looks like it
    # skipped 15 sequence numbers when it didn't.
    seq_num = seq_ctrl >> 4
    fragment_num = seq_ctrl & 0xF
    core_stream = struct.unpack("<H", payload[off:off + 2])[0]
    off += 2
    core = core_stream & 0x7
    spatial_stream = (core_stream >> 3) & 0x7
    chanspec = struct.unpack("<H", payload[off:off + 2])[0]
    off += 2
    chip_version = struct.unpack("<H", payload[off:off + 2])[0]
    off += 2

    assert off == header_len, f"internal error: header parse offset {off} != {header_len}"

    nfft = nfft_override or BW_TO_NFFT.get(bw)
    if nfft is None:
        raise DecodeError(f"unsupported bandwidth {bw} MHz (expected 20/40/80, or pass --nfft)")

    csi_bytes = payload[header_len:header_len + nfft * 4]

    if chip not in CHIP_TABLE:
        raise DecodeError(f"unknown chip '{chip}', expected one of {list(CHIP_TABLE)}")
    spec = CHIP_TABLE[chip]

    if spec["format"] == "int16":
        csi = decode_int16_csi(csi_bytes, nfft)
    else:
        if len(csi_bytes) < nfft * 4:
            raise DecodeError(f"CSI payload too short: {len(csi_bytes)} < {nfft * 4}")
        H = np.frombuffer(csi_bytes[:nfft * 4], dtype="<u4")
        csi = unpack_float_acphy(spec["nbits"], spec["nman"], spec["nexp"], nfft, H)

    return CSIFrame(
        index=index, ts=ts, src_mac=src_mac, seq_num=seq_num, core=core,
        spatial_stream=spatial_stream, chanspec=chanspec,
        chip_version=chip_version, csi=csi, rssi=rssi,
        frame_control=frame_control, fragment_num=fragment_num, raw_len=len(payload),
    )


def iter_pcap_csi_frames(pcap_path: str, chip: str = "43455c0", bw: int = 80,
                          udp_port: int = 5500,
                          nfft_override: Optional[int] = None,
                          src_ip_filter: Optional[str] = "10.10.10.10",
                          header_format: str = "auto",
                          ) -> Iterator[CSIFrame]:
    """
    Stream-decode CSI frames out of a pcap file.

    Skips (with a logged warning, not a crash) any packet that:
      - isn't UDP on udp_port
      - optionally doesn't match src_ip_filter (nexmon CSI frames come from
        10.10.10.10 -> 255.255.255.255 by convention; pass None to disable)
      - fails Nexmon magic-byte / length validation

    header_format: "auto" (default), "legacy", or "rssi" -- see module
    docstring for the two known header layouts.

    Uses scapy's PcapReader for a lazy, memory-friendly read so large
    captures don't need to be loaded in full.
    """
    try:
        from scapy.all import PcapReader, UDP, IP  # noqa: E402  (lazy import)
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "scapy is required to read pcap files. Install with: pip install scapy"
        ) from e

    n_seen = 0
    n_udp_match = 0
    n_decoded = 0
    n_skipped = 0

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            n_seen += 1
            if not pkt.haslayer(UDP):
                continue
            udp = pkt[UDP]
            if udp.dport != udp_port:
                continue
            if src_ip_filter and pkt.haslayer(IP) and pkt[IP].src != src_ip_filter:
                continue
            if udp.payload is None:
                continue
            payload = bytes(udp.payload)
            if len(payload) == 0:
                continue
            n_udp_match += 1

            ts = float(pkt.time) if hasattr(pkt, "time") else 0.0
            try:
                frame = decode_frame_payload(
                    payload, index=n_udp_match - 1, ts=ts, chip=chip, bw=bw,
                    nfft_override=nfft_override, header_format=header_format,
                )
            except DecodeError as e:
                n_skipped += 1
                logger.warning("Skipped packet %d: %s", n_seen, e)
                continue

            n_decoded += 1
            yield frame

    logger.info(
        "pcap scan complete: %d packets seen, %d matched UDP/%d, "
        "%d decoded OK, %d skipped (bad frame)",
        n_seen, n_udp_match, udp_port, n_decoded, n_skipped,
    )
