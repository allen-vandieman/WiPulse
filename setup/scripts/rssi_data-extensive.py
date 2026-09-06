#!/usr/bin/python3
from scapy.all import *
import struct
import time

MY_MAC = "02:11:22:33:44:55"
iface = "wlan0mon"

stats = {
    "received": 0,
    "expected_next": None,
    "lost": 0,
    "first_seen": None,
    "last_seen": None,
}

def get_radiotap_info(pkt):
    """Extract whatever RadioTap fields the adapter/driver provides."""
    info = {}
    if not pkt.haslayer(RadioTap):
        return info

    rt = pkt[RadioTap]
    info["len"] = rt.len

    # Signal / noise
    if hasattr(rt, "dBm_AntSignal"):
        info["rssi"] = rt.dBm_AntSignal
    if hasattr(rt, "dBm_AntNoise"):
        info["noise"] = rt.dBm_AntNoise
    if hasattr(rt, "Antenna"):
        info["antenna"] = rt.Antenna

    # Legacy rate/channel fields
    if hasattr(rt, "Rate"):
        info["rate_mbps"] = rt.Rate
    if hasattr(rt, "ChannelFrequency"):
        info["freq_mhz"] = rt.ChannelFrequency
    if hasattr(rt, "ChannelFlags"):
        info["chan_flags"] = rt.ChannelFlags

    # HT (802.11n) fields - present if the frame has an MCS radiotap field
    if hasattr(rt, "MCS_index"):
        info["mcs_index"] = rt.MCS_index
    if hasattr(rt, "MCS_bandwidth"):
        info["ht_bandwidth"] = rt.MCS_bandwidth
    if hasattr(rt, "MCS_GI"):
        info["guard_interval"] = "short" if rt.MCS_GI else "long"

    # VHT (802.11ac) fields, if adapter/driver exposes them
    if hasattr(rt, "VHT_bandwidth"):
        info["vht_bandwidth"] = rt.VHT_bandwidth
    if hasattr(rt, "VHT_mcs"):
        info["vht_mcs"] = rt.VHT_mcs
    if hasattr(rt, "VHT_nss"):
        info["vht_nss"] = rt.VHT_nss

    return info

def packet(pkt):
    if not pkt.haslayer(Dot11):
        return
    if pkt.addr2 != MY_MAC:
        return
    if not pkt.haslayer(Raw):
        return

    payload = bytes(pkt[Raw].load)
    now = time.time()
    rt_info = get_radiotap_info(pkt)

    print("=" * 60)
    print(pkt.summary())

    # Print whatever RadioTap fields were found, in a stable order
    field_order = [
        ("len", "Radiotap length"),
        ("rssi", "RSSI (dBm)"),
        ("noise", "Noise floor (dBm)"),
        ("antenna", "Antenna index"),
        ("rate_mbps", "Data rate (Mbps)"),
        ("freq_mhz", "Channel frequency (MHz)"),
        ("chan_flags", "Channel flags"),
        ("mcs_index", "MCS index"),
        ("ht_bandwidth", "HT bandwidth"),
        ("guard_interval", "Guard interval"),
        ("vht_bandwidth", "VHT bandwidth"),
        ("vht_mcs", "VHT MCS"),
        ("vht_nss", "VHT spatial streams (NSS)"),
    ]
    for key, label in field_order:
        if key in rt_info:
            print(f"{label}: {rt_info[key]}")

    print(f"Received at: {now:.3f} (local)")

    if len(payload) < 12:
        print(f"Payload too short to unpack ({len(payload)} bytes): {payload}")
        return

    packet_number, sent_ts_ms = struct.unpack("<IQ", payload[:12])
    latency_ms = (now * 1000) - sent_ts_ms

    print(f"Packet #: {packet_number}")
    print(f"Sent timestamp (ms): {sent_ts_ms}")
    print(f"Latency: {latency_ms:.1f} ms")
    print(f"Raw payload: {payload}")

    # --- Update loss stats ---
    stats["received"] += 1
    if stats["first_seen"] is None:
        stats["first_seen"] = now
    stats["last_seen"] = now

    if stats["expected_next"] is None:
        stats["expected_next"] = packet_number
    else:
        if packet_number > stats["expected_next"]:
            gap = packet_number - stats["expected_next"]
            stats["lost"] += gap
            print(f"!! Detected {gap} lost packet(s) (gap in sequence)")
        elif packet_number < stats["expected_next"]:
            print("!! Out-of-order or duplicate packet")

    stats["expected_next"] = packet_number + 1

    total_expected = stats["received"] + stats["lost"]
    loss_pct = (stats["lost"] / total_expected * 100) if total_expected else 0
    duration = stats["last_seen"] - stats["first_seen"] if stats["first_seen"] else 0

    print(f"-- Stats: received={stats['received']} lost={stats['lost']} "
          f"loss_rate={loss_pct:.2f}% duration={duration:.1f}s")

try:
    sniff(iface=iface, store=False, prn=packet)
except KeyboardInterrupt:
    print("\nStopped.")
    total_expected = stats["received"] + stats["lost"]
    loss_pct = (stats["lost"] / total_expected * 100) if total_expected else 0
    print(f"Final: received={stats['received']} lost={stats['lost']} loss_rate={loss_pct:.2f}%")

