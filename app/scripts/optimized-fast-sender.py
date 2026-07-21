#!/usr/bin/python3
from scapy.all import *
import time
import struct

iface = "wlan0mon"
src = "02:11:22:33:44:55"
dst = "ff:ff:ff:ff:ff:ff"

TARGET_HZ = 100
PERIOD = 1.0 / TARGET_HZ

# Build the static frame ONCE - only the payload changes per iteration
base_frame = (
    RadioTap()/
    Dot11(
        type=2,
        subtype=0,
        addr1=dst,
        addr2=src,
        addr3=src
    )/
    LLC()/
    SNAP()
)

# Open the L2 socket ONCE and reuse it for all sends
sock = conf.L2socket(iface=iface)

packet_number = 0
next_send_time = time.perf_counter()

try:
    while True:
        payload = struct.pack("<IQ", packet_number, int(time.time() * 1000))
        frame = base_frame / Raw(load=payload)

        sock.send(frame)

        # Only print occasionally - printing every packet at 100Hz adds overhead
        if packet_number % 100 == 0:
            print(f"Sent packet {packet_number}")

        packet_number += 1

        # Precise pacing: schedule next send based on absolute time,
        # not "sleep 10ms after work is done" (which drifts)
        next_send_time += PERIOD
        sleep_time = next_send_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # We're behind schedule - reset reference to avoid runaway catch-up bursts
            next_send_time = time.perf_counter()

except KeyboardInterrupt:
    print(f"\nStopped. Sent {packet_number} packets.")
finally:
    sock.close()