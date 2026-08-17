#!/usr/bin/env python3

    # ----------------------------------------------------------------------
    # Usage: python read_pcap.py capture.pcap
    # ----------------------------------------------------------------------

import sys
import numpy as np
from scapy.all import rdpcap

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

HOFFSET = 15          # MATLAB 16 and py 15
BW = 80
NFFT = int(BW * 3.2) # 256 for 80 MHz

# ------------------------------------------------------------------


def extract_csi_info(pcap_file):
    packets = rdpcap(pcap_file)

    all_complex = []

    first_stats_done = False

    sequence = None
    core = None
    nss = None
    chanspec = None
    chipver = None

    valid_packets = 0

    for pkt in packets:

        raw = bytes(pkt)

        # Need enough data for metadata extraction
        if len(raw) < 60:
            continue

        # Convert packet bytes into uint32 words exactly like MATLAB
        usable_len = (len(raw) // 4) * 4
        words = np.frombuffer(raw[:usable_len], dtype="<u4")

        if len(words) < (HOFFSET + NFFT):
            continue

        # ------------------------------------------------------------------
        # CSI extraction (43455c0 only)
        # MATLAB:
        # H = payload(HOFFSET:HOFFSET+NFFT-1);
        # Hout = typecast(H,'int16');
        # ------------------------------------------------------------------
        H = words[HOFFSET:HOFFSET + NFFT]

        hout = H.view(np.int16)

        if not first_stats_done:

            print(f"Min int16 value = {hout.min()}")
            print(f"Max int16 value = {hout.max()}")
            print(f"Number of zeros = {np.count_nonzero(hout == 0)}")
            print(f"Total samples   = {len(hout)}")

            first_stats_done = True

        # ------------------------------------------------------------------
        # Metadata
        # MATLAB:
        # payload8 = typecast(payload,'uint8');
        # seq      = typecast(payload8(53:54),'uint16');
        # core_ss  = typecast(payload8(55:56),'uint16');
        # chanspec = typecast(payload8(57:58),'uint16');
        # chipver  = typecast(payload8(59:60),'uint16');
        # ------------------------------------------------------------------

        if sequence is None:

            payload8 = words.view(np.uint8)

            if len(payload8) >= 60:

                sequence = int.from_bytes(payload8[52:54], "little")

                core_ss = int.from_bytes(
                    payload8[54:56],
                    "little"
                )

                chanspec = int.from_bytes(
                    payload8[56:58],
                    "little"
                )

                chipver = int.from_bytes(
                    payload8[58:60],
                    "little"
                )

                core = core_ss & 0x7
                nss = (core_ss >> 3) & 0x7

        # ------------------------------------------------------------------
        # Convert interleaved int16 -> complex CSI
        # MATLAB:
        # Hout = reshape(Hout,2,[]).';
        # cmplx = Hout(:,1) + 1j*Hout(:,2)
        # ------------------------------------------------------------------

        iq = hout.reshape(-1, 2)

        complex_csi = (
            iq[:, 0].astype(np.float64)
            + 1j * iq[:, 1].astype(np.float64)
        )

        all_complex.append(complex_csi)

        valid_packets += 1

    # ----------------------------------------------------------------------
    # Metadata output
    # ----------------------------------------------------------------------

    print()
    print("Connection Information")
    print("----------------------")

    if sequence is not None:
        print(f"Sequence = {sequence}")
        print(f"Core     = {core}")
        print(f"NSS      = {nss}")
        print(f"Chanspec = 0x{chanspec:04X}")
        print(f"ChipVer  = 0x{chipver:04X}")

    # ----------------------------------------------------------------------
    # Strongest 15 subcarriers
    # MATLAB used packet 100:
    # pkt = abs(csi_buff(100,:));
    #
    # If <100 packets, use last packet.
    # ----------------------------------------------------------------------

    if not all_complex:
        print("\nNo valid CSI packets found.")
        return

    csi = np.array(all_complex)

    idx_pkt = min(99, len(csi) - 1)

    magnitudes = np.abs(csi[idx_pkt])

    strongest_idx = np.argsort(magnitudes)[::-1][:15]

    print()
    print(f"15 Strongest Subcarriers (packet {idx_pkt + 1})")
    print("------------------------------------------------")

    for i in strongest_idx:
        print(f"{i:3d} {magnitudes[i]:.1f}")


    print()
    print(f"Valid CSI packets = {valid_packets}")


# ------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} capture.pcap")
        sys.exit(1)

    extract_csi_info(sys.argv[1])
