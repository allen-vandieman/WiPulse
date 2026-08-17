#!/usr/bin/env python3

import sys
import numpy as np
from scapy.all import rdpcap

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

HOFFSET = 16          # same as MATLAB
BW = 80
NFFT = int(BW * 3.2) # 256 for 80 MHz

# ------------------------------------------------------------------


def extract_csi_info(pcap_file):
    packets = rdpcap(pcap_file)
    
    print(f"Total packets in file: {len(packets)}")
    
    all_complex = []

    first_stats_done = False

    sequence = None
    core = None
    nss = None
    chanspec = None
    chipver = None

    valid_packets = 0
    
    # Try different offsets if the standard one doesn't work
    offsets_to_try = [15, 14, 13, 12, 16, 17, 18, 20]
    best_offset = None
    best_packets = 0

    for test_offset in offsets_to_try:
        test_valid = 0
        for pkt in packets[:10]:  # Check first 10 packets
            raw = bytes(pkt)
            if len(raw) < 60:
                continue
            usable_len = (len(raw) // 4) * 4
            words = np.frombuffer(raw[:usable_len], dtype="<u4")
            if len(words) >= (test_offset + NFFT):
                # Try to extract metadata
                payload8 = words.view(np.uint8)
                if len(payload8) >= 60:
                    seq_check = int.from_bytes(payload8[52:54], "little")
                    if seq_check != 0:  # Valid sequence number
                        test_valid += 1
        
        if test_valid > best_packets:
            best_packets = test_valid
            best_offset = test_offset
    
    if best_offset is not None and best_packets > 0:
        HOFFSET = best_offset
        print(f"Using adjusted HOFFSET = {HOFFSET} (found {best_packets} valid packets)")
    else:
        print("Warning: Could not find working offset, trying default")

    for idx, pkt in enumerate(packets):
        raw = bytes(pkt)
        
        # Debug: print first few packet sizes
        if idx < 5:
            print(f"Packet {idx} size: {len(raw)} bytes")

        # Need enough data for metadata extraction
        if len(raw) < 60:
            continue

        # Convert packet bytes into uint32 words exactly like MATLAB
        usable_len = (len(raw) // 4) * 4
        words = np.frombuffer(raw[:usable_len], dtype="<u4")
        
        if idx < 5:
            print(f"  Number of uint32 words: {len(words)}")
            print(f"  Need at least {HOFFSET + NFFT} words, have {len(words)}")

        if len(words) < (HOFFSET + NFFT):
            # Try a smaller NFFT for 40MHz or 20MHz
            for bw_test in [40, 20]:
                nfft_test = int(bw_test * 3.2)
                if len(words) >= (HOFFSET + nfft_test):
                    print(f"  Found match for {bw_test}MHz bandwidth (NFFT={nfft_test})")
                    # Could handle this case here
            continue

        # ------------------------------------------------------------------
        # CSI extraction
        # ------------------------------------------------------------------
        H = words[HOFFSET:HOFFSET + NFFT]

        hout = H.view(np.int16)
        
        if idx < 5:
            print(f"  hout shape: {hout.shape}")
            print(f"  hout min: {hout.min()}, max: {hout.max()}")

        if not first_stats_done:
            print(f"Min int16 value = {hout.min()}")
            print(f"Max int16 value = {hout.max()}")
            print(f"Number of zeros = {np.count_nonzero(hout == 0)}")
            print(f"Total samples   = {len(hout)}")
            first_stats_done = True

        # ------------------------------------------------------------------
        # Metadata extraction
        # ------------------------------------------------------------------
        if sequence is None:
            payload8 = words.view(np.uint8)

            if len(payload8) >= 60:
                sequence = int.from_bytes(payload8[52:54], "little")
                core_ss = int.from_bytes(payload8[54:56], "little")
                chanspec = int.from_bytes(payload8[56:58], "little")
                chipver = int.from_bytes(payload8[58:60], "little")

                core = core_ss & 0x7
                nss = (core_ss >> 3) & 0x7
                
                print(f"\nMetadata extracted from first valid packet:")
                print(f"  Sequence = {sequence}")
                print(f"  Core = {core}")
                print(f"  NSS = {nss}")
                print(f"  Chanspec = 0x{chanspec:04X}")
                print(f"  ChipVer = 0x{chipver:04X}")

        # ------------------------------------------------------------------
        # Convert interleaved int16 -> complex CSI
        # ------------------------------------------------------------------
        iq = hout.reshape(-1, 2)

        complex_csi = (
            iq[:, 0].astype(np.float64)
            + 1j * iq[:, 1].astype(np.float64)
        )

        all_complex.append(complex_csi)
        valid_packets += 1
        
        if idx < 5:
            print(f"  Packet {idx} processed successfully as CSI packet")

    # ----------------------------------------------------------------------
    # Try alternative approach: search for CSI data in raw bytes
    # ----------------------------------------------------------------------
    if valid_packets == 0:
        print("\nTrying alternative parsing method...")
        
        # Look for patterns in the raw data
        sample_pkt = bytes(packets[0])
        
        # Try to find where the CSI data might start
        for offset in range(10, 30):
            # Check if the data at this offset looks like valid int16 pairs
            data_slice = sample_pkt[offset:offset+512]
            if len(data_slice) >= 512:
                int16_data = np.frombuffer(data_slice, dtype=np.int16)
                if len(int16_data) >= 100:
                    # Check if it looks like complex data (I/Q pairs)
                    if len(int16_data) % 2 == 0:
                        iq_data = int16_data.reshape(-1, 2)
                        # Check if values are reasonable
                        if np.abs(iq_data).mean() > 10:
                            print(f"  Found potential CSI data at offset {offset}")
                            print(f"    Mean magnitude: {np.abs(iq_data).mean():.2f}")
                            print(f"    Number of subcarriers: {len(iq_data)}")
                            if len(iq_data) == 256:
                                print(f"    This matches 80MHz bandwidth!")
                            elif len(iq_data) == 128:
                                print(f"    This matches 40MHz bandwidth!")
                            elif len(iq_data) == 64:
                                print(f"    This matches 20MHz bandwidth!")

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
    else:
        print("No metadata could be extracted from any packet")

    # ----------------------------------------------------------------------
    # Strongest 15 subcarriers
    # ----------------------------------------------------------------------

    if not all_complex:
        print("\nNo valid CSI packets found.")
        print(f"Total packets processed: {len(packets)}")
        print(f"Valid CSI packets: {valid_packets}")
        
        # Show first packet hex dump for debugging
        print("\nFirst 100 bytes of first packet (hex):")
        first_pkt = bytes(packets[0])
        for i in range(0, min(100, len(first_pkt)), 16):
            hex_str = ' '.join(f'{b:02x}' for b in first_pkt[i:i+16])
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in first_pkt[i:i+16])
            print(f"{i:04x}: {hex_str:<48} {ascii_str}")
        return

    csi = np.array(all_complex)

    idx_pkt = min(99, len(csi) - 1)

    magnitudes = np.abs(csi[idx_pkt])

    strongest_idx = np.argsort(magnitudes)[::-1][:15]

    print()
    print(f"15 Strongest Subcarriers (packet {idx_pkt + 1})")
    print("------------------------------------------------")

    for i in strongest_idx:
        print(f"{i:3d}  {magnitudes[i]:.1f}")

    print()
    print(f"Valid CSI packets = {valid_packets}")


# ------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} capture.pcap")
        sys.exit(1)

    extract_csi_info(sys.argv[1])