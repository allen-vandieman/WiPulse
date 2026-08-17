import os
from scapy.all import PcapReader, Ether, Dot11, Dot11Elt
import numpy as np
import matplotlib.pyplot as plt

# Configuration
CHIP = '4358'  # WiFi chip (possible values: 4339, 4358, 43455c0, 4366c0)
BW = 80  # Bandwidth in MHz
FILE = 'captures/csi_capture.pcap'  # Capture file path
NPKTS_MAX = 10000  # Maximum number of UDPs to process

# Constants
HOFFSET = 16
NFFT = BW * 3.2

def read_csi_packets(file):
    csi_data = []
    with PcapReader(file) as pcap_reader:
        for packet in pcap_reader:
            if Ether not in packet or Dot11 not in packet:
                continue
            
            dot11 = packet[Dot11]
            if 'CSITag' not in dot11:
                continue
            
            # Extract CSI data from the packet
            csi_payload = bytes(dot11['CSITag'])
            if len(csi_payload) < HOFFSET + NFFT * 4:
                continue
            
            payload = csi_payload[HOFFSET:HOFFSET + NFFT * 4]
            
            if (CHIP == '4339' or CHIP == '43455c0'):
                Hout = np.frombuffer(payload, dtype=np.int16)
            elif (CHIP == '4358'):
                Hout = unpack_float(np.frombuffer(payload, dtype=np.int32))
            elif (CHIP == '4366c0'):
                Hout = unpack_float(np.frombuffer(payload, dtype=np.int32), reverse_endian=True)
            else:
                print("Invalid CHIP")
                return csi_data
            
            Hout = Hout.reshape(2, -1).T
            cmplx = Hout[:, 0] + 1j * Hout[:, 1]
            csi_data.append(cmplx)
    
    return csi_data

def unpack_float(data, reverse_endian=False):
    if len(data) % 4 != 0:
        raise ValueError("Data length must be a multiple of 4")
    
    result = np.zeros(len(data) // 4, dtype=np.float32)
    
    for i in range(0, len(data), 4):
        if reverse_endian:
            byteorder = '<'
        else:
            byteorder = '>'
        
        float_value = struct.unpack(byteorder + 'f', data[i:i+4])[0]
        result[i // 4] = float_value
    
    return result

def plot_csi(csi_data, nfft):
    plt.figure(figsize=(12, 8))
    
    for i, csi in enumerate(csi_data[:NPKTS_MAX]):
        if len(csi) != nfft:
            continue
        
        plt.subplot(3, 4, i + 1)
        plt.plot(np.arange(nfft), np.abs(csi))
        plt.title(f'Packet {i+1} Magnitude')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csi_data = read_csi_packets(FILE)
    plot_csi(csi_data, NFFT)
