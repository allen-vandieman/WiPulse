import numpy as np
from scapy.all import rdpcap

def extract_csi_data(filename):
    csi_data = []
    
    with rdpcap(filename) as cap:
        for packet in cap:
            if 'CSI' in str(packet):  # Check for the presence of "CSI" in the packet string
                # Extract core, NSS, chanspec, chipver
                core_ss = ((packet[125] >> 3) & 0x7) + (packet[126] & 0x7)
                nss = (packet[126] >> 3) & 0x7
                chanspec = packet[128:130]
                chipver = packet[130:132]
                
                # Extract CSI data
                csi_bytes = packet.load[24:]  # Assuming header is 24 bytes long
                csi_data.extend([int.from_bytes(csi_bytes[i:i+2], byteorder='big') for i in range(0, len(csi_bytes), 2)])
    
    return csi_data

def main():
    filename = 'captures/csi_capture.pcap'
    csi_data = extract_csi_data(filename)
    
    # Min int16 value
    min_value = np.min(csi_data)
    print(f'Min int16 value: {min_value}')
    
    # Max int16 value
    max_value = np.max(csi_data)
    print(f'Max int16 value: {max_value}')
    
    # Number of zeros
    num_zeros = csi_data.count(0)
    print(f'Number of zeros: {num_zeros}')
    
    # Total samples
    total_samples = len(csi_data)
    print(f'Total samples: {total_samples}')
    
    # Find 15 strongest values
    strongest_values = np.sort(csi_data)[-15:]
    print('15 strongest values:')
    for val in strongest_values:
        print(val)

if __name__ == '__main__':
    main()
