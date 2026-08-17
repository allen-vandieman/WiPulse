import scapy.all as scapy
import numpy as np
from scapy.layers import IP, TCP, Ethernet
import time

# ... (other imports and functions from your project) ... 

def read_pcap(file_path):
    packets = scapy.rdpcap(file_path)
    return packets
    
def process_packets(packets, chip_type='43455c0', bandwidth=80, max_packets=10000): 
    hoffset = 16 # offset for header in bytes (adjust if needed)
    nfft = bandwidth * 3.2
    csi_buff = []  
    seqs = []  
    cores = []  
    nsses = []  
    chanspecs = []  
    chipvers = [] 
    
    # ... (rest of your process_packets function) ...   

def find_strongest(csi_data, packet_index):
    pkt = np.abs(csi_data[packet_index])
    vals, idx = np.sort(-pkt, axis=1).flatten(), np.argsort(-pkt, axis=1).flatten()
    return vals[:15], idx[:15]  # Find the strongest values in the packet
    
def find_strongest_packet(packets, packet_index):
    if packets is None:
        print('Error: No packets available. Please provide packets.') 
        return None

    pkt = np.abs(packets[packet_index])
    vals, idx = np.sort(-pkt, axis=1).flatten(), np.argsort(-pkt, axis=1).flatten()
    return vals[:15], idx[:15]   # Find the strongest values in the packet
    
def process_packets(packets, chip_type='43455c0', bandwidth=80, max_packets=10000):
    hoffset = 16 # offset for header in bytes (adjust if needed)
    nfft = bandwidth * 3.2
    csi_buff = []  
    seqs = []  
    cores = []  
    nsses = []  
    chanspecs = []  
    chipvers = [] 
    
    # ... (rest of your process_packets function) ...   
    # Handling cut-off packets
    try:
        for i in range(100):  # Example for 100 packets, increase to analyze more
            if len(packets) <= max_packets:
                break
            # Process packet based on your logic here
            # ... (extract data, process as needed) ... 
            csi_buff.append(...)
            seqs.append(...) 
            cores.append(...) 
            nsses.append(...) 
            chanspecs.append(...)  
            chipvers.append(...)  
    except Exception as e:
        print(f'Error: {e}')
        return None
    finally:
        # ... (clean up and close resources) ...   
        if packets is not None:
            packets.close() 
    return np.array(csi_buff), np.array(seqs), np.array(cores), np.array(nsses), np.array(chanspecs), np.array(chipvers)

# Example usage (you'll need to update the file path and packet index)
file_path = 'captures/csi_capture.pcap'
packets = scapy.rdpcap(file_path) 
try:
    csi_data, seqs, cores, nses, chanspecs, chipvers = process_packets(packets)
    # ... (process the data and extract information as needed)  ... 
except Exception as e:
    print(f'Error: {e}')
