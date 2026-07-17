from dataclasses import dataclass

from scapy.all import rdpcap

from scapy.layers.inet import UDP

import numpy as np

import struct



@dataclass

class CSIFrame:

    sequence: int

    core: int

    spatial_stream: int

    chanspec: int

    chip_version: int

    mac: str

    csi: np.ndarray



class NexmonDecoder:



    def __init__(self, bandwidth=80):

        self.bandwidth = bandwidth

        self.nfft = int(bandwidth * 3.2)



    def decode_file(self, filename):

        packets = rdpcap(filename)

        frames = []

        for pkt in packets:

            if UDP not in pkt:

                continue

            if pkt[UDP].dport != 5500:

                continue

            frame = self.decode_packet(bytes(pkt[UDP].payload))

            if frame is not None:

                frames.append(frame)

        return frames



    def decode_packet(self, payload):

        if len(payload) < 18:

            return None



        magic = payload[:2]



        if magic != b"\x11\x11":

            return None



        mac = ":".join(f"{b:02x}" for b in payload[4:10])



        sequence = struct.unpack("<H", payload[10:12])[0]



        css = struct.unpack("<H", payload[12:14])[0]



        core = css & 0x7

        spatial = (css >> 3) & 0x7



        chanspec = struct.unpack("<H", payload[14:16])[0]



        chip = struct.unpack("<H", payload[16:18])[0]



        raw = payload[18:]



        iq = np.frombuffer(raw, dtype="<i2")



        iq = iq.reshape((-1,2))



        csi = iq[:,0].astype(np.float32) + 1j*iq[:,1].astype(np.float32)



        return CSIFrame(

            sequence,

            core,

            spatial,

            chanspec,

            chip,

            mac,

            csi

        )