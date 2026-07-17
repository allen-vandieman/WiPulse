import argparse

import numpy as np

from decoder import NexmonDecoder

from exporter import save_numpy, save_csv

from utils import ensure_directory



parser = argparse.ArgumentParser()



parser.add_argument("pcap")



args = parser.parse_args()



ensure_directory("output")



decoder = NexmonDecoder()



frames = decoder.decode_file(args.pcap)



print(f"Frames : {len(frames)}")



matrix = np.vstack([f.csi for f in frames])



print(matrix.shape)



save_numpy(

    "output/csi.npy",

    matrix

)



save_csv(

    "output/csi_amplitude.csv",

    matrix

)



print("Done.")