import numpy as np
import pandas as pd


def save_numpy(filename, csi):

    np.save(filename, csi)


def save_csv(filename, csi):

    amp = np.abs(csi)

    df = pd.DataFrame(amp)

    df.to_csv(filename, index=False)