# Quick check: Plot a few subcarriers run in PS
python -c "
import numpy as np
import matplotlib.pyplot as plt
from read_csv_csi import CSIDataLoader

loader = CSIDataLoader('captures/Alicia/csi_capture_20260818_193158.csv')
data, meta = loader.load()

# Plot subcarriers 50, 60, 228, 232
fig, ax = plt.subplots(4, 1, figsize=(12, 8))
for idx, sc in enumerate([50, 60, 228, 232]):
    ax[idx].plot(np.abs(data[:, sc]))
    ax[idx].set_title(f'Subcarrier {sc}')
plt.tight_layout()
plt.savefig('check_subcarriers.png')
print('Saved subcarrier plot to check_subcarriers.png')
"