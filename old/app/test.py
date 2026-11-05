from decoder import iter_pcap_csi_frames
from analysis import build_dataset, fftshift_subcarriers, amplitude, select_subcarriers, fft_motion_detection
from filters import butterworth_bandpass, remove_dc_drift

frames = list(iter_pcap_csi_frames("test/baseline_lab6.pcap", chip="43455c0", bw=80))
ds = build_dataset(frames)
csi = select_subcarriers(fftshift_subcarriers(ds.csi), exclude_dc_guard=True, nfft=ds.nfft)
amp = amplitude(csi)[:, 57]   # sc 57, matching your quality-ranked index

bp = butterworth_bandpass(amp, 0.7, 2.5, ds.fs_est)
bp_detrended = remove_dc_drift(bp)

print("bandpass only:     ", fft_motion_detection(bp[:, None], fs=ds.fs_est, subcarrier_idx=0)["dominant_freq_hz"])
print("bandpass+detrend:  ", fft_motion_detection(bp_detrended[:, None], fs=ds.fs_est, subcarrier_idx=0)["dominant_freq_hz"])
