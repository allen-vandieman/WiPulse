# phase_processing.py
# Phase-based heartbeat extraction for CSI data

import numpy as np
from scipy import signal
from scipy.signal import detrend

class PhaseProcessor:
    """Extract heartbeat from CSI phase data"""
    
    def __init__(self, fs=100.0):
        self.fs = fs
        self.phase_data = None
        self.phase_unwrapped = None
        self.phase_detrended = None
        self.phase_filtered = None
        self.phase_quality = None
        
    def extract_phase(self, csi_data):
        """Extract phase from complex CSI"""
        self.phase_data = np.angle(csi_data)
        return self.phase_data
    
    def unwrap_phase(self, phase_data=None):
        """Unwrap phase across time (axis=0)"""
        if phase_data is None:
            phase_data = self.phase_data
        
        # Unwrap each subcarrier independently
        self.phase_unwrapped = np.zeros_like(phase_data)
        for sc in range(phase_data.shape[1]):
            self.phase_unwrapped[:, sc] = np.unwrap(phase_data[:, sc])
        
        return self.phase_unwrapped
    
    def remove_common_phase(self, phase_unwrapped=None):
        """Remove common phase trend across subcarriers"""
        if phase_unwrapped is None:
            phase_unwrapped = self.phase_unwrapped
        
        # Median phase across subcarriers for each packet
        common_phase = np.median(phase_unwrapped, axis=1)
        
        # Subtract common phase
        phase_removed = phase_unwrapped - common_phase[:, np.newaxis]
        
        return phase_removed
    
    def detrend_phase(self, phase_unwrapped=None):
        """Remove linear trend from phase"""
        if phase_unwrapped is None:
            phase_unwrapped = self.phase_unwrapped
        
        self.phase_detrended = detrend(phase_unwrapped, axis=0)
        return self.phase_detrended
    
    def bandpass_phase(self, phase_data=None, low=0.8, high=2.5, order=4):
        """Apply bandpass filter to phase data"""
        if phase_data is None:
            phase_data = self.phase_detrended
        
        nyquist = self.fs / 2
        sos = signal.butter(order, [low/nyquist, high/nyquist], 
                           btype='band', output='sos')
        
        self.phase_filtered = np.zeros_like(phase_data)
        for sc in range(phase_data.shape[1]):
            self.phase_filtered[:, sc] = signal.sosfiltfilt(
                sos, phase_data[:, sc] - np.mean(phase_data[:, sc])
            )
        
        return self.phase_filtered
    
    def compute_phase_differences(self, phase_unwrapped=None, max_diff=10):
        """Compute phase differences between subcarriers"""
        if phase_unwrapped is None:
            phase_unwrapped = self.phase_unwrapped
        
        n_sc = phase_unwrapped.shape[1]
        diffs = []
        diff_indices = []
        
        # Test differences up to max_diff apart
        for d in range(1, min(max_diff + 1, n_sc)):
            for i in range(n_sc - d):
                diff = phase_unwrapped[:, i] - phase_unwrapped[:, i + d]
                diffs.append(diff)
                diff_indices.append((i, i + d))
        
        return diffs, diff_indices
    
    def evaluate_phase_quality(self, phase_filtered, hr_range=(0.8, 2.5)):
        """Evaluate which subcarriers have strongest heartbeat signal"""
        n_sc = phase_filtered.shape[1]
        quality = []
        
        for sc in range(n_sc):
            signal_data = phase_filtered[:, sc]
            
            # FFT
            fft_vals = np.abs(np.fft.fft(signal_data - np.mean(signal_data)))
            freqs = np.fft.fftfreq(len(signal_data), 1/self.fs)
            
            # Power in heartbeat band
            mask = (freqs >= hr_range[0]) & (freqs <= hr_range[1])
            hr_power = np.sum(fft_vals[mask]**2) if np.any(mask) else 0
            
            # Total power
            total_power = np.sum(fft_vals**2)
            
            # SNR
            hr_power_ratio = hr_power / (total_power + 1e-10)
            
            quality.append({
                'sc': sc,
                'hr_power': hr_power,
                'hr_power_ratio': hr_power_ratio,
                'peak_freq': freqs[mask][np.argmax(fft_vals[mask])] if np.any(mask) else 0
            })
        
        self.phase_quality = pd.DataFrame(quality)
        self.phase_quality = self.phase_quality.sort_values('hr_power_ratio', ascending=False)
        
        return self.phase_quality
    
    def extract_heartbeat_signal(self, csi_data, method='phase', 
                                 low=0.8, high=2.5, order=4,
                                 top_n=5, use_phase_diff=False):
        """
        Main method: extract heartbeat signal from CSI
        
        Args:
            csi_data: complex CSI (packets x subcarriers)
            method: 'phase', 'amplitude', or 'phase_diff'
            low, high: bandpass filter range
            order: filter order
            top_n: number of best subcarriers to average
            use_phase_diff: if True, use phase differences instead of raw phase
        
        Returns:
            heartbeat_signal: extracted heartbeat waveform
            metrics: quality metrics
        """
        
        if method == 'amplitude':
            # Use amplitude (existing method)
            signal_data = np.abs(csi_data)
            signal_processed = signal_data
            
        elif method == 'phase':
            # Full phase processing pipeline
            phase = self.extract_phase(csi_data)
            unwrapped = self.unwrap_phase(phase)
            # Use phase difference between subcarriers to remove common noise
            if use_phase_diff:
                # Test phase differences
                diffs, indices = self.compute_phase_differences(unwrapped, max_diff=10)
                # Quality evaluate each difference
                best_diff = None
                best_score = 0
                for diff, (i, j) in zip(diffs, indices):
                    detrended = detrend(diff)
                    filtered = self.bandpass_phase(detrended[:, np.newaxis], low, high, order)
                    score = np.std(filtered) / (np.mean(np.abs(detrended)) + 1e-10)
                    if score > best_score:
                        best_score = score
                        best_diff = filtered[:, 0]
                signal_processed = best_diff if best_diff is not None else np.zeros(len(csi_data))
            else:
                # Standard phase processing
                # Option A: Remove common phase
                phase_removed = self.remove_common_phase(unwrapped)
                # Detrend
                detrended = self.detrend_phase(phase_removed)
                # Bandpass
                filtered = self.bandpass_phase(detrended, low, high, order)
                # Quality evaluate
                quality = self.evaluate_phase_quality(filtered)
                top_scs = quality.head(top_n)['sc'].values.astype(int)
                # Average top subcarriers
                signal_processed = np.mean(filtered[:, top_scs], axis=1)
                
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize
        heartbeat = (signal_processed - np.mean(signal_processed)) / np.std(signal_processed)
        
        return heartbeat, self.phase_quality if hasattr(self, 'phase_quality') else None