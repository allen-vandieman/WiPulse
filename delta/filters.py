#!/usr/bin/env python3

# ----------------------------------------------------------------------
# Filter functions for CSI data processing
# ----------------------------------------------------------------------

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter


class FilterConfig:
    """Configuration for CSI filtering"""
    
    def __init__(self):
        # DC Notch Filter Settings
        self.notch_freq = 0.0  # DC component
        self.notch_Q = 30.0    # Quality factor (higher = narrower notch)
        self.sampling_rate = 100.0  # Hz
        
        # Hampel Filter Settings
        self.hampel_window_size = 11  # Window size for median filter (must be odd)
        self.hampel_threshold = 3.0   # Number of standard deviations for outlier detection
        
        # Zero Detection
        self.zero_threshold = 1e-6   # Threshold to consider a value as zero
        
        # Interpolation Settings
        self.interpolate_problematic = True


class CSIFilter:
    """Apply various filters to CSI data"""
    
    def __init__(self, config=None):
        self.config = config if config else FilterConfig()
        self.problematic_mask = None
        self.dc_mask = None
        self.zero_mask = None
        
    def detect_dc_subcarriers(self, csi_data, threshold=0.1):
        """
        Detect DC subcarriers (those with very low amplitude variation)
        Returns boolean mask of DC subcarriers
        """
        magnitudes = np.abs(csi_data)
        
        # Calculate mean magnitude per subcarrier
        mean_mag = np.mean(magnitudes, axis=0)
        
        # Calculate variance per subcarrier
        var_mag = np.var(magnitudes, axis=0)
        
        # Normalize variance by mean (coefficient of variation)
        cv = np.zeros_like(var_mag)
        mask = mean_mag > 0
        cv[mask] = var_mag[mask] / mean_mag[mask]
        
        # DC subcarriers have very low variation and low mean
        dc_mask = (mean_mag < np.percentile(mean_mag, 5)) | (cv < threshold)
        
        self.dc_mask = dc_mask
        return dc_mask
    
    def detect_permanent_zeros(self, csi_data, zero_threshold=None):
        """
        Detect subcarriers that are permanently zero
        Returns boolean mask of permanent zero subcarriers
        """
        if zero_threshold is None:
            zero_threshold = self.config.zero_threshold
            
        magnitudes = np.abs(csi_data)
        
        # Check if all values for a subcarrier are below threshold
        permanent_zero_mask = np.all(magnitudes < zero_threshold, axis=0)
        
        self.zero_mask = permanent_zero_mask
        return permanent_zero_mask
    
    def detect_problematic_subcarriers(self, csi_data):
        """Detect all problematic subcarriers (DC + permanent zeros)"""
        dc_mask = self.detect_dc_subcarriers(csi_data)
        zero_mask = self.detect_permanent_zeros(csi_data)
        problematic_mask = dc_mask | zero_mask
        
        self.problematic_mask = problematic_mask
        return problematic_mask
    
    def apply_notch_filter(self, csi_data, freq=None, Q=None, fs=None):
        """
        Apply notch filter to remove specific frequency components
        """
        if freq is None:
            freq = self.config.notch_freq
        if Q is None:
            Q = self.config.notch_Q
        if fs is None:
            fs = self.config.sampling_rate
        
        # Calculate normalized frequency
        nyquist = fs / 2
        normalized_freq = freq / nyquist if nyquist > 0 else 0
        
        if normalized_freq <= 0 or normalized_freq >= 1:
            # If DC, use a high-pass filter instead
            b, a = signal.butter(1, 0.01, btype='high', fs=fs)
        else:
            # Create notch filter
            b, a = signal.iirnotch(normalized_freq, Q)
        
        # Apply filter to each subcarrier
        filtered_data = np.zeros_like(csi_data)
        for sc in range(csi_data.shape[1]):
            # Filter real and imaginary parts separately
            filtered_data[:, sc] = signal.filtfilt(
                b, a, csi_data[:, sc], axis=0
            )
        
        return filtered_data
    
    def apply_hampel_filter(self, csi_data, window_size=None, threshold=None):
        """
        Apply Hampel filter to remove outliers
        """
        if window_size is None:
            window_size = self.config.hampel_window_size
        if threshold is None:
            threshold = self.config.hampel_threshold
        
        if window_size % 2 == 0:
            window_size += 1  # Make sure it's odd
        
        half_window = window_size // 2
        filtered_data = csi_data.copy()
        
        for sc in range(csi_data.shape[1]):
            # Process real and imaginary parts separately
            for component_idx in [0, 1]:  # 0 = real, 1 = imag
                component = csi_data[:, sc].real if component_idx == 0 else csi_data[:, sc].imag
                
                for i in range(len(component)):
                    # Get window indices
                    start = max(0, i - half_window)
                    end = min(len(component), i + half_window + 1)
                    window = component[start:end]
                    
                    # Compute median and MAD
                    median_val = np.median(window)
                    mad = np.median(np.abs(window - median_val))
                    
                    # Check if value is outlier
                    if mad > 0:
                        z_score = np.abs(component[i] - median_val) / (1.4826 * mad)
                        if z_score > threshold:
                            # Replace with median
                            if component_idx == 0:
                                filtered_data[i, sc] = median_val + 1j * filtered_data[i, sc].imag
                            else:
                                filtered_data[i, sc] = filtered_data[i, sc].real + 1j * median_val
        
        return filtered_data
    
    def interpolate_problematic_subcarriers(self, csi_data, problematic_mask=None):
        """
        Interpolate problematic subcarriers from neighboring ones
        """
        if problematic_mask is None:
            problematic_mask = self.problematic_mask
            
        if problematic_mask is None or not np.any(problematic_mask):
            return csi_data
        
        filtered_data = csi_data.copy()
        n_subcarriers = csi_data.shape[1]
        
        for sc in range(n_subcarriers):
            if problematic_mask[sc]:
                # Interpolate from neighboring subcarriers
                left_sc = sc - 1
                right_sc = sc + 1
                
                while left_sc >= 0 and problematic_mask[left_sc]:
                    left_sc -= 1
                while right_sc < n_subcarriers and problematic_mask[right_sc]:
                    right_sc += 1
                
                if left_sc >= 0 and right_sc < n_subcarriers:
                    # Interpolate between neighbors
                    weight_left = 1 - (sc - left_sc) / (right_sc - left_sc)
                    weight_right = 1 - (right_sc - sc) / (right_sc - left_sc)
                    
                    # Interpolate both real and imaginary parts
                    filtered_data[:, sc] = (
                        weight_left * filtered_data[:, left_sc] + 
                        weight_right * filtered_data[:, right_sc]
                    )
                elif left_sc >= 0:
                    filtered_data[:, sc] = filtered_data[:, left_sc]
                elif right_sc < n_subcarriers:
                    filtered_data[:, sc] = filtered_data[:, right_sc]
                else:
                    # If no neighbors, use mean of all subcarriers
                    filtered_data[:, sc] = np.mean(filtered_data, axis=1)
        
        return filtered_data
    
    def apply_all_filters(self, csi_data, verbose=True):
        """
        Apply all filters in sequence: detect, notch, hampel, interpolate
        """
        if verbose:
            print("\nApplying Filters...")
            print("="*60)
        
        # Step 1: Detect problematic subcarriers
        if verbose:
            print("  Detecting problematic subcarriers...")
        
        dc_mask = self.detect_dc_subcarriers(csi_data)
        zero_mask = self.detect_permanent_zeros(csi_data)
        self.problematic_mask = dc_mask | zero_mask
        
        if verbose:
            print(f"    DC subcarriers: {np.sum(dc_mask)}")
            print(f"    Permanent zeros: {np.sum(zero_mask)}")
            print(f"    Total problematic: {np.sum(self.problematic_mask)}")
        
        # Step 2: Apply notch filter
        if verbose:
            print("  Applying notch filter...")
        filtered_data = self.apply_notch_filter(csi_data)
        
        # Step 3: Apply Hampel filter
        if verbose:
            print("  Applying Hampel filter...")
        filtered_data = self.apply_hampel_filter(filtered_data)
        
        # Step 4: Interpolate problematic subcarriers
        if self.config.interpolate_problematic and np.any(self.problematic_mask):
            if verbose:
                print("  Interpolating problematic subcarriers...")
            filtered_data = self.interpolate_problematic_subcarriers(filtered_data)
        
        if verbose:
            modified_count = np.sum(np.abs(filtered_data - csi_data) > 0.01)
            print(f"  Modified elements: {modified_count}")
            print("="*60)
        
        return filtered_data


def filter_csi_data(csi_data, config=None, verbose=True):
    """Convenience function to apply all filters"""
    filter_obj = CSIFilter(config)
    return filter_obj.apply_all_filters(csi_data, verbose)


if __name__ == "__main__":
    # Test the filters
    print("Testing filters...")
    test_data = np.random.randn(100, 64) + 1j * np.random.randn(100, 64)
    config = FilterConfig()
    filter_obj = CSIFilter(config)
    
    # Test individual functions
    dc_mask = filter_obj.detect_dc_subcarriers(test_data)
    zero_mask = filter_obj.detect_permanent_zeros(test_data)
    print(f"DC mask: {np.sum(dc_mask)} subcarriers")
    print(f"Zero mask: {np.sum(zero_mask)} subcarriers")
    
    # Test full filter chain
    filtered = filter_obj.apply_all_filters(test_data)
    print(f"Original shape: {test_data.shape}")
    print(f"Filtered shape: {filtered.shape}")
    print("Test completed!")