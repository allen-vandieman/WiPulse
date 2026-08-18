# tune_heartbeat_filter.py
# Fine-Tune Bandpass Filter Systematically
# Create a sweep script to find optimal frequencies:

import numpy as np
from heartbeat_extraction import HeartbeatExtractor, HeartbeatConfig
from validate_heartbeat import HeartbeatValidator

def sweep_filter_parameters(csi_data, gt_df, metadata):
    """Sweep bandpass parameters to find optimal settings"""
    results = []
    
    # Test ranges
    low_freqs = np.arange(0.3, 1.2, 0.1)
    high_freqs = np.arange(1.8, 4.0, 0.2)
    orders = [2, 3, 4, 5]
    
    best_score = 0
    best_params = None
    
    for low in low_freqs:
        for high in high_freqs:
            if low >= high:
                continue
            for order in orders:
                try:
                    validator = HeartbeatValidator()
                    validator.csi_data = csi_data
                    validator.metadata = metadata
                    validator.groundtruth_df = gt_df
                    
                    validator.extract_csi_heartbeat(low, high, order)
                    validator.align_timestamps()
                    validator.compute_validation_metrics()
                    
                    f1 = validator.metrics.get('f1_score', 0)
                    results.append({
                        'low': low, 'high': high, 'order': order,
                        'f1': f1, 'precision': validator.metrics.get('precision', 0),
                        'recall': validator.metrics.get('recall', 0),
                        'hr_mae': validator.metrics.get('heart_rate_mae', 999)
                    })
                    
                    if f1 > best_score:
                        best_score = f1
                        best_params = (low, high, order)
                        
                    print(f"low={low:.1f}, high={high:.1f}, order={order}: F1={f1:.3f}")
                    
                except Exception as e:
                    print(f"Error with low={low}, high={high}: {e}")
                    continue
    
    return results, best_params