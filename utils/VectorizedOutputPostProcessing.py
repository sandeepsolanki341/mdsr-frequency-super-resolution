"""
Vectorized Output Post-Processing for FDSRM Pipeline
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

# Add the parent directory to path to import from SRmodel
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Import vectorized tools
from utils.Vectorizedtools import vectorized_idct , create_frequency_masks


class VectorizedMockDCT:
    """Vectorized mock DCT object for batch IDCT processing."""
    def __init__(self, zscore_coeffs_batch, mean_vals, std_vals):
        self.zscore = zscore_coeffs_batch.astype(np.float32)
        self.mean = mean_vals[:, None, None].astype(np.float32)
        self.std = std_vals[:, None, None].astype(np.float32)


def vectorized_output_postProcessing(low_passed_batch, 
                                   medium_passed_batch, 
                                   high_passed_batch, 
                                   rest_passed_batch, 
                                   normalization_object, # This normalization parameter stores the means and standard deviation for each image patch
                                   normalization = "z-score", 
                                   cutoff_freqs = [0.1, 0.3, 0.85],
                                   target_patch_size=None,
                                   is_flatten: bool = False,
                                   selective_combination = False) -> np.ndarray:
    """
    Vectorized post-processing function for FDSRM pipeline.
    
    Args:
        low_passed_batch: Output from Low Enhancement Network (LEN)
        medium_passed_batch: Output from medium frequency processing 
        high_passed_batch: Output from High Enhancement Network (HRN)
        rest_passed_batch: Output from Texture Restoration Network (TRN)
        normalization_object: Batch normalization object from preprocessing
        target_patch_size: Optional target size for downscaling
    
    Returns:
        np.ndarray: Final reconstructed images 
    """
    low = np.asarray(low_passed_batch, dtype=np.float32)
    medium = np.asarray(medium_passed_batch, dtype=np.float32)
    high = np.asarray(high_passed_batch, dtype=np.float32)
    rest = np.asarray(rest_passed_batch, dtype=np.float32)

    # If flattened inputs are provided, reconstruct full (N,H,W) bands using masks
    if is_flatten:
        # infer reference DCT shape from normalization_object
        coeff_attr = {"z-score": "zscore", "min-max": "min_max", "log": "log_norm"}.get(normalization)
        coeffs_ref = None
        if coeff_attr and hasattr(normalization_object, coeff_attr):
            coeffs_ref = getattr(normalization_object, coeff_attr)
        if coeffs_ref is None and hasattr(normalization_object, "dct_batch"):
            coeffs_ref = getattr(normalization_object, "dct_batch")
        if coeffs_ref is None or not (isinstance(coeffs_ref, np.ndarray) and coeffs_ref.ndim == 3):
            raise ValueError("Cannot infer (N,H,W) from normalization_object for unflattening; provide normalization object with dct batch or normalized coefficients.")

        N_ref, H, W = coeffs_ref.shape
        masks = create_frequency_masks(H, W, cutoff_freqs , "cpu")  # low, mid, texture, high
        print(masks[1].device)
        def _unflatten(flat_arr, mask):
            arr = np.asarray(flat_arr, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError("Expected flattened band array of shape (N, K)")
            if arr.shape[0] != N_ref:
                raise ValueError(f"Batch size mismatch when unflattening: expected {N_ref}, got {arr.shape[0]}")
            mask_bool = (mask.ravel()).cpu().numpy().astype(bool)
            if arr.shape[1] != mask_bool.sum():
                raise ValueError(f"Flattened length {arr.shape[1]} does not match mask true count {mask_bool.sum()}")
            full = np.zeros((N_ref, H * W), dtype=np.float32)
            full[:, mask_bool] = arr
            return full.reshape(N_ref, H, W)

        low = _unflatten(low, masks[0])
        medium = _unflatten(medium, masks[1])
        high = _unflatten(high, masks[2])
        rest = _unflatten(rest, masks[3])

        upscaled_patch_size = H
    else:
        upscaled_patch_size = low.shape[1]
    
    # Step 1: Vectorized combination of frequency bands
    if selective_combination == False:
        combined_coefficients = low + medium + high + rest
    if selective_combination == "low": 
        combined_coefficients = low 
    if selective_combination == "low_medium":
        combined_coefficients = low+medium
    if selective_combination == "low_high":
        combined_coefficients = low+high
    if selective_combination == "low_rest":
        combined_coefficients = low+rest
    if selective_combination == "medium_high":
        combined_coefficients = medium+high
    if selective_combination == "medium_high_rest":
        combined_coefficients = medium+high*5+rest*2  
    if selective_combination == "medium":
        combined_coefficients = medium*2
    if selective_combination == "high":
        combined_coefficients = high*4
    if selective_combination == "rest":
        combined_coefficients = rest
    if selective_combination == "high_rest":
        combined_coefficients = 3*high + 3*rest

    if (normalization.lower() == "none"):
        normalization_object = combined_coefficients # This is for only working purpose , might be confusing which it is 
    else:
        normalization_object.zscore = combined_coefficients
    print(type(normalization_object))
    # Step 3: Apply vectorized IDCT
    reconstructed_batch = vectorized_idct(normalization_object, is_normalized=normalization)

    final_batch = reconstructed_batch

    return final_batch




