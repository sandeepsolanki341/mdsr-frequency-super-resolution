"""
Vectorized Input Preprocessing for FDSRM Pipeline
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
from utils import Vectorizedtools as vt

def vectorized_input_preprocessing(mini_batch_patches: np.ndarray, 
                                 pre_upscale_factor: int, 
                                 upscale_method: str = "bicubic",
                                 normalization = "z-score",
                                 cutoff_freqs = [0.1, 0.3, 0.85],
                                 flatten_frequency: bool = False) -> dict:
    """
    Vectorized preprocessing function for FDSRM pipeline.
    This function mirrors the logic in Processing.py but processes entire batches at once.
    The frequency separation matches exactly:
    - Low pass: frequencies <= 0.1
    - High pass (mid): 0.1 < frequencies <= 0.3  
    - Texture pass: 0.3 < frequencies <= 0.85
    - Rest pass: frequencies > 0.85
    """
    
    # Step 1: Vectorized batch upscaling using PyTorch
    patches_tensor = torch.from_numpy(mini_batch_patches.astype(np.float32)).unsqueeze(1)  # Add channel dim
    upscaled_tensor = F.interpolate(
        patches_tensor, 
        scale_factor=pre_upscale_factor, 
        mode=upscale_method.lower(), 
        align_corners=False
    )
    upscaled_patches = upscaled_tensor.squeeze(1).cpu().numpy().astype(np.float32)
    
    # Step 2: Vectorized DCT transformation for entire batch
    dct_batch_normalized = vt.vectorized_dct(upscaled_patches, normalization=normalization)

    # Step 3: Vectorized frequency band separation 
    if not flatten_frequency:
        frequency_bands = vt.vectorized_frequency_separation(
            dct_batch_normalized,
            cutoff_freqs=cutoff_freqs,
            normalization_type=normalization,
            return_flat=False
        )
        # return only full bands + normalization object when flatten_frequency is False
        return {"frequency bands":[band.astype(np.float32) for band in frequency_bands] ,"norm_params":dct_batch_normalized}
    else:
    # flatten_frequency == True
        bands_flat = vt.vectorized_frequency_separation(
            dct_batch_normalized,
            cutoff_freqs = cutoff_freqs,
            normalization_type = normalization,
            return_flat = True
        )
        # return only flattened bands + normalization object when flatten_frequency is True
        return {"frequency bands":[band.astype(np.float32) for band in bands_flat] ,"norm_params":dct_batch_normalized}

    
def channel_dim_to_first(x: "torch.Tensor") -> "torch.Tensor":
    """
    Move channel dimension to the first position (torch.Tensor only).
    - (N, H, W, C) -> (N, C, H, W)
    - (H, W, C)    -> (C, H, W)

    Raises:
      - TypeError if input is not a torch.Tensor
      - ValueError for unexpected dims
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("channel_dim_to_first expects a torch.Tensor")
    d = x.dim()
    if d == 4:
        return x.permute(0, 3, 1, 2).contiguous()
    elif d == 3:
        return x.permute(2, 0, 1).contiguous()
    else:
        raise ValueError(f"Unexpected tensor dim {d}, expected 3 or 4")


def vectors_to_grid(vectors: "torch.Tensor", channel_first: bool = True) -> "torch.Tensor":
    """
    Arrange N vectors into a sqrt(N) x sqrt(N) grid (torch.Tensor only).

    Accepts:
      - (N, K) -> returns (S, S, K)
      - (B, N, K) -> returns (B, S, S, K)

    If channel_first is True the returned tensor will be channels-first:
      - (S, S, K) -> (K, S, S)
      - (B, S, S, K) -> (B, K, S, S)
    """
    if not isinstance(vectors, torch.Tensor):
        raise TypeError("vectors_to_grid expects a torch.Tensor")
    import math

    d = vectors.dim()
    if d == 2:
        N, K = vectors.shape
        S = math.isqrt(N)
        if S * S != N:
            raise ValueError(f"Number of vectors N={N} is not a perfect square")
        out = vectors.reshape(S, S, K)
    elif d == 3:
        B, N, K = vectors.shape
        S = math.isqrt(N)
        if S * S != N:
            raise ValueError(f"Number of vectors per batch N={N} is not a perfect square")
        out = vectors.reshape(B, S, S, K)
    else:
        raise ValueError("Input must be a torch.Tensor of shape (N,K) or (B,N,K)")
    if channel_first:
        out = channel_dim_to_first(out)
        return out
    else:
        return out


def channel_dim_to_last(x: "torch.Tensor") -> "torch.Tensor":
    """
    Move channel dimension to the last position (torch.Tensor only).

    - (N, C, H, W) -> (N, H, W, C)
    - (C, H, W)    -> (H, W, C)

    Raises TypeError if input is not a torch.Tensor.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("channel_dim_to_last expects a torch.Tensor")
    d = x.dim()
    if d == 4:
        return x.permute(0, 2, 3, 1).contiguous()
    elif d == 3:
        return x.permute(1, 2, 0).contiguous()
    else:
        raise ValueError(f"Unexpected tensor dim {d}, expected 3 or 4")


def grid_to_vector(grid: "torch.Tensor", channel_last: bool = True) -> "torch.Tensor":
    """
    Reverse of vectors_to_grid (torch.Tensor only).

    If channel_last is True expects grid with channels last:
      - (S, S, K) -> (N, K)
      - (B, S, S, K) -> (B, N, K)

    If channel_last is False expects channels-first:
      - (K, S, S) or (B, K, S, S) -> will be converted internally.
    Returns torch.Tensor.
    """
    if not isinstance(grid, torch.Tensor):
        raise TypeError("grid_to_vector expects a torch.Tensor")

    g = grid
    if not channel_last:
        # convert channels-first -> channels-last
        g = channel_dim_to_last(g)

    d = g.dim()
    if d == 3:
        S1, S2, K = g.shape
        if S1 != S2:
            raise ValueError(f"Grid spatial dims must be equal squares, got {S1}x{S2}")
        N = S1 * S2
        return g.reshape(N, K)
    elif d == 4:
        B, S1, S2, K = g.shape
        if S1 != S2:
            raise ValueError(f"Grid spatial dims must be equal squares, got {S1}x{S2}")
        N = S1 * S2
        return g.reshape(B, N, K)
    else:
        raise ValueError("Input must be a torch.Tensor of shape (S,S,K) or (B,S,S,K)")
