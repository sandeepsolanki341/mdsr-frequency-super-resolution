import torch.nn.functional as F
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms.functional import rgb_to_grayscale

from scipy.fftpack import dct, idct
import numpy as np
import torch
from typing import List


def _ensure_numpy(arr):
    """Convert torch tensor to numpy if needed"""
    if isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    return arr

class NoNormalization:
    def __init__(self, batch):
        # We use .zscore as a consistent attribute name for the final output
        self.zscore = batch 
        self.dct_batch = batch
class BatchZscoreNormalization:
    """Vectorized z-score normalization for batch of DCT feature maps."""
    def __init__(self, dct_batch: np.ndarray):
        self.dct_batch = _ensure_numpy(dct_batch).astype(np.float32)
        self.mean = np.mean(self.dct_batch, axis=(1, 2), keepdims=True)
        self.std = np.std(self.dct_batch, axis=(1, 2), keepdims=True)
        self.std = np.where(self.std < 1e-8, 1e-8, self.std)
        self.zscore = ((self.dct_batch - self.mean) / self.std).astype(np.float32)
        self.mean_flat = self.mean.squeeze()
        self.std_flat = self.std.squeeze()


class BatchMinMaxNormalization:
    """Vectorized min-max normalization for batch of DCT feature maps."""
    def __init__(self, dct_batch: np.ndarray):
        self.dct_batch = _ensure_numpy(dct_batch).astype(np.float32)
        self.min = np.min(self.dct_batch, axis=(1, 2), keepdims=True)
        self.max = np.max(self.dct_batch, axis=(1, 2), keepdims=True)
        range_val = self.max - self.min
        range_val = np.where(range_val < 1e-8, 1e-8, range_val)
        self.min_max = ((self.dct_batch - self.min) / range_val).astype(np.float32)
        self.min_flat = self.min.squeeze()
        self.max_flat = self.max.squeeze()


class BatchLogNormalization:
    """Vectorized log normalization for batch of DCT feature maps."""
    def __init__(self, dct_batch: np.ndarray):
        self.dct_batch = _ensure_numpy(dct_batch).astype(np.float32)
        self.sign = np.sign(self.dct_batch)
        self.log_norm = np.log(np.abs(self.dct_batch) + 1e-5).astype(np.float32)


def vectorized_dct(image_batch: np.ndarray, normalization: str = "none"):
    """Vectorized DCT transformation for a batch of images."""
    image_batch = _ensure_numpy(image_batch).astype(np.float32)
    
    if image_batch.ndim != 3:
        raise ValueError(f"Expected 3D input (N, H, W), got shape {image_batch.shape}")
    
    # If input is uint8, assume [0, 255] and normalize to [0, 1]
    if image_batch.dtype == np.uint8:
        image_batch = image_batch.astype(np.float32) / 255.0
    
    dct_batch = np.zeros_like(image_batch, dtype=np.float32)
    for i in range(image_batch.shape[0]):
        # Apply 2D DCT
        dct_result = dct(dct(image_batch[i], axis=0, norm='ortho'), axis=1, norm='ortho')
        dct_batch[i] = dct_result.astype(np.float32)
    
    # --- Check for normalization type ---
    if normalization == "z-score":
        return BatchZscoreNormalization(dct_batch)
    elif normalization == "min-max":
        return BatchMinMaxNormalization(dct_batch)
    elif normalization == "log":
        return BatchLogNormalization(dct_batch)
    elif normalization == "none":
        # Create a simple class to hold the unnormalized data
        return NoNormalization(dct_batch)
    else:
        raise ValueError(f"Unsupported normalization type: {normalization}")


def vectorized_idct(dct_batch_normalized , is_normalized: str = "z-score") -> np.ndarray:
    """
    Vectorized IDCT transformation for a batch of DCT coefficients,
    with corresponding denormalization.
    Accepts 3D (N, H, W) or 4D (N, C, H, W) inputs and returns the reconstructed array
    with the same leading dims.
    """
    dct_denormalized = None
    
    # --- Denormalization Logic ---
    if is_normalized == "z-score":
        dct_denormalized = (dct_batch_normalized.zscore * dct_batch_normalized.std + 
                           dct_batch_normalized.mean)
    elif is_normalized == "min-max":
        range_val = dct_batch_normalized.max - dct_batch_normalized.min
        dct_denormalized = (dct_batch_normalized.min_max * range_val + 
                           dct_batch_normalized.min)
    elif is_normalized == "log":
        dct_denormalized = (dct_batch_normalized.sign * (np.exp(dct_batch_normalized.log_norm) - 1e-5))
    elif is_normalized == "none":
        # Consistently access the .zscore attribute which holds the raw DCT data
        dct_denormalized = _ensure_numpy(NoNormalization(dct_batch_normalized).zscore)
    else:
        raise ValueError(f"Unsupported normalization type: {is_normalized}")

    # Ensure numpy float32
    dct_denormalized = np.asarray(dct_denormalized).astype(np.float32)

    # Support both (N, H, W) and (N, C, H, W) inputs:
    # If 4D, collapse leading (N, C) into a single batch dimension for per-(H,W) IDCT.
    collapsed = False
    if dct_denormalized.ndim == 4:
        N, C, H, W = dct_denormalized.shape
        dct_flat = dct_denormalized.reshape(N * C, H, W)
        collapsed = True
    elif dct_denormalized.ndim == 3:
        dct_flat = dct_denormalized
    else:
        raise ValueError(f"Expected 3D or 4D DCT input, got shape {dct_denormalized.shape}")

    # --- Inverse DCT (IDCT) Application ---
    batch_size = dct_flat.shape[0]
    reconstructed_flat = np.zeros_like(dct_flat, dtype=np.float32)
    
    for i in range(batch_size):
        # Apply 2D IDCT over the last two axes (H, W)
        idct_result = idct(idct(dct_flat[i], axis=0, norm='ortho'), axis=1, norm='ortho')
        reconstructed_flat[i] = idct_result

    # Restore original leading dims if necessary
    if collapsed:
        reconstructed_batch = reconstructed_flat.reshape(N, C, H, W)
    else:
        reconstructed_batch = reconstructed_flat
    return reconstructed_batch

from typing import Tuple

def channel_dim_to_first(x):
    """
    Move channel dimension to the first position.
    Accepts:
      - torch.Tensor shape (N,H,W,C) -> returns (N,C,H,W)
      - torch.Tensor shape (H,W,C)    -> returns (C,H,W)
      - torch.Tensor already (N,C,H,W) or (C,H,W) -> returned unchanged
      - numpy array with same shapes -> returns numpy array with channels first
    Raises ValueError for unexpected dims.
    """
    try:
        import torch as _torch
    except Exception:
        _torch = None

    if _torch is not None and isinstance(x, _torch.Tensor):
        d = x.dim()
        if d == 4:
            # (N, H, W, C) -> (N, C, H, W)
            return x.permute(0, 3, 1, 2).contiguous()
        elif d == 3:
            # (H, W, C) -> (C, H, W)
            return x.permute(2, 0, 1).contiguous()
        else:
            raise ValueError(f"Unexpected tensor ndim {d}, expected 3 or 4")
    else:
        import numpy as _np
        arr = _np.asarray(x)
        d = arr.ndim
        if d == 4:
            # (N, H, W, C) -> (N, C, H, W)
            return arr.transpose(0, 3, 1, 2).copy()
        elif d == 3:
            # (H, W, C) -> (C, H, W)
            return arr.transpose(2, 0, 1).copy()
        else:
            raise ValueError(f"Unexpected array ndim {d}, expected 3 or 4")

def channel_dim_to_last(x):
    """
    Move channel dimension to the last position.

    Accepts:
      - torch.Tensor shape (N,C,H,W) -> returns (N,H,W,C)
      - torch.Tensor shape (C,H,W)    -> returns (H,W,C)
      - numpy array with same shapes -> returns numpy array with channels last

    Raises ValueError for unexpected dims.
    """
    try:
        import torch as _torch
    except Exception:
        _torch = None

    if _torch is not None and isinstance(x, _torch.Tensor):
        if x.dim() == 4:
            # (N, C, H, W) -> (N, H, W, C)
            return x.permute(0, 2, 3, 1).contiguous()
        elif x.dim() == 3:
            # (C, H, W) -> (H, W, C)
            return x.permute(1, 2, 0).contiguous()
        else:
            raise ValueError(f"Unexpected tensor ndim {x.dim()}, expected 3 or 4")
    else:
        import numpy as _np
        arr = _np.asarray(x)
        if arr.ndim == 4:
            # (N, C, H, W) -> (N, H, W, C)
            return arr.transpose(0, 2, 3, 1).copy()
        elif arr.ndim == 3:
            # (C, H, W) -> (H, W, C)
            return arr.transpose(1, 2, 0).copy()
        else:
            raise ValueError(f"Unexpected array ndim {arr.ndim}, expected 3 or 4")
    
def compute_channel_mean_std(x: torch.Tensor, eps: float = 1e-8, unbiased: bool = False):
    # validate input shape
    if x.dim() != 4:
        raise ValueError(f"expected 4D tensor (N,C,H,W), got {x.dim()}D with shape {tuple(x.shape)}")
    # ensure contiguous memory layout (safe before view/reshape or external use)
    x = x.contiguous()
    # mean over batch + spatial dims -> result shape (C,)
    # dims order: (N, C, H, W) -> reduce (0,2,3)
    mean = x.mean(dim=(0, 2, 3))
    # std over same dims; use unbiased=False for population std by default
    std = x.std(dim=(0, 2, 3), unbiased=unbiased)
    # avoid zeros in std (replace very small values with 1.0 to prevent NaNs on division)
    std = torch.where(std < eps, torch.ones_like(std), std)
    device = x.device #FInding where is the tensor located such that we can migrate the mean and std to same device for handling Expected all tensors to be on the same device, but found at least two devices anomly.
    return  mean.to(device), std.to(device)

def zscore_normalize(x: torch.Tensor):
    mean, std = compute_channel_mean_std(x) # If mean and std is not provided explicitly, compute from using compute_channel_mean_std over the x
    if x.dim() != 4:
        raise ValueError(f"expected 4D tensor (N,C,H,W), got {x.dim()}D")
    if mean.dim() != 1 or std.dim() != 1:
        raise ValueError("mean and std must be 1D tensors of shape (C,)")
    # broadcast mean/std to (N,C,H,W) using explicit None indices
    return (x - mean[None, :, None, None]) / std[None, :, None, None] , mean , std

def denormalize_zscore(x_norm: torch.Tensor, mean: torch.Tensor , std: torch.Tensor ) -> torch.Tensor:
    """
    Reverse z-score normalization for a 4D tensor (N, C, H, W).
    x_norm: normalized tensor (N,C,H,W)
    mean: per-channel mean (C,)
    std: per-channel std (C,)
    Returns: denormalized tensor same shape as x_norm
    """
    if x_norm.dim() != 4:
        raise ValueError(f"expected 4D tensor (N,C,H,W), got {x_norm.dim()}D")
    if mean.dim() != 1 or std.dim() != 1:
        raise ValueError("mean and std must be 1D tensors of shape (C,)")
    return x_norm * std[None, :, None, None] + mean[None, :, None, None]

def create_ring_weight_map(height: int, 
                           width: int ,
                           ring_boundaries = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32],
                           ring_weights = [1.0, 1.0, 5.0, 10.0, 10.0, 5.0, 1.0, 1.0]) -> Tuple[np.ndarray, int]:
    """Creates a 2D weight map using concentric rings based on Euclidean distance.
    Returns (weight_map, total_weighted_coeffs).
    """
    ring_boundaries = ring_boundaries
    ring_weights = ring_weights

    weight_map = np.ones((height, width), dtype=np.float32)
    Y, X = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    dist = np.sqrt((X - 0.0)**2 + (Y - 0.0)**2).astype(np.float32)
    min_dim = float(min(height, width))
    dist_norm = dist / min_dim

    total_weighted_coeffs = 0
    last_boundary = 0.0
    for i, boundary in enumerate(ring_boundaries):
        mask = (dist_norm >= last_boundary) & (dist_norm <= boundary)
        weight_map[mask] = ring_weights[i]
        total_weighted_coeffs += np.sum(mask)
        last_boundary = boundary

    return weight_map, int(total_weighted_coeffs)

def create_frequency_masks(
    height: int,
    width: int,
    cutoff_freqs: List[float],
    device: torch.device
) -> List[torch.Tensor]:
    """Create frequency masks as PyTorch tensors to preserve the computation graph."""
    cutoffs = torch.tensor(cutoff_freqs, dtype=torch.float32, device=device)
    if cutoffs.numel() == 0:
        raise ValueError("cutoff_freqs must contain at least one value")
    if torch.any(cutoffs < 0.0) or torch.any(cutoffs > 1.0):
        raise ValueError("cutoff_freqs values must be in [0, 1]")
    if not torch.all(torch.diff(cutoffs) >= 0):
        raise ValueError("cutoff_freqs must be sorted ascending")

    # Use torch.meshgrid instead of np.meshgrid
    Y, X = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij'
    )
    dist = torch.sqrt((X - 0.0)**2 + (Y - 0.0)**2)
    min_dim = float(min(height, width))
    dist_norm = dist / min_dim  # normalized distance

    masks: List[torch.Tensor] = []
    masks.append(dist_norm <= cutoffs[0]) # Return boolean masks
    for i in range(1, cutoffs.numel()):
        masks.append((dist_norm > cutoffs[i-1]) & (dist_norm <= cutoffs[i]))
    masks.append(dist_norm > cutoffs[-1])
    return masks



def vectorized_frequency_separation(dct_batch_normalized, 
                                   cutoff_freqs: List[float] = [0.1, 0.3, 0.85],
                                   normalization_type: str = "z-score",
                                   return_flat: bool = False):
    """Compact vectorized band separation. Returns list of (N,H,W) bands or (bands_full, bands_flat)."""
    # select attribute
    attr = {"z-score": "zscore", "min-max": "min_max", "log": "log_norm","none":"zscore"}.get(normalization_type)
    if attr is None:
        raise ValueError(f"Unsupported normalization type: {normalization_type}")
    dct_coeffs = getattr(dct_batch_normalized, attr)

    if dct_coeffs.ndim != 3:
        raise ValueError(f"Expected (N,H,W) dct_coeffs, got {dct_coeffs.shape}")
    N, H, W = dct_coeffs.shape

    masks = create_frequency_masks(H, W, cutoff_freqs , device=None)  # list of (H,W)
    
    if not return_flat:
        bands_full = [(dct_coeffs * (m[None, :, :].cpu().numpy())).astype(np.float32) for m in masks]
        return bands_full
    else:
        # build flattened compact bands (N, K_i)
        dct_flat = dct_coeffs.reshape(N, -1)
        bands_flat = []
        for m in masks:
            idx = (m.ravel()).cpu().numpy().astype(bool)
            bands_flat.append(dct_flat[:, idx].astype(np.float32) if idx.sum() else np.zeros((N, 0), dtype=np.float32))
        return bands_flat

import torch
import numpy as np
from typing import List

# UPDATED FUNCTION 1: Create masks directly as PyTorch tensors.
def create_frequency_masks_torch(
    height: int,
    width: int,
    cutoff_freqs: List[float],
    device: torch.device
) -> List[torch.Tensor]:
    """Create frequency masks as PyTorch tensors to preserve the computation graph."""
    cutoffs = torch.tensor(cutoff_freqs, dtype=torch.float32, device=device)
    if cutoffs.numel() == 0:
        raise ValueError("cutoff_freqs must contain at least one value")
    if torch.any(cutoffs < 0.0) or torch.any(cutoffs > 1.0):
        raise ValueError("cutoff_freqs values must be in [0, 1]")
    if not torch.all(torch.diff(cutoffs) >= 0):
        raise ValueError("cutoff_freqs must be sorted ascending")

    # Use torch.meshgrid instead of np.meshgrid
    Y, X = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij'
    )
    dist = torch.sqrt((X - 0.0)**2 + (Y - 0.0)**2)
    min_dim = float(min(height, width))
    dist_norm = dist / min_dim  # normalized distance

    masks: List[torch.Tensor] = []
    masks.append(dist_norm <= cutoffs[0]) # Return boolean masks
    for i in range(1, cutoffs.numel()):
        masks.append((dist_norm > cutoffs[i-1]) & (dist_norm <= cutoffs[i]))
    masks.append(dist_norm > cutoffs[-1])
    return masks


def vector_to_dctmap(
    low_band_flat: torch.Tensor,
    rest_band_flat: torch.Tensor,
    height: int = 32,
    width: int = 32,
    cutoff_freq: float = 0.334
):
    """
    Reconstruct a full 2D DCT map using only PyTorch operations,
    making it fully differentiable and suitable for training.
    """
    device = low_band_flat.device
    
    # Use the new PyTorch-native mask creation function
    masks = create_frequency_masks_torch(height, width, [cutoff_freq], device=device)
    low_mask = masks[0].view(-1)  # Flatten to 1D
    rest_mask = masks[1].view(-1) # Flatten to 1D

    # Helper function for aligning dimensions (already pure PyTorch)
    def _align_leading_torch(a, b):
        la, lb = tuple(a.shape[:-1]), tuple(b.shape[:-1])
        if la == lb: return a, b, la
        if len(la) == len(lb) + 1 and la[:-1] == lb:
            N = la[-1]
            b_exp = b.unsqueeze(-2).expand(*lb, N, b.shape[-1])
            return a, b_exp, la
        if len(lb) == len(la) + 1 and lb[:-1] == la:
            N = lb[-1]
            a_exp = a.unsqueeze(-2).expand(*la, N, a.shape[-1])
            return a_exp, b, lb
        raise ValueError(f"Cannot align leading dims: {la} vs {lb}")

    low_al, rest_al, leading_shape = _align_leading_torch(low_band_flat, rest_band_flat)
    
    hwsz = height * width
    
    # Create a zero tensor that will be filled. This operation is tracked by autograd.
    recon = torch.zeros(*leading_shape, hwsz, device=device, dtype=low_al.dtype)

    # Use boolean indexing to place the values. This is differentiable.
    recon[..., low_mask] = low_al
    recon[..., rest_mask] = rest_al

    # Reshape to the final 2D map. .view() is also tracked.
    return recon.view(*leading_shape, height, width)

def extract_patches(image, patch_size, stride):
    """
    Extracts patches from a (C, H, W) image array.
    
    Args:
        image (np.ndarray): Input image array of shape (C, H, W), dtype=np.uint8 or np.float32.
        patch_size (int): Size of the square patch (patch_size x patch_size).
        stride (int): Stride for patch extraction.
        
    Returns:
        np.ndarray: Patches of shape (num_patches, C, patch_size, patch_size), dtype=np.float32 in [0, 1].
    """
    image = _ensure_numpy(image)
    
    # Convert to float32 [0, 1] for processing
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)
    
    C, H, W = img.shape
    patches = []
    
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = img[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    
    patches = np.stack(patches, axis=0)  # (num_patches, C, patch_size, patch_size)
    return patches.astype(np.float32)

def Print_image(image_tensor, file_name, denormalization=True):
    """Save image array to file. Converts to uint8 [0, 255] for saving."""
    image_tensor = _ensure_numpy(image_tensor)
    
    if denormalization == True:
        img_min, img_max = image_tensor.min(), image_tensor.max()
        if img_max - img_min < 1e-8:
            # Constant image
            image_tensor = np.full_like(image_tensor, 127, dtype=np.uint8)
        elif img_max <= 2.0 and img_min >= -2.0:
            #Already in [0,1]
            image_tensor = np.clip(image_tensor * 255.0, 0, 255).astype(np.uint8)
        else:
            # Arbitrary range: normalize to [0,1] first
            image_tensor = (image_tensor - img_min) / (img_max - img_min)
            image_tensor = np.clip(image_tensor * 255.0, 0, 255).astype(np.uint8)
    else:
        image_tensor = image_tensor.astype(np.uint8)
    
    # Handle different shapes
    if image_tensor.ndim == 3:
        if image_tensor.shape[0] == 1:  # Grayscale (1, H, W)
            image_tensor = np.squeeze(image_tensor, axis=0)  # (H, W)
        else:  # RGB (3, H, W) -> (H, W, 3)
            image_tensor = np.transpose(image_tensor, (1, 2, 0))
    
    img_pil = Image.fromarray(image_tensor)
    img_pil.save(file_name)

def Read_ConvertTo_Tensors(path_to_image, scale="grey", image_normalization=True):
    """
    Reads an image from the specified path, converts it to grayscale or keeps RGB, and returns it as a NumPy array.
    
    Args:
        path_to_image (str): Path to the input image file.
        scale (str): "grey" for grayscale, "rgb" for RGB, otherwise RGB
        image_normalization (str): 'True' to normalize to [0,1], 'False' to keep [0,255]
        
    Returns:
        np.ndarray: Image as a NumPy array of dimension (1, H, W) for grayscale or (3, H, W) for RGB, dtype=np.float32.
    """
    # Read the image (returns uint8)
    img = read_image(path_to_image) 
    if img.shape[0] == 4:
        img = img[:3]
    if scale == "grey":
        img = rgb_to_grayscale(img)  # Shape: (1, H, W)

    if image_normalization:
        img = img / 255.0
        # Convert to numpy and float32
        img = img.cpu().numpy().astype(np.float32)
    else :
        img = img.cpu().numpy().astype(np.uint8)


    return img

def Img_scaling(image, scale_factor=4):
    """
    Downscales single image or batch of images using bicubic interpolation.
    
    Args:
        image: image array in float32 [0, 1]
               - Single image: (H, W) or (1, H, W)
               - Batch of images: (N, H, W) where N is number of patches
        downscale_factor: scale of downscaling (default: 4)
        
    Returns:
        torch.tensor: Downscaled image(s) float32 [0, 1]
                   - Single image: (1, H//scale, W//scale)
                   - Batch: (N, H//scale, W//scale)
    """
    image = _ensure_numpy(image)
    image = image.astype(np.float32)
    
    # Convert to torch for interpolation
    image_torch = torch.from_numpy(image)
    original_shape = image_torch.shape
    
    # Handle different input dimensions
    if image_torch.dim() == 2:
        # Single image (H, W) -> (1, 1, H, W)
        image_torch = image_torch.unsqueeze(0).unsqueeze(0)
        batch_mode = False
        single_channel = True
    elif image_torch.dim() == 3:
        if original_shape[0] == 1:
            # Single image with channel (1, H, W) -> (1, 1, H, W)
            image_torch = image_torch.unsqueeze(0)
            batch_mode = False
            single_channel = True
        else:
            # Batch of images (N, H, W) -> (N, 1, H, W)
            image_torch = image_torch.unsqueeze(1)
            batch_mode = True
            single_channel = True
    elif image_torch.dim() == 4:
        # Already in batch format (N, C, H, W)
        batch_mode = True
        single_channel = False
    else:
        raise ValueError(f"Unsupported image dimensions: {image_torch.shape}")
    
    # Perform downscaling using bicubic interpolation
    downscaled_torch = F.interpolate(
        image_torch, 
        scale_factor=scale_factor, 
        mode='bicubic', 
        align_corners=False 
    )
    
    # Convert back to numpy and adjust output shape
    downscaled_numpy = downscaled_torch.cpu().numpy().astype(np.float32)
    
    if not batch_mode:
        # Single image: remove batch dimension, keep channel dimension
        # (1, 1, H, W) -> (1, H, W)
        return downscaled_numpy.squeeze(0)
    else:
        # Batch of images: remove channel dimension if it was added
        if single_channel:
            # (N, 1, H, W) -> (N, H, W)
            return downscaled_numpy.squeeze(1)
        else:
            # (N, C, H, W) -> keep as is
            return downscaled_numpy



'''
IDCT can absolutely produce negative values!

Why IDCT produces negative values:

When you remove low frequencies (DC component):

DC coefficient represents the average brightness/offset
Removing it means reconstructed values oscillate around zero instead of around the image mean
Result: negative values for pixels that were below the average
Mathematical property of DCT/IDCT:

DCT is a linear transform — it doesn't constrain output range
IDCT of any DCT coefficients can produce any real values (positive or negative)
Only full DCT→IDCT of images originally in [0,1] guarantees output in [0,1] 
(approximately, with small numerical errors)

If you want the reconstructed image to show correct overall brightness you must add the 
DC / low-frequency component (or an equivalent offset) before/after IDCT. Medium/high bands 
are AC (detail) and are centered around zero — IDCT of AC-only produces positive and negative 
fluctuations around 0, so the image will look wrong (too dark, clipped, or banded) unless you 
restore the baseline.
'''