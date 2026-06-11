"""
This version writes directly to a single HDF5 (.h5) file in appendable chunks,
so we never keep the whole dataset in RAM and we avoid creating a giant .pt.

logical keys inside HDF5 file :
 - 32by32patches_xtrain # B , 16 , W , H 
 - target_frequency_tensor_xtrain # B , 16 , C
 - cached_frequency_xtrain # B , 16 , #frequnecy coefficient - C
 - groundtruth_frequency # B , 16 , W , H
 - mean # (C ,)
 - std # (C,)

Normalization: After writing all samples, we compute mean/std across samples
for the target_frequency_tensor_xtrain in a streaming, two-pass manner and
write the normalized values back to the same dataset (no full in-memory load).
"""

import os
import sys
import torch
import numpy as np
import gc
from typing import  List,Any
import json 

# Optional dependency for on-disk, chunked storage
try:
    import h5py
except Exception as e:
    h5py = None
    # We don't raise here to allow import of this module;
    # the function will check and provide a clear hint if missing.

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import required modules
import  Vectorizedtools as tl
import  VectorizedInputPreprocessing as vip


def _to_cpu_numpy(x: Any, dtype=np.float32) -> np.ndarray:
    """
    Convert a numpy or torch tensor to a CPU numpy array of given dtype.
    Keeps shape as-is. This ensures we don't keep GPU tensors around and
    that we write compact arrays to HDF5.
    """
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def _ensure_h5_dset(f, name: str, first_batch: np.ndarray,
                    compression: str = "gzip", compression_opts: int = 4):
    """
    Create an HDF5 dataset if it doesn't exist yet. The dataset is created
    with maxshape=(None, ...) to allow appends along the first dimension.
    Chunk shape is chosen so that one chunk roughly equals an input batch.
    """
    if name in f:
        return f[name]
    maxshape = (None,) + tuple(first_batch.shape[1:])
    chunk0 = min(len(first_batch), 1024) if len(first_batch) > 0 else 1
    chunks = (chunk0,) + tuple(first_batch.shape[1:])
    return f.create_dataset(
        name,
        data=first_batch,
        maxshape=maxshape,
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts,
        dtype=first_batch.dtype,
    )


def _append_to_h5(f, name: str, batch: np.ndarray) -> None:
    """
    Append a batch (np.ndarray) to an HDF5 dataset by resizing axis 0.
    If the dataset doesn't exist, it is created first.
    """
    if name not in f:
        _ensure_h5_dset(f, name, batch)
        return
    dset = f[name]
    old_n = dset.shape[0]
    new_n = old_n + batch.shape[0]
    dset.resize((new_n,) + dset.shape[1:])
    dset[old_n:new_n] = batch


def process_and_downscale_dataset(
    root_dir: str,
    output_path: str,
    image_normalized: bool = True,
    maxdir: int = 50000,
    scale_factor: int = 4,
    chunk_size: int = 2000,
    normalize_target_freq: bool = True,
    reconstruction_order: int = 1
):
    """
    Processes image patches from subdirectories, downscales them, and saves them 
    into a single HDF5 file (appendable datasets). This avoids holding the full
    dataset in RAM and prevents memory exhaustion.

    Args:
        root_dir (str): The root directory of the dataset containing subdirectories 
                        of 32x32 image patches.
        output_path (str): The full path (including filename, e.g., /path/to/output.h5)
                           to save the final HDF5 file. If ".pt" is provided, it will
                           still write ".h5" next to it for safety.
        image_normalized (bool): Whether to normalize images during reading.
        maxdir (int): Maximum number of subdirectories to process (cap for big datasets).
        scale_factor (int): Down/up scale factor (e.g., 4 for x4).
        chunk_size (int): How many subdirectories to buffer before flushing to disk.
        normalize_target_freq (bool):performing the normalization technique introduced by the freqnet 
    """

    # Derive .h5 path if user passed a .pt
    h5_path = output_path

    settings = {
    "root_dir": root_dir,
    "output_path": output_path,
    "image_normalized": bool(image_normalized),
    "maxdir": int(maxdir) if maxdir is not None else None,
    "scale_factor": int(scale_factor),
    "normalize_target_freq": bool(normalize_target_freq),
    "reconstruction_order": int(reconstruction_order)
    }
    settings_json = json.dumps(settings)
    with h5py.File(h5_path, "a") as f:
    # store metadata as a JSON string attribute (preferred for small metadata)
        f.attrs["data-Settings"] = settings_json
    '''
    Import json, h5py
    with h5py.File(h5_path, "r") as f:
    settings = json.loads(f.attrs["data-Settings"])
    '''

    # Small in-memory buffers to collect samples and write in chunks
    buf_up: List[np.ndarray] = []
    buf_tf: List[np.ndarray] = []
    buf_cf: List[np.ndarray] = []
    buf_gt: List[np.ndarray] = []

    # Get all subdirectories in the root directory
    try:
        subdirectories = sorted([d.path for d in os.scandir(root_dir) if d.is_dir()])
    except FileNotFoundError:
        print(f"Error: Root directory not found at {root_dir}")
        return

    # If max_dirs is set, slice the list to the desired number
    if maxdir is not None and maxdir > 0:
        print(f"Capping processing to the first {maxdir} subdirectories.")
        subdirectories = subdirectories[:maxdir]
    total_dirs = len(subdirectories)
    print(f"Found {total_dirs} subdirectories to process.")

    #Note :- Each subdirectories should have 16 RGB image patches of spatial size 32 by 32  
    for i, sub_dir_path in enumerate(subdirectories):
        # Find all image files in the subdirectory
        try:
            image_files = sorted([
                os.path.join(sub_dir_path, f) for f in os.listdir(sub_dir_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            ])
        except OSError as e:
            print(f"Warning: Could not access files in {sub_dir_path}. Skipping. Error: {e}")
            continue

        if len(image_files) != 16:
            print(f"Warning: Skipping {sub_dir_path}. Expected 16 images, but found {len(image_files)}.")
            continue

        # 1. Load 16 images, convert to luminance, and stack into a tensor
        patch_tensors = []
        for img_path in image_files:
                luminance_patch = tl.Read_ConvertTo_Tensors(img_path, scale="grey", image_normalization = image_normalized)
                patch_tensors.append(torch.from_numpy(luminance_patch.squeeze(0)))

        # Stack patches into a single tensor for the subdirectory -> (16, 32, 32)
        stacked_tensor = np.stack(patch_tensors).astype(np.float32)

        # 2. Downscale the combined tensor batch
        downscaled_tensor = tl.Img_scaling(stacked_tensor, scale_factor=(1/scale_factor)) # high frequency details permanent loss.
        upscaled_tensor = tl.Img_scaling(downscaled_tensor , scale_factor=scale_factor)

        if reconstruction_order == 1:
            cutoff = [0.334]
        elif reconstruction_order == 2:
            cutoff = [0.478]
        #vectorized_input_preprocessing separates the frequency band given the cutoff frequency .
        groundtruth_entities = vip.vectorized_input_preprocessing(mini_batch_patches = stacked_tensor,pre_upscale_factor = 1,upscale_method = "bicubic",normalization = "none",cutoff_freqs = cutoff,flatten_frequency = False) #0.334 for extracting 100 frequnecy coefficients and 0.478 for extracting 201 frequency coefficients
        degraded_entities  = vip.vectorized_input_preprocessing(mini_batch_patches = downscaled_tensor,pre_upscale_factor = scale_factor,upscale_method = "bicubic",normalization = "none",cutoff_freqs = cutoff,flatten_frequency = True)
        ground_truth_target_frequency  , groundtruth_cached_frequency = groundtruth_entities["frequency bands"]
        target_frequency , cached_frequency = degraded_entities["frequency bands"]
        target_frequency_ = vip.vectors_to_grid(torch.from_numpy(target_frequency), channel_first=True)

        # Convert everything to CPU numpy and buffer
        buf_up.append(_to_cpu_numpy(upscaled_tensor)) # B , 16 , 32 , 32
        buf_tf.append(_to_cpu_numpy(target_frequency_)) # B  , C , 4 , 4 
        buf_cf.append(_to_cpu_numpy(cached_frequency)) # B , 16 , #frequnecy_coefficients - C
        buf_gt.append(_to_cpu_numpy(ground_truth_target_frequency) + _to_cpu_numpy(groundtruth_cached_frequency)) # B , 16 , C

        # Flush to HDF5 when buffer reaches chunk_size or on last iteration
        if len(buf_up) >= chunk_size or (i + 1) == total_dirs:
            # Stack buffered samples into arrays (N_chunk, ...)
            up_batch = np.stack(buf_up, axis=0)
            tf_batch = np.stack(buf_tf, axis=0)
            cf_batch = np.stack(buf_cf, axis=0)
            gt_batch = np.stack(buf_gt, axis=0)

            # Ensure output directory exists
            out_dir = os.path.dirname(h5_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            # Append to HDF5
            with h5py.File(h5_path, "a") as f:
                _append_to_h5(f, "32by32patches_xtrain", up_batch)
                _append_to_h5(f, "target_frequency_tensor_xtrain", tf_batch)
                _append_to_h5(f, "cached_frequency_xtrain", cf_batch)
                _append_to_h5(f, "groundtruth_frequency", gt_batch)
             

            # Clear buffers and free memory
            buf_up.clear(); buf_tf.clear(); buf_cf.clear(); buf_gt.clear()
            del up_batch, tf_batch, cf_batch, gt_batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Provide progress update
        if (i + 1) % 100 == 0 or (i + 1) == total_dirs:
            print(f"Processed {i + 1}/{total_dirs} subdirectories...")

    # 3) Optional: Normalize target_frequency_tensor_xtrain across samples in streaming fashion
    if normalize_target_freq:
        """
        Normalize target_frequency_tensor_xtrain using your pipeline:
        - Convert vectors -> grid via vip.vectors_to_grid (channel_first=True)
        - Compute stats via tl.zscore_normalize (returns normalized, mean, std)
        - Normalize with global mean/std
        - Convert grid -> vectors via vip.grid_to_vector (channel_last=False)
        This is done in two streaming passes over the HDF5 dataset.
        """
        if h5py is None:
            raise ImportError("h5py not available. Install with: pip install h5py")

        print("Normalizing target_frequency_tensor_xtrain using custom freqnet pipeline (streaming)...")
        eps = 1e-6

        with h5py.File(h5_path, "a") as f:
            tfd = f["target_frequency_tensor_xtrain"]
            N = tfd.shape[0]
            # Use moderate chunk size to control RAM (tune if needed)
            batch = max(1, min(4096, N))

            # Pass 1: compute global mean/std using your zscore function per-chunk.
            # We combine chunk statistics into global mean/std via sums/sumsq.
            running_sum = None
            running_sumsq = None

            for start in range(0, N, batch):
                end = min(N, start + batch)
                grid_np = tfd[start:end]  # shape: (B, ...)
                B = end - start

                grid_t = torch.from_numpy(grid_np).to(torch.float32)

                # (we ignore normalized output here; we only need mean/std for accumulating running mean and std)
                _, mean_chunk, std_chunk = tl.zscore_normalize(grid_t)

                mean_np = _to_cpu_numpy(mean_chunk, dtype=np.float64)
                std_np = _to_cpu_numpy(std_chunk, dtype=np.float64)
                # sum and sumsq from chunk stats
                if running_sum is None:
                    running_sum = mean_np * B
                    running_sumsq = (std_np**2 + mean_np**2) * B
                else:
                    running_sum += mean_np * B
                    running_sumsq += (std_np**2 + mean_np**2) * B

                # Free per-iteration tensors
                del grid_t, mean_chunk, std_chunk
                gc.collect()

            # Global mean/std in grid space
            mean = (running_sum / float(N)).astype(np.float32) #(C , )
            var = (running_sumsq / float(N)) - (mean.astype(np.float64) ** 2) 
            var = np.maximum(var, 0.0)
            std = np.sqrt(var).astype(np.float32) #(C ,)

            # Store mean/std in HDF5 (replace if they exist)
            if "mean" in f:
                del f["mean"]
            if "std" in f:
                del f["std"]
            f.create_dataset("mean", data=mean)
            f.create_dataset("std", data=std)

            '''mean and std was of shape (C,) which will cause broadcasting issue with grid_np , bellow 
            statements add channel to mean and std for proper brodcasting '''
            mean_b = mean[None, :, None, None] # (1 , C , 1 , 1)
            std_b  = std[None,  :, None, None] # (1 , C , 1 , 1)

            # Pass 2: normalize in-place using global mean/std and write back
            print("Writing normalized target_frequency_tensor_xtrain back to HDF5 (streaming)...")
            for start in range(0, N, batch):
                end = min(N, start + batch)
                grid_np = tfd[start:end]  # (B , C , H , W)

                '''mean and std was of shape (C,) which will cause broadcasting issue with grid_np , bellow 
                statements add channel to mean and std for proper brodcasting '''
                mean_b = mean[None, :, None, None] # (1 , C , 1 , 1)
                std_b  = std[None,  :, None, None] # (1 , C , 1 , 1)

                grid_np_norm = (grid_np - mean_b) / (std_b + eps) # normalizing grid_np channel wise

                # Write normalized slice back
                tfd[start:end] = grid_np_norm
                # Free per-iteration tensors
                del   grid_np , grid_np_norm
                gc.collect()

        print(f"Normalization complete and saved to {h5_path}")
            

# Example call:
# Note: Pass an .h5 output path. If a .pt path is given, the function will write a sibling .h5.


# Example HDF5 build (uncomment to use; paths come from config.py):
# process_and_downscale_dataset(
#     Config.PATCHES_DIR,
#     Config.H5_PATH,
#     image_normalized=True,
#     maxdir=3,
#     scale_factor=4,
#     chunk_size=1000,
#     normalize_target_freq=True,
#     reconstruction_order=1,
# )

'''
def process_and_downscale_dataset_LMDB(
    root_dir: str,
    lmdb_path: str,
    image_normalized: bool = True,
    maxdir: int = 50000,
    scale_factor: int = 4,
    chunk_size: int = 512,                # how many samples to buffer before writing
    buffer_byte_limit: int | None = None, # optional: flush when total buffered bytes exceed this
    normalize_target_freq: bool = True,
    reconstruction_order: int = 1,
    commit_interval: int = 4096,          # safety: commit inside large flushes every this many puts
    map_size: int = 20 * 1024**3,
    compression: bool = False,
    sync_interval_commits: int | None = None  # optional: call env.sync() every N commits
):
    import os, gc, zlib
    from tqdm import tqdm
    import lmdb, pickle

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    subdirectories = sorted([d.path for d in os.scandir(root_dir) if d.is_dir()])
    if maxdir is not None and maxdir > 0:
        subdirectories = subdirectories[:maxdir]
    total_dirs = len(subdirectories)
    if total_dirs == 0:
        raise RuntimeError("No subdirectories found to process.")

    os.makedirs(os.path.dirname(lmdb_path) or ".", exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=False, lock=True)

    # Buffering state
    buffer = []                   # list of (idx, sample_obj)
    buffered_bytes = 0            # approximate bytes in buffer (sum of np.nbytes)
    sample_idx = 0
    total_commits = 0

    def _sample_size_bytes(sobj):
        # approximate: sum of numpy array nbytes
        s = 0
        for v in (sobj["patch"], sobj["target"], sobj["cached"], sobj["gt"]):
            try:
                s += int(getattr(v, "nbytes", 0))
            except Exception:
                pass
        return s

    txn = env.begin(write=True)
    written_since_commit = 0

    def _flush_buffer_to_txn():
        nonlocal txn, written_since_commit, buffered_bytes, total_commits, buffer
        if not buffer:
            return
        for idx, s_obj in buffer:
            key = f"{idx:08d}".encode("ascii")
            val = pickle.dumps(s_obj, protocol=4)
            if compression:
                val = zlib.compress(val)
            txn.put(key, val)
            written_since_commit += 1
            # safety commit inside a big flush
            if written_since_commit >= commit_interval:
                txn.commit()
                total_commits += 1
                if sync_interval_commits and (total_commits % sync_interval_commits == 0):
                    env.sync()
                txn = env.begin(write=True)
                written_since_commit = 0
        buffer.clear()
        buffered_bytes = 0

    for i, sub_dir_path in enumerate(tqdm(subdirectories, desc="Processing to LMDB")):
        try:
            image_files = sorted([
                os.path.join(sub_dir_path, f) for f in os.listdir(sub_dir_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            ])
        except OSError:
            continue
        if len(image_files) != 16:
            continue

        # build sample (same processing pipeline)
        patch_tensors = []
        for img_path in image_files:
            luminance_patch = tl.Read_ConvertTo_Tensors(img_path, scale="grey", image_normalization=image_normalized)
            patch_tensors.append(torch.from_numpy(luminance_patch.squeeze(0)))
        stacked_tensor = np.stack(patch_tensors).astype(np.float32)
        downscaled_tensor = tl.Img_scaling(stacked_tensor, scale_factor=(1/scale_factor))
        upscaled_tensor = tl.Img_scaling(downscaled_tensor, scale_factor=scale_factor)

        if reconstruction_order == 1:
            cutoff = [0.334]
        elif reconstruction_order == 2:
            cutoff = [0.478]
        else:
            cutoff = [0.334]

        groundtruth_entities = vip.vectorized_input_preprocessing(
            mini_batch_patches=stacked_tensor, pre_upscale_factor=1, upscale_method="bicubic",
            normalization="none", cutoff_freqs=cutoff, flatten_frequency=False
        )
        degraded_entities = vip.vectorized_input_preprocessing(
            mini_batch_patches=downscaled_tensor, pre_upscale_factor=scale_factor, upscale_method="bicubic",
            normalization="none", cutoff_freqs=cutoff, flatten_frequency=True
        )
        ground_truth_target_frequency, groundtruth_cached_frequency = groundtruth_entities["frequency bands"]
        target_frequency, cached_frequency = degraded_entities["frequency bands"]
        target_frequency_ = vip.vectors_to_grid(torch.from_numpy(target_frequency), channel_first=True)

        patch_np = _to_cpu_numpy(upscaled_tensor)
        target_np = _to_cpu_numpy(target_frequency_)
        cached_np = _to_cpu_numpy(cached_frequency)
        gt_np = _to_cpu_numpy(ground_truth_target_frequency + groundtruth_cached_frequency)

        sample_obj = {"patch": patch_np, "target": target_np, "cached": cached_np, "gt": gt_np}

        # buffer
        buffer.append((sample_idx, sample_obj))
        sample_bytes = _sample_size_bytes(sample_obj)
        buffered_bytes += sample_bytes
        sample_idx += 1

        # flush conditions: reached chunk_size (count) or exceeded bytes limit (if provided)
        if len(buffer) >= chunk_size or (buffer_byte_limit is not None and buffered_bytes >= buffer_byte_limit):
            _flush_buffer_to_txn()

    # final flush & commit
    _flush_buffer_to_txn()
    if written_since_commit > 0:
        txn.commit()
        total_commits += 1
        written_since_commit = 0
    if sync_interval_commits and (total_commits % sync_interval_commits == 0):
        env.sync()

    # compute stats (same as before)
    stats = {}
    if normalize_target_freq:
        print("Computing mean/std for target_frequency_tensor_xtrain...")
        with env.begin(write=False) as rtxn:
            running_sum = None
            running_sumsq = None
            N = sample_idx
            batch = max(1, min(4096, N))
            for start in range(0, N, batch):
                arrs = []
                for k in range(start, min(N, start + batch)):
                    key = f"{k:08d}".encode("ascii")
                    val = rtxn.get(key)
                    if val is None:
                        continue
                    if compression:
                        val = zlib.decompress(val)
                    sample = pickle.loads(val)
                    arrs.append(sample["target"])
                if len(arrs) == 0:
                    continue
                grid_np = np.stack(arrs, axis=0)
                grid_t = torch.from_numpy(grid_np).to(torch.float32)
                _, mean_chunk, std_chunk = tl.zscore_normalize(grid_t)
                mean_np = _to_cpu_numpy(mean_chunk, dtype=np.float64)
                std_np = _to_cpu_numpy(std_chunk, dtype=np.float64)
                B = grid_np.shape[0]
                if running_sum is None:
                    running_sum = mean_np * B
                    running_sumsq = (std_np**2 + mean_np**2) * B
                else:
                    running_sum += mean_np * B
                    running_sumsq += (std_np**2 + mean_np**2) * B
                del grid_t, mean_chunk, std_chunk, grid_np
                gc.collect()
            if running_sum is not None:
                mean = (running_sum / float(N)).astype(np.float32)
                var = (running_sumsq / float(N)) - (mean.astype(np.float64) ** 2)
                var = np.maximum(var, 0.0)
                std = np.sqrt(var).astype(np.float32)
                stats["mean"] = mean
                stats["std"] = std
    else:
        stats["mean"] = None
        stats["std"] = None

    with env.begin(write=True) as final_txn:
        final_txn.put(b"stats", pickle.dumps(stats, protocol=4))
        final_txn.put(b"length", pickle.dumps(sample_idx))

    env.sync()
    env.close()
    print(f"LMDB created at {lmdb_path} with {sample_idx} samples.")

if __name__ == "__main__":
    from config import Config
    process_and_downscale_dataset_LMDB(
        Config.PATCHES_DIR,
        Config.LMDB_PATH,
        maxdir=2,
        chunk_size=3000,
        buffer_byte_limit=200_000_000,
    )
'''