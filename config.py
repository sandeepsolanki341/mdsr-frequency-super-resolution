"""
config.py
=========
Central configuration for MDSR training and evaluation.

All filesystem paths live here. Override DATA_DIR / CHECKPOINT_DIR with
environment variables if your data lives elsewhere:

    export MDSR_DATA_DIR=/path/to/data
    export MDSR_CKPT_DIR=/path/to/checkpoints
"""

import os


class Config:
    # ── Paths (relative to repo root by default) ──────────────────────
    DATA_DIR = os.environ.get("MDSR_DATA_DIR", "data")
    CHECKPOINT_DIR = os.environ.get("MDSR_CKPT_DIR", "checkpoints")
    LOG_DIR = os.environ.get("MDSR_LOG_DIR", "TrainingLogs")

    # Single-file LMDB containing training samples + 'stats' + 'length' keys
    LMDB_PATH = os.path.join(DATA_DIR, "train_data.lmdb")

    # Optional HDF5 dataset (legacy path, used by FreqnetTrainDataTemplate)
    H5_PATH = os.path.join(DATA_DIR, "train_data.h5")

    # Directory of raw 32x32 image patches (input to data_preparation.py)
    PATCHES_DIR = os.environ.get("MDSR_PATCHES_DIR", os.path.join(DATA_DIR, "32by32patches"))

    # HDF5/LMDB sample keys
    DATA_KEYS = [
        "32by32patches_xtrain",
        "target_frequency_tensor_xtrain",
        "cached_frequency_xtrain",
        "groundtruth_frequency",
    ]

    # ── Model ─────────────────────────────────────────────────────────
    in_channels = 100
    out_channels = 100
    repeats = 8              # SHCN repeat layers
    layers_per_repeat = 10   # SHCN blocks per repeat (dilation D_d = l + 1)
    apply_weights = False

    # ── Training hyperparameters ──────────────────────────────────────
    batch_size = 64
    grad_accum_steps = 1
    effective_batch_size = batch_size * grad_accum_steps
    learning_rate = 1e-4
    num_epochs = 200

    # Cosine annealing with warm restarts
    scheduler_T0 = 30
    scheduler_eta_min = 1e-7

    # ── Logging / checkpointing ───────────────────────────────────────
    log_interval = 10
    save_interval = 5
    best_checkpoint_name = "mdsr_best.pth"

    # ── DataLoader ────────────────────────────────────────────────────
    num_workers = 6
    prefetch_factor = 2
    pin_memory = True
    persistent_workers = True

    @staticmethod
    def set_random_seed(seed_value):
        """Set seeds across random/numpy/torch for reproducibility.

        Pass None to disable. Note: enabling cuDNN determinism trades away
        some training speed.
        """
        import random

        import numpy as np
        import torch

        if seed_value is None:
            return
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
