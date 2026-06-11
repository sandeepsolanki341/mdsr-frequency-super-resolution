"""
dataset.py
==========
PyTorch Dataset wrappers for MDSR training data.

Two backends are provided:

1. FreqnetTrainDataTemplate  — reads samples lazily from an HDF5 file.
2. FreqnetTrainDataLMDB      — reads samples from a single-file LMDB
   database via memory-mapped I/O (preferred for large datasets).

Why lazy, per-worker file handles?
----------------------------------
DataLoader workers are separate OS processes. Sharing an open h5py.File
or lmdb.Environment created in the main process across workers is fragile
and can cause read errors or corruption. Each worker therefore opens its
own handle on first __getitem__ call (see _ensure_open).

Why LMDB over naive HDF5 slicing?
---------------------------------
Loading the full dataset with f[key][:] copies everything into RAM and
overflows on large datasets. LMDB provides memory-mapped random access,
so the OS pages data in on demand and the working set stays small.
"""

import pickle
from typing import List

import h5py
import lmdb
import torch
from torch.utils.data import Dataset

from config import Config


class FreqnetTrainDataTemplate(Dataset):
    """HDF5-backed dataset with lazy per-worker file handles."""

    def __init__(self, h5_path: str = Config.H5_PATH, keys: List[str] = None):
        self._h5_path = h5_path
        self._keys = list(keys or Config.DATA_KEYS)
        self._h5 = None  # worker-local handle

        # Read length once from disk (cheap)
        with h5py.File(self._h5_path, "r") as f:
            self._len = int(f[self._keys[0]].shape[0])

    def __len__(self):
        return int(self._len)

    def _ensure_open(self):
        """Open an h5py.File handle per worker lazily."""
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, "r")

    def __getitem__(self, idx):
        self._ensure_open()
        patch = torch.from_numpy(self._h5[self._keys[0]][idx])
        target = torch.from_numpy(self._h5[self._keys[1]][idx])
        cached = torch.from_numpy(self._h5[self._keys[2]][idx])
        gt = torch.from_numpy(self._h5[self._keys[3]][idx])
        return patch, target, cached, gt

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FreqnetTrainDataLMDB(Dataset):
    """Single-file LMDB dataset using memory-mapped random access.

    Samples are stored as pickled dicts under zero-padded 8-digit keys
    ("00000000", "00000001", ...). Two metadata keys are expected:
    'length' (pickled int) and 'stats' (pickled {'mean','std'} arrays).
    """

    def __init__(self, db_path: str = Config.LMDB_PATH):
        self.db_path = db_path
        self.env = None

        # Open briefly in the main process to read the dataset length.
        # lock=False is critical for concurrent reads from multiple workers.
        env = lmdb.open(
            self.db_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=False,
        )
        with env.begin(write=False) as txn:
            self._len = pickle.loads(txn.get("length".encode("ascii")))
            stats = pickle.loads(txn.get("stats".encode("ascii")))
        env.close()
        # Normalization statistics, read once here to avoid re-opening the
        # same LMDB environment twice in one process (LMDB disallows it).
        self.mean = stats["mean"]
        self.std = stats["std"]

    def _ensure_open(self):
        """Open a per-worker LMDB env (single-file DB → subdir=False)."""
        if self.env is None:
            self.env = lmdb.open(
                self.db_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=False,
            )

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._ensure_open()
        key = f"{idx:08d}".encode("ascii")

        with self.env.begin(write=False) as txn:
            byte_data = txn.get(key)

        if byte_data is None:
            raise IndexError(
                f"LMDB key not found for idx={idx} (tried key={key!r}). "
                "Check LMDB key format or available keys."
            )

        sample = pickle.loads(byte_data)
        patch = torch.from_numpy(sample["patch"].copy())
        target = torch.from_numpy(sample["target"].copy())
        cached = torch.from_numpy(sample["cached"].copy())
        gt = torch.from_numpy(sample["gt"].copy())
        return patch, target, cached, gt

    def __del__(self):
        if self.env is not None:
            self.env.close()
            self.env = None
