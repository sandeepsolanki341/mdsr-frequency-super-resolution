"""
benchmark_dataloader.py
=======================
Measures DataLoader throughput for the LMDB-backed dataset.

This exists because the original HDF5 full-slice approach
(f[key][:]) loaded the entire dataset into RAM and overflowed on
large datasets. The LMDB + lazy per-worker handle design fixes that;
this script quantifies the resulting throughput.

Usage:
    python benchmark_dataloader.py [--epochs 1]
"""

import argparse
import time

import tqdm
from torch.utils.data import DataLoader

from config import Config
from dataset import FreqnetTrainDataLMDB


def main():
    parser = argparse.ArgumentParser(description="Benchmark LMDB DataLoader throughput")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    dataset = FreqnetTrainDataLMDB(Config.LMDB_PATH)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        persistent_workers=Config.persistent_workers,
        prefetch_factor=Config.prefetch_factor,
        shuffle=True,
        pin_memory=Config.pin_memory,
    )

    print(f"Samples: {len(dataset)} | Batch size: {Config.batch_size} "
          f"| Workers: {Config.num_workers}")

    for epoch in range(args.epochs):
        start = time.perf_counter()
        n_samples = 0
        for patches, target, cached, gt in tqdm.tqdm(dataloader, desc=f"Epoch {epoch}"):
            n_samples += patches.shape[0]
        elapsed = time.perf_counter() - start
        print(f"Epoch {epoch}: {n_samples} samples in {elapsed:.2f}s "
              f"-> {n_samples / elapsed:.1f} samples/sec")


if __name__ == "__main__":
    main()
