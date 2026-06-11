"""
train.py
========
MDSR training entry point.

Trains the Multi-Scale DCT Super-Resolution Network on DCT-domain patch
data stored in LMDB, with gradient accumulation, cosine-annealing warm
restarts, and best-loss checkpointing.

Usage:
    python train.py
    MDSR_DATA_DIR=/path/to/data python train.py
"""

import os

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

import model_utils.MDSR as mdsr
from config import Config
from dataset import FreqnetTrainDataLMDB
from utils.VectorizedInputPreprocessing import grid_to_vector
from utils.Vectorizedtools import denormalize_zscore, vector_to_dctmap


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ──────────────────────────────────────────────────────────
    # The dataset holds only a pointer to the LMDB file; samples are
    # memory-mapped on demand instead of loaded into RAM up front.
    dataset = FreqnetTrainDataLMDB(Config.LMDB_PATH)
    mean_np, std_np = dataset.mean, dataset.std

    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        persistent_workers=Config.persistent_workers,
        prefetch_factor=Config.prefetch_factor,
        shuffle=True,
        pin_memory=Config.pin_memory,
        in_order=False,  # requires torch >= 2.6; remove on older versions
    )

    # ── Model / optimizer / loss ──────────────────────────────────────
    model = mdsr.MDSR(
        in_channels=Config.in_channels,
        out_channels=Config.out_channels,
        repeats=Config.repeats,
        layers_per_repeat=Config.layers_per_repeat,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.scheduler_T0, eta_min=Config.scheduler_eta_min
    )
    criterion = mdsr.FrequencyDomainLoss2()

    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)

    # ── Bookkeeping ───────────────────────────────────────────────────
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    effective_batch = Config.batch_size * Config.grad_accum_steps
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Samples in dataset      : {len(dataset)}")
    print(f"Mini-batches per epoch  : {len(dataloader)}")
    print(f"Grad accumulation steps : {Config.grad_accum_steps} "
          f"(effective batch = {effective_batch})")
    print(f"Model parameters        : {total:,} (trainable: {trainable:,})")

    best_loss = float("inf")
    cost_history = np.array([])

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(Config.num_epochs):
        model.train()
        epoch_losses = np.array([])
        optimizer.zero_grad()

        for i, (patches, normalized_target_freq, cached_freq, ground_truth) in enumerate(
            tqdm.tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}")
        ):
            # Forward pass on normalized DCT feature maps
            output = model(normalized_target_freq.to(device))

            # Post-process model output back into a DCT feature map:
            # denormalize → flatten grid to coefficient vector → merge with
            # cached (reliable low-frequency) coefficients.
            denorm = denormalize_zscore(output, mean, std)
            denorm_vec = grid_to_vector(denorm, channel_last=False)
            reconstructed_dct = vector_to_dctmap(
                denorm_vec, cached_freq.to(device), height=32, width=32
            )

            loss = criterion(reconstructed_dct, ground_truth.to(device))
            epoch_losses = np.append(epoch_losses, loss.item())

            # Scale loss so accumulated gradients average correctly.
            (loss / float(Config.grad_accum_steps)).backward()

            if (i + 1) % Config.grad_accum_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses.size else float("inf")
        cost_history = np.append(cost_history, epoch_loss)
        print(f"Epoch {epoch} mean loss: {epoch_loss:.6f}")

        # Save best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(Config.CHECKPOINT_DIR, Config.best_checkpoint_name)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "loss_history": cost_history.tolist(),
                },
                best_path,
            )
            print(f"Saved best checkpoint -> {best_path} (loss={epoch_loss:.6f})")

    # Persist the per-epoch loss curve
    log_path = os.path.join(Config.LOG_DIR, "training_cost_log.csv")
    np.savetxt(log_path, cost_history, delimiter=",")
    print(f"Training loss log saved to {log_path}")


if __name__ == "__main__":
    main()
