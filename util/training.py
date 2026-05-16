"""Training loop and visual helpers for the baseline Pix2Pix.

The notebook orchestrates everything; the actual loop lives here so it stays
short and reusable across experiments (e.g., training with and without data
augmentation).
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from .data import denorm


@torch.no_grad()
def validate_l1(generator: nn.Module, loader: DataLoader, device) -> float:
    """Mean L1 in pixel space ([0, 1]) over `loader`."""
    generator.eval()
    total, n = 0.0, 0
    for label, real in loader:
        label, real = label.to(device).float(), real.to(device)
        fake = generator(label)
        total += torch.mean(torch.abs(denorm(fake) - denorm(real))).item() * label.size(0)
        n     += label.size(0)
    generator.train()
    return total / max(n, 1)


@torch.no_grad()
def save_sample_grid(generator: nn.Module, dataset, path: Path, device, n: int = 6):
    """Save a (label | fake | real) grid for n fixed samples from `dataset`."""
    generator.eval()
    rows = []
    for i in range(n):
        label, real = dataset[i]
        fake = generator(label.unsqueeze(0).to(device).float()).squeeze(0).cpu()
        label_display = denorm(label.float()).expand(3, -1, -1)
        rows.extend([label_display, denorm(fake), denorm(real)])
    grid = make_grid(torch.stack(rows), nrow=3, padding=2)
    plt.figure(figsize=(6, 2 * n))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off"); plt.title("label  |  fake  |  real")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    generator.train()


def train_pix2pix(G: nn.Module,
                  D: nn.Module,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  val_set,
                  device,
                  *,
                  epochs: int = 200,
                  lr: float = 2e-4,
                  betas=(0.5, 0.999),
                  lambda_l1: float = 100.0,
                  save_dir: Path,
                  samples_dir: Path,
                  sample_every: int = 10,
                  log_every: int = 1,
                  early_stopping_patience: int = 30,
                  early_stopping_min_delta: float = 1e-4):
    """Standard Pix2Pix training loop (BCE adversarial + L1 reconstruction).

    Returns the training history dict and best validation L1. Saves the best
    and last checkpoints to `save_dir` and periodic sample grids to
    `samples_dir`.
    """
    save_dir = Path(save_dir);    save_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = Path(samples_dir); samples_dir.mkdir(parents=True, exist_ok=True)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1  = nn.L1Loss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    history = {"g_loss": [], "d_loss": [], "g_gan": [], "g_l1": [], "val_l1": []}
    best_val = float("inf")
    best_epoch = 0
    es_counter = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        epoch_g, epoch_d, epoch_gan, epoch_l1 = 0.0, 0.0, 0.0, 0.0

        for label, real in train_loader:
            label = label.to(device, non_blocking=True).float()
            real  = real.to(device,  non_blocking=True) 

            # Train D
            opt_D.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = G(label)
            pred_real = D(label, real)
            pred_fake = D(label, fake)
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_D.step()

            # Train G
            opt_G.zero_grad(set_to_none=True)
            fake = G(label)
            pred_fake_for_g = D(label, fake)
            loss_g_gan = criterion_gan(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_g_l1  = criterion_l1(fake, real)
            loss_g = loss_g_gan + lambda_l1 * loss_g_l1
            loss_g.backward()
            opt_G.step()

            epoch_g   += loss_g.item()
            epoch_d   += loss_d.item()
            epoch_gan += loss_g_gan.item()
            epoch_l1  += loss_g_l1.item()

        n_batches = len(train_loader)
        history["g_loss"].append(epoch_g   / n_batches)
        history["d_loss"].append(epoch_d   / n_batches)
        history["g_gan"].append (epoch_gan / n_batches)
        history["g_l1"].append  (epoch_l1  / n_batches)

        val_l1 = validate_l1(G, val_loader, device)
        history["val_l1"].append(val_l1)

        if val_l1 < best_val - early_stopping_min_delta:
            best_val, best_epoch = val_l1, epoch
            es_counter = 0
            torch.save({"G": G.state_dict(), "D": D.state_dict(),
                        "epoch": epoch, "val_l1": val_l1},
                       save_dir / "best.pt")
        else:
            es_counter += 1
            if es_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {early_stopping_patience} epochs). "
                      f"Best val L1 = {best_val:.4f} at epoch {best_epoch}.")
                break

        if epoch % log_every == 0 or epoch == epochs:
            elapsed = time.time() - start
            print(f"epoch {epoch:03d}/{epochs} | "
                  f"D {history['d_loss'][-1]:.3f} | "
                  f"G {history['g_loss'][-1]:.3f} "
                  f"(gan {history['g_gan'][-1]:.3f}, l1 {history['g_l1'][-1]:.3f}) | "
                  f"val L1 {val_l1:.4f}  best {best_val:.4f} | "
                  f"{elapsed/60:.1f} min")

        if epoch % sample_every == 0 or epoch == epochs:
            save_sample_grid(G, val_set, samples_dir / f"epoch_{epoch:03d}.png", device, n=6)

    torch.save({"G": G.state_dict(), "D": D.state_dict(),
                "epoch": epochs, "val_l1": history["val_l1"][-1]},
               save_dir / "last.pt")
    print(f"Done. Best val L1 = {best_val:.4f} at epoch {best_epoch} (ckpt: {save_dir/'best.pt'})")

    return history, best_val, best_epoch


def plot_losses(history: dict):
    """Three-panel loss visualisation for a training run."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["g_loss"], label="G total")
    axes[0].plot(history["d_loss"], label="D")
    axes[0].set_xlabel("epoch"); axes[0].set_title("Adversarial losses"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["g_gan"], label="G_GAN")
    axes[1].plot(history["g_l1"],  label="G_L1 (raw)")
    axes[1].set_xlabel("epoch"); axes[1].set_title("Generator components"); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(history["val_l1"])
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("L1 (pixel space)")
    axes[2].set_title("Validation L1"); axes[2].grid(True)

    plt.tight_layout()
    plt.show()
