"""Test-set evaluation helpers (L1 / PSNR / SSIM / FID)."""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import denorm


def _import_metrics():
    """Import torchmetrics, installing on the fly if missing."""
    try:
        from torchmetrics.image import (
            StructuralSimilarityIndexMeasure,
            PeakSignalNoiseRatio,
        )
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "torchmetrics[image]", "torch-fidelity"])
        from torchmetrics.image import (
            StructuralSimilarityIndexMeasure,
            PeakSignalNoiseRatio,
        )
        from torchmetrics.image.fid import FrechetInceptionDistance
    return StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, FrechetInceptionDistance


@torch.no_grad()
def evaluate_test_set(generator: nn.Module, test_loader: DataLoader, device) -> dict:
    """Compute L1, PSNR, SSIM and FID on a test loader.

    Returns a dict with float values for each metric.
    """
    SSIM, PSNR, FID = _import_metrics()

    ssim_metric = SSIM(data_range=1.0).to(device)
    psnr_metric = PSNR(data_range=1.0).to(device)
    fid_metric  = FID(feature=2048, normalize=True).to("cpu")

    generator.eval()
    l1_total, n = 0.0, 0
    for label, real in test_loader:
        label = label.to(device).float(); real = real.to(device)
        fake = generator(label)

        fake01 = denorm(fake)
        real01 = denorm(real)

        l1_total += torch.mean(torch.abs(fake01 - real01)).item() * label.size(0)
        n        += label.size(0)
        ssim_metric.update(fake01, real01)
        psnr_metric.update(fake01, real01)
        fid_metric.update(real01.to("cpu"), real=True)
        fid_metric.update(fake01.to("cpu"), real=False)

    return {
        "L1":   l1_total / max(n, 1),
        "PSNR": psnr_metric.compute().item(),
        "SSIM": ssim_metric.compute().item(),
        "FID":  fid_metric.compute().item(),
    }


@torch.no_grad()
def show_qualitative(generator: nn.Module, test_set, device, n_show: int = 6, save_path=None):
    """Plot a (label | generated | real) panel for `n_show` test samples."""
    generator.eval()
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    for i in range(n_show):
        label, real = test_set[i]
        fake = generator(label.unsqueeze(0).to(device).float()).squeeze(0).cpu()
        for j, (img, title) in enumerate([
            (denorm(label), "label"),
            (denorm(fake),  "generated"),
            (denorm(real),  "real"),
        ]):
            axes[i, j].imshow(img.float().permute(1, 2, 0).numpy())
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
