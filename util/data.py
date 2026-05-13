"""MoNuSeg paired dataset and split helpers used by the baseline notebook.

Each PNG in the dataset folder is a horizontally-concatenated pair: left half =
real histology image (RGB), right half = binary nuclei mask. The dataset
returns the two halves as `(label_tensor, real_tensor)` in `[-1, 1]`.

Optional paired data augmentation (horizontal/vertical flip, 90 degrees
rotations and color jitter on the real image) is applied **synchronously** so
labels and targets stay aligned.
"""

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF


class MoNuSegDataset(Dataset):
    """Paired dataset for MoNuSeg.

    Parameters
    ----------
    root : Path
        Folder containing the side-by-side PNG pairs.
    img_size : int
        Output spatial size (square).
    augment : bool
        If True, apply paired flips + 90 degrees rotations to (label, real) and
        a color jitter to the real image only. Use this for the training split.
    """

    def __init__(self, root: Path, img_size: int = 256, augment: bool = False):
        self.files = sorted(Path(root).glob("*.png"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No PNG files found in {root}")
        self.img_size = img_size
        self.augment = augment

        self.resize_bicubic = transforms.Resize(
            (img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.resize_nearest = transforms.Resize(
            (img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST
        )
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # Color jitter is applied to the real image only, not the label.
        self.color_jitter = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02
        )

    def __len__(self):
        return len(self.files)

    def _paired_augment(self, real_pil: Image.Image, label_pil: Image.Image):
        """Apply the same geometric transforms to both halves, plus a
        photometric jitter to the real image only."""
        # Horizontal flip
        if random.random() < 0.5:
            real_pil = TF.hflip(real_pil)
            label_pil = TF.hflip(label_pil)
        # Vertical flip
        if random.random() < 0.5:
            real_pil = TF.vflip(real_pil)
            label_pil = TF.vflip(label_pil)
        # Random 90 degrees rotation (square -> stays square)
        k = random.randint(0, 3)
        if k:
            angle = 90 * k
            real_pil = TF.rotate(real_pil, angle, interpolation=TF.InterpolationMode.BICUBIC)
            label_pil = TF.rotate(label_pil, angle, interpolation=TF.InterpolationMode.NEAREST)
        # Color jitter on the real image only
        real_pil = self.color_jitter(real_pil)
        return real_pil, label_pil

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size
        half = w // 2
        real_pil  = img.crop((0,    0, half, h))   # left  -> histology
        label_pil = img.crop((half, 0, w,    h))   # right -> nuclei mask

        # Resize first so geometric augmentations don't change image size
        real_pil  = self.resize_bicubic(real_pil)
        label_pil = self.resize_nearest(label_pil.convert("RGB"))

        if self.augment:
            real_pil, label_pil = self._paired_augment(real_pil, label_pil)

        real_tensor  = self.normalize(TF.to_tensor(real_pil))
        label_tensor = self.normalize(TF.to_tensor(label_pil))
        return label_tensor, real_tensor


def split_indices(n_total: int, ratios=(0.70, 0.15, 0.15), seed: int = 42):
    """Deterministic 70/15/15 (default) shuffle-and-split of [0, n_total)."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    indices = list(range(n_total))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train = int(ratios[0] * n_total)
    n_val   = int(ratios[1] * n_total)
    train = indices[:n_train]
    val   = indices[n_train:n_train + n_val]
    test  = indices[n_train + n_val:]
    return train, val, test


def make_loaders(data_dir: Path,
                 img_size: int = 256,
                 batch_size: int = 16,
                 num_workers: int = 0,
                 seed: int = 42,
                 augment_train: bool = False):
    """Build (train, val, test) DataLoaders sharing a fixed split.

    `augment_train` toggles the paired data augmentation on the training
    split only — validation and test always use the clean transform.
    """
    full_clean = MoNuSegDataset(data_dir, img_size=img_size, augment=False)
    train_idx, val_idx, test_idx = split_indices(len(full_clean), seed=seed)

    if augment_train:
        full_train = MoNuSegDataset(data_dir, img_size=img_size, augment=True)
        train_set = Subset(full_train, train_idx)
    else:
        train_set = Subset(full_clean, train_idx)
    val_set  = Subset(full_clean, val_idx)
    test_set = Subset(full_clean, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return {
        "train_set": train_set, "val_set": val_set, "test_set": test_set,
        "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
    }


def denorm(t: torch.Tensor) -> torch.Tensor:
    """Map a tensor from [-1, 1] back to [0, 1] for display / metrics."""
    return (t.clamp(-1, 1) + 1.0) / 2.0
