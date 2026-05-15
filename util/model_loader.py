"""
Model loader utilities for loading Pix2PixHD models from HuggingFace.
This module provides functions to download and load models from HF, similar to gradio_app.py.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

# HuggingFace configuration
HF_REPO_ID = os.environ.get("HF_REPO_ID", "DavidFlorezM/cv2-monuseg-translation")
HF_TOKEN = os.environ.get("HF_TOKEN")


def _build_choice(suffix: str, epoch: str) -> str:
    """Build choice string from suffix and epoch."""
    if suffix == "pix2pixHD_monusg":
        return epoch
    return f"{suffix}/{epoch}"


def _parse_choice(choice: str) -> Tuple[str, str]:
    """Parse choice string into suffix and epoch."""
    if "/" in choice:
        suffix, epoch = choice.split("/", 1)
    else:
        suffix = "pix2pixHD_monusg"
        epoch = choice
    return suffix, epoch


def _scan_local_pix2pixhd_dir(base_dir: Path) -> Dict[str, Path]:
    """Scan a base directory for Pix2PixHD checkpoint folders.

    Structure:
    - <base_dir>/pix2pixHD_monusg/<epoch>_net_G.pth → choice = "epoch"
    - <base_dir>/pix2pixHD_monusg_<suffix>/<epoch>_net_G.pth → choice = "<suffix>/epoch"

    Returns {"suffix/epoch": Path} mapping.
    """
    out: Dict[str, Path] = {}
    if not base_dir.exists() or not base_dir.is_dir():
        return out

    # Scan base folder (pix2pixHD_monusg)
    base_folder = base_dir / "pix2pixHD_monusg"
    if base_folder.is_dir():
        for p in sorted(base_folder.glob("*_net_G.pth")):
            epoch = p.name.split("_net_")[0]
            out[_build_choice("pix2pixHD_monusg", epoch)] = p

    # Scan suffixed folders (pix2pixHD_monusg_*)
    for subfolder in sorted(base_dir.iterdir()):
        if subfolder.is_dir() and subfolder.name.startswith("pix2pixHD_monusg_"):
            suffix = subfolder.name[len("pix2pixHD_monusg_"):]
            for p in sorted(subfolder.glob("*_net_G.pth")):
                epoch = p.name.split("_net_")[0]
                out[_build_choice(suffix, epoch)] = p

    return out


def list_pix2pixhd_versions(checkpoints_dir: Path) -> Dict[str, Path]:
    """Return mapping of choice label -> path to G checkpoint for each version found locally.

    Args:
        checkpoints_dir: Base directory containing model folders (e.g., checkpoints_pix2pixHD)

    Returns:
        Dict mapping "suffix/epoch" or "epoch" to checkpoint Path
    """
    return _scan_local_pix2pixhd_dir(checkpoints_dir)


def list_pix2pixhd_versions_hf() -> Dict[str, str]:
    """Return mapping of choice label -> HF path for each Pix2PixHD version in the repo.

    Scans pix2pixHD/pix2pixHD_monusg[_suffix]/*_net_G.pth in the HF repo.
    Choice label format: "suffix/epoch" (e.g. "v1/200") or just "epoch" for base.
    """
    out: Dict[str, str] = {}
    if not HF_REPO_ID:
        return out

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return out

    api = HfApi(token=HF_TOKEN)
    try:
        files = api.list_repo_files(HF_REPO_ID)
    except Exception:
        return out

    prefix = "pix2pixHD/"
    for f in files:
        if not f.startswith(prefix) or not f.endswith("_net_G.pth"):
            continue
        remainder = f[len(prefix):]
        if remainder.startswith("pix2pixHD_monusg_"):
            suffix_part, name_part = remainder.split("/", 1)
            suffix = suffix_part[len("pix2pixHD_monusg_"):]
        elif remainder.startswith("pix2pixHD_monusg/"):
            suffix = "pix2pixHD_monusg"
            name_part = remainder[len("pix2pixHD_monusg/"):]
        else:
            suffix = "pix2pixHD_monusg"
            name_part = remainder
        name = Path(name_part).name
        epoch = name.split("_net_")[0] if "_net_" in name else name.replace(".pth", "")
        out[_build_choice(suffix, epoch)] = f
    return out


def download_file_from_hf(path_in_repo: str, dst: Path) -> Path:
    """Download a single file from HF repo and copy it to dst path.

    Args:
        path_in_repo: Path within the HF repo (e.g., "pix2pixHD/pix2pixHD_monusg/175_net_G.pth")
        dst: Destination path where to save the file

    Returns:
        Path to the downloaded file
    """
    if not HF_REPO_ID:
        raise RuntimeError("HF_REPO_ID no está configurado.")

    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=path_in_repo,
        repo_type="model",
        token=HF_TOKEN
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cached, dst)
    return dst


def load_pix2pixhd_from_hf(
    choice: str,
    checkpoints_dir: Path,
    device: torch.device,
    img_size: int = 256,
    label_nc: int = 2,
) -> nn.Module:
    """Load a Pix2PixHD model from HuggingFace.

    This function downloads the checkpoint from HF if needed, then loads the model
    using the same logic as in gradio_app.py.

    Args:
        choice: Model choice in format "epoch" (base) or "suffix/epoch" (e.g., "augmented/175")
        checkpoints_dir: Base directory for checkpoints (e.g., Path("checkpoints_pix2pixHD"))
        device: Torch device to load the model on
        img_size: Image size (default 256)
        label_nc: Number of label classes (default 2 for binary masks)

    Returns:
        Loaded Pix2PixHD model in eval mode

    Example:
        >>> model = load_pix2pixhd_from_hf("175", Path("checkpoints_pix2pixHD"), device)
        >>> model = load_pix2pixhd_from_hf("augmented/175", Path("checkpoints_pix2pixHD"), device)
    """
    from options.train_options import TrainOptions
    from models.models import create_model

    # Parse choice to get suffix and epoch
    suffix, epoch = _parse_choice(choice)

    # Check if model exists locally
    local_versions = list_pix2pixhd_versions(checkpoints_dir)

    local_path: Optional[Path] = None
    if choice in local_versions:
        local_path = local_versions[choice]
    else:
        # Check if exists in expected location
        if suffix == "pix2pixHD_monusg":
            candidate = checkpoints_dir / "pix2pixHD_monusg" / f"{epoch}_net_G.pth"
        else:
            candidate = checkpoints_dir / f"pix2pixHD_monusg_{suffix}" / f"{epoch}_net_G.pth"
        if candidate.exists():
            local_path = candidate

    # Get HF path
    hf_versions = list_pix2pixhd_versions_hf()
    hf_path = hf_versions.get(choice)

    # Determine final checkpoint path
    expected_g: Optional[Path] = None

    if local_path is not None and local_path.exists():
        expected_g = local_path
    elif hf_path is not None:
        # Download to correct folder based on suffix
        if suffix == "pix2pixHD_monusg":
            expected_g = checkpoints_dir / "pix2pixHD_monusg" / f"{epoch}_net_G.pth"
        else:
            expected_g = checkpoints_dir / f"pix2pixHD_monusg_{suffix}" / f"{epoch}_net_G.pth"
        expected_g.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando {hf_path} desde HF...")
        download_file_from_hf(hf_path, expected_g)
    else:
        available = list(hf_versions.keys()) if hf_versions else []
        raise FileNotFoundError(
            f"Checkpoint '{choice}' no encontrado localmente ni en HF. "
            f"Modelos disponibles: {available}"
        )

    # Configure options
    opt = TrainOptions()
    opt.isTrain = False
    opt.device = device
    opt.gpu_ids = [0] if device.type == "cuda" else []
    opt.checkpoints_dir = str(checkpoints_dir)
    if suffix == "pix2pixHD_monusg":
        opt.name = "pix2pixHD_monusg"
    else:
        opt.name = f"pix2pixHD_monusg_{suffix}"
    opt.no_instance = True
    opt.label_nc = label_nc
    opt.input_nc = label_nc
    opt.output_nc = 3
    opt.fineSize = img_size
    opt.loadSize = img_size
    opt.which_epoch = epoch
    opt.verbose = False

    opt = opt.parse(save=False)
    if opt.no_instance:
        opt.input_nc = opt.label_nc

    # Create and load model
    model = create_model(opt)
    net = model.module if hasattr(model, "module") else model
    net.eval()

    return net
