import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
import zipfile

import gradio as gr
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import os
import shutil

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from models.models import create_model
from options.train_options import TrainOptions


PROJECT_ROOT = Path(__file__).resolve().parent
IMG_SIZE = 256

load_dotenv(PROJECT_ROOT / ".env")
HF_REPO_ID = os.environ.get("HF_REPO_ID")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

BASELINE_CKPT_CANDIDATES = [
    PROJECT_ROOT / "checkpoints_baseline" / "best.pt",
    PROJECT_ROOT / "checkpoints_baseline" / "best.pt.zip",
]
BASELINE_CKPT_DIR = PROJECT_ROOT / "checkpoints_baseline" / "best"
BASELINE_REPACKED_CKPT = PROJECT_ROOT / "checkpoints_baseline" / "best_repacked.pt"

PIX2PIXHD_DIR = PROJECT_ROOT / "checkpoints_pix2pixHD" / "pix2pixHD_monusg"


class UNetDown(nn.Module):
    def __init__(self, in_c: int, out_c: int, normalize: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_c,
                out_c,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not normalize,
            )
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_c: int, out_c: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, ngf: int = 64):
        super().__init__()
        self.d1 = UNetDown(in_channels, ngf, normalize=False)
        self.d2 = UNetDown(ngf, ngf * 2)
        self.d3 = UNetDown(ngf * 2, ngf * 4)
        self.d4 = UNetDown(ngf * 4, ngf * 8)
        self.d5 = UNetDown(ngf * 8, ngf * 8)
        self.d6 = UNetDown(ngf * 8, ngf * 8)
        self.d7 = UNetDown(ngf * 8, ngf * 8)
        self.d8 = UNetDown(ngf * 8, ngf * 8, normalize=False)

        self.u1 = UNetUp(ngf * 8, ngf * 8, dropout=0.5)
        self.u2 = UNetUp(ngf * 8 * 2, ngf * 8, dropout=0.5)
        self.u3 = UNetUp(ngf * 8 * 2, ngf * 8, dropout=0.5)
        self.u4 = UNetUp(ngf * 8 * 2, ngf * 8)
        self.u5 = UNetUp(ngf * 8 * 2, ngf * 4)
        self.u6 = UNetUp(ngf * 4 * 2, ngf * 2)
        self.u7 = UNetUp(ngf * 2 * 2, ngf)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)

        return self.final(u7)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = resolve_device()

_loaded_baseline: Optional[nn.Module] = None
_loaded_pix2pixhd: Optional[nn.Module] = None
_loaded_pix2pixhd_versions: Dict[str, nn.Module] = {}


def find_baseline_checkpoint() -> Optional[Path]:
    for ckpt in BASELINE_CKPT_CANDIDATES:
        if ckpt.exists():
            return ckpt
    if BASELINE_CKPT_DIR.exists() and BASELINE_CKPT_DIR.is_dir():
        return BASELINE_CKPT_DIR
    return None


def repack_baseline_checkpoint_from_dir(src_dir: Path, out_file: Path) -> Path:
    if out_file.exists() and out_file.stat().st_mtime >= src_dir.stat().st_mtime:
        return out_file

    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_STORED) as zf:
        for item in sorted(src_dir.rglob("*")):
            if item.is_file():
                arcname = item.relative_to(src_dir.parent)
                zf.write(item, arcname=arcname.as_posix())

    return out_file


def has_pix2pixhd_checkpoint() -> bool:
    return PIX2PIXHD_DIR.exists()


def list_pix2pixhd_versions() -> Dict[str, Path]:
    """Return mapping of choice label -> path to G checkpoint for each version found.

    Version is the prefix before '_net_' in filenames like '<version>_net_G.pth'.
    """
    out: Dict[str, Path] = {}
    if not PIX2PIXHD_DIR.exists():
        return out
    for p in sorted(PIX2PIXHD_DIR.glob("*_net_G.pth")):
        name = p.name
        if "_net_" in name:
            version = name.split("_net_")[0]
        else:
            version = name.replace(".pth", "")
        key = f"Pix2PixHD:{version}"
        out[key] = p
    return out


def list_pix2pixhd_versions_hf() -> Dict[str, str]:
    """Return mapping of version -> remote path in HF repo (if HF_REPO_ID configured)."""
    out: Dict[str, str] = {}
    if not HF_REPO_ID:
        return out
    api = HfApi(token=HF_TOKEN)
    try:
        files = api.list_repo_files(HF_REPO_ID)
    except Exception:
        return out
    prefix = "pix2pixHD/pix2pixHD_monusg/"
    for f in files:
        if f.startswith(prefix) and f.endswith("_net_G.pth"):
            name = Path(f).name
            if "_net_" in name:
                version = name.split("_net_")[0]
            else:
                version = name.replace(".pth", "")
            out[version] = f
    return out


def download_file_from_hf(path_in_repo: str, dst: Path) -> Path:
    """Download a single file from HF repo and copy it to dst path."""
    if not HF_REPO_ID:
        raise RuntimeError("HF_REPO_ID is not configured.")
    # hf_hub_download returns cached file path
    cached = hf_hub_download(repo_id=HF_REPO_ID, filename=path_in_repo, repo_type="model", token=HF_TOKEN)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cached, dst)
    return dst


def available_models() -> Dict[str, str]:
    models: Dict[str, str] = {}
    baseline_ckpt = find_baseline_checkpoint()
    if baseline_ckpt is not None:
        models["Pix2Pix baseline"] = str(baseline_ckpt)
    # Add each Pix2PixHD version as a separate selectable model
    for k, v in list_pix2pixhd_versions().items():
        models[k] = str(v)
    # Also include models available on Hugging Face if configured
    if HF_REPO_ID:
        hf_pix = list_pix2pixhd_versions_hf()
        for ver, remote in hf_pix.items():
            key = f"Pix2PixHD:{ver} (HF)"
            models[key] = f"hf://{HF_REPO_ID}/{remote}"
        # baseline in HF
        api = HfApi(token=HF_TOKEN)
        try:
            files = api.list_repo_files(HF_REPO_ID)
            if "baseline/best_repacked.pt" in files:
                models["Pix2Pix baseline (HF)"] = f"hf://{HF_REPO_ID}/baseline/best_repacked.pt"
        except Exception:
            pass
    return models


def denorm_to_pil(tensor: torch.Tensor) -> Image.Image:
    t = tensor.detach().cpu().clamp(-1, 1)
    t = (t + 1.0) / 2.0
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def preprocess_label_for_baseline(image: Image.Image) -> torch.Tensor:
    gray = image.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
    arr = np.array(gray, dtype=np.float32)
    arr = (arr > 127).astype(np.float32)
    arr = np.stack([arr, arr, arr], axis=0)
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor * 2.0 - 1.0


def preprocess_label_for_pix2pixhd(image: Image.Image) -> torch.Tensor:
    gray = image.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
    arr = np.array(gray, dtype=np.uint8)
    arr = (arr > 127).astype(np.int64)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def load_baseline_model() -> nn.Module:
    global _loaded_baseline
    if _loaded_baseline is not None:
        return _loaded_baseline

    ckpt_path = find_baseline_checkpoint()
    if ckpt_path is None:
        # Try to download from HF if configured
        if HF_REPO_ID:
            hf_path = "baseline/best_repacked.pt"
            try:
                print(f"Downloading baseline from HF: {HF_REPO_ID}/{hf_path}")
                download_file_from_hf(hf_path, BASELINE_REPACKED_CKPT)
                ckpt_path = BASELINE_REPACKED_CKPT
            except Exception:
                raise FileNotFoundError("Baseline checkpoint not found locally or on HF.")
        else:
            raise FileNotFoundError("Baseline checkpoint not found.")

    if ckpt_path.is_dir():
        ckpt_path = repack_baseline_checkpoint_from_dir(ckpt_path, BASELINE_REPACKED_CKPT)

    model = UNetGenerator(ngf=64).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "G" in ckpt:
        model.load_state_dict(ckpt["G"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    _loaded_baseline = model
    return model


def load_pix2pixhd_model(version: Optional[str] = None) -> nn.Module:
    """Load Pix2PixHD model for a specific version.

    If `version` is None and only one version is present, load that one.
    """
    global _loaded_pix2pixhd_versions

    local_versions = list_pix2pixhd_versions()
    hf_versions = list_pix2pixhd_versions_hf() if HF_REPO_ID else {}
    if not local_versions and not hf_versions:
        raise FileNotFoundError("No Pix2PixHD checkpoints found locally or on Hugging Face.")

    if version is None:
        all_versions = []
        all_versions.extend([k.split(":", 1)[1] for k in local_versions.keys()])
        all_versions.extend([v for v in hf_versions.keys()])
        if len(all_versions) == 1:
            version = all_versions[0]
        else:
            raise ValueError("You must specify the Pix2PixHD version when multiple checkpoints are available.")

    choice_key = f"Pix2PixHD:{version}"
    if choice_key in _loaded_pix2pixhd_versions:
        return _loaded_pix2pixhd_versions[choice_key]

    # Build options pointing to the pix2pixHD checkpoints directory
    # Ensure required checkpoint exists locally; if not, try to download from HF
    expected_g = PIX2PIXHD_DIR / f"{version}_net_G.pth"
    if not expected_g.exists():
        if HF_REPO_ID:
            hf_mapping = list_pix2pixhd_versions_hf()
            if version in hf_mapping:
                remote = hf_mapping[version]
                try:
                    print(f"Downloading {remote} from HF...")
                    download_file_from_hf(remote, expected_g)
                except Exception as e:
                    raise FileNotFoundError(f"Could not download {remote} from HF: {e}")
        # after attempt, if still missing, raise
    if not expected_g.exists():
        raise FileNotFoundError(f"Checkpoint G for version {version} not found locally or on HF.")

    opt = TrainOptions()
    opt.isTrain = False
    opt.device = DEVICE
    opt.gpu_ids = [0] if DEVICE.type == "cuda" else []
    opt.checkpoints_dir = str(PROJECT_ROOT / "checkpoints_pix2pixHD")
    opt.name = "pix2pixHD_monusg"
    opt.no_instance = True
    opt.label_nc = 2
    opt.input_nc = 2
    opt.output_nc = 3
    opt.fineSize = IMG_SIZE
    opt.loadSize = IMG_SIZE
    opt.which_epoch = version
    opt.verbose = False

    opt = opt.parse(save=False)
    if opt.no_instance:
        opt.input_nc = opt.label_nc

    model = create_model(opt)
    net = model.module if hasattr(model, "module") else model
    net.eval()

    _loaded_pix2pixhd_versions[choice_key] = net
    return net


def generate_image(label_image: Optional[Image.Image], model_type: str, pix2pix_version: Optional[str]) -> Tuple[Optional[Image.Image], str]:
    if label_image is None:
        return None, "You must upload a label image before generating."

    with torch.no_grad():
        if model_type == "Pix2Pix baseline":
            model = load_baseline_model()
            label_tensor = preprocess_label_for_baseline(label_image).to(DEVICE)
            fake = model(label_tensor).squeeze(0)
            out = denorm_to_pil(fake)
            return out, f"Generated with {model_type} on {DEVICE}."

        if model_type == "Pix2PixHD":
            version = pix2pix_version or None
            model = load_pix2pixhd_model(version)
            label_tensor = preprocess_label_for_pix2pixhd(label_image).to(DEVICE)
            inst = torch.zeros_like(label_tensor, device=DEVICE)
            fake = model.inference(label_tensor, inst, image=None).squeeze(0)
            out = denorm_to_pil(fake)
            return out, f"Generated with Pix2PixHD:{version} on {DEVICE}."

    return None, "Unrecognized model."


def build_demo() -> gr.Blocks:
    models = available_models()

    if not models:
        warning = (
            "No checkpoints found for inference. "
            "Check checkpoints_baseline and checkpoints_pix2pixHD/pix2pixHD_monusg."
        )
        warnings.warn(warning)

    # Determine default model type and available pix2pixhd versions
    has_baseline = any(k == "Pix2Pix baseline" for k in models.keys())
    local_versions = [k.split(":", 1)[1] for k in list_pix2pixhd_versions().keys()]
    hf_versions = list(list_pix2pixhd_versions_hf().keys()) if HF_REPO_ID else []
    # Merge local first, then HF-only versions, preserving order and uniqueness
    seen = set()
    pix2pix_versions = []
    for v in local_versions + hf_versions:
        if v not in seen:
            seen.add(v)
            pix2pix_versions.append(v)
    model_types = ["Pix2Pix baseline", "Pix2PixHD"]
    default_model_type = "Pix2Pix baseline" if has_baseline else ("Pix2PixHD" if pix2pix_versions else "Pix2Pix baseline")
    default_version = pix2pix_versions[0] if pix2pix_versions else None

    with gr.Blocks(title="MoNuSeg Generator") as demo:
        gr.Markdown("# Histology generator from label")
        gr.Markdown(
            "Upload a binary nuclei label. The generator produces a synthetic histology image."
        )

        with gr.Row():
            model_type = gr.Dropdown(
                choices=model_types,
                value=default_model_type,
                label="Model type",
            )
            pix2pix_versions_dropdown = gr.Dropdown(
                        choices=pix2pix_versions,
                        value=default_version,
                        label="Pix2PixHD version",
                        visible=(default_model_type == "Pix2PixHD" and bool(pix2pix_versions)),
                    )
            status = gr.Textbox(
                label="Status",
                value=f"Device: {DEVICE}",
                interactive=False,
            )

        label_input = gr.Image(type="pil", label="Input label")
        output_image = gr.Image(type="pil", label="Generated image")

        generate_btn = gr.Button("Generate")
        # Update versions dropdown visibility/choices when model type changes
        def _on_model_type_change(selected: str):
            # Use precomputed `pix2pix_versions` to avoid filesystem I/O inside the callback
            if selected == "Pix2PixHD":
                if pix2pix_versions:
                    return gr.update(visible=True, choices=pix2pix_versions, value=pix2pix_versions[0])
                return gr.update(visible=True, choices=[], value=None)
            # hide dropdown but keep its choices so re-showing restores them
            return gr.update(visible=False)

        model_type.change(fn=_on_model_type_change, inputs=[model_type], outputs=[pix2pix_versions_dropdown])

        generate_btn.click(
            fn=generate_image,
            inputs=[label_input, model_type, pix2pix_versions_dropdown],
            outputs=[output_image, status],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
