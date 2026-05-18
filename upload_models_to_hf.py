from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo, get_token
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_SUFFIX = "cv2-project-private-models"


load_dotenv(PROJECT_ROOT / ".env")


def _resolve_best_source(ckpt_dir: Path) -> Path | None:
    """Return the local checkpoint source for a baseline-style folder.

    Accepts `best.pt` (file), `best.pt.zip`, or `best/` (PyTorch new-format
    directory). Returns None if nothing usable exists.
    """
    for cand in (ckpt_dir / "best.pt", ckpt_dir / "best.pt.zip"):
        if cand.exists() and cand.is_file():
            return cand
    best_dir = ckpt_dir / "best"
    if best_dir.exists() and best_dir.is_dir():
        return best_dir
    return None


def _repack_if_dir(src: Path, out_file: Path) -> Path:
    """If `src` is a directory, zip its contents into `out_file`; otherwise return `src`."""
    if src.is_file():
        return src
    if out_file.exists() and out_file.stat().st_mtime >= src.stat().st_mtime:
        return out_file
    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_STORED) as zf:
        for item in sorted(src.rglob("*")):
            if item.is_file():
                arcname = item.relative_to(src.parent)
                zf.write(item, arcname=arcname.as_posix())
    return out_file


def _prepare_baseline_file(ckpt_dir: Path) -> Path | None:
    """Resolve and (if needed) repack the baseline checkpoint. Returns the file to upload, or None."""
    if not ckpt_dir.exists():
        return None
    src = _resolve_best_source(ckpt_dir)
    if src is None:
        return None
    return _repack_if_dir(src, ckpt_dir / "best_repacked.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the trained MoNuSeg models to a Hugging Face repository."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face repo id, e.g. your-username/cv2-project-private-models",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or get_token(),
        help="Hugging Face access token. Defaults to .env / HF_TOKEN / HUGGINGFACE_HUB_TOKEN or the cached CLI token.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist yet.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without sending files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    baseline_dir = PROJECT_ROOT / "checkpoints_baseline"
    baseline_aug_dir = PROJECT_ROOT / "checkpoints_baseline_aug"
    pix2pix_dir = PROJECT_ROOT / "checkpoints_pix2pixHD" / "pix2pixHD_monusg"

    baseline_file = _prepare_baseline_file(baseline_dir)
    if baseline_file is None:
        raise FileNotFoundError(
            f"No baseline checkpoint found under {baseline_dir} "
            "(expected `best.pt`, `best.pt.zip`, or a `best/` directory)."
        )
    baseline_metrics = baseline_dir / "test_metrics.json"

    baseline_aug_file = _prepare_baseline_file(baseline_aug_dir)
    baseline_aug_metrics = baseline_aug_dir / "test_metrics.json"

    if not pix2pix_dir.exists():
        raise FileNotFoundError(f"Expected Pix2PixHD folder does not exist: {pix2pix_dir}")

    repo_id = args.repo_id
    token = args.token
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN or run `huggingface-cli login`."
        )

    print(f"Target repository: {repo_id}")
    print(f"Baseline: {baseline_file}")
    if baseline_aug_file is not None:
        print(f"Baseline + aug: {baseline_aug_file}")
    else:
        print(f"Baseline + aug: not found under {baseline_aug_dir} (skipping)")
    print(f"Pix2PixHD: {pix2pix_dir}")

    if args.dry_run:
        print("Dry run enabled. No file will be uploaded.")
        return 0

    create_repo(repo_id, token=token, private=args.private, exist_ok=True)
    api = HfApi(token=token)

    api.upload_file(
        path_or_fileobj=str(baseline_file),
        path_in_repo="baseline/best_repacked.pt",
        repo_id=repo_id,
        repo_type="model",
    )

    if baseline_metrics.exists():
        api.upload_file(
            path_or_fileobj=str(baseline_metrics),
            path_in_repo="baseline/test_metrics.json",
            repo_id=repo_id,
            repo_type="model",
        )

    if baseline_aug_file is not None:
        api.upload_file(
            path_or_fileobj=str(baseline_aug_file),
            path_in_repo="baseline_aug/best_repacked.pt",
            repo_id=repo_id,
            repo_type="model",
        )
        if baseline_aug_metrics.exists():
            api.upload_file(
                path_or_fileobj=str(baseline_aug_metrics),
                path_in_repo="baseline_aug/test_metrics.json",
                repo_id=repo_id,
                repo_type="model",
            )

    api.upload_folder(
        folder_path=str(pix2pix_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="pix2pixHD/pix2pixHD_monusg",
        ignore_patterns=[".DS_Store"],
    )

    print("Upload complete.")
    print(
        "Tip: the repo is organised with baseline/, baseline_aug/ and pix2pixHD/pix2pixHD_monusg/ so the app can detect them easily later."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())