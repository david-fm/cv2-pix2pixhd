from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, get_token
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_SUFFIX = "cv2-project-private-models"


load_dotenv(PROJECT_ROOT / ".env")


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

    baseline_file = PROJECT_ROOT / "checkpoints_baseline" / "best_repacked.pt"
    baseline_metrics = PROJECT_ROOT / "checkpoints_baseline" / "test_metrics.json"
    pix2pix_dir = PROJECT_ROOT / "checkpoints_pix2pixHD" / "pix2pixHD_monusg"

    if not baseline_file.exists():
        raise FileNotFoundError(
            f"No existe el archivo baseline esperado: {baseline_file}. "
            "Genera `best_repacked.pt` antes de subirlo."
        )
    if not pix2pix_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta Pix2PixHD esperada: {pix2pix_dir}")

    repo_id = args.repo_id
    token = args.token
    if not token:
        raise RuntimeError(
            "No se encontró token de Hugging Face. Configura HF_TOKEN o ejecuta `huggingface-cli login`."
        )

    print(f"Repositorio destino: {repo_id}")
    print(f"Baseline: {baseline_file}")
    print(f"Pix2PixHD: {pix2pix_dir}")

    if args.dry_run:
        print("Dry run activado. No se subirá ningún archivo.")
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

    api.upload_folder(
        folder_path=str(pix2pix_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="pix2pixHD/pix2pixHD_monusg",
        ignore_patterns=[".DS_Store"],
    )

    print("Subida completada.")
    print(
        "Sugerencia: el repo queda organizado con baseline/ y pix2pixHD/pix2pixHD_monusg/ para que luego la app los detecte fácilmente."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())