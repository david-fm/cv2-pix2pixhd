#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def extract_label_half(input_image: Path, output_image: Path, binarize: bool) -> None:
    with Image.open(input_image) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        half = width // 2

        if half == 0:
            raise ValueError(f"Invalid image width for {input_image}")

        label = rgb.crop((half, 0, width, height)).convert("L")

        if binarize:
            label = label.point(lambda p: 255 if p > 127 else 0)

        label.save(output_image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract right-half label masks from MoNuSeg paired images."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("MoNuSeg"),
        help="Path to the MoNuSeg folder (default: ./MoNuSeg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for labels (default: sibling folder named MoNuSegLabels)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label files in output directory.",
    )
    parser.add_argument(
        "--no-binarize",
        action="store_true",
        help="Keep grayscale labels without thresholding to 0/255.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {input_dir}")

    output_dir = args.output_dir or (input_dir.parent / "MoNuSegLabels")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"No PNG files found in {input_dir}")

    written = 0
    skipped = 0
    for src in image_paths:
        dst = output_dir / src.name

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        extract_label_half(src, dst, binarize=not args.no_binarize)
        written += 1

    print(f"Input folder : {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Total images : {len(image_paths)}")
    print(f"Written      : {written}")
    print(f"Skipped      : {skipped}")


if __name__ == "__main__":
    main()
