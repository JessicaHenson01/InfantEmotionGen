#!/usr/bin/env python3
"""Download the private infant StyleGAN dataset from Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from stylegan2_baby.hf_dataset import download_dataset_zip  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="InfantEmotionGen/baby_samples_gan",
        help="Private Hugging Face dataset repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "baby_samples_gan.zip",
        help="Normalized local output ZIP.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit hash for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        path = download_dataset_zip(args.repo_id, args.output, args.revision)
    except Exception as exc:
        raise SystemExit(
            f"Dataset download failed: {exc}\n"
            "Confirm that `hf auth login` was run with an account that has "
            "access to the private dataset."
        ) from exc

    print(f"Dataset ready: {path}")


if __name__ == "__main__":
    main()
