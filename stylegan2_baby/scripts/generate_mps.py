#!/usr/bin/env python3
"""Generate class-conditional StyleGAN2 images using MPS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STYLEGAN_DIR = PROJECT_ROOT / "vendor" / "stylegan2-ada-pytorch"
sys.path.insert(0, str(STYLEGAN_DIR))

import dnnlib  # noqa: E402
import legacy  # noqa: E402


def parse_range(value: str) -> list[int]:
    """Parse values such as '0-15' or '0,4,9'."""
    values: list[int] = []

    for part in value.split(","):
        part = part.strip()

        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)

            if end < start:
                raise ValueError(f"Invalid range: {part}")

            values.extend(range(start, end + 1))
        else:
            values.append(int(part))

    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--network",
        type=Path,
        required=True,
        help="Path to network-snapshot-*.pkl",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory for generated images",
    )
    parser.add_argument(
        "--seeds",
        default="0-15",
        help="Seed list or range, such as 0-15 or 1,4,8",
    )
    parser.add_argument(
        "--classes",
        default="0,1,2",
        help="Comma-separated conditional class indices",
    )
    parser.add_argument(
        "--trunc",
        type=float,
        default=0.7,
        help="Truncation psi",
    )
    parser.add_argument(
        "--noise-mode",
        choices=["const", "random", "none"],
        default="const",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.network.is_file():
        raise SystemExit(f"Snapshot not found: {args.network}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Loading: {args.network}")

    with dnnlib.util.open_url(str(args.network)) as network_file:
        network_data = legacy.load_network_pkl(network_file)

    generator = (
        network_data["G_ema"]
        .eval()
        .requires_grad_(False)
        .to(device)
    )

    seeds = parse_range(args.seeds)
    class_indices = parse_range(args.classes)

    if generator.c_dim == 0:
        raise SystemExit("This snapshot is not conditional.")

    print(f"Classes: {generator.c_dim}")
    print(f"Resolution: {generator.img_resolution}")

    for class_idx in class_indices:
        if class_idx < 0 or class_idx >= generator.c_dim:
            raise ValueError(
                f"Class {class_idx} is invalid for c_dim={generator.c_dim}"
            )

        class_dir = args.outdir / f"class_{class_idx}"
        class_dir.mkdir(parents=True, exist_ok=True)

        label = torch.zeros(
            [1, generator.c_dim],
            device=device,
            dtype=torch.float32,
        )
        label[:, class_idx] = 1

        for seed in seeds:
            rng = np.random.RandomState(seed)

            latent = torch.from_numpy(
                rng.randn(1, generator.z_dim).astype(np.float32)
            ).to(device)

            with torch.inference_mode():
                image = generator(
                    latent,
                    label,
                    truncation_psi=args.trunc,
                    noise_mode=args.noise_mode,
                    force_fp32=True,
                )

            image = (
                image.permute(0, 2, 3, 1) * 127.5 + 128
            ).clamp(0, 255).to(torch.uint8)

            output_path = class_dir / f"seed_{seed:06d}.png"

            Image.fromarray(
                image[0].cpu().numpy(),
                mode="RGB",
            ).save(output_path)

            print(f"Saved {output_path}")


if __name__ == "__main__":
    main()