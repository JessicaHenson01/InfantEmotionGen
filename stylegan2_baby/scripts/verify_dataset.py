#!/usr/bin/env python3
"""Validate a StyleGAN2 ZIP dataset and summarize its labels."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=PROJECT_ROOT / "data" / "baby_samples_gan.zip",
    )
    parser.add_argument(
        "--sample-images",
        type=int,
        default=25,
        help="Number of images to inspect for dimensions and mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = args.zip_path.expanduser().resolve()
    if not zip_path.is_file():
        raise SystemExit(
            f"Dataset not found: {zip_path}\n"
            "Run `python scripts/download_dataset.py` first."
        )

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        if "dataset.json" not in names:
            raise SystemExit("dataset.json is missing from the dataset ZIP.")

        metadata = json.loads(archive.read("dataset.json"))
        labels = metadata.get("labels")
        if not isinstance(labels, list) or not labels:
            raise SystemExit("dataset.json does not contain a non-empty labels list.")

        missing = [name for name, _ in labels if name not in names]
        if missing:
            preview = "\n".join(f"  - {name}" for name in missing[:10])
            raise SystemExit(
                f"{len(missing)} labeled files are missing from the ZIP:\n{preview}"
            )

        counts = Counter(label for _, label in labels)
        print(f"ZIP: {zip_path}")
        print(f"Labeled images: {len(labels)}")
        print("Class counts:")
        for label, count in sorted(counts.items(), key=lambda item: str(item[0])):
            print(f"  {label!r}: {count}")

        image_info = Counter()
        sample_count = min(args.sample_images, len(labels))
        for filename, _ in labels[:sample_count]:
            with archive.open(filename) as stream:
                with Image.open(io.BytesIO(stream.read())) as image:
                    image_info[(image.size, image.mode)] += 1

        print(f"Sampled image properties ({sample_count} images):")
        for (size, mode), count in image_info.items():
            print(f"  size={size}, mode={mode}: {count}")

        print("\nDataset validation passed.")
        print(
            "Confirm the semantic mapping from numeric labels to "
            "angry/cry/happy with the teammate."
        )


if __name__ == "__main__":
    main()
