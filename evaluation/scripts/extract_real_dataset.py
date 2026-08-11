#!/usr/bin/env python3
"""Extract a labeled StyleGAN ZIP into class folders."""
from __future__ import annotations
import argparse
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json


def parse_args():
    """Parse command line arguments for dataset extraction."""
    parser = argparse.ArgumentParser(
        description="Extract labeled StyleGAN images into class folders"
    )
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        required=True,
        help="Path to the StyleGAN ZIP file"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for extracted images"
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        required=True,
        help="JSON file mapping label indices to class names"
    )
    parser.add_argument(
        "--resize",
        type=int,
        help="Resize images to this size (width and height)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory"
    )
    return parser.parse_args()


def label_key(label):
    """Convert a label to a string key for class mapping.

    Handles integer, float, and one-hot encoded labels.

    Args:
        label: Label value (int, float, or list of probabilities).

    Returns:
        String representation of the label index.

    Raises:
        ValueError: If label format is unsupported or ambiguous.
    """
    if isinstance(label, (int, float)) and not isinstance(label, bool):
        return str(int(label))
    if isinstance(label, list) and label:
        max_value = max(label)
        indices = [i for i, value in enumerate(label) if value == max_value]
        if len(indices) == 1:
            return str(indices[0])
    raise ValueError(f"Unsupported or ambiguous label: {label!r}")


def main():
    """Extract images from StyleGAN ZIP file organized by class."""
    args = parse_args()

    class_map = load_json(args.class_map)

    if args.outdir.exists() and any(args.outdir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {args.outdir}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    counts = Counter()

    with zipfile.ZipFile(args.zip_path) as archive:
        metadata = json.loads(archive.read("dataset.json"))
        labels = metadata.get("labels")

        if not labels:
            raise SystemExit("dataset.json does not contain labels.")

        for filename, raw_label in tqdm(labels, desc="Extracting real images"):
            key = label_key(raw_label)

            if key not in class_map:
                raise SystemExit(f"Label {key} is missing from {args.class_map}")

            destination_dir = args.outdir / f"class_{key}"
            destination_dir.mkdir(parents=True, exist_ok=True)

            with Image.open(io.BytesIO(archive.read(filename))) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                if args.resize:
                    image = image.resize(
                        (args.resize, args.resize),
                        Image.Resampling.LANCZOS
                    )
                image.save(destination_dir / (Path(filename).stem + ".png"))

            counts[f"class_{key}:{class_map[key]}"] += 1

    summary = {
        "source_zip": str(args.zip_path),
        "output_directory": str(args.outdir),
        "counts": dict(sorted(counts.items()))
    }

    (args.outdir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
