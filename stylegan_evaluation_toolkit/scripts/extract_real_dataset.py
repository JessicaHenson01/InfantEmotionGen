#!/usr/bin/env python3
"""Extract a labeled StyleGAN ZIP into class folders."""
from __future__ import annotations
import argparse, io, json, sys, zipfile
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", dest="zip_path", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--class-map", type=Path, required=True)
    p.add_argument("--resize", type=int)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def label_key(label):
    if isinstance(label, (int, float)) and not isinstance(label, bool):
        return str(int(label))
    if isinstance(label, list) and label:
        maximum = max(label)
        indices = [i for i, value in enumerate(label) if value == maximum]
        if len(indices) == 1:
            return str(indices[0])
    raise ValueError(f"Unsupported or ambiguous label: {label!r}")

def main():
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
                    image = image.resize((args.resize, args.resize), Image.Resampling.LANCZOS)
                image.save(destination_dir / (Path(filename).stem + ".png"))
            counts[f"class_{key}:{class_map[key]}"] += 1
    summary = {"source_zip": str(args.zip_path), "output_directory": str(args.outdir), "counts": dict(sorted(counts.items()))}
    (args.outdir / "extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
