#!/usr/bin/env python3
"""Calculate overall and optional per-class FID."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as TF
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import image_paths, save_json

class ImageDataset(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return TF.pil_to_tensor(image.convert("RGB"))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real", type=Path, required=True)
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--feature", type=int, default=2048, choices=[64,192,768,2048])
    p.add_argument("--per-class", action="store_true")
    return p.parse_args()

def update(metric, paths, real, args):
    loader = DataLoader(ImageDataset(paths), batch_size=args.batch_size, num_workers=args.workers)
    for batch in tqdm(loader, desc="Real FID" if real else "Generated FID"):
        metric.update(batch.to("cpu"), real=real)

def calculate(real_paths, generated_paths, args):
    metric = FrechetInceptionDistance(feature=args.feature, normalize=False).to("cpu")
    update(metric, real_paths, True, args)
    update(metric, generated_paths, False, args)
    return float(metric.compute().item())

def class_dirs(root):
    return {p.name: p for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith("class_")}

def main():
    args = parse_args()
    real_all, generated_all = image_paths(args.real), image_paths(args.generated)
    result = {"metric":"FID", "feature_dimension":args.feature, "real_image_count":len(real_all), "generated_image_count":len(generated_all), "overall_fid":calculate(real_all, generated_all, args), "notes":"Lower is better. Use equal sample counts and identical settings when comparing runs."}
    if args.per_class:
        real_classes, generated_classes = class_dirs(args.real), class_dirs(args.generated)
        per_class = {}
        for name in sorted(set(real_classes) & set(generated_classes)):
            rp, gp = image_paths(real_classes[name]), image_paths(generated_classes[name])
            per_class[name] = {"real_image_count":len(rp), "generated_image_count":len(gp), "fid":calculate(rp, gp, args)}
        result["per_class"] = per_class
    save_json(result, args.output)
    print(f"Overall FID: {result['overall_fid']:.4f}")
    print(f"Saved: {args.output}")
if __name__ == "__main__": main()
