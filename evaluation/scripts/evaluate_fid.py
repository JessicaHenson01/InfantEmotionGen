#!/usr/bin/env python3
"""Calculate overall and optional per-class FID."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import image_paths, save_json


class ImageDataset(Dataset):
    """Dataset for loading images from a list of paths."""

    def __init__(self, paths):
        """Initialize dataset with list of image paths.

        Args:
            paths: List of Path objects pointing to image files.
        """
        self.paths = paths

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.paths)

    def __getitem__(self, index):
        """Load and return image as tensor at the given index.

        Args:
            index: Index of the image to load.

        Returns:
            RGB image as PyTorch tensor with values in [0, 255].
        """
        with Image.open(self.paths[index]) as image:
            return TF.pil_to_tensor(image.convert("RGB"))


def parse_args():
    """Parse command line arguments for FID evaluation."""
    parser = argparse.ArgumentParser(
        description="Calculate Fréchet Inception Distance (FID) between real and generated images"
    )
    parser.add_argument(
        "--real",
        type=Path,
        required=True,
        help="Directory containing real images"
    )
    parser.add_argument(
        "--generated",
        type=Path,
        required=True,
        help="Directory containing generated images"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing images"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading"
    )
    parser.add_argument(
        "--feature",
        type=int,
        default=2048,
        choices=[64, 192, 768, 2048],
        help="Feature dimension of InceptionV3"
    )
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="Calculate FID per class as well as overall"
    )
    return parser.parse_args()


def update_metric(metric, paths, real, args):
    """Update FID metric with images from paths.

    Args:
        metric: FrechetInceptionDistance metric instance.
        paths: List of image paths to process.
        real: Boolean indicating if these are real images.
        args: Command line arguments containing batch_size and workers.
    """
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    for batch in tqdm(loader, desc="Real FID" if real else "Generated FID"):
        metric.update(batch.to("cpu"), real=real)


def calculate_fid(real_paths, generated_paths, args):
    """Calculate FID between real and generated image sets.

    Args:
        real_paths: List of real image paths.
        generated_paths: List of generated image paths.
        args: Command line arguments containing feature dimension.

    Returns:
        FID score as a float.
    """
    metric = FrechetInceptionDistance(
        feature=args.feature,
        normalize=False
    ).to("cpu")
    update_metric(metric, real_paths, True, args)
    update_metric(metric, generated_paths, False, args)
    return float(metric.compute().item())


def get_class_dirs(root):
    """Get class subdirectories from a root directory.

    Args:
        root: Root path containing class subdirectories.

    Returns:
        Dictionary mapping class name to Path object for each class directory.
    """
    return {
        p.name: p
        for p in sorted(root.iterdir())
        if p.is_dir() and p.name.startswith("class_")
    }


def main():
    """Run FID evaluation on real and generated images."""
    args = parse_args()

    real_all = image_paths(args.real)
    generated_all = image_paths(args.generated)

    result = {
        "metric": "FID",
        "feature_dimension": args.feature,
        "real_image_count": len(real_all),
        "generated_image_count": len(generated_all),
        "overall_fid": calculate_fid(real_all, generated_all, args),
        "notes": "Lower is better. Use equal sample counts and identical settings when comparing runs."
    }

    if args.per_class:
        real_classes = get_class_dirs(args.real)
        generated_classes = get_class_dirs(args.generated)
        per_class = {}

        for name in sorted(set(real_classes) & set(generated_classes)):
            real_paths = image_paths(real_classes[name])
            generated_paths = image_paths(generated_classes[name])
            per_class[name] = {
                "real_image_count": len(real_paths),
                "generated_image_count": len(generated_paths),
                "fid": calculate_fid(real_paths, generated_paths, args)
            }
        result["per_class"] = per_class

    save_json(result, args.output)
    print(f"Overall FID: {result['overall_fid']:.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
