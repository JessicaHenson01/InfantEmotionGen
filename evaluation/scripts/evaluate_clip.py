#!/usr/bin/env python3
"""Evaluate generated classes with OpenCLIP image-text similarity."""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import class_name_from_path, image_paths, load_json, save_json, select_torch_device


def parse_args():
    """Parse command line arguments for CLIP evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated images using OpenCLIP similarity"
    )
    parser.add_argument(
        "--generated",
        type=Path,
        required=True,
        help="Directory containing generated images organized by class"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="JSON file mapping class names to prompt templates"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path for results"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cuda, mps, or cpu"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing images"
    )
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        help="OpenCLIP model name"
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b79k",
        help="Pretrained weights to use"
    )
    return parser.parse_args()


def main():
    """Run CLIP evaluation on generated images."""
    args = parse_args()
    device = select_torch_device(args.device)

    config = load_json(args.prompts)
    paths = image_paths(args.generated)
    class_names = sorted(config)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    # Prepare text prompts for each class
    flat_prompts = []
    prompt_owners = []
    for i, name in enumerate(class_names):
        for prompt in config[name]:
            flat_prompts.append(prompt)
            prompt_owners.append(i)

    with torch.inference_mode():
        # Encode all prompts
        prompt_features = model.encode_text(
            tokenizer(flat_prompts).to(device)
        )
        prompt_features = prompt_features / prompt_features.norm(
            dim=-1, keepdim=True
        )

        # Average prompts per class
        class_features = []
        for i in range(len(class_names)):
            indices = [j for j, owner in enumerate(prompt_owners) if owner == i]
            feature = prompt_features[indices].mean(dim=0)
            class_features.append(feature / feature.norm())
        target_features = torch.stack(class_features)

    records = []
    target_scores = defaultdict(list)
    clip_scores = defaultdict(list)
    margins = defaultdict(list)
    confusion = Counter()

    # Process images in batches
    for start in tqdm(
        range(0, len(paths), args.batch_size),
        desc="CLIP evaluation"
    ):
        batch_paths = paths[start:start + args.batch_size]
        tensors = []

        # Load and preprocess images
        for path in batch_paths:
            with Image.open(path) as image:
                tensors.append(preprocess(image.convert("RGB")))

        with torch.inference_mode():
            # Encode images
            image_features = model.encode_image(
                torch.stack(tensors).to(device)
            )
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

            # Compute similarity
            cosine = image_features @ target_features.T
            probs = (100 * cosine).softmax(dim=-1)

        cosine = cosine.float().cpu().numpy()
        probs = probs.float().cpu().numpy()

        # Record results for each image
        for row, path in enumerate(batch_paths):
            target = class_name_from_path(path, args.generated)
            target_idx = class_names.index(target)
            pred_idx = int(np.argmax(probs[row]))
            predicted = class_names[pred_idx]

            target_cosine = float(cosine[row, target_idx])
            score = float(target_cosine * 100)
            clip_score = float(2.5 * max(target_cosine, 0.0))
            ordered = np.sort(cosine[row])
            margin = float((ordered[-1] - ordered[-2]) * 100)

            confusion[(target, predicted)] += 1
            target_scores[target].append(score)
            clip_scores[target].append(clip_score)
            margins[target].append(margin)

            records.append({
                "image": str(path),
                "target_class": target,
                "predicted_class": predicted,
                "correct": target == predicted,
                "target_cosine_x100": score,
                "target_clip_score": clip_score,
                "top1_margin_x100": margin
            })

    # Build results dictionary
    result = {
        "metric": "OpenCLIP emotion alignment",
        "model": args.model,
        "pretrained": args.pretrained,
        "image_count": len(records),
        "zero_shot_accuracy": sum(r["correct"] for r in records) / len(records),
        "mean_target_cosine_x100": float(np.mean(
            [r["target_cosine_x100"] for r in records]
        )),
        "mean_target_clip_score": float(np.mean(
            [r["target_clip_score"] for r in records]
        )),
        "per_class": {},
        "confusion": {
            f"{a}->{b}": n
            for (a, b), n in sorted(confusion.items())
        },
        "records": records
    }

    # Per-class statistics
    for name in class_names:
        subset = [r for r in records if r["target_class"] == name]
        if subset:
            result["per_class"][name] = {
                "count": len(subset),
                "zero_shot_accuracy": sum(r["correct"] for r in subset) / len(subset),
                "mean_target_cosine_x100": float(np.mean(target_scores[name])),
                "mean_target_clip_score": float(np.mean(clip_scores[name])),
                "mean_top1_margin_x100": float(np.mean(margins[name]))
            }

    save_json(result, args.output)
    print(f"CLIP agreement: {result['zero_shot_accuracy']:.4f}")
    print(f"CLIPScore: {result['mean_target_clip_score']:.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
