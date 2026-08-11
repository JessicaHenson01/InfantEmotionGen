#!/usr/bin/env python3
"""Populate deterministic real-reference and pipeline-test evaluation folders."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import snapshot_download
from PIL import Image, ImageOps

CLASS_MAPPING: dict[int, str] = {
    0: "angry",
    1: "crying",
    2: "happy",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


class SplitError(RuntimeError):
    """Raised when the evaluation split cannot be created or validated."""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for populating real evaluation folders."""
    parser = argparse.ArgumentParser(
        description="Create a deterministic stratified smoke-test split from a StyleGAN dataset ZIP."
    )
    parser.add_argument(
        "--repo-id",
        default="InfantEmotionGen/baby_samples_gan",
        help="Hugging Face dataset repository ID"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for output splits"
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Fraction of images to use for test split"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--zip-name",
        help="ZIP filename to use when the Hugging Face dataset snapshot contains multiple ZIP files.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing split using split_manifest.json without downloading or extracting.",
    )
    return parser.parse_args()


def label_key(label: Any) -> int:
    """Convert a label to an integer class key.

    Handles integer, float, and one-hot encoded labels.

    Args:
        label: Label value (int, float, or list of probabilities).

    Returns:
        Integer class ID.

    Raises:
        SplitError: If label format is unsupported or ambiguous.
    """
    if isinstance(label, bool):
        raise SplitError(f"Boolean labels are unsupported: {label!r}")
    if isinstance(label, (int, float)):
        value = int(label)
        if float(label) != float(value):
            raise SplitError(f"Non-integer numeric label is unsupported: {label!r}")
        return value
    if isinstance(label, list) and label:
        maximum = max(label)
        indices = [index for index, value in enumerate(label) if value == maximum]
        if len(indices) == 1:
            return int(indices[0])
        raise SplitError(f"Ambiguous one-hot label: {label!r}")
    raise SplitError(f"Unsupported label format: {label!r}")


def find_zip(snapshot_dir: Path, zip_name: str | None) -> Path:
    """Find the StyleGAN ZIP file in the snapshot directory.

    Args:
        snapshot_dir: Directory containing downloaded dataset snapshot.
        zip_name: Optional specific ZIP filename to use.

    Returns:
        Path to the ZIP file.

    Raises:
        SplitError: If no ZIP found, multiple ZIPs found without name, or requested ZIP not found.
    """
    zip_paths = sorted(path for path in snapshot_dir.rglob("*.zip") if path.is_file())
    if zip_name:
        matches = [path for path in zip_paths if path.name == zip_name or str(path.relative_to(snapshot_dir)) == zip_name]
        if not matches:
            raise SplitError(f"Requested ZIP {zip_name!r} was not found under {snapshot_dir}")
        if len(matches) > 1:
            choices = "\n".join(str(path.relative_to(snapshot_dir)) for path in matches)
            raise SplitError(f"ZIP name {zip_name!r} matched multiple files:\n{choices}")
        return matches[0]
    if not zip_paths:
        raise SplitError(f"No ZIP files found in downloaded dataset snapshot: {snapshot_dir}")
    if len(zip_paths) > 1:
        choices = "\n".join(str(path.relative_to(snapshot_dir)) for path in zip_paths)
        raise SplitError(f"Multiple ZIP files found. Re-run with --zip-name.\n{choices}")
    return zip_paths[0]


def load_stylegan_labels(archive: zipfile.ZipFile) -> list[tuple[str, int]]:
    """Load and validate labels from a StyleGAN dataset.json file.

    Args:
        archive: Open ZIP file containing the dataset.

    Returns:
        List of (filename, class_id) tuples for labeled images.

    Raises:
        SplitError: If dataset.json is missing, invalid, or contains unsupported labels.
    """
    try:
        metadata = json.loads(archive.read("dataset.json"))
    except KeyError as exc:
        raise SplitError("StyleGAN ZIP does not contain dataset.json") from exc
    except json.JSONDecodeError as exc:
        raise SplitError(f"dataset.json is invalid JSON: {exc}") from exc

    labels = metadata.get("labels")
    if not isinstance(labels, list) or not labels:
        raise SplitError("dataset.json does not contain a non-empty labels list")

    archive_names = set(archive.namelist())
    labeled: list[tuple[str, int]] = []
    for item in labels:
        if not isinstance(item, list) or len(item) != 2:
            raise SplitError(f"Unsupported dataset.json label entry: {item!r}")
        filename, raw_label = item
        if not isinstance(filename, str):
            raise SplitError(f"Label entry filename is not a string: {filename!r}")
        if filename not in archive_names:
            raise SplitError(f"Labeled image is missing from ZIP: {filename}")
        key = label_key(raw_label)
        if key not in CLASS_MAPPING:
            raise SplitError(f"Unsupported class label {key}; expected one of {sorted(CLASS_MAPPING)}")
        if Path(filename).suffix.lower() not in IMAGE_SUFFIXES:
            raise SplitError(f"Labeled file does not look like a supported image: {filename}")
        labeled.append((filename, key))
    return labeled


def stratified_split(
    grouped: dict[int, list[str]], test_fraction: float, seed: int
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Create stratified train/test split based on class labels.

    Args:
        grouped: Dictionary mapping class IDs to list of filenames.
        test_fraction: Fraction of data to use for test split.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (reference_split, test_split) dictionaries.

    Raises:
        SplitError: If test_fraction is invalid or a class has too few images.
    """
    if not 0 < test_fraction < 1:
        raise SplitError(f"--test-fraction must be between 0 and 1, got {test_fraction}")

    reference: dict[int, list[str]] = {}
    test: dict[int, list[str]] = {}
    rng = random.Random(seed)

    for class_id, class_name in CLASS_MAPPING.items():
        filenames = sorted(grouped.get(class_id, []))
        if len(filenames) < 2:
            raise SplitError(
                f"Class {class_id} ({class_name}) needs at least 2 labeled images; found {len(filenames)}"
            )
        shuffled = list(filenames)
        rng.shuffle(shuffled)
        test_count = int(round(len(shuffled) * test_fraction))
        test_count = max(1, min(len(shuffled) - 1, test_count))
        test[class_id] = sorted(shuffled[:test_count])
        reference[class_id] = sorted(shuffled[test_count:])
    return reference, test


def clear_directory_contents(directory: Path) -> None:
    """Clear all contents of a directory while preserving the directory itself.

    Args:
        directory: Directory to clear.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def output_filename(source_filename: str, used: set[str]) -> str:
    """Generate a unique output filename based on source filename.

    Args:
        source_filename: Original filename from the ZIP.
        used: Set of already-used filenames to avoid collisions.

    Returns:
        Unique output filename.
    """
    source_path = Path(source_filename)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_path.stem).strip("._") or "image"
    digest = hashlib.sha1(source_filename.encode("utf-8")).hexdigest()[:10]
    candidate = f"{stem}__{digest}.png"
    suffix = 1
    while candidate in used:
        candidate = f"{stem}__{digest}_{suffix}.png"
        suffix += 1
    used.add(candidate)
    return candidate


def extract_split(
    archive: zipfile.ZipFile,
    assignments: dict[int, list[str]],
    split_root: Path,
) -> dict[str, dict[str, str]]:
    """Extract images for a split from the ZIP archive.

    Args:
        archive: Open ZIP file containing images.
        assignments: Dictionary mapping class IDs to filenames to extract.
        split_root: Root directory for the split.

    Returns:
        Dictionary mapping source filenames to output information.
    """
    records: dict[str, dict[str, str]] = {}
    used_names: set[str] = set()
    for class_id, filenames in assignments.items():
        class_name = CLASS_MAPPING[class_id]
        class_dir = split_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for source_filename in filenames:
            destination_name = output_filename(source_filename, used_names)
            destination = class_dir / destination_name
            with Image.open(io.BytesIO(archive.read(source_filename))) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.save(destination, format="PNG")
            records[source_filename] = {
                "class": class_name,
                "output": str(destination.relative_to(split_root.parent)),
            }
    return records


def count_images(directory: Path) -> int:
    """Count the number of image files in a directory recursively.

    Args:
        directory: Directory to count images in.

    Returns:
        Number of image files found.
    """
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_split(output_root: Path, manifest: dict[str, Any]) -> None:
    """Validate an existing split against its manifest.

    Args:
        output_root: Root directory containing the splits.
        manifest: Manifest dictionary to validate against.

    Raises:
        SplitError: If validation fails.
    """
    reference_root = output_root / "real_reference"
    test_root = output_root / "pipeline_test"

    for split_root in (reference_root, test_root):
        if not split_root.is_dir():
            raise SplitError(f"Missing split directory: {split_root}")
        for class_name in CLASS_MAPPING.values():
            class_dir = split_root / class_name
            if not class_dir.is_dir():
                raise SplitError(f"Missing emotion folder: {class_dir}")
            if count_images(class_dir) < 1:
                raise SplitError(f"Emotion folder contains no images: {class_dir}")

    reference_outputs = {
        path.name
        for path in reference_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    test_outputs = {
        path.name
        for path in test_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    overlap = reference_outputs & test_outputs
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise SplitError(f"Output filename overlap between splits: {examples}")

    source_reference = set(manifest["assignments"]["real_reference"])
    source_test = set(manifest["assignments"]["pipeline_test"])
    source_overlap = source_reference & source_test
    if source_overlap:
        examples = ", ".join(sorted(source_overlap)[:5])
        raise SplitError(f"Source filename overlap between splits: {examples}")

    expected_total = int(manifest["source_labeled_image_count"])
    actual_total = count_images(reference_root) + count_images(test_root)
    if actual_total != expected_total:
        raise SplitError(
            f"Extracted image count ({actual_total}) does not match dataset.json labeled count ({expected_total})"
        )


def print_counts(counts: dict[str, dict[str, int]]) -> None:
    """Print counts of images per class for both splits.

    Args:
        counts: Dictionary containing counts for real_reference and pipeline_test splits.
    """
    for class_name in CLASS_MAPPING.values():
        reference_count = counts["real_reference"].get(class_name, 0)
        test_count = counts["pipeline_test"].get(class_name, 0)
        print(f"{class_name}: real_reference={reference_count}, pipeline_test={test_count}")


def load_manifest(output_root: Path) -> dict[str, Any]:
    """Load and parse the split manifest JSON file.

    Args:
        output_root: Root directory containing the manifest.

    Returns:
        Manifest dictionary.

    Raises:
        SplitError: If manifest not found or invalid JSON.
    """
    manifest_path = output_root / "split_manifest.json"
    if not manifest_path.is_file():
        raise SplitError(f"Manifest not found: {manifest_path}")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SplitError(f"Manifest is invalid JSON: {manifest_path}: {exc}") from exc


def build_manifest(
    repo_id: str,
    zip_path: Path,
    seed: int,
    test_fraction: float,
    labeled_count: int,
    reference_assignments: dict[int, list[str]],
    test_assignments: dict[int, list[str]],
    reference_records: dict[str, dict[str, str]],
    test_records: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Build the complete split manifest dictionary.

    Args:
        repo_id: Hugging Face dataset repository ID.
        zip_path: Path to the source ZIP file.
        seed: Random seed used for splitting.
        test_fraction: Fraction used for test split.
        labeled_count: Total number of labeled images.
        reference_assignments: Dictionary of reference split assignments.
        test_assignments: Dictionary of test split assignments.
        reference_records: Records for reference split images.
        test_records: Records for test split images.

    Returns:
        Complete manifest dictionary.
    """
    counts = {
        "real_reference": {
            CLASS_MAPPING[class_id]: len(filenames)
            for class_id, filenames in sorted(reference_assignments.items())
        },
        "pipeline_test": {
            CLASS_MAPPING[class_id]: len(filenames)
            for class_id, filenames in sorted(test_assignments.items())
        },
    }
    return {
        "source_repo": repo_id,
        "source_zip": str(zip_path),
        "source_labeled_image_count": labeled_count,
        "seed": seed,
        "test_fraction": test_fraction,
        "class_mapping": {str(key): value for key, value in CLASS_MAPPING.items()},
        "counts": counts,
        "assignments": {
            "real_reference": {
                source: reference_records[source]
                for source in sorted(reference_records)
            },
            "pipeline_test": {
                source: test_records[source]
                for source in sorted(test_records)
            },
        },
    }


def group_by_class(labeled: Iterable[tuple[str, int]]) -> dict[int, list[str]]:
    """Group labeled images by class ID.

    Args:
        labeled: List of (filename, class_id) tuples.

    Returns:
        Dictionary mapping class IDs to list of filenames.

    Raises:
        SplitError: If any expected class is missing.
    """
    grouped: dict[int, list[str]] = defaultdict(list)
    for filename, class_id in labeled:
        grouped[class_id].append(filename)
    missing = [name for class_id, name in CLASS_MAPPING.items() if not grouped.get(class_id)]
    if missing:
        raise SplitError(f"dataset.json is missing labeled images for classes: {', '.join(missing)}")
    return grouped


def populate(args: argparse.Namespace) -> None:
    """Populate the evaluation splits by downloading and extracting the dataset.

    Args:
        args: Parsed command line arguments.

    Raises:
        SplitError: If any step of the population process fails.
    """
    output_root = args.output_root.resolve()
    snapshot_dir = Path(snapshot_download(repo_id=args.repo_id, repo_type="dataset")).resolve()
    zip_path = find_zip(snapshot_dir, args.zip_name)

    with zipfile.ZipFile(zip_path) as archive:
        labeled = load_stylegan_labels(archive)
        grouped = group_by_class(labeled)
        reference_assignments, test_assignments = stratified_split(
            grouped, args.test_fraction, args.seed
        )

        reference_root = output_root / "real_reference"
        test_root = output_root / "pipeline_test"
        clear_directory_contents(reference_root)
        clear_directory_contents(test_root)

        reference_records = extract_split(archive, reference_assignments, reference_root)
        test_records = extract_split(archive, test_assignments, test_root)

    manifest = build_manifest(
        repo_id=args.repo_id,
        zip_path=zip_path,
        seed=args.seed,
        test_fraction=args.test_fraction,
        labeled_count=len(labeled),
        reference_assignments=reference_assignments,
        test_assignments=test_assignments,
        reference_records=reference_records,
        test_records=test_records,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    validate_split(output_root, manifest)
    print_counts(manifest["counts"])
    print(f"Saved manifest: {manifest_path}")
    print("Validation passed.")


def main() -> int:
    """Main entry point for the script.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    try:
        if args.validate_only:
            manifest = load_manifest(args.output_root.resolve())
            validate_split(args.output_root.resolve(), manifest)
            print_counts(manifest["counts"])
            print("Validation passed.")
        else:
            populate(args)
    except SplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
