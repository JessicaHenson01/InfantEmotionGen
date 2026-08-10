#!/usr/bin/env python3
"""Populate final external-reference folders from a Hugging Face image dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import snapshot_download
from PIL import Image, ImageOps

CLASS_MAPPING: dict[int, str] = {
    0: "angry",
    1: "crying",
    2: "happy",
}
EMOTIONS = {"angry", "crying", "happy"}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
METADATA_NAMES = {"metadata.jsonl", "metadata.csv"}


class ExternalReferenceError(RuntimeError):
    """Raised when the external reference dataset cannot be imported."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a Hugging Face image dataset into evaluation/data/external_reference/<emotion>/."
    )
    parser.add_argument("--repo-id", default="InfantEmotionGen/InfantEmotionGen_Dataset")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zip-name", help="ZIP filename to use if the dataset snapshot contains ZIP files.")
    parser.add_argument(
        "--zip-label-json",
        default="final_test_samples/test_samples.json",
        help="Label JSON inside the ZIP. Defaults to the held-out final_test_samples labels.",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        raise ExternalReferenceError(f"Boolean labels are unsupported: {value!r}")
    if isinstance(value, (int, float)):
        class_id = int(value)
        if float(value) != float(class_id):
            raise ExternalReferenceError(f"Non-integer numeric label is unsupported: {value!r}")
        if class_id not in CLASS_MAPPING:
            raise ExternalReferenceError(f"Unsupported class label {class_id}; expected {sorted(CLASS_MAPPING)}")
        return CLASS_MAPPING[class_id]
    if isinstance(value, list) and value:
        maximum = max(value)
        indices = [index for index, item in enumerate(value) if item == maximum]
        if len(indices) != 1:
            raise ExternalReferenceError(f"Ambiguous one-hot label: {value!r}")
        return normalize_label(indices[0])
    if isinstance(value, str):
        normalized = " ".join(value.lower().replace("_", " ").replace("-", " ").split())
        aliases = {
            "angry": "angry",
            "anger": "angry",
            "mad": "angry",
            "cry": "crying",
            "crying": "crying",
            "sad crying": "crying",
            "happy": "happy",
            "smile": "happy",
            "smiling": "happy",
        }
        if normalized in aliases:
            return aliases[normalized]
        if normalized.isdigit():
            return normalize_label(int(normalized))
    raise ExternalReferenceError(f"Unsupported or unmapped label: {value!r}")


def clear_directory_contents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def output_filename(source_filename: str, used: set[str]) -> str:
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


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def class_from_path(path: Path, snapshot_dir: Path) -> str | None:
    for part in reversed(path.relative_to(snapshot_dir).parts[:-1]):
        try:
            return normalize_label(part)
        except ExternalReferenceError:
            continue
    return None


def metadata_records(snapshot_dir: Path) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    for metadata_path in sorted(path for path in snapshot_dir.rglob("*") if path.name in METADATA_NAMES):
        if metadata_path.name == "metadata.jsonl":
            rows = []
            with metadata_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            raise ExternalReferenceError(f"Invalid JSONL at {metadata_path}:{line_number}: {exc}") from exc
        else:
            with metadata_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        base = metadata_path.parent
        for row in rows:
            if not isinstance(row, dict):
                continue
            filename = row.get("file_name") or row.get("filename") or row.get("image") or row.get("path")
            raw_label = row.get("label", row.get("class", row.get("emotion")))
            if filename is None or raw_label is None:
                continue
            path = (base / str(filename)).resolve()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                records.append((path, normalize_label(raw_label)))
    return records


def directory_records(snapshot_dir: Path) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    for path in image_paths(snapshot_dir):
        class_name = class_from_path(path, snapshot_dir)
        if class_name:
            records.append((path, class_name))
    return records


def find_zip(snapshot_dir: Path, zip_name: str | None) -> Path | None:
    zip_paths = sorted(path for path in snapshot_dir.rglob("*.zip") if path.is_file())
    if zip_name:
        matches = [path for path in zip_paths if path.name == zip_name or str(path.relative_to(snapshot_dir)) == zip_name]
        if not matches:
            raise ExternalReferenceError(f"Requested ZIP {zip_name!r} was not found under {snapshot_dir}")
        if len(matches) > 1:
            choices = "\n".join(str(path.relative_to(snapshot_dir)) for path in matches)
            raise ExternalReferenceError(f"ZIP name {zip_name!r} matched multiple files:\n{choices}")
        return matches[0]
    if len(zip_paths) > 1:
        choices = "\n".join(str(path.relative_to(snapshot_dir)) for path in zip_paths)
        raise ExternalReferenceError(f"Multiple ZIP files found. Re-run with --zip-name.\n{choices}")
    return zip_paths[0] if zip_paths else None


def label_key(label: Any) -> int:
    emotion = normalize_label(label)
    reverse = {value: key for key, value in CLASS_MAPPING.items()}
    return reverse[emotion]


def zip_records(archive: zipfile.ZipFile, label_json: str) -> list[tuple[str, str]]:
    metadata_name = label_json.strip("/")
    if metadata_name not in archive.namelist():
        if "dataset.json" in archive.namelist():
            metadata_name = "dataset.json"
        else:
            return []
    try:
        metadata = json.loads(archive.read(metadata_name))
    except json.JSONDecodeError as exc:
        raise ExternalReferenceError(f"{metadata_name} inside ZIP is invalid JSON: {exc}") from exc

    labels = metadata.get("labels")
    if not isinstance(labels, list):
        return []

    archive_names = set(archive.namelist())
    metadata_parent = str(Path(metadata_name).parent)
    records: list[tuple[str, str]] = []
    for item in labels:
        if not isinstance(item, list) or len(item) != 2:
            raise ExternalReferenceError(f"Unsupported dataset.json label entry: {item!r}")
        filename, raw_label = item
        if not isinstance(filename, str):
            raise ExternalReferenceError(f"Labeled image filename is invalid: {filename!r}")
        candidates = [filename]
        if metadata_parent and metadata_parent != ".":
            candidates.append(str(Path(metadata_parent) / filename))
        source_name = next((candidate for candidate in candidates if candidate in archive_names), None)
        if source_name is None:
            raise ExternalReferenceError(f"Labeled image is missing from ZIP: {filename!r}")
        if Path(source_name).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        records.append((source_name, CLASS_MAPPING[label_key(raw_label)]))
    return records


def validate_records(records: Iterable[tuple[Any, str]]) -> dict[str, int]:
    counts: dict[str, int] = {emotion: 0 for emotion in sorted(EMOTIONS)}
    for _, class_name in records:
        if class_name not in EMOTIONS:
            raise ExternalReferenceError(f"Unsupported class {class_name!r}; expected {sorted(EMOTIONS)}")
        counts[class_name] += 1
    missing = [emotion for emotion, count in counts.items() if count < 1]
    if missing:
        raise ExternalReferenceError(f"Missing images for class folder(s): {', '.join(missing)}")
    return counts


def copy_path_records(records: list[tuple[Path, str]], output_root: Path) -> dict[str, Any]:
    used: set[str] = set()
    manifest_records: list[dict[str, str]] = []
    external_root = output_root / "external_reference"
    clear_directory_contents(external_root)
    for emotion in sorted(EMOTIONS):
        (external_root / emotion).mkdir(parents=True, exist_ok=True)

    for source_path, class_name in sorted(records, key=lambda item: str(item[0])):
        destination = external_root / class_name / output_filename(str(source_path), used)
        with Image.open(source_path) as image:
            ImageOps.exif_transpose(image).convert("RGB").save(destination, format="PNG")
        manifest_records.append(
            {
                "source": str(source_path),
                "class": class_name,
                "output": str(destination.relative_to(output_root)),
            }
        )
    return {"records": manifest_records, "counts": validate_records([(r["source"], r["class"]) for r in manifest_records])}


def copy_zip_records(archive_path: Path, output_root: Path, label_json: str) -> dict[str, Any]:
    used: set[str] = set()
    manifest_records: list[dict[str, str]] = []
    external_root = output_root / "external_reference"
    clear_directory_contents(external_root)
    for emotion in sorted(EMOTIONS):
        (external_root / emotion).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        records = zip_records(archive, label_json)
        if not records:
            raise ExternalReferenceError(f"ZIP does not contain supported labels at {label_json!r}: {archive_path}")
        validate_records(records)
        for source_name, class_name in sorted(records):
            destination = external_root / class_name / output_filename(source_name, used)
            with Image.open(io.BytesIO(archive.read(source_name))) as image:
                ImageOps.exif_transpose(image).convert("RGB").save(destination, format="PNG")
            manifest_records.append(
                {
                    "source": source_name,
                    "class": class_name,
                    "output": str(destination.relative_to(output_root)),
                }
            )
    return {"records": manifest_records, "counts": validate_records([(r["source"], r["class"]) for r in manifest_records])}


def validate_existing(output_root: Path) -> None:
    external_root = output_root / "external_reference"
    if not external_root.is_dir():
        raise ExternalReferenceError(f"Missing external reference directory: {external_root}")
    records = []
    for emotion in sorted(EMOTIONS):
        class_dir = external_root / emotion
        if not class_dir.is_dir():
            raise ExternalReferenceError(f"Missing emotion folder: {class_dir}")
        paths = image_paths(class_dir)
        if not paths:
            raise ExternalReferenceError(f"Emotion folder contains no images: {class_dir}")
        records.extend((path, emotion) for path in paths)

    name_counts = Counter(path.name for path, _ in records)
    duplicates = [name for name, count in name_counts.items() if count > 1]
    if duplicates:
        examples = ", ".join(sorted(duplicates)[:5])
        raise ExternalReferenceError(f"Duplicate output filenames across external reference folders: {examples}")

    counts = validate_records(records)
    print_counts(counts)
    print(f"Validated external reference directory: {external_root}")


def print_counts(counts: dict[str, int]) -> None:
    for emotion in sorted(EMOTIONS):
        print(f"{emotion}: external_reference={counts.get(emotion, 0)}")


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.validate_only:
        try:
            validate_existing(output_root)
            return 0
        except ExternalReferenceError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    try:
        snapshot_dir = Path(snapshot_download(repo_id=args.repo_id, repo_type="dataset"))
        zip_path = find_zip(snapshot_dir, args.zip_name)
        if zip_path:
            copied = copy_zip_records(zip_path, output_root, args.zip_label_json)
            source_format = "labeled_zip"
            source_detail = str(zip_path)
        else:
            records = metadata_records(snapshot_dir)
            source_format = "metadata"
            if not records:
                records = directory_records(snapshot_dir)
                source_format = "class_directories"
            if not records:
                raise ExternalReferenceError(
                    "Could not infer labels. Expected class folders named angry/crying/happy, "
                    "metadata.jsonl/metadata.csv with file_name + label/class/emotion, or a StyleGAN ZIP."
                )
            validate_records(records)
            copied = copy_path_records(records, output_root)
            source_detail = str(snapshot_dir)

        manifest = {
            "source_repo": args.repo_id,
            "source_format": source_format,
            "source_path": source_detail,
            "class_mapping": {str(key): value for key, value in CLASS_MAPPING.items()},
            "counts": copied["counts"],
            "records": copied["records"],
        }
        manifest_path = output_root / "external_reference_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print_counts(copied["counts"])
        print(f"Saved manifest: {manifest_path}")
        print(f"External reference images saved under: {output_root / 'external_reference'}")
    except ExternalReferenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
