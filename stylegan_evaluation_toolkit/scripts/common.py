"""Shared helpers for image evaluation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc

def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

def image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"Image directory not found: {directory}")
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No supported images found in: {directory}")
    return paths

def class_name_from_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) < 2:
        raise ValueError(f"Expected class subdirectories under {root}; found {path}")
    return relative.parts[0]

def select_torch_device(requested: str) -> str:
    import torch
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable.")
    if requested not in {"cuda", "mps", "cpu"}:
        raise SystemExit("Device must be auto, cuda, mps, or cpu.")
    return requested

def normalize_label(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())
