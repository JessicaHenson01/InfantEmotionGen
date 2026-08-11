"""Shared helpers for image evaluation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def load_json(path: Path) -> Any:
    """Load and parse a JSON file from the given path.
    
    Args:
        path: Path to the JSON file.
    
    Returns:
        Parsed JSON data as a Python object.
    
    Raises:
        SystemExit: If file is not found or JSON is invalid.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc

def save_json(data: Any, path: Path) -> None:
    """Save Python data to a JSON file with pretty formatting.
    
    Args:
        data: Data to serialize to JSON.
        path: Path where the JSON file will be saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

def image_paths(directory: Path) -> list[Path]:
    """Get all supported image file paths recursively from a directory.
    
    Args:
        directory: Root directory to search for images.
    
    Returns:
        Sorted list of paths to image files.
    
    Raises:
        SystemExit: If directory doesn't exist or no images are found.
    """
    if not directory.is_dir():
        raise SystemExit(f"Image directory not found: {directory}")
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No supported images found in: {directory}")
    return paths

def class_name_from_path(path: Path, root: Path) -> str:
    """Extract the class/subfolder name from a file path.
    
    Args:
        path: Full path to the file.
        root: Root directory containing class subdirectories.
    
    Returns:
        Name of the immediate parent directory (class name).
    
    Raises:
        ValueError: If path doesn't have a parent directory under root.
    """
    relative = path.relative_to(root)
    if len(relative.parts) < 2:
        raise ValueError(f"Expected class subdirectories under {root}; found {path}")
    return relative.parts[0]

def select_torch_device(requested: str) -> str:
    """Select the appropriate PyTorch device based on user preference.
    
    Args:
        requested: Device preference ('auto', 'cuda', 'mps', or 'cpu').
    
    Returns:
        Selected device string ('cuda', 'mps', or 'cpu').
    
    Raises:
        SystemExit: If requested device is unavailable or invalid.
    """
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
    """Normalize a label string by lowercasing and normalizing spaces.
    
    Replaces underscores and hyphens with spaces, then collapses multiple spaces.
    
    Args:
        value: Raw label string to normalize.
    
    Returns:
        Normalized label string.
    """
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())
