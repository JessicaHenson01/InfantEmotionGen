#!/usr/bin/env python3
"""Count cached Hugging Face Diffusers/LoRA parameters from safetensor shapes."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from safetensors import safe_open


DEFAULT_CACHE_ROOTS = [
    Path("evaluation/.cache/huggingface/hub"),
    Path.home() / ".cache/huggingface/hub",
]


@dataclass(frozen=True)
class RepoSpec:
    """Specification for a model repository to count parameters from."""
    name: str
    cache_dir_name: str
    kind: str


REPOS = [
    RepoSpec(
        name="sdxl_base",
        cache_dir_name="models--stabilityai--stable-diffusion-xl-base-1.0",
        kind="base",
    ),
    RepoSpec(
        name="sdxl_primary_lora",
        cache_dir_name="models--InfantEmotionGen--SDXLPrimary",
        kind="adapter",
    ),
    RepoSpec(
        name="sd35_base",
        cache_dir_name="models--stabilityai--stable-diffusion-3.5-medium",
        kind="base",
    ),
    RepoSpec(
        name="sd35_medium_lora",
        cache_dir_name="models--InfantEmotionGen--stable-diffusion-3.5-medium",
        kind="adapter",
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for counting model parameters.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Count cached model parameters from safetensors files.")
    parser.add_argument(
        "--cache-root",
        type=Path,
        action="append",
        help="Hugging Face hub cache root. Can be repeated. Defaults to repo-local and user cache.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to write machine-readable counts."
    )
    return parser.parse_args()


def snapshot_dirs(repo_cache_dir: Path) -> list[Path]:
    """Get all snapshot directories in a repository cache.

    Args:
        repo_cache_dir: Path to the repository cache directory.

    Returns:
        List of snapshot directory paths, sorted by name.
    """
    snapshots = repo_cache_dir / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(path for path in snapshots.iterdir() if path.is_dir())


def latest_snapshot(repo_cache_dir: Path) -> Path | None:
    """Find the most recently modified snapshot in a repository cache.

    Args:
        repo_cache_dir: Path to the repository cache directory.

    Returns:
        Path to the latest snapshot, or None if no snapshots exist.
    """
    snapshots = snapshot_dirs(repo_cache_dir)
    if not snapshots:
        return None
    return max(snapshots, key=lambda path: path.stat().st_mtime)


def count_safetensors(path: Path) -> int:
    """Count the total number of parameters in a safetensors file.

    Args:
        path: Path to the safetensors file.

    Returns:
        Total number of parameters (sum of all tensor sizes).
    """
    total = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shape = handle.get_slice(key).get_shape()
            size = 1
            for dimension in shape:
                size *= int(dimension)
            total += size
    return total


def component_name(snapshot: Path, tensor_file: Path) -> str:
    """Extract the component name from a tensor file path relative to snapshot.

    Args:
        snapshot: Path to the snapshot directory.
        tensor_file: Path to the tensor file.

    Returns:
        Component name (directory name or 'root' if file is at root level).
    """
    relative = tensor_file.relative_to(snapshot)
    if len(relative.parts) == 1:
        return "root"
    return relative.parts[0]


def find_repo_cache(spec: RepoSpec, cache_roots: Iterable[Path]) -> Path | None:
    """Find a repository cache directory across multiple cache roots.

    Args:
        spec: Repository specification.
        cache_roots: Iterable of cache root directories to search.

    Returns:
        Path to the repository cache, or None if not found.
    """
    for cache_root in cache_roots:
        candidate = cache_root.expanduser().resolve() / spec.cache_dir_name
        if candidate.is_dir():
            return candidate
    return None


def count_repo(spec: RepoSpec, cache_roots: Iterable[Path]) -> dict[str, object]:
    """Count parameters for a single repository.

    Args:
        spec: Repository specification.
        cache_roots: Iterable of cache root directories to search.

    Returns:
        Dictionary containing count results with status, total parameters, and components.
    """
    repo_cache = find_repo_cache(spec, cache_roots)
    if repo_cache is None:
        return {
            "name": spec.name,
            "kind": spec.kind,
            "status": "missing_cache",
            "total_parameters": 0,
            "components": {},
            "snapshot": None,
        }

    snapshot = latest_snapshot(repo_cache)
    if snapshot is None:
        return {
            "name": spec.name,
            "kind": spec.kind,
            "status": "missing_snapshot",
            "total_parameters": 0,
            "components": {},
            "snapshot": None,
        }

    components: dict[str, int] = {}
    for tensor_file in sorted(snapshot.rglob("*.safetensors")):
        component = component_name(snapshot, tensor_file)
        components[component] = components.get(component, 0) + count_safetensors(tensor_file)

    return {
        "name": spec.name,
        "kind": spec.kind,
        "status": "ok",
        "total_parameters": sum(components.values()),
        "components": dict(sorted(components.items())),
        "snapshot": str(snapshot),
    }


def format_count(count: int) -> str:
    """Format a parameter count in billions with 3 decimal places.

    Args:
        count: Number of parameters.

    Returns:
        Formatted string with B suffix (e.g., "2.500B").
    """
    return f"{count / 1_000_000_000:.3f}B"


def print_report(results: list[dict[str, object]]) -> None:
    """Print a human-readable report of parameter counts.

    Args:
        results: List of result dictionaries from count_repo calls.
    """
    for result in results:
        print(f"{result['name']} ({result['kind']})")
        print(f"  status: {result['status']}")
        print(f"  total:  {format_count(int(result['total_parameters']))}")
        components = result["components"]
        if isinstance(components, dict):
            for component, count in components.items():
                print(f"  {component}: {format_count(int(count))}")
        print()


def main() -> int:
    """Main entry point for the script.

    Returns:
        Exit code (0 for success).
    """
    args = parse_args()
    cache_roots = args.cache_root or DEFAULT_CACHE_ROOTS
    results = [count_repo(spec, cache_roots) for spec in REPOS]
    print_report(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
