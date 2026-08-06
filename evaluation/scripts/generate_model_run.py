#!/usr/bin/env python3
"""Resolve a named model run into the fixed evaluation-generation command."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images for a named evaluation model run.")
    parser.add_argument("run_name", help="Run name from evaluation/configs/model_runs.json.")
    parser.add_argument("--config", type=Path, default=Path("evaluation/configs/model_runs.json"))
    parser.add_argument("--num-images", type=int, help="Override protocol image count per class.")
    parser.add_argument("--height", type=int, help="Override protocol height.")
    parser.add_argument("--width", type=int, help="Override protocol width.")
    parser.add_argument("--num-inference-steps", type=int, help="Override protocol inference steps.")
    parser.add_argument("--guidance-scale", type=float, help="Override protocol guidance scale.")
    parser.add_argument("--seed", type=int, help="Override protocol seed.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--disable-xet", action="store_true")
    parser.add_argument("--hf-transfer-workers", type=int, default=1)
    return parser.parse_args()


def load_runs(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Model run config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in model run config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Model run config must be a JSON object: {path}")
    return data


def append_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    runs = load_runs(repo_root / args.config)
    if args.run_name not in runs:
        choices = ", ".join(sorted(runs))
        raise SystemExit(f"Unknown run {args.run_name!r}. Available runs: {choices}")
    run = runs[args.run_name]

    script = repo_root / "evaluation/scripts/generate_sdxl_lora_eval.py"
    command = [
        sys.executable,
        str(script),
        "--run-name",
        args.run_name,
        "--pipeline-type",
        str(run["pipeline_type"]),
        "--protocol",
        str(repo_root / run["protocol"]),
        "--output-root",
        str(repo_root / "evaluation/generated"),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]

    if run["pipeline_type"] in {"sdxl_lora", "sd3_lora"}:
        if run.get("base_model"):
            command.extend(["--base-model", str(run["base_model"])])
        command.extend(["--lora-repo", str(run["lora_repo"])])
        if run.get("lora_subfolder"):
            command.extend(["--lora-subfolder", str(run["lora_subfolder"])])
        if run.get("adapter_format"):
            command.extend(["--adapter-format", str(run["adapter_format"])])
    elif run["pipeline_type"] == "sd3_pipeline":
        command.extend(["--base-model", str(run["base_model"])])
    else:
        raise SystemExit(f"Unsupported pipeline_type for {args.run_name}: {run['pipeline_type']}")

    append_optional(command, "--num-images", args.num_images)
    append_optional(command, "--height", args.height)
    append_optional(command, "--width", args.width)
    append_optional(command, "--num-inference-steps", args.num_inference_steps)
    append_optional(command, "--guidance-scale", args.guidance_scale)
    append_optional(command, "--seed", args.seed)
    if args.overwrite:
        command.append("--overwrite")
    if args.local_files_only:
        command.append("--local-files-only")
    if args.disable_xet:
        command.append("--disable-xet")
    append_optional(command, "--hf-transfer-workers", args.hf_transfer_workers)

    print("Running:")
    print(" ".join(command))
    return subprocess.call(command, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
