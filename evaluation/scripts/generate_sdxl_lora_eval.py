#!/usr/bin/env python3
"""Generate fixed-protocol Diffusers evaluation images from Hugging Face models."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image
from peft import PeftModel


DEFAULT_PROTOCOL = Path("evaluation/configs/generation_protocol.sdxl.json")


class GenerationError(RuntimeError):
    """Raised when generation setup is invalid."""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for image generation."""
    parser = argparse.ArgumentParser(
        description="Generate comparable Diffusers image folders for FID/CLIP/FER evaluation."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Output/result run name, e.g. sdxl_primary."
    )
    parser.add_argument(
        "--pipeline-type",
        choices=["sdxl_lora", "sd3_pipeline", "sd3_lora"],
        default="sdxl_lora",
        help="sdxl_lora loads SDXL base plus adapter. sd3_lora loads SD3/SD3.5 base plus adapter. "
             "sd3_pipeline loads a full SD3/SD3.5 Diffusers pipeline.",
    )
    parser.add_argument(
        "--lora-repo",
        help="Hugging Face repo ID or local adapter directory for --pipeline-type sdxl_lora."
    )
    parser.add_argument(
        "--lora-subfolder",
        help="Optional subfolder inside the LoRA repo, e.g. unet_lora_final."
    )
    parser.add_argument(
        "--adapter-format",
        choices=["auto", "peft_unet", "peft_transformer", "diffusers_lora"],
        default="auto",
        help="Use peft_unet or peft_transformer for adapter_config.json + adapter_model.safetensors folders.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
        help="Path to generation protocol JSON file"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation/generated"),
        help="Root directory for generated images"
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("evaluation/.cache"),
        help="Cache directory for models and downloads"
    )
    parser.add_argument(
        "--base-model",
        help="Override protocol base_model."
    )
    parser.add_argument(
        "--vae-model",
        help="Override protocol vae_model. Use empty string to disable fixed VAE."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        help="Override protocol num_images_per_class."
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Override protocol height."
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Override protocol width."
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        help="Override protocol num_inference_steps."
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help="Override protocol guidance_scale."
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override protocol seed."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="Data type to use for model"
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use local files (don't download from Hugging Face)"
    )
    parser.add_argument(
        "--disable-xet",
        action="store_true",
        help="Disable Hugging Face Xet downloads. Useful when large-model reconstruction fails.",
    )
    parser.add_argument(
        "--hf-transfer-workers",
        type=int,
        default=1,
        help="Limit Hugging Face parallel file download workers for large models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear the run output directory before generation."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume a run by leaving existing expected PNG files in place and only generating missing files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load and parse JSON from a file path.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data as a dictionary.

    Raises:
        GenerationError: If file not found, invalid JSON, or not a dictionary.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GenerationError(f"Protocol file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GenerationError(f"Protocol file is invalid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GenerationError(f"Protocol file must contain a JSON object: {path}")
    return data


def select_device(requested: str) -> str:
    """Select the appropriate PyTorch device based on user preference.

    Args:
        requested: Device preference ('auto', 'cuda', 'mps', or 'cpu').

    Returns:
        Selected device string ('cuda', 'mps', or 'cpu').

    Raises:
        GenerationError: If requested device is unavailable.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise GenerationError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise GenerationError("MPS was requested but is not available.")
    return requested


def select_dtype(requested: str, device: str) -> torch.dtype:
    """Select the appropriate PyTorch data type.

    Args:
        requested: Data type preference ('auto', 'float16', 'float32', or 'bfloat16').
        device: Device being used (affects auto selection).

    Returns:
        Selected PyTorch dtype.
    """
    if requested == "float16":
        return torch.float16
    if requested == "float32":
        return torch.float32
    if requested == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def protocol_value(protocol: dict[str, Any], key: str, override: Any) -> Any:
    """Get a protocol value, using override if provided.

    Args:
        protocol: Protocol dictionary.
        key: Key to look up in protocol.
        override: Override value (if not None).

    Returns:
        Protocol value or override.
    """
    return protocol[key] if override is None else override


def resolve_protocol(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the generation protocol with command line overrides.

    Args:
        args: Parsed command line arguments.

    Returns:
        Resolved protocol dictionary.

    Raises:
        GenerationError: If protocol is invalid or missing required fields.
    """
    protocol = load_json(args.protocol)
    emotions = protocol.get("emotions")
    if not isinstance(emotions, dict) or not emotions:
        raise GenerationError("Protocol must define a non-empty emotions object.")
    for emotion, config in emotions.items():
        if not isinstance(config, dict) or not config.get("prompt"):
            raise GenerationError(f"Emotion {emotion!r} must define a prompt.")

    resolved = {
        "base_model": args.base_model or protocol.get("base_model"),
        "vae_model": protocol.get("vae_model") if args.vae_model is None else args.vae_model,
        "num_images_per_class": protocol_value(protocol, "num_images_per_class", args.num_images),
        "height": protocol_value(protocol, "height", args.height),
        "width": protocol_value(protocol, "width", args.width),
        "num_inference_steps": protocol_value(protocol, "num_inference_steps", args.num_inference_steps),
        "guidance_scale": protocol_value(protocol, "guidance_scale", args.guidance_scale),
        "seed": protocol_value(protocol, "seed", args.seed),
        "negative_prompt": protocol.get("negative_prompt", ""),
        "emotions": emotions,
    }
    if not resolved["base_model"]:
        raise GenerationError("A base_model must be provided in the protocol or via --base-model.")
    if int(resolved["num_images_per_class"]) < 1:
        raise GenerationError("num_images_per_class must be at least 1.")
    return resolved


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments for consistency.

    Args:
        args: Parsed command line arguments.

    Raises:
        GenerationError: If arguments are invalid or inconsistent.
    """
    if args.pipeline_type in {"sdxl_lora", "sd3_lora"} and not args.lora_repo:
        raise GenerationError(f"--lora-repo is required for --pipeline-type {args.pipeline_type}.")
    if args.pipeline_type == "sd3_pipeline" and args.lora_repo:
        raise GenerationError("--lora-repo is not used with --pipeline-type sd3_pipeline; use --base-model.")


def looks_like_peft_unet_adapter(repo_or_path: str, subfolder: str | None) -> bool:
    """Check if a path looks like a PEFT UNET adapter directory.

    Args:
        repo_or_path: Path or repo ID to check.
        subfolder: Optional subfolder within the path.

    Returns:
        True if the path contains adapter_config.json, False otherwise.
    """
    path = Path(repo_or_path)
    if subfolder:
        path = path / subfolder
    return path.is_dir() and (path / "adapter_config.json").is_file()


def load_adapter(
    pipe: StableDiffusionXLPipeline | StableDiffusion3Pipeline,
    repo_or_path: str,
    subfolder: str | None,
    adapter_format: str,
    pipeline_type: str,
    local_files_only: bool,
) -> str:
    """Load a LoRA or PEFT adapter into the pipeline.

    Args:
        pipe: Diffusers pipeline to load adapter into.
        repo_or_path: Hugging Face repo ID or local path.
        subfolder: Optional subfolder within the repo/path.
        adapter_format: Format of the adapter ('auto', 'peft_unet', 'peft_transformer', 'diffusers_lora').
        pipeline_type: Type of pipeline ('sdxl_lora' or 'sd3_lora').
        local_files_only: Whether to only use local files.

    Returns:
        String indicating the loaded adapter format.

    Raises:
        GenerationError: If adapter loading fails or format is incompatible.
    """
    if adapter_format == "auto":
        if pipeline_type == "sd3_lora":
            adapter_format = "peft_transformer"
        else:
            adapter_format = "peft_unet" if looks_like_peft_unet_adapter(repo_or_path, subfolder) else "diffusers_lora"

    if adapter_format == "peft_unet":
        pipe.unet = PeftModel.from_pretrained(
            pipe.unet,
            repo_or_path,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )
        pipe.unet.eval()
        return "peft_unet"

    if adapter_format == "peft_transformer":
        if not hasattr(pipe, "transformer"):
            raise GenerationError("Selected peft_transformer, but the pipeline has no transformer component.")
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            repo_or_path,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )
        pipe.transformer.eval()
        return "peft_transformer"

    pipe.load_lora_weights(
        repo_or_path,
        subfolder=subfolder,
        local_files_only=local_files_only,
    )
    return "diffusers_lora"


def prepare_output(run_dir: Path, emotions: list[str], overwrite: bool) -> None:
    """Prepare the output directory structure for generated images.

    Args:
        run_dir: Directory for this run's outputs.
        emotions: List of emotion class names.
        overwrite: Whether to overwrite existing directory.
    """
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for emotion in emotions:
        emotion_dir = run_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)


def image_matches_size(path: Path, width: int, height: int) -> bool:
    """Check if an image file matches the expected width and height.

    Args:
        path: Path to the image file.
        width: Expected width in pixels.
        height: Expected height in pixels.

    Returns:
        True if the image dimensions match, False otherwise.
    """
    try:
        with Image.open(path) as image:
            return image.size == (width, height)
    except OSError:
        return False


def build_pipeline(
    protocol: dict[str, Any],
    pipeline_type: str,
    device: str,
    dtype: torch.dtype,
    local_files_only: bool,
) -> StableDiffusionXLPipeline | StableDiffusion3Pipeline:
    """Build the Diffusers pipeline for image generation.

    Args:
        protocol: Resolved protocol dictionary.
        pipeline_type: Type of pipeline to build.
        device: Device to load the model on.
        dtype: Data type for model weights.
        local_files_only: Whether to only use local files.

    Returns:
        Configured Diffusers pipeline.

    Raises:
        GenerationError: If pipeline building fails.
    """
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
    }
    vae_model = protocol.get("vae_model")
    if vae_model and pipeline_type == "sdxl_lora":
        kwargs["vae"] = AutoencoderKL.from_pretrained(
            vae_model,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )

    if pipeline_type in {"sd3_pipeline", "sd3_lora"}:
        pipe = StableDiffusion3Pipeline.from_pretrained(protocol["base_model"], **kwargs)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(protocol["base_model"], **kwargs)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True
        )

    pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)
    return pipe


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Save the generation manifest to a JSON file.

    Args:
        path: Path where the manifest should be saved.
        manifest: Manifest dictionary to save.
    """
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    """Run the main image generation process.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    cache_root = args.cache_root.resolve()
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_MAX_WORKERS", str(args.hf_transfer_workers))
    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    for key in ("HF_HOME", "TORCH_HOME", "MPLCONFIGDIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    try:
        validate_args(args)
        protocol = resolve_protocol(args)
        device = select_device(args.device)
        dtype = select_dtype(args.dtype, device)
        run_dir = (args.output_root / args.run_name).resolve()
        emotions = sorted(protocol["emotions"])
        prepare_output(run_dir, emotions, args.overwrite)

        print(f"Loading base model: {protocol['base_model']}")
        pipe = build_pipeline(protocol, args.pipeline_type, device, dtype, args.local_files_only)
        loaded_format = None
        if args.pipeline_type in {"sdxl_lora", "sd3_lora"}:
            print(f"Loading LoRA adapter: {args.lora_repo}")
            loaded_format = load_adapter(
                pipe=pipe,
                repo_or_path=args.lora_repo,
                subfolder=args.lora_subfolder,
                adapter_format=args.adapter_format,
                pipeline_type=args.pipeline_type,
                local_files_only=args.local_files_only,
            )
            pipe.to(device)

        manifest: dict[str, Any] = {
            "run_name": args.run_name,
            "pipeline_type": args.pipeline_type,
            "base_model": protocol["base_model"],
            "vae_model": protocol.get("vae_model"),
            "lora_repo": args.lora_repo,
            "lora_subfolder": args.lora_subfolder,
            "adapter_format": loaded_format,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "num_images_per_class": int(protocol["num_images_per_class"]),
            "height": int(protocol["height"]),
            "width": int(protocol["width"]),
            "num_inference_steps": int(protocol["num_inference_steps"]),
            "guidance_scale": float(protocol["guidance_scale"]),
            "seed": int(protocol["seed"]),
            "negative_prompt": protocol.get("negative_prompt", ""),
            "outputs": {},
        }

        images_per_class = int(protocol["num_images_per_class"])
        base_seed = int(protocol["seed"])
        width = int(protocol["width"])
        height = int(protocol["height"])
        with torch.inference_mode():
            for class_index, emotion in enumerate(emotions):
                prompt = str(protocol["emotions"][emotion]["prompt"])
                emotion_dir = run_dir / emotion
                manifest["outputs"][emotion] = []
                print(f"Generating {images_per_class} images for {emotion}: {prompt}")
                for image_index in range(images_per_class):
                    seed = base_seed + (class_index * 100000) + image_index
                    filename = f"{emotion}_{image_index:04d}_seed{seed}.png"
                    output_path = emotion_dir / filename
                    if args.skip_existing and output_path.is_file():
                        if image_matches_size(output_path, width, height):
                            manifest["outputs"][emotion].append(
                                {
                                    "prompt": prompt,
                                    "seed": seed,
                                    "path": str(output_path.relative_to(repo_root)),
                                    "status": "existing",
                                }
                            )
                            print(f"  kept {output_path.relative_to(repo_root)}")
                            continue
                        print(f"  replacing wrong-size image {output_path.relative_to(repo_root)}")

                    generator_device = "cuda" if device == "cuda" else "cpu"
                    generator = torch.Generator(device=generator_device).manual_seed(seed)
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=protocol.get("negative_prompt", ""),
                        num_inference_steps=int(protocol["num_inference_steps"]),
                        guidance_scale=float(protocol["guidance_scale"]),
                        generator=generator,
                        height=height,
                        width=width,
                    )
                    result.images[0].save(output_path)
                    manifest["outputs"][emotion].append(
                        {
                            "prompt": prompt,
                            "seed": seed,
                            "path": str(output_path.relative_to(repo_root)),
                            "status": "generated",
                        }
                    )
                    print(f"  saved {output_path.relative_to(repo_root)}")

        save_manifest(run_dir / "generation_manifest.json", manifest)
        print(f"Saved generation manifest: {run_dir / 'generation_manifest.json'}")
        print(f"Generated images under: {run_dir}")
    except GenerationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
