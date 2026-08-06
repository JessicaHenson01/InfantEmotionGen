"""
Image generation script for Stable Diffusion 3.5 Medium with LoRA weights.
T5-XXL is excluded (only CLIP-L + OpenCLIP-G).
Full WandB logging integration.
"""

import argparse
import os

import torch
import wandb
from diffusers import AutoPipelineForText2Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with SD 3.5 Medium")
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Base model ID"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="./models/infant_lora_sd35/mmdit_lora_final",
        help="Path to LoRA weights"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images/sd35",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of images per emotion"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Guidance scale for generation"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="infant-emotion-generation",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity/username"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sd35-inference",
        help="WandB run name"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or "sd35-inference",
        config={
            "model_id": args.model_id,
            "lora_path": args.lora_path,
            "num_images": args.num_images,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
        }
    )

    print(f"📊 WandB initialized!")
    print(f"   Project: {args.wandb_project}")
    print(f"   Run: {args.wandb_run_name}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SD 3.5 Medium pipeline (T5-XXL excluded)...")

    # Load pipeline without T5-XXL
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to("cuda")

    # Try to enable xFormers
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xFormers enabled")
    except (ModuleNotFoundError, ImportError) as error:
        print(f"⚠️ xFormers not available: {error}")

    # Load LoRA weights
    print(f"Loading LoRA weights from: {args.lora_path}")
    pipe.load_lora_weights(args.lora_path)

    # Define prompts (same as SDXL)
    emotion_prompts = {
        "angry": "a photo of an angry sks infant",
        "crying": "a photo of a crying sks infant",
        "happy": "a photo of a happy sks infant",
    }

    print("Generating images...")
    generated_files = []

    for emotion, prompt in emotion_prompts.items():
        emotion_dir = os.path.join(args.output_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)

        print(f"Generating {args.num_images} {emotion} images...")
        for idx in range(args.num_images):
            generator = torch.Generator("cuda").manual_seed(args.seed + idx)

            with torch.cuda.amp.autocast():
                result = pipe(
                    prompt=prompt,
                    negative_prompt="cartoon, drawing, blurry, low quality, distorted, deformed",
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    height=1024,
                    width=1024,
                )

            save_path = os.path.join(emotion_dir, f"{emotion}_{idx:04d}.png")
            result.images[0].save(save_path)
            generated_files.append(save_path)

            if (idx + 1) % 10 == 0:
                print(f"  Generated {idx + 1}/{args.num_images}")

            # Log individual images to WandB (every 10th image)
            if idx % 10 == 0:
                wandb.log({
                    f"generated/{emotion}_sample_{idx}": wandb.Image(save_path, caption=f"{emotion} baby {idx}")
                })

    # Log all generated images as a WandB artifact
    artifact = wandb.Artifact(
        name="sd35_generated_images",
        type="dataset",
        description=f"Generated images ({args.num_images} per emotion)",
    )
    artifact.add_dir(args.output_dir)
    wandb.log_artifact(artifact)

    # Log sample images for each emotion
    for emotion in emotion_prompts:
        sample_dir = os.path.join(args.output_dir, emotion)
        if os.listdir(sample_dir):
            sample_path = os.path.join(sample_dir, os.listdir(sample_dir)[0])
            wandb.log({f"samples/{emotion}": wandb.Image(sample_path, caption=f"{emotion} baby")})

    print(f"Generation complete! Images saved to {args.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()