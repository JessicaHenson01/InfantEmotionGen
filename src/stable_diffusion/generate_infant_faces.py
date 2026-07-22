"""
Image generation script for SDXL with LoRA weights.
"""

import argparse
import os

import torch
import wandb
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler


def main() -> None:
    """
    Main function to generate infant emotion images using SDXL with LoRA weights.

    Generates images for each emotion (angry, crying, happy) using the trained
    LoRA weights and saves them to the specified output directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora_path", type=str, default="./models/infant_lora/unet_lora_final")
    parser.add_argument("--output_dir", type=str, default="./generated_images/sdxl")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="infant-emotion-generation")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or "sdxl-inference",
        config=vars(args),
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to("cuda")

    # Try to enable xFormers, but continue if not available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xFormers enabled")
    except (ModuleNotFoundError, ImportError) as error:
        print(f"⚠️ xFormers not available: {error}")
        print("⚠️ Continuing without xFormers")

    # Load LoRA weights
    pipe.load_lora_weights(args.lora_path)

    # Use DPM++ sampler for better quality
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )

    # Define prompts
    emotion_prompts = {
        "angry": "a photo of an angry sks infant",
        "crying": "a photo of a crying sks infant",
        "happy": "a photo of a happy sks infant",
    }

    print("Generating images...")
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

            if (idx + 1) % 10 == 0:
                print(f"  Generated {idx + 1}/{args.num_images}")

    # Log to wandb
    artifact = wandb.Artifact(
        name="sdxl_generated_images",
        type="dataset",
        description=f"Generated images ({args.num_images} per emotion)",
    )
    artifact.add_dir(args.output_dir)
    wandb.log_artifact(artifact)

    for emotion in emotion_prompts:
        sample_dir = os.path.join(args.output_dir, emotion)
        if os.listdir(sample_dir):
            sample_path = os.path.join(sample_dir, os.listdir(sample_dir)[0])
            wandb.log({f"samples/{emotion}": wandb.Image(sample_path)})

    print(f"Generation complete! Images saved to {args.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
