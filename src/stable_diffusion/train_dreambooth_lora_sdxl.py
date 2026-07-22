"""
Training script for SDXL using diffusers' built-in methods.
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import torch
import wandb
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from data_utils import InfantEmotionDataset


class DreamBoothDataset(torch.utils.data.Dataset):
    """Dataset wrapper for DreamBooth training with instance prompts."""

    def __init__(self, base_dataset: InfantEmotionDataset, instance_prompt_template: str) -> None:
        """
        Initialize DreamBooth dataset.

        Args:
            base_dataset: Base infant emotion dataset
            instance_prompt_template: Template for instance prompts
        """
        self.base_dataset = base_dataset
        self.instance_prompt_template = instance_prompt_template

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get an item with formatted prompt."""
        item = self.base_dataset[index]
        emotion = item["emotion"]
        prompt = self.instance_prompt_template.format(emotion)
        return {
            "image": item["image"],
            "prompt": prompt,
            "emotion": emotion,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for dataloader.

    Args:
        batch: List of samples

    Returns:
        Dictionary with batched tensors and prompts
    """
    images = torch.stack([item["image"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    emotions = [item["emotion"] for item in batch]
    return {
        "images": images,
        "prompts": prompts,
        "emotions": emotions,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SDXL")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/baby_emotion_samples",
        help="Directory containing images"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="./data/labels_formatted.json",
        help="Path to JSON labels file"
    )
    parser.add_argument(
        "--instance_prompt_template",
        type=str,
        default="a photo of a {} sks infant",
        help="Template for instance prompts"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1500,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/infant_lora",
        help="Directory to save model"
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
        default=None,
        help="WandB run name"
    )
    parser.add_argument(
        "--wandb_offline",
        action="store_true",
        help="Run wandb in offline mode"
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="InfantEmotionGen",
        help="Hugging Face repository name"
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Run in Colab mode"
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()
    torch.manual_seed(args.seed)

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode="offline" if args.wandb_offline else "online",
        config=vars(args),
    )

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )
    device = accelerator.device

    # Load the pipeline to get all components
    print("Loading models...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Extract components
    vae = pipe.vae.to(device)
    unet = pipe.unet.to(device)
    text_encoder = pipe.text_encoder.to(device)
    text_encoder_2 = pipe.text_encoder_2.to(device)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    noise_scheduler = pipe.scheduler

    # Freeze everything except UNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Enable gradient checkpointing for UNet
    unet.enable_gradient_checkpointing()

    # Create dataset
    print("Loading dataset...")
    base_dataset = InfantEmotionDataset(
        data_dir=args.data_dir,
        json_path=args.json_path,
        size=args.resolution,
    )

    dream_dataset = DreamBoothDataset(
        base_dataset=base_dataset,
        instance_prompt_template=args.instance_prompt_template,
    )

    dataloader = DataLoader(
        dream_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
    )

    # Prepare with accelerator
    unet, optimizer, dataloader = accelerator.prepare(
        unet, optimizer, dataloader
    )

    # Training loop
    print("Starting training...")
    global_step = 0
    running_loss = 0.0
    progress_bar = tqdm(range(args.max_train_steps))

    for _step in progress_bar:
        for batch in dataloader:
            # Move to device
            images = batch["images"].to(device, dtype=torch.float16)
            prompts = batch["prompts"]

            # Encode prompts with both text encoders (SDXL has two)
            with torch.no_grad():
                # First text encoder - CLIP (768 dims) - full sequence output
                tokenized_prompts = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
                # (batch, 77, 768)
                text_embeddings = text_encoder(tokenized_prompts)[0]

                # Second text encoder - CLIP (1280 dims) - full sequence output
                tokenized_prompts_2 = tokenizer_2(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
                # FIX: Use last_hidden_state to get 3D tensor (batch, 77, 1280)
                text_embeddings_2 = (
                    text_encoder_2(tokenized_prompts_2).last_hidden_state
                )

                # Concatenate along the last dimension for SDXL
                # Result: (batch, 77, 2048)
                text_embeddings = torch.cat(
                    [text_embeddings, text_embeddings_2], dim=-1
                )

            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215

            # Sample random timestep
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (images.shape[0],),
                device=device,
            ).long()

            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # SDXL requires added_cond_kwargs
            batch_size = images.shape[0]

            # Pooled text_embeds from second encoder
            text_embeds = text_embeddings_2.mean(dim=1)  # (batch, 1280)
            time_ids = torch.zeros(
                batch_size, 6, device=device, dtype=torch.float16
            )

            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }

            # Predict noise
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # Compute loss
            # Compute loss in float32
            loss = torch.nn.functional.mse_loss(
                noise_pred.float(), noise.float(), reduction="mean"
            )

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            running_loss += loss.item()
            avg_loss = running_loss / global_step

            wandb.log({
                "train/loss": loss.item(),
                "train/avg_loss": avg_loss,
                "train/global_step": global_step,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
            })

            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": avg_loss})

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Save final model
    print("Saving final model...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "unet_lora_final")
        os.makedirs(final_dir, exist_ok=True)

        # Save the UNet
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(final_dir)

        print(f"Training complete! Final model saved to {final_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
