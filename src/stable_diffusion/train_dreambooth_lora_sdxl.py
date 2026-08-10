"""
Training script for SDXL using diffusers' built-in methods with LoRA.

This script fine-tunes a pretrained Stable Diffusion XL (SDXL) model on an infant
facial expression dataset using DreamBooth and Low-Rank Adaptation (LoRA).
The model is trained to generate synthetic infant faces with controlled emotions
(angry, crying, happy) conditioned on text prompts.
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import torch
import wandb
from accelerate import Accelerator
from diffusers import AutoencoderKL, StableDiffusionXLPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import InfantEmotionDataset


class DreamBoothDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for DreamBooth training with formatted instance prompts.

    Generates prompt strings such as "a photo of a happy sks infant" on the fly
    from base dataset emotion labels.
    """

    def __init__(self, base_dataset: InfantEmotionDataset, instance_prompt_template: str) -> None:
        """
        Initialize the DreamBooth dataset wrapper.

        Args:
            base_dataset: InfantEmotionDataset containing image tensors and emotion labels.
            instance_prompt_template: Prompt template with a '{}' placeholder for emotion.
        """
        self.base_dataset = base_dataset
        self.instance_prompt_template = instance_prompt_template

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single training item with its emotion-specific prompt.

        Returns:
            Dict containing 'image', 'prompt', and 'emotion'.
        """
        item = self.base_dataset[index]
        prompt = self.instance_prompt_template.format(item["emotion"])
        return {
            "image": item["image"],
            "prompt": prompt,
            "emotion": item["emotion"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a batch of samples into tensors and lists for the DataLoader.

    Args:
        batch: List of individual sample dicts.

    Returns:
        Dict with batched 'images' tensor, 'prompts' list, and 'emotions' list.
    """
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "prompts": [item["prompt"] for item in batch],
        "emotions": [item["emotion"] for item in batch],
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train SDXL with LoRA")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/baby_emotion_samples",
        help="Directory containing images",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="./data/labels_formatted.json",
        help="Path to JSON labels file",
    )
    parser.add_argument(
        "--instance_prompt_template",
        type=str,
        default="a photo of a {} sks infant",
        help="Template for instance prompts",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Training batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1500,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/infant_lora",
        help="Directory to save model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="infant-emotion-generation",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity/username",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name",
    )
    parser.add_argument(
        "--wandb_offline",
        action="store_true",
        help="Run wandb in offline mode",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="InfantEmotionGen",
        help="Hugging Face repository name",
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Run in Colab mode",
    )
    return parser.parse_args()


def check_tensor(tensor: torch.Tensor, name: str, step: int) -> bool:
    """
    Check a tensor for NaN or Inf values for debugging.

    Args:
        tensor: Input tensor to check.
        name: Name of the tensor for logging.
        step: Current training step.

    Returns:
        True if NaN or Inf detected, False otherwise.
    """
    if torch.isnan(tensor).any():
        print(f"Warning: NaN in {name} at step {step}!")
        return True
    if torch.isinf(tensor).any():
        print(f"Warning: Inf in {name} at step {step}!")
        return True
    return False


def main() -> None:
    """Main training loop with model loading, optimization, and logging."""
    args = parse_args()

    # Set memory optimization for CUDA to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.manual_seed(args.seed)

    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode="offline" if args.wandb_offline else "online",
        config=vars(args),
    )

    # Initialize Accelerator for mixed-precision distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )
    device = accelerator.device

    # Load VAE in FP32 to maintain numerical stability during latent encoding
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.requires_grad_(False)

    # Load UNet in FP16 for memory efficiency
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    ).to(device)

    # Configure LoRA for parameter-efficient fine-tuning of cross-attention layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Load frozen text encoders and tokenizers from the base pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    text_encoder = pipe.text_encoder.to(device)
    text_encoder_2 = pipe.text_encoder_2.to(device)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    noise_scheduler = pipe.scheduler

    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Enable gradient checkpointing to reduce memory usage during backpropagation
    unet.enable_gradient_checkpointing()

    # Load the infant emotion dataset and wrap with DreamBooth prompt formatting
    base_dataset = InfantEmotionDataset(
        data_dir=args.data_dir,
        json_path=args.json_path,
        size=args.resolution,
    )
    dream_dataset = DreamBoothDataset(
        base_dataset=base_dataset,
        instance_prompt_template=args.instance_prompt_template,
    )

    # Single-sample batch with gradient accumulation yields effective batch size
    dataloader = DataLoader(
        dream_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # AdamW optimizer optimizes only the LoRA parameters attached to the UNet
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    # Prepare model, optimizer, and dataloader for accelerator
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    global_step = 0
    running_loss = 0.0
    progress_bar = tqdm(range(args.max_train_steps), desc="Training")

    # Main training loop
    for batch in dataloader:
        # Move images to device and convert to FP16
        images = batch["images"].to(device, dtype=torch.float16)
        prompts = batch["prompts"]

        if check_tensor(images, "images", global_step):
            optimizer.zero_grad()
            continue

        # Encode text prompts using both CLIP text encoders
        with torch.no_grad():
            tokenized_prompts = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            text_embeddings = text_encoder(tokenized_prompts)[0]

            tokenized_prompts_2 = tokenizer_2(
                prompts,
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            text_embeddings_2 = text_encoder_2(tokenized_prompts_2).last_hidden_state

            # Concatenate outputs from both encoders for SDXL conditioning
            text_embeddings = torch.cat([text_embeddings, text_embeddings_2], dim=-1)

        # Encode input images to latent space using frozen VAE (FP32)
        with torch.no_grad():
            latents = vae.encode(images.float()).latent_dist.sample()
            latents = latents * 0.18215  # VAE scaling factor for SDXL

            if check_tensor(latents, "latents", global_step):
                optimizer.zero_grad()
                continue

        # Sample random timestep and add noise to latents
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=device,
        ).long()
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Prepare SDXL-specific conditioning
        batch_size = images.shape[0]
        text_embeds = text_embeddings_2.mean(dim=1)
        time_ids = torch.zeros(batch_size, 6, device=device, dtype=torch.float16)
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}

        # Forward pass through UNet to predict added noise
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if check_tensor(noise_pred, "noise_pred", global_step):
            optimizer.zero_grad()
            continue

        # Compute MSE loss between predicted and actual noise (in FP32 for stability)
        loss = torch.nn.functional.mse_loss(
            noise_pred.float(), noise.float(), reduction="mean"
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at step {global_step}! Skipping...")
            optimizer.zero_grad()
            continue

        # Backward pass with gradient clipping and step update
        with accelerator.accumulate(unet):
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Update metrics and logging
        global_step += 1
        running_loss += loss.item()
        avg_loss = running_loss / global_step

        wandb.log({
            "train/loss": loss.item(),
            "train/avg_loss": avg_loss,
            "train/global_step": global_step,
        })

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item(), "avg_loss": avg_loss})

        if global_step >= args.max_train_steps:
            break

    # Save the fine-tuned LoRA adapter weights after training completes
    print("Saving final model...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "unet_lora_final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(final_dir)
        print(f"Training complete! Final model saved to {final_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
