"""
Training script for SDXL using diffusers' built-in methods with LoRA.

This script fine-tunes a pretrained Stable Diffusion XL (SDXL) model on an infant 
facial expression dataset using DreamBooth and Low-Rank Adaptation (LoRA). 
The model is trained to generate synthetic infant faces with controlled 
emotions (angry, crying, happy) conditioned on text prompts.
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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from data_utils import InfantEmotionDataset


class DreamBoothDataset(torch.utils.data.Dataset):
    """Dataset wrapper for DreamBooth training with instance prompts.

    This class wraps the base dataset and generates instance-specific prompts
    for each image, such as "a photo of a happy sks infant".
    """

    def __init__(self, base_dataset: InfantEmotionDataset, instance_prompt_template: str) -> None:
        """
        Initialize DreamBooth dataset.

        Args:
            base_dataset: Base infant emotion dataset containing images and labels
            instance_prompt_template: Template string with {} placeholder for emotion
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
        # Create prompt like "a photo of a happy sks infant"
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
    parser = argparse.ArgumentParser(description="Train SDXL with LoRA")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base model to fine-tune"
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
        default=512,
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
        default=5e-6,
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


def check_tensor(
    tensor: torch.Tensor,
    name: str,
    step: int
) -> bool:
    """
    Debug helper to check for NaN/Inf in tensors.

    Args:
        tensor: Tensor to check
        name: Name of the tensor for logging
        step: Current training step

    Returns:
        True if NaN/Inf found, False otherwise
    """
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN in {name} at step {step}!")
        return True
    if torch.isinf(tensor).any():
        print(f"⚠️ Inf in {name} at step {step}!")
        return True
    return False


def main() -> None:
    """Main training function."""
    args = parse_args()

    # ============================================================
    # Step 1: Environment Setup
    # ============================================================
    
    # Set memory optimization to reduce fragmentation on CUDA
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.manual_seed(args.seed)  # Set random seed for reproducibility

    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode="offline" if args.wandb_offline else "online",
        config=vars(args),  # Log all hyperparameters
    )

    # ============================================================
    # Step 2: Initialize Accelerator for Distributed Training
    # ============================================================
    
    # Accelerator handles device placement, mixed precision, and gradient accumulation
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",  # FP16 for faster training and lower memory
    )
    device = accelerator.device

    print("Loading models...")

    # ============================================================
    # Step 3: Load Pretrained Models (VAE, UNet, Text Encoders)
    # ============================================================
    
    # Load VAE in FP32 for numerical stability (prevents NaN in latents)
    # VAE encodes images to latent space and decodes latents back to images
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,  # FP32 for stability
    ).to(device)
    vae.requires_grad_(False)  # Freeze VAE - only UNet is trained

    # Load UNet in FP16 for memory efficiency
    # UNet is the denoising network that predicts noise in the latent space
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    ).to(device)

    # ============================================================
    # Step 4: Apply LoRA to UNet (Parameter-Efficient Fine-Tuning)
    # ============================================================
    
    # LoRA freezes the original UNet weights and adds trainable low-rank matrices
    # Only cross-attention layers (attn2) are adapted to preserve model knowledge
    lora_config = LoraConfig(
        r=16,  # Rank of the low-rank matrices
        lora_alpha=16,  # Scaling factor
        target_modules=[
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ],  # Only cross-attention layers
        lora_dropout=0.1,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()  # Prints trainable parameter count

    # Load text encoders from pipeline (frozen)
    # Text encoders convert text prompts to embeddings for conditioning
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # xFormers disabled to avoid installation issues (memory-efficient attention)
    # pipe.enable_xformers_memory_efficient_attention()

    # Extract components from pipeline
    text_encoder = pipe.text_encoder.to(device)  # First CLIP text encoder
    text_encoder_2 = pipe.text_encoder_2.to(device)  # Second CLIP text encoder
    tokenizer = pipe.tokenizer  # Tokenizer for first text encoder
    tokenizer_2 = pipe.tokenizer_2  # Tokenizer for second text encoder
    noise_scheduler = pipe.scheduler  # Controls the denoising schedule

    # Freeze text encoders (only UNet LoRA parameters are trained)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Gradient checkpointing reduces memory usage at the cost of compute
    unet.enable_gradient_checkpointing()

    # ============================================================
    # Step 5: Load Dataset and Create DataLoader
    # ============================================================
    
    print("Loading dataset...")
    base_dataset = InfantEmotionDataset(
        data_dir=args.data_dir,
        json_path=args.json_path,
        size=args.resolution,
    )

    # Wrap base dataset with DreamBooth prompt formatting
    dream_dataset = DreamBoothDataset(
        base_dataset=base_dataset,
        instance_prompt_template=args.instance_prompt_template,
    )

    # DataLoader with batch_size=1 (effective batch size = gradient_accumulation_steps)
    dataloader = DataLoader(
        dream_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ============================================================
    # Step 6: Setup Optimizer
    # ============================================================
    
    # AdamW optimizer only updates LoRA parameters (trainable params)
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
    )

    # ============================================================
    # Step 7: Prepare Model with Accelerator
    # ============================================================
    
    # Accelerator prepares the model, optimizer, and dataloader for distributed training
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    print("Starting training...")
    global_step = 0
    running_loss = 0.0
    progress_bar = tqdm(range(args.max_train_steps))

    # ============================================================
    # Step 8: Training Loop
    # ============================================================
    
    for batch in dataloader:
        # ----- 8a: Move batch to device -----
        # Images are in FP16 for memory efficiency
        images = batch["images"].to(device, dtype=torch.float16)
        prompts = batch["prompts"]

        # ----- 8b: Check for NaN in input images -----
        if check_tensor(images, "images", global_step):
            print(f"   Images range: min={images.min():.4f}, max={images.max():.4f}")
            optimizer.zero_grad()
            continue

        # ----- 8c: Encode text prompts using both CLIP text encoders -----
        with torch.no_grad():
            # First text encoder (CLIP-ViT-L/14) - 768 dims
            tokenized_prompts = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            text_embeddings = text_encoder(tokenized_prompts)[0]  # (batch, 77, 768)

            # Second text encoder (CLIP-ViT-G/14) - 1280 dims
            tokenized_prompts_2 = tokenizer_2(
                prompts,
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            text_embeddings_2 = text_encoder_2(
                tokenized_prompts_2
            ).last_hidden_state  # (batch, 77, 1280)

            # Concatenate embeddings for SDXL: (batch, 77, 2048)
            text_embeddings = torch.cat(
                [text_embeddings, text_embeddings_2], dim=-1
            )

        # ----- 8d: Encode images to latent space using VAE -----
        with torch.no_grad():
            # Convert images to FP32 for VAE encoding (VAE is in FP32)
            latents = vae.encode(images.float()).latent_dist.sample()
            latents = latents * 0.18215  # VAE scaling factor

            if check_tensor(latents, "latents", global_step):
                print("   VAE produced NaN! Skipping...")
                optimizer.zero_grad()
                continue

        # ----- 8e: Sample random timestep and add noise -----
        # Diffusion models are trained to denoise images at various noise levels
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=device,
        ).long()

        noise = torch.randn_like(latents)  # Random Gaussian noise
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # ----- 8f: Prepare SDXL-specific conditioning -----
        batch_size = images.shape[0]
        text_embeds = text_embeddings_2.mean(dim=1)  # Pooled text embeddings (batch, 1280)
        time_ids = torch.zeros(
            batch_size, 6, device=device, dtype=torch.float16
        )  # Dummy time IDs for SDXL

        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }  # SDXL requires these additional conditioning parameters

        # ----- 8g: Forward pass through UNet -----
        # UNet predicts the noise that was added to the latents
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

        # ----- 8h: Compute loss (MSE between predicted and actual noise) -----
        # Loss is computed in FP32 for numerical stability
        loss = torch.nn.functional.mse_loss(
            noise_pred.float(), noise.float(), reduction="mean"
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf loss at step {global_step}! Skipping...")
            optimizer.zero_grad()
            continue

        # ----- 8i: Backward pass with gradient accumulation -----
        # accelerator.accumulate handles gradient accumulation automatically
        with accelerator.accumulate(unet):
            accelerator.backward(loss)

            # Gradient clipping prevents exploding gradients
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), max_norm=1.0)

            # Update weights (only LoRA parameters are updated)
            optimizer.step()
            optimizer.zero_grad()

        # ----- 8j: Log metrics -----
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

        # Stop if we've reached the maximum number of steps
        if global_step >= args.max_train_steps:
            break

    # ============================================================
    # Step 9: Save Fine-Tuned Model
    # ============================================================
    
    print("Saving final model...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save the LoRA adapter weights (not the full model)
        final_dir = os.path.join(args.output_dir, "unet_lora_final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(final_dir)  # Saves adapter_config.json and adapter_model.safetensors
        print(f"Training complete! Final model saved to {final_dir}")

    wandb.finish()  # End Weights & Biases run


if __name__ == "__main__":
    main()
