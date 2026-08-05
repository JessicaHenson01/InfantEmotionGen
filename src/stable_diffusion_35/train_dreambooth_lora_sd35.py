"""
Training script for SD 3.5 Medium using diffusers' built-in methods with LoRA.
Mirrors the SDXL training setup but adapted for SD 3.5 Medium architecture.

This script fine-tunes a pretrained Stable Diffusion 3.5 Medium model on an infant 
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
from diffusers import DDPMScheduler, SD3Transformer2DModel, AutoencoderKL
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

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
    parser = argparse.ArgumentParser(description="Train SD 3.5 Medium with LoRA")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
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
        default="./models/infant_lora_sd35",
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
        default="InfantEmotionGen/SD35Primary",
        help="Hugging Face repository name"
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Run in Colab mode"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for gated models"
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
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.manual_seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode="offline" if args.wandb_offline else "online",
        config=vars(args),
    )

    # ============================================================
    # Step 2: Initialize Accelerator
    # ============================================================
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )
    device = accelerator.device

    print("Loading SD 3.5 Medium models...")

    # ============================================================
    # Step 3: Load Pretrained Models
    # ============================================================
    
    # Load VAE in FP32 for stability (same as SDXL)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        token=args.token or True,
    ).to(device)
    vae.requires_grad_(False)

    # Load MMDiT Transformer in FP16 (this is the equivalent of UNet for SD3)
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float16,
        token=args.token or True,
    ).to(device)

    # ============================================================
    # Step 4: Apply LoRA to MMDiT (Parameter-Efficient Fine-Tuning)
    # ============================================================
    
    # For SD3.5 Medium, we target the attention layers in the MMDiT
    # The key difference from SDXL: using q_proj, k_proj, v_proj, out_proj
    # instead of attn2.to_q, attn2.to_k, etc.
    lora_config = LoraConfig(
        r=16,  # Rank of the low-rank matrices
        lora_alpha=16,  # Scaling factor
        target_modules=[
            "to_q",
            "to_k", 
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ],  # All attention layers in MMDiT
        lora_dropout=0.1,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # ============================================================
    # Step 5: Load Text Encoders and Tokenizers (T5-XXL excluded)
    # ============================================================
    
    # Load text encoders (only CLIP-L and OpenCLIP-G, skip T5-XXL)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        token=args.token or True,
    ).to(device)
    
    text_encoder_2 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
        token=args.token or True,
    ).to(device)
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        token=args.token or True,
    )
    
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        token=args.token or True,
    )

    # Freeze text encoders
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # ============================================================
    # Step 6: Load Scheduler (DDPM for training)
    # ============================================================
    
    # Use DDPM scheduler for training (same as SDXL approach)
    scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        token=args.token or True,
    )

    # ============================================================
    # Step 7: Load Dataset and Create DataLoader
    # ============================================================
    
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

    # ============================================================
    # Step 8: Setup Optimizer
    # ============================================================
    
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
    )

    # ============================================================
    # Step 9: Prepare Model with Accelerator
    # ============================================================
    
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )

    print("Starting training...")
    global_step = 0
    running_loss = 0.0
    progress_bar = tqdm(range(args.max_train_steps))

    # ============================================================
    # Step 10: Training Loop
    # ============================================================
    
    for batch in dataloader:
        # ----- 10a: Move batch to device -----
        images = batch["images"].to(device, dtype=torch.float16)
        prompts = batch["prompts"]

        if check_tensor(images, "images", global_step):
            optimizer.zero_grad()
            continue

        # ----- 10b: Encode text prompts using both CLIP text encoders -----
        with torch.no_grad():
            # First text encoder (CLIP-L)
            tokenized_prompts = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            
            text_encoder_output = text_encoder(
                tokenized_prompts,
                output_hidden_states=True,
                return_dict=True,
            )
            
            text_embeddings_1 = text_encoder_output.last_hidden_state
            
            # Get pooled projection for MMDiT conditioning
            if hasattr(text_encoder_output, 'pooler_output') and text_encoder_output.pooler_output is not None:
                pooled_projection_1 = text_encoder_output.pooler_output
            else:
                pooled_projection_1 = text_encoder_output.last_hidden_state.mean(dim=1)

            # Second text encoder (OpenCLIP-G)
            tokenized_prompts_2 = tokenizer_2(
                prompts,
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            text_encoder_output_2 = text_encoder_2(
                tokenized_prompts_2,
                output_hidden_states=True,
                return_dict=True,
            )

            text_embeddings_2 = text_encoder_output_2.last_hidden_state
            
            if hasattr(text_encoder_output_2, 'pooler_output') and text_encoder_output_2.pooler_output is not None:
                pooled_projection_2 = text_encoder_output_2.pooler_output
            else:
                pooled_projection_2 = text_encoder_output_2.last_hidden_state.mean(dim=1)

            # Concatenate pooled projections for MMDiT (2048 dims)
            pooled_projections = torch.cat([pooled_projection_1, pooled_projection_2], dim=-1)
            
            # Pad text embeddings to 4096 to match T5-XXL dimension (excluded)
            t5_dim = 4096 - 768 - 1280
            zero_padding = torch.zeros(
                text_embeddings_1.shape[0], 
                text_embeddings_1.shape[1], 
                t5_dim,
                device=text_embeddings_1.device,
                dtype=text_embeddings_1.dtype
            )
            text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2, zero_padding], dim=-1)

        # ----- 10c: Encode images to latent space using VAE -----
        with torch.no_grad():
            latents = vae.encode(images.float()).latent_dist.sample()
            latents = latents * 0.18215

            if check_tensor(latents, "latents", global_step):
                optimizer.zero_grad()
                continue

        # ----- 10d: Sample random timestep and add noise -----
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=device,
        ).long()

        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # ----- 10e: Forward pass through MMDiT -----
        # CRITICAL: MMDiT forward pass requires:
        # - hidden_states: noisy latents
        # - timestep: timestep indices
        # - encoder_hidden_states: text embeddings (padded to 4096)
        # - pooled_projections: pooled embeddings (2048 dims)
        noise_pred = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled_projections,
            return_dict=False,
        )[0]

        if check_tensor(noise_pred, "noise_pred", global_step):
            optimizer.zero_grad()
            continue

        # ----- 10f: Compute loss (MSE between predicted and actual noise) -----
        loss = torch.nn.functional.mse_loss(
            noise_pred.float(), noise.float(), reduction="mean"
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf loss at step {global_step}! Skipping...")
            optimizer.zero_grad()
            continue

        # ----- 10g: Backward pass with gradient accumulation -----
        with accelerator.accumulate(transformer):
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

        # ----- 10h: Log metrics -----
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

    # ============================================================
    # Step 11: Save Fine-Tuned Model
    # ============================================================
    
    print("Saving final model...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "mmdit_lora_final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        unwrapped_transformer.save_pretrained(final_dir)
        print(f"Training complete! Final model saved to {final_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()