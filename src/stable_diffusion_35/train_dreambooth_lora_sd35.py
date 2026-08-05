"""
FIXED: SD 3.5 Training with Emotion-Specific Prompts
Uses official Diffusers Flow Matching implementation
"""

import argparse
import os
import sys
import json
from typing import Any, Dict, List
from collections import Counter

import torch
import wandb
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, AutoencoderKL
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils.torch_utils import randn_tensor

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class InfantEmotionDataset(Dataset):
    """Dataset for infant emotion images."""
    
    def __init__(self, data_dir: str, json_path: str, size: int = 512):
        self.data_dir = data_dir
        self.size = size
        
        with open(json_path, 'r') as f:
            self.labels = json.load(f)
        
        self.image_paths = []
        self.emotions = []
        
        # Look for images in subdirectories
        emotion_dirs = ['angry', 'crying', 'happy']
        for emotion in emotion_dirs:
            emotion_path = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_path):
                for file in os.listdir(emotion_path):
                    if file.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        self.image_paths.append(os.path.join(emotion_path, file))
                        self.emotions.append(emotion)
        
        # If no images found, try root directory
        if not self.image_paths:
            for file in os.listdir(data_dir):
                if file.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    img_name = os.path.splitext(file)[0]
                    if img_name in self.labels:
                        self.image_paths.append(os.path.join(data_dir, file))
                        self.emotions.append(self.labels[img_name])
        
        print(f"Loaded {len(self.image_paths)} images")
        emotion_counts = Counter(self.emotions)
        print(f"Distribution: {dict(emotion_counts)}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_path = self.image_paths[index]
        emotion = self.emotions[index]
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return {
            "image": image,
            "emotion": emotion,
            "image_path": img_path,
        }


class DreamBoothDataset(Dataset):
    """Dataset wrapper with emotion-specific prompts."""
    
    def __init__(self, base_dataset: InfantEmotionDataset, instance_prompt_template: str) -> None:
        self.base_dataset = base_dataset
        self.instance_prompt_template = instance_prompt_template

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.base_dataset[index]
        emotion = item["emotion"]
        prompt = self.instance_prompt_template.format(emotion)
        return {
            "image": item["image"],
            "prompt": prompt,
            "emotion": emotion,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    emotions = [item["emotion"] for item in batch]
    return {
        "images": images,
        "prompts": prompts,
        "emotions": emotions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--data_dir", type=str, default="./data/baby_emotion_samples")
    parser.add_argument("--json_path", type=str, default="./data/labels_formatted.json")
    parser.add_argument("--instance_prompt_template", type=str, default="a photo of a {} sks infant")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_train_steps", type=int, default=1500)
    parser.add_argument("--output_dir", type=str, default="./models/infant_lora_sd35")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="infant-emotion-generation")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="sd35-training-fixed")
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--checkpoint_steps", type=int, default=500)
    return parser.parse_args()


def compute_padding(shape, pad, dtype, device):
    """Compute padding for text embeddings."""
    return torch.zeros(shape, dtype=dtype, device=device)


def main() -> None:
    args = parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.manual_seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode="offline" if args.wandb_offline else "online",
        config=vars(args),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )
    device = accelerator.device

    print("=" * 60)
    print("SD 3.5 Medium Training (FIXED - Flow Matching)")
    print("With Emotion-Specific Prompts")
    print("=" * 60)
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Max steps: {args.max_train_steps}")
    print("=" * 60)

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        token=args.token or True,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # Load MMDiT Transformer
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float16,
        token=args.token or True,
    ).to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Load text encoders
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
    text_encoder.eval()
    text_encoder_2.requires_grad_(False)
    text_encoder_2.eval()

    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        token=args.token or True,
    )

    # Load dataset
    print("Loading dataset...")
    base_dataset = InfantEmotionDataset(
        data_dir=args.data_dir,
        json_path=args.json_path,
        size=args.resolution,
    )

    if len(base_dataset) == 0:
        print("❌ ERROR: No images found!")
        return

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
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # Prepare with accelerator
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )
    transformer.train()

    print("Starting training...")
    print("Expected: ~8-12 seconds per step on A100")
    
    global_step = 0
    running_loss = 0.0
    progress_bar = tqdm(range(args.max_train_steps))
    data_iterator = iter(dataloader)

    for step_idx in range(args.max_train_steps):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch = next(data_iterator)
        
        images = batch["images"].to(device, dtype=torch.float16)
        prompts = batch["prompts"]

        # CRITICAL: Encode text with proper gradient handling
        with torch.no_grad():
            # CLIP-L
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
            
            pooled_projection_1 = (
                text_encoder_output.pooler_output 
                if hasattr(text_encoder_output, 'pooler_output') and text_encoder_output.pooler_output is not None
                else text_encoder_output.last_hidden_state.mean(dim=1)
            )

            # OpenCLIP-G
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
            
            pooled_projection_2 = (
                text_encoder_output_2.pooler_output
                if hasattr(text_encoder_output_2, 'pooler_output') and text_encoder_output_2.pooler_output is not None
                else text_encoder_output_2.last_hidden_state.mean(dim=1)
            )

            pooled_projections = torch.cat([pooled_projection_1, pooled_projection_2], dim=-1)
            
            # Pad to 4096 (T5-XXL excluded)
            t5_dim = 4096 - 768 - 1280
            zero_padding = torch.zeros(
                text_embeddings_1.shape[0], 
                text_embeddings_1.shape[1], 
                t5_dim,
                device=text_embeddings_1.device,
                dtype=text_embeddings_1.dtype
            )
            text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2, zero_padding], dim=-1)

        # Encode images
        with torch.no_grad():
            latents = vae.encode(images.float()).latent_dist.sample()
            latents = latents * 0.18215

        # ============================================================
        # CRITICAL FIX: Correct Flow Matching implementation
        # Based on official Diffusers SD3 training
        # ============================================================
        
        # Sample random timesteps with logit-normal distribution
        # This is the official way to sample timesteps for SD3
        u = torch.normal(mean=0.0, std=1.0, size=(images.shape[0],), device=device)
        timesteps = torch.sigmoid(u)
        
        # Ensure timesteps are in [0, 1] range
        timesteps = timesteps.clamp(0.0, 1.0)
        
        # Expand timesteps for broadcasting
        timesteps_expanded = timesteps.view(-1, 1, 1, 1)
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Flow Matching interpolation (official implementation)
        noisy_latents = (1.0 - timesteps_expanded) * latents + timesteps_expanded * noise
        
        # Target velocity (official implementation)
        target_velocity = noise - latents

        # Convert to correct dtype
        noisy_latents = noisy_latents.to(dtype=torch.float16)
        timesteps = timesteps.to(dtype=torch.float16)
        text_embeddings = text_embeddings.to(dtype=torch.float16)
        pooled_projections = pooled_projections.to(dtype=torch.float16)

        # ============================================================
        # CRITICAL: Forward pass with proper gradient computation
        # ============================================================
        # The model predicts velocity (noise - latents)
        # This must be done with gradient tracking
        predicted_velocity = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled_projections,
            return_dict=False,
        )[0]

        # Compute loss (same as SDXL)
        loss = torch.nn.functional.mse_loss(
            predicted_velocity.float(), target_velocity.float(), reduction="mean"
        )

        # ============================================================
        # Backward pass
        # ============================================================
        accelerator.backward(loss)
        
        if (step_idx + 1) % args.gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1
        running_loss += loss.item()
        avg_loss = running_loss / global_step

        # Log metrics
        wandb.log({
            "train/loss": loss.item(),
            "train/avg_loss": avg_loss,
            "train/global_step": global_step,
            "train/timestep_mean": timesteps.mean().item(),
        })

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item(), "avg_loss": avg_loss})

        # Save checkpoint
        if global_step % args.checkpoint_steps == 0:
            unwrapped = accelerator.unwrap_model(transformer)
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped.save_pretrained(checkpoint_dir)
            print(f"\n✅ Checkpoint saved at step {global_step}")

    # Save final model
    print("Saving final model...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "mmdit_lora_final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(final_dir)
        print(f"\n✅ Training complete! Model saved to {final_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()