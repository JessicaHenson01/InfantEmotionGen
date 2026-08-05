"""
DreamBooth + LoRA training for Stable Diffusion 3.5 Medium.
"""

import argparse
import os
import sys
import shutil
import json
from typing import Any, Dict, List
from PIL import Image

# ============================================================
# FIX: Disable torchao in PEFT BEFORE importing peft
# ============================================================
os.environ["PEFT_DISABLE_TORCHAO"] = "1"

import torch
import wandb
from accelerate import Accelerator
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import numpy as np
import time
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from stable_diffusion_35.config import CONFIG


class InfantEmotionDataset(Dataset):
    """Dataset for infant emotion images."""
    
    def __init__(self, data_dir: str, json_path: str, size: int = 512):
        self.data_dir = data_dir
        self.size = size
        
        # Load labels
        with open(json_path, 'r') as f:
            self.labels = json.load(f)
        
        # Get all image files
        self.image_paths = []
        self.emotions = []
        
        # Look for images in subdirectories
        emotion_dirs = ['angry', 'crying', 'happy']
        for emotion in emotion_dirs:
            emotion_path = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_path):
                for file in os.listdir(emotion_path):
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(emotion_path, file))
                        self.emotions.append(emotion)
        
        # If no images found in subdirs, try root directory
        if not self.image_paths:
            for file in os.listdir(data_dir):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    img_name = os.path.splitext(file)[0]
                    if img_name in self.labels:
                        self.image_paths.append(os.path.join(data_dir, file))
                        self.emotions.append(self.labels[img_name])
        
        print(f"Loaded {len(self.image_paths)} images")
        
        # Print distribution
        from collections import Counter
        emotion_counts = Counter(self.emotions)
        print(f"Distribution: {dict(emotion_counts)}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_path = self.image_paths[index]
        emotion = self.emotions[index]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return {
            "image": image,
            "emotion": emotion,
            "image_path": img_path,
        }


class DreamBoothDataset(Dataset):
    """Dataset wrapper for DreamBooth training with instance prompts."""

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
    parser = argparse.ArgumentParser(description="Train SD 3.5 Medium with LoRA")
    parser.add_argument(
        "--model_id",
        type=str,
        default=CONFIG["model_id"],
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
        default=CONFIG["instance_prompt_template"],
        help="Template for instance prompts"
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=CONFIG["class_prompt"],
        help="Class prompt for prior preservation"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=CONFIG["resolution"],
        help="Image resolution (matches SDXL)"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=CONFIG["train_batch_size"],
        help="Training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=CONFIG["gradient_accumulation_steps"],
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=CONFIG["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=CONFIG["max_train_steps"],
        help="Maximum training steps"
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        default=CONFIG["checkpoint_steps"],
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=CONFIG["output_dir"],
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
        default=CONFIG["wandb_project"],
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
        default=CONFIG["wandb_run_name"],
        help="WandB run name"
    )
    parser.add_argument(
        "--wandb_offline",
        action="store_true",
        help="Run wandb in offline mode"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for gated models"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.manual_seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode="offline" if args.wandb_offline else "online",
        config={
            "model": args.model_id,
            "resolution": args.resolution,
            "learning_rate": args.learning_rate,
            "max_train_steps": args.max_train_steps,
            "checkpoint_steps": args.checkpoint_steps,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lora_rank": CONFIG["lora_rank"],
            "lora_alpha": CONFIG["lora_alpha"],
            "mixed_precision": CONFIG["mixed_precision"],
            "text_encoders": "CLIP-L + OpenCLIP-G (T5-XXL excluded)",
            "architecture": "MMDiT (SD 3.5 Medium)",
        }
    )

    print("📊 WandB initialized!")
    print(f"   Project: {args.wandb_project}")
    print(f"   Run: {args.wandb_run_name}")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=CONFIG["mixed_precision"],
    )
    device = accelerator.device

    print("Loading SD 3.5 Medium model...")

    # ============================================================
    # Load the model components separately for training
    # ============================================================
    from diffusers import AutoencoderKL, SD3Transformer2DModel
    from transformers import CLIPTextModel, CLIPTokenizer
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.float16,
        token=args.token or True,
    ).to(device)
    
    # Load transformer (MMDiT)
    transformer = SD3Transformer2DModel.from_pretrained(
        args.model_id,
        subfolder="transformer",
        torch_dtype=torch.float16,
        token=args.token or True,
    ).to(device)
    
    # Load text encoders (CLIP-L and OpenCLIP-G, skip T5)
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        token=args.token or True,
    ).to(device)
    
    text_encoder_2 = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
        token=args.token or True,
    ).to(device)
    
    # Load tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id,
        subfolder="tokenizer",
        token=args.token or True,
    )
    
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.model_id,
        subfolder="tokenizer_2",
        token=args.token or True,
    )

    # ============================================================
    # Use DDPM scheduler for training
    # ============================================================
    scheduler = DDPMScheduler.from_pretrained(
        args.model_id,
        subfolder="scheduler",
        token=args.token or True,
    )

    print("✅ Model components loaded!")

    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"📊 Total transformer parameters: {total_params:,}")

    # ============================================================
    # Find ALL attention modules across ALL blocks
    # ============================================================
    print("\n🔍 Finding attention modules...")
    
    target_modules = []
    for name, module in transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(x in name for x in ['to_q', 'to_k', 'to_v', 'to_out']):
                target_modules.append(name)
    
    # Filter to only transformer blocks
    target_modules = [m for m in target_modules if 'transformer_blocks' in m]
    print(f"Found {len(target_modules)} target modules")

    lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
    )

    transformer = get_peft_model(transformer, lora_config)
    
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"📊 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Freeze text encoders
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    text_encoder_2.requires_grad_(False)
    text_encoder_2.eval()

    print("Loading dataset...")
    base_dataset = InfantEmotionDataset(
        data_dir=args.data_dir,
        json_path=args.json_path,
        size=args.resolution,
    )

    if len(base_dataset) == 0:
        print("❌ ERROR: No images found in dataset!")
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

    transformer.train()
    print("✅ Model set to training mode")

    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )
    
    transformer.train()

    emotion_prompts = {
        "angry": "a photo of an angry sks infant",
        "crying": "a photo of a crying sks infant",
        "happy": "a photo of a happy sks infant",
    }

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Total steps: {args.max_train_steps}")
    print(f"Batch size: {args.train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.train_batch_size * args.gradient_accumulation_steps}")
    print(f"Expected time: ~4-5 hours on A100")
    print(f"Each step should take ~8-12 seconds")
    print("=" * 60 + "\n")
    
    global_step = 0
    running_loss = 0.0
    progress_bar = tqdm(range(args.max_train_steps))
    
    step_times = []

    for batch in dataloader:
        step_start_time = time.time()
        
        images = batch["images"].to(device, dtype=torch.float16)
        prompts = batch["prompts"]

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
            
            if hasattr(text_encoder_output, 'pooler_output') and text_encoder_output.pooler_output is not None:
                pooled_projection_1 = text_encoder_output.pooler_output
            else:
                pooled_projection_1 = text_encoder_output.last_hidden_state.mean(dim=1)

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
            
            if hasattr(text_encoder_output_2, 'pooler_output') and text_encoder_output_2.pooler_output is not None:
                pooled_projection_2 = text_encoder_output_2.pooler_output
            else:
                pooled_projection_2 = text_encoder_output_2.last_hidden_state.mean(dim=1)

            pooled_projections = torch.cat([pooled_projection_1, pooled_projection_2], dim=-1)
            
            t5_dim = 4096 - 768 - 1280
            zero_padding = torch.zeros(
                text_embeddings_1.shape[0], 
                text_embeddings_1.shape[1], 
                t5_dim,
                device=text_embeddings_1.device,
                dtype=text_embeddings_1.dtype
            )
            text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2, zero_padding], dim=-1)

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215

        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=device,
        ).long()

        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        noisy_latents = noisy_latents.to(device=device, dtype=torch.float16)
        timesteps = timesteps.to(device)
        text_embeddings = text_embeddings.to(device=device, dtype=torch.float16)
        pooled_projections = pooled_projections.to(device=device, dtype=torch.float16)

        # Forward pass
        noise_pred = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled_projections,
            return_dict=False,
        )[0]

        # Compute loss
        loss = torch.nn.functional.mse_loss(
            noise_pred.float(), 
            noise.float(), 
            reduction="mean"
        )

        loss = loss / args.gradient_accumulation_steps

        with accelerator.accumulate(transformer):
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

        global_step += 1
        
        actual_loss = loss.item() * args.gradient_accumulation_steps
        running_loss += actual_loss
        avg_loss = running_loss / global_step
        
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times[-10:]) / min(len(step_times), 10)

        wandb.log({
            "train/loss": actual_loss,
            "train/avg_loss": avg_loss,
            "train/global_step": global_step,
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            "train/step_time": step_time,
        })

        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{actual_loss:.4f}", 
            "avg_loss": f"{avg_loss:.4f}",
            "time": f"{avg_step_time:.1f}s"
        })

        if global_step >= args.max_train_steps:
            break

    print("\nSaving final model...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "mmdit_lora_final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(final_dir)

        print(f"\n✅ Training complete! Final model saved to {final_dir}")
        
        if step_times:
            avg_time = sum(step_times) / len(step_times)
            total_time = sum(step_times)
            print(f"\n📊 Training Statistics:")
            print(f"   Total steps: {global_step}")
            print(f"   Total time: {total_time/60:.1f} minutes")
            print(f"   Average time per step: {avg_time:.2f} seconds")
            print(f"   Final loss: {actual_loss:.4f}")
            print(f"   Final avg loss: {avg_loss:.4f}")

    wandb.log({"status": "training_complete"})
    wandb.finish()


if __name__ == "__main__":
    main()