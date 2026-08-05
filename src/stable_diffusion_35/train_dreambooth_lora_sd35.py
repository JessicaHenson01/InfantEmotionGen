"""
DreamBooth + LoRA training for Stable Diffusion 3.5 Medium.

Key differences from SDXL:
1. Uses AutoPipelineForText2Image instead of StableDiffusionXLPipeline
2. Target module is pipe.transformer instead of pipe.unet
3. LoRA target modules are MMDiT attention layers
4. T5-XXL text encoder is EXCLUDED (only CLIP-L + OpenCLIP-G)
5. Training resolution matches SDXL at 512×512
6. Full WandB logging integration
7. Checkpoint saving matching SDXL
8. Uses DDPM scheduler for training compatibility
"""

import argparse
import os
import sys
import shutil
from typing import Any, Dict, List

# ============================================================
# FIX: Disable torchao in PEFT BEFORE importing peft
# ============================================================
os.environ["PEFT_DISABLE_TORCHAO"] = "1"

import torch
import wandb
from accelerate import Accelerator
from diffusers import AutoPipelineForText2Image, DDPMScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from data_utils import InfantEmotionDataset
from stable_diffusion_35.config import CONFIG


class DreamBoothDataset(torch.utils.data.Dataset):
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


def check_tensor(tensor: torch.Tensor, name: str, step: int) -> bool:
    """Debug helper to check for NaN/Inf in tensors."""
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN in {name} at step {step}!")
        return True
    if torch.isinf(tensor).any():
        print(f"⚠️ Inf in {name} at step {step}!")
        return True
    return False


def save_checkpoint(transformer, optimizer, global_step, output_dir, wandb_run=None):
    """Save checkpoint locally."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    unwrapped = transformer
    if hasattr(transformer, 'unwrap'):
        unwrapped = transformer.unwrap()
    elif hasattr(transformer, 'module'):
        unwrapped = transformer.module
    
    unwrapped.save_pretrained(checkpoint_dir)
    
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }, os.path.join(checkpoint_dir, "optimizer.pt"))
    
    print(f"Checkpoint saved at step {global_step}: {checkpoint_dir}")
    return checkpoint_dir


def cleanup_old_checkpoints(output_dir, keep_last=3):
    """Keep only the last N checkpoints."""
    checkpoint_dirs = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, item)):
            step = int(item.split("-")[1])
            checkpoint_dirs.append((step, item))
    
    checkpoint_dirs.sort(key=lambda x: x[0])
    
    if len(checkpoint_dirs) > keep_last:
        for step, dir_name in checkpoint_dirs[:-keep_last]:
            dir_path = os.path.join(output_dir, dir_name)
            shutil.rmtree(dir_path)
            print(f"Removed old checkpoint: {dir_name}")


def log_sample_images(pipe, transformer, device, step, emotion_prompts, output_dir):
    """Generate and log sample images to WandB."""
    os.makedirs(f"{output_dir}/samples", exist_ok=True)

    sample_images = []
    
    # Handle wrapped transformer
    model_to_eval = transformer
    if hasattr(transformer, 'unwrap'):
        model_to_eval = transformer.unwrap()
    elif hasattr(transformer, 'module'):
        model_to_eval = transformer.module
    
    model_to_eval.eval()
    with torch.no_grad():
        for emotion, prompt in emotion_prompts.items():
            generator = torch.Generator(device).manual_seed(42)
            result = pipe(
                prompt=prompt,
                negative_prompt="cartoon, drawing, blurry, low quality, distorted, deformed",
                num_inference_steps=30,
                guidance_scale=7.0,
                generator=generator,
                height=512,
                width=512,
            )
            img = result.images[0]
            save_path = f"{output_dir}/samples/step_{step}_{emotion}.png"
            img.save(save_path)
            sample_images.append(wandb.Image(save_path, caption=f"{emotion} baby"))
    
    model_to_eval.train()
    wandb.log({f"samples/step_{step}": sample_images})


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
            "dataset_size": 1200,
            "emotions": ["angry", "crying", "happy"],
        }
    )

    print("📊 WandB initialized!")
    print(f"   Project: {args.wandb_project}")
    print(f"   Run: {args.wandb_run_name}")
    print(f"   View at: https://wandb.ai/{wandb.run.entity}/{args.wandb_project}/runs/{wandb.run.id}")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=CONFIG["mixed_precision"],
    )
    device = accelerator.device

    print("Loading SD 3.5 Medium model (T5-XXL excluded)...")

    token = args.token or True
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        token=token,
        tokenizer_3=None,
        text_encoder_3=None,
    )
    pipe.to(device)

    print("Switching to DDPM scheduler for training compatibility...")
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    print("✅ Using DDPM scheduler for training")

    transformer = pipe.transformer

    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"📊 Total transformer parameters: {total_params:,}")

    # ============================================================
    # CRITICAL FIX: Use correct target modules for SD 3.5 Medium
    # ============================================================
    # For SD 3.5 Medium, the attention layers are named differently
    # We need to find all linear layers in attention blocks
    
    print("\n🔍 Finding attention modules...")
    
    # First, print all module names to debug
    print("Module names in transformer:")
    module_names = []
    for name, _ in transformer.named_modules():
        if any(x in name for x in ['attn', 'attention', 'q', 'k', 'v', 'proj']):
            module_names.append(name)
            if len(module_names) < 30:  # Print first 30
                print(f"  - {name}")
    
    # Common patterns for SD 3.5 Medium
    target_modules = [
        "attn.to_q",
        "attn.to_k", 
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.add_out_proj"
    ]
    
    # Also try to find any modules with these patterns
    found_modules = []
    for name, module in transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            for target in ["q", "k", "v", "out"]:
                if target in name.lower() and any(x in name.lower() for x in ["attn", "proj"]):
                    if name not in found_modules:
                        found_modules.append(name)
    
    if found_modules:
        print(f"Found modules: {found_modules[:10]}")
        target_modules = found_modules[:10]  # Use first 10 found modules
    
    print(f"Using target modules: {target_modules}")

    # LoRA config
    lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        modules_to_save=None,  # Don't save any extra modules
    )

    transformer = get_peft_model(transformer, lora_config)
    
    # CRITICAL: Print trainable parameters
    trainable_params = 0
    print("\n🔍 Trainable parameters:")
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  - {name}: {param.numel():,} parameters")
    
    print(f"\n📊 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    if trainable_params < 1000000:  # Less than 1M parameters
        print("⚠️ WARNING: Very few trainable parameters! This might indicate LoRA isn't properly applied.")
        print("   Expected: ~7-8 million parameters")

    wandb.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params,
    })

    # Freeze text encoders
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.text_encoder_2.requires_grad_(False)
    pipe.text_encoder_2.eval()

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

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # CRITICAL: Set model to training mode
    transformer.train()
    print("✅ Model set to training mode")

    # CRITICAL: Verify model is in training mode
    print(f"Model training mode: {transformer.training}")
    
    # Count trainable parameters after accelerator prepare
    trainable_params_after = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"Trainable parameters after prepare: {trainable_params_after:,}")

    # Prepare with accelerator
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )
    
    # CRITICAL: Verify model is still in training mode after prepare
    transformer.train()
    print(f"Model training mode after prepare: {transformer.training}")

    emotion_prompts = {
        "angry": "a photo of an angry sks infant",
        "crying": "a photo of a crying sks infant",
        "happy": "a photo of a happy sks infant",
    }

    print("Starting training...")
    print(f"Expected time: ~4-5 hours on A100")
    print(f"Each step should take ~8-12 seconds")
    
    global_step = 0
    running_loss = 0.0
    progress_bar = tqdm(range(args.max_train_steps))
    
    # Track time per step
    step_times = []

    wandb.log({"status": "training_started"})
    log_sample_images(pipe, transformer, device, 0, emotion_prompts, args.output_dir)

    for batch in dataloader:
        step_start_time = time.time()
        
        images = batch["images"].to(device, dtype=torch.float16)
        prompts = batch["prompts"]

        if check_tensor(images, "images", global_step):
            optimizer.zero_grad()
            continue

        with torch.no_grad():
            # CLIP-L
            tokenized_prompts = pipe.tokenizer(
                prompts,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            text_encoder_output = pipe.text_encoder(
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
            tokenized_prompts_2 = pipe.tokenizer_2(
                prompts,
                padding="max_length",
                max_length=pipe.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            text_encoder_output_2 = pipe.text_encoder_2(
                tokenized_prompts_2,
                output_hidden_states=True,
                return_dict=True,
            )

            text_embeddings_2 = text_encoder_output_2.last_hidden_state
            
            if hasattr(text_encoder_output_2, 'pooler_output') and text_encoder_output_2.pooler_output is not None:
                pooled_projection_2 = text_encoder_output_2.pooler_output
            else:
                pooled_projection_2 = text_encoder_output_2.last_hidden_state.mean(dim=1)

            # Pooled projections: (batch, 2048)
            pooled_projections = torch.cat([pooled_projection_1, pooled_projection_2], dim=-1)
            
            # Encoder hidden states: pad to 4096
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
            latents = pipe.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215

            if check_tensor(latents, "latents", global_step):
                optimizer.zero_grad()
                continue

        timesteps = torch.randint(
            0,
            pipe.scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=device,
        ).long()

        noise = torch.randn_like(latents)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        noisy_latents = noisy_latents.to(device=device, dtype=torch.float16)
        timesteps = timesteps.to(device)
        text_embeddings = text_embeddings.to(device=device, dtype=torch.float16)
        pooled_projections = pooled_projections.to(device=device, dtype=torch.float16)

        # CRITICAL: Ensure model is in training mode for forward pass
        if not transformer.training:
            transformer.train()
            print("⚠️ Model was not in training mode, fixing...")

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

        loss = torch.nn.functional.mse_loss(
            noise_pred.float(), 
            noise.float(), 
            reduction="mean"
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf loss at step {global_step}! Skipping...")
            optimizer.zero_grad()
            continue

        loss = loss / args.gradient_accumulation_steps

        # CRITICAL: Check if gradients are being computed
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
        
        # Track step time
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

        if global_step % args.checkpoint_steps == 0:
            unwrapped_unet = accelerator.unwrap_model(transformer)
            save_checkpoint(
                unwrapped_unet,
                optimizer,
                global_step,
                args.output_dir,
                wandb
            )
            cleanup_old_checkpoints(args.output_dir, keep_last=3)
            log_sample_images(pipe, transformer, device, global_step, emotion_prompts, args.output_dir)

        if global_step >= args.max_train_steps:
            break

    log_sample_images(pipe, transformer, device, args.max_train_steps, emotion_prompts, args.output_dir)

    print("Saving final model...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "mmdit_lora_final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(final_dir)

        print(f"Training complete! Final model saved to {final_dir}")
        
        # Print final stats
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