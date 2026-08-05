"""
Training script for SD 3.5 Medium using Flow Matching (native training objective).
Mirrors the SDXL training setup for fair comparison.

COMPARABLE TO SDXL:
- Same hyperparameters (lr=5e-6, batch_size=1, grad_accum=4, steps=1500)
- Same dataset (1200 images, 400 per emotion)
- Same LoRA rank (16)
- Same prompt template ("a photo of a {} sks infant")
- Same loss function (MSE)
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import torch
import wandb
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, AutoencoderKL
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from data_utils import InfantEmotionDataset


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
    """Parse command line arguments - matches SDXL setup."""
    parser = argparse.ArgumentParser(description="Train SD 3.5 Medium with LoRA (Flow Matching)")
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
        help="Template for instance prompts (same as SDXL)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution (same as SDXL)"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Training batch size (same as SDXL)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (same as SDXL)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate (same as SDXL)"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1500,
        help="Maximum training steps (same as SDXL)"
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
        help="Random seed (same as SDXL)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="infant-emotion-generation",
        help="WandB project name (same as SDXL)"
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
        default="sd35-flow-matching",
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
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    return parser.parse_args()


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
) -> torch.Tensor:
    """Compute density for timestep sampling (Flow Matching)."""
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,))
        return torch.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,))
        return 1 - u ** (1 / mode_scale)
    else:
        return torch.rand(size=(batch_size,))


def check_tensor(tensor: torch.Tensor, name: str, step: int) -> bool:
    """Check for NaN/Inf in tensors."""
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN in {name} at step {step}!")
        return True
    if torch.isinf(tensor).any():
        print(f"⚠️ Inf in {name} at step {step}!")
        return True
    return False


def save_checkpoint(transformer, optimizer, global_step, output_dir, wandb_run=None):
    """Save checkpoint locally and log to wandb."""
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
    import shutil
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
            "model": "SD3.5-Medium",
            "architecture": "MMDiT",
            "training_objective": "Flow Matching",
            "resolution": args.resolution,
            "learning_rate": args.learning_rate,
            "max_train_steps": args.max_train_steps,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lora_rank": 16,
            "dataset_size": 1200,
            "emotions": ["angry", "crying", "happy"],
            "weighting_scheme": "logit_normal",
        }
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )
    device = accelerator.device

    print("=" * 60)
    print("SD 3.5 Medium Training (Flow Matching)")
    print("Comparable to SDXL Training")
    print("=" * 60)
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Max steps: {args.max_train_steps}")
    print("=" * 60)

    print("Loading SD 3.5 Medium models...")

    # ============================================================
    # Load VAE (FP32 for stability - same as SDXL)
    # ============================================================
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        token=args.token or True,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # ============================================================
    # Load MMDiT Transformer (FP16 for memory)
    # ============================================================
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float16,
        token=args.token or True,
    ).to(device)

    # ============================================================
    # Apply LoRA to MMDiT (rank=16, same as SDXL)
    # ============================================================
    lora_config = LoraConfig(
        r=16,  # Same rank as SDXL
        lora_alpha=16,  # Same alpha as SDXL
        target_modules=[
            "to_q",
            "to_k", 
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ],
        lora_dropout=0.1,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # ============================================================
    # Load Text Encoders and Tokenizers
    # ============================================================
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

    # Freeze text encoders (same as SDXL)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    text_encoder_2.requires_grad_(False)
    text_encoder_2.eval()

    # ============================================================
    # Use FlowMatchEulerDiscreteScheduler (native to SD3.5)
    # ============================================================
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        token=args.token or True,
    )

    # ============================================================
    # Load Dataset (same dataset as SDXL)
    # ============================================================
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

    # ============================================================
    # Setup Optimizer (AdamW, same as SDXL)
    # ============================================================
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # ============================================================
    # CRITICAL: Prepare with accelerator but be explicit about training
    # ============================================================
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )
    
    # Ensure training mode
    transformer.train()
    
    # Verify training mode
    print(f"Transformer training mode: {transformer.training}")
    print(f"Gradients enabled: {torch.is_grad_enabled()}")

    print("Starting training...")
    print(f"Expected time: ~4-5 hours on A100")
    print(f"Each step should take ~8-12 seconds")
    
    global_step = 0
    running_loss = 0.0
    progress_bar = tqdm(range(args.max_train_steps))

    # ============================================================
    # Training Loop (Flow Matching - native to SD3.5)
    # ============================================================
    data_iterator = iter(dataloader)
    
    for step_idx in range(args.max_train_steps):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch = next(data_iterator)
        
        # Move data to device
        images = batch["images"].to(device, dtype=torch.float16)
        prompts = batch["prompts"]

        # ----- Encode text prompts (no gradients) -----
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

            # Concatenate pooled projections for MMDiT
            pooled_projections = torch.cat([pooled_projection_1, pooled_projection_2], dim=-1)
            
            # Pad text embeddings to 4096 (T5-XXL excluded)
            t5_dim = 4096 - 768 - 1280
            zero_padding = torch.zeros(
                text_embeddings_1.shape[0], 
                text_embeddings_1.shape[1], 
                t5_dim,
                device=text_embeddings_1.device,
                dtype=text_embeddings_1.dtype
            )
            text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2, zero_padding], dim=-1)

        # ----- Encode images to latents (no gradients) -----
        with torch.no_grad():
            latents = vae.encode(images.float()).latent_dist.sample()
            latents = latents * 0.18215

        # ----- Flow Matching: Sample timestep with density -----
        timesteps = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=images.shape[0],
        ).to(device)

        # ----- Sample noise and apply flow matching -----
        noise = torch.randn_like(latents)
        
        # For flow matching, we interpolate between noise and latents
        timesteps = timesteps.view(-1, 1, 1, 1)
        noisy_latents = (1.0 - timesteps) * latents + timesteps * noise

        # Move to device with correct dtype
        noisy_latents = noisy_latents.to(device=device, dtype=torch.float16)
        timesteps_flat = timesteps.view(-1).to(device)
        text_embeddings = text_embeddings.to(device=device, dtype=torch.float16)
        pooled_projections = pooled_projections.to(device=device, dtype=torch.float16)

        # Ensure transformer is in training mode
        if not transformer.training:
            transformer.train()
            print(f"⚠️ Step {global_step}: Transformer was not in training mode, fixed!")

        # ----- Forward pass (WITH gradients) -----
        # The model predicts the "velocity" (noise - latents) in flow matching
        predicted_velocity = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps_flat,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled_projections,
            return_dict=False,
        )[0]

        # Target: noise - latents (the flow direction)
        target_velocity = noise - latents
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(
            predicted_velocity.float(), target_velocity.float(), reduction="mean"
        )

        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps

        # ----- Backward pass -----
        accelerator.backward(loss)
        
        # Step optimizer when we've accumulated enough gradients
        if (step_idx + 1) % args.gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1
        
        # Track loss (unscaled)
        unscaled_loss = loss.item() * args.gradient_accumulation_steps
        running_loss += unscaled_loss
        avg_loss = running_loss / global_step

        wandb.log({
            "train/loss": unscaled_loss,
            "train/avg_loss": avg_loss,
            "train/global_step": global_step,
            "train/learning_rate": optimizer.param_groups[0]['lr'],
        })

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": unscaled_loss, "avg_loss": avg_loss})

        # Save checkpoint
        if global_step % args.checkpoint_steps == 0:
            unwrapped = accelerator.unwrap_model(transformer)
            save_checkpoint(
                unwrapped,
                optimizer,
                global_step,
                args.output_dir,
                wandb
            )
            cleanup_old_checkpoints(args.output_dir, keep_last=3)

    # ============================================================
    # Save Final Model
    # ============================================================
    print("Saving final model...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "mmdit_lora_final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        unwrapped_transformer.save_pretrained(final_dir)
        print(f"Training complete! Final model saved to {final_dir}")
        
        # Log to wandb
        artifact = wandb.Artifact(
            name="sd35_lora_weights",
            type="model",
            description=f"SD 3.5 Medium LoRA weights after {args.max_train_steps} steps",
        )
        artifact.add_dir(final_dir)
        wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()