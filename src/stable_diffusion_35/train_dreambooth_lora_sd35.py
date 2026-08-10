"""
SD 3.5 Medium training script using Flow Matching and LoRA.

This script fine-tunes a pretrained Stable Diffusion 3.5 Medium model on an infant
facial expression dataset using DreamBooth and Low-Rank Adaptation (LoRA).
The model is trained using a flow matching objective with logit-normal sigma sampling,
matching the official SD3.5 training recipe. T5-XXL is excluded to maintain a fair
parameter count comparison with SDXL.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import wandb
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, AutoencoderKL
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import InfantEmotionDataset


class DreamBoothDataset(torch.utils.data.Dataset):
    """Dataset wrapper that appends emotion-specific DreamBooth prompts to each image."""

    def __init__(self, base_dataset: InfantEmotionDataset, instance_prompt_template: str) -> None:
        self.base_dataset = base_dataset
        self.instance_prompt_template = instance_prompt_template

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.base_dataset[index]
        prompt = self.instance_prompt_template.format(item["emotion"])
        return {
            "image": item["image"],
            "prompt": prompt,
            "emotion": item["emotion"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of samples into stacked tensors and prompt lists."""
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "prompts": [item["prompt"] for item in batch],
        "emotions": [item["emotion"] for item in batch],
    }


@torch.no_grad()
def generate_preview_image(
    prompt: str,
    transformer,
    vae,
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    device,
    resolution: int,
    vae_scaling_factor: float,
    vae_shift_factor: float,
    num_inference_steps: int = 15,
    seed: int = 0,
    num_train_timesteps: int = 1000,
):
    """
    Generate a quick in training preview image for WandB logging.

    This uses a short Euler integration of the learned velocity field to produce
    a low fidelity preview. It mirrors the same text encoding and flow matching
    logic used during training, allowing visual inspection of progress without
    waiting for a full inference run.
    """
    was_training = transformer.training
    transformer.eval()

    # Encode text prompts using both CLIP encoders.
    tokenized = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    out1 = text_encoder(tokenized, output_hidden_states=True, return_dict=True)
    emb1 = out1.last_hidden_state
    pooled1 = out1.pooler_output if getattr(out1, "pooler_output", None) is not None else emb1.mean(dim=1)

    tokenized_2 = tokenizer_2(
        [prompt],
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    out2 = text_encoder_2(tokenized_2, output_hidden_states=True, return_dict=True)
    emb2 = out2.last_hidden_state
    pooled2 = out2.pooler_output if getattr(out2, "pooler_output", None) is not None else emb2.mean(dim=1)

    # Concatenate pooled projections and pad to 4096 dimensions.
    pooled_projections = torch.cat([pooled1, pooled2], dim=-1).to(dtype=torch.float16)
    t5_dim = 4096 - 768 - 1280
    zero_padding = torch.zeros(
        emb1.shape[0],
        emb1.shape[1],
        t5_dim,
        device=device,
        dtype=emb1.dtype
    )
    text_embeddings = torch.cat([emb1, emb2, zero_padding], dim=-1).to(dtype=torch.float16)

    # Initialize latent noise and run Euler integration.
    latent_size = resolution // 8
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (1, transformer.config.in_channels, latent_size, latent_size),
        device=device, dtype=torch.float16, generator=generator,
    )

    dt = 1.0 / num_inference_steps
    for step in range(num_inference_steps):
        t_val = 1.0 - step * dt
        model_t = t_val * num_train_timesteps
        timestep = torch.full((1,), model_t, device=device, dtype=torch.float16)
        velocity = transformer(
            hidden_states=x,
            timestep=timestep,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled_projections,
            return_dict=False,
        )[0]
        x = x - velocity * dt

    # Undo SD3.5 latent normalization and decode through VAE.
    latents = (x.float() / vae_scaling_factor) + vae_shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

    if was_training:
        transformer.train()

    from PIL import Image
    return Image.fromarray(image)


def check_tensor(tensor: torch.Tensor, name: str, step: int) -> bool:
    """
    Check a tensor for NaN or Inf values to catch training instability early.
    """
    if torch.isnan(tensor).any():
        print(f"Warning: NaN in {name} at step {step}!")
        return True
    if torch.isinf(tensor).any():
        print(f"Warning: Inf in {name} at step {step}!")
        return True
    return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium"
    )
    parser.add_argument("--data_dir", type=str, default="./data/baby_emotion_samples")
    parser.add_argument("--json_path", type=str, default="./data/labels_formatted.json")
    parser.add_argument(
        "--instance_prompt_template",
        type=str,
        default="a photo of a {} sks infant"
    )
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
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=250,
        help="Log a quick preview image to WandB every N steps (0 to disable)"
    )
    parser.add_argument(
        "--sample_inference_steps",
        type=int,
        default=15,
        help="Number of Euler steps used for the quick in-training preview sampler"
    )
    return parser.parse_args()


def _encode_text(
    prompts: List[str],
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode text prompts using CLIP-L and OpenCLIP-G.

    Returns:
        text_embeddings: Concatenated embeddings (padded to 4096 dims).
        pooled_projections: Pooled projections from both encoders.
    """
    with torch.no_grad():
        # CLIP-L
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        out1 = text_encoder(tokenized, output_hidden_states=True, return_dict=True)
        emb1 = out1.last_hidden_state
        pooled1 = out1.pooler_output if (hasattr(out1, 'pooler_output') and
                                         out1.pooler_output is not None) else emb1.mean(dim=1)

        # OpenCLIP-G
        tokenized_2 = tokenizer_2(
            prompts,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        out2 = text_encoder_2(tokenized_2, output_hidden_states=True, return_dict=True)
        emb2 = out2.last_hidden_state
        pooled2 = out2.pooler_output if (hasattr(out2, 'pooler_output') and
                                         out2.pooler_output is not None) else emb2.mean(dim=1)

        pooled_projections = torch.cat([pooled1, pooled2], dim=-1).to(dtype=torch.float16)
        t5_dim = 4096 - 768 - 1280
        zero_padding = torch.zeros(
            emb1.shape[0],
            emb1.shape[1],
            t5_dim,
            device=emb1.device,
            dtype=emb1.dtype
        )
        text_embeddings = torch.cat([emb1, emb2, zero_padding], dim=-1).to(dtype=torch.float16)

    return text_embeddings, pooled_projections


def _encode_images(images: torch.Tensor, vae, vae_shift_factor, vae_scaling_factor):
    """
    Encode images to latent space using the frozen VAE.

    Returns:
        latents: Normalized latents ready for flow matching.
    """
    with torch.no_grad():
        latents = vae.encode(images.float()).latent_dist.sample()
        latents = (latents - vae_shift_factor) * vae_scaling_factor
    return latents


def _train_loop(
    transformer, optimizer, dataloader, accelerator, vae, tokenizer, tokenizer_2,
    text_encoder, text_encoder_2, device, args, scheduler, vae_scaling_factor, vae_shift_factor
) -> None:
    """Main training loop containing the flow matching logic."""
    global_step = 0
    running_loss = 0.0
    progress_bar = tqdm(range(args.max_train_steps))
    data_iterator = iter(dataloader)

    for _ in range(args.max_train_steps):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch = next(data_iterator)

        images = batch["images"].to(device, dtype=torch.float16)
        prompts = batch["prompts"]

        if check_tensor(images, "images", global_step):
            optimizer.zero_grad()
            continue

        # Encode text
        text_embeddings, pooled_projections = _encode_text(
            prompts, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device
        )

        # Encode images
        latents = _encode_images(images, vae, vae_shift_factor, vae_scaling_factor)
        if check_tensor(latents, "latents", global_step):
            print("   VAE produced NaN! Skipping...")
            optimizer.zero_grad()
            continue

        # Flow matching
        u = torch.normal(mean=0.0, std=1.0, size=(images.shape[0],), device=device)
        sigmas = torch.sigmoid(u)
        sigmas = sigmas.clamp(0.0, 1.0)

        # Map sigma to model's timestep scale
        # Pylint falsely flags no-member on scheduler.config. It does exist.
        model_timesteps = sigmas * scheduler.config.num_train_timesteps  # pylint: disable=no-member
        sigmas_expanded = sigmas.view(-1, 1, 1, 1)

        noise = torch.randn_like(latents)
        noisy_latents = (1.0 - sigmas_expanded) * latents + sigmas_expanded * noise
        target_velocity = noise - latents

        noisy_latents = noisy_latents.to(dtype=torch.float16)
        model_timesteps = model_timesteps.to(dtype=torch.float16)
        text_embeddings = text_embeddings.to(dtype=torch.float16)
        pooled_projections = pooled_projections.to(dtype=torch.float16)

        # Forward pass
        predicted_velocity = transformer(
            hidden_states=noisy_latents,
            timestep=model_timesteps,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled_projections,
            return_dict=False,
        )[0]

        if check_tensor(predicted_velocity, "predicted_velocity", global_step):
            optimizer.zero_grad()
            continue

        # Compute loss
        loss = torch.nn.functional.mse_loss(
            predicted_velocity.float(), target_velocity.float(), reduction="mean"
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at step {global_step}! Skipping...")
            optimizer.zero_grad()
            continue

        # Backward pass
        with accelerator.accumulate(transformer):
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1
        running_loss += loss.item()
        avg_loss = running_loss / global_step

        wandb.log({
            "train/loss": loss.item(),
            "train/avg_loss": avg_loss,
            "train/global_step": global_step,
            "train/sigma_mean": sigmas.mean().item(),
            "train/model_timestep_mean": model_timesteps.mean().item(),
        })

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item(), "avg_loss": avg_loss})

        # Save checkpoint
        if global_step % args.checkpoint_steps == 0:
            unwrapped = accelerator.unwrap_model(transformer)
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved at step {global_step}")

        # Generate preview samples
        if args.sample_steps > 0 and global_step % args.sample_steps == 0 and accelerator.is_main_process:
            print(f"Logging preview samples at step {global_step}...")
            unwrapped = accelerator.unwrap_model(transformer)
            for emotion in ["angry", "crying", "happy"]:
                preview_prompt = args.instance_prompt_template.format(emotion)
                try:
                    preview_img = generate_preview_image(
                        prompt=preview_prompt,
                        transformer=unwrapped,
                        vae=vae,
                        tokenizer=tokenizer,
                        tokenizer_2=tokenizer_2,
                        text_encoder=text_encoder,
                        text_encoder_2=text_encoder_2,
                        device=device,
                        resolution=args.resolution,
                        vae_scaling_factor=vae_scaling_factor,
                        vae_shift_factor=vae_shift_factor,
                        num_inference_steps=args.sample_inference_steps,
                        seed=args.seed,
                        num_train_timesteps=scheduler.config.num_train_timesteps,
                    )
                    wandb.log({
                        f"preview/{emotion}": wandb.Image(
                            preview_img,
                            caption=f"{emotion} @ step {global_step}"
                        ),
                        "train/global_step": global_step,
                    })
                except (RuntimeError, ValueError) as preview_error:
                    print(f"Preview generation failed for '{emotion}': {preview_error}")


def main() -> None:
    """Main training loop for SD3.5 Medium using Flow Matching and LoRA."""
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

    print("SD 3.5 Medium Training (Flow Matching)")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Max steps: {args.max_train_steps}")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        token=args.token or True,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    vae_scaling_factor = vae.config.scaling_factor
    vae_shift_factor = vae.config.shift_factor
    print(f"VAE scaling_factor={vae_scaling_factor}, shift_factor={vae_shift_factor}")

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

    transformer.enable_gradient_checkpointing()

    # Load frozen text encoders
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
    print("Loading dataset using existing data_utils...")
    base_dataset = InfantEmotionDataset(
        data_dir=args.data_dir,
        json_path=args.json_path,
        size=args.resolution,
        center_crop=False,
    )

    if len(base_dataset) == 0:
        print("Error: No images found!")
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
    )

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )
    transformer.train()

    print("Starting training...")

    _train_loop(
        transformer, optimizer, dataloader, accelerator, vae, tokenizer, tokenizer_2,
        text_encoder, text_encoder_2, device, args, scheduler, vae_scaling_factor, vae_shift_factor
    )

    # Save final model
    print("Saving final model...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "mmdit_lora_final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(final_dir)
        print(f"Training complete! Model saved to {final_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
