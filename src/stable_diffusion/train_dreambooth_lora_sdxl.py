"""
Training script for SDXL with DreamBooth and LoRA on infant emotion dataset.
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import torch
import wandb
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules (pylint: disable=wrong-import-position)
from data_utils import InfantEmotionDataset
from save_utils import save_checkpoint_local


class DreamBoothDataset(torch.utils.data.Dataset):
    """Dataset wrapper for DreamBooth training with instance prompts."""

    def __init__(self, base_dataset: InfantEmotionDataset, instance_prompt_template: str) -> None:
        self.base_dataset = base_dataset
        self.instance_prompt_template = instance_prompt_template

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.base_dataset[index]
        # Use positional formatting if template has {}
        prompt = self.instance_prompt_template.format(item["emotion"])
        return {
            "image": item["image"],
            "prompt": prompt,
            "emotion": item["emotion"],
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


def _upload_to_huggingface(final_dir: str, hf_repo: str) -> None:
    """
    Upload model to Hugging Face.

    Args:
        final_dir: Directory containing model weights
        hf_repo: Hugging Face repository name
    """
    try:
        # pylint: disable=import-outside-toplevel
        from huggingface_hub import upload_folder
        print(f"📤 Uploading to Hugging Face: {hf_repo}")
        upload_folder(
            folder_path=final_dir,
            repo_id=hf_repo,
            repo_type="model",
            path_in_repo=".",
        )
        print("✅ Model uploaded to Hugging Face!")
    except ImportError as error:
        print(f"⚠️ Could not import huggingface_hub: {error}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SDXL with DreamBooth and LoRA on infant emotion dataset"
    )
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
    # In the parse_args() function, change the default template
    parser.add_argument(
        "--instance_prompt_template",
        type=str,
        default="a photo of a {emotion} sks infant",  # Changed {} to {emotion}
        help="Template for instance prompts"
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a photo of an infant",
        help="Class prompt for prior preservation"
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
        "--checkpoint_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
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
        help="Run in Colab mode with Drive saving"
    )
    return parser.parse_args()


def _load_models(args: argparse.Namespace, device: torch.device):
    """
    Load all models for training.

    Args:
        args: Command line arguments
        device: Torch device

    Returns:
        Tuple of (vae, unet, text_encoder, tokenizer, noise_scheduler)
    """
    print("Loading models...")

    # Load VAE (frozen)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    ).to(device)
    vae.requires_grad_(False)

    # Load UNet (with LoRA)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    ).to(device)

    # Apply LoRA to UNet
    # Note: task_type is not needed for UNet in newer PEFT versions
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
        ],
        lora_dropout=0.1,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Load Text Encoder (frozen)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    ).to(device)
    text_encoder.requires_grad_(False)

    # Load Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    # Load Noise Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    return vae, unet, text_encoder, tokenizer, noise_scheduler


def _setup_dataloader(args: argparse.Namespace) -> DataLoader:
    """
    Set up the dataloader.

    Args:
        args: Command line arguments

    Returns:
        DataLoader for training
    """
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

    return DataLoader(
        dream_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def _run_training_loop(
    args: argparse.Namespace,
    accelerator: Accelerator,
    unet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
) -> None:
    """
    Run the main training loop.

    Args:
        args: Command line arguments
        accelerator: Accelerator for distributed training
        unet: UNet model with LoRA
        optimizer: Optimizer
        dataloader: Training dataloader
        text_encoder: CLIP text encoder
        tokenizer: CLIP tokenizer
        vae: VAE model
        noise_scheduler: Noise scheduler
        device: Torch device
    """
    print("Starting training...")
    global_step = 0
    running_loss = 0.0
    progress_bar = tqdm(range(args.max_train_steps))

    for _step in progress_bar:
        for batch in dataloader:
            images = batch["images"].to(device, dtype=torch.float16)
            prompts = batch["prompts"]

            tokenized_prompts = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            with torch.no_grad():
                text_embeddings = text_encoder(tokenized_prompts)[0]

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (images.shape[0],),
                device=device,
            ).long()

            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            accelerator.backward(loss)
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

            if global_step % args.checkpoint_steps == 0:
                unwrapped_unet = accelerator.unwrap_model(unet)
                save_checkpoint_local(
                    unwrapped_unet,
                    optimizer,
                    global_step,
                    args.output_dir,
                    wandb
                )

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break
# pylint: enable=too-many-arguments, too-many-positional-arguments, too-many-locals


def _save_final_model(
    args: argparse.Namespace,
    accelerator: Accelerator,
    unet: torch.nn.Module,
) -> None:
    """
    Save final model and upload to Hugging Face.

    Args:
        args: Command line arguments
        accelerator: Accelerator for distributed training
        unet: UNet model with LoRA
    """
    print("Saving final model...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "unet_lora_final")
        os.makedirs(final_dir, exist_ok=True)

        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(final_dir)

        artifact = wandb.Artifact(
            name="unet_lora_final",
            type="model",
            description="Final UNet LoRA weights after full training",
        )
        artifact.add_dir(final_dir)
        wandb.log_artifact(artifact)

        # Save to Hugging Face if in Colab mode
        if args.colab:
            _upload_to_huggingface(final_dir, args.hf_repo)

        print(f"Training complete! Final model saved to {final_dir}")


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Set seed
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

    # Load models
    vae, unet, text_encoder, tokenizer, noise_scheduler = _load_models(args, device)

    # Setup dataloader
    dataloader = _setup_dataloader(args)

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
    )

    # Prepare with accelerator
    unet, optimizer, dataloader = accelerator.prepare(
        unet, optimizer, dataloader
    )

    # Run training loop
    _run_training_loop(
        args=args,
        accelerator=accelerator,
        unet=unet,
        optimizer=optimizer,
        dataloader=dataloader,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        noise_scheduler=noise_scheduler,
        device=device,
    )

    # Save final model
    _save_final_model(args, accelerator, unet)

    wandb.finish()


if __name__ == "__main__":
    main()
