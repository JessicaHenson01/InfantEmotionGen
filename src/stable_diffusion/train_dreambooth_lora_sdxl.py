import torch
import argparse
import os
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import wandb

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import InfantEmotionDataset
from .save_utils import save_checkpoint_local, save_all_outputs


class DreamBoothDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, instance_prompt_template):
        self.base_dataset = base_dataset
        self.instance_prompt_template = instance_prompt_template
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, index):
        item = self.base_dataset[index]
        prompt = self.instance_prompt_template.format(emotion=item["emotion"])
        return {
            "image": item["image"],
            "prompt": prompt,
            "emotion": item["emotion"],
        }


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    emotions = [item["emotion"] for item in batch]
    return {
        "images": images,
        "prompts": prompts,
        "emotions": emotions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--data_dir", type=str, default="./data/baby_emotion_samples")
    parser.add_argument("--json_path", type=str, default="./data/labels.json")
    parser.add_argument("--instance_prompt_template", type=str, default="a photo of a {} sks infant")
    parser.add_argument("--class_prompt", type=str, default="a photo of an infant")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=1500)
    parser.add_argument("--output_dir", type=str, default="./models/infant_lora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_steps", type=int, default=500)
    parser.add_argument("--wandb_project", type=str, default="infant-emotion-generation")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--hf_repo", type=str, default="your_username/infant-lora-model")
    parser.add_argument("--colab", action="store_true", help="Run in Colab mode with Drive saving")
    args = parser.parse_args()

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
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="UNET",
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

    for step in progress_bar:
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

    # Save final model
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
        
        # Save to Hugging Face and GitHub if in Colab mode
        if args.colab:
            save_all_outputs(
                model_dir=final_dir,
                images_dir="./generated_images/sdxl",
                hf_repo=args.hf_repo,
            )
        
        print(f"Training complete! Final model saved to {final_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()