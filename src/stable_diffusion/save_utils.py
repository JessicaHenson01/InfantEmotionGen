"""
Utility functions for saving model checkpoints and outputs.
"""

import os
import shutil
import subprocess
from typing import Optional, Any

import torch
import wandb

# pylint: disable=import-error
from huggingface_hub import HfApi, create_repo, upload_folder
# pylint: enable=import-error


def save_checkpoint_to_hub(local_path: str, repo_name: str, checkpoint_name: str) -> None:
    """
    Upload checkpoint to Hugging Face Hub.

    Args:
        local_path: Local path to checkpoint
        repo_name: Hugging Face repository name
        checkpoint_name: Name for the checkpoint in the repo
    """
    api = HfApi()

    try:
        create_repo(repo_name, exist_ok=True, repo_type="model")
    except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"Repo creation error: {error}")

    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_name,
        repo_type="model",
        path_in_repo=checkpoint_name,
    )
    print(f"Uploaded {checkpoint_name} to {repo_name}")


def save_images_to_github(local_dir: str, repo_dir: str = "generated_images") -> None:
    """
    Copy generated images to GitHub repo directory.

    Args:
        local_dir: Local directory with generated images
        repo_dir: Target directory in GitHub repo
    """
    target_dir = os.path.join("/content/InfantEmotionGen", repo_dir)
    os.makedirs(target_dir, exist_ok=True)

    emotions = ["angry", "crying", "happy"]
    for emotion in emotions:
        src_dir = os.path.join(local_dir, emotion)
        dst_dir = os.path.join(target_dir, emotion)
        if os.path.exists(src_dir):
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            print(f"Copied {emotion} images to {dst_dir}")


def push_to_github() -> None:
    """Commit and push changes to GitHub."""
    os.chdir("/content/InfantEmotionGen")
    subprocess.run(["git", "add", "."], capture_output=True, check=False)
    subprocess.run(
        ["git", "commit", "-m", "Update model outputs and generated images"],
        capture_output=True,
        check=False
    )
    subprocess.run(["git", "push", "origin", "main"], capture_output=True, check=False)
    print("Pushed to GitHub!")


def save_all_outputs(
    model_dir: str = "./models/infant_lora",
    images_dir: str = "./generated_images/sdxl",
    hf_repo: str = "InfantEmotionGen",
    github_repo: str = "/content/InfantEmotionGen"
) -> None:
    """
    Save all outputs to Hugging Face and GitHub.

    Args:
        model_dir: Directory containing model weights
        images_dir: Directory containing generated images
        hf_repo: Hugging Face repository name
        github_repo: Local path to GitHub repository
    """
    # Ensure HF repo exists
    try:
        create_repo(hf_repo, exist_ok=True, repo_type="model")
    except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"Repo creation error: {error}")
        return

    # Upload model
    print(f"Uploading model to Hugging Face: {hf_repo}")
    upload_folder(
        folder_path=model_dir,
        repo_id=hf_repo,
        repo_type="model",
        path_in_repo=".",
    )

    # Upload images
    print(f"Uploading images to Hugging Face: {hf_repo}")
    upload_folder(
        folder_path=images_dir,
        repo_id=hf_repo,
        repo_type="model",
        path_in_repo="generated_images",
    )

    # Push to GitHub
    print("Pushing to GitHub...")
    os.chdir(github_repo)
    subprocess.run(["git", "add", "."], capture_output=True, check=False)
    subprocess.run(
        ["git", "commit", "-m", "Update model outputs and images"],
        capture_output=True,
        check=False
    )
    subprocess.run(["git", "push", "origin", "main"], capture_output=True, check=False)

    print("All saved!")


def save_checkpoint_local(
    unet: Any,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    output_dir: str,
    wandb_run: Optional[Any] = None
) -> str:
    """
    Save checkpoint locally and log to wandb.

    Args:
        unet: UNet model with LoRA
        optimizer: Optimizer
        global_step: Current training step
        output_dir: Directory to save checkpoint
        wandb_run: Optional wandb run for logging

    Returns:
        Path to the saved checkpoint
    """
    # Save locally
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    unet.save_pretrained(checkpoint_dir)

    # Save optimizer state
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }, os.path.join(checkpoint_dir, "optimizer.pt"))

    # Log to wandb as artifact
    if wandb_run:
        artifact = wandb.Artifact(
            name=f"unet_lora_checkpoint_{global_step}",
            type="model",
            description=f"UNet LoRA checkpoint at step {global_step}",
        )
        artifact.add_dir(checkpoint_dir)
        wandb_run.log_artifact(artifact)

    print(f"Checkpoint saved at step {global_step}: {checkpoint_dir}")
    return checkpoint_dir
