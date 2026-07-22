import os
import shutil
import subprocess
import torch
from huggingface_hub import HfApi, create_repo, upload_folder
import wandb


def save_checkpoint_to_hub(local_path, repo_name, checkpoint_name):
    """
    Upload checkpoint to Hugging Face Hub.
    """
    api = HfApi()
    
    # Ensure repo exists
    try:
        create_repo(repo_name, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Repo creation error: {e}")
    
    # Upload files to Hugging Face
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_name,
        repo_type="model",
        path_in_repo=checkpoint_name,
    )
    print(f"Uploaded {checkpoint_name} to {repo_name}")


def save_images_to_github(local_dir, repo_dir="generated_images"):
    """
    Copy generated images to GitHub repo directory.
    """
    target_dir = os.path.join("/content/infant_project/repo", repo_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy images
    for emotion in ["angry", "crying", "happy"]:
        src_dir = os.path.join(local_dir, emotion)
        dst_dir = os.path.join(target_dir, emotion)
        if os.path.exists(src_dir):
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            print(f"Copied {emotion} images to {dst_dir}")


def push_to_github():
    """
    Commit and push changes to GitHub.
    """
    os.chdir("/content/infant_project/repo")
    subprocess.run(["git", "add", "."], capture_output=True)
    subprocess.run(["git", "commit", "-m", "Update model outputs and generated images"], capture_output=True)
    subprocess.run(["git", "push", "origin", "main"], capture_output=True)
    print("Pushed to GitHub!")


def save_all_outputs(
    model_dir="./models/infant_lora",
    images_dir="./generated_images/sdxl",
    hf_repo="your_username/infant-lora-model",
    github_repo="/content/infant_project/repo"
):
    """
    Save all outputs to Hugging Face and GitHub.
    """
    api = HfApi()
    
    # Ensure HF repo exists
    try:
        create_repo(hf_repo, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Repo creation error: {e}")
    
    # Upload model
    print(f"Uploading model to Hugging Face: {hf_repo}")
    upload_folder(
        folder_path=model_dir,
        repo_id=hf_repo,
        path_in_repo=".",
    )
    
    # Upload images
    print(f"Uploading images to Hugging Face: {hf_repo}")
    upload_folder(
        folder_path=images_dir,
        repo_id=hf_repo,
        path_in_repo="generated_images",
    )
    
    # Push to GitHub
    print("Pushing to GitHub...")
    os.chdir(github_repo)
    subprocess.run(["git", "add", "."], capture_output=True)
    subprocess.run(["git", "commit", "-m", "Update model outputs and images"], capture_output=True)
    subprocess.run(["git", "push", "origin", "main"], capture_output=True)
    
    print("All saved!")


def save_checkpoint_local(unet, optimizer, global_step, output_dir, wandb_run=None):
    """
    Save checkpoint locally and log to wandb.
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