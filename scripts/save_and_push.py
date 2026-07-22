import os
import subprocess
from huggingface_hub import HfApi, upload_folder, create_repo
import wandb


def save_and_push(
    model_dir="./models/infant_lora",
    images_dir="./generated_images/sdxl",
    hf_repo="your_username/infant-lora-model",
    github_repo=".",
    wandb_project="infant-emotion-generation"
):
    """
    Save all outputs to Hugging Face and GitHub.
    """
    api = HfApi()
    
    # Ensure HF repo exists
    try:
        create_repo(hf_repo, exist_ok=True, repo_type="model")
        print(f"Using Hugging Face repo: {hf_repo}")
    except Exception as e:
        print(f"Repo creation error: {e}")
    
    # Upload model
    print(f"Uploading model to Hugging Face: {hf_repo}")
    upload_folder(
        folder_path=model_dir,
        repo_id=hf_repo,
        path_in_repo=".",
    )
    
    # Upload images if they exist
    if os.path.exists(images_dir):
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
    
    # Log to wandb
    wandb.init(project=wandb_project)
    artifact = wandb.Artifact(
        name="infant-lora-model",
        type="model",
        description="LoRA weights for infant emotion generation"
    )
    artifact.add_dir(model_dir)
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    save_and_push()