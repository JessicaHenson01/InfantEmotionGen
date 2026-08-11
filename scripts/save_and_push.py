"""Save model outputs, images, and push to Hugging Face and GitHub."""
import os
import subprocess

from huggingface_hub import create_repo, upload_folder
import wandb


def save_and_push(
    model_dir: str = "./models/infant_lora",
    images_dir: str = "./generated_images/sdxl",
    hf_repo: str = "your_username/infant-lora-model",
    github_repo: str = ".",
    wandb_project: str = "infant-emotion-generation"
) -> None:
    """
    Save all outputs to Hugging Face and GitHub.

    Args:
        model_dir: Directory containing the model files.
        images_dir: Directory containing generated images.
        hf_repo: Hugging Face repository ID.
        github_repo: Path to the GitHub repository.
        wandb_project: Weights & Biases project name.
    """
    # Ensure HF repo exists
    try:
        create_repo(hf_repo, exist_ok=True, repo_type="model")
        print(f"Using Hugging Face repo: {hf_repo}")
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Repo creation error: {e}")
        return

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
    original_dir = os.getcwd()
    try:
        os.chdir(github_repo)

        subprocess.run(["git", "add", "."], capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Update model outputs and images"],
            capture_output=True,
            check=False  # Allow commit to fail if nothing to commit
        )
        subprocess.run(["git", "push", "origin", "main"], capture_output=True, check=True)
        print("GitHub push completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Git operation failed: {e}")
        print(f"Error output: {e.stderr.decode() if e.stderr else 'No stderr'}")
    finally:
        os.chdir(original_dir)

    print("All saved!")

    # Log to wandb
    try:
        wandb.init(project=wandb_project)
        artifact = wandb.Artifact(
            name="infant-lora-model",
            type="model",
            description="LoRA weights for infant emotion generation"
        )
        artifact.add_dir(model_dir)
        wandb.log_artifact(artifact)
        wandb.finish()
        print("Weights & Biases logging completed!")
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"WandB logging failed: {e}")


if __name__ == "__main__":
    save_and_push()
