"""
Configuration for Stable Diffusion 3.5 Medium fine-tuning.
T5-XXL is EXCLUDED (only CLIP-L + OpenCLIP-G).
Training resolution matches SDXL at 512×512.
"""

CONFIG = {
    # Model - T5-XXL is excluded for fair parameter comparison
    "model_id": "stabilityai/stable-diffusion-3.5-medium",
    "resolution": 512,  # Matches SDXL training resolution
    "inference_resolution": 1024,  # Generation can be higher

    # Text encoders - only CLIP-L + OpenCLIP-G (T5-XXL excluded)
    "use_t5": False,
    "text_encoder_1": "clip-l",
    "text_encoder_2": "openclip-g",

    # Training (matching SDXL setup except learning rate)
    "learning_rate": 1e-4,  # Explicitly set to match notebook override
    "max_train_steps": 1500,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "mixed_precision": "fp16",
    "gradient_checkpointing": True,

    # LoRA (matching SDXL rank)
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "train_text_encoder": False,

    # Checkpoint saving (matching your notebook args)
    "checkpoint_steps": 500,  # Save checkpoint every 500 steps
    "save_checkpoints": True,
    "keep_checkpoints": 3,  # Keep only the last 3 checkpoints

    # WandB logging
    "wandb_project": "infant-emotion-generation",
    "wandb_run_name": "sd35-timestep-fix",  # Matches your notebook run name

    # Output
    "output_dir": "./models/infant_lora_sd35",

    # Dataset
    "instance_prompt_template": "a photo of a {} sks infant",
    "class_prompt": "a photo of an infant",
}
