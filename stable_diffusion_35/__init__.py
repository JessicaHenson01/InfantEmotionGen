"""
Stable Diffusion 3.5 Medium module for infant emotion generation.
T5-XXL is excluded (only CLIP-L + OpenCLIP-G).
Full WandB logging integration.
"""

from .train_dreambooth_lora_sd35 import main as train
from .generate_infant_faces_sd35 import main as generate

__all__ = ["train", "generate"]