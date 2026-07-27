# Interim Experimental Progress Report

## Project Overview

Our final project investigates whether generative models can produce realistic infant faces with controllable facial expressions. The project compares a conditional StyleGAN2-ADA model with a fine-tuned Stable Diffusion XL model. The current dataset contains three emotion labels, and the final evaluation will compare image realism, diversity, and emotion consistency across the two approaches.

# Fine-tuned Stable Diffusion XL Model

## Baseline Setup

We use the official Stability AI Stable Diffusion XL base model (`stabilityai/stable-diffusion-xl-base-1.0`) as our foundation. The model has approximately 2.6 billion parameters in the UNet backbone and was pretrained on large-scale image-text datasets including LAION-5B. For our task, we fine-tune the model using DreamBooth with Low-Rank Adaptation (LoRA) on our 1,200 infant facial expression images, which are labeled with three emotion classes: angry, crying, and happy.

The fine-tuning pipeline loads the pretrained SDXL model in FP16 precision, applies LoRA with rank 16 to the cross-attention layers (`attn2.to_q`, `attn2.to_k`, `attn2.to_v`, `attn2.to_out.0`) of the UNet, and freezes the VAE and text encoders. DreamBooth personalizes the model by binding a unique identifier token (`sks`) to the infant concept, while LoRA reduces the number of trainable parameters to approximately 12.6 million (0.49% of the total). The model is fine-tuned at a resolution of 512×512 for 1,500 steps with a batch size of 1, gradient accumulation of 4, and a learning rate of 5e-6. Training is accelerated using mixed precision and gradient checkpointing.

## Current Implementation Status

The SDXL fine-tuning pipeline is fully implemented and functional. We successfully trained the model on the 1,200 infant emotion images, achieving a final average loss of 0.67. The training run completed on an A100 GPU and took approximately 20 minutes. The trained LoRA weights were saved as a Hugging Face adapter (`adapter_model.safetensors`), enabling efficient model sharing and inference.

We have also implemented the generation pipeline, which loads the fine-tuned LoRA weights and generates 100 synthetic infant faces for each emotion class (angry, crying, happy) at 1024×1024 resolution. The generation pipeline uses the DPM++ 2M Karras scheduler and includes negative prompting to avoid common artifacts. The generated images have been saved locally and uploaded as a WandB artifact for review. The training configuration, hyperparameters, and results have been logged to Weights & Biases, providing full reproducibility of the experiment.

## Next Steps

The next step is to evaluate the generated SDXL images using our planned evaluation metrics. We will compute Fréchet Inception Distance (FID) to assess image realism and distributional similarity to real infant faces. CLIP Score will be computed to evaluate semantic alignment between each generated image and its corresponding text prompt. We will also apply a pretrained Facial Expression Recognition (FER) classifier to assess how accurately the generated images portray the intended expressions. These results will then be compared against the StyleGAN2-ADA baseline using the same emotion classes and sample counts. The generated images will also be saved to our Hugging Face repository for public access.

# StyleGAN2-ADA

## Baseline Setup

For the StyleGAN2 baseline, we are using NVIDIA’s StyleGAN2-ADA PyTorch implementation with transfer learning from a pretrained FFHQ face model. The training dataset contains 1,200 labeled infant-face images stored in a private Hugging Face dataset repository. The images are stored in the StyleGAN ZIP format with a `dataset.json` file containing the conditional labels. Horizontal mirroring increases the effective training set to 2,400 samples.

The current local baseline uses 256 × 256 images, conditional training, adaptive discriminator augmentation, horizontal mirroring, and the pretrained FFHQ-256 checkpoint. The model is trained with a batch size of one because of laptop memory constraints. Training statistics are stored locally and synchronized with Weights & Biases.

## Current Implementation Status

The original StyleGAN2-ADA implementation was designed for NVIDIA CUDA GPUs, so several compatibility changes were required to run it on an Apple Silicon laptop using PyTorch’s MPS backend. These changes included adding MPS device selection, disabling CUDA-specific timing and fused operations, using full-precision computation, replacing unsupported float64 statistics with float32 on MPS, and fixing compatibility issues with newer PyTorch versions.

Additional changes were made to support transfer learning from an unconditional FFHQ model into the new three-class conditional model. Compatible generator and discriminator weights are loaded from the pretrained checkpoint, while the newly introduced class-embedding parameters remain randomly initialized.

The complete local training pipeline is now functional. A 1-kimg smoke test successfully completed on MPS in approximately 19 minutes. The run completed generator and discriminator optimization, path-length regularization, R1 regularization, ADA setup, checkpoint creation, image generation, and W&B synchronization without errors. It produced initial and final model snapshots, generated-image grids, a real-image grid, TensorBoard logs, and training statistics.

## Next Steps

The next step is to run a longer StyleGAN2 experiment using the full dataset and compare checkpoints throughout training. Generated samples will be evaluated both qualitatively and quantitatively. Planned evaluation metrics include FID for realism, precision and recall for image quality and diversity, CLIP-based emotion alignment, and facial-expression classification accuracy and macro F1. We also plan to compare generated images with their nearest training-set neighbors to check for memorization.

The StyleGAN2 results will ultimately be compared with the SDXL generation pipeline using the same emotion classes, sample counts, and evaluation procedures.
