# Interim Experimental Progress Report

## Project Overview

Our final project investigates whether generative models can produce realistic infant faces with controllable facial expressions. The project compares a conditional StyleGAN2-ADA model with a fine-tuned Stable Diffusion XL model. The current dataset contains three emotion labels, and the final evaluation will compare image realism, diversity, and emotion consistency across the two approaches.

# Fine-tuned Stable Diffusion XL model

## Baseline Setup

To-Do

## Current Implementation Status

To-Do

## Next Steps

To-Do

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
