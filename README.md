# Synthetic Infant Facial Expression Generation Using SDXL and Stable Diffusion 3.5

This repository contains the code and configuration for fine-tuning two state-of-the-art diffusion models—**SDXL** and **Stable Diffusion 3.5 Medium**—for controlled infant facial expression generation. The models are fine-tuned using **DreamBooth** personalization and **Low-Rank Adaptation (LoRA)** on a curated dataset of infant faces with three emotion classes: `happy`, `angry`, and `crying`.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Training Pipelines](#training-pipelines)
  - [Train SDXL](#1-train-sdxl-unet-based)
  - [Train SD3.5 Medium](#2-train-sd35-medium-mmdit-based)
- [Generating Images](#generating-images)
  - [Generate with SDXL](#1-generate-with-sdxl)
  - [Generate with SD3.5 Medium](#2-generate-with-sd35-medium)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [License](#license)

---

## Project Structure

```text
InfantEmotionGen/
│
├── src/
│   ├── stable_diffusion/                 # SDXL training and inference scripts
│   │   ├── train_dreambooth_lora_sdxl.py
│   │   ├── generate_infant_faces.py
│   │   └── save_utils.py
│   │
│   └── stable_diffusion_35/              # SD3.5 Medium training and inference scripts
│       ├── train_dreambooth_lora_sd35.py
│       └── generate_infant_faces_sd35.py
│
├── data/
│   ├── baby_emotion_samples/             # Training images (1200 total, 400 per emotion)
│   └── labels_formatted.json             # JSON file mapping images to emotion labels
│
├── models/
│   ├── infant_lora/                      # SDXL LoRA weights (output)
│   └── infant_lora_sd35/                 # SD3.5 LoRA weights (output)
│
├── generated_images/
│   ├── sdxl/                             # SDXL generated images
│   └── sd35/                             # SD3.5 generated images
│
├── evaluation/                           # Evaluation scripts, configs, and outputs
│   └── generated/                        # Evaluation-generated images
│
├── configs.py                            # Shared hyperparameter configuration
├── .pylintrc                             # Pylint configuration
└── README.md                             # This file
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Diffusers 0.27+
- Accelerate 0.27+
- PEFT 0.8+
- WandB

Install dependencies via pip using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Training Pipelines

Both models are trained on the same dataset (1,200 images, 400 per emotion) using identical DreamBooth + LoRA configurations. The learning rate is adjusted per architecture to account for differences in the denoising backbones.

### 1. Train SDXL (UNet-based)

Run the following command from the **project root**:

```bash
python src/stable_diffusion/train_dreambooth_lora_sdxl.py \
  --data_dir ./data/baby_emotion_samples \
  --json_path ./data/labels_formatted.json \
  --output_dir ./models/infant_lora \
  --wandb_project "infant-emotion-generation" \
  --wandb_run_name "sdxl-training" \
  --instance_prompt_template "a photo of a {} sks infant" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-6 \
  --max_train_steps 1500 \
  --seed 42
```

**Expected training time:** ~2–4 hours on a single NVIDIA GPU (e.g., A100 or RTX 4090).
**Output:** LoRA adapter weights saved to `./models/infant_lora/unet_lora_final/`.

---

### 2. Train SD3.5 Medium (MMDiT-based)

Run the following command from the **project root**:

```bash
python src/stable_diffusion_35/train_dreambooth_lora_sd35.py \
  --data_dir ./data/baby_emotion_samples \
  --json_path ./data/labels_formatted.json \
  --output_dir ./models/infant_lora_sd35 \
  --wandb_project "infant-emotion-generation" \
  --wandb_run_name "sd35-training" \
  --instance_prompt_template "a photo of a {} sks infant" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_train_steps 1500 \
  --seed 42
```

> **Note:** SD3.5 Medium uses a higher learning rate (`1e-4`) compared to SDXL (`5e-6`) due to its MMDiT architecture and flow matching objective.

**Output:** LoRA adapter weights saved to `./models/infant_lora_sd35/mmdit_lora_final/`.

---

## Generating Images

After training is complete, generate 100 images per emotion for each model.

### 1. Generate with SDXL

```bash
python src/stable_diffusion/generate_infant_faces.py \
  --lora_path ./models/infant_lora/unet_lora_final \
  --output_dir ./generated_images/sdxl \
  --num_images 100 \
  --guidance_scale 7.5 \
  --num_inference_steps 30 \
  --seed 42
```

### 2. Generate with SD3.5 Medium

```bash
python src/stable_diffusion_35/generate_infant_faces_sd35.py \
  --lora_path ./models/infant_lora_sd35/mmdit_lora_final \
  --output_dir ./generated_images/sd35 \
  --num_images 100 \
  --guidance_scale 7.0 \
  --num_inference_steps 30 \
  --seed 42
```

**Output format:** Images are saved in emotion-specific subfolders:

```text
generated_images/
  ├── sdxl/
  │   ├── happy/
  │   ├── angry/
  │   └── crying/
  └── sd35/
      ├── happy/
      ├── angry/
      └── crying/
```

---

## Evaluation

Evaluation is performed with the scripts under `evaluation/`. Final evaluation uses the held-out Hugging Face dataset `InfantEmotionGen/InfantEmotionGen_Dataset`, specifically `final_test_samples/test_samples.json`, which provides 750 balanced reference images (250 per emotion). The generated images are compared with:

- **FID:** Measures distributional realism against the reference set.
- **CLIP Agreement and CLIPScore:** Measures semantic alignment between generated images and emotion prompts.
- **FER Accuracy & Macro F1:** Measures emotional expression recognizability using a frozen facial expression classifier.

Populate and validate the final external reference set:

```bash
evaluation/scripts/populate_external_reference.sh
evaluation/scripts/populate_external_reference.sh --validate-only
```

Run the full configured evaluation:

```bash
evaluation/scripts/run_full_model_evaluation.sh
```

If SDXL is already complete and only SD3.5 needs to run:

```bash
RUN_SDXL=0 DEVICE=mps evaluation/scripts/run_full_model_evaluation.sh
```

Evaluation outputs are saved under `evaluation/results/<run_name>/`, and the final comparison table is written to:

```text
evaluation/results/comparison.csv
evaluation/results/comparison.md
```

Current final comparison on the external reference set:

| Model | FID ↓ | CLIP Agreement ↑ | CLIPScore ↑ | FER Acc. ↑ | FER Macro F1 ↑ | Images |
|-------|------:|-----------------:|------------:|-----------:|---------------:|-------:|
| SDXLPrimary | 144.6880 | 0.9167 | 0.8925 | 0.5533 | 0.5598 | 300 |
| SD3.5 Medium | **125.2550** | **0.9433** | **0.9241** | **0.7967** | **0.8649** | 300 |

See `evaluation/README.md` for the complete evaluation workflow, smoke-test split, model registry, generation protocol, and metric details.

---

## Configuration

All hyperparameters for training and generation are configured via the `configs.py` file. Key parameters include:

| Parameter | SDXL | SD3.5 Medium |
|-----------|------|--------------|
| Learning Rate | `5e-6` | `1e-4` |
| LoRA Rank | `16` | `16` |
| LoRA Alpha | `16` | `16` |
| Batch Size | `1` | `1` |
| Gradient Accumulation | `4` | `4` |
| Max Steps | `1500` | `1500` |

Edit `configs.py` to adjust these values globally across training runs.
