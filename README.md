# InfantEmotionGen

InfantEmotionGen investigates whether modern generative models can create realistic infant face images with controllable emotional expressions. The project focuses on three target classes:

```text
0 = angry
1 = crying
2 = happy
```

The current final comparison is between two LoRA-adapted diffusion models published under the `InfantEmotionGen` Hugging Face organization:

```text
InfantEmotionGen/SDXLPrimary
InfantEmotionGen/stable-diffusion-3.5-medium
```

The repository also contains StyleGAN2-ADA work used as a baseline/related model track.

## Project Goal

Real infant facial-expression datasets are difficult to collect because of privacy, consent, and data-scarcity constraints. This project explores synthetic infant emotion generation as a way to produce controlled images for research workflows while still evaluating realism and expression consistency carefully.

The main questions are:

1. Can fine-tuned diffusion models generate convincing infant faces?
2. Can the generated faces reliably express `angry`, `crying`, and `happy`?
3. How do SDXL LoRA and SD3.5 Medium LoRA compare under the same generation and evaluation protocol?

## Repository Layout

```text
data/
  baby_emotion_samples.json        Local training metadata
  baby_emotion_samples/            Local training images, not committed

src/
  data_utils/                      Shared infant emotion dataset loader
  stable_diffusion/                SDXL DreamBooth/LoRA training and generation
  stable_diffusion_35/             SD3.5 Medium DreamBooth/LoRA training and generation

stylegan2_baby/
  scripts/                         StyleGAN2-ADA setup, dataset prep, training, generation
  configs/                         StyleGAN2 config examples
  patches/                         Apple Silicon/MPS compatibility patches

evaluation/
  configs/                         Model registry, generation protocol, CLIP/FER mappings
  scripts/                         Dataset import, generation, FID, CLIP, FER, comparison
  data/                            Local reference datasets, ignored by git
  generated/                       Generated model images, ignored by git
  results/                         Metric outputs, ignored by git

InfantGeneration_SD35.ipynb        SD3.5 experimentation notebook
stylegan2_baby.ipynb               StyleGAN2 experimentation notebook
Project Proposal.md                Project proposal and metric definitions
InterimReport.md                   Interim experimental progress report
```

## Data

Training data is local/private and is not committed to git. The project uses infant face images labeled as `angry`, `crying`, or `happy`.

The local dataset loader is:

```text
src/data_utils/dataset.py
```

It expects JSON labels in this format:

```json
{
  "image_001.jpg": 0,
  "image_002.jpg": 1,
  "image_003.jpg": 2
}
```

The current local metadata file is:

```text
data/baby_emotion_samples.json
```

Some training scripts still default to `data/labels_formatted.json`; pass the current metadata file explicitly when needed:

```bash
--json_path data/baby_emotion_samples.json
```

## Final Evaluation Reference

Final evaluation should use the held-out dataset that was not used for model training:

```text
InfantEmotionGen/InfantEmotionGen_Dataset
```

The external reference importer uses the ZIP file's held-out labels:

```text
final_test_samples/test_samples.json
```

It populates:

```text
evaluation/data/external_reference/
  angry/   250 images
  crying/  250 images
  happy/   250 images
```

Populate and validate it with:

```bash
evaluation/scripts/populate_external_reference.sh
evaluation/scripts/populate_external_reference.sh --validate-only
```

The older `evaluation/data/real_reference` and `evaluation/data/pipeline_test` folders are for smoke testing the evaluation pipeline, not final reporting.

## Model Tracks

### SDXLPrimary

The SDXL model uses:

```text
base model: stabilityai/stable-diffusion-xl-base-1.0
LoRA repo:  InfantEmotionGen/SDXLPrimary
adapter:    unet_lora_final
target:     SDXL UNet
```

The SDXL training code is:

```text
src/stable_diffusion/train_dreambooth_lora_sdxl.py
```

This track uses DreamBooth with LoRA and the `sks` trigger token for the infant concept.

### SD3.5 Medium

The SD3.5 model uses:

```text
base model: stabilityai/stable-diffusion-3.5-medium
LoRA repo:  InfantEmotionGen/stable-diffusion-3.5-medium
adapter:    mmdit_lora_final
target:     SD3.5 MMDiT transformer
```

The SD3.5 training code is:

```text
src/stable_diffusion_35/train_dreambooth_lora_sd35.py
```

Each user must accept Stability AI's gated model access before loading SD3.5:

```text
https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
```

### StyleGAN2-ADA

The StyleGAN2-ADA work lives in:

```text
stylegan2_baby/
```

It includes dataset preparation, smoke tests, Apple Silicon compatibility patches, and training/generation scripts. This track is useful for the GAN baseline and project comparison context, while the current automated evaluation workflow is centered on the two Hugging Face diffusion model runs.

## Environment Setup

Use a dedicated conda environment instead of `base`:

```bash
conda create -n infant-eval python=3.11 -y
conda activate infant-eval
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r evaluation/requirements.txt
python -m pip install -r evaluation/requirements_generation.txt
python -m pip install -r evaluation/requirements_eval.txt
```

Authenticate to Hugging Face:

```bash
hf auth login
```

On Apple Silicon, check whether PyTorch can use MPS:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
print(torch.ones(1, device="mps"))
PY
```

If MPS is unavailable, generation will fall back to CPU unless `DEVICE=mps` is explicitly requested. Full SD3.5 generation is very slow on CPU.

## Fixed Generation Protocol

The evaluation generation settings live in:

```text
evaluation/configs/generation_protocol.sdxl.json
evaluation/configs/generation_protocol.sd35.json
```

The full protocol is:

```text
100 images per class
300 images per model
1024x1024 resolution
30 inference steps
seed 4242
guidance scale 7.5
```

SDXL prompts:

```text
a photo of an angry sks infant face
a photo of a crying sks infant face
a photo of a happy sks infant face
```

SD3.5 prompts:

```text
a photo of an angry infant face
a photo of a crying infant face
a photo of a happy infant face
```

The generation script assigns deterministic seeds by class and image index, so interrupted runs can resume reproducibly.

## Run Final Evaluation

The configured model runs are listed in:

```text
evaluation/configs/model_runs.json
```

Run the complete final workflow:

```bash
evaluation/scripts/run_full_model_evaluation.sh
```

If SDXL is already complete and only SD3.5 still needs generation/evaluation:

```bash
RUN_SDXL=0 DEVICE=mps evaluation/scripts/run_full_model_evaluation.sh
```

The runner:

1. checks disk space;
2. uses `evaluation/data/external_reference` as the real-image reference;
3. generates missing images with `--skip-existing`;
4. evaluates each model with FID, CLIP, and FER;
5. writes comparison files.

Generated images:

```text
evaluation/generated/<run_name>/
  angry/
  crying/
  happy/
  generation_manifest.json
```

Metric outputs:

```text
evaluation/results/<run_name>/
  fid.json
  clip.json
  fer.json
```

Comparison outputs:

```text
evaluation/results/comparison.csv
evaluation/results/comparison.md
```

## Metrics

FID compares the generated image distribution against the held-out external reference dataset. Lower is better.

CLIP measures whether generated images align with the intended emotion text prompts. Higher agreement and target cosine are better.

FER runs a frozen facial-expression classifier on generated images and compares the predicted expression against the image folder label. Higher accuracy and macro F1 are better. The current FER model is a proxy and should not be described as infant-specific ground truth.

The generated images are not directly paired with specific real images. Each model is evaluated separately against the same reference distribution and then compared side by side.

## Smoke Tests

Before final generated images are available, the pipeline can be tested with the private StyleGAN-format dataset:

```text
InfantEmotionGen/baby_samples_gan
```

Populate the deterministic 80/20 smoke-test split:

```bash
evaluation/scripts/populate_real_eval_folders.sh
```

Run smoke tests:

```bash
evaluation/scripts/run_smoke_tests.sh
```

Smoke-test FID compares `evaluation/data/pipeline_test` against `evaluation/data/real_reference`. This proves the pipeline works, but it is not the final model evaluation.

## Git And Data Policy

Generated images, downloaded datasets, model caches, and metric outputs are local artifacts and should not be committed.

Ignored local artifacts include:

```text
evaluation/data/
evaluation/generated/
evaluation/results/
evaluation/.cache/
*.zip
root-level extracted image files
```

Track source code and configuration:

```text
src/
stylegan2_baby/scripts/
stylegan2_baby/configs/
evaluation/scripts/
evaluation/configs/
README.md
evaluation/README.md
requirements files
```
