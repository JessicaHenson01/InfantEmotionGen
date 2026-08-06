# InfantEmotionGen Evaluation

This folder contains the evaluation workflow for generated infant emotion images. It supports:

- deterministic real-data smoke-test splits;
- fixed-protocol image generation from teammate Hugging Face models;
- FID, CLIP, and FER evaluation;
- comparison tables across model runs.

## Current Status

The evaluation pipeline has been tested end to end.

Smoke-test real-data split:

```text
evaluation/data/real_reference/
  angry/   320 images
  crying/  320 images
  happy/   320 images

evaluation/data/pipeline_test/
  angry/    80 images
  crying/   80 images
  happy/    80 images
```

Tiny model-generation tests have also run for both configured models:

```text
evaluation/generated/sdxl_primary/
  angry/
  crying/
  happy/
  generation_manifest.json

evaluation/generated/sd35_medium/
  angry/
  crying/
  happy/
  generation_manifest.json
```

Tiny comparison outputs exist at:

```text
evaluation/results/comparison.csv
evaluation/results/comparison.md
```

These tiny-test scores are only pipeline checks because they use 1 image per class, 512x512 resolution, and 5 inference steps.

## Install

Authenticate to Hugging Face first:

```bash
hf auth login
```

Install the dataset-splitting dependencies:

```bash
python -m pip install -r evaluation/requirements.txt
```

Install generation and evaluation dependencies:

```bash
python -m pip install -r evaluation/requirements_generation.txt
python -m pip install -r evaluation/requirements_eval.txt
```

For SD3.5, each person must accept access to the gated Stability AI base model:

```text
https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
```

## Folder Layout

Real reference data:

```text
evaluation/data/real_reference/
  angry/
  crying/
  happy/
```

Generated model outputs:

```text
evaluation/generated/<run_name>/
  angry/
  crying/
  happy/
  generation_manifest.json
```

Evaluation results:

```text
evaluation/results/<run_name>/
  fid.json
  clip.json
  fer.json
```

## Real-Data Split

Populate the deterministic 80/20 stratified split from the private dataset:

```bash
evaluation/scripts/populate_real_eval_folders.sh
```

The source dataset is:

```text
InfantEmotionGen/baby_samples_gan
```

The label mapping is:

```text
0 = angry
1 = crying
2 = happy
```

The split manifest is saved at:

```text
evaluation/data/split_manifest.json
```

Validate an existing split without downloading or extracting:

```bash
evaluation/scripts/populate_real_eval_folders.sh --validate-only
```

Optional overrides:

```bash
REPO_ID=InfantEmotionGen/baby_samples_gan \
OUTPUT_ROOT="$PWD/evaluation/data" \
TEST_FRACTION=0.20 \
SEED=42 \
evaluation/scripts/populate_real_eval_folders.sh
```

## Smoke Test

Run the real-data smoke-test suite:

```bash
evaluation/scripts/run_smoke_tests.sh
```

This validates the split, evaluates `pipeline_test` against `real_reference`, and writes:

```text
evaluation/results/pipeline_test/
  fid.json
  clip.json
  fer.json
```

The default FER classifier is a generic placeholder:

```text
trpakov/vit-face-expression
```

Use a different FER model with:

```bash
FER_MODEL_ID=YOUR_FER_MODEL_OR_LOCAL_PATH evaluation/scripts/run_smoke_tests.sh
```

## Configured Model Runs

Model runs are registered in:

```text
evaluation/configs/model_runs.json
```

Currently configured:

```text
sdxl_primary
  base: stabilityai/stable-diffusion-xl-base-1.0
  adapter: InfantEmotionGen/SDXLPrimary/unet_lora_final
  adapter type: PEFT UNet LoRA
  generated: evaluation/generated/sdxl_primary/
  results: evaluation/results/sdxl_primary/

sd35_medium
  base: stabilityai/stable-diffusion-3.5-medium
  adapter: InfantEmotionGen/stable-diffusion-3.5-medium/mmdit_lora_final
  adapter type: PEFT transformer/MMDiT LoRA
  generated: evaluation/generated/sd35_medium/
  results: evaluation/results/sd35_medium/
```

Generation protocols:

```text
evaluation/configs/generation_protocol.sdxl.json
evaluation/configs/generation_protocol.sd35.json
```

Keep prompts, seeds, resolution, image count, steps, and guidance scale fixed across models for fair comparison.

## Generate Tiny Test Images

SDXLPrimary:

```bash
evaluation/scripts/generate_model_run.sh sdxl_primary \
  --num-images 1 \
  --height 512 \
  --width 512 \
  --num-inference-steps 5 \
  --device auto \
  --overwrite
```

SD3.5 Medium:

```bash
evaluation/scripts/generate_model_run.sh sd35_medium \
  --num-images 1 \
  --height 512 \
  --width 512 \
  --num-inference-steps 5 \
  --device auto \
  --dtype float16 \
  --disable-xet \
  --hf-transfer-workers 1 \
  --overwrite
```

Use `--disable-xet --hf-transfer-workers 1` for SD3.5 if Hugging Face large-file reconstruction fails.

## Evaluate Generated Runs

Evaluate SDXLPrimary:

```bash
evaluation/scripts/evaluate_generated_run.sh sdxl_primary
```

Evaluate SD3.5 Medium:

```bash
evaluation/scripts/evaluate_generated_run.sh sd35_medium
```

Each command writes:

```text
evaluation/results/<run_name>/
  fid.json
  clip.json
  fer.json
```

## Compare Runs

After both model runs have `fid.json`, `clip.json`, and `fer.json`:

```bash
evaluation/scripts/compare_model_runs.sh
```

This writes:

```text
evaluation/results/comparison.csv
evaluation/results/comparison.md
```

Latest tiny-test comparison:

```text
SDXLPrimary: FID 593.6742, CLIP agreement 0.6667, FER accuracy 0.3333, generated images 3
SD35Medium:  FID 311.5380, CLIP agreement 0.3333, FER accuracy 0.3333, generated images 3
```

Again: these are not final quality scores. They only prove the generation and evaluation plumbing works.

## Scaling Up

Start with a small real batch before trying full protocol generation:

```bash
evaluation/scripts/generate_model_run.sh sdxl_primary \
  --num-images 10 \
  --height 512 \
  --width 512 \
  --num-inference-steps 10 \
  --device auto \
  --overwrite
```

```bash
evaluation/scripts/generate_model_run.sh sd35_medium \
  --num-images 10 \
  --height 512 \
  --width 512 \
  --num-inference-steps 10 \
  --device auto \
  --dtype float16 \
  --disable-xet \
  --hf-transfer-workers 1 \
  --overwrite
```

Then evaluate and compare:

```bash
evaluation/scripts/evaluate_generated_run.sh sdxl_primary
evaluation/scripts/evaluate_generated_run.sh sd35_medium
evaluation/scripts/compare_model_runs.sh
```

For final evaluation, generate the same number of images per class for every model.

## Metrics

FID:

- compares generated images to `evaluation/data/real_reference`;
- lower is better;
- noisy with small image counts, especially tiny tests.

CLIP:

- uses `evaluation/configs/clip_prompts.smoke.json`;
- measures image-text alignment against `angry`, `crying`, and `happy` prompts;
- higher agreement and target cosine are better.

FER:

- uses a Hugging Face image-classification model;
- default is `trpakov/vit-face-expression`;
- current FER is a generic proxy, not an infant-specific evaluator;
- mapping is in `evaluation/configs/fer_mapping.smoke.json`.

## Fair-Comparison Rules

1. Use the same real reference set.
2. Use the same class folders: `angry`, `crying`, `happy`.
3. Generate the same number of images per class for every model.
4. Use the same prompts, seeds, resolution, steps, and guidance scale.
5. Use the same CLIP prompts.
6. Use the same FER classifier and label mapping.
7. Preserve each run's `generation_manifest.json`.
8. Treat tiny-test scores as pipeline checks only.
9. Do not choose a final model using FID alone.

## Disk And Runtime Notes

SDXL and especially SD3.5 are large. Keep an eye on disk space:

```bash
df -h .
```

SD3.5 requires gated access to the Stability AI base model. If downloads fail midway, free disk space and retry with:

```bash
--disable-xet --hf-transfer-workers 1
```

Full generation may be better suited for Colab or another GPU machine if local runs become unstable.
