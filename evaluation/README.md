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

Use a dedicated conda environment for evaluation instead of `base`:

```bash
conda create -n infant-eval python=3.11 -y
conda activate infant-eval
```

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

## Final External Reference Dataset

For final model evaluation, use the held-out dataset that was not used for training:

```text
InfantEmotionGen/InfantEmotionGen_Dataset
```

Populate the final reference folders:

```bash
evaluation/scripts/populate_external_reference.sh
```

This writes:

```text
evaluation/data/external_reference/
  angry/   250 images
  crying/  250 images
  happy/   250 images

evaluation/data/external_reference_manifest.json
```

Validate the existing external reference folders:

```bash
evaluation/scripts/populate_external_reference.sh --validate-only
```

Optional override:

```bash
REPO_ID=InfantEmotionGen/InfantEmotionGen_Dataset \
OUTPUT_ROOT="$PWD/evaluation/data" \
evaluation/scripts/populate_external_reference.sh
```

The default ZIP labels path is:

```text
final_test_samples/test_samples.json
```

Override it only if the dataset layout changes:

```bash
evaluation/scripts/populate_external_reference.sh \
  --zip-label-json final_test_samples/test_samples.json
```

The importer supports common Hugging Face image dataset layouts:

- class folders named `angry`, `crying`, and `happy`;
- `metadata.jsonl` or `metadata.csv` with `file_name` plus `label`, `class`, or `emotion`;
- a StyleGAN-format ZIP with `dataset.json`.

The smoke-test split remains useful for checking the pipeline, but final FID should use this external reference dataset.

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

## Parameter Counts

Count cached Hugging Face model parameters from safetensor metadata:

```bash
python evaluation/scripts/count_model_parameters.py \
  --json evaluation/results/parameter_counts.json
```

Current measured counts:

```text
sdxl_base
  total: 3.469B
  unet: 2.567B
  text_encoder: 0.123B
  text_encoder_2: 0.695B
  vae_1_0: 0.084B

sdxl_primary_lora
  total adapter parameters: 0.013B / 12.57M

sd35_base
  total: 8.134B
  transformer: 2.470B
  text_encoder: 0.124B
  text_encoder_2: 0.695B
  text_encoder_3: 4.762B
  vae: 0.084B

sd35_medium_lora
  total adapter parameters: 0.007B / 7.27M
```

Important distinction: SDXL and SD3.5 have similarly sized core denoisers, but SD3.5's full inference pipeline is much larger because it includes `text_encoder_3`, a T5 text encoder. If reporting only the core denoiser, compare `unet` for SDXL against `transformer` for SD3.5. If reporting total inference pipeline size, include all text encoders and the VAE.

## Scaling Up

Start with a small real batch before trying full protocol generation:

```bash
evaluation/scripts/generate_model_run.sh sdxl_primary \
  --num-images 10 \
  --height 512 \
  --width 512 \
  --num-inference-steps 10 \
  --device auto \
  --skip-existing
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
  --skip-existing
```

Then evaluate and compare:

```bash
evaluation/scripts/evaluate_generated_run.sh sdxl_primary
evaluation/scripts/evaluate_generated_run.sh sd35_medium
evaluation/scripts/compare_model_runs.sh
```

For final evaluation, generate the same number of images per class for every model.

Use `--skip-existing` for longer runs. It keeps any already-generated expected PNGs and only fills in missing files, so a crashed run can be restarted with the exact same command. Use `--overwrite` only when you intentionally want to clear that model's generated output folder and start again.

Recommended local progression:

```text
1 image/class, 512x512, 5 steps     pipeline check
10 images/class, 512x512, 10 steps  small comparison
25 images/class, 512x512, 15 steps  better local comparison if disk/time allow
100 images/class, 1024x1024, 30 steps  final protocol, preferably on a stronger GPU machine
```

## Full Evaluation

Run the complete configured evaluation:

```bash
evaluation/scripts/run_full_model_evaluation.sh
```

By default this uses the protocol files exactly:

```text
100 images per class
1024x1024 resolution
30 inference steps
seed 4242
guidance scale 7.5
```

The full runner:

1. checks available disk space;
2. uses `evaluation/data/external_reference` as the default real-image reference;
3. resumes SDXLPrimary generation with `--skip-existing`;
4. evaluates SDXLPrimary with FID, CLIP, and FER;
5. resumes SD3.5 Medium generation with `--skip-existing`;
6. evaluates SD3.5 Medium with FID, CLIP, and FER;
7. writes `evaluation/results/comparison.csv` and `evaluation/results/comparison.md`.

If the laptop crashes, rerun the same command. Existing expected PNGs are kept and missing images are generated.
If an existing PNG has the wrong size for the requested protocol, the generator replaces it instead of keeping it. This matters when a folder already contains tiny-test `512x512` images and the final protocol requests `1024x1024`.

The runner refuses to start unless at least `60GiB` is free. To intentionally use a lower threshold:

```bash
MIN_FREE_GB=25 evaluation/scripts/run_full_model_evaluation.sh
```

To run only one model:

```bash
RUN_SD35=0 evaluation/scripts/run_full_model_evaluation.sh
RUN_SDXL=0 evaluation/scripts/run_full_model_evaluation.sh
```

To do a smaller full-pipeline rehearsal through the same runner:

```bash
NUM_IMAGES=10 \
HEIGHT=512 \
WIDTH=512 \
NUM_INFERENCE_STEPS=10 \
evaluation/scripts/run_full_model_evaluation.sh
```

## Metrics

FID:

- compares generated images to the configured real-reference folder;
- the full runner defaults to `evaluation/data/external_reference`;
- smoke tests use `evaluation/data/real_reference`;
- resizes images to a fixed square size before batching so mixed source dimensions do not crash evaluation;
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
