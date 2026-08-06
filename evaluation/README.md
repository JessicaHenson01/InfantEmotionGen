# Generated Image Evaluation Toolkit

This toolkit evaluates image folders after StyleGAN2 or SDXL generation. It calculates:

- overall and optional per-class FID;
- OpenCLIP emotion alignment and zero-shot class agreement;
- FER classifier agreement, accuracy, macro F1, and confusion matrix;
- comparison tables across checkpoints or models.

## Expected generated-image layout

```text
generated/stylegan2_50k/
├── class_0/
├── class_1/
└── class_2/
```

Use the same number of generated images per class for every model being compared.

## Install

Copy this folder into the project as `stylegan2_baby/evaluation/`, then run:

```bash
python -m pip install -r evaluation/requirements_eval.txt
```

FID is forced to CPU to avoid MPS float64/covariance issues. CLIP and FER can run on MPS.

## Real-data pipeline smoke test

Use this workflow to validate that the FID, CLIP, and FER scripts can read a clean pair of evaluation folders before final generated images are available. This is a pipeline-validation split only. FID from `real_reference` versus `pipeline_test` is a smoke-test baseline, not final generated-model evaluation.

Authenticate to Hugging Face before downloading the private dataset:

```bash
hf auth login
```

Install the small dependency set needed for this folder-population workflow:

```bash
python -m pip install -r evaluation/requirements.txt
```

Populate the deterministic 80/20 stratified split:

```bash
evaluation/scripts/populate_real_eval_folders.sh
```

The script downloads or reuses the cached Hugging Face dataset snapshot for `InfantEmotionGen/baby_samples_gan`, finds the StyleGAN-format ZIP, reads `dataset.json`, maps labels as `0=angry`, `1=crying`, `2=happy`, and extracts RGB PNG files into:

```text
evaluation/data/
├── real_reference/
│   ├── angry/
│   ├── crying/
│   └── happy/
├── pipeline_test/
│   ├── angry/
│   ├── crying/
│   └── happy/
└── split_manifest.json
```

Environment-variable overrides:

```bash
REPO_ID=InfantEmotionGen/baby_samples_gan \
OUTPUT_ROOT="$PWD/evaluation/data" \
TEST_FRACTION=0.20 \
SEED=42 \
evaluation/scripts/populate_real_eval_folders.sh
```

If the downloaded dataset snapshot contains more than one ZIP, pass the ZIP name:

```bash
evaluation/scripts/populate_real_eval_folders.sh --zip-name baby_samples_gan.zip
```

Validate an existing split without downloading or extracting:

```bash
evaluation/scripts/populate_real_eval_folders.sh --validate-only
```

Validation checks that all six emotion folders exist, each contains at least one image, no source or output filename appears in both splits, and the total extracted image count matches the labeled image count recorded from `dataset.json` in `split_manifest.json`.

Run the smoke-test suite:

```bash
evaluation/scripts/run_smoke_tests.sh
```

This validates the split, runs FID from `real_reference` to `pipeline_test`, and runs CLIP with `evaluation/configs/clip_prompts.smoke.json`. FER is skipped unless a classifier is supplied:

```bash
FER_MODEL_ID=YOUR_FER_MODEL_OR_LOCAL_PATH evaluation/scripts/run_smoke_tests.sh
```

Smoke-test outputs are written to:

```text
evaluation/results/pipeline_test/
├── fid.json
├── clip.json
└── fer.json
```

The smoke-test FER mapping is in `evaluation/configs/fer_mapping.smoke.json`. It maps common classifier labels like `sad`/`sadness` to the project class `crying`; adjust this if your FER model uses different labels.

Run an overall FID smoke test:

```bash
python evaluation/scripts/evaluate_fid.py \
  --real evaluation/data/real_reference \
  --generated evaluation/data/pipeline_test \
  --output evaluation/results/pipeline_test/fid.json \
  --batch-size 32
```

For CLIP and FER smoke tests, make sure the config keys match the emotion folder names: `angry`, `crying`, and `happy`.

```bash
python evaluation/scripts/evaluate_clip.py \
  --generated evaluation/data/pipeline_test \
  --prompts evaluation/configs/clip_prompts.smoke.json \
  --output evaluation/results/pipeline_test/clip.json \
  --device auto
```

```bash
python evaluation/scripts/evaluate_fer.py \
  --generated evaluation/data/pipeline_test \
  --model-id YOUR_FER_MODEL_OR_LOCAL_PATH \
  --mapping evaluation/configs/fer_mapping.smoke.json \
  --output evaluation/results/pipeline_test/fer.json \
  --device auto
```

## Configure the class meanings

```bash
cp evaluation/configs/class_map.example.json evaluation/configs/class_map.json
cp evaluation/configs/clip_prompts.example.json evaluation/configs/clip_prompts.json
cp evaluation/configs/fer_mapping.example.json evaluation/configs/fer_mapping.json
```

Edit all three files after confirming which emotion belongs to class 0, 1, and 2. Do not guess the label order.

## Generate SDXL LoRA model outputs

For actual model evaluation, first generate images from each teammate model into a fixed folder layout:

```text
evaluation/generated/<run_name>/
├── angry/
├── crying/
└── happy/
```

Install generation dependencies:

```bash
python -m pip install -r evaluation/requirements_generation.txt
```

The fixed SDXL generation protocol is stored in:

```text
evaluation/configs/generation_protocol.sdxl.json
```

It controls prompts, image count, image size, inference steps, guidance scale, negative prompt, and seed. Keep it unchanged while generating all models you want to compare.

Generate a small one-image-per-class test from the current SDXLPrimary Hugging Face LoRA:

```bash
evaluation/scripts/generate_model_run.sh sdxl_primary \
  --num-images 1 \
  --height 512 \
  --width 512 \
  --num-inference-steps 5 \
  --device auto \
  --overwrite
```

Generate the full protocol batch:

```bash
evaluation/scripts/generate_model_run.sh sdxl_primary \
  --device auto \
  --overwrite
```

The SDXLPrimary adapter is PEFT-style because the uploaded folder contains `adapter_config.json` and `adapter_model.safetensors`.

The named model runs are configured in:

```text
evaluation/configs/model_runs.json
```

Currently configured runs:

```text
sdxl_primary -> evaluation/generated/sdxl_primary/ -> evaluation/results/sdxl_primary/
sd35_medium  -> evaluation/generated/sd35_medium/  -> evaluation/results/sd35_medium/
```

Generate a small one-image-per-class test from the Stable Diffusion 3.5 Medium repo:

```bash
evaluation/scripts/generate_model_run.sh sd35_medium \
  --num-images 1 \
  --height 512 \
  --width 512 \
  --num-inference-steps 5 \
  --device auto \
  --overwrite
```

If the SD3.5 download fails with a Hugging Face Xet reconstruction error, retry with Xet disabled and one download worker:

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

The `sd35_medium` run uses the Stability AI SD3.5 Medium base model plus the team PEFT adapter at:

```text
InfantEmotionGen/stable-diffusion-3.5-medium/mmdit_lora_final/
```

That adapter targets the SD3 transformer/MMDiT component, so the registry uses `pipeline_type=sd3_lora` and `adapter_format=peft_transformer`.

Generate the full SD3.5 protocol batch:

```bash
evaluation/scripts/generate_model_run.sh sd35_medium \
  --device auto \
  --overwrite
```

After generation, evaluate a run:

```bash
evaluation/scripts/evaluate_generated_run.sh sdxl_primary
```

or:

```bash
evaluation/scripts/evaluate_generated_run.sh sd35_medium
```

That writes:

```text
evaluation/results/<run_name>/
├── fid.json
├── clip.json
└── fer.json
```

After both runs have result JSON files, create the comparison table:

```bash
evaluation/scripts/compare_model_runs.sh
```

## Extract the real reference images

```bash
python evaluation/scripts/extract_real_dataset.py \
  --zip data/baby_samples_gan_256.zip \
  --outdir evaluation/data/real_256 \
  --class-map evaluation/configs/class_map.json
```

## Generate an evaluation image set

For 500 images per class:

```bash
python scripts/generate_mps.py \
  --network outputs_mps/<run>/network-snapshot-000050.pkl \
  --outdir generated/stylegan2_50k \
  --seeds 0-499 \
  --classes 0,1,2 \
  --trunc 1.0
```

Use truncation 1.0 for formal evaluation because lower truncation tends to improve average appearance while reducing diversity.

## FID

```bash
python evaluation/scripts/evaluate_fid.py \
  --real evaluation/data/real_256 \
  --generated generated/stylegan2_50k \
  --output evaluation/results/stylegan2_50k/fid.json \
  --batch-size 32 \
  --per-class
```

Lower FID is better. FID is noisy with small image counts, especially per class, so compare runs with equal sample counts and identical settings.

## CLIP emotion alignment

Edit `clip_prompts.json` first, then run:

```bash
python evaluation/scripts/evaluate_clip.py \
  --generated generated/stylegan2_50k \
  --prompts evaluation/configs/clip_prompts.json \
  --output evaluation/results/stylegan2_50k/clip.json \
  --device mps
```

The script reports zero-shot class agreement, mean target cosine similarity, per-class scores, and a confusion summary. Higher is better. CLIP should be treated as a supporting metric because subtle infant expressions may not be well represented by generic CLIP models.

## FER classifier agreement

The FER script accepts any Hugging Face image-classification model or a local model saved in Hugging Face format:

```bash
python evaluation/scripts/evaluate_fer.py \
  --generated generated/stylegan2_50k \
  --model-id YOUR_FER_MODEL_OR_LOCAL_PATH \
  --mapping evaluation/configs/fer_mapping.json \
  --output evaluation/results/stylegan2_50k/fer.json \
  --device mps
```

The mapping connects the FER model's labels to your three project classes. For example, a project class called `crying` might accept a classifier label called `sad`, but that assumption must be documented.

For the strongest evaluation, use an independent FER classifier trained on real infant images with a held-out split. An adult-face FER classifier is only a proxy and should be described as a limitation.

## Compare checkpoints or models

```bash
python evaluation/scripts/compare_results.py \
  --run StyleGAN2-10k \
    evaluation/results/stylegan2_10k/fid.json \
    evaluation/results/stylegan2_10k/clip.json \
    evaluation/results/stylegan2_10k/fer.json \
  --run StyleGAN2-50k \
    evaluation/results/stylegan2_50k/fid.json \
    evaluation/results/stylegan2_50k/clip.json \
    evaluation/results/stylegan2_50k/fer.json \
  --run SDXL \
    evaluation/results/sdxl/fid.json \
    evaluation/results/sdxl/clip.json \
    evaluation/results/sdxl/fer.json \
  --csv evaluation/results/comparison.csv \
  --markdown evaluation/results/comparison.md
```

## Fair-comparison rules

1. Generate the same number of images per class.
2. Use the same output resolution.
3. Use the same real-image reference set.
4. Use the same CLIP prompts.
5. Use the same FER classifier and label mapping.
6. Preserve the generation seeds and model snapshot names.
7. Report qualitative grids and memorization checks alongside the metrics.
8. Do not choose a model using FID alone.
