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

## Configure the class meanings

```bash
cp evaluation/configs/class_map.example.json evaluation/configs/class_map.json
cp evaluation/configs/clip_prompts.example.json evaluation/configs/clip_prompts.json
cp evaluation/configs/fer_mapping.example.json evaluation/configs/fer_mapping.json
```

Edit all three files after confirming which emotion belongs to class 0, 1, and 2. Do not guess the label order.

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
