# Evidence and Log Files

This folder contains reproducibility evidence for the final SDXLPrimary and
Stable Diffusion 3.5 Medium comparison.

## Training Logs

- `sdxl_train.log`: SDXL LoRA training log. The terminal progress capture jumps
  from `1200/1500` to final model saving because the progress-bar output was
  truncated in the captured log. The final lines show `Saving final model...`
  and `Training complete! Final model saved to ./models/infant_lora/unet_lora_final`.
  If a complete raw SDXL log is available, replace this file with the complete
  capture.
- `sd35_train.log`: SD3.5 Medium LoRA training log. This log shows
  `1500/1500`, checkpoint saving at step 1500, and final model saving to
  `./models/infant_lora_sd35/mmdit_lora_final`.

## Generation Logs

- `sdxl_generate.log`: SDXL generation evidence.
- `sd35_generate.log`: SD3.5 Medium generation evidence.

## Evaluation Results

The `evaluation_results/` folder contains tracked JSON summaries copied from
the ignored `evaluation/results/` directory:

- `evaluation_results/sdxl_primary/`: FID, CLIP, and FER metrics for SDXLPrimary.
- `evaluation_results/sd35_medium/`: FID, CLIP, and FER metrics for SD3.5 Medium.
- `evaluation_results/pipeline_test/`: smoke-test evaluation outputs.
- `evaluation_results/parameter_counts.json`: model parameter-count summary.

The final generated-image evaluation uses the held-out external reference set
from `InfantEmotionGen/InfantEmotionGen_Dataset`.
