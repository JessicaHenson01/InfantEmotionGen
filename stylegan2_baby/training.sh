export WANDB_PROJECT="InfantEmotionGen"
export WANDB_RUN_NAME="stylegan2-baby-mps-50kimg"

caffeinate -dimsu env \
  KIMG=50 \
  SNAP=2 \
  BATCH=1 \
  METRICS=none \
  bash scripts/train_mps.sh