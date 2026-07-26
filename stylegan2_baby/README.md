# StyleGAN2-ADA Infant Emotion Training

This directory converts the original Colab notebook workflow into normal,
version-controlled scripts.

## Important backend note

The official NVIDIA `stylegan2-ada-pytorch` training implementation targets
Linux/Windows, NVIDIA GPUs, CUDA, and custom CUDA extensions. The scripts in
this folder therefore use CUDA for actual StyleGAN2-ADA training.

An Apple Silicon Mac can still be used to:

- authenticate with Hugging Face;
- download the private dataset;
- validate the StyleGAN2 ZIP and labels;
- inspect class counts and image resolution;
- prepare and commit the project code;
- run a StyleGAN2 training dry run.

Running the official trainer on MPS would require a separate port of the
training loop and custom operations. That port would not be the same tested
backend used by the teammate's notebook.

## Directory layout

```text
stylegan2_baby/
├── configs/
│   ├── class_names.example.json
│   └── train.env.example
├── data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── patches/
│   └── README.md
├── scripts/
│   ├── apply_partner_patches.sh
│   ├── bootstrap_stylegan2.sh
│   ├── check_device.py
│   ├── download_dataset.py
│   ├── generate_samples.sh
│   ├── train_cuda.sh
│   ├── train_smoke.sh
│   └── verify_dataset.py
├── src/
│   └── stylegan2_baby/
│       ├── __init__.py
│       └── hf_dataset.py
├── .gitignore
└── requirements.txt
```

The NVIDIA source repository is cloned into `vendor/stylegan2-ada-pytorch/`
and ignored by Git. The infant dataset, checkpoints, and generated images are
also ignored.

## 1. Put this folder in the project

From the root of `InfantEmotionGen`:

```bash
cp -R /path/to/stylegan2_baby ./stylegan2_baby
cd stylegan2_baby
```

## 2. Create a local environment on the Mac

A separate environment prevents the SDXL dependencies in the root project
from conflicting with StyleGAN2.

```bash
conda create -n infant-stylegan python=3.10 -y
conda activate infant-stylegan

python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

Check PyTorch and MPS:

```bash
python scripts/check_device.py
```

## 3. Authenticate with Hugging Face

Because the dataset repository is private, the logged-in Hugging Face account
must have access.

```bash
hf auth login
```

Do not place a Hugging Face token in Git, a shell script, or a notebook.

## 4. Download the private dataset

```bash
python scripts/download_dataset.py
```

The default source is:

```text
InfantEmotionGen/baby_samples_gan
```

The script downloads the dataset repository and copies its single ZIP file to:

```text
data/baby_samples_gan.zip
```

You can override either path:

```bash
python scripts/download_dataset.py \
  --repo-id InfantEmotionGen/baby_samples_gan \
  --output data/baby_samples_gan.zip
```

## 5. Verify the dataset and labels

```bash
python scripts/verify_dataset.py
```

The teammate's notebook reported 1,200 labeled images. This script confirms:

- `dataset.json` exists;
- every labeled image exists in the ZIP;
- labels are valid;
- class counts;
- image sizes and color modes from a sample.

The numeric-to-name mapping is not guaranteed to be stored in the ZIP.
Confirm which indices correspond to angry, cry, and happy, then copy
`configs/class_names.example.json` to `configs/class_names.json` and edit it.

## 6. Clone the NVIDIA implementation

```bash
bash scripts/bootstrap_stylegan2.sh
```

This installs the lightweight dependencies and clones:

```text
vendor/stylegan2-ada-pytorch/
```

## 7. Preserve the teammate's modifications

The notebook copied custom versions of these files from Google Drive:

```text
training/training_loop.py
torch_utils/misc.py
```

Ask for the exact versions that produced the teammate's run. Place them at:

```text
patches/training_loop.py
patches/misc.py
```

Then apply them:

```bash
bash scripts/apply_partner_patches.sh
```

This is important for reproducibility. The original notebook depended on
files that are currently outside Git.

## 8. Run a configuration smoke test

This prints the complete training configuration without starting training:

```bash
bash scripts/train_smoke.sh
```

## 9. Train with CUDA

The default command mirrors the notebook:

- conditional model;
- ADA;
- horizontal mirroring;
- FFHQ-512 transfer learning;
- batch size 16;
- snapshots every 25 ticks;
- 500 kimg.

```bash
bash scripts/train_cuda.sh
```

For an initial real smoke run, reduce the duration and disable the expensive
FID metric:

```bash
KIMG=1 SNAP=1 METRICS=none BATCH=4 bash scripts/train_cuda.sh
```

For a lower-memory GPU:

```bash
BATCH=4 WORKERS=2 METRICS=none bash scripts/train_cuda.sh
```

For the full teammate-style run:

```bash
KIMG=500 BATCH=16 SNAP=25 METRICS=fid50k_full \
  bash scripts/train_cuda.sh
```

Outputs are written under `outputs/` unless `OUTDIR` is set.

## 10. Run from Google Colab without putting logic in a notebook

Use a GPU runtime. The notebook only needs to launch the repository scripts:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone https://github.com/JessicaHenson01/InfantEmotionGen.git
%cd /content/InfantEmotionGen/stylegan2_baby
!python -m pip install -r requirements.txt
!hf auth login
!python scripts/download_dataset.py
!bash scripts/bootstrap_stylegan2.sh
!bash scripts/apply_partner_patches.sh
!KIMG=1 SNAP=1 METRICS=none BATCH=4 bash scripts/train_cuda.sh
```

After the smoke run succeeds:

```bash
!OUTDIR=/content/drive/MyDrive/stylegan2_baby_runs \
  KIMG=500 BATCH=16 SNAP=25 METRICS=fid50k_full \
  bash scripts/train_cuda.sh
```

All meaningful logic remains in Git. Colab is only the remote CUDA terminal.

## 11. Generate images from a snapshot

Set the network snapshot and class index:

```bash
NETWORK=outputs/<run>/network-snapshot-000500.pkl \
CLASS_IDX=0 \
SEEDS=0-15 \
bash scripts/generate_samples.sh
```

Generated files are written to `outputs/generated/` by default.
