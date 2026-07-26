#!/usr/bin/env python3
"""Apply single-device MPS compatibility changes to NVIDIA StyleGAN2-ADA.

This patch targets NVlabs/stylegan2-ada-pytorch commit:
d72cc7d041b42ec8e806021a205ed9349f87c6a4

It is intentionally conservative:
- keeps CUDA behavior unchanged;
- fixes modern PyTorch Sampler initialization;
- selects MPS through STYLEGAN_DEVICE=mps;
- uses ordinary PyTorch reference operations on non-CUDA devices;
- disables CUDA timing/memory calls on MPS;
- disables pinned-memory assumptions on MPS.

Run this after copying any project-owned training_loop.py or misc.py into the
vendor checkout.
"""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR = PROJECT_ROOT / "vendor" / "stylegan2-ada-pytorch"
MISC = VENDOR / "torch_utils" / "misc.py"
TRAINING_LOOP = VENDOR / "training" / "training_loop.py"


def backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".pre_mps")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
        print(f"Backup created: {backup_path}")


def replace_once(path: Path, old: str, new: str, marker: str) -> None:
    text = path.read_text(encoding="utf-8-sig")
    if marker in text:
        print(f"Already patched: {path.name} ({marker})")
        return
    if old not in text:
        raise RuntimeError(
            f"Could not find the expected block in {path}.\n"
            "The vendor file may differ from the pinned NVIDIA commit or may "
            "already contain unrelated edits."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"Patched: {path} ({marker})")


def main() -> None:
    for path in (MISC, TRAINING_LOOP):
        if not path.is_file():
            raise SystemExit(
                f"Missing {path}. Run scripts/bootstrap_stylegan2.sh first."
            )
        backup(path)

    replace_once(
        MISC,
        "        super().__init__(dataset)\n",
        "        super().__init__()\n",
        "super().__init__()",
    )

    replace_once(
        TRAINING_LOOP,
        """    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
""",
        """    # STYLEGAN_MPS_DEVICE_PATCH
    requested_device = os.environ.get('STYLEGAN_DEVICE', 'cuda').lower()
    if requested_device == 'auto':
        if torch.cuda.is_available():
            requested_device = 'cuda'
        elif torch.backends.mps.is_available():
            requested_device = 'mps'
        else:
            requested_device = 'cpu'

    if requested_device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('STYLEGAN_DEVICE=cuda, but CUDA is unavailable.')
        device = torch.device('cuda', rank)
    elif requested_device == 'mps':
        if num_gpus != 1:
            raise RuntimeError('The MPS training path supports exactly one device.')
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                'STYLEGAN_DEVICE=mps, but torch.backends.mps.is_available() is False.'
            )
        device = torch.device('mps')
    elif requested_device == 'cpu':
        if num_gpus != 1:
            raise RuntimeError('The CPU training path supports exactly one device.')
        device = torch.device('cpu')
    else:
        raise ValueError(
            f'Unsupported STYLEGAN_DEVICE={requested_device!r}; '
            'use cuda, mps, cpu, or auto.'
        )

    if rank == 0:
        print(f'Using training device: {device}')

    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    # NVIDIA's custom grad fixes and fused kernels are CUDA-oriented. On MPS,
    # the existing StyleGAN operators automatically use ordinary PyTorch
    # reference implementations.
    conv2d_gradfix.enabled = (device.type == 'cuda')
    grid_sample_gradfix.enabled = (device.type == 'cuda')
""",
        "STYLEGAN_MPS_DEVICE_PATCH",
    )

    replace_once(
        TRAINING_LOOP,
        """    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
""",
        """    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    data_loader_kwargs = dict(data_loader_kwargs)
    if device.type != 'cuda':
        data_loader_kwargs['pin_memory'] = False
    training_set_iterator = iter(torch.utils.data.DataLoader(
        dataset=training_set,
        sampler=training_set_sampler,
        batch_size=batch_size // num_gpus,
        **data_loader_kwargs,
    ))
""",
        "data_loader_kwargs['pin_memory'] = False",
    )

    replace_once(
        TRAINING_LOOP,
        """        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)
""",
        """        if rank == 0 and device.type == 'cuda':
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)
""",
        "rank == 0 and device.type == 'cuda'",
    )

    replace_once(
        TRAINING_LOOP,
        """            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
""",
        """            all_gen_c = torch.from_numpy(np.stack(all_gen_c))
            if device.type == 'cuda':
                all_gen_c = all_gen_c.pin_memory()
            all_gen_c = all_gen_c.to(device)
""",
        "all_gen_c = all_gen_c.pin_memory()",
    )

    replace_once(
        TRAINING_LOOP,
        """        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
""",
        """        if device.type == 'cuda':
            accelerator_mem_gb = torch.cuda.max_memory_allocated(device) / 2**30
            torch.cuda.reset_peak_memory_stats()
        elif device.type == 'mps':
            accelerator_mem_gb = torch.mps.current_allocated_memory() / 2**30
        else:
            accelerator_mem_gb = 0.0
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', accelerator_mem_gb):<6.2f}"]
""",
        "torch.mps.current_allocated_memory()",
    )

    replace_once(
        TRAINING_LOOP,
        """            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
""",
        """            value = 0.0
            if (phase.start_event is not None) and (phase.end_event is not None):
""",
        "value = 0.0",
    )

    print("\nMPS compatibility patch applied successfully.")
    print("CUDA behavior remains available when STYLEGAN_DEVICE=cuda.")


if __name__ == "__main__":
    main()
