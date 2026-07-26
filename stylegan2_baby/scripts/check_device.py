#!/usr/bin/env python3
"""Report local PyTorch accelerator availability."""

import platform
import sys

import torch

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    x = torch.ones(4, device="mps")
    print(f"MPS smoke tensor: {x}")
else:
    print(
        "MPS is not available in this environment. Confirm that this is an "
        "Apple Silicon Mac and that the installed PyTorch build includes MPS."
    )
