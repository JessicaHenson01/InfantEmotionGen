"""Download and normalize the private Hugging Face dataset repository."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def download_dataset_zip(
    repo_id: str,
    output_path: Path,
    revision: str | None = None,
) -> Path:
    """Download a private dataset repo and copy its only ZIP to output_path."""
    output_path = output_path.expanduser().resolve()
    snapshot_dir = output_path.parent / "_hf_snapshot"

    token = os.environ.get("HF_TOKEN")
    downloaded = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=snapshot_dir,
        token=token,
    )

    zip_files = sorted(Path(downloaded).rglob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(
            f"No ZIP file was found in Hugging Face dataset {repo_id!r}."
        )
    if len(zip_files) > 1:
        names = "\n".join(f"  - {path}" for path in zip_files)
        raise RuntimeError(
            "More than one ZIP file was found. Select the intended file "
            f"explicitly before training:\n{names}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_files[0], output_path)
    return output_path
