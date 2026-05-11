#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub"]
# ///
"""
Download model weights from HuggingFace Hub.

Output directory: auto-detected from repo location, or set MODELS_DIR env var.

Usage:
    uv run utils/download_models.py                   # download all
    uv run utils/download_models.py gpt-oss-20b       # download specific ones
    uv run utils/download_models.py --list            # show available models
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PARENT = _REPO_ROOT.parent
OUT_ROOT = Path(os.environ.get("MODELS_DIR", str(_DEFAULT_PARENT / "models")))

# model_name -> HuggingFace repo ID
MODELS = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "Nemotron-Nano-9B-v2": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "Nemotron-3-Nano-30B-A3B": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "Ministral-3-14B-Reasoning-2512": "mistralai/Ministral-3-14B-Reasoning-2512",
}


def download_model(name: str, repo_id: str) -> None:
    out_dir = OUT_ROOT / name
    print(f"  Downloading {repo_id} -> {out_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print(f"  Done: {out_dir}")


if __name__ == "__main__":
    if "--list" in sys.argv:
        for name, repo_id in MODELS.items():
            print(f"  {name:40s} {repo_id}")
        sys.exit(0)

    targets = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not targets:
        targets = list(MODELS.keys())

    for name in targets:
        if name not in MODELS:
            print(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
            sys.exit(1)
        print(f"Downloading {name}...")
        download_model(name, MODELS[name])
    print("Done.")
