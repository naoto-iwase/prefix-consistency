#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["datasets", "Pillow", "requests"]
# ///
"""
Download datasets and save as local JSONL files.

Keys are preserved as-is from the source (MathArena uses "problem",
FrontierScience also uses "problem"). config.py's question_key/gold_key
must match.

Usage:
    uv run utils/download_datasets.py                       # download all
    uv run utils/download_datasets.py aime2025 brumo hmmt   # download specific ones
"""

import json
import os
import sys
from pathlib import Path

import requests
from datasets import load_dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PARENT = _REPO_ROOT.parent
OUT_ROOT = Path(os.environ.get("DATASETS_DIR", str(_DEFAULT_PARENT / "datasets")))


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records to {path}")


# -----------------------------------------------------------------
# MathArena helpers (AIME, HMMT, BRUMO share the same schema)
# -----------------------------------------------------------------
def _download_matharena(dataset_id: str, split: str, out_path: Path) -> None:
    """Download a MathArena dataset. Keeps original keys (problem, answer)."""
    ds = load_dataset(dataset_id, split=split)
    records = [{"problem": str(row["problem"]), "answer": str(row["answer"])} for row in ds]
    save_jsonl(records, out_path)


# -----------------------------------------------------------------
# AIME 2025 (MathArena/aime_2025)
# -----------------------------------------------------------------
def download_aime2025():
    _download_matharena("MathArena/aime_2025", "train", OUT_ROOT / "AIME2025" / "aime2025.jsonl")


# -----------------------------------------------------------------
# HMMT (MathArena/hmmt_feb_2026)
# -----------------------------------------------------------------
def download_hmmt():
    _download_matharena("MathArena/hmmt_feb_2026", "train", OUT_ROOT / "HMMT" / "hmmt.jsonl")


# -----------------------------------------------------------------
# BRUMO (MathArena/brumo_2025)
# -----------------------------------------------------------------
def download_brumo():
    _download_matharena("MathArena/brumo_2025", "train", OUT_ROOT / "BRUMO" / "brumo.jsonl")


# -----------------------------------------------------------------
# FrontierScience (openai/frontierscience)
# -----------------------------------------------------------------
def download_frontierscience():
    # olympiad subset only (olympiad/test.jsonl in the repo)
    url = "https://huggingface.co/datasets/openai/frontierscience/resolve/main/olympiad/test.jsonl"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    # Strip the embedded instruction suffix common to all problems
    # (starts with "\n\nThink step by step...")
    raw_rows = [json.loads(line) for line in resp.text.strip().splitlines()]
    # Detect common suffix length
    ref = raw_rows[0]["problem"]
    common = 0
    min_len = min(len(r["problem"]) for r in raw_rows)
    for i in range(1, min_len + 1):
        if all(r["problem"][-i] == ref[-i] for r in raw_rows):
            common = i
        else:
            break
    records = []
    for row in raw_rows:
        problem = row["problem"][:-common] if common else row["problem"]
        records.append({"problem": problem, "answer": row["answer"]})
    save_jsonl(records, OUT_ROOT / "FrontierScience" / "frontierscience_olympiad.jsonl")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
DOWNLOADERS = {
    "aime2025": download_aime2025,
    "hmmt": download_hmmt,
    "brumo": download_brumo,
    "frontierscience": download_frontierscience,
}

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(DOWNLOADERS.keys())
    for name in targets:
        if name not in DOWNLOADERS:
            print(f"Unknown dataset: {name}. Available: {list(DOWNLOADERS.keys())}")
            sys.exit(1)
        print(f"Downloading {name}...")
        DOWNLOADERS[name]()
    print("Done.")
