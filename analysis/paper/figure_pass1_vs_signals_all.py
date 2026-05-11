#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "scipy", "statsmodels"]
# ///
"""Driver: emit every pass1_vs_signals PDF for the paper.

For each of the 5 main-paper models and the 2 confidence baselines
used in the paper (DeepConf tail, P(True)), calls
``figure_pass1_vs_signals.run_one`` and writes
``pass1_vs_signals_<model_stem>_<sig_stem>.pdf``.

Filename stems match the ``\\includegraphics`` paths in main.tex.

Usage:
    cd analysis
    uv run --script paper/figure_pass1_vs_signals_all.py \\
        --root ../data-self-judge \\
        --out-dir ../overleaf/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _defs import MAIN_PAPER_MODELS, MODEL_LABELS, MODEL_STEMS
from _utils import infer_benchmark_label, setup_tex_rendering
from figure_pass1_vs_signals import run_one

setup_tex_rendering()


# (filename suffix in main.tex, --signal value).
SIGNALS = [
    ("tail",  "tail_conf"),
    ("ptrue", "p_true"),
]


def _model_dirs(root: Path, model_label: str) -> list[Path]:
    keys = [k for k, v in MODEL_LABELS.items() if v == model_label]
    if not keys:
        return []
    key = keys[0]
    out: list[Path] = []
    for d in root.iterdir():
        if not d.is_dir() or not d.name.endswith("_jsonl"):
            continue
        if key not in d.name.lower():
            continue
        if infer_benchmark_label(d.name):
            out.append(d)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Emit every paper pass1_vs_signals PDF.")
    ap.add_argument("--root", type=Path, default=Path("../data-self-judge"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("../overleaf/figures"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model_label in MAIN_PAPER_MODELS:
        stem = MODEL_STEMS.get(model_label)
        if not stem:
            print(f"!! no stem mapping for {model_label}; skip")
            continue
        dirs = _model_dirs(args.root, model_label)
        if not dirs:
            print(f"!! no jsonl dirs for {model_label} under {args.root}; skip")
            continue
        for sig_stem, sig_value in SIGNALS:
            out = args.out_dir / f"pass1_vs_signals_{stem}_{sig_stem}.pdf"
            print(f"=== {model_label} :: {sig_value} -> {out.name} ===")
            run_one(
                dirs, out, signal=sig_value,
                model_label=model_label,
                n_boot=args.n_boot, seed=args.seed, no_title=False,
            )


if __name__ == "__main__":
    main()
