# Reliable Chain-of-Thought via Prefix Consistency

Naoto Iwase, Yuki Ichihara, Mohammad Atif Quamar, Junpei Komiyama

[arXiv](https://arxiv.org/abs/2605.07654) | [Project page](https://naoto-iwase.github.io/prefix-consistency-page) | [Data (Zenodo)](https://doi.org/10.5281/zenodo.20082164)

When we truncate a Chain-of-Thought (CoT) partway through and regenerate the remainder, traces with correct answers reproduce their original answer more often than traces with wrong answers. We use this gap as a reliability signal, *prefix consistency*, that weights majority voting without access to logits or self-rating prompts.

## Repository structure

```
generation/             # Inference pipeline (initial generation, regeneration, enrichment)
analysis/               # WMV evaluation and paper figures/tables (scripts only)
utils/                  # Dataset and model download scripts
```

See each subdirectory's README for details:
- [analysis/README.md](analysis/README.md): WMV evaluation, paper figures and tables, per-cell data layout.
- [generation/README.md](generation/README.md): inference pipeline, vLLM server, and `run_regen.sh` for regenerating the JSONLs from scratch (GPU required).

## Reproducing the paper from the released data (no GPU)

Per-cell JSONLs and `wmv_result.json` files for all 5 models × 4 benchmarks under both self-judge and external judge are published on Zenodo at https://doi.org/10.5281/zenodo.20082164. From the repository root:

```bash
curl -L 'https://zenodo.org/records/20082164/files/prefix-consistency-data.zip?download=1' -o prefix-consistency-data.zip
unzip prefix-consistency-data.zip   # creates data-self-judge/ and data-external-judge/
uv sync                             # requires uv (https://docs.astral.sh/uv/)
bash analysis/run_paper_all.sh
```

Writes every table and figure into `paper-out/{tables,figures}/`.

## Workspace layout

```
<parent>/                  # auto-detected as workspace root
├── prefix-consistency/    # this repo
├── models/                # model weights (only needed to re-run generation)
└── datasets/              # evaluation datasets (only needed to re-run generation)
```

`models/` and `datasets/` are only needed if you want to re-run `generation/`. To regenerate paper figures and tables from the released data, neither is required.

<details>
<summary>Using a different storage location</summary>

```bash
# Option 1: symlink
ln -s /data/shared/models  <parent>/models
ln -s /data/shared/datasets <parent>/datasets

# Option 2: env vars (per command or in .bashrc)
export MODELS_DIR=/data/shared/models
export DATASETS_DIR=/data/shared/datasets
```

</details>

## Citation

```bibtex
@article{iwase2026prefixconsistency,
  title   = {Reliable Chain-of-Thought via Prefix Consistency},
  author  = {Naoto Iwase and Yuki Ichihara and Mohammad Atif Quamar and Junpei Komiyama},
  journal = {arXiv preprint arXiv:2605.07654},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.07654},
}
```
