#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "scikit-learn"]
# ///
"""5 model x 4 benchmark macro ROC grid for the appendix.

Single PDF with a 5x4 panel grid (rows = models, columns = benchmarks):
each panel overlays per-problem ROC curves (averaged on a common FPR
grid) for prefix consistency and the introspective baselines, with a
shared legend at the bottom. Companion to ``extra_figure_roc.py`` which
produces a single-cell ROC panel for diagnostic use.

The macro AUROC printed in the legend matches the convention of
Tables~\\ref{tab:signal} and~\\ref{tab:auroc}: per-problem AUROC averaged
over problems with at least one correct and one wrong initial sample.

Output (default):
    figures/signal_roc_grid.pdf

Usage:
    cd analysis
    uv run --script paper/figure_roc_all.py \\
        --root ../data-self-judge \\
        --out-dir ../../overleaf/figures \\
        --formats pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve

from _utils import (
    setup_tex_rendering,
    infer_benchmark_label,
    bench_sort_key, model_sort_key,
    find_regen_files, load_init_records, load_regen_records,
    is_usable_answer,
    METHOD_COLORS,
)
from _defs import (
    METHOD_CITATIONS,
    MAIN_PAPER_MODELS, BENCHMARK_ORDER,
    MODEL_LABELS,
)

setup_tex_rendering()


# (display_label, jsonl_key, method_key). jsonl_key="prefix" is computed
# from regen rates instead of read from init JSONL. method_key indexes
# METHOD_COLORS.
SIGNALS = [
    ("Prefix Consistency",   "prefix",        "prefix_cubic"),
    ("DeepConf tail",        "tail_conf",     "deepconf_tail"),
    ("DeepConf bottom-10\\%", "bottom10_conf", "deepconf_bottom10"),
    ("Self-certainty",       "mean_conf",     "deepconf_mean"),
    ("P(True)",              "p_true",        "p_true_raw"),
    ("Verbal 0--100",        "verbal_0_100",  "verbal_0_100_raw"),
]


def _stacked_label(display: str, method_key: str) -> str:
    cite = METHOD_CITATIONS.get(method_key, "")
    return f"{display}\n{cite}" if cite else display


def _extract_baseline_scores(records, key):
    """Return per-problem (y_pp, s_pp) dicts. Skips traces where the
    score or the parsed answer is missing."""
    y_pp: dict = {}
    s_pp: dict = {}
    for rec in records:
        pnum = rec["problem_num"]
        gold = str(rec["gold_answer"])
        if key in ("p_true", "verbal_0_100"):
            vals = rec.get(f"{key}_confidences") or []
        else:
            confs = rec.get("confidences") or []
            vals = [(c.get(key) if c else None) for c in confs]
        for i, entry in enumerate(rec["all_answers"]):
            if not is_usable_answer(entry):
                continue
            if i >= len(vals) or vals[i] is None:
                continue
            v = float(vals[i])
            if not np.isfinite(v):
                continue
            label = 1 if str(entry[0]) == gold else 0
            y_pp.setdefault(pnum, []).append(label)
            s_pp.setdefault(pnum, []).append(v)
    return y_pp, s_pp


def _extract_prefix_scores(init_records, regen_data, k):
    y_pp: dict = {}
    s_pp: dict = {}
    for rec in init_records:
        pnum = rec["problem_num"]
        gold = str(rec["gold_answer"])
        regen_rec = regen_data.get(pnum)
        if not regen_rec:
            continue
        regens = regen_rec.get("regen_answers") or []
        for i, entry in enumerate(rec["all_answers"]):
            if not is_usable_answer(entry):
                continue
            if i >= len(regens) or not regens[i]:
                continue
            init_ans = str(entry[0])
            regen_answers = [str(r[0]) for r in regens[i]
                             if is_usable_answer(r)][:k]
            if not regen_answers:
                continue
            group = [init_ans] + regen_answers
            n = len(group)
            freq = group.count(init_ans) / n
            label = 1 if init_ans == gold else 0
            y_pp.setdefault(pnum, []).append(label)
            s_pp.setdefault(pnum, []).append(freq)
    return y_pp, s_pp


def _macro_auroc(y_pp, s_pp):
    aurocs = []
    for pnum in y_pp:
        ya = np.array(y_pp[pnum])
        sa = np.array(s_pp[pnum])
        if len(ya) < 2 or len(np.unique(ya)) < 2:
            continue
        aurocs.append(roc_auc_score(ya, sa))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def _macro_roc(y_pp, s_pp, n_grid=201):
    """Per-problem ROC interpolated to a common FPR grid, then averaged."""
    fpr_grid = np.linspace(0.0, 1.0, n_grid)
    tprs = []
    for pnum in y_pp:
        ya = np.array(y_pp[pnum])
        sa = np.array(s_pp[pnum])
        if len(ya) < 2 or len(np.unique(ya)) < 2:
            continue
        fpr_i, tpr_i, _ = roc_curve(ya, sa)
        tprs.append(np.interp(fpr_grid, fpr_i, tpr_i))
    if not tprs:
        return fpr_grid, np.full_like(fpr_grid, np.nan)
    return fpr_grid, np.mean(tprs, axis=0)


def _draw_panel(ax, jsonl_dir: Path, regen_k: int) -> bool:
    """Draw one panel of overlaid macro ROC curves. Returns True if any
    curve was drawn."""
    init_records = load_init_records(jsonl_dir)
    regen_files = find_regen_files(jsonl_dir)
    regen_path = None
    for rm_pct, scope, k_avail, path in regen_files:
        if rm_pct == 25 and k_avail >= regen_k and scope == "full":
            regen_path = path
            break
    if not regen_path:
        return False
    regen_data = load_regen_records(regen_path)

    drew = False
    for display, key, method_key in SIGNALS:
        if key == "prefix":
            y_pp, s_pp = _extract_prefix_scores(
                init_records, regen_data, regen_k)
            zorder = 5
        else:
            y_pp, s_pp = _extract_baseline_scores(init_records, key)
            zorder = 3
        lw = 1.3
        if not y_pp:
            continue
        macro_auroc = _macro_auroc(y_pp, s_pp)
        if not np.isfinite(macro_auroc):
            continue
        fpr, tpr = _macro_roc(y_pp, s_pp)
        color = METHOD_COLORS.get(method_key, "#888888")
        ax.plot(fpr, tpr, color=color, lw=lw, ls="-", zorder=zorder)
        drew = True

    ax.plot([0, 1], [0, 1], "--", color="0.7", lw=0.6, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="both", labelsize=7, length=2.5, pad=2)
    ax.grid(alpha=0.3)
    return drew


def _find_jsonl_dir(root: Path, model_label: str, bench_label: str
                    ) -> Path | None:
    model_keys = [k for k, v in MODEL_LABELS.items() if v == model_label]
    if not model_keys:
        return None
    model_key = model_keys[0]
    candidates = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.endswith("_jsonl"):
            continue
        if model_key not in child.name.lower():
            continue
        if infer_benchmark_label(child.name) != bench_label:
            continue
        candidates.append(child)
    if not candidates:
        return None
    return sorted(candidates)[0]


def main():
    parser = argparse.ArgumentParser(
        description="5x4 macro ROC grid (one figure for the appendix)")
    parser.add_argument("--root", type=Path,
                        default=Path("../data-self-judge"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("../../overleaf/figures"))
    parser.add_argument("--regen-k", type=int, default=1)
    parser.add_argument("--formats", nargs="+", default=["pdf"])
    parser.add_argument("--stem", default="signal_roc_grid")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        sys.exit(f"ERROR: --root {root} does not exist")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    benches = sorted(BENCHMARK_ORDER, key=bench_sort_key)
    models = sorted(MAIN_PAPER_MODELS, key=model_sort_key)
    n_rows, n_cols = len(models), len(benches)

    # Sized so equal-aspect panels with matched wspace/hspace come out
    # roughly square; legend gets a 0.45 row to fit two rows.
    fig_w, fig_h = 7.0, 9.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=n_rows + 1, ncols=n_cols,
        height_ratios=[1.0] * n_rows + [0.55],
        left=0.12, right=0.99, top=0.97, bottom=0.02,
        wspace=0.18, hspace=0.18,
    )
    legend_ax = fig.add_subplot(gs[n_rows, :])
    legend_ax.axis("off")

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    for r, model in enumerate(models):
        for c, bench in enumerate(benches):
            ax = axes[r, c]
            jsonl_dir = _find_jsonl_dir(root, model, bench)
            print(f"  [{r},{c}] {model} / {bench}: {jsonl_dir}")
            if jsonl_dir is None or not _draw_panel(ax, jsonl_dir,
                                                    args.regen_k):
                ax.set_axis_off()
                continue
            if r == 0:
                ax.set_title(bench.replace("~", " "), fontsize=10, pad=4)
            if r == n_rows - 1:
                ax.set_xlabel("False positive rate", fontsize=8)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel("True positive rate", fontsize=8)
            else:
                ax.set_yticklabels([])
        leftmost = axes[r, 0]
        leftmost.text(-0.45, 0.5, model,
                      transform=leftmost.transAxes,
                      rotation=90, ha="center", va="center",
                      fontsize=12, fontweight="bold")

    handles, labels = [], []
    for display, _key, method_key in SIGNALS:
        color = METHOD_COLORS.get(method_key, "#888888")
        handles.append(Line2D([0], [0], color=color, lw=1.3, ls="-"))
        labels.append(_stacked_label(display, method_key))
    legend_ax.legend(handles, labels, loc="center", ncol=3,
                     bbox_to_anchor=(0.5, 0.35),
                     fontsize=9, frameon=False,
                     handlelength=1.4, columnspacing=1.5,
                     handletextpad=0.4, labelspacing=0.4)

    for fmt in args.formats:
        out = out_dir / f"{args.stem}.{fmt}"
        fig.savefig(out, dpi=300)
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
