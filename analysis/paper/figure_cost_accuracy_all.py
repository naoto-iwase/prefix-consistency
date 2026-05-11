#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""5 model x 4 benchmark cost-accuracy grid for the appendix.

Single PDF with a 5x4 panel grid (rows = models, columns = benchmarks):
column headers on the top row, model labels rotated on the left edge,
shared horizontal legend at the bottom, and y/x-axis labels only on the
boundary panels. Companion to ``figure_cost_accuracy.py`` which produces
the single-cell main-paper Fig 1 panels.

Output (default):
    figures/cost_accuracy_grid.pdf

Usage:
    cd analysis
    uv run --script paper/figure_cost_accuracy_all.py \\
        --root ../data-self-judge \\
        --out-dir ../overleaf/figures \\
        --formats pdf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _utils import (
    setup_tex_rendering,
    infer_benchmark_label,
    bench_sort_key, model_sort_key,
    load_natural_stopping_curves,
    get_pass_at_1,
    METHOD_COLORS, METHOD_DISPLAY,
)
from _defs import (
    METHOD_CITATIONS,
    MAIN_PAPER_MODELS, BENCHMARK_ORDER,
    MODEL_LABELS,
    HEADLINE_METHOD_KEYS,
    NATURAL_FAMILIES, NATURAL_MARKERS,
)

setup_tex_rendering()


METHODS = list(HEADLINE_METHOD_KEYS)


def _stacked_label(key: str) -> str:
    base = METHOD_DISPLAY.get(key, key)
    cite = METHOD_CITATIONS.get(key, "")
    return f"{base}\n{cite}" if cite else base


def _resolve_method_entries(wmv_path: Path, key: str, condition: str):
    with open(wmv_path) as f:
        d = json.load(f)
    if "token_budget_dense" not in d:
        return None
    tbd = d["token_budget_dense"]
    m = tbd.get("shared_methods", {}).get(key)
    if m is None:
        m = (tbd.get("conditions", {}).get(condition, {})
             .get("methods", {}).get(key))
    if m is None or not m.get("entries"):
        return None
    entries = m["entries"]
    tok = np.array([e["mean_tokens"] for e in entries], dtype=float)
    acc = np.array([e["acc"] for e in entries], dtype=float)
    ci = np.array([e.get("ci", 0.0) for e in entries], dtype=float)
    order = np.argsort(tok)
    return tok[order], acc[order], ci[order]


def _draw_panel(ax, jsonl_dir: Path, condition: str) -> bool:
    """Draw one cost-accuracy panel. Returns True if any curve was drawn."""
    wmv_path = jsonl_dir / "wmv_result.json"
    drew = False
    plateau_max = -np.inf
    pass1 = get_pass_at_1(jsonl_dir)
    for key in METHODS:
        data = _resolve_method_entries(wmv_path, key, condition)
        if data is None:
            continue
        tok, acc, ci = data
        color = METHOD_COLORS.get(key, "#888888")
        lw = 1.3
        zorder = 5 if key.startswith("prefix") else 3
        ax.plot(tok, acc * 100, color=color, lw=lw, ls="-", zorder=zorder)
        ax.fill_between(tok, (acc - ci) * 100, (acc + ci) * 100,
                        color=color, alpha=0.12, zorder=zorder - 1,
                        linewidth=0)
        drew = True
        plateau_max = max(plateau_max, float(acc.max()))

    natural_curves = load_natural_stopping_curves(wmv_path)
    for family in NATURAL_FAMILIES:
        entries = natural_curves.get(family, [])
        if not entries:
            continue
        tok = np.array([e[0] for e in entries], dtype=float)
        acc = np.array([e[1] for e in entries], dtype=float)
        ci = np.array([e[2] for e in entries], dtype=float)
        cost_ci = np.array([e[3] for e in entries], dtype=float)
        order = np.argsort(tok)
        tok = tok[order]
        acc = acc[order]
        ci = ci[order]
        cost_ci = cost_ci[order]
        color = METHOD_COLORS.get(family, "#888888")
        ax.errorbar(tok, acc * 100, yerr=ci * 100, xerr=cost_ci,
                    color=color, ecolor="black", lw=1.3, ls="-",
                    marker=NATURAL_MARKERS.get(family, "o"), markersize=3.0,
                    elinewidth=0.6, capsize=1.5, capthick=0.6,
                    zorder=4)
        drew = True
        plateau_max = max(plateau_max, float(acc.max()))

    if pass1 is not None and plateau_max > pass1:
        gap = plateau_max - pass1
        y0 = max(0.0, pass1 - 0.10 * gap)
        y1 = min(1.0, plateau_max + 0.15 * gap)
        ax.set_ylim(y0 * 100, y1 * 100)

    ax.set_xscale("log")
    ax.set_xlim(1e3, 1e7)
    ax.grid(which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=7, length=2.5, pad=2)
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
        description="5x4 cost-accuracy grid (one figure for the appendix)")
    parser.add_argument("--root", type=Path,
                        default=Path("../data-self-judge"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("../overleaf/figures"))
    parser.add_argument("--condition", default="rm25pct_full_x1")
    parser.add_argument("--formats", nargs="+", default=["pdf"])
    parser.add_argument("--stem", default="cost_accuracy_grid")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        sys.exit(f"ERROR: --root {root} does not exist")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    benches = sorted(BENCHMARK_ORDER, key=bench_sort_key)
    models = sorted(MAIN_PAPER_MODELS, key=model_sort_key)
    n_rows, n_cols = len(models), len(benches)

    # Reserve bottom strip for the legend via gridspec; height ratio
    # keeps the legend at ~0.35 inch regardless of overall figsize.
    # Extra left padding leaves room for rotated row labels (model
    # names) outside the leftmost-column y-axis ticks.
    fig_w, fig_h = 11.0, 9.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=n_rows + 1, ncols=n_cols,
        height_ratios=[1.0] * n_rows + [0.32],
        left=0.10, right=0.99, top=0.97, bottom=0.04,
        wspace=0.20, hspace=0.32,
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
            if jsonl_dir is None or not _draw_panel(ax, jsonl_dir,
                                                   args.condition):
                ax.set_axis_off()
                continue
            if r == 0:
                ax.set_title(bench.replace("~", " "), fontsize=10,
                             pad=4)
            if r == n_rows - 1:
                ax.set_xlabel("Tokens / problem", fontsize=8)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel("Accuracy (\\%)", fontsize=8)
        # Row label outside the leftmost panel's y-axis tick labels.
        leftmost = axes[r, 0]
        leftmost.text(-0.45, 0.5, model,
                      transform=leftmost.transAxes,
                      rotation=90, ha="center", va="center",
                      fontsize=12, fontweight="bold")

    handles, labels = [], []
    for key in METHODS:
        color = METHOD_COLORS.get(key, "#888888")
        lw = 1.3
        handles.append(Line2D([0], [0], color=color, lw=lw, ls="-"))
        labels.append(_stacked_label(key))
    for family in NATURAL_FAMILIES:
        color = METHOD_COLORS.get(family, "#888888")
        handles.append(Line2D([0], [0], color=color, lw=1.3, ls="-",
                              marker=NATURAL_MARKERS.get(family, "o"),
                              markersize=3.0))
        labels.append(_stacked_label(family))
    legend_ax.legend(handles, labels, loc="center", ncol=len(handles),
                     fontsize=9, frameon=False,
                     handlelength=1.6, columnspacing=1.4,
                     handletextpad=0.4, labelspacing=0.2)

    for fmt in args.formats:
        out = out_dir / f"{args.stem}.{fmt}"
        fig.savefig(out, dpi=300)
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
