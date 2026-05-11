#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "scipy"]
# ///
"""5 model x 4 benchmark scatter of per-problem init vs regen Pass@1.

Companion to ``figure_cost_accuracy_oracle.py``: shows that the
marginal correctness rate of regenerated answers matches that of
initial answers at the per-problem level. Each panel scatters one
point per problem, with x-axis the fraction of correct init answers
and y-axis the fraction of correct K=1 regen answers (one regen per
init), at tau=0.75. A y=x diagonal is drawn for reference, and the
panel-level means (mean init, mean regen) are reported in the panel
title.

Also runs a paired Wilcoxon signed-rank test on the per-problem
differences within each cell, plus a two-sided binomial sign test on
the cell-level direction across the 20 cells. The aggregate stats are
printed to stdout and back the statistical claims in the
corresponding appendix subsection.

Output (default):
    figures/pool_correctness_scatter.pdf

Usage:
    cd analysis
    uv run --script paper/figure_pool_correctness_scatter.py \\
        --root ../data-self-judge \\
        --out-dir ../overleaf/figures \\
        --formats pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _utils import (
    setup_tex_rendering,
    infer_benchmark_label,
    bench_sort_key, model_sort_key,
    find_regen_for_condition, load_regen_records,
    is_usable_answer,
)
from _defs import (
    MAIN_PAPER_MODELS, BENCHMARK_ORDER,
    MODEL_LABELS,
)

setup_tex_rendering()


def _per_problem_rates(regen_path: Path) -> list[tuple[float, float]]:
    """Read a regen JSONL and return per-problem (init, regen) Pass@1 pairs.

    For each problem, init_rate = (# i: all_answers[i][0] == gold) / N_init
    and regen_rate = (# i: regen_answers[i][0][0] == gold) / N_regen, where
    the denominators count only positions with usable answers on that side.
    Problems with no usable init or no usable regen are skipped.
    """
    rates = []
    for rec in load_regen_records(regen_path).values():
        gold = str(rec["gold_answer"])
        n_init = init_correct = n_regen = regen_correct = 0
        for i, init_pair in enumerate(rec["all_answers"]):
            if is_usable_answer(init_pair):
                n_init += 1
                if str(init_pair[0]) == gold:
                    init_correct += 1
            if i < len(rec["regen_answers"]):
                rs = rec["regen_answers"][i]
                if rs:
                    r0 = rs[0]
                    if is_usable_answer(r0):
                        n_regen += 1
                        if str(r0[0]) == gold:
                            regen_correct += 1
        if n_init == 0 or n_regen == 0:
            continue
        rates.append((init_correct / n_init, regen_correct / n_regen))
    return rates


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


def _find_rm25_x1(jsonl_dir: Path) -> Path | None:
    return find_regen_for_condition(jsonl_dir, "rm25pct_full_x1")


def _draw_panel(ax, rates: list[tuple[float, float]],
                bonferroni_alpha: float) -> dict | None:
    """Draw one panel and return its summary stats (or None if empty)."""
    if not rates:
        ax.set_axis_off()
        return None
    xs = np.array([r[0] for r in rates])
    ys = np.array([r[1] for r in rates])
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.7, ls="--",
            zorder=1, alpha=0.8)
    ax.scatter(xs, ys, s=10, color="#1f7a1f", alpha=0.45,
               edgecolors="none", zorder=3)
    mx, my = float(xs.mean()), float(ys.mean())
    # Panel-mean marker: filled diamond at the cell mean of (pi(a*), pi^->(a*)).
    # Dotted projection lines drop from the diamond to both axes, and the
    # exact mean values are printed at the axis intersections.
    ax.plot([mx, mx], [0.0, my], color="#a63603", lw=0.6,
            ls=(0, (1, 1.5)), alpha=0.55, zorder=4)
    ax.plot([0.0, mx], [my, my], color="#a63603", lw=0.6,
            ls=(0, (1, 1.5)), alpha=0.55, zorder=4)
    ax.scatter([mx], [my], s=44, color="#a63603", marker="D",
               edgecolors="white", linewidths=0.6, zorder=5)
    label_box = dict(boxstyle="round,pad=0.12",
                     facecolor="white", edgecolor="none", alpha=0.8)
    # When both means are small, the per-axis labels crowd near the
    # origin and overlap each other. In that case, drop a single
    # combined "(mx, my)" annotation just above the diamond and skip
    # the per-axis labels.
    if mx < 0.20 and my < 0.20:
        ax.text(mx + 0.04, my + 0.04,
                f"$({mx:.2f},\\,{my:.2f})$",
                color="#a63603", fontsize=6.5,
                ha="left", va="bottom", zorder=6, bbox=label_box)
    else:
        ax.text(mx, 0.03, f"{mx:.2f}", color="#a63603", fontsize=6.5,
                ha="center", va="bottom", zorder=6, bbox=label_box)
        ax.text(0.03, my, f"{my:.2f}", color="#a63603", fontsize=6.5,
                ha="left", va="center", zorder=6, bbox=label_box)
    # Significance marker: paired Wilcoxon signed-rank on per-problem
    # (init - regen) differences. Cells that reject equality at the
    # Bonferroni-corrected alpha get an upper-left annotation showing
    # the p-value in scientific notation.
    diffs = xs - ys
    p_val = (float(stats.wilcoxon(xs, ys,
                                  alternative="two-sided").pvalue)
             if not np.allclose(diffs, 0.0) else 1.0)
    if p_val < bonferroni_alpha:
        exp_neg = int(np.ceil(-np.log10(p_val)))
        mantissa = p_val * 10 ** exp_neg
        ax.text(0.05, 0.95,
                f"$\\ast~p\\!=\\!{mantissa:.1f}"
                f"{{\\times}}10^{{-{exp_neg}}}$",
                transform=ax.transAxes,
                color="#a63603", fontsize=8,
                ha="left", va="top", zorder=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=7, length=2.5, pad=2)
    return {
        "n": len(rates),
        "mean_init": mx,
        "mean_regen": my,
        "mean_diff": mx - my,
        "wilcoxon_p": p_val,
    }


def _print_aggregate_stats(cells, bonferroni_alpha):
    """Print per-cell stats and the across-cell sign test."""
    print()
    print(f"# Per-cell paired Wilcoxon (init - regen)")
    print(f"# Bonferroni alpha = {bonferroni_alpha:.4g}")
    header = ("model", "benchmark", "n", "mean_init", "mean_regen",
              "mean_diff", "wilcoxon_p", "sig")
    widths = [16, 26, 4, 10, 11, 10, 12, 4]
    fmt_row = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt_row.format(*header))
    print(fmt_row.format(*("-" * w for w in widths)))
    n_sig = n_init_above = n_regen_above = 0
    for c in cells:
        sig = c["wilcoxon_p"] < bonferroni_alpha
        if sig:
            n_sig += 1
        if c["mean_diff"] > 0:
            n_init_above += 1
        elif c["mean_diff"] < 0:
            n_regen_above += 1
        print(fmt_row.format(
            c["model"], c["bench"], c["n"],
            f"{c['mean_init']:.3f}", f"{c['mean_regen']:.3f}",
            f"{c['mean_diff']:+.3f}", f"{c['wilcoxon_p']:.2e}",
            "*" if sig else "."))
    print()
    print(f"# Significant cells: {n_sig} / {len(cells)}")
    print(f"# init mean > regen mean: {n_init_above}")
    print(f"# regen mean > init mean: {n_regen_above}")
    n_signed = n_init_above + n_regen_above
    if n_signed > 0:
        binom = stats.binomtest(n_init_above, n_signed, p=0.5,
                                alternative="two-sided")
        print(f"# Sign test (k of n signed cells): "
              f"k={n_init_above}, n={n_signed}, p={binom.pvalue:.3g}")


def main():
    parser = argparse.ArgumentParser(
        description="5x4 per-problem init-vs-regen Pass@1 scatter")
    parser.add_argument("--root", type=Path,
                        default=Path("../data-self-judge"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("../overleaf/figures"))
    parser.add_argument("--formats", nargs="+", default=["pdf"])
    parser.add_argument("--stem", default="pool_correctness_scatter")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        sys.exit(f"ERROR: --root {root} does not exist")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    benches = sorted(BENCHMARK_ORDER, key=bench_sort_key)
    models = sorted(MAIN_PAPER_MODELS, key=model_sort_key)
    n_rows, n_cols = len(models), len(benches)
    # Bonferroni correction over the full 5x4 grid.
    bonferroni_alpha = 0.05 / (n_rows * n_cols)

    # Square panels keep the y=x diagonal at 45 degrees. Inner-panel
    # tick labels are hidden so wspace/hspace can stay tight without
    # collisions; only the boundary panels carry tick labels.
    # fig_w is sized so that with equal-aspect panels the column slots
    # are nearly square (matching the row slots), which keeps the visual
    # column gap close to the visual row gap. Reducing fig_w further
    # makes the rendered figure taller on the page; this value is the
    # smallest fig_w that keeps the column gap from looking pinched.
    fig_w, fig_h = 6.5, 7.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=n_rows, ncols=n_cols,
        left=0.10, right=0.99, top=0.97, bottom=0.05,
        wspace=0.18, hspace=0.18,
    )

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    cells = []
    for r, model in enumerate(models):
        for c, bench in enumerate(benches):
            ax = axes[r, c]
            jsonl_dir = _find_jsonl_dir(root, model, bench)
            if jsonl_dir is None:
                ax.set_axis_off()
                continue
            regen_path = _find_rm25_x1(jsonl_dir)
            if regen_path is None:
                ax.set_axis_off()
                continue
            rates = _per_problem_rates(regen_path)
            stats_row = _draw_panel(ax, rates, bonferroni_alpha)
            if stats_row is not None:
                cells.append({"model": model, "bench": bench,
                              **stats_row})
            if r == 0:
                ax.set_title(bench.replace("~", " "), fontsize=10, pad=4)
            if r == n_rows - 1:
                ax.set_xlabel(r"$\hat\pi(a^\star)$", fontsize=9)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(r"$\widehat{\pi^{\rightarrow}}(a^\star)$",
                              fontsize=9)
            else:
                ax.set_yticklabels([])
        leftmost = axes[r, 0]
        leftmost.text(-0.40, 0.5, model,
                      transform=leftmost.transAxes,
                      rotation=90, ha="center", va="center",
                      fontsize=12, fontweight="bold")

    for fmt in args.formats:
        out = out_dir / f"{args.stem}.{fmt}"
        fig.savefig(out, dpi=300)
        print(f"  Saved: {out}")
    plt.close(fig)

    _print_aggregate_stats(cells, bonferroni_alpha)


if __name__ == "__main__":
    main()
