#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""5 model x 4 benchmark cost-accuracy grid with the full baseline set.

Companion to ``figure_cost_accuracy_all.py``: same 5x4 layout but draws
every baseline (PC linear/quadratic/cubic, Standard MV, all five
DeepConf aggregations and their filtered variants, the four CISC
verbalized-confidence raws, AC and ESC sweeps), plus the oracle upper
bounds. Colors are grouped by method family so the figure stays
readable despite the larger curve count: greens for PC, blues for
DeepConf, purples for DeepConf filtered, oranges for CISC, gray for
Standard MV, red for AC sweep, dark purple for ESC sweep, and faded
dotted lines for the oracles.

Output (default):
    figures/cost_accuracy_grid_full.pdf

Usage:
    cd analysis
    uv run --script paper/figure_cost_accuracy_full.py \\
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
    METHOD_COLORS,
)
from _defs import (
    MAIN_PAPER_MODELS, BENCHMARK_ORDER,
    MODEL_LABELS,
    NATURAL_FAMILIES, NATURAL_MARKERS,
    ORACLE_KEYS,
)

setup_tex_rendering()


# Back-to-front: PC last (on top). DeepConf order from paper Sec 3.2.
DRAW_ORDER = [
    "oracle_init", "oracle_prefix",
    "standard_mv",
    "markers",
    "deepconf_first_token", "deepconf_mean", "deepconf_bottom10",
    "deepconf_block_min", "deepconf_tail",
    "deepconf_bottom10_top10pct", "deepconf_bottom10_top90pct",
    "deepconf_tail_top10pct", "deepconf_tail_top90pct",
    "response_prob_raw", "verbal_binary_raw",
    "verbal_0_100_raw", "p_true_raw",
    # ac_sweep, esc_sweep are drawn separately from natural-stopping.
    "prefix_linear", "prefix_quadratic", "prefix_cubic",
]


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


def _style_for(key: str, color: str) -> dict:
    """Per-curve linewidth / linestyle / zorder.

    All main methods share lw to keep visual weight equal; zorder
    layering keeps PC visible on top when curves overlap. Oracles use
    a thinner dotted style to mark them as upper-bound references.
    SubthoughtReasoner is dashed to disambiguate from CISC oranges.
    """
    if key in ORACLE_KEYS:
        return dict(color=color, lw=0.7, ls=(0, (1, 2)),
                    marker="", zorder=2, alpha=0.8)
    if key.startswith("prefix_"):
        return dict(color=color, lw=1.0, ls="-",
                    marker="", zorder=10, alpha=0.95)
    if key == "markers":
        return dict(color=color, lw=1.0, ls=(0, (4, 2)),
                    marker="", zorder=5, alpha=0.95)
    if key == "standard_mv":
        return dict(color=color, lw=1.0, ls="-",
                    marker="", zorder=6, alpha=0.9)
    return dict(color=color, lw=1.0, ls="-",
                marker="", zorder=4, alpha=0.9)


def _draw_panel(ax, jsonl_dir: Path, condition: str) -> bool:
    wmv_path = jsonl_dir / "wmv_result.json"
    pass1 = get_pass_at_1(jsonl_dir)
    plateau_max = -np.inf
    drew = False

    for key in DRAW_ORDER:
        data = _resolve_method_entries(wmv_path, key, condition)
        if data is None:
            continue
        tok, acc, ci = data
        st = _style_for(key, METHOD_COLORS[key])
        ax.plot(tok, acc * 100, **st)
        # Skip CI band for oracles (already visually busy enough).
        if key not in ORACLE_KEYS:
            ax.fill_between(tok, (acc - ci) * 100,
                            (acc + ci) * 100,
                            color=st["color"], alpha=0.08,
                            zorder=st["zorder"] - 1, linewidth=0)
        drew = True
        if key not in ORACLE_KEYS:
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
                    color=color, ecolor="black", lw=1.0, ls="-",
                    marker=NATURAL_MARKERS.get(family, "o"), markersize=2.5,
                    elinewidth=0.5, capsize=1.2, capthick=0.5,
                    zorder=5, alpha=0.9)
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


# One column per method family; entries fall back to per-entry
# citations when not listed at the group level.
LEGEND_COLUMNS = [
    {
        "title": "Prefix consistency",
        "cite": "(ours)",
        "entries": [
            ("PC-linear", "prefix_linear", "line"),
            ("PC-quadratic", "prefix_quadratic", "line"),
            ("PC-cubic", "prefix_cubic", "line"),
        ],
    },
    {
        "title": "DeepConf",
        "cite": "(Fu et al. 2026)",
        "entries": [
            ("first-token", "deepconf_first_token", "line"),
            ("Self-certainty", "deepconf_mean", "line"),
            (r"bottom-10\%", "deepconf_bottom10", "line"),
            ("block-min", "deepconf_block_min", "line"),
            ("tail", "deepconf_tail", "line"),
        ],
    },
    {
        "title": "DeepConf filtered",
        "cite": "(Fu et al. 2026)",
        "entries": [
            (r"bottom-10\% (top-10\%)",
             "deepconf_bottom10_top10pct", "line"),
            (r"bottom-10\% (top-90\%)",
             "deepconf_bottom10_top90pct", "line"),
            (r"tail (top-10\%)", "deepconf_tail_top10pct", "line"),
            (r"tail (top-90\%)", "deepconf_tail_top90pct", "line"),
        ],
    },
    {
        "title": "CISC",
        "cite": "(Taubenfeld et al. 2025)",
        "entries": [
            ("Response prob. (Wang et al. 2023)",
             "response_prob_raw", "line"),
            ("Verbal binary (Lin et al. 2022)",
             "verbal_binary_raw", "line"),
            ("Verbal 0--100 (Lin et al. 2022)",
             "verbal_0_100_raw", "line"),
            ("P(True) (Kadavath et al. 2022)",
             "p_true_raw", "line"),
        ],
    },
    {
        "title": "Adaptive stopping",
        "cite": "",
        "entries": [
            ("AC sweep (Aggarwal et al. 2023)", "ac_sweep", "marker"),
            ("ESC sweep (Li et al. 2024)", "esc_sweep", "marker"),
        ],
    },
    {
        "title": "Other",
        "cite": "",
        "entries": [
            ("Standard MV (Wang et al. 2023)", "standard_mv", "line"),
            ("SubthoughtReasoner (Hammoud et al. 2025)",
             "markers", "line"),
            ("Oracle (Standard MV)", "oracle_init", "line"),
            ("Oracle (Prefix Consistency)",
             "oracle_prefix", "line"),
        ],
    },
]


def _split_label(label: str, max_inline: int = 22) -> str:
    """Wrap an entry label onto two lines at the citation boundary."""
    if " (" not in label or len(label) <= max_inline:
        return label
    base, rest = label.split(" (", 1)
    return f"{base}\n({rest}"


# Per-column widths sized to each column's widest entry.
COL_RATIOS = [1.3, 1.6, 1.6, 1.7, 1.4, 1.8]


def _draw_custom_legend(legend_ax):
    """One column per family, COL_RATIOS-scaled widths, two-line entry
    labels wrapped at the citation boundary."""
    legend_ax.axis("off")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    total = sum(COL_RATIOS)
    norm = [r / total for r in COL_RATIOS]
    col_x = []
    acc = 0.0
    for r in norm:
        col_x.append(acc)
        acc += r

    title_y = 0.96
    cite_y = 0.83
    first_entry_y = 0.68
    line_h = 0.155
    line_x_off = 0.006
    line_w = 0.026
    text_x_off = 0.038

    for ci, col in enumerate(LEGEND_COLUMNS):
        x0 = col_x[ci]
        legend_ax.text(x0 + line_x_off, title_y, col["title"],
                       fontsize=8.5, fontweight="bold",
                       ha="left", va="center",
                       transform=legend_ax.transAxes)
        if col["cite"]:
            legend_ax.text(x0 + line_x_off, cite_y, col["cite"],
                           fontsize=7.0, color="#555",
                           ha="left", va="center",
                           transform=legend_ax.transAxes)
            entry_y = first_entry_y
        else:
            entry_y = first_entry_y + line_h * 0.45
        for label, key, kind in col["entries"]:
            color = METHOD_COLORS[key]
            st = _style_for(key, color)
            text_label = _split_label(label)
            is_two_line = "\n" in text_label
            x_a = x0 + line_x_off
            x_b = x0 + line_x_off + line_w
            line = Line2D([x_a, x_b], [entry_y, entry_y],
                          color=st["color"], lw=st["lw"],
                          ls=st["ls"],
                          marker=NATURAL_MARKERS.get(key, "")
                                  if kind == "marker" else "",
                          markersize=3.0,
                          transform=legend_ax.transAxes,
                          clip_on=False)
            legend_ax.add_line(line)
            legend_ax.text(x0 + text_x_off, entry_y, text_label,
                           fontsize=7.0, ha="left", va="center",
                           linespacing=1.05,
                           transform=legend_ax.transAxes)
            # Two-line entries take ~2x the vertical room of single-line
            # ones; bump the next entry's y by an extra 0.04 so wrapped
            # citations never collide with the following entry.
            entry_y -= line_h + (0.045 if is_two_line else 0.0)


def main():
    parser = argparse.ArgumentParser(
        description="5x4 cost-accuracy grid with the full baseline set")
    parser.add_argument("--root", type=Path,
                        default=Path("../data-self-judge"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("../overleaf/figures"))
    parser.add_argument("--condition", default="rm25pct_full_x1")
    parser.add_argument("--formats", nargs="+", default=["pdf"])
    parser.add_argument("--stem", default="cost_accuracy_grid_full")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        sys.exit(f"ERROR: --root {root} does not exist")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    benches = sorted(BENCHMARK_ORDER, key=bench_sort_key)
    models = sorted(MAIN_PAPER_MODELS, key=model_sort_key)
    n_rows, n_cols = len(models), len(benches)

    # Larger figure than the 6-curve variant to accommodate the multi-row
    # legend strip at the bottom.
    fig_w, fig_h = 11.0, 10.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=n_rows + 1, ncols=n_cols,
        height_ratios=[1.0] * n_rows + [0.85],
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
                ax.set_title(bench.replace("~", " "), fontsize=10, pad=4)
            if r == n_rows - 1:
                ax.set_xlabel("Tokens / problem", fontsize=8)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel("Accuracy (\\%)", fontsize=8)
        leftmost = axes[r, 0]
        leftmost.text(-0.45, 0.5, model,
                      transform=leftmost.transAxes,
                      rotation=90, ha="center", va="center",
                      fontsize=12, fontweight="bold")

    _draw_custom_legend(legend_ax)

    for fmt in args.formats:
        out = out_dir / f"{args.stem}.{fmt}"
        fig.savefig(out, dpi=300)
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
