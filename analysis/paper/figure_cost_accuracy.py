#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Generate cost-equivalent accuracy curve for the paper's Fig 1.

Plots accuracy vs total tokens (log x) for a fixed (model, benchmark)
using the dense-grid wmv evaluation. Reads `token_budget_dense` from
wmv_result.json (run `wmv.py --dense` to populate). Methods with zero
dense entries are skipped.

Usage:
    cd analysis
    uv run --script paper/figure_cost_accuracy.py \\
        gpt-oss-120b_frontierscience_olympiad_jsonl/ \\
        --formats pdf png \\
        --title "GPT-OSS-120B / FrontierScience"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _utils import (
    setup_tex_rendering,
    infer_model_label, infer_benchmark_label,
    load_natural_stopping_curves,
    get_pass_at_1,
    METHOD_COLORS, METHOD_DISPLAY,
)
from _defs import (
    METHOD_CITATIONS,
    HEADLINE_METHOD_KEYS,
    NATURAL_FAMILIES, NATURAL_MARKERS,
)

setup_tex_rendering()


# Wider than square: x is log-tokens (2-4 decades), y is a narrow
# accuracy band. Fixed margins (no bbox_inches='tight') keep the axes
# box identical across panels regardless of xlim/ylim.
COST_ACC_FIGSIZE = (5.0, 2.7)
COST_ACC_MARGINS = dict(left=0.09, right=0.97, top=0.822, bottom=0.20)


# PC first so it heads the legend; visual draw order is via zorder.
DEFAULT_METHODS = list(HEADLINE_METHOD_KEYS)


def _legend_label(key: str) -> str:
    base = METHOD_DISPLAY.get(key, key)
    cite = METHOD_CITATIONS.get(key)
    return f"{base} {cite}" if cite else base


def _resolve_method_entries(wmv_path: Path, key: str, condition: str):
    """Return (tok, acc, ci) for `key`, trying shared then per-condition.

    Returns None if the method has no dense entries in either location.
    """
    with open(wmv_path) as f:
        d = json.load(f)
    if "token_budget_dense" not in d:
        sys.exit(f"ERROR: {wmv_path} has no token_budget_dense. "
                 f"Re-run wmv.py with --dense.")
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


def main():
    parser = argparse.ArgumentParser(description="Cost-accuracy curve")
    parser.add_argument("jsonl_dir", type=Path)
    parser.add_argument("--condition", default="rm25pct_full_x1",
                        help="τ/K condition for prefix methods (default: "
                             "rm25pct_full_x1, i.e. τ=0.75 K=1)")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("-o", "--out-dir", type=Path, default=None)
    parser.add_argument("--stem", default="cost_accuracy")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xlim", nargs=2, type=float, default=None,
                        help="Override token-axis limits (e.g. 1e4 1e7)")
    parser.add_argument("--ylim", nargs=2, type=float, default=None,
                        help="Accuracy axis limits in percent (e.g. 28 52)")
    parser.add_argument("--headline-alpha", type=float, default=None,
                        help="If set (e.g. 0.99), draw a horizontal reference "
                             "line at alpha * MV-plateau, drop vertical lines "
                             "where each curve first crosses it, and label "
                             "the token-savings ratio.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        help=f"Method json_keys to plot in order (default: "
                             f"{' '.join(DEFAULT_METHODS)}). Display labels "
                             f"and colors are looked up from "
                             f"_defs.METHOD_DISPLAY / METHOD_COLORS.")
    parser.add_argument("--headline-pc-key", default=None,
                        help="Method key used as the PC reference for the "
                             "speedup headline (default: first prefix_* in "
                             "--methods, if any).")
    parser.add_argument("--no-legend", action="store_true",
                        help="Skip the in-panel legend. Use this when "
                             "panels share an external legend strip.")
    parser.add_argument("--legend-out", type=Path, default=None,
                        help="If set, also save a horizontal legend-only "
                             "strip (using all curves drawn on this panel) "
                             "to this stem. Implies --no-legend on the "
                             "panel itself. Files: <stem>.pdf, <stem>.png.")
    args = parser.parse_args()

    jsonl_dir = args.jsonl_dir
    out_dir = args.out_dir or (jsonl_dir / "plots_signal_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    wmv_path = jsonl_dir / "wmv_result.json"
    if not wmv_path.exists():
        sys.exit(f"ERROR: {wmv_path} not found")

    fig, ax = plt.subplots(figsize=COST_ACC_FIGSIZE)

    series = {}
    for key in args.methods:
        display = METHOD_DISPLAY.get(key, key)
        data = _resolve_method_entries(wmv_path, key, args.condition)
        if data is None:
            print(f"  WARNING: {display} ({key}): no dense entries in "
                  f"{wmv_path.parent.name}; omitted from the figure. "
                  f"Re-run `wmv.py {jsonl_dir.name} --dense` to populate.",
                  file=sys.stderr)
            continue
        tok, acc, ci = data
        color = METHOD_COLORS.get(key, "#888888")
        lw = 1.6
        zorder = 5 if key.startswith("prefix") else 3
        ax.plot(tok, acc * 100, color=color, lw=lw, ls="-", zorder=zorder,
                label=_legend_label(key))
        ax.fill_between(tok, (acc - ci) * 100, (acc + ci) * 100,
                        color=color, alpha=0.12, zorder=zorder - 1,
                        linewidth=0)
        print(f"  {display}: {len(tok)} pts, acc {acc.min()*100:.1f}-"
              f"{acc.max()*100:.1f}%")
        series[key] = (display, tok, acc, color)

    # AC/ESC operating points; registered in `series` so the crossing
    # block below also reads them.
    natural_curves = load_natural_stopping_curves(wmv_path)
    for family, entries in natural_curves.items():
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
        display = METHOD_DISPLAY.get(family, family)
        ax.errorbar(tok, acc * 100, yerr=ci * 100, xerr=cost_ci,
                    color=color, ecolor="black", lw=1.6, ls="-",
                    marker=NATURAL_MARKERS.get(family, "o"), markersize=4,
                    elinewidth=0.8, capsize=2, capthick=0.8,
                    zorder=4, label=_legend_label(family))
        print(f"  {display}: {len(entries)} pts, acc "
              f"{acc.min()*100:.1f}-{acc.max()*100:.1f}%")
        series[family] = (display, tok, acc, color)

    # Set limits first; axvline's ymax= below is in axes fraction.
    ax.set_xscale("log")
    if args.xlim:
        ax.set_xlim(*args.xlim)
    if args.ylim:
        ax.set_ylim(*args.ylim)

    # Headline annotation at pass@1 + alpha * (plateau - pass@1),
    # matching table_token_savings.tex.
    if args.headline_alpha is not None and "standard_mv" in series:
        def _envelope(acc):
            return np.maximum.accumulate(acc)

        def _budget_at(tok, env, target):
            mask = env >= target
            if not mask.any():
                return None
            i = int(np.argmax(mask))
            if i == 0:
                return float(tok[0])
            t1, a1 = tok[i - 1], env[i - 1]
            t2, a2 = tok[i], env[i]
            if a2 <= a1:
                return float(t2)
            frac = (target - a1) / (a2 - a1)
            return float(np.exp(np.log(t1)
                                + frac * (np.log(t2) - np.log(t1))))

        p1_acc = get_pass_at_1(jsonl_dir)
        _, _, mv_acc, _ = series["standard_mv"]
        # Plateau = MV's acc at B_max (matches table_token_savings).
        plateau = float(mv_acc[-1])
        gap = plateau - p1_acc
        target = p1_acc + args.headline_alpha * gap

        ax.axhline(target * 100, ls=":", color="0.4", lw=0.8, zorder=1)

        def _fmt(t):
            if t >= 1e6:
                return f"{t/1e6:.1f}M"
            if t >= 1e3:
                return f"{t/1e3:.0f}k"
            return f"{t:.0f}"

        crossings = {}
        for key, (display, tok, acc, color) in series.items():
            env = _envelope(acc)
            b = _budget_at(tok, env, target)
            if b is None:
                continue
            crossings[key] = b
            y0, y1 = ax.get_ylim()
            ymax_frac = (target * 100 - y0) / (y1 - y0)
            ax.axvline(b, ymax=ymax_frac,
                       color=color, ls="--", lw=0.9, alpha=0.7, zorder=2)
            ax.scatter([b], [target * 100], color=color, s=22,
                       zorder=6, edgecolor="white", linewidth=0.8)

        # Per-marker numeric labels around the dotted line. Adjacent
        # labels can collide on log-x (when crossings are clustered).
        # We pack labels into slots: slot 0=above row 0, 1=below row 0,
        # 2=above row 1, 3=below row 1, ... and assign each label the
        # smallest slot that does not collide with already-placed labels
        # in the same slot. This spreads dense clusters across both sides
        # of the target line instead of stacking everything upward.
        sorted_keys = sorted(crossings, key=lambda k: crossings[k])
        BASE_DY = 4         # offset_pt of innermost row
        ROW_DY = 12         # vertical spacing between rows
        # Negative PAD allows label bboxes to abut or slightly overlap
        # before stacking. Round-corner bboxes look generous on screen
        # but the visible text is narrower; otherwise common cases like
        # "57k" "95k" stack unnecessarily.
        PAD_PX = -8

        # First pass: place all labels, render, measure widths, then drop.
        renderer = fig.canvas.get_renderer()
        bbox_centers_px: list[float] = []
        bbox_half_widths_px: list[float] = []
        for key in sorted_keys:
            txt = ax.annotate(_fmt(crossings[key]),
                              xy=(crossings[key], target * 100),
                              xytext=(0, BASE_DY),
                              textcoords="offset points",
                              ha="center", va="bottom",
                              fontsize=9, fontweight="bold",
                              color=METHOD_COLORS.get(key, "#444"),
                              zorder=20,
                              bbox=dict(boxstyle="round,pad=0.15",
                                        fc="white", ec="none", alpha=0.85))
            bb = txt.get_window_extent(renderer=renderer)
            bbox_centers_px.append(0.5 * (bb.x0 + bb.x1))
            bbox_half_widths_px.append(0.5 * (bb.x1 - bb.x0))
            txt.remove()

        def _slot_offset(slot: int) -> tuple[float, str]:
            row = slot // 2
            above = (slot % 2 == 0)
            dy = (BASE_DY + row * ROW_DY) * (1 if above else -1)
            return dy, ("bottom" if above else "top")

        placed: list[tuple[int, float, float]] = []  # (slot, cx, hw)
        slots: list[int] = []
        for cx, hw in zip(bbox_centers_px, bbox_half_widths_px):
            slot = 0
            while any(s == slot
                      and abs(cx - ocx) < (hw + ohw + PAD_PX)
                      for (s, ocx, ohw) in placed):
                slot += 1
            placed.append((slot, cx, hw))
            slots.append(slot)

        for key, slot in zip(sorted_keys, slots):
            color = METHOD_COLORS.get(key, "#444")
            dy, va = _slot_offset(slot)
            ax.annotate(_fmt(crossings[key]),
                        xy=(crossings[key], target * 100),
                        xytext=(0, dy),
                        textcoords="offset points",
                        ha="center", va=va,
                        fontsize=9, fontweight="bold", color=color,
                        zorder=20,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="white", ec="none", alpha=0.85))

        headline_pc_key = args.headline_pc_key
        if headline_pc_key is None:
            pc_keys = [k for k in args.methods if k.startswith("prefix_")]
            headline_pc_key = pc_keys[0] if pc_keys else ""
        if (headline_pc_key and headline_pc_key in crossings
                and "standard_mv" in crossings):
            ratio = crossings["standard_mv"] / crossings[headline_pc_key]
            print(f"  Speedup {headline_pc_key} vs MV at "
                  f"alpha={args.headline_alpha}: {ratio:.2f}x")
            speedup_ratio = ratio
            speedup_color = METHOD_COLORS.get(headline_pc_key, "#2ca02c")
        else:
            speedup_ratio = None
            speedup_color = "#2ca02c"
    else:
        speedup_ratio = None
        speedup_color = "#2ca02c"

    ax.set_xlabel("Tokens per problem (log scale)", fontsize=10)
    ax.set_ylabel(r"Accuracy (\%)", fontsize=10)
    title = args.title
    if title is None:
        model = infer_model_label(jsonl_dir.name)
        bench = infer_benchmark_label(jsonl_dir.name)
        if model and bench:
            title = f"{model} / {bench}"
    if title:
        if speedup_ratio is not None:
            # Two-line title: line 1 = model/bench (small, gray title),
            # line 2 = speedup headline (bold, just above axes).
            ax.set_title(title, fontsize=10, pad=18)
            headline = (f"PC-WMV uses {speedup_ratio:.1f}$\\times$ fewer "
                        f"tokens than Standard MV")
            if plt.rcParams.get("text.usetex", False):
                # fontweight="bold" is ignored in usetex mode; wrap with
                # \textbf so LaTeX actually renders the headline bold.
                headline = rf"\textbf{{{headline}}}"
            ax.text(0.5, 1.01, headline,
                    transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold",
                    color=speedup_color)
        else:
            ax.set_title(title, fontsize=10, pad=6)
    ax.grid(which="both", alpha=0.3)

    skip_legend = args.no_legend or args.legend_out is not None
    if not skip_legend:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

    fig.subplots_adjust(**COST_ACC_MARGINS)
    for fmt in args.formats:
        out = out_dir / f"{args.stem}.{fmt}"
        fig.savefig(out, dpi=300)
        print(f"  Saved: {out}")
    plt.close(fig)

    # Stand-alone legend strip with stacked (name, citation) entries
    # so six entries fit in a narrow strip (used by main.tex side-by-side).
    if args.legend_out is not None:
        handles, labels = [], []
        from matplotlib.lines import Line2D
        def _stacked_label(key):
            base = METHOD_DISPLAY.get(key, key)
            cite = METHOD_CITATIONS.get(key, "")
            return f"{base}\n{cite}" if cite else base
        for key in args.methods:
            if key not in series:
                continue
            color = METHOD_COLORS.get(key, "#888888")
            lw = 1.6
            handles.append(Line2D([0], [0], color=color, lw=lw, ls="-"))
            labels.append(_stacked_label(key))
        for family in NATURAL_FAMILIES:
            if family not in series:
                continue
            color = METHOD_COLORS.get(family, "#888888")
            handles.append(Line2D([0], [0], color=color, lw=1.6, ls="-",
                                  marker=NATURAL_MARKERS.get(family, "o"),
                                  markersize=4))
            labels.append(_stacked_label(family))
        fig_leg = plt.figure(figsize=(6.5, 0.95))
        fig_leg.legend(handles, labels, loc="center",
                       ncol=len(handles), fontsize=10, frameon=False,
                       handlelength=1.6, columnspacing=1.0,
                       handletextpad=0.4, labelspacing=0.2)
        for fmt in args.formats:
            leg_out = args.legend_out.with_suffix(f".{fmt}")
            leg_out.parent.mkdir(parents=True, exist_ok=True)
            fig_leg.savefig(leg_out, dpi=300, bbox_inches="tight",
                            pad_inches=0.02)
            print(f"  Legend saved: {leg_out}")
        plt.close(fig_leg)


if __name__ == "__main__":
    main()
