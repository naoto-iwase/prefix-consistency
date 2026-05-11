#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Judge-robustness artifacts for the appendix.

Compares evaluation under self-judge against re-scoring with an external
LLM judge. Reads two parallel data roots and produces:

  1. Lead scatter (PDF): one point per cell. ``x`` is PC's
     $\\overline{\\mathrm{AUROC}}$ lead over the best per-cell baseline
     under the model's own judge; ``y`` is the same lead under the
     external judge. Quadrants encode preserved vs.\\ flipped sign.

  2. WMV cost--accuracy grid (PDF): a model x benchmark grid; each
     panel overlays self-judge curves (faded) on top of external-judge
     curves (saturated). The four methods plotted match
     ``figure_cost_accuracy_all.py``.

  3. LaTeX table: per-cell PC and best-baseline
     $\\overline{\\mathrm{AUROC}}$ under both judges.

The model and benchmark sets follow ``MAIN_PAPER_MODELS`` and
``BENCHMARK_ORDER`` from ``_defs.py``, with one of each excluded for
this comparison: AIME~2025 is exact-match (judge-free), and Ministral
is excluded by ``--exclude-model``.

Usage:
    cd analysis
    uv run --script paper/figure_judge_robustness.py \\
        --self-root ../data-self-judge \\
        --external-root ../data-external-judge \\
        --out-fig-lead ../../overleaf/figures/fig_judge_robustness_lead.pdf \\
        --out-fig-grid ../../overleaf/figures/fig_judge_robustness_grid.pdf \\
        --out-table   ../../overleaf/tables/table_judge_robustness.tex
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
    find_init_file, find_regen_for_condition,
    bench_sort_key, model_sort_key,
    infer_benchmark_label,
    METHOD_COLORS, METHOD_DISPLAY,
)
from _defs import (
    MAIN_PAPER_MODELS, BENCHMARK_ORDER,
    MODEL_LABELS,
    SIGNAL_KEY_TO_METHOD,
    METHOD_CITATIONS,
    HEADLINE_METHOD_KEYS,
    BENCHMARK_COLOR,
)
from table_auroc import compute_baseline_aurocs
from table_transition import compute_transition_rates

setup_tex_rendering()


EXCLUDE_MODELS = {"Ministral3-14B"}
EXCLUDE_BENCHMARKS = {"AIME~2025"}  # exact-match -> judge-free

GRID_METHODS = list(HEADLINE_METHOD_KEYS)

# Lead scatter: color = benchmark (BENCHMARK_COLOR), marker = model.
# GPT-OSS uses square/diamond, Nemotron uses up/down triangle.
MODEL_MARKER_LEAD = {
    "GPT-OSS-120B":  "s",
    "GPT-OSS-20B":   "D",
    "Nemotron3-30B": "^",
    "Nemotron2-9B":  "v",
}

SIGNAL_TO_DISPLAY = {"PC": "PC"}
for _sig_key, _meth_key in SIGNAL_KEY_TO_METHOD.items():
    if _meth_key in METHOD_DISPLAY and _sig_key in {
        "mean_conf", "tail_conf", "bottom10_conf", "p_true", "verbal_0_100",
    }:
        SIGNAL_TO_DISPLAY[_sig_key] = METHOD_DISPLAY[_meth_key]

DENSE = "token_budget_dense"


# ── Helpers ──────────────────────────────────────────────────────────────

def _selected_models() -> list[str]:
    return [m for m in sorted(MAIN_PAPER_MODELS, key=model_sort_key)
            if m not in EXCLUDE_MODELS]


def _selected_benches() -> list[str]:
    return [b for b in sorted(BENCHMARK_ORDER, key=bench_sort_key)
            if b not in EXCLUDE_BENCHMARKS]


def _find_jsonl_dir(root: Path, model_label: str, bench_label: str
                    ) -> Path | None:
    """Find the per-cell JSONL directory under ``root`` for one cell.

    Mirrors ``figure_cost_accuracy_all.py``: matches the model via the
    dir-name substring registered in ``MODEL_LABELS`` and the benchmark
    via ``infer_benchmark_label``.
    """
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


# ── Per-cell collection (AUROC) ──────────────────────────────────────────

def _collect_aurocs(jsonl_dir: Path, condition: str) -> dict:
    """Return {signal_key: AUROC or None} including PC."""
    out: dict[str, float | None] = {k: None for k in SIGNAL_TO_DISPLAY}
    regen_path = find_regen_for_condition(jsonl_dir, condition)
    if regen_path:
        rates = compute_transition_rates(regen_path, K=1)["rates"]
        rc, rw = rates.get("c_to_c"), rates.get("w_to_same_w")
        D = (rc - rw) if (rc is not None and rw is not None) else None
        out["PC"] = ((1 + D) / 2) if D is not None else None
    init_path = find_init_file(jsonl_dir)
    if init_path:
        for k, v in compute_baseline_aurocs(init_path).items():
            if k in SIGNAL_TO_DISPLAY:
                out[k] = v
    return out


def _best_baseline(aurocs: dict[str, float | None]
                   ) -> tuple[str, float | None]:
    bls = {k: v for k, v in aurocs.items() if k != "PC" and v is not None}
    if not bls:
        return "", None
    bk = max(bls, key=lambda k: bls[k])
    return SIGNAL_TO_DISPLAY.get(bk, bk), bls[bk]


def collect(self_root: Path, external_root: Path, condition: str,
            skip: set[tuple[str, str]]) -> list[tuple]:
    rows = []
    for model in _selected_models():
        for bench in _selected_benches():
            if (model, bench) in skip:
                continue
            sd = _find_jsonl_dir(self_root, model, bench)
            ad = _find_jsonl_dir(external_root, model, bench)
            if sd is None or ad is None:
                print(f"  skip {model}/{bench}: missing data dir")
                continue
            print(f"  collecting {model}/{bench} ...", flush=True)
            asd = _collect_aurocs(sd, condition)
            ada = _collect_aurocs(ad, condition)
            rows.append((model, bench, asd, ada))
    return rows


# ── Lead scatter ─────────────────────────────────────────────────────────

def make_lead_scatter(rows, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    lo, hi = -0.10, 0.25
    ax.axhline(0, color="#888", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#888", linewidth=0.6, zorder=0)
    ax.plot([lo, hi], [lo, hi], "--", color="#aaa", linewidth=0.7, zorder=1)
    ax.fill_between([0, hi], [0, 0], [hi, hi], color="#2ca02c",
                    alpha=0.06, zorder=0)
    ax.fill_between([lo, 0], [lo, lo], [0, 0], color="#d62728",
                    alpha=0.06, zorder=0)

    for model, bench, asd, ada in rows:
        _, best_s = _best_baseline(asd)
        _, best_a = _best_baseline(ada)
        pc_s, pc_a = asd.get("PC"), ada.get("PC")
        if None in (pc_s, pc_a, best_s, best_a):
            continue
        ls, la = pc_s - best_s, pc_a - best_a
        ax.scatter([ls], [la], marker=MODEL_MARKER_LEAD[model], s=70,
                   facecolor=BENCHMARK_COLOR[bench],
                   edgecolor="black", linewidth=0.6, alpha=0.9, zorder=3)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\Delta\overline{\mathrm{AUROC}}$ under self-judge")
    ax.set_ylabel(r"$\Delta\overline{\mathrm{AUROC}}$ under external judge")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.text(hi - 0.005, hi - 0.005,
            r"$\Delta\overline{\mathrm{AUROC}} > 0$ under both judges",
            ha="right", va="top", fontsize=7.5, color="#2ca02c", alpha=0.95)
    ax.text(lo + 0.005, lo + 0.005,
            r"$\Delta\overline{\mathrm{AUROC}} < 0$ under both judges",
            ha="left", va="bottom", fontsize=7.5, color="#d62728", alpha=0.95)

    bench_handles = [Line2D([0], [0], marker="o", color="none",
                            markerfacecolor=BENCHMARK_COLOR[b], markersize=8,
                            markeredgecolor="black", markeredgewidth=0.5,
                            label=b.replace("~", " "))
                     for b in _selected_benches()]
    model_handles = [Line2D([0], [0], marker=MODEL_MARKER_LEAD[m],
                            color="none", markerfacecolor="#777",
                            markersize=8, markeredgecolor="black",
                            markeredgewidth=0.5, label=m)
                     for m in _selected_models()]
    leg1 = ax.legend(handles=bench_handles, title="Benchmark",
                     loc="upper left", fontsize=7.5, title_fontsize=7.5,
                     framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Model",
              loc="lower right", bbox_to_anchor=(1.0, 0.033),
              fontsize=7.5, title_fontsize=7.5,
              framealpha=0.9, ncol=2)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved: {out_path}")


# ── WMV cost--accuracy grid ──────────────────────────────────────────────

def _resolve_method_entries(wmv_path: Path, key: str, condition: str):
    with open(wmv_path) as f:
        d = json.load(f)
    if DENSE not in d:
        return None
    tbd = d[DENSE]
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


def _draw_grid_panel(ax, self_dir: Path, external_dir: Path, condition: str,
                     pending: bool):
    if pending:
        ax.text(0.5, 0.5, "rejudge in progress",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#888", style="italic")
        ax.set_xscale("log"); ax.set_xlim(1e3, 1e7)
        ax.tick_params(axis="both", labelsize=7, length=2.5, pad=2)
        ax.grid(which="both", alpha=0.3)
        return

    plateau_max = -np.inf
    pass1_min = +np.inf
    for tag, jsonl_dir in [("self", self_dir), ("external", external_dir)]:
        wmv = jsonl_dir / "wmv_result.json"
        from _utils import get_pass_at_1
        p1 = get_pass_at_1(jsonl_dir)
        if p1 is not None:
            pass1_min = min(pass1_min, p1)
        for key in GRID_METHODS:
            data = _resolve_method_entries(wmv, key, condition)
            if data is None:
                continue
            tok, acc, ci = data
            color = METHOD_COLORS.get(key, "#888888")
            is_pc = key.startswith("prefix")
            zorder = (5 if is_pc else 3) + (1 if tag == "external" else 0)
            alpha = 0.30 if tag == "self" else 0.95
            ax.plot(tok, acc * 100, color=color, lw=1.3, ls="-",
                    alpha=alpha, zorder=zorder)
            # CI band, attenuated for self so external (focus) reads
            # cleanly; same alpha pairing as the curve itself.
            band_alpha = 0.04 if tag == "self" else 0.12
            ax.fill_between(tok, (acc - ci) * 100, (acc + ci) * 100,
                            color=color, alpha=band_alpha,
                            zorder=zorder - 1, linewidth=0)
            plateau_max = max(plateau_max, float(acc.max()))

    if pass1_min < np.inf and plateau_max > pass1_min:
        gap = plateau_max - pass1_min
        y0 = max(0.0, pass1_min - 0.10 * gap)
        y1 = min(1.0, plateau_max + 0.15 * gap)
        ax.set_ylim(y0 * 100, y1 * 100)

    ax.set_xscale("log")
    ax.set_xlim(1e3, 1e7)
    ax.grid(which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=7, length=2.5, pad=2)


def make_wmv_grid(self_root: Path, external_root: Path, condition: str,
                  pending: set[tuple[str, str]], out_path: Path):
    models = _selected_models()
    benches = _selected_benches()
    n_rows, n_cols = len(models), len(benches)
    fig_w, fig_h = 10.5, 8.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=n_rows + 1, ncols=n_cols,
        height_ratios=[1.0] * n_rows + [0.45],
        left=0.10, right=0.99, top=0.96, bottom=0.04,
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
            sd = _find_jsonl_dir(self_root, model, bench)
            ad = _find_jsonl_dir(external_root, model, bench)
            if sd is None or ad is None:
                ax.set_axis_off()
                continue
            _draw_grid_panel(ax, sd, ad, condition,
                             pending=(model, bench) in pending)
            if r == 0:
                ax.set_title(bench.replace("~", " "), fontsize=10, pad=4)
            if r == n_rows - 1:
                ax.set_xlabel("Tokens / problem", fontsize=8)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel("Accuracy (\\%)", fontsize=8)
        first = axes[r, 0]
        first.text(-0.32, 0.5, model,
                   transform=first.transAxes, ha="center", va="center",
                   fontsize=10, rotation=90)

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS.get(k, "#888"), lw=2,
               label=METHOD_DISPLAY.get(k, k)
               + (f" {METHOD_CITATIONS[k]}" if k in METHOD_CITATIONS else ""))
        for k in GRID_METHODS
    ]
    style_handles = [
        Line2D([0], [0], color="#444", lw=1.4, alpha=0.30,
               label="self-judge"),
        Line2D([0], [0], color="#444", lw=1.4, alpha=0.95,
               label="external judge (Claude Sonnet 4.6)"),
    ]
    leg1 = legend_ax.legend(handles=method_handles, loc="upper center",
                            ncol=4, fontsize=8, frameon=False,
                            bbox_to_anchor=(0.5, 1.0))
    legend_ax.add_artist(leg1)
    legend_ax.legend(handles=style_handles, loc="lower center",
                     ncol=2, fontsize=8, frameon=False,
                     bbox_to_anchor=(0.5, 0.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved: {out_path}")


# ── Table (AUROC-only) ───────────────────────────────────────────────────

def _fmt(v, prec=3):
    return f"{v:.{prec}f}" if v is not None else "--"


def _bold_if(txt: str, cond: bool) -> str:
    return r"\textbf{" + txt + "}" if cond else txt


def make_table(rows, out_path: Path):
    # Group rows by model (outer), preserving canonical orders.
    by_model: dict[str, list] = {}
    for model, bench, asd, ada in rows:
        by_model.setdefault(model, []).append((bench, asd, ada))
    model_order = [m for m in _selected_models() if m in by_model]
    bench_order = _selected_benches()
    for m in model_order:
        by_model[m].sort(key=lambda be: bench_order.index(be[0]))

    lines = []
    lines.append(r"% Auto-generated by figure_judge_robustness.py")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{\textbf{Per-cell PC and best-baseline "
                 r"$\overline{\mathrm{AUROC}}$ under self-judge vs.\ an "
                 r"external judge (Claude Sonnet 4.6).} Bold marks the "
                 r"higher of PC and the best baseline within each cell, "
                 r"with the best baseline's identity in parentheses. "
                 r"AIME~2025 is exact-match (judge-free) and is omitted.}")
    lines.append(r"\label{tab:judge_robustness}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll cc cc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{Model} & \multirow{2}{*}{Benchmark} & "
                 r"\multicolumn{2}{c}{PC} & "
                 r"\multicolumn{2}{c}{best baseline} \\")
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    lines.append(r" & & self & external & self & external \\")
    lines.append(r"\midrule")

    for i, model in enumerate(model_order):
        if i > 0:
            lines.append(r"\addlinespace")
        cells = by_model[model]
        n = len(cells)
        for j, (bench, asd, ada) in enumerate(cells):
            bn_s, best_s = _best_baseline(asd)
            bn_a, best_a = _best_baseline(ada)
            pc_s, pc_a = asd.get("PC"), ada.get("PC")
            # Bold the winner of each (PC vs.\ best-baseline) comparison.
            pc_wins_s = pc_s is not None and best_s is not None and pc_s > best_s
            pc_wins_a = pc_a is not None and best_a is not None and pc_a > best_a
            base_wins_s = pc_s is not None and best_s is not None and best_s > pc_s
            base_wins_a = pc_a is not None and best_a is not None and best_a > pc_a
            pc_s_txt = _bold_if(_fmt(pc_s), pc_wins_s)
            pc_a_txt = _bold_if(_fmt(pc_a), pc_wins_a)
            bb_s_txt = f"{_bold_if(_fmt(best_s), base_wins_s)}\\,({bn_s})"
            bb_a_txt = f"{_bold_if(_fmt(best_a), base_wins_a)}\\,({bn_a})"
            left = f"\\multirow{{{n}}}{{*}}{{{model}}}" if j == 0 else ""
            lines.append(f"{left} & {bench} & {pc_s_txt} & {pc_a_txt} & "
                         f"{bb_s_txt} & {bb_a_txt} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"saved: {out_path}")


# ── Driver ───────────────────────────────────────────────────────────────

def _parse_cells(seq, valid_models, valid_benches):
    """Parse 'Model/Benchmark' strings, validating against the canonical
    label sets."""
    out = set()
    for s in seq:
        if "/" not in s:
            sys.exit(f"--skip/--pending entry must be 'Model/Bench': {s}")
        m, b = s.split("/", 1)
        if m not in valid_models:
            sys.exit(f"--skip/--pending: unknown model '{m}'. "
                     f"Choices: {sorted(valid_models)}")
        if b not in valid_benches:
            sys.exit(f"--skip/--pending: unknown benchmark '{b}'. "
                     f"Choices: {sorted(valid_benches)}")
        out.add((m, b))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-root", required=True, type=Path)
    p.add_argument("--external-root", required=True, type=Path)
    p.add_argument("--condition", default="rm25pct_full_x1")
    p.add_argument("--skip", nargs="*", default=[],
                   help="Cells to omit from the lead scatter and table, "
                        "as '<Model display label>/<Benchmark display label>' "
                        "(use the same labels as in MAIN_PAPER_MODELS / "
                        "BENCHMARK_ORDER, e.g. 'GPT-OSS-20B/FrontierScience-Olympiad').")
    p.add_argument("--pending", nargs="*", default=[],
                   help="Cells to render as pending (placeholder text) in "
                        "the WMV grid; same format as --skip.")
    p.add_argument("--out-fig-lead", required=True, type=Path)
    p.add_argument("--out-fig-grid", required=True, type=Path)
    p.add_argument("--out-table",    required=True, type=Path)
    args = p.parse_args()

    valid_models = set(_selected_models())
    valid_benches = set(_selected_benches())
    skip = _parse_cells(args.skip, valid_models, valid_benches)
    pending = _parse_cells(args.pending, valid_models, valid_benches)

    rows = collect(args.self_root, args.external_root, args.condition, skip)
    if not rows:
        print("No cells collected; aborting.", file=sys.stderr)
        sys.exit(1)
    make_lead_scatter(rows, args.out_fig_lead)
    make_table(rows, args.out_table)
    make_wmv_grid(args.self_root, args.external_root, args.condition,
                  pending, args.out_fig_grid)


if __name__ == "__main__":
    main()
