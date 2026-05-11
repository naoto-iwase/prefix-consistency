#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate LaTeX sensitivity table for the appendix.

Row axis:    9 ($\\tau$, $K$) configurations x 3 weight variants
             (PC-linear, PC-quadratic, PC-cubic).
Column axis: (model, benchmark) groups x token budgets.

Each cell is absolute accuracy with CI. The Standard MV baseline row
is shown at the top so readers can read the gain off directly. Bold
marks the best $(\\tau, K, \\text{weight})$ per (model, benchmark) at
each budget.

Usage:
    cd analysis && uv run python paper/table_sensitivity.py \\
        gpt-oss-20b_frontierscience_olympiad_jsonl \\
        gpt-oss-20b_hmmt_jsonl \\
        gpt-oss-20b_aime2025_jsonl \\
        gpt-oss-20b_brumo_jsonl \\
        -o tables/table_wmv_sensitivity.tex
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from _utils import tau_from_rm_pct, bench_sort_key, model_sort_key
from _utils import (
    format_acc,
    format_tp,
    infer_benchmark_label,
    infer_model_label,
    unwrap_wmv_data,
)


# ── Configuration ──

# (rm_pct, K). The condition label is built as f"rm{rm_pct}pct_full_x{K}".
TAU_K_GRID: list[tuple[int, int]] = [
    (25, 1), (25, 2), (25, 3),
    (50, 1), (50, 2), (50, 3),
    (75, 1), (75, 2), (75, 3),
]

VARIANTS: list[tuple[str, str]] = [
    ("prefix_linear",    "PC-linear"),
    ("prefix_quadratic", "PC-quadratic"),
    ("prefix_cubic",     "PC-cubic"),
]

DEFAULT_TOKEN_POINTS: list[int] = [250_000, 1_000_000, 5_000_000]


def _condition_label(rm_pct: int, K: int) -> str:
    return f"rm{rm_pct}pct_full_x{K}"


# ── Data loading ──

def load_benchmark(jsonl_dir: Path,
                   token_points: list[int]) -> dict | None:
    """Load Standard MV and PC-variant accuracies (with CI).

    Returns None when ``wmv_result.json`` is missing or empty.
    Otherwise returns ``{"sm": {tp: (acc, ci)}, "cells": {(cond, tp,
    variant_key): (acc, ci)}}``.
    """
    json_path = jsonl_dir / "wmv_result.json"
    if not json_path.exists() or json_path.stat().st_size == 0:
        return None
    try:
        raw = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return None
    tb = unwrap_wmv_data(raw, "token_budget")
    tp_set = set(token_points)
    sm_entries = (tb.get("shared_methods", {})
                    .get("standard_mv", {})
                    .get("entries", []))
    sm = {e["token_point"]: (e["acc"], e.get("ci", 0.0))
          for e in sm_entries if e["token_point"] in tp_set}
    cells: dict[tuple[str, int, str], tuple[float, float]] = {}
    for cond, cdata in tb.get("conditions", {}).items():
        methods = cdata.get("methods", {})
        for v_key, _ in VARIANTS:
            for e in methods.get(v_key, {}).get("entries", []):
                tp = e["token_point"]
                if tp in tp_set:
                    cells[(cond, tp, v_key)] = (e["acc"], e.get("ci", 0.0))
    return {"sm": sm, "cells": cells}


# ── Bold selection ──

def _best_cell_per_budget(
    cells: dict[tuple[str, int, str], tuple[float, float]],
    token_points: list[int],
) -> dict[int, tuple[str, str] | None]:
    """For each budget, return (cond, variant_key) of best accuracy."""
    best: dict[int, tuple[str, str] | None] = {}
    for tp in token_points:
        winner = None
        winner_acc = None
        for rm_pct, K in TAU_K_GRID:
            cond = _condition_label(rm_pct, K)
            for v_key, _ in VARIANTS:
                v = cells.get((cond, tp, v_key))
                if v is None:
                    continue
                if winner_acc is None or v[0] > winner_acc:
                    winner_acc = v[0]
                    winner = (cond, v_key)
        best[tp] = winner
    return best


# ── LaTeX generation ──

def _model_spans(benchmarks: list[dict]) -> list[tuple[str, int]]:
    spans: list[tuple[str, int]] = []
    for b in benchmarks:
        if spans and spans[-1][0] == b["model"]:
            spans[-1] = (b["model"], spans[-1][1] + 1)
        else:
            spans.append((b["model"], 1))
    return spans


def generate_tex(benchmarks: list[dict],
                 token_points: list[int]) -> str:
    n_src = len(benchmarks)
    n_tp = len(token_points)
    spans = _model_spans(benchmarks)
    only_one_model = len(spans) == 1
    bm_qualifier = "benchmark" if only_one_model else "(model, benchmark)"
    # When all rows belong to one model, name it in the caption (the
    # spanning header row is suppressed in that case, so the caption is
    # the only place the model appears).
    model_in_caption = f" on {spans[0][0]}" if only_one_model else ""
    budget_str = " / ".join(format_tp(tp) for tp in token_points)
    n_data_cols = 3 + n_src * n_tp  # tau, K, method, then data cells
    cmid_within_tau = f"\\cmidrule(l){{2-{n_data_cols}}}"

    bold_per_src = [
        _best_cell_per_budget(b["data"]["cells"], token_points)
        if b["data"] is not None else {}
        for b in benchmarks
    ]

    lines: list[str] = []
    lines.append("% Auto-generated by analysis/paper/table_sensitivity.py. "
                 "Do not edit by hand.")
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(
        f"\\caption{{\\textbf{{Sensitivity of PC-WMV accuracy to "
        f"$(\\tau, K)${model_in_caption} (higher is better).}} "
        f"Accuracy of PC-linear, PC-quadratic, PC-cubic at "
        f"budgets {budget_str} for every $(\\tau, K)$ in the sweep. "
        f"Standard MV at the top for reference. "
        f"Bold marks the best $(\\tau, K, \\text{{weight}})$ per "
        f"{bm_qualifier} at each budget.}}"
    )
    lines.append(r"\label{tab:wmv_sensitivity}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2.2pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{ccl *{{{n_src}}}{{{'c' * n_tp}}}}}")
    lines.append(r"\toprule")

    # Top header: model spans (skipped when there is only one model).
    if not only_one_model:
        head = "& & "
        cmid = []
        col_start = 4
        for model, span in spans:
            ncols = span * n_tp
            head += f"& \\multicolumn{{{ncols}}}{{c}}{{{model}}} "
            cmid.append(f"\\cmidrule(lr){{{col_start}-{col_start+ncols-1}}}")
            col_start += ncols
        head += r"\\"
        lines.append(head)
        lines.append(" ".join(cmid))
    else:
        # Single model: still group benchmarks under its name for clarity.
        model = spans[0][0]
        ncols = n_src * n_tp
        lines.append(
            f"& & & \\multicolumn{{{ncols}}}{{c}}{{{model}}} \\\\")
        lines.append(f"\\cmidrule(lr){{4-{3 + ncols}}}")

    # Benchmark header.
    head = "& & "
    cmid = []
    col_start = 4
    for b in benchmarks:
        head += f"& \\multicolumn{{{n_tp}}}{{c}}{{{b['bench']}}} "
        cmid.append(f"\\cmidrule(lr){{{col_start}-{col_start+n_tp-1}}}")
        col_start += n_tp
    head += r"\\"
    lines.append(head)
    lines.append(" ".join(cmid))

    # Budget header row.
    head = r"$\tau$ & $K$ & Method "
    for _ in benchmarks:
        for tp in token_points:
            head += f"& $B{{=}}${format_tp(tp)} "
    head += r"\\"
    lines.append(head)
    lines.append(r"\midrule")

    # Standard MV baseline.
    base = r"\multicolumn{3}{c}{Standard MV} "
    for b in benchmarks:
        sm = b["data"]["sm"] if b["data"] is not None else {}
        for tp in token_points:
            v = sm.get(tp)
            if v is None:
                base += "& -- "
            else:
                base += f"& {format_acc(v[0], v[1], is_best=False)} "
    base += r"\\"
    lines.append(base)
    lines.append(r"\midrule")

    # Data rows: (tau, K) blocks of 3 sub-rows (one per variant).
    prev_rm_pct = None
    for rm_pct, K in TAU_K_GRID:
        if prev_rm_pct is not None and rm_pct != prev_rm_pct:
            lines.append(r"\midrule")
        elif K != 1 and prev_rm_pct == rm_pct:
            lines.append(cmid_within_tau)
        prev_rm_pct = rm_pct
        cond = _condition_label(rm_pct, K)
        tau = tau_from_rm_pct(rm_pct)
        for vi, (v_key, v_label) in enumerate(VARIANTS):
            if vi == 0 and K == 1:
                row_start = (f"\\multirow{{9}}{{*}}{{{tau:.2f}}} "
                             f"& \\multirow{{3}}{{*}}{{{K}}} ")
            elif vi == 0:
                row_start = f" & \\multirow{{3}}{{*}}{{{K}}} "
            else:
                row_start = " & "
            row = row_start + f"& {v_label} "
            for b, bold_per_tp in zip(benchmarks, bold_per_src):
                cells = b["data"]["cells"] if b["data"] is not None else {}
                for tp in token_points:
                    v = cells.get((cond, tp, v_key))
                    if v is None:
                        row += "& -- "
                        continue
                    is_best = bold_per_tp.get(tp) == (cond, v_key)
                    row += f"& {format_acc(v[0], v[1], is_best=is_best)} "
            row += r"\\"
            lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}% end resizebox")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Entry point ──

def collect_benchmarks(dirs: list[Path],
                       analysis_dir: Path,
                       token_points: list[int]) -> list[dict]:
    """Resolve, load, and label each input directory."""
    out: "OrderedDict[Path, dict]" = OrderedDict()
    for d in dirs:
        dir_path = d if d.is_absolute() else analysis_dir / d
        if not dir_path.exists():
            print(f"  SKIP (missing): {d}")
            continue
        data = load_benchmark(dir_path, token_points)
        if data is None:
            print(f"  SKIP (no wmv_result.json): {d}")
            continue
        label = infer_benchmark_label(dir_path.name) or dir_path.name
        model = infer_model_label(dir_path.name) or "Unknown"
        n_filled = sum(1 for v in data["cells"].values() if v is not None)
        n_total = len(TAU_K_GRID) * len(VARIANTS) * len(token_points)
        print(f"  loaded {dir_path.name} -> {model} / {label}  "
              f"({n_filled}/{n_total} cells)")
        out[dir_path] = {
            "dir": dir_path,
            "model": model,
            "bench": label,
            "data": data,
        }
    items = list(out.values())
    items.sort(key=lambda b: (model_sort_key(b["model"]), bench_sort_key(b["bench"])))
    return items


def main():
    ap = argparse.ArgumentParser(
        description="Generate LaTeX sensitivity table for the appendix")
    ap.add_argument("jsonl_dirs", nargs="+", type=Path,
                    help="Analysis JSONL directories (one per benchmark)")
    ap.add_argument("--analysis-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent,
                    help="Base analysis directory (default: parent of paper/)")
    ap.add_argument("--token-points", nargs="+", type=int,
                    default=DEFAULT_TOKEN_POINTS,
                    help=f"Token budget cells (default: {DEFAULT_TOKEN_POINTS})")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Write to file instead of stdout")
    args = ap.parse_args()

    benchmarks = collect_benchmarks(
        args.jsonl_dirs, args.analysis_dir, args.token_points)
    if not benchmarks:
        raise SystemExit("No benchmarks loaded.")

    tex = generate_tex(benchmarks, args.token_points)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex + "\n")
        print(f"\nWritten: {args.output}")
    else:
        print()
        print(tex)


if __name__ == "__main__":
    main()
