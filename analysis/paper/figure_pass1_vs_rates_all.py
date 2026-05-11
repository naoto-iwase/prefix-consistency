#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "scipy", "statsmodels"]
# ///
"""Driver: emit every pass1_vs_rates PDF + GLM slope tables for the paper.

For each of the 5 main-paper models, calls
``figure_pass1_vs_rates.run_one`` four times:

  * method=glm,    group=domain     -> pass1_vs_rates_<stem>_glm_rates.pdf
  * method=binned, group=domain     -> pass1_vs_rates_<stem>_binned_rates.pdf
  * method=lowess, group=domain     -> pass1_vs_rates_<stem>_lowess_rates.pdf
  * method=glm,    group=benchmark  -> pass1_vs_rates_<stem>_per_benchmark_rates.pdf

The two ``method=glm`` calls also surface their per-cell GLM slope fits,
which we accumulate to emit the matching tables in one pass:

  tables/table_glm_beta.tex                (per (model, domain))
  tables/table_glm_beta_per_benchmark.tex  (per (model, benchmark))

Sharing the bootstrap with the figure means the table CIs match the
panel CI bands exactly, instead of drifting from a re-bootstrap. Pass
``--no-tex`` to skip the table emission.

Filename stems match the ``\\includegraphics`` paths in main.tex.

Usage:
    cd analysis
    uv run --script paper/figure_pass1_vs_rates_all.py \\
        --root ../data-self-judge \\
        --out-dir ../overleaf/figures
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from _defs import MAIN_PAPER_MODELS, MODEL_LABELS, MODEL_STEMS
from _utils import (
    bench_sort_key, infer_benchmark_label, model_sort_key,
    setup_tex_rendering,
)
from figure_pass1_vs_rates import run_one

setup_tex_rendering()


# (method, group, filename middle).
PANELS = [
    ("glm",    "domain",    "glm"),
    ("binned", "domain",    "binned"),
    ("lowess", "domain",    "lowess"),
    ("glm",    "benchmark", "per_benchmark"),
]


# Display order within each model group, for the domain-pooled table.
DOMAIN_ROW_ORDER = ["Science", "Math"]


# ── Filename helpers ────────────────────────────────────────────────────

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


# ── GLM slope table rendering ───────────────────────────────────────────

def _format_p(p: float) -> str:
    if not isinstance(p, float) or math.isnan(p):
        return "--"
    if p < 0.001:
        return r"$<$.001"
    if p < 0.01:
        return f"${p:.3f}$"
    return f"${p:.2g}$"


def _format_beta(rec: dict) -> str:
    beta = rec["beta"]
    if not isinstance(beta, float) or math.isnan(beta):
        return "--"
    return (f"${beta:+.2f}$ "
            f"$[{rec['lo']:+.2f}, {rec['hi']:+.2f}]$")


def _render_table(rows: list[tuple[str, str, dict]],
                  caption: str, label: str,
                  group_col_header: str,
                  size: str = r"\footnotesize") -> str:
    """Render the GLM-slope LaTeX table.

    ``rows`` are (model_label, group_label, fit) triples sorted by
    (model order, group order). ``group_col_header`` is the second-
    column header (``Benchmark`` or ``Category``).
    """
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        size,
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        rf"\multirow{{2}}{{*}}{{Model}} & \multirow{{2}}{{*}}{{{group_col_header}}}"
        r" & \multicolumn{2}{c}{$\beta(r_C)$}"
        r" & \multicolumn{2}{c}{$\beta(r_W)$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" & & $\hat\beta$ [$2\sigma$ CI] & $p$"
        r"   & $\hat\beta$ [$2\sigma$ CI] & $p$ \\",
        r"\midrule",
    ]
    rows_per_model: dict[str, int] = defaultdict(int)
    for model, _, _ in rows:
        rows_per_model[model] += 1
    last_model: str | None = None
    emitted_for_model: dict[str, int] = defaultdict(int)
    for model, group, fit in rows:
        emitted_for_model[model] += 1
        if model != last_model and emitted_for_model[model] == 1:
            n = rows_per_model[model]
            model_cell = rf"\multirow{{{n}}}{{*}}{{{model}}}"
            if last_model is not None:
                lines.append(r"\addlinespace")
        else:
            model_cell = ""
        beta_c = _format_beta(fit["C"])
        beta_w = _format_beta(fit["W"])
        p_c = _format_p(fit["C"]["p"])
        p_w = _format_p(fit["W"]["p"])
        lines.append(
            f"{model_cell} & {group} & {beta_c} & {p_c} & {beta_w} & {p_w} \\\\"
        )
        last_model = model
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


CAPTION_DOMAIN = (
    r"\textbf{Logistic GLM slope estimates on "
    r"$\textnormal{logit}(r) = \beta_0 + \beta \cdot$ "
    r"\textnormal{Pass@1} per (model, category).} $2\sigma$ CIs are from "
    r"cluster bootstrap over problems (1000 replicates). The $p$-column "
    r"gives the two-sided bootstrap $p$-value for $H_0: \beta = 0$, with "
    r"``$<$.001'' indicating that no replicate crossed zero."
)
CAPTION_BENCH = (
    r"\textbf{Per-(model, benchmark) GLM slope estimates on "
    r"$\textnormal{logit}(r) = \beta_0 + \beta \cdot$ "
    r"\textnormal{Pass@1}.} Per-benchmark variant of "
    r"Table~\ref{tab:glm_beta}, fit on each (model, benchmark) cell "
    r"separately. $2\sigma$ CIs from a cluster bootstrap over problems "
    r"(1000 replicates), and two-sided bootstrap $p$-values for "
    r"$H_0: \beta = 0$ (``$<$.001'' indicates that no replicate crossed "
    r"zero)."
)


def _emit_tables(domain_rows, bench_rows, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    if domain_rows:
        domain_rows.sort(key=lambda r: (
            model_sort_key(r[0]),
            DOMAIN_ROW_ORDER.index(r[1]) if r[1] in DOMAIN_ROW_ORDER
            else len(DOMAIN_ROW_ORDER),
        ))
        out = tables_dir / "table_glm_beta.tex"
        out.write_text(_render_table(
            domain_rows, CAPTION_DOMAIN, "tab:glm_beta",
            group_col_header="Category"))
        print(f"  Saved: {out}")
    if bench_rows:
        bench_rows.sort(key=lambda r: (
            model_sort_key(r[0]), bench_sort_key(r[1])))
        out = tables_dir / "table_glm_beta_per_benchmark.tex"
        out.write_text(_render_table(
            bench_rows, CAPTION_BENCH, "tab:glm_beta_per_benchmark",
            group_col_header="Benchmark"))
        print(f"  Saved: {out}")


# ── Driver ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Emit every paper pass1_vs_rates PDF (and the two "
                    "GLM-slope tables built from the same bootstrap).")
    ap.add_argument("--root", type=Path, default=Path("../data-self-judge"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("../overleaf/figures"))
    ap.add_argument("--tables-dir", type=Path, default=None,
                    help="Where to write table_glm_beta{,_per_benchmark}.tex"
                         " (default: <out-dir>/../tables).")
    ap.add_argument("--no-tex", action="store_true",
                    help="Skip emitting the GLM-slope tables; only PDFs.")
    ap.add_argument("--condition", default="rm25pct_full_x1")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.tables_dir or (args.out_dir.parent / "tables")

    domain_rows: list[tuple[str, str, dict]] = []
    bench_rows: list[tuple[str, str, dict]] = []

    for model_label in MAIN_PAPER_MODELS:
        stem = MODEL_STEMS.get(model_label)
        if not stem:
            print(f"!! no stem mapping for {model_label}; skip")
            continue
        dirs = _model_dirs(args.root, model_label)
        if not dirs:
            print(f"!! no jsonl dirs for {model_label} under {args.root}; skip")
            continue
        for method, group, mid in PANELS:
            out = args.out_dir / f"pass1_vs_rates_{stem}_{mid}_rates.pdf"
            print(f"=== {model_label} :: {method}/{group} -> {out.name} ===")
            fits = run_one(
                dirs, out,
                condition=args.condition, model_label=model_label,
                method=method, group=group, panel="rates",
                n_boot=args.n_boot, seed=args.seed, no_title=False,
            )
            if method != "glm" or args.no_tex:
                continue
            target = domain_rows if group == "domain" else bench_rows
            for group_label, fit in fits.items():
                target.append((model_label, group_label, fit))

    if not args.no_tex:
        _emit_tables(domain_rows, bench_rows, tables_dir)


if __name__ == "__main__":
    main()
