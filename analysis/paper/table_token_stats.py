#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate LaTeX token-statistics table from analysis JSONL dirs.

Row axis:    Model x Benchmark
Column axis: $N$, Init, Regen (tau=0.75 / 0.50 / 0.25), Verbal queries
             (Verbal 0--100 actual tokens, Binary actual tokens).

Init tokens come from the per-problem init JSONL (`all_answers[i][1]`).
Regen tokens come from each regen condition JSONL
(`regen_answers[i][r][1]`, i.e. new tokens generated after the kept prefix).
Verbal queries are secondary completions made on each init sample to
elicit a confidence value: ``verbal_0_100_actual_tokens`` for the
0--100 confidence prompt and ``binary_query_actual_tokens`` for the
binary "Is your answer correct? 0/1" prompt (whose response logprob also
serves as the P(True) signal). Both columns are means of the actual
generated tokens (whether or not the parser later succeeded).
By default only $K{=}1$ regen conditions are used; override with --K.

Usage:
    cd analysis && uv run python paper/table_token_stats.py \\
        gpt-oss-120b_frontierscience_olympiad_jsonl \\
        gpt-oss-120b_aime2025_jsonl \\
        Nemotron-3-Nano-30B-A3B_hmmt_jsonl \\
        -o tables/table_token_stats.tex
"""

import argparse
import json
import statistics
from collections import OrderedDict
from pathlib import Path

from _utils import (
    bench_sort_key, model_sort_key, tau_from_rm_pct,
    detect_subset_footnote,
    find_init_file,
    find_regen_for_condition,
    infer_benchmark_label,
    infer_model_label,
    load_base_cells_checked, fallback_cell,
)


# ── Merge-base helpers ──

def _strip_row_markers(first_cell: str) -> str:
    import re
    s = first_cell.strip()
    s = re.sub(r"^(?:\\(?:midrule|toprule|bottomrule|addlinespace(?:\[[^\]]*\])?))\s*",
               "", s)
    m = re.match(r"^\\multirow\{\d+\}\{\*\}\{(.*)\}\s*$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    return s


def _tokenstats_group_key(cells):
    """Emit a new group when cells[0] carries either a ``\\multirow`` model
    label (multi-benchmark model) or a bare non-empty model name
    (single-benchmark model written without ``\\multirow``)."""
    first = cells[0].strip()
    if not first:
        return None
    return _strip_row_markers(first)


def _tokenstats_row_key(cells, current_group):
    """Row key = (model, benchmark). Benchmark is cells[1] with any
    ``$^{x}$`` footnote marker stripped so the key is stable across
    rerunning with different subsets."""
    if current_group is None or len(cells) < 2:
        return None
    import re
    bench = re.sub(r"\s*\$\^\{[^}]*\}\$\s*$", "", cells[1].strip())
    if not bench:
        return None
    return (current_group, bench)


TAU_COLS = [0.75, 0.50, 0.25]  # default regen columns


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def load_init_tokens(jsonl_dir: Path) -> tuple[int | None, int | None]:
    """Return (N, mean_init_tokens) for the init JSONL in *jsonl_dir*.

    ``N`` is the maximum ``len(all_answers)`` across problems, which
    represents the intended sample budget (a few problems may lose
    samples to upstream failures).
    """
    init_path = find_init_file(jsonl_dir)
    if not init_path:
        return None, None
    n_init = 0
    toks: list[int] = []
    for line in open(init_path):
        rec = json.loads(line)
        n_init = max(n_init, len(rec["all_answers"]))
        for a in rec["all_answers"]:
            # [letter, total_tokens, cot, final]
            if len(a) >= 2 and a[1] is not None:
                toks.append(a[1])
    return (n_init or None), int(round(_mean(toks))) if toks else None


def load_regen_tokens(regen_path: Path) -> int | None:
    """Return mean regen new tokens (`regen_answers[i][r][1]`)."""
    toks: list[int] = []
    for line in open(regen_path):
        rec = json.loads(line)
        for regens in rec.get("regen_answers", []):
            for reg in regens:
                # [letter, new_tokens, ?, final, init_total, kept, cut]
                if len(reg) >= 2 and reg[1] is not None:
                    toks.append(reg[1])
    return int(round(_mean(toks))) if toks else None


def load_verbal_tokens(jsonl_dir: Path) -> tuple[int | None, int | None]:
    """Return (verbal_0_100_mean, binary_query_mean) actual-token counts.

    The binary query is shared by the Verbal binary and P(True) signals.
    None values in the JSONL (rare for ``actual_tokens``) are skipped.
    """
    init_path = find_init_file(jsonl_dir)
    if not init_path:
        return None, None
    v0: list[int] = []
    bq: list[int] = []
    for line in open(init_path):
        rec = json.loads(line)
        for v in rec.get("verbal_0_100_actual_tokens", []):
            if v is not None:
                v0.append(v)
        for v in rec.get("binary_query_actual_tokens", []):
            if v is not None:
                bq.append(v)
    return (
        int(round(_mean(v0))) if v0 else None,
        int(round(_mean(bq))) if bq else None,
    )


def _fmt_int(n: int | None) -> str:
    if n is None:
        return "--"
    s = f"{n:,}"
    return s.replace(",", "{,}")




def collect_rows(dirs: list[Path], K: int) -> list[dict]:
    """Collect one row dict per (model, benchmark)."""
    rows = []
    for d in dirs:
        if not d.exists():
            print(f"  SKIP (missing): {d}")
            continue
        model = infer_model_label(d.name) or d.name
        # Distinguish reasoning-effort variants encoded as dir-name suffix
        # (e.g. "gpt-oss-120b_aime2025_high_jsonl").
        lower = d.name.lower()
        if model == "GPT-OSS-120B" and "_high" in lower:
            model = "GPT-OSS-120B (high)"
        bench = infer_benchmark_label(d.name)
        print(f"Loading {d.name} -> {model} / {bench}")

        n_init, init_mean = load_init_tokens(d)
        if n_init is None and init_mean is None:
            print(f"  SKIP (no init data): {d.name}")
            continue
        regen_by_tau: dict[float, int | None] = {t: None for t in TAU_COLS}
        for tau in TAU_COLS:
            rm_pct = int(round((1 - tau) * 100))
            rpath = find_regen_for_condition(d, f"rm{rm_pct}pct_full_x{K}")
            if rpath is None:
                continue
            regen_by_tau[tau] = load_regen_tokens(rpath)
            print(f"  tau={tau}: {regen_by_tau[tau]}")

        verbal_0_100, binary_query = load_verbal_tokens(d)

        wmv_path = d / "wmv_result.json"
        footnote = None
        if wmv_path.exists():
            try:
                footnote = detect_subset_footnote(wmv_path, d.name)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  WARN: unreadable wmv_result.json ({e})")

        rows.append({
            "model": model,
            "bench": bench,
            "n_init": n_init,
            "init": init_mean,
            "regen": regen_by_tau,
            "verbal_0_100": verbal_0_100,
            "binary_query": binary_query,
            "footnote": footnote,
        })
    return rows


def generate_tex(rows: list[dict], K: int,
                 base_cells: dict | None = None,
                 suppress_subset_footnote: bool = False) -> str:
    # Footnote mapping across all rows
    footnote_map: OrderedDict[str, str] = OrderedDict()
    if not suppress_subset_footnote:
        for r in rows:
            fn = r.get("footnote")
            if fn and fn not in footnote_map:
                footnote_map[fn] = chr(ord("a") + len(footnote_map))

    # Group by model, ordering by canonical MAIN_PAPER_MODELS.
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["model"], []).append(r)
    by_model: "OrderedDict[str, list[dict]]" = OrderedDict(
        sorted(grouped.items(), key=lambda kv: model_sort_key(kv[0]))
    )

    # Sort benchmarks inside each model by canonical BENCHMARK_ORDER
    for m in by_model:
        by_model[m].sort(key=lambda r: bench_sort_key(r["bench"]))

    n_tau = len(TAU_COLS)
    # Column layout: l l | r r | r*ntau | r r
    #                Model Dataset N Generations [Continuations] [Verbal]
    cont_start = 5            # column index of the first Continuations column
    cont_end = 4 + n_tau      # column index of the last Continuations column
    verbal_start = cont_end + 1
    verbal_end = verbal_start + 1
    tabular_spec = "ll r r " + "r" * n_tau + " rr"

    lines = []
    lines.append("% Auto-generated by analysis/paper/table_token_stats.py. Do not edit by hand.")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{\\textbf{Average output tokens per generation.} "
        "Generations is the original full generation. "
        "Continuations are completions from the truncation point onward "
        f"($K{{=}}{K}$). "
        "Verbal queries are secondary completions on each init sample: "
        "0--100 elicits the CISC confidence value, Binary elicits a 0/1 "
        "verdict (the binary completion's logprob is also the P(True) "
        "signal). "
        "All values are means of actual generated tokens over every "
        "(problem, sample) pair. $N$ is the number of answers per problem.}")
    lines.append("\\label{tab:token_stats}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append(f"\\begin{{tabular}}{{{tabular_spec}}}")
    lines.append("\\toprule")
    lines.append(
        "& & & & "
        f"\\multicolumn{{{n_tau}}}{{c}}{{Continuations}} & "
        "\\multicolumn{2}{c}{Verbal queries} \\\\")
    lines.append(
        f"\\cmidrule(lr){{{cont_start}-{cont_end}}} "
        f"\\cmidrule(lr){{{verbal_start}-{verbal_end}}}")
    lines.append(
        "Model & Dataset & $N$ & Generations & "
        "$\\tau{=}0.75$ & $\\tau{=}0.50$ & $\\tau{=}0.25$ & "
        "0--100 & Binary \\\\")
    lines.append("\\midrule")

    model_items = list(by_model.items())
    for mi, (model, entries) in enumerate(model_items):
        if mi > 0:
            lines.append("\\midrule")
        n = len(entries)
        for ei, r in enumerate(entries):
            if ei == 0:
                if n == 1:
                    row_label = model
                else:
                    row_label = f"\\multirow{{{n}}}{{*}}{{{model}}}"
            else:
                row_label = ""

            bench_label = r["bench"]
            fn = r.get("footnote")
            if fn and fn in footnote_map:
                bench_label += f"$^{{{footnote_map[fn]}}}$"

            row_key = (r["model"], r["bench"])
            # cells[0]=row_label, cells[1]=bench_label, data starts at cells[2]
            cells = [row_label, bench_label]
            n_cell = str(r["n_init"]) if r["n_init"] is not None else "--"
            cells.append(fallback_cell(n_cell, (row_key, 2), base_cells))
            cells.append(fallback_cell(_fmt_int(r["init"]),
                                       (row_key, 3), base_cells))
            col_idx = 4
            for tau in TAU_COLS:
                raw = _fmt_int(r["regen"].get(tau))
                cells.append(fallback_cell(raw, (row_key, col_idx), base_cells))
                col_idx += 1
            cells.append(fallback_cell(_fmt_int(r.get("verbal_0_100")),
                                       (row_key, col_idx), base_cells))
            col_idx += 1
            cells.append(fallback_cell(_fmt_int(r.get("binary_query")),
                                       (row_key, col_idx), base_cells))
            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if footnote_map:
        notes = " ".join(
            f"$^{{{v}}}${k}."
            for k, v in footnote_map.items()
        )
        lines.append("\\vspace{2pt}")
        lines.append(f"\\par\\raggedright\\footnotesize {notes}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Generate LaTeX token statistics table")
    ap.add_argument("jsonl_dirs", nargs="+", type=Path,
                    help="Analysis JSONL directories")
    ap.add_argument("-K", type=int, default=1,
                    help="Regen count to report (default: 1)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Write to file instead of stdout")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Only emit rows for these model labels "
                         "(e.g. for the paper version keep the five "
                         "evaluated models only).")
    ap.add_argument("--merge-base", type=Path, default=None,
                    help="Existing output .tex; cells this run cannot "
                         "compute locally fall back to its values.")
    ap.add_argument("--suppress-subset-footnote", action="store_true",
                    help="Omit per-cell ``N=...`` / problem-subset footnotes.")
    args = ap.parse_args()

    rows = collect_rows(args.jsonl_dirs, args.K)
    if args.models:
        allowed = set(args.models)
        rows = [r for r in rows if r["model"] in allowed]
        print(f"Filtered to --models={args.models}: {len(rows)} rows")
    if not rows:
        raise SystemExit("No rows collected.")

    base_cells = None
    if args.merge_base:
        # Bench label + N + Generations + tau cols + 2 verbal cols.
        expected_cols = 1 + 2 + len(TAU_COLS) + 2
        base_cells = load_base_cells_checked(
            args.merge_base,
            _tokenstats_row_key,
            expected_cols=expected_cols,
            label="merge-base",
            group_key_fn=_tokenstats_group_key,
        )

    tex = generate_tex(rows, args.K, base_cells=base_cells,
                       suppress_subset_footnote=args.suppress_subset_footnote)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex + "\n")
        print(f"\nWritten: {args.output}")
    else:
        print()
        print(tex)


if __name__ == "__main__":
    main()
