#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate LaTeX table of answer-extraction failure rates.

Two failure modes are reported per (model, benchmark) cell:

1. Boxed extraction (Generations / Continuations): fraction of generated
   answers where the parser found neither a ``\\boxed{...}`` expression
   nor a usable numeric fallback (``PARSE_FAILED`` / ``None`` in
   ``all_answers[i][0]`` and ``regen_answers[i][r][0]``). Column names
   mirror those of ``table_token_stats.py``.

2. Verbal queries (0--100 / Binary / P(True)): fraction of secondary
   completions whose confidence value could not be extracted (``None``
   in ``verbal_0_100_confidences``, ``verbal_binary_confidences``, and
   ``p_true_confidences`` respectively). The Binary completion's logprob
   is also the P(True) signal, so the two columns share the same
   secondary call.

Row axis:    Model x Benchmark.
Column axis: $N$, Generations %, Continuations %, Verbal 0--100 %,
             Verbal binary %, P(True) %.

Generations counts pool over (problem, init sample). Continuations
counts pool over (problem, init sample, regen 1..K) using only the regen
condition with $K{=}1$ at $\\tau{=}0.75$ (the main-paper setting; failure
rate is essentially constant across $\\tau$). Verbal columns pool over
(problem, init sample); they are properties of the init pool, not the
regenerations.

Usage:
    cd analysis && uv run python paper/table_format_failure.py \\
        gpt-oss-120b_aime2025_jsonl \\
        ... -o /tmp/table_format_failure.tex
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from _utils import (
    bench_sort_key, model_sort_key,
    detect_subset_footnote,
    find_init_file,
    find_regen_for_condition,
    infer_benchmark_label,
    infer_model_label,
    signal_keys_in_order,
)


def init_failure_counts(jsonl_dir: Path) -> tuple[int, int, int | None]:
    """Return (n_fail, n_total, n_init_per_problem) for the init JSONL.

    ``n_init_per_problem`` is the maximum ``len(all_answers)`` across problems,
    which represents the intended sample budget (a few problems may lose
    samples to upstream failures).
    """
    p = find_init_file(jsonl_dir)
    if p is None:
        return 0, 0, None
    fail = total = 0
    n_init = 0
    for line in open(p):
        rec = json.loads(line)
        n_init = max(n_init, len(rec["all_answers"]))
        for a in rec["all_answers"]:
            total += 1
            if a[0] == "PARSE_FAILED" or a[0] is None:
                fail += 1
    return fail, total, (n_init or None)


def regen_failure_counts(regen_path: Path, K: int) -> tuple[int, int]:
    """Return (n_fail, n_total) pooled over (problem, init, regen 1..K)."""
    fail = total = 0
    for line in open(regen_path):
        rec = json.loads(line)
        for regens in rec.get("regen_answers", []):
            for r in regens[:K]:
                total += 1
                if r[0] == "PARSE_FAILED" or r[0] is None:
                    fail += 1
    return fail, total


# Confidence fields whose ``None`` entries indicate a failed extraction
# of the secondary verbal/P(True) call. Ordered by ALL_BASELINE_GROUPS
# (paper-natural) via signal_keys_in_order so the column order tracks
# the rest of the paper. Response probability is not included since it
# never has a "secondary call extraction failure" mode.
_VERBAL_CONF_SUBSET = ("verbal_binary", "verbal_0_100", "p_true")
VERBAL_CONF_FIELDS = tuple(
    (k, f"{k}_confidences")
    for k in signal_keys_in_order(_VERBAL_CONF_SUBSET)
)


def verbal_failure_counts(jsonl_dir: Path) -> dict[str, tuple[int, int]]:
    """Return ``{key: (n_fail, n_total)}`` for each verbal-confidence field."""
    init_path = find_init_file(jsonl_dir)
    out = {key: (0, 0) for key, _ in VERBAL_CONF_FIELDS}
    if init_path is None:
        return out
    for line in open(init_path):
        rec = json.loads(line)
        for key, field in VERBAL_CONF_FIELDS:
            confs = rec.get(field, [])
            f, t = out[key]
            for v in confs:
                t += 1
                if v is None:
                    f += 1
            out[key] = (f, t)
    return out


def _fmt_pct(fail: int, total: int) -> str:
    if total == 0:
        return "--"
    pct = 100.0 * fail / total
    if pct == 0.0:
        return "0.00"
    return f"{pct:.2f}"


def collect_rows(dirs: list[Path], K: int, tau: float) -> list[dict]:
    rows = []
    for d in dirs:
        if not d.exists():
            print(f"  SKIP (missing): {d}")
            continue
        model = infer_model_label(d.name) or d.name
        bench = infer_benchmark_label(d.name)
        print(f"Loading {d.name} -> {model} / {bench}")

        ifail, itot, n_init = init_failure_counts(d)
        if itot == 0:
            print(f"  SKIP (no init data): {d.name}")
            continue

        rm_pct = int(round((1 - tau) * 100))
        rpath = find_regen_for_condition(d, f"rm{rm_pct}pct_full_x{K}")
        rfail = rtot = 0
        if rpath is not None:
            rfail, rtot = regen_failure_counts(rpath, K=K)
        else:
            print(f"  WARN: no rm{rm_pct}pct_full_x{K} in {d.name}")

        wmv_path = d / "wmv_result.json"
        footnote = None
        if wmv_path.exists():
            try:
                footnote = detect_subset_footnote(wmv_path, d.name)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  WARN: unreadable wmv_result.json ({e})")

        verbal = verbal_failure_counts(d)

        rows.append({
            "model": model,
            "bench": bench,
            "n_init": n_init,
            "init_fail": ifail,
            "init_total": itot,
            "regen_fail": rfail,
            "regen_total": rtot,
            "verbal": verbal,
            "footnote": footnote,
        })
    return rows


def generate_tex(rows: list[dict], K: int, tau: float,
                 suppress_subset_footnote: bool = False) -> str:
    footnote_map: OrderedDict[str, str] = OrderedDict()
    if not suppress_subset_footnote:
        for r in rows:
            fn = r.get("footnote")
            if fn and fn not in footnote_map:
                footnote_map[fn] = chr(ord("a") + len(footnote_map))

    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["model"], []).append(r)
    by_model: "OrderedDict[str, list[dict]]" = OrderedDict(
        sorted(grouped.items(), key=lambda kv: model_sort_key(kv[0]))
    )
    for m in by_model:
        by_model[m].sort(key=lambda r: bench_sort_key(r["bench"]))

    total_ifail = sum(r["init_fail"] for r in rows)
    total_itot = sum(r["init_total"] for r in rows)
    total_rfail = sum(r["regen_fail"] for r in rows)
    total_rtot = sum(r["regen_total"] for r in rows)
    verbal_totals: dict[str, tuple[int, int]] = {
        key: (sum(r["verbal"][key][0] for r in rows),
              sum(r["verbal"][key][1] for r in rows))
        for key, _ in VERBAL_CONF_FIELDS
    }

    # Column layout: l l | r | r r | r r r
    #                Model Dataset N [Boxed: Generations Continuations]
    #                                [Verbal: 0-100 Bin P(True)]
    boxed_start = 4
    boxed_end = 5
    verbal_start = 6
    verbal_end = 8

    lines = []
    lines.append("% Auto-generated by analysis/paper/table_format_failure.py. "
                 "Do not edit by hand.")
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{\\textbf{Extraction failure rate.} "
        "Boxed columns: percentage of generated answers for which the parser "
        "found neither a \\texttt{\\textbackslash boxed\\{...\\}} expression "
        "nor a usable numeric fallback. "
        "Generations pools over (problem, sample) on the original full "
        f"generation; Continuations pools over (problem, sample, regen) on "
        f"$K{{=}}{K}$ completions from the truncation point onward "
        f"($\\tau{{=}}{tau:g}$). "
        "Verbal columns: percentage of secondary completions whose "
        "confidence value could not be extracted (0--100 = CISC value, "
        "Binary = ``Is your answer correct? 0/1'' verdict, "
        "P(True) = logprob of the binary call's ``1'' token; the latter "
        "two share the same secondary call). "
        "Failure handling in downstream aggregators is described in "
        "Appendix~\\ref{app:answer_extraction}. "
        "Pooled totals across all 20 cells: "
        f"{100*total_ifail/total_itot:.2f}\\% (Generations), "
        f"{100*total_rfail/total_rtot:.2f}\\% (Continuations), "
        f"{100*verbal_totals['verbal_binary'][0]/verbal_totals['verbal_binary'][1]:.2f}\\% (Binary), "
        f"{100*verbal_totals['verbal_0_100'][0]/verbal_totals['verbal_0_100'][1]:.2f}\\% (0--100), "
        f"{100*verbal_totals['p_true'][0]/verbal_totals['p_true'][1]:.2f}\\% (P(True)). "
        "Boxed-extraction rates are flat across "
        "$\\tau \\in \\{0.25, 0.50, 0.75\\}$ (within $0.05$ pp) on the cells "
        "where multiple $\\tau$ are available.}")
    lines.append("\\label{tab:format_failure}")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{ll r rr rrr}")
    lines.append("\\toprule")
    lines.append(
        "& & & \\multicolumn{2}{c}{Boxed extraction (\\%)} & "
        "\\multicolumn{3}{c}{Verbal queries (\\%)} \\\\")
    lines.append(
        f"\\cmidrule(lr){{{boxed_start}-{boxed_end}}} "
        f"\\cmidrule(lr){{{verbal_start}-{verbal_end}}}")
    lines.append(
        "Model & Dataset & $N$ & "
        "Generations & Continuations & "
        "0--100 & Binary & P(True) \\\\")
    lines.append("\\midrule")

    model_items = list(by_model.items())
    for mi, (model, entries) in enumerate(model_items):
        if mi > 0:
            lines.append("\\midrule")
        n = len(entries)
        for ei, r in enumerate(entries):
            if ei == 0:
                row_label = (model if n == 1
                             else f"\\multirow{{{n}}}{{*}}{{{model}}}")
            else:
                row_label = ""

            bench_label = r["bench"]
            fn = r.get("footnote")
            if fn and fn in footnote_map:
                bench_label += f"$^{{{footnote_map[fn]}}}$"

            n_cell = (str(r["n_init"]) if r["n_init"] is not None else "--")
            init_cell = _fmt_pct(r["init_fail"], r["init_total"])
            regen_cell = _fmt_pct(r["regen_fail"], r["regen_total"])
            v0_cell = _fmt_pct(*r["verbal"]["verbal_0_100"])
            vb_cell = _fmt_pct(*r["verbal"]["verbal_binary"])
            pt_cell = _fmt_pct(*r["verbal"]["p_true"])
            lines.append(
                f"{row_label} & {bench_label} & {n_cell} & "
                f"{init_cell} & {regen_cell} & "
                f"{v0_cell} & {vb_cell} & {pt_cell} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if footnote_map:
        notes = " ".join(f"$^{{{v}}}${k}." for k, v in footnote_map.items())
        lines.append("\\vspace{2pt}")
        lines.append(f"\\par\\raggedright\\footnotesize {notes}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Generate LaTeX answer-extraction failure rate table")
    ap.add_argument("jsonl_dirs", nargs="+", type=Path,
                    help="Analysis JSONL directories")
    ap.add_argument("-K", type=int, default=1,
                    help="Regen count to pool over (default: 1)")
    ap.add_argument("--tau", type=float, default=0.75,
                    help="Kept-fraction tau for regen column (default: 0.75)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Write to file instead of stdout")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Only emit rows for these model labels")
    ap.add_argument("--suppress-subset-footnote", action="store_true",
                    help="Omit per-cell ``N=...`` / problem-subset footnotes.")
    args = ap.parse_args()

    rows = collect_rows(args.jsonl_dirs, args.K, args.tau)
    if args.models:
        allowed = set(args.models)
        rows = [r for r in rows if r["model"] in allowed]
        print(f"Filtered to --models={args.models}: {len(rows)} rows")
    if not rows:
        raise SystemExit("No rows collected.")

    tex = generate_tex(rows, args.K, args.tau,
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
