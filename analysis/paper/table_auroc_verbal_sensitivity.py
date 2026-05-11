#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Sensitivity table: verbal-confidence AUROC under drop vs median-imputed.

Background. The main AUROC table (``table_auroc.tex``) drops samples
whose verbal score failed to parse, while the WMV pipeline imputes the
per-problem median for the same failures so the parsed answer still
casts a neutrally-weighted vote (Appendix~I; Appendix~E
``app:baseline_impl``). The two failure-handling modes can disagree
when the failure rate is high, which is the case for Verbal 0--100 on
GPT-OSS and Nemotron3-30B (19--56 percent).

This script produces a side-by-side comparison so reviewers can see how
much of the Verbal 0--100 / Verbal binary AUROC depends on which mode
is used.

Usage (mirrors ``table_auroc.py`` CLI):
    cd analysis
    uv run python paper/table_auroc_verbal_sensitivity.py \\
        --condition rm25pct_full_x1 \\
        -o tables/table_auroc_verbal_sensitivity.tex \\
        --model "GPT-OSS-120B"   gpt-oss-120b_*_jsonl \\
        --model "GPT-OSS-20B"    gpt-oss-20b_*_jsonl \\
        --model "Nemotron3-30B"  Nemotron-3-Nano-30B-A3B_*_jsonl \\
        --model "Nemotron2-9B"   Nemotron-Nano-9B-v2_*_jsonl \\
        --model "Ministral3-14B" Ministral-3-14B-Reasoning-2512_*_jsonl
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from _utils import (
    infer_benchmark_label, infer_model_label, find_init_file,
    compute_macro_auroc,
    is_usable_answer,
)
from _defs import BENCHMARK_ORDER, BENCHMARK_ABBREV
from table_wmv import parse_model_args


# ── Configuration ──

# (display label, JSONL key) for the two signals with material failure
# rates. P(True) is omitted (~0% missing) since drop and imputed match.
VERBAL_SIGNALS = [
    ("Verbal 0--100", "verbal_0_100"),
    ("Verbal binary", "verbal_binary"),
]

BENCH_ORDER = list(BENCHMARK_ORDER)


# ── Per-(model, benchmark) computation ──

def compute_verbal_aurocs(init_path: Path) -> dict[str, dict[str, float | None]]:
    """For each verbal signal, compute macro AUROC under drop and imputed.

    Returns:
        {
            "verbal_0_100": {"drop": auroc, "imputed": auroc, "fail_rate": rate},
            "verbal_binary": {...},
        }

    The two modes only differ on samples with a None signal value. Drop
    excludes them per-signal; imputed substitutes the per-problem median
    of the non-None values (matching ``get_confs`` in ``analysis/wmv/voters.py``).
    """
    # Per-problem labels and signal arrays. Keep label/score paired by
    # index so drop and imputed see the same problem support.
    y_pp: dict[int, list[int]] = defaultdict(list)
    s_pp: dict[str, dict[int, list[float | None]]] = {
        k: defaultdict(list) for _, k in VERBAL_SIGNALS
    }

    with open(init_path) as f:
        for line in f:
            rec = json.loads(line)
            pnum = rec["problem_num"]
            gold = rec["gold_answer"]
            answers = rec.get("all_answers") or []
            verbal_0_100 = rec.get("verbal_0_100_confidences") or []
            verbal_binary = rec.get("verbal_binary_confidences") or []

            for i, pair in enumerate(answers):
                if not is_usable_answer(pair):
                    continue
                ans = str(pair[0])
                label = 1 if ans == str(gold) else 0
                y_pp[pnum].append(label)
                s_pp["verbal_0_100"][pnum].append(
                    verbal_0_100[i] if i < len(verbal_0_100) else None)
                s_pp["verbal_binary"][pnum].append(
                    verbal_binary[i] if i < len(verbal_binary) else None)

    out: dict[str, dict[str, float | None]] = {}
    for _, key in VERBAL_SIGNALS:
        # Drop: per-problem, drop None entries (and the matching label).
        y_drop: dict[int, list[int]] = {}
        s_drop: dict[int, list[float]] = {}
        # Imputed: per-problem median substituted for None.
        y_imp: dict[int, list[int]] = {}
        s_imp: dict[int, list[float]] = {}

        n_total = 0
        n_failed = 0
        for pnum, ys in y_pp.items():
            ss = s_pp[key][pnum]
            n_total += len(ss)
            n_failed += sum(1 for v in ss if v is None)

            ys_drop, ss_drop = [], []
            for y, s in zip(ys, ss):
                if s is None:
                    continue
                ys_drop.append(y)
                ss_drop.append(s)
            if ys_drop:
                y_drop[pnum] = ys_drop
                s_drop[pnum] = ss_drop

            valid = [v for v in ss if v is not None]
            if valid:
                fill = float(statistics.median(valid))
                ss_imp = [v if v is not None else fill for v in ss]
                y_imp[pnum] = list(ys)
                s_imp[pnum] = ss_imp

        out[key] = {
            "drop": compute_macro_auroc(y_drop, s_drop),
            "imputed": compute_macro_auroc(y_imp, s_imp),
            "fail_rate": (n_failed / n_total) if n_total else None,
        }
    return out


def compute_entry(jsonl_dir: Path) -> dict | None:
    model = infer_model_label(jsonl_dir.name)
    bench = infer_benchmark_label(jsonl_dir.name)
    if not model or not bench:
        print(f"  WARNING: could not infer model/benchmark for {jsonl_dir.name}")
        return None

    init_path = find_init_file(jsonl_dir)
    if init_path is None:
        print(f"  WARNING: no init file in {jsonl_dir.name}")
        return None

    return {
        "model": model,
        "benchmark": bench,
        "verbal": compute_verbal_aurocs(init_path),
    }


# ── Formatting ──

def _fmt_auroc(v: float | None) -> str:
    if v is None:
        return "--"
    return f"{v:.3f}"[1:]  # .XXX (drop leading 0)


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "--"
    return f"{100 * v:.1f}"


# ── Table generator ──

def generate_table(entries: list[dict], models: list[str]) -> str:
    """Build sensitivity table.

    Layout mirrors ``table_auroc.tex``:
      - Outer header: 5 model groups, each spanning 4 benchmark columns.
      - Inner header: benchmark abbreviations (FSci / HMMT / AIME / Brumo).
      - Rows: 4 rows = (Verbal 0--100 drop, imputed) and (Verbal binary
        drop, imputed). Two extra rows show the per-cell failure rate
        for context.
    """
    by_mb = {(e["model"], e["benchmark"]): e for e in entries}
    n_benches = len(BENCH_ORDER)
    col_spec = "l " + " ".join(["c" * n_benches for _ in models])

    lines = []
    lines.append("% Verbal-confidence AUROC sensitivity (drop vs imputed)")
    lines.append("\\begin{table}")
    lines.append("\\centering")
    lines.append(
        "\\caption{\\textbf{Verbal-confidence "
        "$\\overline{\\mathrm{AUROC}}$ sensitivity to failure handling.} "
        "For Verbal 0--100 and Verbal binary, macro-averaged "
        "$\\overline{\\mathrm{AUROC}}$ under "
        "(i)~\\emph{drop}, the convention used by "
        "Table~\\ref{tab:auroc} that excludes samples whose verbal "
        "score failed to parse, and (ii)~\\emph{imputed}, the "
        "convention used by the WMV pipeline that substitutes the "
        "per-problem median of the non-missing scores so the parsed "
        "answer still casts a neutrally-weighted vote "
        "(Appendix~\\ref{app:baseline_impl}). "
        "Per-cell verbal-extraction failure rates are shown for "
        "context. Cells where every sample fails ($\\mathrm{fail}=100\\%$) "
        "leave both modes undefined (``--'').}")
    lines.append("\\label{tab:auroc_verbal_sensitivity}")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{1.5pt}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Outer model header
    lines.append("& " + " & ".join(
        f"\\multicolumn{{{n_benches}}}{{c}}{{{m}}}" for m in models
    ) + " \\\\")
    cmids = []
    col = 2
    for _ in models:
        cmids.append(f"\\cmidrule(lr){{{col}-{col + n_benches - 1}}}")
        col += n_benches
    lines.append(" ".join(cmids))

    # Inner benchmark abbrev header (replicated per model)
    bench_abbrevs = [BENCHMARK_ABBREV.get(b, b) for b in BENCH_ORDER]
    inner = "Signal / mode"
    for _ in models:
        for bl in bench_abbrevs:
            inner += f" & {bl}"
    lines.append(inner + " \\\\")
    lines.append("\\midrule")

    def _row(label: str, getter) -> str:
        row = label
        for m in models:
            for bench in BENCH_ORDER:
                e = by_mb.get((m, bench))
                row += f" & {getter(e)}"
        return row + " \\\\"

    for sig_label, sig_key in VERBAL_SIGNALS:
        lines.append(f"\\multicolumn{{{1 + len(models) * n_benches}}}{{l}}"
                     f"{{\\emph{{{sig_label}}}}} \\\\")
        lines.append(_row(
            "\\quad drop",
            lambda e, k=sig_key: _fmt_auroc(
                e["verbal"][k]["drop"]) if e else "--"))
        lines.append(_row(
            "\\quad imputed",
            lambda e, k=sig_key: _fmt_auroc(
                e["verbal"][k]["imputed"]) if e else "--"))
        lines.append(_row(
            "\\quad fail (\\%)",
            lambda e, k=sig_key: _fmt_pct(
                e["verbal"][k]["fail_rate"]) if e else "--"))
        lines.append("\\midrule")

    # Replace the trailing \midrule with \bottomrule
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")

    lines.append("\\end{tabular}%")
    lines.append("}% end resizebox")
    lines.append("\\vspace{2pt}")

    glossary_parts = []
    for b in BENCH_ORDER:
        abbr = BENCHMARK_ABBREV.get(b)
        if abbr and abbr != b:
            glossary_parts.append(f"{abbr} = {b}")
    footer = "Abbreviations: " + ", ".join(glossary_parts) + "."
    lines.append(f"\\par\\raggedright\\footnotesize {footer}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Verbal-confidence AUROC sensitivity table.")
    parser.add_argument(
        "--condition", default="rm25pct_full_x1",
        help="Unused (kept for CLI parity with table_auroc.py).")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path for the LaTeX table (default: stdout).")
    args, remaining = parser.parse_known_args()

    if "--model" not in remaining:
        parser.error("Provide --model NAME dir [dir ...] groups.")
    model_groups = parse_model_args(remaining)
    if not model_groups:
        parser.error("--model requires NAME followed by one or more dirs")

    entries = []
    models = []
    for name, dirs in model_groups:
        models.append(name)
        for d in dirs:
            print(f"Processing {d.name} (as {name})")
            e = compute_entry(d)
            if e is None:
                continue
            e["model"] = name
            entries.append(e)
            v = e["verbal"]
            print(f"  -> {name} / {e['benchmark']}: "
                  f"V0-100 drop={v['verbal_0_100']['drop']} "
                  f"imp={v['verbal_0_100']['imputed']} "
                  f"fail={v['verbal_0_100']['fail_rate']:.1%} | "
                  f"Vbin drop={v['verbal_binary']['drop']} "
                  f"imp={v['verbal_binary']['imputed']} "
                  f"fail={v['verbal_binary']['fail_rate']:.1%}")

    print(f"\nModel columns: {models}")

    tex = generate_table(entries, models)
    if args.output:
        args.output.write_text(tex + "\n")
        print(f"Wrote: {args.output}")
    else:
        print("\n== table_auroc_verbal_sensitivity.tex ==")
        print(tex)


if __name__ == "__main__":
    main()
