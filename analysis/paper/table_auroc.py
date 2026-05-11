#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate table_auroc.tex and table_signal.tex from jsonl dirs.

Both tables use macro-averaged per-problem values across problems with
both correct and incorrect init answers (consistent with table_transition.py).

- table_auroc.tex: AUROC per (signal, benchmark, model). Signals:
  Prefix consistency, Self-certainty, DeepConf (tail), DeepConf (bot-10%),
  P(True), Verbal 0-100.
- table_signal.tex: r_C, r_W, D = r_C - r_W for prefix consistency, per
  (benchmark, model). AUROC = (1 + D) / 2 holds by construction for PC.

Usage (explicit --model groups, preferred; mirrors table_wmv.py CLI):
    cd analysis
    uv run python paper/table_auroc.py \\
        --condition rm25pct_full_x1 \\
        --auroc-out tables/table_auroc.tex \\
        --signal-out tables/table_signal.tex \\
        --model "GPT-OSS-120B"              gpt-oss-120b_*_jsonl \\
        --model "GPT-OSS-20B"               gpt-oss-20b_*_jsonl \\
        --model "Nemotron-3-Nano-30B-A3B"   Nemotron-3-Nano-30B-A3B_*_jsonl \\
        --model "Ministral-3-14B-Reasoning" Ministral-3-14B-Reasoning-2512_*_jsonl

Alternative (positional dirs, model inferred from dir name via _defs.MODEL_LABELS):
    uv run python paper/table_auroc.py \\
        gpt-oss-120b_*_jsonl gpt-oss-20b_*_jsonl ... \\
        --condition rm25pct_full_x1 --auroc-out ... --signal-out ...
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _utils import (
    infer_benchmark_label, infer_model_label, find_init_file,
    find_regen_for_condition,
    compute_macro_auroc, model_sort_key,
    load_base_cells_checked, fallback_cell,
    METHOD_DISPLAY,
    signal_keys_in_order,
    is_usable_answer,
)
from _defs import (
    BENCHMARK_ORDER, BENCHMARK_ABBREV,
    SIGNAL_KEY_TO_METHOD,
)
from table_transition import compute_transition_rates
from table_wmv import parse_model_args


# ── Display configuration ──

# Benchmarks shown in the auroc table.
# BENCH_ORDER is the canonical column order; centralized in _defs.py so
# this table stays aligned with table_wmv, table_token_savings, etc.
BENCH_ORDER = list(BENCHMARK_ORDER)

# Signal rows for table_auroc.tex (display_label, jsonl_key).
# Baseline subset: the per-sample signals the JSONL records; ordered
# automatically by ALL_BASELINE_GROUPS via signal_keys_in_order so the
# row order tracks the rest of the paper. "prefix" is special (computed
# from regen rates, not from init JSONL) and is prepended.
_AUROC_BASELINE_SUBSET = (
    "mean_conf", "tail_conf", "bottom10_conf", "p_true", "verbal_0_100",
)
AUROC_SIGNALS = (
    [("Prefix consistency", "prefix")]
    + [(METHOD_DISPLAY[SIGNAL_KEY_TO_METHOD[k]], k)
       for k in signal_keys_in_order(_AUROC_BASELINE_SUBSET)]
)


# ── Baseline-signal AUROC (from init JSONL) ──

def compute_baseline_aurocs(init_path: Path) -> dict[str, float | None]:
    """Compute macro AUROC for each baseline signal from init JSONL.

    Returns dict keyed by the same signal keys used in AUROC_SIGNALS
    (mean_conf, tail_conf, bottom10_conf, p_true, verbal_0_100).
    Missing/None values are skipped per-sample.
    """
    signal_keys = signal_keys_in_order(_AUROC_BASELINE_SUBSET)
    y_pp: dict[int, list] = defaultdict(list)
    s_pp: dict[str, dict[int, list]] = {k: defaultdict(list) for k in signal_keys}

    with open(init_path) as f:
        for line in f:
            rec = json.loads(line)
            pnum = rec["problem_num"]
            gold = rec["gold_answer"]
            answers = rec.get("all_answers") or []
            confs = rec.get("confidences") or []
            verbal = rec.get("verbal_0_100_confidences") or []
            p_true = rec.get("p_true_confidences") or []

            for i, pair in enumerate(answers):
                if not is_usable_answer(pair):
                    continue
                ans = str(pair[0])
                label = 1 if ans == str(gold) else 0

                # DeepConf variants from confidences dict
                c = confs[i] if i < len(confs) and confs[i] else {}
                vals = {
                    "mean_conf":     c.get("mean_conf"),
                    "tail_conf":     c.get("tail_conf"),
                    "bottom10_conf": c.get("bottom10_conf"),
                    "verbal_0_100":  verbal[i] if i < len(verbal) else None,
                    "p_true":        p_true[i] if i < len(p_true) else None,
                }
                # Only include the sample if at least one signal is available;
                # NaN signals will drop the sample for that signal only.
                if any(v is not None for v in vals.values()):
                    y_pp[pnum].append(label)
                    for k in signal_keys:
                        s_pp[k][pnum].append(vals[k])

    # Per-signal: drop samples where that signal is None, then macro AUROC.
    result: dict[str, float | None] = {}
    for k in signal_keys:
        y_filtered: dict[int, list] = {}
        s_filtered: dict[int, list] = {}
        for pnum in y_pp:
            ys, ss = [], []
            for y, s in zip(y_pp[pnum], s_pp[k][pnum]):
                if s is None:
                    continue
                ys.append(y)
                ss.append(s)
            if ys:
                y_filtered[pnum] = ys
                s_filtered[pnum] = ss
        result[k] = compute_macro_auroc(y_filtered, s_filtered)
    return result


# ── Per-(model, benchmark) computation ──

def compute_entry(jsonl_dir: Path, condition: str) -> dict | None:
    """Compute all values needed for both tables for one (model, benchmark).

    Returns dict with keys:
      model, benchmark, prefix_auroc, baseline_aurocs, r_C, r_W, D
    (values may be None if unavailable).
    """
    model = infer_model_label(jsonl_dir.name)
    bench = infer_benchmark_label(jsonl_dir.name)
    if not model or not bench:
        print(f"  WARNING: could not infer model/benchmark for {jsonl_dir.name}")
        return None

    regen_path = find_regen_for_condition(jsonl_dir, condition)
    init_path = find_init_file(jsonl_dir)

    if regen_path is None:
        print(f"  WARNING: no regen file for {condition} in {jsonl_dir.name}")
        rates = None
    else:
        rates = compute_transition_rates(regen_path, K=1)["rates"]

    if init_path is None:
        print(f"  WARNING: no init file in {jsonl_dir.name}")
        baseline_aurocs = {}
    else:
        baseline_aurocs = compute_baseline_aurocs(init_path)

    r_C = rates["c_to_c"] if rates else None
    r_W = rates["w_to_same_w"] if rates else None
    D = (r_C - r_W) if (r_C is not None and r_W is not None) else None
    # For K=1 binary PC scores, macro AUROC = (1 + macro D) / 2 exactly.
    # Derive from D to keep the two tables algebraically consistent.
    prefix_auroc = (1 + D) / 2 if D is not None else None

    return {
        "model": model,
        "benchmark": bench,
        "prefix_auroc": prefix_auroc,
        "baseline_aurocs": baseline_aurocs,
        "r_C": r_C,
        "r_W": r_W,
        "D": D,
    }


# ── Formatting ──

def _fmt_auroc(v: float | None, bold: bool = False) -> str:
    if v is None:
        return "--"
    s = f"{v:.3f}"[1:]  # .XXX (drop leading 0)
    return f"\\textbf{{{s}}}" if bold else s


def _fmt_pct(v: float | None, bold: bool = False) -> str:
    if v is None:
        return "--"
    s = f"{100 * v:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def _get_auroc(entry: dict, signal_key: str) -> float | None:
    if entry is None:
        return None
    if signal_key == "prefix":
        return entry["prefix_auroc"]
    return entry["baseline_aurocs"].get(signal_key)


# ── Table generators ──

def generate_auroc_table(
    entries: list[dict],
    models: list[str],
    base_cells: dict | None = None,
    suppress_condition: bool = False,
) -> str:
    """Build table_auroc.tex (model-outer × benchmark-inner layout).

    If ``suppress_condition`` is True, the caption omits the
    ``$\\tau{=}0.75$, $K{=}1$`` phrase and the related editorial gloss,
    matching the camera-ready convention from ``regen_paper.sh`` (those
    facts live in Appendix~\\ref{appendix:reprod}).
    """
    by_mb = {(e["model"], e["benchmark"]): e for e in entries}

    # Best signal per (model, benchmark) column for bolding.
    best = {}
    for m in models:
        for bench in BENCH_ORDER:
            vals = []
            for _, sig_key in AUROC_SIGNALS:
                v = _get_auroc(by_mb.get((m, bench)), sig_key)
                if v is not None:
                    vals.append(v)
            best[(m, bench)] = max(vals) if vals else None

    n_benches = len(BENCH_ORDER)
    col_spec = "l " + " ".join(["c" * n_benches for _ in models])

    cond_clause = ("" if suppress_condition
                   else " at $\\tau{=}0.75$, $K{=}1$ "
                        "(Section~\\ref{sec:signal})")
    editorial = ("" if suppress_condition
                 else " Prefix consistency is the strongest correctness "
                      "signal on most cells, and dominates the logit-based "
                      "and introspective baselines on the harder benchmarks.")

    lines = []
    lines.append("% AUROC table (macro-averaged per problem, model-outer layout)")
    lines.append("\\begin{table}")
    lines.append("\\centering")
    lines.append("\\caption{\\textbf{$\\overline{\\mathrm{AUROC}}$ for "
                 "correctness discrimination (higher is better).} "
                 "Macro-averaged $\\overline{\\mathrm{AUROC}}$ per "
                 f"(model, benchmark){cond_clause}, with the best per column "
                 f"in bold.{editorial}}}")
    lines.append("\\label{tab:auroc}")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{1.5pt}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Model group header (outer)
    lines.append("& " + " & ".join(
        f"\\multicolumn{{{n_benches}}}{{c}}{{{m}}}" for m in models
    ) + " \\\\")
    # cmidrule per model block
    cmids = []
    col = 2
    for _ in models:
        cmids.append(f"\\cmidrule(lr){{{col}-{col + n_benches - 1}}}")
        col += n_benches
    lines.append(" ".join(cmids))

    # Inner benchmark abbrev row (replicated per model)
    bench_abbrevs = [BENCHMARK_ABBREV.get(b, b) for b in BENCH_ORDER]
    inner = "Signal"
    for _ in models:
        for bl in bench_abbrevs:
            inner += f" & {bl}"
    lines.append(inner + " \\\\")
    lines.append("\\midrule")

    # Signal rows
    for sig_label, sig_key in AUROC_SIGNALS:
        row = sig_label
        col_idx = 1
        for m in models:
            for bench in BENCH_ORDER:
                v = _get_auroc(by_mb.get((m, bench)), sig_key)
                b = best[(m, bench)]
                is_best = (v is not None and b is not None
                           and abs(v - b) < 1e-9)
                cell = _fmt_auroc(v, bold=is_best)
                cell = fallback_cell(cell, (sig_label, col_idx), base_cells)
                row += f" & {cell}"
                col_idx += 1
        lines.append(row + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}% end resizebox")
    lines.append("\\vspace{2pt}")

    # Footnote: dataset abbreviation glossary
    glossary_parts = []
    for b in BENCH_ORDER:
        abbr = BENCHMARK_ABBREV.get(b)
        if abbr and abbr != b:
            glossary_parts.append(f"{abbr} = {b}")
    footer = "Abbreviations: " + ", ".join(glossary_parts) + "."
    lines.append(f"\\par\\raggedright\\footnotesize {footer}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_signal_table(
    entries: list[dict],
    models: list[str],
    base_cells: dict | None = None,
    suppress_condition: bool = False,
) -> str:
    """Build table_signal.tex content.

    If ``suppress_condition`` is True, the caption omits the
    ``$\\tau{=}0.75$, $K{=}1$`` phrase and the trailing
    ``Section~\\ref{sec:token_efficiency}`` cross-reference, matching
    the camera-ready convention from ``regen_paper.sh`` (those facts
    live in Appendix~\\ref{appendix:reprod}).
    """
    by_mb = {(e["model"], e["benchmark"]): e for e in entries}
    n_models = len(models)
    col_spec = "l " + " ".join(["ccc" for _ in models])

    cond_clause = "" if suppress_condition else " at $\\tau{=}0.75$, $K{=}1$"
    section_ref = ("" if suppress_condition
                   else ", Section~\\ref{sec:token_efficiency}")

    lines = []
    lines.append("% Prefix consistency signal table (macro-averaged, "
                 "problems with both correct and incorrect only)")
    lines.append("\\begin{table}")
    lines.append("\\centering")
    # Canonical caption is kept commented for history; the active caption
    # below is the paper-body form (bold lead + narrative).
    lines.append(
        "% \\caption{Class-conditional reproduction rates $r_C$, $r_W$ and "
        "discrimination gap $D = r_C - r_W$ (Eq.~\\eqref{eq:asymmetry}) for "
        "prefix consistency at $\\tau{=}0.75$, $K{=}1$. Values are "
        "macro-averaged across problems that have both correct and incorrect "
        "init answers, so that $\\overline{\\mathrm{AUROC}} = (1+D)/2$ holds "
        "(Table~\\ref{tab:auroc}).}")
    lines.append("% \\label{tab:signal}")
    lines.append(
        "\\caption{\\textbf{Reproduction rates $r_C$, $r_W$ and discrimination "
        "gap $D = r_C - r_W$ for prefix consistency "
        "(larger $D$ is better).} "
        "Macro-averaged over problems with at least one correct and one wrong "
        f"initial sample{cond_clause}. "
        "$r_C \\geq r_W$ holds on every (model, benchmark) cell, and a larger "
        "$D$ predicts a larger PC-WMV advantage over Standard MV "
        f"(Theorem~\\ref{{thm:improvement}}{section_ref}).}}")
    lines.append("\\label{tab:signal}")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Multicolumn header
    lines.append("& " + " & ".join(
        f"\\multicolumn{{3}}{{c}}{{{m}}}" for m in models) + " \\\\")
    cmids = []
    col = 2
    for _ in models:
        cmids.append(f"\\cmidrule(lr){{{col}-{col + 2}}}")
        col += 3
    lines.append(" ".join(cmids))
    header = "Benchmark"
    for _ in models:
        header += " & $r_C$ & $r_W$ & $D$"
    lines.append(header + " \\\\")
    lines.append("\\midrule")

    for bench in BENCH_ORDER:
        row = bench
        col_idx = 1
        for m in models:
            e = by_mb.get((m, bench))
            if e is None:
                triple = ("--", "--", "--")
            else:
                triple = (
                    _fmt_pct(e["r_C"]),
                    _fmt_pct(e["r_W"]),
                    _fmt_pct(e["D"]),
                )
            for cell in triple:
                cell = fallback_cell(cell, (bench, col_idx), base_cells)
                row += f" & {cell}"
                col_idx += 1
        lines.append(row + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{2pt}")
    lines.append("\\par\\raggedright\\footnotesize")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Generate table_auroc.tex and table_signal.tex")
    parser.add_argument("--condition", required=True,
                        help="Condition label (e.g. rm25pct_full_x1)")
    parser.add_argument("--auroc-out", type=Path, default=None,
                        help="Output path for table_auroc.tex (default: stdout)")
    parser.add_argument("--signal-out", type=Path, default=None,
                        help="Output path for table_signal.tex (default: stdout)")
    parser.add_argument("--auroc-merge-base", type=Path, default=None,
                        help="Existing table_auroc.tex whose non-empty cells "
                             "fill in wherever this run lacks data.")
    parser.add_argument("--signal-merge-base", type=Path, default=None,
                        help="Existing table_signal.tex whose non-empty cells "
                             "fill in wherever this run lacks data.")
    parser.add_argument("--suppress-condition", action="store_true",
                        help="Drop ``$\\tau{=}0.75$, $K{=}1$`` and related "
                             "cross-references from the captions; the camera-"
                             "ready convention (those facts live in "
                             "Appendix~\\ref{appendix:reprod}).")
    args, remaining = parser.parse_known_args()

    # Two input modes:
    #   (a) --model NAME dir [dir ...] (explicit grouping, same CLI shape
    #       as table_wmv.py / table_token_savings.py)
    #   (b) positional dirs, model inferred via _defs.MODEL_LABELS (legacy)
    if "--model" in remaining:
        model_groups = parse_model_args(remaining)
        if not model_groups:
            parser.error("--model requires NAME followed by one or more dirs")
        entries = []
        models = []
        for name, dirs in model_groups:
            models.append(name)
            for d in dirs:
                print(f"Processing {d.name} (as {name})")
                e = compute_entry(d, args.condition)
                if e is None:
                    continue
                e["model"] = name
                entries.append(e)
                print(f"  -> {name} / {e['benchmark']}: "
                      f"PC AUROC={e['prefix_auroc']}, D={e['D']}")
    else:
        jsonl_dirs = [Path(a) for a in remaining if not a.startswith("-")]
        if not jsonl_dirs:
            parser.error(
                "Provide either --model NAME dir [dir ...] groups or "
                "positional directories.")
        entries = []
        for d in jsonl_dirs:
            print(f"Processing {d.name}")
            e = compute_entry(d, args.condition)
            if e is None:
                continue
            entries.append(e)
            print(f"  -> {e['model']} / {e['benchmark']}: "
                  f"PC AUROC={e['prefix_auroc']}, D={e['D']}")
        seen = []
        for e in entries:
            if e["model"] not in seen:
                seen.append(e["model"])
        seen.sort(key=model_sort_key)
        models = seen

    print(f"\nModel columns: {models}")

    # Merge-base cells (cell-level fallback for data we don't have locally).
    # Positional merge: we refuse to merge when the base's column count
    # differs from this run's, otherwise values would bind to the wrong
    # columns (e.g. a new --models list that drops a model).
    auroc_base = None
    if args.auroc_merge_base:
        auroc_base = load_base_cells_checked(
            args.auroc_merge_base,
            [sig for sig, _ in AUROC_SIGNALS],
            expected_cols=len(BENCH_ORDER) * len(models),
            label="auroc-merge-base",
        )
    signal_base = None
    if args.signal_merge_base:
        signal_base = load_base_cells_checked(
            args.signal_merge_base,
            BENCH_ORDER,
            expected_cols=3 * len(models),
            label="signal-merge-base",
        )

    # Generate both tables
    auroc_tex = generate_auroc_table(
        entries, models, base_cells=auroc_base,
        suppress_condition=args.suppress_condition)
    signal_tex = generate_signal_table(
        entries, models, base_cells=signal_base,
        suppress_condition=args.suppress_condition)

    if args.auroc_out:
        args.auroc_out.write_text(auroc_tex + "\n")
        print(f"Wrote: {args.auroc_out}")
    else:
        print("\n== table_auroc.tex ==")
        print(auroc_tex)

    if args.signal_out:
        args.signal_out.write_text(signal_tex + "\n")
        print(f"Wrote: {args.signal_out}")
    else:
        print("\n== table_signal.tex ==")
        print(signal_tex)


if __name__ == "__main__":
    main()
