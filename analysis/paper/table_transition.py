#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate LaTeX answer-transition-rate tables from regen JSONL files.

Computes per-generation transition rates (W->C, W->sameW, W->otherW,
C->C, C->W) and outputs publication-ready LaTeX tables.

Three output modes:

  rates  Single condition, all model x dataset rows grouped by dataset.
         Produces table_transition_rates.tex (Appendix).

  tau    Multiple truncation fractions for selected model x dataset pairs.
         Produces table_transition_tau.tex (Appendix).

  size   Model-family comparison (larger vs smaller within same family).
         Produces table_analysis_flip_vs_size.tex (main paper).

Usage:
    cd analysis

    # Appendix: transition rates at tau=0.75
    uv run python paper/table_transition.py \
        gpt-oss-20b_hmmt_jsonl \
        gpt-oss-120b_hmmt_jsonl \
        Nemotron-3-Nano-30B-A3B_hmmt_jsonl \
        --condition rm25pct_full_x1 \
        --mode rates \
        -o tables/table_transition_rates.tex

    # Appendix: transition rates vs truncation fraction
    uv run python paper/table_transition.py \
        gpt-oss-20b_aime2025_jsonl \
        gpt-oss-20b_hmmt_jsonl \
        --conditions rm25pct_full_x1 rm50pct_full_x1 rm75pct_full_x1 \
        --mode tau \
        -o tables/table_transition_tau.tex

    # Model size comparison
    uv run python paper/table_transition.py \
        gpt-oss-120b_aime2025_jsonl \
        gpt-oss-20b_aime2025_jsonl \
        Nemotron-3-Nano-30B-A3B_aime2025_jsonl \
        Nemotron-Nano-9B-v2_aime2025_jsonl \
        --condition rm25pct_full_x1 \
        --mode size \
        --families "GPT-OSS:GPT-OSS-120B,GPT-OSS-20B" \
                   "Nemotron:Nemotron3-30B,Nemotron2-9B" \
        -o tables/table_analysis_flip_vs_size.tex
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _defs import BENCHMARK_LABELS, DATASET_TOTAL_PROBLEMS
from _utils import (
    find_init_file, parse_condition,
    find_regen_for_condition,
    infer_benchmark_label, infer_model_label,
    compute_macro_auroc, bench_sort_key, model_sort_key,
    load_base_cells_checked, fallback_cell,
    is_usable_answer,
)


# ── Transition rate columns ──

RATE_COLS = [
    ("c_to_c",       r"\textbf{$r_C$}"),
    ("w_to_same_w",  r"\textbf{$r_W$}"),
    ("w_to_c",       r"$\phi_{WC}$"),
]

# Extra columns appended after RATE_COLS.
#
# The default is D + one AUROC column (computed from the init-answer
# PC score c_i(a_i), which is the standard $(1+D)/2$ identity for PC at
# K=1, so we label it simply ``AUROC``). ``auroc_dist`` is an alternative
# measure over distinct in-group answers and is opt-in via --include-auroc-dist.
EXTRA_COLS_BASE = [
    ("D", r"\textbf{$D$}"),
    ("auroc_init", r"$\overline{\mathrm{AUROC}}$"),
]
EXTRA_COLS_DIST = [
    ("auroc_dist", r"$\overline{\mathrm{AUROC}}_\text{dist}$"),
]


def _extra_cols(include_dist: bool) -> list[tuple[str, str]]:
    return list(EXTRA_COLS_BASE) + (list(EXTRA_COLS_DIST) if include_dist else [])

# ── AUROC loading ──


def compute_auroc_from_regen(regen_path: Path,
                             problems: set[int] | None = None,
                             K: int | None = None,
                             ) -> tuple[float | None, float | None]:
    """Compute init_freq and distinct_freqs AUROC from regen JSONL.

    Uses per-problem macro averaging over the same slice as
    ``compute_transition_rates`` (problems with both correct and incorrect
    init answers), so ``AUROC_init = (1 + D) / 2`` holds for binary K=1
    scores by construction.

    Returns (auroc_init, auroc_dist).
    """
    from collections import Counter, defaultdict

    init_y_pp: dict[int, list[int]] = defaultdict(list)
    init_s_pp: dict[int, list[float]] = defaultdict(list)
    dist_y_pp: dict[int, list[int]] = defaultdict(list)
    dist_s_pp: dict[int, list[float]] = defaultdict(list)

    with open(regen_path) as f:
        for line in f:
            rec = json.loads(line)
            if problems is not None and rec["problem_num"] not in problems:
                continue
            pnum = rec["problem_num"]
            gold = str(rec["gold_answer"])
            regen_answers = rec.get("regen_answers", [])
            for i, pair in enumerate(rec["all_answers"]):
                if not is_usable_answer(pair):
                    continue
                if i >= len(regen_answers):
                    continue
                ans = str(pair[0])
                raw_regens = regen_answers[i] if K is None else regen_answers[i][:K]
                regens = [str(r[0]) for r in raw_regens
                          if is_usable_answer(r)]
                if not regens:
                    continue
                group = [ans] + regens
                counts = Counter(group)
                n = len(group)

                init_y_pp[pnum].append(1 if ans == gold else 0)
                init_s_pp[pnum].append(counts[ans] / n)

                for distinct_ans, count in counts.items():
                    dist_y_pp[pnum].append(1 if distinct_ans == gold else 0)
                    dist_s_pp[pnum].append(count / n)

    return (compute_macro_auroc(init_y_pp, init_s_pp),
            compute_macro_auroc(dist_y_pp, dist_s_pp))


# ── Computation ──

def _extract_problem_nums(regen_path: Path) -> set[int]:
    """Extract all problem_num values from a regen JSONL."""
    with open(regen_path) as f:
        return {json.loads(line)["problem_num"] for line in f}


def _total_problems_for(benchmark_label: str) -> int | None:
    """Look up total problem count for a benchmark label."""
    for key, label in BENCHMARK_LABELS.items():
        if label == benchmark_label:
            return DATASET_TOTAL_PROBLEMS.get(key)
    return None


def compute_transition_rates(regen_path: Path,
                             problems: set[int] | None = None,
                             K: int | None = None) -> dict:
    """Compute answer transition rates from a regen JSONL file.

    For each (init_answer, regen_answer) pair, classify into one of five
    transition types: C->C, C->W, W->C, W->sameW, W->otherW.

    Uses macro-averaging: rates are computed per problem, then averaged
    across problems that have both correct and incorrect init answers.
    This ensures consistency with AUROC (where AUROC = (1+D)/2 for
    binary scores).

    Returns dict with 'counts' (pooled) and 'rates' (macro-averaged).
    """
    # Per-problem counts
    per_problem = {}

    with open(regen_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            pnum = rec["problem_num"]
            if problems is not None and pnum not in problems:
                continue
            gold = str(rec["gold_answer"])
            regen_answers = rec.get("regen_answers", [])

            p_counts = {
                "c_to_c": 0, "c_to_w": 0,
                "w_to_c": 0, "w_to_same_w": 0, "w_to_other_w": 0,
                "total_correct": 0, "total_wrong": 0,
            }

            for i, pair in enumerate(rec["all_answers"]):
                if not is_usable_answer(pair):
                    continue
                init_ans = str(pair[0])
                init_correct = (init_ans == gold)

                if i >= len(regen_answers):
                    continue
                regs_i = regen_answers[i] if K is None else regen_answers[i][:K]
                for r in regs_i:
                    if not is_usable_answer(r):
                        continue
                    regen_ans = str(r[0])
                    regen_correct = (regen_ans == gold)

                    if init_correct:
                        p_counts["total_correct"] += 1
                        if regen_correct:
                            p_counts["c_to_c"] += 1
                        else:
                            p_counts["c_to_w"] += 1
                    else:
                        p_counts["total_wrong"] += 1
                        if regen_correct:
                            p_counts["w_to_c"] += 1
                        elif regen_ans == init_ans:
                            p_counts["w_to_same_w"] += 1
                        else:
                            p_counts["w_to_other_w"] += 1

            per_problem[pnum] = p_counts

    # Pooled counts (for backward compat / reference)
    counts = {k: 0 for k in ["c_to_c", "c_to_w", "w_to_c", "w_to_same_w",
                              "w_to_other_w", "total_correct", "total_wrong"]}
    for pc in per_problem.values():
        for k in counts:
            counts[k] += pc[k]

    # Macro-averaged rates (only problems with both correct and incorrect)
    r_c_list, r_w_list = [], []
    wc_list, wsw_list, wow_list = [], [], []

    for pc in per_problem.values():
        tc = pc["total_correct"]
        tw = pc["total_wrong"]
        if tc > 0 and tw > 0:
            r_c_list.append(pc["c_to_c"] / tc)
            r_w_list.append((pc["w_to_same_w"] + pc["w_to_other_w"]) / tw
                            if (pc["w_to_same_w"] + pc["w_to_other_w"]) > 0
                            else 0.0)
            wc_list.append(pc["w_to_c"] / tw)
            wsw_list.append(pc["w_to_same_w"] / tw)
            wow_list.append(pc["w_to_other_w"] / tw)

    import numpy as _np
    rates = {}
    rates["c_to_c"] = float(_np.mean(r_c_list)) if r_c_list else 0.0
    rates["c_to_w"] = 1.0 - rates["c_to_c"]
    rates["w_to_c"] = float(_np.mean(wc_list)) if wc_list else 0.0
    rates["w_to_same_w"] = float(_np.mean(wsw_list)) if wsw_list else 0.0
    rates["w_to_other_w"] = float(_np.mean(wow_list)) if wow_list else 0.0

    return {"counts": counts, "rates": rates}


# ── LaTeX helpers ──

_BOLD_COLS = {"c_to_c", "w_to_same_w", "D"}


def _fmt_rate(val: float | None, key: str = "") -> str:
    """Format a rate as percentage with 1 decimal place."""
    if val is None:
        return "---"
    s = f"{val * 100:.1f}"
    if key in _BOLD_COLS:
        return rf"\textbf{{{s}}}"
    return s


def _fmt_auroc(val: float | None) -> str:
    """Format AUROC as 3-decimal fraction."""
    if val is None:
        return "---"
    return f"{val:.3f}"


# ── Mode: rates ──

def _strip_footnote_mark(s: str) -> str:
    """Drop a trailing ``$^{x}$`` footnote marker, if any."""
    import re
    return re.sub(r"\s*\$\^\{[^}]*\}\$\s*$", "", s.strip())


def _strip_leading_markup(s: str) -> str:
    """Remove leading ``\\midrule`` / ``\\toprule`` / ``\\cmidrule(...){...}``
    etc. so only the data cell content remains."""
    import re
    return re.sub(
        r"^(?:\\(?:midrule|toprule|bottomrule|addlinespace(?:\[[^\]]*\])?"
        r"|cmidrule(?:\([^)]*\))?(?:\{[^}]*\})?)\s*)+",
        "", s.strip())


def _rates_group_key(cells):
    """Extract benchmark label from a ``\\multirow{N}{*}{Benchmark}`` on the
    first row of a benchmark block in table_transition_rates.tex merge-base.
    """
    if not cells:
        return None
    first = _strip_leading_markup(cells[0])
    import re
    m = re.match(r"^\\multirow\{\d+\}\{\*\}\{(.*)\}\s*$", first, re.DOTALL)
    if m:
        return _strip_footnote_mark(m.group(1).strip())
    return None


def _rates_row_key(cells, current_group):
    """Row-key extractor for table_transition_rates.tex merge-base.

    Layout: ``Benchmark(multirow) & Model & r_C & ...``. cells[0] carries
    the ``\\multirow`` marker on the first row of each group and is empty
    on subsequent rows; cells[1] is always the Model label.
    """
    if current_group is None or len(cells) < 2:
        return None
    model = _strip_footnote_mark(cells[1])
    if not model:
        return None
    return (current_group, model)


def _tau_group_key(cells):
    """Update the current ``(model, benchmark)`` group when this row
    begins a new block (non-empty first cell after markup stripping)."""
    first = _strip_leading_markup(cells[0])
    if not first:
        return None
    if len(cells) < 2:
        return None
    return (_strip_footnote_mark(first), _strip_footnote_mark(cells[1]))


def _tau_row_key(cells, current_group):
    """Row key is ``(model, benchmark, tau_str)``. cells[2] carries the
    truncation fraction as a plain string such as ``0.75``."""
    if current_group is None or len(cells) < 3:
        return None
    tau_str = cells[2].strip()
    if not tau_str:
        return None
    model, bench = current_group
    return (model, bench, tau_str)


def generate_rates_table(rows: list[dict], condition: str,
                         caption_note: str = "",
                         intersected: bool = False,
                         base_cells: dict | None = None,
                         include_auroc_dist: bool = False) -> str:
    """Generate table_transition_rates style LaTeX (single tau, grouped by dataset)."""
    tau, K = parse_condition(condition)
    cond_str = f"$\\tau{{=}}{tau}$, $K{{=}}{K}$" if tau else condition

    # Group by benchmark_label, ordered by canonical BENCHMARK_ORDER.
    # Within each benchmark, models are ordered by canonical MAIN_PAPER_MODELS.
    by_benchmark: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark_label"]].append(row)
    bench_order: list[str] = sorted(by_benchmark.keys(), key=bench_sort_key)
    for bl in bench_order:
        by_benchmark[bl].sort(key=lambda r: model_sort_key(r["model"]))

    # Build footnotes for subset experiments (deduplicate identical notes)
    fn_marks: dict[tuple[str, str], str] = {}  # (model, benchmark) -> mark
    fn_texts: list[str] = []
    text_to_mark: dict[str, str] = {}  # footnote text -> mark letter
    mark_idx = 0
    for bl in bench_order:
        total = _total_problems_for(bl)
        if total is None:
            continue
        for row in by_benchmark[bl]:
            n_prob = row.get("n_problems")
            if n_prob is not None and n_prob < total:
                fn_text = f"{n_prob} of {total} problems, $N$=64"
                if fn_text in text_to_mark:
                    fn_marks[(row["model"], bl)] = text_to_mark[fn_text]
                else:
                    mark = chr(ord("a") + mark_idx)
                    mark_idx += 1
                    text_to_mark[fn_text] = mark
                    fn_marks[(row["model"], bl)] = mark
                    fn_texts.append(f"$^{{{mark}}}${fn_text}.")

    note = f" {caption_note}" if caption_note else ""
    extra_cols = _extra_cols(include_auroc_dist)
    all_cols = RATE_COLS + extra_cols
    n_all_cols = len(all_cols)

    if intersected:
        subset_note = (
            " Within each dataset group, only the intersection of "
            "problems shared by all models is used.")
        label = r"\label{tab:transition_rates_intersect}"
    else:
        subset_note = ""
        label = r"\label{tab:transition_rates}"

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Per-problem macro-averaged transition rates (\%) "
        f"at {cond_str}. "
        r"The three rate columns $r_C, r_W, \phi_{WC}$ "
        r"are defined in Eq.~\eqref{eq:rates}; "
        r"$D = r_C - r_W$ is the discrimination gap. "
        r"The reported $\overline{\mathrm{AUROC}}$ uses the init-frequency score. "
        r"At $K = 1$ it satisfies "
        r"$\overline{\mathrm{AUROC}} = (1 + D)/2$ by construction."
        f"{subset_note}{note}}}")
    lines.append(label)
    lines.append(r"\small")

    # Two label cols (Benchmark, Model) with benchmark carried by
    # \multirow on the first model row of each group, so the group
    # header \multicolumn row is unnecessary.
    col_spec = "@{}ll" + "r" * n_all_cols + "@{}"
    col_headers = " & ".join(
        ["Benchmark", "Model"] + [h for _, h in all_cols])
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(col_headers + r" \\")
    lines.append(r"\midrule")

    for bi, bl in enumerate(bench_order):
        if bi > 0:
            lines.append(r"\midrule")

        group_rows = by_benchmark[bl]
        n_rows = len(group_rows)
        for ri, row in enumerate(group_rows):
            rates = row["rates"]
            model_display = row["model"]
            mark = fn_marks.get((row["model"], bl))
            if mark:
                model_display = f"{model_display}$^{{{mark}}}$"
            if ri == 0:
                bench_cell = f"\\multirow{{{n_rows}}}{{*}}{{{bl}}}"
            else:
                bench_cell = ""
            cells = [bench_cell, model_display]
            row_key = (bl, row["model"])
            col_idx = 2  # data starts after (Benchmark, Model) label cells
            for key, _ in RATE_COLS:
                cell = _fmt_rate(rates.get(key), key)
                cell = fallback_cell(cell, (row_key, col_idx), base_cells)
                cells.append(cell)
                col_idx += 1
            # D = C->C - W->sameW
            d_val = rates.get("c_to_c", 0) - rates.get("w_to_same_w", 0)
            cells.append(fallback_cell(
                _fmt_rate(d_val, "D"), (row_key, col_idx), base_cells))
            col_idx += 1
            # Canonical AUROC (init-answer PC score) AUROC = (1+D)/2 for PC at K=1.
            cells.append(fallback_cell(
                _fmt_auroc(row.get("auroc_init")),
                (row_key, col_idx), base_cells))
            col_idx += 1
            if include_auroc_dist:
                cells.append(fallback_cell(
                    _fmt_auroc(row.get("auroc_dist")),
                    (row_key, col_idx), base_cells))
                col_idx += 1
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if fn_texts:
        lines.append(r"\par\vspace{2pt}")
        lines.append(
            r"\parbox{\textwidth}{\raggedright\footnotesize "
            + " ".join(fn_texts) + "}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ── Mode: tau ──

def generate_tau_table(rows: list[dict], caption_note: str = "",
                       base_cells: dict | None = None,
                       include_auroc_dist: bool = False) -> str:
    """Generate table_transition_tau style LaTeX (multi-tau, grouped by model x dataset)."""
    # Group by (model, benchmark_label), ordered by canonical model and benchmark order.
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_group[(row["model"], row["benchmark_label"])].append(row)
    group_order: list[tuple[str, str]] = sorted(
        by_group.keys(),
        key=lambda mb: (model_sort_key(mb[0]), bench_sort_key(mb[1])),
    )

    note = f" {caption_note}" if caption_note else ""
    n_rate_cols = len(RATE_COLS)

    lines = []
    lines.append(r"\begin{table}[H]")
    # Tau mode reports r_C, r_W, D, AUROC; phi_{WC} is omitted to keep
    # the table compact and aligned with the caption / main.tex narrative.
    rate_cols = [c for c in RATE_COLS if c[0] != "w_to_c"]
    extra_cols = _extra_cols(include_auroc_dist)
    all_cols = rate_cols + extra_cols
    n_all_cols = len(all_cols)

    lines.append(r"\centering")
    lines.append(
        r"\caption{\textbf{Reproduction rates (\%) across truncation "
        r"fractions $\tau \in \{0.75, 0.50, 0.25\}$ at $K{=}1$, "
        r"GPT-OSS-20B (larger $D$ is better).} "
        r"Column symbols ($r_C$, $r_W$, $D$) follow Table~\ref{tab:signal}, "
        r"with $\overline{\mathrm{AUROC}} = (1+D)/2$ at $K{=}1$. "
        r"Macro-averaged over problems with at least one correct and one "
        r"wrong initial sample."
        f"{note}}}")
    lines.append(r"\label{tab:transition_tau}")
    lines.append(r"\small")

    col_spec = "@{}lll" + "r" * n_all_cols + "@{}"
    col_headers = " & ".join(
        ["Model", "Benchmark", r"$\tau$"] + [h for _, h in all_cols])
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(col_headers + r" \\")
    lines.append(r"\midrule")

    for gi, (model, bl) in enumerate(group_order):
        if gi > 0:
            lines.append(r"\midrule")

        group_rows = sorted(by_group[(model, bl)],
                            key=lambda r: r["tau"], reverse=True)

        for ri, row in enumerate(group_rows):
            rates = row["rates"]
            m_cell = model if ri == 0 else ""
            d_cell = bl if ri == 0 else ""
            tau_str = f"{row['tau']:.2f}"

            cells = [m_cell, d_cell, tau_str]
            row_key = (model, bl, tau_str)
            col_idx = 3  # data starts after (model, benchmark, tau) label cells
            for key, _ in rate_cols:
                cell = _fmt_rate(rates.get(key), key)
                cells.append(fallback_cell(cell, (row_key, col_idx), base_cells))
                col_idx += 1
            d_val = rates.get("c_to_c", 0) - rates.get("w_to_same_w", 0)
            cells.append(fallback_cell(
                _fmt_rate(d_val, "D"), (row_key, col_idx), base_cells))
            col_idx += 1
            cells.append(fallback_cell(
                _fmt_auroc(row.get("auroc_init")),
                (row_key, col_idx), base_cells))
            col_idx += 1
            if include_auroc_dist:
                cells.append(fallback_cell(
                    _fmt_auroc(row.get("auroc_dist")),
                    (row_key, col_idx), base_cells))
                col_idx += 1
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ── Mode: size ──

def _parse_families(family_strs: list[str]) -> list[tuple[str, list[str]]]:
    """Parse --families args like 'GPT-OSS:GPT-OSS-120B (high),GPT-OSS-20B'."""
    families = []
    for s in family_strs:
        if ":" not in s:
            raise SystemExit(
                f"Invalid --families format: '{s}'. "
                f"Expected 'FamilyName:Model1,Model2'.")
        name, members_str = s.split(":", 1)
        members = [m.strip() for m in members_str.split(",")]
        families.append((name.strip(), members))
    return families


def generate_size_tables(rows: list[dict],
                         families: list[tuple[str, list[str]]],
                         condition: str,
                         footnotes: dict[str, str] | None = None,
                         include_auroc_dist: bool = False,
                         ) -> str:
    """Generate table_analysis_flip_vs_size style LaTeX (one table per family)."""
    tau, K = parse_condition(condition)
    cond_str = f"$\\tau{{=}}{tau}$, $K{{=}}{K}$" if tau else condition

    # Index rows by (model, benchmark_label)
    row_index: dict[tuple[str, str], dict] = {}
    for row in rows:
        row_index[(row["model"], row["benchmark_label"])] = row

    footnotes = footnotes or {}
    extra_cols = _extra_cols(include_auroc_dist)
    all_cols = RATE_COLS + extra_cols
    n_all_cols = len(all_cols)
    tables = []

    for family_name, family_models in families:
        # Collect benchmarks in input order
        benchmarks: list[str] = []
        seen: set[str] = set()
        for row in rows:
            bl = row["benchmark_label"]
            if row["model"] in family_models and bl not in seen:
                seen.add(bl)
                benchmarks.append(bl)

        # Build footnote map for this family's benchmarks
        fn_map: dict[str, str] = {}
        for bl in benchmarks:
            if bl in footnotes:
                fn_map[bl] = chr(ord("a") + len(fn_map))

        label_slug = (family_name.lower()
                      .replace(" ", "_").replace("-", "_"))

        # Caption with footnote references
        fn_refs = ""
        if fn_map:
            fn_refs = (
                " "
                + " ".join(
                    f"$^{{{mark}}}${footnotes[bl]}."
                    for bl, mark in fn_map.items())
            )

        lines = []
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(
            f"\\caption{{Answer transition rates (\\%) for "
            f"{family_name} family ({cond_str}). "
            r"The larger model shows higher W\textrightarrow C and "
            r"lower W\textrightarrow sameW, indicating a wider "
            r"discrimination gap $D$."
            f"{fn_refs}}}")
        lines.append(f"\\label{{tab:flip_rate_{label_slug}}}")
        lines.append(r"\footnotesize")
        lines.append(r"\resizebox{\textwidth}{!}{%")

        col_headers = " & ".join(
            ["Model", "Benchmark"] + [h for _, h in all_cols])
        lines.append(
            r"\begin{tabular}{ll" + "r" * n_all_cols + r"}")
        lines.append(r"\toprule")
        lines.append(col_headers + r" \\")
        lines.append(r"\midrule")

        for bi, bl in enumerate(benchmarks):
            if bi > 0:
                lines.append(r"\midrule")

            bl_display = bl
            if bl in fn_map:
                bl_display = f"{bl}$^{{{fn_map[bl]}}}$"

            for model in family_models:
                key = (model, bl)
                if key in row_index:
                    rates = row_index[key]["rates"]
                    cells = [model, bl_display]
                    for rk, _ in RATE_COLS:
                        cells.append(_fmt_rate(rates.get(rk), rk))
                    d_val = (rates.get("c_to_c", 0)
                             - rates.get("w_to_same_w", 0))
                    cells.append(_fmt_rate(d_val, "D"))
                    cells.append(_fmt_auroc(
                        row_index[key].get("auroc_init")))
                    if include_auroc_dist:
                        cells.append(_fmt_auroc(
                            row_index[key].get("auroc_dist")))
                else:
                    cells = [model, bl_display] + ["--"] * n_all_cols
                lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}% end resizebox")
        lines.append(r"\end{table}")
        tables.append("\n".join(lines))

    return "\n\n".join(tables)


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX transition rate tables")
    parser.add_argument("jsonl_dirs", nargs="+", type=Path,
                        help="JSONL directories to analyze")
    parser.add_argument("--mode", choices=["rates", "tau", "size"],
                        default="rates",
                        help="Table format (default: rates)")
    parser.add_argument("--condition", type=str, default=None,
                        help="Condition for rates/size mode "
                             "(e.g. rm25pct_full_x1)")
    parser.add_argument("--conditions", nargs="+", type=str, default=None,
                        help="Multiple conditions for tau mode")
    parser.add_argument("--families", nargs="+", type=str, default=None,
                        help="Family specs for size mode "
                             "(e.g. 'GPT-OSS:GPT-OSS-120B,GPT-OSS-20B')")
    parser.add_argument("--caption-note", type=str, default="",
                        help="Extra text appended to table caption")
    parser.add_argument("--intersect", action="store_true",
                        help="Intersect problem sets per dataset group "
                             "(rates mode only)")
    parser.add_argument("--problems", type=int, nargs="+", default=None,
                        help="Problem indices to include (default: all)")
    parser.add_argument("--problems-from", type=Path, default=None,
                        help="JSONL dir to extract problem indices from")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output .tex file (default: stdout)")
    parser.add_argument("--merge-base", type=Path, default=None,
                        help="Existing output .tex; cells this run cannot "
                             "compute locally fall back to its values. "
                             "Supported for rates and tau modes.")
    parser.add_argument("--K", type=int, default=1,
                        help="Truncate regen_answers[i][:K] for each init. "
                             "Default 1 enforces K=1 (binary score) where "
                             "AUROC = (1+D)/2 holds. Use larger K for "
                             "K-sweep sensitivity.")
    parser.add_argument("--include-auroc-dist", action="store_true",
                        help="Also emit an AUROC$_\\text{dist}$ column "
                             "(AUROC over every distinct candidate answer's "
                             "PC score). Off by default; the single ``AUROC'' "
                             "column in the default output uses the init-"
                             "answer PC score $c_i(a_i)$, for which "
                             "$\\mathrm{AUROC} = (1 + D) / 2$ at $K=1$.")
    args = parser.parse_args()

    # Resolve conditions
    if args.mode == "tau":
        conditions = args.conditions or (
            [args.condition] if args.condition else [])
        if not conditions:
            parser.error("--conditions (or --condition) required for tau mode")
    else:
        if not args.condition:
            parser.error("--condition required for rates/size mode")
        conditions = [args.condition]

    if args.mode == "size" and not args.families:
        parser.error("--families required for size mode")

    # Resolve problem filter
    problem_set = None
    if args.problems:
        problem_set = set(args.problems)
    elif args.problems_from:
        init_path = find_init_file(args.problems_from)
        if init_path:
            with open(init_path) as f:
                problem_set = {json.loads(line)["problem_num"]
                               for line in f}
            print(f"Problem filter from {args.problems_from.name}: "
                  f"{len(problem_set)} problems")

    # ── Size mode: intersect problem sets per (family, benchmark) ──
    if args.mode == "size":
        families = _parse_families(args.families)
        family_of: dict[str, str] = {}
        for fname, fmodels in families:
            for m in fmodels:
                family_of[m] = fname

        # Pass 1: discover regen paths and problem sets
        cond = conditions[0]
        regen_info: dict[tuple[str, str], tuple[Path, set[int]]] = {}
        for d in args.jsonl_dirs:
            dir_name = d.name
            model = infer_model_label(dir_name)
            bl = infer_benchmark_label(dir_name)
            regen_path = find_regen_for_condition(d, cond)
            if regen_path is None:
                print(f"  WARNING: no regen file for {cond} in {dir_name}")
                continue
            pnums = _extract_problem_nums(regen_path)
            regen_info[(model, bl)] = (regen_path, pnums)

        # Intersect problem sets per (family, benchmark)
        intersected: dict[tuple[str, str], set[int]] = {}
        footnotes: dict[str, str] = {}  # benchmark_label -> footnote text
        for fname, fmodels in families:
            benchmarks_seen: set[str] = set()
            for (m, bl) in regen_info:
                if m in fmodels:
                    benchmarks_seen.add(bl)
            for bl in benchmarks_seen:
                sets = [regen_info[(m, bl)][1]
                        for m in fmodels if (m, bl) in regen_info]
                if not sets:
                    continue
                common = sets[0].intersection(*sets[1:])
                intersected[(fname, bl)] = common
                total = _total_problems_for(bl)
                if total and len(common) < total:
                    footnotes[bl] = (
                        f"{len(common)} of {total} problems")

        # Pass 2: compute rates on intersected problem sets
        all_rows: list[dict] = []
        for (model, bl), (regen_path, _) in regen_info.items():
            fname = family_of.get(model)
            ps = intersected.get((fname, bl), problem_set)
            tau, K = parse_condition(cond)
            data = compute_transition_rates(regen_path, ps, K=args.K)
            tc = data["counts"]["total_correct"]
            tw = data["counts"]["total_wrong"]
            if tc + tw == 0:
                print(f"  WARNING: no regen pairs for {model}/{bl}, "
                      f"skipping")
                continue
            w2c = data["rates"]["w_to_c"] * 100
            n_prob = len(ps) if ps else "all"
            auroc_init, auroc_dist = compute_auroc_from_regen(regen_path, ps, K=args.K)
            print(f"  {model} / {bl} @ tau={tau}: "
                  f"W->C={w2c:.1f}%  (n_correct={tc}, n_wrong={tw}, "
                  f"problems={n_prob})")
            all_rows.append({
                "model": model,
                "benchmark_label": bl,
                "condition": cond,
                "tau": tau,
                "rates": data["rates"],
                "counts": data["counts"],
                "auroc_init": auroc_init,
                "auroc_dist": auroc_dist,
            })

        if not all_rows:
            print("No data found.")
            return
        tex = generate_size_tables(
            all_rows, families, conditions[0], footnotes,
            include_auroc_dist=args.include_auroc_dist)

    # ── Rates / tau mode ──
    else:
        # Pass 1: discover regen paths and problem sets
        regen_paths: dict[tuple[str, str, str], tuple[Path, set[int], Path]] = {}
        for d in args.jsonl_dirs:
            dir_name = d.name
            model = infer_model_label(dir_name)
            benchmark_label = infer_benchmark_label(dir_name)
            for cond in conditions:
                regen_path = find_regen_for_condition(d, cond)
                if regen_path is None:
                    print(f"  WARNING: no regen file for {cond} "
                          f"in {dir_name}")
                    continue
                pnums = _extract_problem_nums(regen_path)
                regen_paths[(model, benchmark_label, cond)] = (
                    regen_path, pnums, d)

        # Intersect problem sets per (benchmark, condition) if requested
        intersected: dict[tuple[str, str], set[int]] = {}
        if args.intersect and args.mode == "rates":
            by_bench_cond: dict[tuple[str, str], list[set[int]]] = (
                defaultdict(list))
            for (model, bl, cond), (_, pnums, _d) in regen_paths.items():
                by_bench_cond[(bl, cond)].append(pnums)
            for (bl, cond), pnum_sets in by_bench_cond.items():
                common = pnum_sets[0].intersection(*pnum_sets[1:])
                intersected[(bl, cond)] = common
                total = _total_problems_for(bl)
                n_all = max(len(s) for s in pnum_sets)
                print(f"  Intersect {bl} ({cond}): "
                      f"{n_all} -> {len(common)} common problems")

        # Pass 2: compute transition rates
        all_rows = []
        for (model, benchmark_label, cond), (regen_path, pnums, jsonl_dir) in (
                regen_paths.items()):
            tau, K = parse_condition(cond)
            ps = intersected.get(
                (benchmark_label, cond), problem_set)
            n_problems = len(ps) if ps else len(pnums)
            data = compute_transition_rates(regen_path, ps, K=args.K)

            tc = data["counts"]["total_correct"]
            tw = data["counts"]["total_wrong"]
            if tc + tw == 0:
                print(f"  WARNING: no regen pairs for "
                      f"{model}/{benchmark_label} ({cond}), skipping")
                continue
            w2c = data["rates"]["w_to_c"] * 100
            auroc_init, auroc_dist = compute_auroc_from_regen(
                regen_path, ps, K=args.K)
            print(f"  {model} / {benchmark_label} @ tau={tau}: "
                  f"W->C={w2c:.1f}%  (n_correct={tc}, n_wrong={tw})")

            all_rows.append({
                "model": model,
                "benchmark_label": benchmark_label,
                "condition": cond,
                "tau": tau,
                "rates": data["rates"],
                "counts": data["counts"],
                "n_problems": n_problems,
                "auroc_init": auroc_init,
                "auroc_dist": auroc_dist,
            })

        if not all_rows:
            print("No data found.")
            return

        extra_col_count = len(_extra_cols(args.include_auroc_dist))
        if args.mode == "rates":
            base_cells = None
            if args.merge_base:
                # Data cols per row = 1 model label + RATE_COLS + extras.
                # cells[0] is the benchmark (multirow, carried by group key).
                expected_cols = 1 + len(RATE_COLS) + extra_col_count
                base_cells = load_base_cells_checked(
                    args.merge_base,
                    _rates_row_key,
                    expected_cols=expected_cols,
                    label="merge-base",
                    group_key_fn=_rates_group_key,
                )
            tex = generate_rates_table(
                all_rows, conditions[0], args.caption_note,
                intersected=bool(intersected),
                base_cells=base_cells,
                include_auroc_dist=args.include_auroc_dist)
        else:
            base_cells = None
            if args.merge_base:
                # Tau mode cells[1..] = benchmark + tau + RATE_COLS + extras.
                expected_cols = 2 + len(RATE_COLS) + extra_col_count
                base_cells = load_base_cells_checked(
                    args.merge_base,
                    _tau_row_key,
                    expected_cols=expected_cols,
                    label="merge-base",
                    group_key_fn=_tau_group_key,
                )
            tex = generate_tau_table(all_rows, args.caption_note,
                                     base_cells=base_cells,
                                     include_auroc_dist=args.include_auroc_dist)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex + "\n")
        print(f"\nWritten: {args.output}")
    else:
        print()
        print(tex)


if __name__ == "__main__":
    main()
