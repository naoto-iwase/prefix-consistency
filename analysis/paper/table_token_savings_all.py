#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Generate per-model LaTeX appendix tables: all-baselines token efficiency.

One table per model.  Columns: (benchmark x threshold).  Rows: methods
grouped via ALL_BASELINE_GROUPS, preceded by 3 header rows (Pass@1,
Standard MV plateau, Standard MV budget per threshold).  Mirrors
table_wmv_all.py for the appendix.  Standard MV is omitted as a row
(it is the reference denominator).

Each --model group with no jsonl directories renders as an all-``---``
placeholder.

Usage:
    cd analysis && \\
    uv run python paper/table_token_savings_all.py \\
        --condition rm25pct_full_x1 \\
        --thresholds 0.75 0.90 0.99 \\
        --output-dir tables \\
        --model "GPT-OSS-120B" \\
            gpt-oss-120b_hmmt_jsonl ... \\
        --model "Nemotron2-9B"   # placeholder
"""

import argparse
import math
import re
import sys
from pathlib import Path

from _utils import (
    ALL_BASELINE_GROUPS,
    parse_condition, infer_benchmark_label, detect_subset_footnote,
)
from table_token_savings import (
    compute_plateau_savings,
    _format_ratio, _format_acc, _format_budget,
    RATIO_LOW_CAP, RATIO_HIGH_CAP,
)
from _defs import BENCHMARK_ORDER


DEFAULT_THRESHOLDS = [0.75, 0.90, 0.99]
DEFAULT_BENCHMARKS = list(BENCHMARK_ORDER)


def _model_slug(model_name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    return s


def _format_threshold(thr: float) -> str:
    pct = thr * 100
    if abs(pct - round(pct)) < 1e-6:
        return f"{int(round(pct))}\\%"
    return f"{pct:.1f}\\%"


def _parse_model_args(raw: list[str]) -> list[tuple[str, list[Path]]]:
    groups: list[tuple[str, list[Path]]] = []
    current_name: str | None = None
    current_dirs: list[Path] = []
    for arg in raw:
        if arg == "--model":
            if current_name is not None:
                groups.append((current_name, current_dirs))
            current_name = None
            current_dirs = []
        elif current_name is None and not Path(arg).exists():
            current_name = arg
        else:
            current_dirs.append(Path(arg))
    if current_name is not None:
        groups.append((current_name, current_dirs))
    return groups


def load_model_data(
    dirs: list[Path],
    condition: str,
    thresholds: list[float],
    method_keys: list[str],
    analysis_dir: Path,
) -> tuple[dict[str, dict], dict[str, str | None]]:
    """Per-bench: {pass_at_1, mv_plateau, mv_budgets, ratios}."""
    by_bench: dict[str, dict] = {}
    footnotes: dict[str, str | None] = {}
    for d in dirs:
        if not d.is_absolute() and not d.exists():
            d = analysis_dir / d
        json_path = d / "wmv_result.json"
        if not json_path.exists():
            print(f"  WARNING: {json_path} not found, skipping",
                  file=sys.stderr)
            continue
        label = infer_benchmark_label(d.name)
        print(f"  Loading {d.name} -> {label}")
        plateau, p1_acc, mv_budgets, ratios = compute_plateau_savings(
            json_path, condition, method_keys, thresholds)
        by_bench[label] = {
            "pass_at_1": p1_acc,
            "mv_plateau": plateau,
            "mv_budgets": mv_budgets,
            "ratios": ratios,
        }
        footnotes[label] = detect_subset_footnote(json_path, d.name)
    return by_bench, footnotes


def generate_tex(
    model_name: str,
    by_bench: dict[str, dict],
    footnotes: dict[str, str | None],
    benchmarks: list[str],
    thresholds: list[float],
    condition: str,
    suppress_condition: bool = False,
    suppress_subset_footnote: bool = False,
) -> str:
    n_thr = len(thresholds)
    n_b = len(benchmarks)
    n_data_cols = n_b * n_thr

    # Footnote bookkeeping (table-level if all benches share, else per-col).
    if suppress_subset_footnote:
        fn_texts = [None for _ in benchmarks]
    else:
        fn_texts = [footnotes.get(b) for b in benchmarks]
    nonempty = [t for t in fn_texts if t]
    table_level_footnote: str | None = None
    bench_marks: dict[str, str] = {}
    text_to_letter: dict[str, str] = {}
    if nonempty and len(set(nonempty)) == 1 and len(nonempty) == len(benchmarks):
        table_level_footnote = nonempty[0]
    else:
        for b in benchmarks:
            fn = None if suppress_subset_footnote else footnotes.get(b)
            if not fn:
                continue
            if fn not in text_to_letter:
                text_to_letter[fn] = chr(ord("a") + len(text_to_letter))
            bench_marks[b] = text_to_letter[fn]

    tau, K = parse_condition(condition)
    if tau is not None and K is not None:
        cond_str = f"$\\tau{{=}}{tau}$, $K{{=}}{K}$"
    else:
        cond_str = condition.replace("_", "\\_")
    cond_suffix = "" if suppress_condition else f", {cond_str}"

    slug = _model_slug(model_name)
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{\\textbf{{Token efficiency, {model_name} "
        f"(smaller is better{cond_suffix}).}} "
        f"Cells $<$1 indicate more cost-efficient than Standard MV. "
        f"``N/A'' indicates target unreachable. "
        f"Bold marks the best per column.}}")
    lines.append(f"\\label{{tab:token_savings_{slug}_all}}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{1.5pt}")
    lines.append("\\renewcommand{\\arraystretch}{0.8}")
    lines.append("\\resizebox{\\textwidth}{!}{%")

    col_spec = "l " + " ".join(["c" * n_thr] * n_b)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Row 1: benchmark labels spanning thresholds.
    bench_headers = []
    for b in benchmarks:
        mark = bench_marks.get(b)
        suffix = f"$^{{{mark}}}$" if mark else ""
        label = b.replace("&", "\\&") + suffix
        bench_headers.append(f"\\multicolumn{{{n_thr}}}{{c}}{{{label}}}")
    lines.append(" & " + " & ".join(bench_headers) + " \\\\")

    cmid_parts = []
    col_idx = 2
    for _ in benchmarks:
        cmid_parts.append(
            f"\\cmidrule(lr){{{col_idx}-{col_idx + n_thr - 1}}}")
        col_idx += n_thr
    lines.append(" ".join(cmid_parts))

    # Row 2: per-benchmark threshold labels.
    thr_hdrs = [f"$\\alpha{{=}}{_format_threshold(t)}$" for t in thresholds]
    lines.append(" & " + " & ".join(thr_hdrs * n_b) + " \\\\")
    lines.append("\\midrule")

    # Header rows: combined Pass@1 / Standard MV plateau (per-bench scalar
    # pair, in the first threshold cell with the rest left empty, mirroring
    # Table~\ref{tab:token_savings}), then Standard MV budget (per-threshold).
    p1plat_cells = ["Pass@1 / Standard MV plateau"]
    bud_cells = ["Standard MV budget ($B_{\\mathrm{MV}}$)"]
    for b in benchmarks:
        bdata = by_bench.get(b, {})
        p1 = bdata.get("pass_at_1")
        plat = bdata.get("mv_plateau")
        mv_b = bdata.get("mv_budgets", {})
        p1_s = _format_acc(p1) if p1 is not None else "---"
        plat_s = _format_acc(plat) if plat is not None else "---"
        p1plat_cells.append(f"{p1_s} / {plat_s}")
        for _ in range(n_thr - 1):
            p1plat_cells.append("")
        for t in thresholds:
            bud_cells.append(_format_budget(mv_b.get(t)))
    lines.append(" & ".join(p1plat_cells) + " \\\\")
    lines.append(" & ".join(bud_cells) + " \\\\")
    lines.append("\\midrule")

    # Filter out methods with no data for this model, except when the
    # model is a pure placeholder (no benchmarks loaded): in that case
    # we keep the core schema visible and only drop the optional
    # ``markers`` row.  ``standard_mv`` is the denominator and always
    # omitted from the rendered method rows.
    is_placeholder = not by_bench
    OPTIONAL_KEYS = {"markers"}
    def _has_any_data(key: str) -> bool:
        for b in benchmarks:
            ratios = by_bench.get(b, {}).get("ratios", {})
            if key in ratios:
                return True
        return False
    def _include(key: str) -> bool:
        if key == "standard_mv":
            return False
        if _has_any_data(key):
            return True
        if is_placeholder and key not in OPTIONAL_KEYS:
            return True
        return False
    rendered_groups: list[tuple[str, list[tuple[str, str]]]] = []
    for group_name, methods in ALL_BASELINE_GROUPS:
        members = [(k, d) for k, d in methods if _include(k)]
        if members:
            rendered_groups.append((group_name, members))

    # Best ratio per (bench, thr) for bolding (Standard MV excluded).
    best_per_col: dict[tuple[str, float], float] = {}
    candidate_keys: list[str] = []
    for _, methods in rendered_groups:
        for k, _ in methods:
            candidate_keys.append(k)
    for b in benchmarks:
        bdata = by_bench.get(b, {})
        ratios = bdata.get("ratios", {})
        for t in thresholds:
            valid: list[float] = []
            for k in candidate_keys:
                if k not in ratios:
                    continue
                entry = ratios[k].get(t)
                if entry is None:
                    continue
                r, _ = entry
                if RATIO_LOW_CAP <= r <= RATIO_HIGH_CAP:
                    valid.append(round(r, 2))
            best_per_col[(b, t)] = min(valid) if valid else float("inf")

    # Method rows (rendered_groups already filters out standard_mv and
    # any methods without data for this model).  ``first_emitted``
    # ensures we don't double-\midrule when an early group is empty.
    first_emitted = True
    for group_name, members in rendered_groups:
        if not first_emitted:
            lines.append("\\midrule")
        first_emitted = False
        lines.append(
            f"\\multicolumn{{{1 + n_data_cols}}}{{l}}"
            f"{{\\textit{{{group_name}}}}} \\\\")
        for key, display in members:
            cells = [f"\\quad {display}"]
            for b in benchmarks:
                bdata = by_bench.get(b, {})
                ratios = bdata.get("ratios", {})
                for t in thresholds:
                    # ``---`` means the method has no entry at all for this
                    # (bench, model); ``N/A`` (emitted by _format_ratio when
                    # entry is None) means we have data but the method's
                    # plateau is below target.  Distinguishing the two is
                    # what the original table_token_savings.py does.
                    if not ratios or key not in ratios:
                        cells.append("---")
                        continue
                    entry = ratios[key].get(t)
                    if entry is None:
                        is_best = False
                    else:
                        r, _ = entry
                        best = best_per_col.get((b, t), float("inf"))
                        is_best = (
                            math.isfinite(best)
                            and best < 0.95
                            and RATIO_LOW_CAP <= r <= RATIO_HIGH_CAP
                            and round(r, 2) == best
                        )
                    cells.append(_format_ratio(entry, is_best))
            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}% end resizebox")

    if table_level_footnote:
        lines.append("\\vspace{2pt}")
        lines.append(
            f"\\par\\raggedright\\footnotesize {table_level_footnote}.")
    elif text_to_letter:
        notes = " ".join(
            f"$^{{{letter}}}${text}."
            for text, letter in text_to_letter.items())
        lines.append("\\vspace{2pt}")
        lines.append(f"\\par\\raggedright\\footnotesize {notes}")

    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Per-model all-baselines token-efficiency table")
    parser.add_argument("--condition", default="rm25pct_full_x1")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=DEFAULT_THRESHOLDS)
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename-template",
                        default="table_token_savings_{slug}_all.tex")
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--suppress-condition", action="store_true",
                        help="Omit `tau, K` from the caption.")
    parser.add_argument("--suppress-subset-footnote", action="store_true",
                        help="Omit per-cell ``N=...`` / problem-subset footnotes.")
    args, remaining = parser.parse_known_args()

    for t in args.thresholds:
        if not (0 < t <= 1):
            parser.error(f"thresholds must be in (0, 1]; got {t}")

    method_keys: list[str] = []
    for _, methods in ALL_BASELINE_GROUPS:
        method_keys.extend(k for k, _ in methods)

    model_groups = _parse_model_args(remaining)
    if not model_groups:
        parser.error("Pass at least one --model NAME [dir ...]")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, dirs in model_groups:
        print(f"\n[{model_name}] {len(dirs)} dirs")
        by_bench, footnotes = load_model_data(
            dirs, args.condition, args.thresholds, method_keys,
            args.analysis_dir)
        slug = _model_slug(model_name)
        out_path = args.output_dir / args.filename_template.format(slug=slug)
        tex = generate_tex(
            model_name, by_bench, footnotes,
            args.benchmarks, args.thresholds, args.condition,
            suppress_condition=args.suppress_condition,
            suppress_subset_footnote=args.suppress_subset_footnote)
        out_path.write_text(tex + "\n")
        print(f"  Written: {out_path}")


if __name__ == "__main__":
    main()
