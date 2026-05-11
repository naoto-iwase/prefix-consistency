#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate per-model LaTeX appendix tables comparing all baseline methods.

One table per model.  Columns: (benchmark x token_budget).  Rows: all
baseline methods grouped via ALL_BASELINE_GROUPS (Baseline / DeepConf /
DeepConf filtered / CISC / Prefix consistency).  Used for the paper appendix.

Each --model group with no jsonl directories renders as an all-``---``
placeholder (intended for models whose runs are not yet finished).

Usage:
    cd analysis && \\
    uv run python paper/table_wmv_all.py \\
        --condition rm25pct_full_x1 \\
        --token-points 250000 1000000 5000000 \\
        --output-dir tables \\
        --model "GPT-OSS-120B" \\
            gpt-oss-120b_hmmt_jsonl \\
            gpt-oss-120b_frontierscience_olympiad_jsonl \\
            gpt-oss-120b_aime2025_jsonl \\
            gpt-oss-120b_brumo_jsonl \\
        --model "Nemotron-Nano-9B-v2"   # placeholder
"""

import argparse
import json
import re
from pathlib import Path

from _utils import (
    ALL_BASELINE_GROUPS, format_tp, format_acc, load_wmv_methods,
    parse_condition, infer_benchmark_label,
    detect_subset_footnote,
)
from _defs import BENCHMARK_ORDER


DEFAULT_TOKEN_POINTS = [250_000, 1_000_000, 5_000_000]
DEFAULT_BENCHMARKS = list(BENCHMARK_ORDER)


def _model_slug(model_name: str) -> str:
    """Lowercased, non-alphanum -> underscores; for filenames/labels."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    return s


def _parse_model_args(raw: list[str]) -> list[tuple[str, list[Path]]]:
    """Parse repeated ``--model NAME [dir ...]``; allow zero dirs."""
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
    token_points: list[int],
    analysis_dir: Path,
) -> tuple[dict[str, dict], dict[str, str | None]]:
    """Return ({bench_label: {key: {tp: (acc, ci)}}}, {bench_label: footnote})."""
    by_bench: dict[str, dict] = {}
    footnotes: dict[str, str | None] = {}
    for d in dirs:
        # Resolve relative dirs against --analysis-dir.
        if not d.is_absolute() and not d.exists():
            d = analysis_dir / d
        json_path = d / "wmv_result.json"
        if not json_path.exists():
            print(f"  WARNING: {json_path} not found, skipping")
            continue
        if json_path.stat().st_size == 0:
            print(f"  WARNING: {json_path} empty, skipping")
            continue
        try:
            data = load_wmv_methods(json_path, condition, token_points)
        except json.JSONDecodeError as e:
            print(f"  WARNING: {json_path} invalid JSON ({e}), skipping")
            continue
        label = infer_benchmark_label(d.name)
        by_bench[label] = data
        footnotes[label] = detect_subset_footnote(json_path, d.name)
        print(f"  Loaded {d.name} -> {label} ({len(data)} methods)")
    return by_bench, footnotes


def generate_tex(
    model_name: str,
    by_bench: dict[str, dict],
    footnotes: dict[str, str | None],
    benchmarks: list[str],
    token_points: list[int],
    condition: str,
    suppress_condition: bool = False,
    suppress_subset_footnote: bool = False,
) -> str:
    """One per-model LaTeX table (full all-baselines layout)."""
    n_tp = len(token_points)
    n_b = len(benchmarks)
    n_data_cols = n_b * n_tp

    # Footnote bookkeeping.  If every benchmark for this model carries the
    # *same* footnote text, attach it once at the table foot (no per-bench
    # superscript).  Otherwise, mark each affected bench column.
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
        f"\\caption{{\\textbf{{All baselines, {model_name} "
        f"(higher accuracy is better{cond_suffix}).}} "
        f"Bold marks the best non-oracle per column. "
        f"Subscripts are $\\pm 2\\sigma$ CI.}}")
    lines.append(f"\\label{{tab:wmv_{slug}_all}}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{1.5pt}")
    lines.append("\\renewcommand{\\arraystretch}{0.8}")
    lines.append("\\resizebox{\\textwidth}{!}{%")

    col_spec = "l " + " ".join(["c" * n_tp] * n_b)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row 1: benchmark labels spanning n_tp cols each.
    bench_headers = []
    for b in benchmarks:
        mark = bench_marks.get(b)
        suffix = f"$^{{{mark}}}$" if mark else ""
        label = b.replace("&", "\\&") + suffix
        bench_headers.append(f"\\multicolumn{{{n_tp}}}{{c}}{{{label}}}")
    lines.append(" & " + " & ".join(bench_headers) + " \\\\")

    cmid_parts = []
    col_idx = 2
    for _ in benchmarks:
        cmid_parts.append(
            f"\\cmidrule(lr){{{col_idx}-{col_idx + n_tp - 1}}}")
        col_idx += n_tp
    lines.append(" ".join(cmid_parts))

    # Header row 2: per-benchmark token-point labels.
    tp_hdrs = [f"$B{{=}}${format_tp(tp)}" for tp in token_points]
    lines.append("Method & " + " & ".join(tp_hdrs * n_b) + " \\\\")
    lines.append("\\midrule")

    # Filter out methods with no data for this model, except when the
    # model is a pure placeholder (no benchmarks loaded at all): in that
    # case we keep the core schema visible and only drop the optional
    # ``markers`` row, which is patchy by design and not yet planned for
    # every (model, benchmark) cell.
    is_placeholder = not by_bench
    OPTIONAL_KEYS = {"markers"}
    def _has_any_data(key: str) -> bool:
        for b in benchmarks:
            bdata = by_bench.get(b, {})
            if key in bdata and any(tp in bdata[key] for tp in token_points):
                return True
        return False
    def _include(key: str) -> bool:
        if _has_any_data(key):
            return True
        if is_placeholder and key not in OPTIONAL_KEYS:
            return True
        return False
    # Adaptive stopping is dominated by Standard MV at the same budget; its
    # value lies on the cost axis, so it appears only in cost-savings tables.
    _SKIP_GROUPS = {"Adaptive stopping"}

    rendered_groups: list[tuple[str, list[tuple[str, str]]]] = []
    for group_name, methods in ALL_BASELINE_GROUPS:
        if group_name in _SKIP_GROUPS:
            continue
        members = [(k, d) for k, d in methods if _include(k)]
        if members:
            rendered_groups.append((group_name, members))

    # Best per (bench, tp) column for bolding.  Rounded to 3 decimals so
    # ties at the displayed precision all get bolded.
    method_keys: list[str] = []
    for _, methods in rendered_groups:
        method_keys.extend(k for k, _ in methods)

    best_per_col: dict[tuple[str, int], float] = {}
    for b in benchmarks:
        bdata = by_bench.get(b, {})
        for tp in token_points:
            best = -1.0
            for k in method_keys:
                if k.startswith("oracle"):
                    continue
                if k in bdata and tp in bdata[k]:
                    acc = round(bdata[k][tp][0], 3)
                    if acc > best:
                        best = acc
            best_per_col[(b, tp)] = best

    # Method rows, grouped.
    for gi, (group_name, methods) in enumerate(rendered_groups):
        if gi > 0:
            lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{{1 + n_data_cols}}}{{l}}"
            f"{{\\textit{{{group_name}}}}} \\\\")

        for key, display in methods:
            cells = [f"\\quad {display}"]
            for b in benchmarks:
                bdata = by_bench.get(b, {})
                for tp in token_points:
                    if key in bdata and tp in bdata[key]:
                        acc, ci = bdata[key][tp]
                        is_best = (round(acc, 3) >= best_per_col[(b, tp)]
                                   and best_per_col[(b, tp)] >= 0)
                        cells.append(format_acc(acc, ci, is_best))
                    else:
                        cells.append("---")
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
        description="Per-model all-baselines accuracy table for the appendix")
    parser.add_argument("--condition", default="rm25pct_full_x1")
    parser.add_argument("--token-points", nargs="+", type=int,
                        default=DEFAULT_TOKEN_POINTS)
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
                        help="Display labels of benchmarks (column order)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write per-model .tex files")
    parser.add_argument("--filename-template",
                        default="table_wmv_{slug}_all.tex",
                        help="Output filename template; {slug} = model slug")
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent,
                        help="Base dir for resolving relative jsonl paths")
    parser.add_argument("--suppress-condition", action="store_true",
                        help="Omit `tau, K` from the caption.")
    parser.add_argument("--suppress-subset-footnote", action="store_true",
                        help="Omit per-cell ``N=...`` / problem-subset footnotes.")
    args, remaining = parser.parse_known_args()

    model_groups = _parse_model_args(remaining)
    if not model_groups:
        parser.error("Pass at least one --model NAME [dir ...]")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, dirs in model_groups:
        print(f"\n[{model_name}] {len(dirs)} dirs")
        by_bench, footnotes = load_model_data(
            dirs, args.condition, args.token_points, args.analysis_dir)

        slug = _model_slug(model_name)
        out_path = args.output_dir / args.filename_template.format(slug=slug)

        tex = generate_tex(
            model_name, by_bench, footnotes,
            args.benchmarks, args.token_points, args.condition,
            suppress_condition=args.suppress_condition,
            suppress_subset_footnote=args.suppress_subset_footnote)
        out_path.write_text(tex + "\n")
        print(f"  Written: {out_path}")


if __name__ == "__main__":
    main()
