#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate LaTeX WMV results tables from wmv_result.json files.

Two output modes:

* ``--emit-per-model``  -> one table per model (``tables/extra/``).
* ``--emit-canonical``  -> single multi-model table used as
                           ``table_wmv.tex`` / ``_aux.tex`` in the
                           main paper body.

Usage (per-model extras):
    uv run python paper/table_wmv.py --emit-per-model \\
        --condition rm25pct_full_x1 \\
        gpt-oss-20b_frontierscience_olympiad_jsonl \\
        gpt-oss-20b_hmmt_jsonl \\
        gpt-oss-20b_aime2025_jsonl \\
        -o tables/extra/table_wmv_gpt-oss-20b.tex

Usage (canonical main-body table):
    uv run python paper/table_wmv.py --emit-canonical \\
        --condition rm25pct_full_x1 \\
        --token-points 250000 1000000 5000000 \\
        --label tab:wmv \\
        --merge-base tables/table_wmv.tex \\
        -o tables/table_wmv.tex \\
        --model "GPT-OSS-120B"           gpt-oss-120b_*_jsonl \\
        --model "Nemotron-3-30B"         Nemotron-3-Nano-30B-A3B_*_jsonl \\
        --model "Ministral-3-14B"        Ministral-3-14B_*_jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _utils import (
    MAIN_METHODS, METHOD_DISPLAY, bench_sort_key,
    format_tp, format_acc, load_wmv_methods,
    detect_subset_footnote, parse_condition,
    infer_benchmark_label, infer_model_label,
    load_base_cells_checked, fallback_cell,
)


# ── Merge-base helpers ──

def _strip_row_markers(first_cell: str) -> str:
    """Strip leading ``\\midrule`` / ``\\addlinespace`` and ``\\multirow{..}{*}{...}``
    wrappers to recover the raw group label (e.g. the benchmark name)."""
    import re
    s = first_cell.strip()
    s = re.sub(
        r"^(?:\\(?:midrule|toprule|bottomrule|addlinespace(?:\[[^\]]*\])?"
        r"|cmidrule(?:\([^)]*\))?(?:\{[^}]*\})?)\s*)+",
        "", s)
    m = re.match(r"^\\multirow\{\d+\}\{\*\}\{(.*)\}\s*$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    m = re.match(r"^\\shortstack(?:\[[^\]]*\])?\{(.*)\}\s*$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    # Normalize ``A\\B`` (shortstack line break) to ``A B``.
    s = s.replace("\\\\", " ").strip()
    # Drop trailing footnote marker ``$^{x}$``.
    s = re.sub(r"\s*\$\^\{[^}]*\}\$\s*$", "", s)
    return s


def _wmv_group_key(cells: list[str]):
    """Extract the benchmark label from a multirow group-header first cell."""
    first = cells[0].strip()
    if not first or "\\multirow" not in first:
        return None
    return _strip_row_markers(first)


def _wmv_row_key(cells: list[str], current_group):
    """Row key is ``(benchmark, method)``. Method is cells[1] (stripped);
    data rows always have a non-empty second cell and the row belongs to
    whichever benchmark ``\\multirow`` last opened the current group."""
    if current_group is None or len(cells) < 2:
        return None
    method = cells[1].strip()
    if not method or method.startswith("\\"):
        return None
    return (current_group, method)


def load_wmv_data(jsonl_dir: Path, condition: str | None,
                  methods: list[tuple[str, str]],
                  token_points: list[int],
                  sample_points: list[int] | None = None,
                  ) -> tuple[dict | None, str | None]:
    """Load accuracy values for specified methods and points."""
    json_path = jsonl_dir / "wmv_result.json"
    if not json_path.exists():
        print(f"  WARNING: {json_path} not found, skipping")
        return None, None

    all_data = load_wmv_methods(json_path, condition, token_points)
    if sample_points:
        samp_data = load_wmv_methods(
            json_path, condition, sample_points,
            section="sample_count", point_key="sample_point")
        for key in samp_data:
            all_data.setdefault(key, {}).update(samp_data[key])

    result = {}
    for key, _ in methods:
        result[key] = all_data.get(key, {})

    return result, condition


def _format_col(pt, is_sample: bool) -> str:
    if is_sample:
        return f"$n{{=}}{pt}$"
    return f"$B{{=}}${format_tp(pt)}"


def _generate_subtable(model: str, benchmarks: list[dict],
                       methods: list[tuple[str, str]],
                       columns: list[tuple[int, bool]],
                       is_subtable: bool,
                       base_cells: dict | None = None) -> list[str]:
    """Generate lines for one model's (sub)table.

    *columns* is a list of ``(point_value, is_sample)`` pairs.
    """
    n_cols = len(columns)
    col_headers = [_format_col(pt, is_samp) for pt, is_samp in columns]

    lines = []
    if is_subtable:
        lines.append(f"\\begin{{subtable}}[t]{{\\textwidth}}")
        lines.append(f"\\centering")
        lines.append(f"\\caption{{{model}}}")

    lines.append(f"\\begin{{tabular}}{{ll {'c ' * n_cols}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(["Benchmark", "Method"] + col_headers) + " \\\\")
    lines.append("\\midrule")

    for bi, bench in enumerate(benchmarks):
        if bi > 0:
            lines.append("\\midrule")

        data = bench["data"]
        n_methods = len(methods)
        bench_label = bench["label"].replace("&", "\\&")
        fn = bench.get("footnote_mark")
        if fn:
            bench_label += fn

        # Find best acc per column (bolding uses rounded values so that
        # ties at the displayed precision are all bolded)
        best_rounded = {}
        for pt, _ in columns:
            best = -1.0
            for key, _ in methods:
                if data and key in data and pt in data[key]:
                    acc = round(data[key][pt][0], 3)
                    if acc > best:
                        best = acc
            best_rounded[pt] = best

        bench_key = bench["label"]
        for mi, (key, display_name) in enumerate(methods):
            if mi == 0:
                row_label = (f"\\multirow{{{n_methods}}}{{*}}"
                             f"{{\\shortstack[l]{{{bench_label}}}}}")
            else:
                row_label = ""

            cells = [row_label, display_name]
            row_key = (bench_key, display_name)
            col_idx = 2  # data cells start at col 2 (cells[0]=group, cells[1]=method)
            for pt, _ in columns:
                if data and key in data and pt in data[key]:
                    acc, ci = data[key][pt]
                    is_best = round(acc, 3) >= best_rounded[pt]
                    cell = format_acc(acc, ci, is_best)
                else:
                    cell = "---"
                cell = fallback_cell(cell, (row_key, col_idx), base_cells)
                cells.append(cell)
                col_idx += 1

            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if is_subtable:
        lines.append("\\end{subtable}")

    return lines


def generate_tex(benchmarks: list[dict], methods: list[tuple[str, str]],
                 columns: list[tuple[int, bool]],
                 condition_used: str | None = None,
                 base_cells: dict | None = None,
                 suppress_condition: bool = False,
                 suppress_subset_footnote: bool = False) -> str:
    """Generate one table per model, concatenated into a single string."""
    # Build footnote map across all benchmarks
    footnote_map: dict[str, str] = {}
    for b in benchmarks:
        fn = None if suppress_subset_footnote else b.get("footnote")
        if fn and fn not in footnote_map:
            footnote_map[fn] = chr(ord("a") + len(footnote_map))
        b["footnote_mark"] = (f"$^{{{footnote_map[fn]}}}$"
                              if fn else "")

    # Group benchmarks by model
    by_model = defaultdict(list)
    model_order = []
    for b in benchmarks:
        m = b["model"] or "Unknown"
        if m not in by_model:
            model_order.append(m)
        by_model[m].append(b)

    tau, K = parse_condition(condition_used)
    if tau is not None and K is not None:
        cond_str = f"$\\tau{{=}}{tau}$, $K{{=}}{K}$"
    elif condition_used:
        cond_str = condition_used.replace("_", "\\_")
    else:
        cond_str = "default condition"
    cond_in_caption = "" if suppress_condition else f" ({cond_str})"

    has_samples = any(is_s for _, is_s in columns)
    axis_desc = "sample counts" if has_samples else "token budgets"

    tables = []
    for model in model_order:
        model_benchmarks = by_model[model]
        model_slug = model.replace(" ", "_").replace("(", "").replace(")", "")

        # Collect footnotes used in this model's benchmarks
        model_fns = {b.get("footnote") for b in model_benchmarks
                     if b.get("footnote")}

        lines = []
        lines.append("\\begin{table}")
        lines.append("\\centering")
        lines.append(
            f"\\caption{{Weighted majority voting accuracy for "
            f"{model}{cond_in_caption}.}}")
        lines.append(f"\\label{{tab:wmv_{model_slug}}}")
        lines.append("\\footnotesize")
        lines.append("\\setlength{\\tabcolsep}{4pt}")
        lines.extend(_generate_subtable(
            model, model_benchmarks, methods, columns,
            is_subtable=False, base_cells=base_cells))

        # Footnotes for this model
        if model_fns:
            notes = " ".join(
                f"$^{{{footnote_map[fn]}}}${fn}."
                for fn in sorted(model_fns,
                                 key=lambda x: footnote_map.get(x, "z"))
                if fn in footnote_map
            )
            lines.append(f"\\par\\vspace{{2pt}}")
            lines.append(
                f"\\parbox{{\\textwidth}}{{\\raggedright\\footnotesize "
                f"{notes}}}")

        lines.append("\\end{table}")
        tables.append("\n".join(lines))

    return "\n\n".join(tables)


# ── Canonical multi-model table (main/aux body) ──

def parse_model_args(raw_args: list[str]) -> list[tuple[str, list[Path]]]:
    """Parse ``--model NAME dir [dir ...] --model NAME dir ...`` into groups.

    Mirrors the same pattern used by ``table_token_savings.py`` so the
    two canonical generators take identical CLI shape.
    """
    groups: list[tuple[str, list[Path]]] = []
    current_name: str | None = None
    current_dirs: list[Path] = []
    for arg in raw_args:
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


def _load_model_group(
    dirs: list[Path],
    condition: str,
    methods: list[tuple[str, str]],
    token_points: list[int],
) -> list[dict]:
    """Load per-benchmark data for one model and order via BENCHMARK_ORDER."""
    benches = []
    for d in dirs:
        data, _ = load_wmv_data(d, condition, methods, token_points, None)
        if data is None:
            continue
        label = infer_benchmark_label(d.name) or d.name
        wmv_path = d / "wmv_result.json"
        footnote = (detect_subset_footnote(wmv_path, d.name)
                    if wmv_path.exists() else None)
        benches.append({
            "label": label, "data": data, "footnote": footnote,
            "dir_name": d.name,
        })

    benches.sort(key=lambda b: bench_sort_key(b["label"]))
    return benches


def generate_canonical_tex(
    model_groups: list[tuple[str, list[dict]]],
    methods: list[tuple[str, str]],
    token_points: list[int],
    condition_used: str | None,
    label: str,
    base_cells: dict | None = None,
    suppress_condition: bool = False,
    suppress_subset_footnote: bool = False,
) -> str:
    """Multi-model canonical layout (``table_wmv.tex``-style).

    Rows: one per (benchmark, method).  Columns: one cell per
    (model, token_point).  Benchmarks from each model are unified by
    display label; cells absent for a given model are ``---`` (or
    filled from ``--merge-base``).
    """
    model_names = [name for name, _ in model_groups]

    # Union of benchmarks across models, ordered canonically.
    bench_set: set[str] = set()
    for _, benches in model_groups:
        for b in benches:
            bench_set.add(b["label"])
    bench_order: list[str] = sorted(bench_set, key=bench_sort_key)

    # Footnote bookkeeping.  Each unique footnote *text* (e.g. "$N$=64")
    # gets one letter; the scope string names the exact (model[, bench])
    # pairs it applies to so the reader knows which column a mark refers
    # to.  Collapses to ``"Model"`` when all benches of that model share
    # the same footnote text.
    per_pair_text: dict[tuple[str, str], str] = {}
    model_bench_total: dict[str, int] = {}
    for mname, benches in model_groups:
        model_bench_total[mname] = len(benches)
        if suppress_subset_footnote:
            continue
        for b in benches:
            fn = b.get("footnote")
            if fn:
                per_pair_text[(mname, b["label"])] = fn

    from collections import defaultdict as _dd
    text_to_pairs: dict[str, list[tuple[str, str]]] = _dd(list)
    for pair, text in per_pair_text.items():
        text_to_pairs[text].append(pair)

    cell_mark: dict[tuple[str, str], str] = {}
    legend_entries: list[tuple[str, str, str]] = []
    for idx, (text, pairs) in enumerate(text_to_pairs.items()):
        mark = chr(ord("a") + idx)
        by_model: dict[str, list[str]] = _dd(list)
        for m, b in pairs:
            by_model[m].append(b)
        scope_parts = []
        for m in model_names:
            if m not in by_model:
                continue
            bs = by_model[m]
            if len(bs) == model_bench_total.get(m, 0):
                scope_parts.append(m)
            else:
                for b in bs:
                    scope_parts.append(f"{m} $\\times$ {b}")
        legend_entries.append((mark, ", ".join(scope_parts), text))
        for pair in pairs:
            cell_mark[pair] = mark

    # Data lookup: (model, benchmark) -> {method: {tp: (acc, ci)}}
    data_map: dict[tuple[str, str], dict] = {}
    for mname, benches in model_groups:
        for b in benches:
            data_map[(mname, b["label"])] = b["data"]

    tau, K = parse_condition(condition_used)
    if tau is not None and K is not None:
        cond_str = f"$\\tau{{=}}{tau}$, $K{{=}}{K}$"
    else:
        cond_str = (condition_used or "default condition"
                    ).replace("_", "\\_")

    n_tp = len(token_points)
    n_models = len(model_names)
    tp_headers = [format_tp(tp) for tp in token_points]

    # Column spec: 2 label cols + (n_tp per model) × n_models
    col_spec = "ll " + " ".join(["c" * n_tp] * n_models)

    lines: list[str] = []
    lines.append("\\begin{table}")
    lines.append("\\centering")
    # Wrap the "k"/"M" suffix in \text{} so it renders upright inside math.
    def _tp_math(h: str) -> str:
        for suf in ("k", "M"):
            if h.endswith(suf):
                return h[:-1] + f"\\text{{{suf}}}"
        return h
    tp_list_str = ", ".join(_tp_math(h) for h in tp_headers)
    cond_suffix = "" if suppress_condition else f", {cond_str}"
    lines.append(
        f"\\caption{{\\textbf{{Weighted majority voting accuracy at fixed "
        f"token budget $B$ (higher is better{cond_suffix}).}} Each method's "
        f"accuracy at $B \\in \\{{{tp_list_str}\\}}$ tokens sampled from the "
        f"shared pool, with the best per (model, $B$) column in bold.}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{2.2pt}")
    lines.append("\\renewcommand{\\arraystretch}{0.95}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header_multicol = " & ".join(
        [f"\\multicolumn{{{n_tp}}}{{c}}{{{m}}}" for m in model_names])
    lines.append(f"& & {header_multicol} \\\\")
    cmid_parts = []
    for i in range(n_models):
        start = 3 + i * n_tp
        end = start + n_tp - 1
        cmid_parts.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmid_parts))
    tp_col_headers = [f"$B{{=}}${h}" for h in tp_headers]
    col_hdrs = " & ".join(tp_col_headers * n_models)
    lines.append(f"Benchmark & Method & {col_hdrs} \\\\")
    lines.append("\\midrule")

    for bi, bench in enumerate(bench_order):
        if bi > 0:
            lines.append("\\midrule")

        # Per-model best-cell lookup for bolding.
        best_rounded: dict[tuple[str, int], float] = {}
        for mname in model_names:
            data = data_map.get((mname, bench))
            for pt in token_points:
                best = -1.0
                if data:
                    for key, _ in methods:
                        if key in data and pt in data[key]:
                            acc = round(data[key][pt][0], 3)
                            if acc > best:
                                best = acc
                best_rounded[(mname, pt)] = best

        # Benchmark row label with footnote markers for any (model, bench)
        # pair that carries a subset footnote.
        marks = []
        for mname in model_names:
            m = cell_mark.get((mname, bench))
            if m and m not in marks:
                marks.append(m)
        marks.sort()
        fn_suffix = ("$^{" + ",".join(marks) + "}$") if marks else ""
        bench_label_tex = bench.replace("&", "\\&") + fn_suffix
        row_label_full = (f"\\multirow{{{len(methods)}}}{{*}}"
                           f"{{\\shortstack[l]{{{bench_label_tex}}}}}")

        for mi, (key, display_name) in enumerate(methods):
            cells = [row_label_full if mi == 0 else "", display_name]
            row_key = (bench, display_name)
            col_idx = 2
            for mname in model_names:
                data = data_map.get((mname, bench))
                # Gate merge-base fallback on whether this (model, bench)
                # has any local data at all.  Without this check, cells
                # for a (model, bench) pair we did not run leak stale
                # values from the old hand-maintained tex — and method
                # renames (e.g. ``DeepConf (tail)`` -> ``DeepConf tail``)
                # fail to match, yielding partial rows.
                has_local = data is not None
                for pt in token_points:
                    if data and key in data and pt in data[key]:
                        acc, ci = data[key][pt]
                        is_best = (round(acc, 3) >= best_rounded[(mname, pt)]
                                   and best_rounded[(mname, pt)] >= 0)
                        cell = format_acc(acc, ci, is_best)
                    else:
                        cell = "---"
                    if has_local:
                        cell = fallback_cell(
                            cell, (row_key, col_idx), base_cells)
                    cells.append(cell)
                    col_idx += 1
            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}% end resizebox")
    if legend_entries:
        notes = " ".join(
            f"$^{{{m}}}${scope}: {text}."
            for m, scope, text in legend_entries)
        lines.append("\\vspace{2pt}")
        lines.append(f"\\par\\raggedright\\footnotesize {notes}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _run_canonical(args, methods, remaining):
    model_groups_raw = parse_model_args(remaining)
    if not model_groups_raw:
        raise SystemExit(
            "--emit-canonical requires at least one --model NAME dir [dir ...]")

    model_groups: list[tuple[str, list[dict]]] = []
    for name, dirs in model_groups_raw:
        print(f"[{name}] {len(dirs)} dirs")
        benches = _load_model_group(dirs, args.condition, methods,
                                     args.token_points)
        for b in benches:
            print(f"  {b['dir_name']} -> {b['label']}")
        model_groups.append((name, benches))

    base_cells = None
    if args.merge_base:
        n_models = len(model_groups)
        expected_cols = 1 + n_models * len(args.token_points)
        base_cells = load_base_cells_checked(
            args.merge_base,
            _wmv_row_key,
            expected_cols=expected_cols,
            label="merge-base",
            group_key_fn=_wmv_group_key,
        )

    tex = generate_canonical_tex(
        model_groups, methods, args.token_points,
        condition_used=args.condition,
        label=args.label,
        base_cells=base_cells,
        suppress_condition=args.suppress_condition,
        suppress_subset_footnote=args.suppress_subset_footnote,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex + "\n")
        print(f"\nWritten: {args.output}")
    else:
        print()
        print(tex)


def main():
    import sys
    parser = argparse.ArgumentParser(
        description="Generate LaTeX WMV results tables")
    parser.add_argument("--condition", required=True,
                        help="Condition label (e.g. rm50pct_full_x3)")
    parser.add_argument("--token-points", nargs="+", type=int,
                        default=[25_000, 250_000, 1_000_000],
                        help="Token budget points for columns")
    parser.add_argument("--sample-points", nargs="+", type=int,
                        default=None,
                        help="Sample count points for additional columns "
                             "(per-model mode only)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Method keys to include (default: standard set)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output .tex file (default: stdout)")
    parser.add_argument("--emit-per-model", action="store_true",
                        help="Emit the per-model extras (tables/extra/"
                             "table_wmv_<model>.tex).")
    parser.add_argument("--emit-canonical", action="store_true",
                        help="Emit the canonical multi-model table "
                             "(main paper body: table_wmv.tex / "
                             "_aux.tex). Takes directories via "
                             "--model NAME dir [dir ...].")
    parser.add_argument("--label", default="tab:wmv_canonical",
                        help="LaTeX label for canonical output.")
    parser.add_argument("--merge-base", type=Path, default=None,
                        help="Existing output .tex; cells this run cannot "
                             "compute locally fall back to its values.")
    parser.add_argument("--suppress-condition", action="store_true",
                        help="Omit `tau, K` from the caption.")
    parser.add_argument("--suppress-subset-footnote", action="store_true",
                        help="Omit per-cell ``N=...`` / problem-subset footnotes.")
    args, remaining = parser.parse_known_args()

    if args.emit_canonical == args.emit_per_model:
        print("Pass exactly one of --emit-canonical or --emit-per-model.")
        return

    methods = list(MAIN_METHODS)
    if args.methods:
        methods = [(m, METHOD_DISPLAY.get(m, m)) for m in args.methods]

    if args.emit_canonical:
        _run_canonical(args, methods, remaining)
        return

    # Per-model mode: remaining args are jsonl dir paths.
    jsonl_dirs = [Path(a) for a in remaining if not a.startswith("-")]
    if not jsonl_dirs:
        parser.error("--emit-per-model requires one or more JSONL directories "
                      "as positional arguments.")
    args.jsonl_dirs = jsonl_dirs

    benchmarks = []
    condition_used = None
    for d in args.jsonl_dirs:
        dir_name = d.name
        label = infer_benchmark_label(dir_name)
        model = infer_model_label(dir_name)
        print(f"Loading {dir_name} -> {label} ({model})")
        data, cond = load_wmv_data(
            d, args.condition, methods,
            args.token_points, args.sample_points)
        if condition_used is None and cond:
            condition_used = cond
        wmv_path = d / "wmv_result.json"
        footnote = (detect_subset_footnote(wmv_path, dir_name)
                    if wmv_path.exists() else None)
        benchmarks.append({
            "dir_name": dir_name,
            "label": label,
            "model": model,
            "data": data,
            "footnote": footnote,
        })

    # Load merge-base cells if requested. Column layout in the base must
    # match the (token-points + sample-points) schema the current run
    # produces; otherwise positional merge would bind values to the
    # wrong columns and we skip it with a warning.
    base_cells = None
    if args.merge_base:
        expected_cols = 1 + len(args.token_points)
        if args.sample_points:
            expected_cols += len(args.sample_points)
        base_cells = load_base_cells_checked(
            args.merge_base,
            _wmv_row_key,
            expected_cols=expected_cols,
            label="merge-base",
            group_key_fn=_wmv_group_key,
        )

    # Generate separate tables for token budget and sample count.
    tables = []
    tok_cols = [(tp, False) for tp in args.token_points]
    tables.append(generate_tex(
        benchmarks, methods, tok_cols, condition_used,
        base_cells=base_cells,
        suppress_condition=args.suppress_condition,
        suppress_subset_footnote=args.suppress_subset_footnote))
    if args.sample_points:
        samp_cols = [(sp, True) for sp in args.sample_points]
        tables.append(generate_tex(
            benchmarks, methods, samp_cols, condition_used,
            base_cells=base_cells,
            suppress_condition=args.suppress_condition,
            suppress_subset_footnote=args.suppress_subset_footnote))
    tex = "\n\n".join(tables)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex)
        print(f"Written: {args.output}")
    else:
        print()
        print(tex)


if __name__ == "__main__":
    main()
