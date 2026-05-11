#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Generate the LaTeX token-efficiency table (Table~\\ref{tab:token_savings}).

For each threshold ``alpha in {0.75, 0.90, 0.99}`` we report
``B_method / B_MV`` at target accuracy ``Pass@1 + alpha * (plateau - Pass@1)``,
with plateau = MV's stored ``acc`` at the largest evaluated budget
($B_{\\max}$=10^7) and Pass@1 = MV's stored ``acc`` at sample_count=1.
Ratios < 1 mean the method is more cost-efficient than Standard MV at
that target.  See Appendix~F of the paper for the plateau definition
and the parametric trial-MC bootstrap that produces the 2σ CIs.

Usage:
    cd analysis && uv run python paper/table_token_savings.py \\
        --condition rm25pct_full_x1 \\
        --model GPT-OSS-120B <dir-containing-wmv_result.json> ...
"""

import argparse
import json
import math
import sys
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np

from _utils import (
    MAIN_METHODS, bench_sort_key,
    load_wmv_methods,
    load_natural_stopping_curves,
    get_pass_at_1,
    detect_subset_footnote, parse_condition,
    infer_benchmark_label, infer_model_label,
    load_base_cells_checked, fallback_cell,
)
from _defs import METHOD_DISPLAY, ALL_BASELINE_GROUPS


# ── Merge-base helpers ──

def _strip_row_markers(first_cell: str) -> str:
    import re
    s = first_cell.strip()
    s = re.sub(r"^(?:\\(?:midrule|toprule|bottomrule|addlinespace(?:\[[^\]]*\])?"
               r"|cmidrule(?:\([^)]*\))?(?:\{[^}]*\})?))\s*",
               "", s)
    m = re.match(r"^\\multirow\{\d+\}\{\*\}\{(.*)\}\s*$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    m = re.match(r"^\\shortstack(?:\[[^\]]*\])?\{(.*)\}\s*$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    s = s.replace("\\\\", " ").strip()
    import re as _re
    s = _re.sub(r"\s*\$\^\{[^}]*\}\$\s*$", "", s)
    return s


def _savings_group_key(cells):
    first = cells[0].strip()
    if not first or "\\multirow" not in first:
        return None
    return _strip_row_markers(first)


def _savings_row_key(cells, current_group):
    """Row key is ``(benchmark, row_label)``. row_label is cells[1]
    (``Pass@1 / Standard MV plateau``, ``Standard MV budget``, or a method name)."""
    if current_group is None or len(cells) < 2:
        return None
    label = cells[1].strip()
    if not label or label.startswith("\\"):
        return None
    return (current_group, label)


ALL_TOKEN_POINTS = [
    1_000, 2_500, 5_000, 10_000, 25_000, 50_000,
    100_000, 250_000, 500_000, 1_000_000,
    2_500_000, 5_000_000, 10_000_000,
]

# wmv.py writes a dense-grid curve under this section when run with
# incremental voters (default).  We prefer it when available.
DENSE_SECTION = "token_budget_dense"


def _interpolate_budget_loglog(entries: list[tuple[int, float]],
                                target_acc: float) -> float | None:
    """First budget where acc crosses target, linear in log(tp).

    *entries* is expected to be already monotone non-decreasing in acc
    (pass it through ``_monotone_envelope`` first if not).  Returns
    None if the curve never reaches target_acc.
    """
    if not entries:
        return None
    for i in range(len(entries)):
        tp, acc = entries[i]
        if acc >= target_acc:
            if i == 0:
                return float(tp)
            tp_prev, acc_prev = entries[i - 1]
            if acc == acc_prev:
                return float(tp_prev)
            frac = (target_acc - acc_prev) / (acc - acc_prev)
            log_tp = (math.log(tp_prev)
                      + frac * (math.log(tp) - math.log(tp_prev)))
            return math.exp(log_tp)
    return None


def _monotone_envelope(
    curve: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Return the running-max (monotone non-decreasing) envelope of *curve*.

    ``_interpolate_budget_loglog`` looks for the first budget at which
    acc crosses *target*.  With raw noisy curves the first crossing can
    flicker due to sampling noise near plateau.  Applying the running
    max implements the ``min budget at which the method has ever
    reached this accuracy'' semantics that ``B_M^{-1}`` really wants.
    """
    out = []
    best = -1.0
    for tp, acc in curve:
        if acc > best:
            best = acc
        out.append((tp, best))
    return out


def _extract_section_points(json_path: Path, section: str) -> list[int]:
    """Return the union of token_points across methods in *section* of
    ``wmv_result.json``.  Empty list if section missing or empty."""
    with open(json_path) as f:
        data = json.load(f)
    sec = data.get(section, {})
    if not sec:
        return []
    pts: set[int] = set()
    for group in ("shared_methods", "conditions", "per_regen_count"):
        blob = sec.get(group, {})
        if group == "shared_methods":
            methods = blob
        else:
            methods = {}
            for child in blob.values():
                methods.update(child.get("methods", child))
        for m in methods.values():
            for e in m.get("entries", []):
                tp = e.get("token_point")
                if tp is not None:
                    pts.add(int(tp))
    return sorted(pts)


N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
BOOTSTRAP_MIN_SUCCESS = 0.5  # below this fraction of successful draws, drop CI


def _load_mv_plateau(wmv_path: Path
                     ) -> tuple[float | None, float | None]:
    """Return ``(acc, ci)`` for MV at the largest dense budget."""
    with open(wmv_path) as f:
        data = json.load(f)
    entries = (data.get(DENSE_SECTION, {})
                   .get("shared_methods", {})
                   .get("standard_mv", {}).get("entries", []))
    if not entries:
        return None, None
    last = entries[-1]
    return (float(last["acc"]), float(last.get("ci", 0.0)))


def _eval_crossings(
    curves_acc: dict[str, list[tuple[int, float]]],
    thresholds: list[float],
    plateau: float,
    pass_at_1: float,
) -> tuple[dict[float, float | None],
           dict[str, dict[float, float | None]]]:
    """Compute ``(mv_budgets, ratios)`` at each threshold.

    Methods that do not reach the target on their monotone envelope
    yield ``ratio=None``.
    """
    gap = plateau - pass_at_1
    degenerate = gap <= 0

    mv_curve = curves_acc.get("standard_mv", [])
    mv_mono = _monotone_envelope(mv_curve) if mv_curve else []

    mv_budgets: dict[float, float | None] = {}
    targets: dict[float, float | None] = {}
    for thr in thresholds:
        if degenerate or not mv_mono:
            mv_budgets[thr] = None
            targets[thr] = None
        else:
            t = pass_at_1 + thr * gap
            targets[thr] = t
            mv_budgets[thr] = _interpolate_budget_loglog(mv_mono, t)

    ratios: dict[str, dict[float, float | None]] = {}
    for key, entries in curves_acc.items():
        mono = _monotone_envelope(entries)
        ratios[key] = {}
        for thr in thresholds:
            mvb = mv_budgets.get(thr)
            t = targets.get(thr)
            if mvb is None or t is None:
                ratios[key][thr] = None
                continue
            mb = _interpolate_budget_loglog(mono, t)
            ratios[key][thr] = (mb / mvb) if mb is not None else None

    return mv_budgets, ratios


def _curves_from_aggregate(
    aggregate: dict[str, list[tuple]],
    natural_rng: np.random.Generator | None = None,
    tb_rng: np.random.Generator | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Strip ``ci``/``cost_ci`` from aggregate tuples, optionally adding
    Gaussian perturbation: acc by ``N(0, (ci/2)^2)`` and (for 4-tuples)
    cost by ``N(0, (cost_ci/2)^2)``. RNG is split by tuple shape so each
    family's bootstrap stream is independent.
    """
    curves: dict[str, list[tuple[float, float]]] = {}
    for k, pts in aggregate.items():
        new_pts: list[tuple[float, float]] = []
        for pt in pts:
            tp = float(pt[0])
            acc = float(pt[1])
            ci = float(pt[2]) if len(pt) >= 3 else 0.0
            is_natural = len(pt) >= 4
            rng = natural_rng if is_natural else tb_rng
            if rng is not None and ci > 0:
                acc += float(rng.normal(0.0, ci / 2.0))
            if is_natural:
                cost_ci = float(pt[3])
                if rng is not None and cost_ci > 0:
                    tp += float(rng.normal(0.0, cost_ci / 2.0))
                    tp = max(tp, 1.0)  # log-budget interpolation needs tp > 0
            new_pts.append((tp, acc))
        curves[k] = new_pts
    return curves


def _summarise_bootstrap(
    boot: dict[str, dict[float, list[float]]],
    ratios_full: dict[str, dict[float, float | None]],
    n_bootstrap: int,
) -> dict[str, dict[float, tuple | None]]:
    """Reduce per-cell bootstrap draws to ``(point, sigma)`` tuples.

    ``sigma`` is ``None`` when fewer than ``BOOTSTRAP_MIN_SUCCESS`` of
    the draws (or fewer than 50) reach the target.
    """
    out: dict[str, dict[float, tuple | None]] = {}
    for k in boot:
        out[k] = {}
        for thr in boot[k]:
            full = ratios_full.get(k, {}).get(thr)
            if full is None:
                out[k][thr] = None
                continue
            vs = boot[k][thr]
            success = len(vs) / max(n_bootstrap, 1)
            if len(vs) >= 50 and success >= BOOTSTRAP_MIN_SUCCESS:
                sigma = 2.0 * float(np.std(vs, ddof=1))
            else:
                sigma = None
            out[k][thr] = (full, sigma)
    return out


def compute_plateau_savings(
    wmv_path: Path,
    condition: str,
    method_keys: list[str],
    thresholds: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float | None, float | None,
           dict[float, float | None],
           dict[str, dict[float, tuple | None]]]:
    """Return ``(plateau, pass1, mv_budgets, ratios)``.

    Parametric trial-MC bootstrap over the stored CIs: each replicate
    perturbs every operating point's accuracy by ``N(0, (ci/2)^2)``
    (and natural-stopping cost by ``N(0, (cost_ci/2)^2)``), perturbs
    the plateau by its own CI, and recomputes the ratio. Pass@1 is
    closed-form over the init pool (held fixed across draws).
    """
    if not wmv_path.exists():
        raise FileNotFoundError(f"wmv_result.json not found at {wmv_path}")

    plateau_pt, plateau_ci = _load_mv_plateau(wmv_path)
    if plateau_pt is None or plateau_ci is None:
        return None, None, {}, {}
    p1_pt = get_pass_at_1(wmv_path.parent)

    if plateau_pt - p1_pt <= 0:
        warnings.warn(
            f"Pass@1 is at or above the Standard MV plateau for "
            f"{wmv_path} (gap={plateau_pt - p1_pt:.4f}); all token-savings "
            "ratios will be N/A.",
            stacklevel=2,
        )

    # natural-stopping: 4-tuples (cost, acc, ci, cost_ci);
    # fixed-budget: 3-tuples (tp, acc, ci).
    dense_pts = _extract_section_points(wmv_path, DENSE_SECTION)
    dense_data = (load_wmv_methods(wmv_path, condition, dense_pts,
                                    section=DENSE_SECTION)
                  if dense_pts else {})
    legacy_data = load_wmv_methods(wmv_path, condition, ALL_TOKEN_POINTS)
    natural_curves = load_natural_stopping_curves(wmv_path)
    agg: dict[str, list[tuple]] = {}
    for key in method_keys:
        if key in natural_curves and natural_curves[key]:
            agg[key] = list(natural_curves[key])
            continue
        pts_dict = dense_data.get(key) or legacy_data.get(key) or {}
        if not pts_dict:
            continue
        agg[key] = [(tp, acc, ci)
                    for tp, (acc, ci) in sorted(pts_dict.items())]
    if "standard_mv" not in agg:
        return None, None, {}, {}

    # --- Point estimate ---
    curves_full = _curves_from_aggregate(agg)
    mv_b_full, ratios_full = _eval_crossings(
        curves_full, thresholds, plateau_pt, p1_pt)

    # Per-family seed offsets, mirroring wmv/eval.py, so each family's
    # CI is stable against method-set size and unrelated perturbations.
    plateau_rng = np.random.default_rng(seed)
    natural_rng = np.random.default_rng(seed + 1)
    tb_rng = np.random.default_rng(seed + 2)
    boot: dict[str, dict[float, list[float]]] = {
        k: {thr: [] for thr in thresholds} for k in agg}

    plateau_sd = plateau_ci / 2.0
    for _ in range(n_bootstrap):
        plateau_b = plateau_pt + float(plateau_rng.normal(0.0, plateau_sd)) \
            if plateau_sd > 0 else plateau_pt
        curves_b = _curves_from_aggregate(
            agg, natural_rng=natural_rng, tb_rng=tb_rng)
        _, ratios_b = _eval_crossings(
            curves_b, thresholds, plateau_b, p1_pt)
        for k in boot:
            rk = ratios_b.get(k, {})
            for thr in thresholds:
                v = rk.get(thr)
                if v is not None and math.isfinite(v):
                    boot[k][thr].append(v)

    ratios_out = _summarise_bootstrap(boot, ratios_full, n_bootstrap)
    return plateau_pt, p1_pt, mv_b_full, ratios_out


RATIO_HIGH_CAP = 10.0
RATIO_LOW_CAP = 0.01


def _format_ratio(entry, is_best: bool) -> str:
    """Format ``(point, sigma)`` or None as a LaTeX cell.

    ``sigma`` may be None when the bootstrap is unreliable, in which
    case the cell shows only the point estimate (no subscript).
    """
    if entry is None:
        return "N/A"
    ratio, sigma = entry
    if ratio > RATIO_HIGH_CAP:
        return f"$>${RATIO_HIGH_CAP:.0f}$\\times$"
    if ratio < RATIO_LOW_CAP:
        return f"$<${RATIO_LOW_CAP:.2f}$\\times$"
    ratio_s = f"{ratio:.2f}"
    if sigma is None:
        s = f"{ratio_s}$\\times$"
    else:
        sigma_s = f"{sigma:.2f}"
        s = f"{ratio_s}$_{{\\pm{sigma_s}}}$$\\times$"
    if is_best:
        return f"\\textbf{{{s}}}"
    return s


def _format_acc(acc: float | None) -> str:
    if acc is None:
        return "---"
    if acc >= 0.9995:
        return "1.000"
    return f"{acc:.3f}"[1:]


def _format_budget(tokens: float | None) -> str:
    if tokens is None:
        return "---"
    if tokens >= 1_000_000:
        return f"{tokens/1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens/1_000:.0f}k"
    return str(int(round(tokens)))




def _format_threshold(thr: float) -> str:
    # 0.90 -> "90\%"
    pct = thr * 100
    if abs(pct - round(pct)) < 1e-6:
        return f"{int(round(pct))}\\%"
    return f"{pct:.1f}\\%"


def generate_tex(
    model_order: list[str],
    model_benchmarks: dict[str, dict[str, dict]],
    methods: list[tuple[str, str]],
    thresholds: list[float],
    condition_used: str | None,
    label: str = "tab:token_savings",
    base_cells: dict | None = None,
    suppress_condition: bool = False,
    suppress_subset_footnote: bool = False,
) -> str:
    tau, K = parse_condition(condition_used)
    if tau is not None and K is not None:
        cond_str = f"$\\tau{{=}}{tau}$, $K{{=}}{K}$"
    else:
        cond_str = condition_used or "default"

    n_thr = len(thresholds)
    n_models = len(model_order)
    thr_headers = [_format_threshold(t) for t in thresholds]

    all_benchmarks = OrderedDict()
    for model in model_order:
        for bench_label in model_benchmarks[model]:
            if bench_label not in all_benchmarks:
                all_benchmarks[bench_label] = None
    bench_order = sorted(all_benchmarks.keys(), key=bench_sort_key)

    # Footnote bookkeeping.  Each unique footnote *text* gets one letter;
    # the scope string names the exact (model[, bench]) pairs so the
    # reader can tell which column the mark refers to.
    from collections import defaultdict as _dd
    per_pair_text: dict[tuple[str, str], str] = {}
    model_bench_total: dict[str, int] = {}
    for model in model_order:
        model_bench_total[model] = len(model_benchmarks[model])
        if suppress_subset_footnote:
            continue
        for bl, bdata in model_benchmarks[model].items():
            fn = bdata.get("footnote")
            if fn:
                per_pair_text[(model, bl)] = fn
    text_to_pairs: dict[str, list[tuple[str, str]]] = _dd(list)
    for pair, text in per_pair_text.items():
        text_to_pairs[text].append(pair)
    footnote_map: dict[str, str] = {}
    cell_mark: dict[tuple[str, str], str] = {}
    legend_entries: list[tuple[str, str, str]] = []
    for idx, (text, pairs) in enumerate(text_to_pairs.items()):
        mark = chr(ord("a") + idx)
        footnote_map[text] = mark
        by_model: dict[str, list[str]] = _dd(list)
        for m, b in pairs:
            by_model[m].append(b)
        scope_parts = []
        for m in model_order:
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

    col_spec = "ll " + " ".join(["c" * n_thr] * n_models)

    lines = []
    lines.append("\\begin{table}")
    lines.append("\\centering")
    thr_list_str = ", ".join(thr_headers)
    cond_suffix = "" if suppress_condition else f", {cond_str}"
    lines.append(
        f"\\caption{{\\textbf{{Token efficiency ratio "
        f"$B_{{\\mathrm{{method}}}} / B_{{\\mathrm{{MV}}}}$ at target "
        f"accuracy $\\alpha$ between Pass@1 and the Standard MV plateau "
        f"(smaller is better{cond_suffix}).}} "
        f"$B_X$ is the budget method $X$ needs to reach Pass@1 $+\\, "
        f"\\alpha \\times$ (Standard MV plateau $-$ Pass@1) for "
        f"$\\alpha \\in \\{{{thr_list_str}\\}}$. "
        f"Cells $<$1 indicate more cost-efficient than Standard MV. "
        f"``N/A'' indicates the method's plateau is below the target. "
        f"Best method per column in bold.}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{2.2pt}")
    lines.append("\\renewcommand{\\arraystretch}{0.95}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    model_headers = []
    cmidrules = []
    col_idx = 3
    for model in model_order:
        end = col_idx + n_thr - 1
        model_headers.append(
            f"\\multicolumn{{{n_thr}}}{{c}}{{{model}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{end}}}")
        col_idx = end + 1
    lines.append("& & " + " & ".join(model_headers) + " \\\\")
    lines.append(" ".join(cmidrules))

    alpha_col_headers = [f"$\\alpha{{=}}{h}$" for h in thr_headers]
    sub_headers: list[str] = []
    for _ in model_order:
        sub_headers.extend(alpha_col_headers)
    lines.append(
        "Benchmark & Method & " + " & ".join(sub_headers) + " \\\\")
    lines.append("\\midrule")

    ratio_methods = [(k, d) for k, d in methods if k != "standard_mv"]

    # Drop methods with no data across every (benchmark, model).
    def _has_any_data(key: str) -> bool:
        for model in model_order:
            for bl in model_benchmarks[model]:
                bdata = model_benchmarks[model][bl]
                ratios = bdata.get("ratios", {})
                if key in ratios:
                    return True
        return False

    ratio_methods = [(k, d) for k, d in ratio_methods if _has_any_data(k)]

    # Rows per benchmark: 1 (Pass@1 / MV plat.) + 1 (MV bud.) + methods.
    n_rows_per_bench = 2 + len(ratio_methods)

    for bi, bench_label in enumerate(bench_order):
        if bi > 0:
            lines.append("\\midrule")

        display_label = bench_label.replace("&", "\\&")
        bench_marks = []
        for model in model_order:
            m = cell_mark.get((model, bench_label))
            if m and m not in bench_marks:
                bench_marks.append(m)
        bench_marks.sort()
        if bench_marks:
            display_label += "$^{" + ",".join(bench_marks) + "}$"

        row_label = (f"\\multirow{{{n_rows_per_bench}}}{{*}}"
                     f"{{\\shortstack[l]{{{display_label}}}}}")

        # Per-model "has local data" flag.  When a (model, bench) has no
        # local data at all, suppress merge-base fallback so stale cells
        # from older runs do not leak into the rebuilt table.
        has_local: dict[str, bool] = {}
        for model in model_order:
            has_local[model] = bool(model_benchmarks[model].get(bench_label))

        # Combined Pass@1 / Standard MV plateau row: single value per model,
        # spans threshold columns.
        p1_cells = [row_label, "Pass@1 / Standard MV plateau"]
        p1_row_key = (bench_label, "Pass@1 / Standard MV plateau")
        col_idx = 2
        for model in model_order:
            bdata = model_benchmarks[model].get(bench_label, {})
            p1 = bdata.get("pass_at_1")
            plateau = bdata.get("mv_plateau")
            p1_s = _format_acc(p1) if p1 is not None else "---"
            plat_s = _format_acc(plateau)
            raw_cell = f"{p1_s} / {plat_s}"
            # Treat a pure "--- / ---" as empty for merge purposes.
            if "---" in raw_cell and p1 is None and plateau is None:
                raw_cell = "---"
            cell = (fallback_cell(raw_cell, (p1_row_key, col_idx), base_cells)
                    if has_local[model] else raw_cell)
            p1_cells.append(cell)
            col_idx += 1
            for _ in range(n_thr - 1):
                extra = (fallback_cell("", (p1_row_key, col_idx), base_cells)
                         if has_local[model] else "")
                p1_cells.append(extra)
                col_idx += 1
        lines.append(" & ".join(p1_cells) + " \\\\")

        # Standard MV budget: absolute budget Standard MV needs per threshold
        bud_cells = ["", "Standard MV budget ($B_{\\mathrm{MV}}$)"]
        bud_row_key = (bench_label, "Standard MV budget ($B_{\\mathrm{MV}}$)")
        col_idx = 2
        for model in model_order:
            bdata = model_benchmarks[model].get(bench_label, {})
            mv_budgets = bdata.get("mv_budgets", {})
            for thr in thresholds:
                cell = _format_budget(mv_budgets.get(thr))
                if has_local[model]:
                    cell = fallback_cell(
                        cell, (bud_row_key, col_idx), base_cells)
                bud_cells.append(cell)
                col_idx += 1
        lines.append(" & ".join(bud_cells) + " \\\\")
        lines.append(
            "\\cmidrule(l){2-" + str(2 + n_models * n_thr) + "}")

        # Best ratio per column (across methods, within the valid band).
        # Ratio cells are now ``(point, sigma)`` tuples (or None);
        # bolding still uses the point estimate at 2-decimal precision.
        best_per_col: list[float] = []
        for model in model_order:
            bdata = model_benchmarks[model].get(bench_label, {})
            ratios_by_method = bdata.get("ratios", {})
            for thr in thresholds:
                valid: list[float] = []
                for m, _ in ratio_methods:
                    if m not in ratios_by_method:
                        continue
                    entry = ratios_by_method[m].get(thr)
                    if entry is None:
                        continue
                    r, _ = entry
                    if RATIO_LOW_CAP <= r <= RATIO_HIGH_CAP:
                        valid.append(round(r, 2))
                best_per_col.append(min(valid) if valid else float("inf"))

        for key, display_name in ratio_methods:
            cells = ["", display_name]
            method_row_key = (bench_label, display_name)
            tex_col_idx = 2
            col_i = 0
            for model in model_order:
                bdata = model_benchmarks[model].get(bench_label, {})
                ratios_by_method = bdata.get("ratios", {})
                for thr in thresholds:
                    if not ratios_by_method:
                        raw_cell = "---"
                    elif key not in ratios_by_method:
                        raw_cell = "---"
                    else:
                        entry = ratios_by_method[key].get(thr)
                        best = best_per_col[col_i]
                        if entry is None:
                            is_best = False
                        else:
                            r, _ = entry
                            is_best = (
                                math.isfinite(best)
                                and best < 0.95
                                and RATIO_LOW_CAP <= r <= RATIO_HIGH_CAP
                                and round(r, 2) == best
                            )
                        raw_cell = _format_ratio(entry, is_best)
                    if has_local[model]:
                        raw_cell = fallback_cell(
                            raw_cell,
                            (method_row_key, tex_col_idx), base_cells)
                    cells.append(raw_cell)
                    tex_col_idx += 1
                    col_i += 1
            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}% end resizebox")

    if legend_entries:
        notes = " ".join(
            f"$^{{{m}}}${scope}: {text}."
            for m, scope, text in legend_entries)
        lines.append("\\vspace{2pt}")
        lines.append(
            f"\\par\\raggedright\\footnotesize {notes}")

    lines.append("\\end{table}")
    return "\n".join(lines)


def parse_model_args(raw_args: list[str]) -> list[tuple[str, list[Path]]]:
    groups = []
    current_name = None
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX token efficiency table "
                    "(plateau-fraction axis)")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.75, 0.90, 0.99],
                        help="Gap fractions for column targets: "
                             "target = Pass@1 + thr * (MV_plateau - Pass@1)")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--label", default="tab:token_savings",
                        help="LaTeX label for the table")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--merge-base", type=Path, default=None,
                        help="Existing output .tex; cells this run cannot "
                             "compute locally fall back to its values.")
    parser.add_argument("--suppress-condition", action="store_true",
                        help="Omit `tau, K` from the caption.")
    parser.add_argument("--suppress-subset-footnote", action="store_true",
                        help="Omit per-cell ``N=...`` / problem-subset footnotes.")

    args, remaining = parser.parse_known_args()

    for t in args.thresholds:
        if not (0 < t <= 1):
            parser.error(f"thresholds must be in (0, 1]; got {t}")

    # Insert AC sweep and ESC sweep just before the PC family in the
    # canonical token-efficiency table. They have no useful row in the
    # WMV table (dominated by MV at fixed budget), but their cost-axis
    # behavior is informative and serves as a baseline against PC.
    natural_rows = [
        ("ac_sweep", METHOD_DISPLAY.get("ac_sweep", "AC sweep")),
        ("esc_sweep", METHOD_DISPLAY.get("esc_sweep", "ESC sweep")),
    ]
    methods = []
    natural_inserted = False
    for k, d in MAIN_METHODS:
        if k.startswith("prefix_") and not natural_inserted:
            methods.extend(natural_rows)
            natural_inserted = True
        methods.append((k, d))
    if not natural_inserted:
        methods.extend(natural_rows)
    if args.methods:
        methods = [(m, METHOD_DISPLAY.get(m, m)) for m in args.methods]
    method_keys = [k for k, _ in methods]

    # Bootstrap CI depends on the order in which `rng.normal` is consumed
    # across methods inside `_curves_from_aggregate`. To keep the canonical
    # table's CIs identical to the per-model `_all` tables (which evaluate
    # the full ALL_BASELINE_GROUPS), we run the bootstrap on the union of
    # both method sets here and only filter to `method_keys` at display time.
    if args.methods:
        compute_method_keys = method_keys
    else:
        all_keys = [k for _, ms in ALL_BASELINE_GROUPS for k, _ in ms]
        seen: set[str] = set()
        compute_method_keys = []
        for k in all_keys + method_keys:
            if k not in seen:
                seen.add(k)
                compute_method_keys.append(k)

    if "--model" in remaining:
        model_groups = parse_model_args(remaining)
    else:
        dirs = [Path(a) for a in remaining if Path(a).exists()]
        by_model: dict[str, list[Path]] = defaultdict(list)
        order = []
        for d in dirs:
            m = infer_model_label(d.name) or d.name
            if m not in by_model:
                order.append(m)
            by_model[m].append(d)
        model_groups = [(m, by_model[m]) for m in order]

    model_order = []
    model_benchmarks: dict[str, dict[str, dict]] = {}

    for model_name, dirs in model_groups:
        model_order.append(model_name)
        benchmarks: dict[str, dict] = {}

        for d in dirs:
            json_path = d / "wmv_result.json"
            if not json_path.exists():
                print(f"  WARNING: {json_path} not found, skipping",
                      file=sys.stderr)
                continue

            label = infer_benchmark_label(d.name)
            print(f"Loading {d.name} -> {label} ({model_name})")

            plateau, p1_acc, mv_budgets, ratios = compute_plateau_savings(
                json_path, args.condition, compute_method_keys, args.thresholds)

            if plateau is not None:
                p1_s = f"{p1_acc:.3f}" if p1_acc is not None else "N/A"
                print(f"  MV_plateau={plateau:.3f}  Pass@1={p1_s}")
                for thr in args.thresholds:
                    mv_b = mv_budgets.get(thr)
                    pc_entry = ratios.get("prefix_cubic", {}).get(thr)
                    if pc_entry is None:
                        pc_s = "N/A"
                    else:
                        pc, pc_sig = pc_entry
                        if pc_sig is not None:
                            pc_s = f"{pc:.2f}±{pc_sig:.2f}x"
                        else:
                            pc_s = f"{pc:.2f}x"
                    thr_s = f"{thr*100:.0f}%"
                    print(f"  @{thr_s} (MV bud={_format_budget(mv_b)}): "
                          f"PC-cubic={pc_s}")

            footnote = detect_subset_footnote(json_path, d.name)
            benchmarks[label] = {
                "mv_plateau": plateau,
                "mv_budgets": mv_budgets,
                "ratios": ratios,
                "pass_at_1": p1_acc,
                "footnote": footnote,
            }

        model_benchmarks[model_name] = benchmarks

    base_cells = None
    if args.merge_base:
        # Per-benchmark row pattern: data starts at col 2 (col 1 = row
        # label / method name). Cells per model = len(thresholds).
        expected_cols = 1 + len(args.thresholds) * len(model_order)
        base_cells = load_base_cells_checked(
            args.merge_base,
            _savings_row_key,
            expected_cols=expected_cols,
            label="merge-base",
            group_key_fn=_savings_group_key,
        )

    tex = generate_tex(model_order, model_benchmarks, methods,
                       args.thresholds, args.condition,
                       label=args.label, base_cells=base_cells,
                       suppress_condition=args.suppress_condition,
                       suppress_subset_footnote=args.suppress_subset_footnote)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex)
        print(f"\nWritten: {args.output}")
    else:
        print()
        print(tex)


if __name__ == "__main__":
    main()
