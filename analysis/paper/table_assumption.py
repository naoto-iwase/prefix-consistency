#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Generate the assumption-verification table for Theorem~\\ref{thm:convergence}.

For each (model, dataset) cell at tau, K=1, evaluate on Q' and report

  n            number of problems in Q' (correct and wrong both observed)
  P(A1)        per-wrong-answer boundary: for every a != a*,
               p r_C > pi(a) T(a -> a)
  P(A2 | A1)   pooled-mass dominance pi(a*)+rho(a*) > pi(a)+rho(a) for all wrong a,
               conditional on A1
  P(Delta_{w^(n)} > 0 | A1)  for w^(n)(c) = c^n, n in {1, 2, 3}, the empirical conclusion

The theorem predicts P(Delta_{w^(n)} > 0 | A1 and A2) = 1 for any convex w.
The table shows the empirical implication strength: when A1 alone holds,
how often does Delta_{w^(n)} > 0 also hold?

Usage (from analysis/):
    uv run --script paper/table_assumption.py \\
        gpt-oss-120b_aime2025_jsonl \\
        gpt-oss-120b_hmmt_jsonl \\
        ... \\
        --condition rm25pct_full_x1 \\
        -o tables/table_assumption.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from _utils import (
    bench_sort_key, model_sort_key,
    find_init_file, find_regen_files,
    infer_benchmark_label, infer_model_label,
    is_usable_answer,
)


# --------------------------------------------------------------------------
# Per-problem stats
# --------------------------------------------------------------------------

def _per_problem_stats(init_rec, regen_rec):
    """Return dict with a*, A, pi, rho, T, p, r_C, r_W, plus counts.

    Returns None if the problem is unusable (no pairs or no a* in init/regen).
    """
    gold = str(init_rec["gold_answer"])
    init_answers = init_rec["all_answers"]
    regen_answers = regen_rec.get("regen_answers", []) if regen_rec else []

    pairs = []
    for i, init_pair in enumerate(init_answers):
        if not is_usable_answer(init_pair):
            continue
        a_init = str(init_pair[0])
        if i >= len(regen_answers):
            continue
        r_list = regen_answers[i] or []
        if not r_list:
            continue
        r = r_list[0]
        if not is_usable_answer(r):
            continue
        a_regen = str(r[0])
        pairs.append((a_init, a_regen))

    if not pairs:
        return None

    A = set()
    for a_i, a_t in pairs:
        A.add(a_i)
        A.add(a_t)

    N = len(pairs)
    pi_count = Counter(a_i for a_i, _ in pairs)
    rho_count = Counter(a_t for _, a_t in pairs)
    pi = {a: pi_count.get(a, 0) / N for a in A}
    rho = {a: rho_count.get(a, 0) / N for a in A}

    T = {b: {a: 0.0 for a in A} for b in A}
    counts_b = Counter(a_i for a_i, _ in pairs)
    counts_ba = Counter(pairs)
    for b in A:
        if counts_b.get(b, 0):
            for a in A:
                T[b][a] = counts_ba.get((b, a), 0) / counts_b[b]

    n_correct = pi_count.get(gold, 0)
    n_wrong = N - n_correct
    if n_correct == 0 or n_wrong == 0:
        # Excluded from Q'
        return None

    p_pass = pi[gold]
    r_C = T[gold][gold]
    r_W = (sum(1 for a_i, a_t in pairs
               if a_i != gold and a_t == a_i) / n_wrong)

    return {
        "a_star": gold,
        "A": A, "pi": pi, "rho": rho, "T": T,
        "p": p_pass, "r_C": r_C, "r_W": r_W,
    }


def _assumption_flags(s):
    """Return (A1, A2, [Δ_n > 0 for n in 1..3]) for a per-problem stats dict.

    A1 (per-wrong-answer boundary): for every wrong ``a``,
    ``pi(a*) * r_C > pi(a) * T(a -> a)``. The aggregate form
    ``p r_C > (1-p) r_W`` is a stronger sufficient condition.
    """
    a_star = s["a_star"]
    p_rc = s["p"] * s["r_C"]
    a1 = all(p_rc > s["pi"][a] * s["T"][a][a]
             for a in s["A"] if a != a_star)
    pooled = {a: s["pi"].get(a, 0) + s["rho"].get(a, 0) for a in s["A"]}
    pooled_other_max = max((pooled[a] for a in s["A"] if a != s["a_star"]),
                           default=0.0)
    a2 = pooled[a_star] > pooled_other_max
    deltas = []
    for n in (1, 2, 3):
        w_half = 0.5 ** n
        lam = 1.0 - 2 * w_half
        phi = {a: (w_half * (s["pi"].get(a, 0) + s["rho"].get(a, 0))
                   + lam * s["pi"].get(a, 0) * s["T"][a].get(a, 0))
               for a in s["A"]}
        others = [phi[a] for a in s["A"] if a != a_star]
        deltas.append(phi[a_star] - (max(others) if others else 0.0) > 0)
    return a1, a2, deltas


def aggregate_cell(jsonl_dir: Path, condition: str):
    """Compute assumption-verification stats for one (model, dataset) cell.

    Matches `condition` (e.g. ``rm25pct_full_x1``) on (rm_pct, scope) and
    uses only the first ``K`` regens per init when the available file has
    ``K' >= K`` regens, mirroring the fallback in wmv/jsonl_loader.py.
    """
    init_path = find_init_file(jsonl_dir)
    if not init_path:
        return None
    import re
    m = re.match(r"rm(\d+)pct_(\w+)_x(\d+)$", condition)
    if not m:
        return None
    want_rm, want_scope, want_k = int(m.group(1)), m.group(2), int(m.group(3))
    candidates = [(rm, sc, k, p) for rm, sc, k, p in find_regen_files(jsonl_dir)
                  if rm == want_rm and sc == want_scope and k >= want_k]
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[2])  # smallest K first
    regen_path = candidates[0][3]

    inits = {}
    with open(init_path) as f:
        for line in f:
            rec = json.loads(line)
            inits[rec["problem_num"]] = rec
    regens = {}
    with open(regen_path) as f:
        for line in f:
            rec = json.loads(line)
            regens[rec["problem_num"]] = rec

    n_problems = 0
    a1_count = a2_and_a1_count = 0
    delta_given_a1_num = {1: 0, 2: 0, 3: 0}

    for pnum, init in inits.items():
        s = _per_problem_stats(init, regens.get(pnum))
        if s is None:
            continue
        n_problems += 1
        a1, a2, deltas = _assumption_flags(s)
        if a1:
            a1_count += 1
            if a2:
                a2_and_a1_count += 1
            for n, d in zip((1, 2, 3), deltas):
                if d:
                    delta_given_a1_num[n] += 1

    if n_problems == 0:
        return None

    return {
        "n_problems": n_problems,
        "n_a1": a1_count,
        "frac_a1": a1_count / n_problems,
        "frac_a2_and_a1": a2_and_a1_count / n_problems,
        "P_a2_given_a1": (a2_and_a1_count / a1_count) if a1_count else float("nan"),
        "P_delta_given_a1": {n: ((delta_given_a1_num[n] / a1_count)
                                 if a1_count else float("nan"))
                             for n in (1, 2, 3)},
    }


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------

def _fmt_pct(x):
    if x is None or x != x:  # NaN
        return "--"
    return f"{x * 100:.1f}"


def generate_latex(rows, condition):
    del condition  # Caption omits the per-cell condition string; defaults
                   # for tau/K live in Appendix F (Hyperparameter defaults).

    by_bench = defaultdict(list)
    for r in rows:
        by_bench[r["benchmark_label"]].append(r)
    bench_order = sorted(by_bench.keys(), key=bench_sort_key)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{\textbf{Empirical verification of "
        r"Theorem~\ref{thm:convergence}'s assumptions} on $\mathcal{Q}'$, "
        r"checking $\mathrm{A1}$ and $\mathrm{A2}$ for every wrong $a$. "
        r"All probabilities are macro-averaged: each problem in $\mathcal{Q}'$ "
        r"contributes one indicator per event. "
        r"Theorem~\ref{thm:convergence} predicts $\Pr[\Delta_{w^{(n)}} > 0 \mid \mathrm{A1} \cap \mathrm{A2}] = 1$ "
        r"for every convex $w$. The last three columns report the weaker "
        r"$\Pr[\Delta_{w^{(n)}} > 0 \mid \mathrm{A1}]$ for $w^{(n)}(c) = c^n$, $n \in \{1, 2, 3\}$.}")
    lines.append(r"\label{tab:assumption}")
    lines.append(r"\small")
    col_spec = "@{}llrrrrrr@{}"
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{Benchmark} & \multirow{2}{*}{Model} & "
        r"\multirow{2}{*}{$|\mathcal{Q}'|$} & "
        r"\multirow{2}{*}{$\Pr[\mathrm{A1}]$} & "
        r"\multirow{2}{*}{$\Pr[\mathrm{A2} \mid \mathrm{A1}]$} & "
        r"\multicolumn{3}{c}{$\Pr[\Delta_{w^{(n)}} > 0 \mid \mathrm{A1}]$} \\"
    )
    lines.append(r"\cmidrule(lr){6-8}")
    lines.append(
        r" & & & & & $n{=}1$ & $n{=}2$ & $n{=}3$ \\"
    )
    lines.append(r"\midrule")

    for bi, bl in enumerate(bench_order):
        if bi > 0:
            lines.append(r"\midrule")
        group = sorted(by_bench[bl], key=lambda r: model_sort_key(r["model"]))
        for ri, r in enumerate(group):
            if ri == 0:
                bench_cell = rf"\multirow{{{len(group)}}}{{*}}{{{bl}}}"
            else:
                bench_cell = ""
            cells = [
                bench_cell,
                r["model"],
                str(r["n_problems"]),
                _fmt_pct(r["frac_a1"]),
                _fmt_pct(r["P_a2_given_a1"]),
                _fmt_pct(r["P_delta_given_a1"][1]),
                _fmt_pct(r["P_delta_given_a1"][2]),
                _fmt_pct(r["P_delta_given_a1"][3]),
            ]
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="JSONL dirs (e.g. gpt-oss-120b_aime2025_jsonl)")
    ap.add_argument("--condition", default="rm25pct_full_x1",
                    help="Condition tag (default: rm25pct_full_x1, i.e. tau=0.75 K=1)")
    ap.add_argument("--analysis-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent,
                    help="Base dir to resolve relative paths against")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output LaTeX path (default: stdout)")
    args = ap.parse_args()

    rows = []
    for d in args.dirs:
        if not d.is_absolute():
            d = args.analysis_dir / d
        cell = aggregate_cell(d, args.condition)
        if cell is None:
            print(f"!! skip {d.name}: no condition {args.condition} or no usable problems",
                  file=sys.stderr)
            continue
        cell["model"] = infer_model_label(d.name) or d.name
        cell["benchmark_label"] = infer_benchmark_label(d.name)
        rows.append(cell)

    if not rows:
        sys.exit("No rows to emit.")

    latex = generate_latex(rows, args.condition)
    if args.output:
        args.output.write_text(latex)
        print(f"Wrote {args.output}")
    else:
        print(latex)


if __name__ == "__main__":
    main()
