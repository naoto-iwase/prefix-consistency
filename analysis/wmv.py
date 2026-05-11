#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""
Weighted majority vote evaluation with token-fair comparison.

Evaluates multiple WMV methods (prefix consistency, logit confidence,
verbal confidence, branching, marker, ESC) at matched token budgets.

When --removal-tag and --regen-count are omitted, all conditions are
auto-detected from the JSONL directory and processed in one invocation.
Shared families (init, verbal, ESC, marker) are evaluated once;
branching once per regen-count; prefix once per condition.

Usage:
    # All conditions (auto-detect):
    uv run python wmv.py <jsonl_dir>

    # Single condition:
    uv run python wmv.py <jsonl_dir> \\
        --removal-tag rm50pct_full --regen-count 3
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from wmv.jsonl_loader import (
    attach_branching_data, attach_marker_data, find_branching_jsonls,
    detect_conditions, detect_data_availability,
    load_branching_data, load_data, load_init_data,
    load_marker_regen_data,
)
from wmv.eval import evaluate_curves
from wmv.voters import (
    ACSweepCore,
    ACSweepVoter,
    ESCSweepCore,
    ESCSweepVoter,
    MultiPointsVoter,
    OracleGroupVoter,
    OracleInitVoter,
    PrefixUnanimousVoter,
    PrefixWeightedVoter,
    SoftmaxVoter,
    StandardVoter,
    TopFilterVoter,
    WeightedVoter,
)


def _default_dense_token_points() -> List[int]:
    """100 points per decade, log-uniform, covering 10^3 to 10^7 tokens."""
    pts = np.logspace(3, 7, 401)
    return sorted(set(int(round(x)) for x in pts))


# Method tuple: (name, family, fn, cost_type).
# ``fn`` is unused now that every method has a voter in
# ``voter_factories`` and ``evaluate_curves`` is the only evaluator;
# the field is retained as ``None`` to keep the tuple shape stable for
# downstream consumers (e.g. ``--methods`` filtering, family grouping).
MethodEntry = Tuple[str, str, Optional[Callable], str]


# =====================================================================
# Helpers
# =====================================================================



def apply_verbal_cap(problems, cap):
    """Cap verbal query tokens; discard confidence for over-budget answers."""
    if cap is None:
        return
    n_capped_v0100 = 0
    n_capped_binary = 0
    for p in problems:
        for i, t in enumerate(p["init_verbal_0_100_actual_tokens"]):
            if t > cap:
                p["init_verbal_0_100_actual_tokens"][i] = cap
                p["init_verbal_0_100_confs"][i] = 50.0
                n_capped_v0100 += 1
        for i, t in enumerate(p["init_binary_query_actual_tokens"]):
            if t > cap:
                p["init_binary_query_actual_tokens"][i] = cap
                p["init_verbal_binary_confs"][i] = 0.5
                p["init_p_true_confs"][i] = 0.5
                n_capped_binary += 1
    n_capped = n_capped_v0100 + n_capped_binary
    if n_capped:
        print(f"--verbal-max-tokens {cap}: capped {n_capped} answers "
              f"(v0100={n_capped_v0100}, binary={n_capped_binary}), "
              f"confidence discarded for those answers")


def n_label(per_problem_counts):
    lo, hi = min(per_problem_counts), max(per_problem_counts)
    if lo == hi:
        return str(lo)
    return f"{lo}..{hi}"


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Weighted MV evaluation (cost-fair)")
    parser.add_argument("jsonl_dir", type=Path)
    parser.add_argument("--removal-tag", type=str, default=None,
                        help="Removal tag (e.g. rm50pct_cot). "
                             "Omit to auto-detect all conditions.")
    parser.add_argument("--regen-count", type=int, default=None,
                        help="Regen count. Omit to auto-detect.")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--token-points", nargs="+", type=int,
                        default=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000,
                                 100_000, 250_000, 500_000, 1_000_000,
                                 2_500_000, 5_000_000, 10_000_000],
                        help="Sparse token budget grid for the evaluate_curves "
                             "sparse snapshots (default: geometric sequence "
                             "1k..10M). Output goes to wmv_result.json "
                             "'token_budget'.")
    parser.add_argument("--dense", action="store_true",
                        help="Additionally snapshot evaluate_curves at a dense "
                             "token-budget grid. Output goes to "
                             "wmv_result.json 'token_budget_dense' and feeds "
                             "the token-efficiency table. Off by default to "
                             "preserve the analyze_jsonls.sh workflow; opt in "
                             "explicitly.")
    parser.add_argument("--token-points-dense", nargs="+", type=int,
                        default=None,
                        help="Dense grid used by --dense. Default: 100 "
                             "points per decade, log-uniform, 10^3..10^7 "
                             "(401 points). Ignored without --dense.")
    parser.add_argument("--sample-points", nargs="+", type=int,
                        default=[1],
                        help="Fixed sample count evaluation points "
                             "(default: [1]). Unused by the paper tables; "
                             "extra points are snapshotted for free in the "
                             "unified evaluate_curves pass.")
    parser.add_argument("--verbal-max-tokens", type=int, default=None,
                        help="Cap per-answer verbal query tokens at this value. "
                             "Limits cost impact of runaway generation "
                             "(e.g. model reopening analysis channel).")
    parser.add_argument("--marker-regen-jsonl", type=Path, default=None,
                        help="Marker-based regen JSONL (from regenerate_from_markers.py). "
                             "Auto-detected from jsonl_dir if not specified.")
    parser.add_argument("--esc-window-sizes", nargs="+", type=int,
                        default=[2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help="ESC window sizes (default: 2..10). The ESC "
                             "paper (Li et al. ICLR 2024) uses w=5 for most "
                             "tasks and w=8 for MATH; the sweep produces a "
                             "natural-stopping cost-accuracy curve.")
    parser.add_argument("--ac-conf-thresholds", nargs="+", type=float,
                        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99,
                                 0.995, 0.999],
                        help="Adaptive Consistency confidence thresholds. "
                             "AC (Aggarwal et al. EMNLP 2023) sweeps "
                             "C_thresh from 0.5 to 1.0 in Figure 2; 0.95 "
                             "is the official default. Spacing is "
                             "log-uniform in (1 - threshold).")
    parser.add_argument("--problems", nargs="+", type=int, default=None,
                        help="Filter to specific problem numbers (0-indexed, matching JSONL). "
                             "E.g., --problems 30 11")
    parser.add_argument("--methods", nargs="+", type=str, default=None,
                        help="Filter to specific method names. "
                             "E.g., --methods standard_mv prefix_unanimous")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for wmv_result.json. "
                             "If a directory (or suffix-less path), writes "
                             "wmv_result.json and wmv_detail.json inside it. "
                             "If omitted and --problems/--methods filters are active, "
                             "no file is written (stdout only). "
                             "If omitted on a full run, defaults to <jsonl_dir>/wmv_result.json.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and print summary, then exit "
                             "(no evaluation or plotting).")
    args = parser.parse_args()

    # =================================================================
    # Auto-detect conditions
    # =================================================================
    conditions = detect_conditions(args.jsonl_dir, args.removal_tag,
                                    args.regen_count)
    if not conditions:
        raise FileNotFoundError(
            f"No regen JSONL files found in {args.jsonl_dir}")
    unique_rcs = sorted(set(rc for _, rc in conditions))
    print(f"Conditions: {len(conditions)} "
          f"({', '.join(f'{t}_x{rc}' for t, rc in conditions)})")

    # Auto-detect marker JSONL
    if args.marker_regen_jsonl is None:
        marker_candidates = sorted(args.jsonl_dir.glob("*_markers_x*.jsonl"))
        if marker_candidates:
            args.marker_regen_jsonl = marker_candidates[0]
            print(f"Auto-detected marker JSONL: {args.marker_regen_jsonl.name}")

    is_filtered = bool(args.problems or args.methods)
    if args.output:
        if args.output.is_dir() or args.output.suffix == "":
            args.output.mkdir(parents=True, exist_ok=True)
            output_path = args.output / "wmv_result.json"
        else:
            output_path = args.output
    elif is_filtered:
        output_path = None
        print("NOTE: --problems/--methods active without --output — results will not be saved.")
    else:
        output_path = args.jsonl_dir / "wmv_result.json"

    # =================================================================
    # Phase 1: Shared setup (load init, build methods, marker, verbal)
    # =================================================================
    init_data, budget = load_init_data(args.jsonl_dir)
    problems = [{"pnum": pnum, **d, "groups": []}
                for pnum, d in sorted(init_data.items())]

    if args.problems:
        problem_set = set(args.problems)
        problems = [p for p in problems if p["pnum"] in problem_set]
        if not problems:
            raise ValueError(f"No problems found for --problems {args.problems}")
        print(f"Problem filter: {sorted(args.problems)} -> {len(problems)} problems")

    # Drop problems that lost all init answers to the loader filter.
    nonempty = [p for p in problems if p["init_tokens"]]
    if not nonempty:
        raise ValueError("All problems empty after filtering; check source data.")
    if len(nonempty) < len(problems):
        print(f"WARNING: {len(problems) - len(nonempty)} problems have "
              f"no usable init answers and are excluded.")
        problems = nonempty
    min_init_cost = min(min(p["init_tokens"]) for p in problems)

    avail = detect_data_availability(problems)
    has_confidence = avail["has_confidence"]
    has_verbal_0_100 = avail["has_verbal_0_100"]
    has_verbal_binary = avail["has_verbal_binary"]
    has_p_true = avail["has_p_true"]
    has_response_prob = avail["has_response_prob"]
    has_v0100_tokens = avail["has_v0100_tokens"]
    has_binary_tokens = avail["has_binary_tokens"]

    apply_verbal_cap(problems, args.verbal_max_tokens)

    print(f"Problems: {len(problems)}, budget={budget}")
    print(f"Min init token cost: {min_init_cost:,}")
    print(f"Confidence: {has_confidence}, verbal_0_100: {has_verbal_0_100}, "
          f"verbal_binary: {has_verbal_binary}, p_true: {has_p_true}, "
          f"response_prob: {has_response_prob}")
    print(f"Trials: {args.trials}\n")

    # Build methods in legend order: baseline, our method, competitors.
    # ``voter_factories`` is a parallel registry mapping method name to a
    # voter factory (``factory(problem) -> Voter``).  Every method has a
    # voter, so registration here is mandatory and ``evaluate_curves``
    # populates both ``token_budget`` and ``token_budget_dense``.
    methods: List[MethodEntry] = []
    voter_factories: Dict[str, Callable] = {}

    methods.append(("standard_mv", "init", None, "init"))
    voter_factories["standard_mv"] = StandardVoter
    methods.append(("oracle_init", "oracle", None, "init"))
    voter_factories["oracle_init"] = OracleInitVoter

    methods.append(("oracle_prefix", "oracle", None, "prefix"))
    voter_factories["oracle_prefix"] = OracleGroupVoter
    methods.extend([
        ("prefix_unanimous", "prefix", None, "prefix"),
        ("prefix_cubic", "prefix", None, "prefix"),
        ("prefix_quadratic", "prefix", None, "prefix"),
        ("prefix_linear", "prefix", None, "prefix"),
    ])
    voter_factories["prefix_unanimous"] = PrefixUnanimousVoter
    voter_factories["prefix_cubic"] = partial(
        PrefixWeightedVoter, weight_fn=lambda f: f ** 3)
    voter_factories["prefix_quadratic"] = partial(
        PrefixWeightedVoter, weight_fn=lambda f: f ** 2)
    voter_factories["prefix_linear"] = partial(
        PrefixWeightedVoter, weight_fn=lambda f: f)

    if has_confidence:
        for name, ck in [
            ("deepconf_mean",      "mean_conf"),
            ("deepconf_bottom10",  "bottom10_conf"),
            ("deepconf_tail",      "tail_conf"),
            ("deepconf_first_token", "first_token_conf"),
            ("deepconf_block_min", "block_min_conf"),
        ]:
            methods.append((name, "deepconf", None, "init"))
            voter_factories[name] = partial(
                WeightedVoter, conf_key=ck, weight_fn=lambda c: c)
        for name, ck, top_pct in [
            ("deepconf_tail_top10pct",      "tail_conf",      10),
            ("deepconf_tail_top90pct",      "tail_conf",      90),
            ("deepconf_bottom10_top10pct",  "bottom10_conf",  10),
            ("deepconf_bottom10_top90pct",  "bottom10_conf",  90),
        ]:
            methods.append((name, "deepconf_filtered", None, "init"))
            voter_factories[name] = partial(
                TopFilterVoter, conf_key=ck, top_pct=top_pct)
    # CISC methods (Taubenfeld et al., ACL Findings 2025)
    # Two normalization strategies reported in the paper:
    #   _raw:         w = c  (no normalization; best untuned, see paper Table 8)
    #   _softmax_T1:  w = softmax(c, T=1)  (paper Definition 3.1; needs T-tuning)
    # p_true's voters share the verbal_binary cost vector
    # (init_tokens + init_binary_query_min_tokens) but use
    # init_p_true_confs as their weight source, hence
    # cost_type="verbal_binary".
    if has_response_prob:
        _rp = "init_response_probs"
        methods.extend([
            ("response_prob_raw", "cisc", None, "init"),
            ("response_prob_softmax_T1", "cisc", None, "init"),
        ])
        voter_factories["response_prob_raw"] = partial(
            WeightedVoter, conf_key=_rp, weight_fn=lambda c: c)
        voter_factories["response_prob_softmax_T1"] = partial(
            SoftmaxVoter, conf_key=_rp, temp=1.0)

    _v0100 = "init_verbal_0_100_confs"
    _vbin = "init_verbal_binary_confs"
    _ptrue = "init_p_true_confs"
    if has_verbal_0_100:
        methods.extend([
            ("verbal_0_100_raw", "cisc", None, "verbal_0_100"),
            ("verbal_0_100_softmax_T1", "cisc", None, "verbal_0_100"),
        ])
        voter_factories["verbal_0_100_raw"] = partial(
            WeightedVoter, conf_key=_v0100,
            weight_fn=lambda c: c / 100)
        voter_factories["verbal_0_100_softmax_T1"] = partial(
            SoftmaxVoter, conf_key=_v0100, temp=1.0)
    if has_verbal_binary:
        methods.extend([
            ("verbal_binary_raw", "cisc", None, "verbal_binary"),
            ("verbal_binary_softmax_T1", "cisc", None, "verbal_binary"),
        ])
        voter_factories["verbal_binary_raw"] = partial(
            WeightedVoter, conf_key=_vbin, weight_fn=lambda c: c)
        voter_factories["verbal_binary_softmax_T1"] = partial(
            SoftmaxVoter, conf_key=_vbin, temp=1.0)
    if has_p_true:
        methods.extend([
            ("p_true_raw", "cisc", None, "verbal_binary"),
            ("p_true_softmax_T1", "cisc", None, "verbal_binary"),
        ])
        voter_factories["p_true_raw"] = partial(
            WeightedVoter, conf_key=_ptrue, weight_fn=lambda c: c)
        voter_factories["p_true_softmax_T1"] = partial(
            SoftmaxVoter, conf_key=_ptrue, temp=1.0)

    # Load marker regen data (once, condition-independent)
    marker_data = None
    if args.marker_regen_jsonl:
        marker_data = load_marker_regen_data(args.marker_regen_jsonl)
        attach_marker_data(problems, marker_data)
        methods.append(("oracle_marker", "oracle", None, "marker"))
        methods.append(("markers", "marker", None, "marker"))
        voter_factories["oracle_marker"] = OracleGroupVoter
        voter_factories["markers"] = MultiPointsVoter

        n_m_groups = sum(len(entry["groups"]) for entry in marker_data.values())
        all_m_toks = [g["group_tokens"] for e in marker_data.values()
                      for g in e["groups"]]
        avg_m = sum(all_m_toks) / len(all_m_toks) if all_m_toks else 0
        print(f"Marker regen data: {len(marker_data)} problems, "
              f"avg groups/problem: {n_m_groups / len(marker_data):.1f}, "
              f"avg_group_tokens: {avg_m:.0f}")

    # One sweep core per (trial, problem) shared by every voter in the
    # sweep; recreated on the next factory call once ``n_added > 0``.
    def _register_sweep(entries, core_factory, family, cost_type):
        cell = {"core": None}
        def take_core(problem):
            c = cell["core"]
            if c is None or c.n_added > 0:
                c = core_factory(problem)
                cell["core"] = c
            return c
        for name, voter_factory in entries:
            methods.append((name, family, None, cost_type))
            voter_factories[name] = (
                lambda problem, vf=voter_factory: vf(take_core(problem)))

    if args.esc_window_sizes:
        ws_list = list(args.esc_window_sizes)
        _register_sweep(
            [(f"esc_w{w}",
              (lambda i: lambda c: ESCSweepVoter(c, i))(i))
             for i, w in enumerate(ws_list)],
            lambda p: ESCSweepCore(p, ws_list),
            family="esc", cost_type="init")

    if args.ac_conf_thresholds:
        ac_thr = list(args.ac_conf_thresholds)
        _register_sweep(
            [(f"ac_t{int(round(t * 1000)):04d}",
              (lambda i: lambda c: ACSweepVoter(c, i))(i))
             for i, t in enumerate(ac_thr)],
            lambda p: ACSweepCore(p, ac_thr),
            family="ac", cost_type="init")

    # Branching methods (added once, evaluated per-rc)
    has_any_branching = any(
        find_branching_jsonls(args.jsonl_dir, rc)
        for rc in unique_rcs
    )
    if has_any_branching:
        methods.append(("oracle_branching", "oracle", None, "branching"))
        methods.append(("multi_cut_points", "branching", None, "branching"))
        voter_factories["oracle_branching"] = OracleGroupVoter
        voter_factories["multi_cut_points"] = MultiPointsVoter

    if args.methods:
        method_set = set(args.methods)
        unknown = method_set - {n for n, _, _, _ in methods}
        if unknown:
            print(f"WARNING: unknown method names: {sorted(unknown)}")
        methods = [(n, f, fn, c) for n, f, fn, c in methods if n in method_set]
        if not methods:
            raise ValueError(f"No methods matched --methods {args.methods}")
        voter_factories = {n: f for n, f in voter_factories.items()
                         if n in method_set}
        print(f"Method filter: {[n for n, _, _, _ in methods]}")

    # Dense grid resolution (only consulted when --dense is passed).
    dense_token_points_cfg = args.token_points_dense
    if dense_token_points_cfg is None:
        dense_token_points_cfg = _default_dense_token_points()
    run_dense = bool(args.dense) and bool(voter_factories) and bool(
        dense_token_points_cfg)

    print(f"Token points: {args.token_points}")
    print(f"Sample points: {args.sample_points}  (fixed-count mode)")
    if run_dense:
        print(f"Dense token points: {len(dense_token_points_cfg)} "
              f"(min={dense_token_points_cfg[0]:,}, "
              f"max={dense_token_points_cfg[-1]:,})")
        print(f"Dense-eligible methods: "
              f"{sorted(voter_factories)}")
    else:
        print("Dense evaluation: off (pass --dense to enable)")

    if args.dry_run:
        # Load and summarize all conditions without running evaluation
        print(f"\n{'='*60}")
        print("Per-condition data summary (dry-run)")
        print(f"{'='*60}")
        for cond_tag, cond_rc in conditions:
            cond_label = f"{cond_tag}_x{cond_rc}"
            problems_cond, _ = load_data(args.jsonl_dir, cond_tag, cond_rc, init_data)
            apply_verbal_cap(problems_cond, args.verbal_max_tokens)
            if marker_data:
                attach_marker_data(problems_cond, marker_data)
            n_groups = [len(p["groups"]) for p in problems_cond]
            print(f"\n  {cond_label}")
            print(f"    prefix groups: "
                  f"min={min(n_groups)} max={max(n_groups)} "
                  f"avg={sum(n_groups)/len(n_groups):.1f}")

        active_tps = [tp for tp in args.token_points
                      if tp >= min_init_cost]
        skipped_tps = [tp for tp in args.token_points
                       if tp < min_init_cost]

        print(f"\n{'='*60}")
        print("Methods and token points")
        print(f"{'='*60}")
        families = {}
        for name, family, _, cost_type in methods:
            families.setdefault(family, []).append(name)
        for fam, names in families.items():
            print(f"  {fam}: {', '.join(names)}")
        print(f"\n  Active token points ({len(active_tps)}): "
              + ", ".join(f"{tp:,}" for tp in active_tps))
        if skipped_tps:
            print(f"  Skipped (budget < 1 init): "
                  + ", ".join(f"{tp:,}" for tp in skipped_tps))
        print(f"\n  Total methods: {len(methods)}")
        print(f"  Total evaluations: "
              f"{len(active_tps)} tp x {len(conditions)} conditions x "
              f"{args.trials} trials")
        return

    # Method metadata
    method_cost_type = {name: cost_type for name, _, _, cost_type in methods}
    method_families = {name: family for name, family, _, _ in methods}

    # Look up registered voter factories by cost-type family.  Every
    # method has a voter, so this also functions as the source of truth
    # for which methods are evaluated in each pass of evaluate_curves.
    def _voter_factories_for(cost_type):
        return [(n, voter_factories[n])
                for n, _, _, ct in methods
                if ct == cost_type and n in voter_factories]

    # Canonical legend order
    canonical_order = [
        "oracle_init", "oracle_prefix", "oracle_branching", "oracle_marker",
        "standard_mv",
        "prefix_unanimous", "prefix_cubic", "prefix_quadratic", "prefix_linear",
        "deepconf_mean", "deepconf_bottom10", "deepconf_tail",
        "deepconf_first_token", "deepconf_block_min",
        "deepconf_tail_top10pct", "deepconf_tail_top90pct",
        "deepconf_bottom10_top10pct", "deepconf_bottom10_top90pct",
        "response_prob_raw", "response_prob_softmax_T1",
        "verbal_0_100_raw", "verbal_0_100_softmax_T1",
        "verbal_binary_raw", "verbal_binary_softmax_T1",
        "p_true_raw", "p_true_softmax_T1",
        *[f"esc_w{ws}" for ws in args.esc_window_sizes],
        *[f"ac_t{int(round(ct * 1000)):04d}" for ct in args.ac_conf_thresholds],
        "multi_cut_points", "markers",
    ]
    def _reorder(d):
        ordered = {}
        for name in canonical_order:
            if name in d:
                ordered[name] = d[name]
        for name in d:
            if name not in ordered:
                ordered[name] = d[name]
        return ordered

    # =================================================================
    # Phase 2: Shared evaluation (init, verbal, marker)
    #
    # Every method is evaluated as a voter via evaluate_curves;
    # each cost-vector family (init, verbal_0_100, verbal_binary,
    # marker) is one pass over a sample sequence per trial, snapshotting
    # at token_points_sparse (-> token_budget), token_points_dense
    # (-> token_budget_dense) and sample_points (-> sample_count) in a
    # single pass.
    # =================================================================
    active_tp = sorted(t for t in args.token_points if t >= min_init_cost)
    skipped_tp = sorted(t for t in args.token_points if t < min_init_cost)
    if skipped_tp:
        print(f"Skipping {len(skipped_tp)} token points < min_init_cost "
              f"({min_init_cost:,})")

    dense_tp_arg = dense_token_points_cfg if run_dense else []

    init_factories = _voter_factories_for("init")
    prefix_factories = _voter_factories_for("prefix")
    branching_factories = _voter_factories_for("branching")
    marker_factories = _voter_factories_for("marker")
    v0100_factories = _voter_factories_for("verbal_0_100")
    vbin_factories = _voter_factories_for("verbal_binary")

    print(f"\n{'='*60}")
    print("Phase 2: Shared evaluation (init, verbal, ESC, marker)")
    print(f"{'='*60}")

    shared_tok: Dict[int, dict] = {tp: {} for tp in active_tp}
    shared_samp: Dict[int, dict] = {sp: {} for sp in args.sample_points}
    shared_dense: Dict[int, dict] = {}
    natural_stopping: Dict[str, dict] = {}
    natural_stopping_detail: Dict[str, dict] = {}

    # ── 2a. evaluate_curves for the shared-pool methods ──
    #        init_factories use init_tokens as cost; verbal_0_100 /
    #        verbal_binary / p_true factories use init_tokens +
    #        verbal-query overhead and run as parallel sub-passes
    #        within the same trial loop.
    if init_factories or v0100_factories or vbin_factories:
        all_factories = init_factories + v0100_factories + vbin_factories
        print(f"  [shared]  voters: {[n for n, _ in all_factories]}")
        t0 = time.time()
        curves = evaluate_curves(
            problems,
            init_voter_factories=init_factories,
            prefix_voter_factories=[],
            verbal_0_100_voter_factories=v0100_factories,
            verbal_binary_voter_factories=vbin_factories,
            token_points_sparse=active_tp,
            token_points_dense=dense_tp_arg,
            sample_points=args.sample_points,
            n_trials=args.trials,
            progress_label="shared",
        )
        print(f"  [shared]  done in {time.time() - t0:.1f}s")
        for tp in active_tp:
            shared_tok[tp].update(curves["token_budget"].get(tp, {}))
        for sp in args.sample_points:
            shared_samp[sp].update(curves["sample_count"].get(sp, {}))
        for tp, row in curves["token_budget_dense"].items():
            shared_dense.setdefault(tp, {}).update(row)
        natural_stopping.update(curves.get("natural_stopping", {}))
        natural_stopping_detail.update(
            curves.get("natural_stopping_detail", {}))

    # ── 2b. evaluate_curves for the marker methods.  Markers are global ──
    #       (attached to problems, not condition-specific).
    marker_tok: Dict[int, dict] = {}
    marker_samp: Dict[int, dict] = {}
    marker_dense: Dict[int, dict] = {}
    if marker_factories:
        print(f"  [marker]  voters: {[n for n, _ in marker_factories]}")
        t0 = time.time()
        curves = evaluate_curves(
            problems,
            init_voter_factories=[],
            prefix_voter_factories=[],
            marker_voter_factories=marker_factories,
            token_points_sparse=active_tp,
            token_points_dense=dense_tp_arg,
            sample_points=args.sample_points,
            n_trials=args.trials,
            progress_label="marker",
        )
        print(f"  [marker]  done in {time.time() - t0:.1f}s")
        marker_tok = curves["token_budget"]
        marker_samp = curves["sample_count"]
        marker_dense = curves["token_budget_dense"]

    # =================================================================
    # Phase 3: Per-rc branching evaluation (all branching methods are inc)
    # =================================================================
    branching_tok_by_rc = {}
    branching_samp_by_rc = {}
    branching_dense_by_rc = {}
    branching_data_by_rc = {}

    for rc in unique_rcs:
        branching_jsonls = find_branching_jsonls(args.jsonl_dir, rc)
        if not branching_jsonls or not branching_factories:
            continue

        print(f"\n{'='*60}")
        print(f"Phase 3: Branching (rc={rc})")
        print(f"{'='*60}")

        branching_data, branch_info = load_branching_data(
            branching_jsonls, rc)
        branching_data_by_rc[rc] = branching_data
        attach_branching_data(problems, branching_data)

        for tag_str, pct_frac in branch_info:
            print(f"  {tag_str} (removal_frac={pct_frac:.2f})")

        t0 = time.time()
        curves = evaluate_curves(
            problems,
            init_voter_factories=[],
            prefix_voter_factories=[],
            branching_voter_factories=branching_factories,
            token_points_sparse=active_tp,
            token_points_dense=dense_tp_arg,
            sample_points=args.sample_points,
            n_trials=args.trials,
            progress_label=f"branching rc={rc}",
        )
        print(f"  [branching rc={rc}]  done in {time.time() - t0:.1f}s")
        branching_tok_by_rc[rc] = curves["token_budget"]
        branching_samp_by_rc[rc] = curves["sample_count"]
        branching_dense_by_rc[rc] = curves["token_budget_dense"]

    # =================================================================
    # Phase 4: Per-condition prefix evaluation + merge + output build
    # =================================================================
    SHARED_COST_TYPES = {"init", "verbal_0_100", "verbal_binary",
                          "esc", "marker"}
    K_ONLY_COST_TYPES = {"branching"}

    def _build_entries(eval_dict, point_key):
        """Convert {point: {method: (acc,ci,pp,pp_tok,mn)}} to method entries."""
        results: Dict[str, List[dict]] = {n: [] for n in method_cost_type}
        pp_results: Dict[str, List[dict]] = {n: [] for n in method_cost_type}
        for pt in sorted(eval_dict):
            for name, tup in eval_dict[pt].items():
                if name not in method_cost_type:
                    continue
                acc, ci, pp, pp_tok, mean_n = tup
                results[name].append({
                    point_key: pt, "acc": acc, "ci": ci,
                    "mean_tokens": float(np.mean(pp_tok)),
                    "n_answers": mean_n, "n_samples": mean_n,
                })
                pp_results[name].append({
                    point_key: pt, "per_problem": pp,
                    "pp_tokens": pp_tok,
                    "n_answers": mean_n, "n_samples": mean_n,
                })
        return results, pp_results

    def _format_cond_methods(results, pp_results, point_key):
        """Build JSON-serialisable method dicts for one condition."""
        cond_methods = {}
        cond_methods_detail = {}
        for name in results:
            summary_entries = []
            detail_entries = []
            pp_tokens_dedup = {}
            for r, pp_r in zip(results[name], pp_results.get(name, [])):
                n_samples = round(r.get("n_samples", 0), 1)
                n_key = str(int(n_samples)) if n_samples else "0"
                pv = r[point_key]
                summary_entries.append({
                    point_key: pv,
                    "acc": round(r["acc"], 6),
                    "ci": round(r["ci"], 6),
                    "mean_tokens": round(r.get("mean_tokens", 0), 1),
                    "n_answers": round(r.get("n_answers", 0), 1),
                    "n_samples": n_samples,
                })
                detail_entries.append({
                    point_key: pv,
                    "n_samples": n_samples,
                    "per_problem": [round(v, 4) for v in pp_r["per_problem"]],
                })
                pp_tok = pp_r.get("pp_tokens", [])
                if pp_tok and n_key not in pp_tokens_dedup:
                    pp_tokens_dedup[n_key] = [round(v, 1) for v in pp_tok]
            cond_methods[name] = {
                "family": method_families[name],
                "cost_type": method_cost_type[name],
                "entries": summary_entries,
            }
            cond_methods_detail[name] = {
                "pp_tokens_by_n_samples": pp_tokens_dedup,
                "entries": detail_entries,
            }
        return cond_methods, cond_methods_detail

    def _hoist(all_cond_data, all_cond_detail):
        """Extract shared / per-rc methods from conditions."""
        shared_m, shared_d = {}, {}
        per_rc_m, per_rc_d = {}, {}
        first = next(iter(all_cond_data.values()), None)
        first_d = next(iter(all_cond_detail.values()), None)
        if not first:
            return shared_m, shared_d, per_rc_m, per_rc_d
        for name, md in list(first["methods"].items()):
            if md["cost_type"] in SHARED_COST_TYPES:
                shared_m[name] = md
                if first_d and name in first_d["methods"]:
                    shared_d[name] = first_d["methods"][name]
        seen_rc = set()
        for cl, cd in all_cond_data.items():
            rc = str(cd["regen_count"])
            if rc in seen_rc:
                continue
            seen_rc.add(rc)
            rm, rd = {}, {}
            for name, md in cd["methods"].items():
                if md["cost_type"] in K_ONLY_COST_TYPES:
                    rm[name] = md
                    cdet = all_cond_detail.get(cl, {})
                    if "methods" in cdet and name in cdet["methods"]:
                        rd[name] = cdet["methods"][name]
            if rm:
                per_rc_m[rc] = rm
                per_rc_d[rc] = rd
        hoisted = set(shared_m) | {n for v in per_rc_m.values() for n in v}
        for cd in all_cond_data.values():
            cd["methods"] = {n: m for n, m in cd["methods"].items()
                             if n not in hoisted}
        for cdet in all_cond_detail.values():
            cdet["methods"] = {n: m for n, m in cdet["methods"].items()
                               if n not in hoisted}
        return shared_m, shared_d, per_rc_m, per_rc_d

    # Build per-condition data for both modes
    tok_all_cond = {}
    tok_all_detail = {}
    samp_all_cond = {}
    samp_all_detail = {}
    dense_all_cond = {}
    dense_all_detail = {}
    last_problems_cond = problems

    for cond_tag, cond_rc in conditions:
        cond_label = f"{cond_tag}_x{cond_rc}"
        print(f"\n{'='*60}")
        print(f"Phase 4: {cond_label}")
        print(f"{'='*60}")

        problems_cond, _ = load_data(
            args.jsonl_dir, cond_tag, cond_rc, init_data)
        if args.problems:
            problems_cond = [p for p in problems_cond
                             if p["pnum"] in set(args.problems)]
        apply_verbal_cap(problems_cond, args.verbal_max_tokens)
        if marker_data:
            attach_marker_data(problems_cond, marker_data)
        if cond_rc in branching_data_by_rc:
            attach_branching_data(
                problems_cond, branching_data_by_rc[cond_rc])
        last_problems_cond = problems_cond

        # ── Prefix evaluation ──
        prefix_tok: Dict[int, dict] = {tp: {} for tp in active_tp}
        prefix_samp: Dict[int, dict] = {sp: {} for sp in args.sample_points}
        prefix_dense: Dict[int, dict] = {}
        if prefix_factories:
            t0 = time.time()
            curves = evaluate_curves(
                problems_cond,
                init_voter_factories=[],
                prefix_voter_factories=prefix_factories,
                token_points_sparse=active_tp,
                token_points_dense=dense_tp_arg,
                sample_points=args.sample_points,
                n_trials=args.trials,
                progress_label=f"prefix {cond_label}",
            )
            print(f"  [prefix]  done in {time.time() - t0:.1f}s")
            for tp in active_tp:
                prefix_tok[tp] = curves["token_budget"].get(tp, {})
            for sp in args.sample_points:
                prefix_samp[sp] = curves["sample_count"].get(sp, {})
            prefix_dense = curves["token_budget_dense"]

        # ── Merge token-budget results ──
        merged_tok = {}
        for tp in active_tp:
            merged_tok[tp] = {}
            merged_tok[tp].update(shared_tok.get(tp, {}))
            merged_tok[tp].update(marker_tok.get(tp, {}))
            if cond_rc in branching_tok_by_rc:
                merged_tok[tp].update(
                    branching_tok_by_rc[cond_rc].get(tp, {}))
            merged_tok[tp].update(prefix_tok.get(tp, {}))

        tok_results, tok_pp = _build_entries(merged_tok, "token_point")
        tok_results = _reorder(tok_results)
        tok_pp = _reorder(tok_pp)

        # ── Merge sample-count results ──
        merged_samp = {}
        for sp in args.sample_points:
            merged_samp[sp] = {}
            merged_samp[sp].update(shared_samp.get(sp, {}))
            merged_samp[sp].update(marker_samp.get(sp, {}))
            if cond_rc in branching_samp_by_rc:
                merged_samp[sp].update(
                    branching_samp_by_rc[cond_rc].get(sp, {}))
            merged_samp[sp].update(prefix_samp.get(sp, {}))

        samp_results, samp_pp = _build_entries(merged_samp, "sample_point")
        samp_results = _reorder(samp_results)
        samp_pp = _reorder(samp_pp)

        # ── Merge dense token-budget results ──
        dense_tok: Dict[int, dict] = {}
        if run_dense:
            dense_tps = sorted(set(shared_dense) | set(marker_dense)
                               | set(branching_dense_by_rc.get(cond_rc, {}))
                               | set(prefix_dense))
            for tp in dense_tps:
                dense_tok[tp] = {}
                dense_tok[tp].update(shared_dense.get(tp, {}))
                dense_tok[tp].update(marker_dense.get(tp, {}))
                if cond_rc in branching_dense_by_rc:
                    dense_tok[tp].update(
                        branching_dense_by_rc[cond_rc].get(tp, {}))
                dense_tok[tp].update(prefix_dense.get(tp, {}))

        # ── Print summary ──
        if active_tp:
            max_tp = active_tp[-1]
            for name in tok_results:
                if tok_results[name]:
                    r = tok_results[name][-1]
                    print(f"  tp={max_tp:>9,}  {name}={r['acc']:.4f}")
        if args.sample_points:
            max_sp = args.sample_points[-1]
            for name in samp_results:
                if samp_results[name]:
                    r = samp_results[name][-1]
                    print(f"  n={max_sp:>5}      {name}={r['acc']:.4f}")

        # ── N labels ──
        n_labels: Dict[str, str] = {}
        n_label_by_ct: Dict[str, str] = {}
        n_label_by_ct["init"] = n_label(
            [len(p["init_answers"]) for p in problems_cond])
        n_label_by_ct["prefix"] = n_label(
            [len(p["groups"]) for p in problems_cond])
        n_label_by_ct["esc"] = n_label_by_ct["init"]
        n_label_by_ct["verbal_0_100"] = n_label(
            [len(p["init_verbal_0_100_confs"]) for p in problems_cond])
        n_label_by_ct["verbal_binary"] = n_label(
            [len(p["init_verbal_binary_confs"]) for p in problems_cond])
        if cond_rc in branching_tok_by_rc:
            bg = [p.get("branching_groups", []) for p in problems_cond]
            nb = [len(g) for g in bg if g]
            n_label_by_ct["branching"] = n_label(nb) if nb else ""
        if marker_data:
            nm = [len(p["marker_groups"]) for p in problems_cond
                  if "marker_groups" in p]
            n_label_by_ct["marker"] = n_label(nm) if nm else ""
        for name in tok_results:
            n_labels[name] = n_label_by_ct.get(
                method_cost_type.get(name, ""), "")

        problem_nums = [p["pnum"] for p in problems_cond]

        tok_cm, tok_cd = _format_cond_methods(
            tok_results, tok_pp, "token_point")
        tok_all_cond[cond_label] = {
            "removal_tag": cond_tag, "regen_count": cond_rc,
            "problem_nums": problem_nums, "n_available": n_labels,
            "methods": tok_cm,
        }
        tok_all_detail[cond_label] = {"methods": tok_cd}

        if args.sample_points:
            samp_cm, samp_cd = _format_cond_methods(
                samp_results, samp_pp, "sample_point")
            samp_all_cond[cond_label] = {
                "removal_tag": cond_tag, "regen_count": cond_rc,
                "problem_nums": problem_nums, "n_available": n_labels,
                "methods": samp_cm,
            }
            samp_all_detail[cond_label] = {"methods": samp_cd}

        # Dense: format and record per-condition
        if run_dense and dense_tok:
            dense_results, dense_pp = _build_entries(dense_tok, "token_point")
            dense_results = _reorder(dense_results)
            dense_pp = _reorder(dense_pp)
            dense_cm, dense_cd = _format_cond_methods(
                dense_results, dense_pp, "token_point")
            dense_all_cond[cond_label] = {
                "removal_tag": cond_tag, "regen_count": cond_rc,
                "problem_nums": problem_nums, "n_available": n_labels,
                "methods": dense_cm,
            }
            dense_all_detail[cond_label] = {"methods": dense_cd}
            # Progress: largest-tp snapshot for a sanity check
            if dense_results.get("standard_mv"):
                last = dense_results["standard_mv"][-1]
                print(f"  dense@{last['token_point']:>10,}: "
                      f"standard_mv={last['acc']:.4f}")

    # ── Hoist shared / per-rc methods ──
    (tok_shared, tok_shared_d,
     tok_per_rc, tok_per_rc_d) = _hoist(tok_all_cond, tok_all_detail)
    if args.sample_points:
        (samp_shared, samp_shared_d,
         samp_per_rc, samp_per_rc_d) = _hoist(
            samp_all_cond, samp_all_detail)
    if run_dense and dense_all_cond:
        (dense_shared, dense_shared_d,
         dense_per_rc, dense_per_rc_d) = _hoist(
            dense_all_cond, dense_all_detail)
    else:
        dense_shared = dense_shared_d = {}
        dense_per_rc = dense_per_rc_d = {}

    meta = {
        "n_problems": len(last_problems_cond),
        "n_init_generations": budget,
        "n_trials": args.trials,
    }

    print(f"\n{'='*60}")
    print(f"All {len(conditions)} conditions complete.")

    if output_path is None:
        print("Output: skipped (filtered run, no --output specified).")
    else:
        # Token-budget section
        tok_section = {
            "shared_methods": tok_shared,
            "per_regen_count": tok_per_rc,
            "conditions": tok_all_cond,
        }
        output = {"meta": meta, "token_budget": tok_section}

        # Dense-grid section
        if run_dense and dense_all_cond:
            dense_section = {
                "shared_methods": dense_shared,
                "per_regen_count": dense_per_rc,
                "conditions": dense_all_cond,
            }
            output["token_budget_dense"] = dense_section

        # Sample-count section
        if args.sample_points:
            samp_section = {
                "shared_methods": samp_shared,
                "per_regen_count": samp_per_rc,
                "conditions": samp_all_cond,
            }
            output["sample_count"] = samp_section

        # One (cost, acc) operating point per early-stopping voter
        # (ESC w sweep, AC threshold sweep).
        if natural_stopping:
            output["natural_stopping"] = natural_stopping

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        # Detail JSON
        tok_detail_sec = {
            "shared_methods": tok_shared_d,
            "per_regen_count": tok_per_rc_d,
            "conditions": tok_all_detail,
        }
        detail_output = {"meta": meta, "token_budget": tok_detail_sec}
        if run_dense and dense_all_detail:
            dense_detail_sec = {
                "shared_methods": dense_shared_d,
                "per_regen_count": dense_per_rc_d,
                "conditions": dense_all_detail,
            }
            detail_output["token_budget_dense"] = dense_detail_sec
        if args.sample_points:
            samp_detail_sec = {
                "shared_methods": samp_shared_d,
                "per_regen_count": samp_per_rc_d,
                "conditions": samp_all_detail,
            }
            detail_output["sample_count"] = samp_detail_sec
        if natural_stopping_detail:
            detail_output["natural_stopping"] = natural_stopping_detail
        # Only the default wmv_result.json keeps the canonical pair name
        # ("wmv_detail.json"). Custom --output paths get a <stem>_detail.json
        # sibling so that ablation runs do not clobber the full-run
        # wmv_detail.json.
        if output_path.name == "wmv_result.json":
            detail_path = output_path.parent / "wmv_detail.json"
        else:
            detail_path = output_path.parent / (output_path.stem + "_detail.json")
        with open(detail_path, "w") as f:
            json.dump(detail_output, f, indent=2)

        print(f"Saved: {output_path}")
        print(f"Saved: {detail_path}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
