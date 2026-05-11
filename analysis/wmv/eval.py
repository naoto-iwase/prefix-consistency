"""Evaluation engine for WMV.

``evaluate_curves`` draws one long sample sequence per
(trial, problem, family) up to ``max(max_budget/min_cost, max_n)`` and
updates voters from ``voters.py`` one sample at a time.  It snapshots
voter state at three grids in a single pass:

    token_points_sparse  ->  ``token_budget``       (e.g. legacy 13 tp)
    token_points_dense   ->  ``token_budget_dense`` (e.g. 401-point grid)
    sample_points        ->  ``sample_count``

Because snapshots are O(1) per voter, adding the dense grid or more
sample counts is effectively free; the cost is dominated by sampling
and voter updates.  Every method exposed in ``wmv.py`` has a voter
class, so ``evaluate_curves`` is the only evaluator the script needs.
"""

import random
import time
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np


class _Voter(Protocol):
    """Structural type for the voters in ``wmv.voters``."""

    def add(self, item) -> None: ...
    def is_correct(self) -> bool: ...


# =====================================================================
# Unified evaluator (one sample sequence per trial)
# =====================================================================

# A voter factory produces a voter bound to one problem.
VoterFactory = Callable[[dict], _Voter]


def _process_sequence(
    rng: random.Random,
    pool_items: list,
    pool_costs: List[int],
    voter_factories: List[Tuple[str, VoterFactory]],
    problem: dict,
    pi: int,
    token_points: List[int],       # sorted union for cost-based snapshots
    sample_points: List[int],      # sorted for count-based snapshots
    token_correct: Dict[str, np.ndarray],
    token_tok_sum: Dict[str, np.ndarray],
    token_nsamp_sum: Dict[str, np.ndarray],
    samp_correct: Dict[str, np.ndarray],
    samp_tok_sum: Dict[str, np.ndarray],
    samp_nsamp_sum: Dict[str, np.ndarray],
    nat_correct: Optional[Dict[str, np.ndarray]] = None,
    nat_cost_sum: Optional[Dict[str, np.ndarray]] = None,
    nat_cost_sq_sum: Optional[Dict[str, np.ndarray]] = None,
    nat_locked: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Sample one sequence; snapshot at token budgets and sample counts."""
    n_pool = len(pool_items)
    if n_pool == 0 or not voter_factories:
        return
    if not pool_costs:
        return
    min_cost = min(pool_costs)
    if min_cost <= 0:
        return

    max_budget = token_points[-1] if token_points else 0
    max_n = sample_points[-1] if sample_points else 0
    n_max_budget = max_budget // min_cost if max_budget > 0 else 0
    n_max = max(n_max_budget, max_n)
    if n_max <= 0:
        return

    picks = rng.choices(range(n_pool), k=n_max)
    sample_costs = [pool_costs[i] for i in picks]

    voters = [(name, factory(problem)) for name, factory in voter_factories]
    n_tp = len(token_points)
    n_sp = len(sample_points)

    # Optional zero-sample snapshots for sample_points[0] == 0.
    j_sp = 0
    while j_sp < n_sp and sample_points[j_sp] == 0:
        for name, voter in voters:
            if voter.is_correct():
                samp_correct[name][j_sp, pi] += 1
            # tok_sum and nsamp_sum remain 0
        j_sp += 1

    cum_cost = 0
    j_tp = 0
    track_nat = nat_correct is not None
    nat_voter_idx = [i for i, (_, v) in enumerate(voters)
                     if hasattr(v, "lock_idx")] if track_nat else []
    for k in range(n_max):
        # State before the k-th add: k samples in, cum_cost = total cost.
        new_cum = cum_cost + sample_costs[k]
        # Budget snapshots with cum_cost <= tp < new_cum: snapshot state@k.
        while j_tp < n_tp and token_points[j_tp] < new_cum:
            for name, voter in voters:
                if voter.is_correct():
                    token_correct[name][j_tp, pi] += 1
                token_tok_sum[name][j_tp, pi] += cum_cost
                token_nsamp_sum[name][j_tp, pi] += k
            j_tp += 1
        # Add the k-th sample.
        for _, voter in voters:
            voter.add(pool_items[picks[k]])
        cum_cost = new_cum
        # State after the k-th add: (k+1) samples in.
        while j_sp < n_sp and sample_points[j_sp] == k + 1:
            for name, voter in voters:
                if voter.is_correct():
                    samp_correct[name][j_sp, pi] += 1
                samp_tok_sum[name][j_sp, pi] += cum_cost
                samp_nsamp_sum[name][j_sp, pi] += (k + 1)
            j_sp += 1
        # Early exit: outputs are unchanged once snapshots and locks are done.
        if j_tp >= n_tp and j_sp >= n_sp:
            if not track_nat or all(
                voters[i][1].lock_idx is not None for i in nat_voter_idx
            ):
                break

    # Remaining budget snapshots: budget >= full cum_cost, use all n_max.
    while j_tp < n_tp:
        for name, voter in voters:
            if voter.is_correct():
                token_correct[name][j_tp, pi] += 1
            token_tok_sum[name][j_tp, pi] += cum_cost
            token_nsamp_sum[name][j_tp, pi] += n_max
        j_tp += 1
    # Remaining sample snapshots: count exceeds n_max, cap at n_max.
    while j_sp < n_sp:
        for name, voter in voters:
            if voter.is_correct():
                samp_correct[name][j_sp, pi] += 1
            samp_tok_sum[name][j_sp, pi] += cum_cost
            samp_nsamp_sum[name][j_sp, pi] += n_max
        j_sp += 1

    # Natural stopping for voters that expose ``lock_idx``: record the
    # cost at lock (or full sequence cost if never locked) and the
    # answer's correctness at that point.
    if (nat_correct is not None and nat_cost_sum is not None
            and nat_locked is not None):
        cum_costs = np.cumsum(sample_costs) if sample_costs else None
        for name, voter in voters:
            if not hasattr(voter, "lock_idx"):
                continue
            lock_idx = voter.lock_idx
            if lock_idx is not None and cum_costs is not None:
                cost = float(cum_costs[lock_idx - 1])
                nat_locked[name][pi] += 1
            else:
                cost = float(cum_cost)
            if voter.is_correct():
                nat_correct[name][pi] += 1
            nat_cost_sum[name][pi] += cost
            if nat_cost_sq_sum is not None:
                nat_cost_sq_sum[name][pi] += cost * cost


def evaluate_curves(
    problems: List[dict],
    init_voter_factories: List[Tuple[str, VoterFactory]],
    prefix_voter_factories: List[Tuple[str, VoterFactory]],
    token_points_sparse: List[int],
    token_points_dense: List[int],
    sample_points: List[int],
    n_trials: int,
    branching_voter_factories: Optional[List[Tuple[str, VoterFactory]]] = None,
    marker_voter_factories: Optional[List[Tuple[str, VoterFactory]]] = None,
    verbal_0_100_voter_factories: Optional[List[Tuple[str, VoterFactory]]] = None,
    verbal_binary_voter_factories: Optional[List[Tuple[str, VoterFactory]]] = None,
    seed: int = 42,
    progress_label: str = "curves",
) -> Dict[str, dict]:
    """Unified evaluator.

    One sample sequence per (trial, problem, family); snapshot at every
    ``token_points_sparse``, ``token_points_dense`` and ``sample_points``
    point during that pass.

    Returns {
        'token_budget':        {tp: {name: (acc, ci, pp_acc, pp_tok, mean_n)}},
        'token_budget_dense':  {tp: {name: ...}},
        'sample_count':        {sp: {name: ...}},
    }

    ``token_budget`` is populated only at points in ``token_points_sparse``;
    ``token_budget_dense`` only at points in ``token_points_dense``.  If a
    point appears in both, both sections get the same value.
    """
    if branching_voter_factories is None:
        branching_voter_factories = []
    if marker_voter_factories is None:
        marker_voter_factories = []
    if verbal_0_100_voter_factories is None:
        verbal_0_100_voter_factories = []
    if verbal_binary_voter_factories is None:
        verbal_binary_voter_factories = []

    token_points = sorted(set(token_points_sparse) | set(token_points_dense))
    sparse_set = set(token_points_sparse)
    dense_set = set(token_points_dense)
    sample_points = sorted(set(sample_points))

    n_tp = len(token_points)
    n_sp = len(sample_points)
    n_problems = len(problems)

    all_factories = (init_voter_factories + prefix_voter_factories
                     + branching_voter_factories + marker_voter_factories
                     + verbal_0_100_voter_factories
                     + verbal_binary_voter_factories)

    def _alloc(shape_sz):
        return {
            "correct": {name: np.zeros((shape_sz, n_problems), dtype=np.int64)
                        for name, _ in all_factories},
            "tok_sum": {name: np.zeros((shape_sz, n_problems), dtype=np.float64)
                        for name, _ in all_factories},
            "nsamp_sum": {name: np.zeros((shape_sz, n_problems), dtype=np.float64)
                          for name, _ in all_factories},
        }

    # ``_alloc(0)`` returns dicts of (0, n_problems) arrays, which is what
    # we want when the corresponding grid is empty -- the inner-loop
    # ``while j_tp < n_tp`` / ``while j_sp < n_sp`` guards never enter, so
    # the snapshots are skipped without further branching.
    tok = _alloc(n_tp)
    samp = _alloc(n_sp)

    # Per-method, per-problem accumulators for voters with ``lock_idx``.
    nat = {
        "correct": {name: np.zeros(n_problems, dtype=np.int64)
                    for name, _ in all_factories},
        "cost_sum": {name: np.zeros(n_problems, dtype=np.float64)
                     for name, _ in all_factories},
        "cost_sq_sum": {name: np.zeros(n_problems, dtype=np.float64)
                        for name, _ in all_factories},
        "locked": {name: np.zeros(n_problems, dtype=np.int64)
                   for name, _ in all_factories},
    }

    init_rng = random.Random(seed)
    v0100_rng = random.Random(seed + 1)
    vbin_rng = random.Random(seed + 2)
    prefix_rng = random.Random(seed + 3)
    branching_rng = random.Random(seed + 4)
    marker_rng = random.Random(seed + 5)

    def _run_verbal_pass(factories, rng, conf_fields, overhead_field,
                         problem, pi):
        """Sample one verbal-cost sequence for ``problem`` and update
        snapshots via ``_process_sequence``.

        ``conf_fields`` lists the ``problem`` keys whose length defines
        the pool size; the first present key wins.  This is what couples
        ``p_true_raw`` to the verbal_binary cost vector: when only
        ``init_p_true_confs`` is populated, it stands in for
        ``init_verbal_binary_confs`` as the pool descriptor.
        """
        if not factories:
            return
        pool_n = 0
        for field in conf_fields:
            pool_n = len(problem.get(field) or [])
            if pool_n > 0:
                break
        if pool_n == 0:
            return
        base = problem["init_tokens"][:pool_n]
        overhead = (problem.get(overhead_field) or [0] * pool_n)[:pool_n]
        costs = [t + o for t, o in zip(base, overhead)]
        _process_sequence(
            rng, list(range(pool_n)), costs,
            factories, problem, pi,
            token_points, sample_points,
            tok["correct"], tok["tok_sum"], tok["nsamp_sum"],
            samp["correct"], samp["tok_sum"], samp["nsamp_sum"],
            nat["correct"], nat["cost_sum"], nat["cost_sq_sum"], nat["locked"],
        )

    t_start = time.time()
    report_every = max(1, n_trials // 10)
    for trial in range(n_trials):
        for pi, p in enumerate(problems):
            if init_voter_factories:
                _process_sequence(
                    init_rng,
                    list(range(len(p["init_answers"]))),
                    p["init_tokens"],
                    init_voter_factories, p, pi,
                    token_points, sample_points,
                    tok["correct"], tok["tok_sum"], tok["nsamp_sum"],
                    samp["correct"], samp["tok_sum"], samp["nsamp_sum"],
                    nat["correct"], nat["cost_sum"], nat["cost_sq_sum"], nat["locked"],
                )
            if prefix_voter_factories and p.get("groups"):
                groups = p["groups"]
                _process_sequence(
                    prefix_rng,
                    groups,
                    [g["group_tokens"] for g in groups],
                    prefix_voter_factories, p, pi,
                    token_points, sample_points,
                    tok["correct"], tok["tok_sum"], tok["nsamp_sum"],
                    samp["correct"], samp["tok_sum"], samp["nsamp_sum"],
                    nat["correct"], nat["cost_sum"], nat["cost_sq_sum"], nat["locked"],
                )
            if branching_voter_factories and p.get("branching_groups"):
                groups = p["branching_groups"]
                _process_sequence(
                    branching_rng,
                    groups,
                    [g["group_tokens"] for g in groups],
                    branching_voter_factories, p, pi,
                    token_points, sample_points,
                    tok["correct"], tok["tok_sum"], tok["nsamp_sum"],
                    samp["correct"], samp["tok_sum"], samp["nsamp_sum"],
                    nat["correct"], nat["cost_sum"], nat["cost_sq_sum"], nat["locked"],
                )
            if marker_voter_factories and p.get("marker_groups"):
                groups = p["marker_groups"]
                _process_sequence(
                    marker_rng,
                    groups,
                    [g["group_tokens"] for g in groups],
                    marker_voter_factories, p, pi,
                    token_points, sample_points,
                    tok["correct"], tok["tok_sum"], tok["nsamp_sum"],
                    samp["correct"], samp["tok_sum"], samp["nsamp_sum"],
                    nat["correct"], nat["cost_sum"], nat["cost_sq_sum"], nat["locked"],
                )
            _run_verbal_pass(
                verbal_0_100_voter_factories, v0100_rng,
                ["init_verbal_0_100_confs"],
                "init_verbal_0_100_min_tokens", p, pi)
            _run_verbal_pass(
                verbal_binary_voter_factories, vbin_rng,
                ["init_verbal_binary_confs", "init_p_true_confs"],
                "init_binary_query_min_tokens", p, pi)
        done = trial + 1
        if done == n_trials or done % report_every == 0:
            elapsed = time.time() - t_start
            eta = elapsed * (n_trials - done) / done if done else 0.0
            print(f"  [{progress_label}] trial {done:>4}/{n_trials}  "
                  f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s",
                  flush=True)

    def _build(corr_row, tsum_row, nsum_row):
        p_hat = corr_row / n_trials
        mean_acc = float(np.mean(p_hat))
        var_acc = float(np.sum(p_hat * (1.0 - p_hat))
                        / (n_trials * n_problems ** 2))
        pp_tok = tsum_row / n_trials
        mean_n = float(np.mean(nsum_row / n_trials))
        return (mean_acc, 2.0 * np.sqrt(var_acc),
                [float(v) for v in p_hat],
                [float(v) for v in pp_tok],
                mean_n)

    token_budget: Dict[int, Dict[str, Tuple]] = {}
    token_budget_dense: Dict[int, Dict[str, Tuple]] = {}
    for j, tp in enumerate(token_points):
        row = {}
        for name, _ in all_factories:
            row[name] = _build(tok["correct"][name][j],
                                tok["tok_sum"][name][j],
                                tok["nsamp_sum"][name][j])
        if tp in sparse_set:
            token_budget[tp] = row
        if tp in dense_set:
            token_budget_dense[tp] = row

    sample_count: Dict[int, Dict[str, Tuple]] = {}
    for j, sp in enumerate(sample_points):
        row = {}
        for name, _ in all_factories:
            row[name] = _build(samp["correct"][name][j],
                                samp["tok_sum"][name][j],
                                samp["nsamp_sum"][name][j])
        sample_count[sp] = row

    natural_stopping: Dict[str, dict] = {}
    natural_stopping_detail: Dict[str, dict] = {}
    for name, _ in all_factories:
        if not nat["cost_sum"][name].any():
            continue
        p_hat = nat["correct"][name] / n_trials
        mean_acc = float(np.mean(p_hat))
        # Wald-Bernoulli, problems fixed.
        var_acc = float(np.sum(p_hat * (1.0 - p_hat))
                        / (n_trials * n_problems ** 2))
        per_problem_cost = nat["cost_sum"][name] / n_trials
        per_problem_cost_sq = nat["cost_sq_sum"][name] / n_trials
        per_problem_var_pop = np.maximum(
            per_problem_cost_sq - per_problem_cost ** 2, 0.0)
        if n_trials > 1:
            per_problem_cost_var = (per_problem_var_pop
                                    * n_trials / (n_trials - 1))
        else:
            per_problem_cost_var = per_problem_var_pop
        var_mean_cost = (float(np.sum(per_problem_cost_var / n_trials))
                         / (n_problems ** 2))
        per_problem_lock = nat["locked"][name] / n_trials
        natural_stopping[name] = {
            "mean_cost": float(np.mean(per_problem_cost)),
            "cost_ci": 2.0 * float(np.sqrt(var_mean_cost)),
            "acc": mean_acc,
            "ci": 2.0 * float(np.sqrt(var_acc)),
            "lock_rate": float(np.mean(per_problem_lock)),
        }
        natural_stopping_detail[name] = {
            "per_problem_cost": [float(v) for v in per_problem_cost],
            "per_problem_acc": [float(v) for v in p_hat],
            "per_problem_lock_rate": [float(v) for v in per_problem_lock],
        }

    return {
        "token_budget": token_budget,
        "token_budget_dense": token_budget_dense,
        "sample_count": sample_count,
        "natural_stopping": natural_stopping,
        "natural_stopping_detail": natural_stopping_detail,
    }
