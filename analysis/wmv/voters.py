"""Stateful voters fed one sample at a time.

Each class takes a ``problem`` dict in the constructor, exposes an
``add(item)`` method that accepts either a sample index (init family)
or a group dict (prefix / branching / marker family), and a
``winner()`` method that returns the currently predicted answer or
None if no samples have been added yet.  ``is_correct()`` returns
``winner() == problem["gold"]``.

``evaluate_curves`` calls ``add`` and ``is_correct`` to compute voter
predictions at many budget points within a single sample sequence;
update is O(1) (or O(log n) for the top-k filter, O(|group|) for
group-family voters), so the whole trial is near-linear in the
number of samples regardless of how dense the grid is.

Softmax / CISC-softmax variants store per-answer running sums of
``exp(c/T)`` (the softmax denominator is constant in the answer, so
dropping it preserves the argmax).  Top-k-filtered DeepConf variants
maintain two heaps (currently-active top-k and inactive remainder)
so each add costs O(log n) and the votes table reflects the active
set after every snapshot.  ESC keeps a small window buffer plus a
Counter of completed-window contents and an "early stop" flag for
unanimous windows.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from heapq import heappop, heappush, heapreplace
from typing import Callable

import numpy as np
from scipy.special import betainc


# ── Shared helpers ───────────────────────────────────────────────────

def get_confs(p: dict, conf_key: str):
    """Get a confidence array from a problem dict.

    ``conf_key`` may be either a top-level key in ``p`` or a key inside
    ``p["init_confidences"]``.  ``None`` entries (missing confidence)
    are replaced with the median of the non-None entries so that every
    answer participates in voting with a neutral weight regardless of
    the confidence scale.
    """
    if conf_key in p:
        raw = p[conf_key]
    else:
        raw = p["init_confidences"][conf_key]
    if not any(v is None for v in raw):
        return raw
    valid = [v for v in raw if v is not None]
    fill = float(np.median(valid)) if valid else 0.0
    return [v if v is not None else fill for v in raw]


# ── Shared base ──────────────────────────────────────────────────────

class _VoterBase:
    """Common ``winner()`` for classes that accumulate a float vote dict."""

    def __init__(self, problem: dict):
        self.p = problem
        self.gold = problem["gold"]
        self.votes: dict = defaultdict(float)

    def winner(self):
        if not self.votes:
            return None
        return max(self.votes, key=lambda a: self.votes[a])

    def is_correct(self) -> bool:
        w = self.winner()
        return w is not None and w == self.gold


# ── Init-family (add(idx)) ──────────────────────────────────────────

class StandardVoter(_VoterBase):
    """Unweighted majority vote over init answers."""

    def add(self, idx: int) -> None:
        self.votes[self.p["init_answers"][idx]] += 1.0


class WeightedVoter(_VoterBase):
    """Weighted MV: ``w = weight_fn(conf[i])`` on the init answer."""

    def __init__(self, problem: dict, conf_key: str, weight_fn: Callable):
        super().__init__(problem)
        self.confs = get_confs(problem, conf_key)
        self.weight_fn = weight_fn

    def add(self, idx: int) -> None:
        w = self.weight_fn(self.confs[idx])
        self.votes[self.p["init_answers"][idx]] += w


class OracleInitVoter:
    """Oracle: correct if any sampled init answer equals gold."""

    def __init__(self, problem: dict):
        self.p = problem
        self.gold = problem["gold"]
        self._seen_correct = False

    def add(self, idx: int) -> None:
        if not self._seen_correct and self.p["init_answers"][idx] == self.gold:
            self._seen_correct = True

    def winner(self):
        return self.gold if self._seen_correct else None

    def is_correct(self) -> bool:
        return self._seen_correct


# ── Group-family (add(group)) ────────────────────────────────────────

class PrefixWeightedVoter(_VoterBase):
    """PC-WMV: votes[a] += weight_fn(c_i(a)) per distinct a per group."""

    def __init__(self, problem: dict, weight_fn: Callable):
        super().__init__(problem)
        self.weight_fn = weight_fn

    def add(self, group: dict) -> None:
        answers = group["answers"]
        if not answers:
            return
        counts = Counter(answers)
        total = len(answers)
        for ans, c in counts.items():
            w = self.weight_fn(c / total)
            if w > 0:
                self.votes[ans] += w


class PrefixUnanimousVoter:
    """Tiered unanimity MV.

    With ``K+1`` answers per group, a group's top in-group count ``c``
    is some ``c in {1, ..., K+1}``.  Each group contributes ``+1`` to
    its top answer(s) (ties at the top all contribute) inside bucket
    ``c``; the winner is ``argmax`` of the largest non-empty bucket.
    This steps through ``c = K+1, K, ..., 1`` instead of jumping from
    full unanimity (``K+1``) straight to pooled MV (``1``).
    """

    def __init__(self, problem: dict):
        self.p = problem
        self.gold = problem["gold"]
        self.buckets: dict[int, Counter] = defaultdict(Counter)
        self.max_k: int = 0

    def add(self, group: dict) -> None:
        answers = group["answers"]
        if not answers:
            return
        counts = Counter(answers)
        top_c = max(counts.values())
        for a, c in counts.items():
            if c == top_c:
                self.buckets[top_c][a] += 1
        if top_c > self.max_k:
            self.max_k = top_c

    def winner(self):
        if self.max_k == 0:
            return None
        return self.buckets[self.max_k].most_common(1)[0][0]

    def is_correct(self) -> bool:
        w = self.winner()
        return w is not None and w == self.gold


class MultiPointsVoter:
    """Uniform MV over pooled answers from sampled groups (branching/marker)."""

    def __init__(self, problem: dict):
        self.p = problem
        self.gold = problem["gold"]
        self.votes: Counter = Counter()

    def add(self, group: dict) -> None:
        self.votes.update(group["answers"])

    def winner(self):
        if not self.votes:
            return None
        return self.votes.most_common(1)[0][0]

    def is_correct(self) -> bool:
        w = self.winner()
        return w is not None and w == self.gold


class OracleGroupVoter:
    """Oracle: correct if any sampled group contains gold."""

    def __init__(self, problem: dict):
        self.p = problem
        self.gold = problem["gold"]
        self._seen_correct = False

    def add(self, group: dict) -> None:
        if not self._seen_correct:
            if any(a == self.gold for a in group["answers"]):
                self._seen_correct = True

    def winner(self):
        return self.gold if self._seen_correct else None

    def is_correct(self) -> bool:
        return self._seen_correct


# ── Init-family with non-trivial state ──────────────────────────────

class SoftmaxVoter(_VoterBase):
    """CISC-softmax MV: weight ``w_i = softmax(c_i / T)`` per sample.

    The softmax denominator ``Z = sum_j exp(c_j / T)`` is constant in
    the answer, so it drops out of ``argmax_a sum_{i: ans_i==a} w_i``.
    We accumulate the unnormalized sums ``S[a] += exp(c/T)`` and take
    ``argmax_a S[a]``; equivalent to ``softmax_vote`` for argmax with
    O(1) per add and no numerical-stability scratchpad.
    """

    def __init__(self, problem: dict, conf_key: str, temp: float):
        super().__init__(problem)
        self.confs = get_confs(problem, conf_key)
        self.temp = temp

    def add(self, idx: int) -> None:
        w = math.exp(self.confs[idx] / self.temp)
        self.votes[self.p["init_answers"][idx]] += w


class TopFilterVoter:
    """DeepConf top-k filter: keep top ``top_pct`` % of samples by conf,
    then confidence-weighted MV among those.

    The active set is the top ``ceil(n * top_pct / 100)`` samples by
    confidence at each step.  As ``n`` grows, ``keep_k`` may grow by 1;
    with each add we either: (a) the new sample beats the worst active
    and becomes part of the top (evicting the old worst when ``keep_k``
    is unchanged, or just joining when ``keep_k`` grew), or (b) it does
    not beat the worst active, in which case it goes to ``inactive``
    unless ``keep_k`` grew, in which case the higher of the new sample
    and the best inactive is promoted.

    State:
      ``active``   min-heap of ``(conf, -seq, ans)`` for currently active
                   samples; smallest-conf-then-latest-seq at the top, so
                   ``heapreplace`` evicts the latest entry on conf ties
                   (matches Python's stable-sort-reverse: at conf ties
                   the EARLIEST insertion is kept).
      ``inactive`` min-heap of ``(-conf, seq, ans)`` for samples below
                   the threshold; top is largest-conf then earliest-seq,
                   matching the order in which a stable-sorted descending
                   list would promote them as ``keep_k`` grows.
      ``votes``   running ``defaultdict(float)`` over the active set
                   only; updated on every set membership change so that
                   ``winner()`` is just ``argmax`` over ``votes``.

    Each ``add`` is O(log n).
    """

    def __init__(self, problem: dict, conf_key: str, top_pct: float):
        self.p = problem
        self.gold = problem["gold"]
        self.confs = get_confs(problem, conf_key)
        self.ans = problem["init_answers"]
        self.top_pct = top_pct
        self.n = 0
        self.keep_k = 0
        self.seq = 0
        self.active: list = []
        self.inactive: list = []
        self.votes: dict = defaultdict(float)

    def _new_keep_k(self) -> int:
        return max(1, math.ceil(self.n * self.top_pct / 100))

    def add(self, idx: int) -> None:
        self.seq += 1
        self.n += 1
        new_keep = self._new_keep_k()
        c = self.confs[idx]
        a = self.ans[idx]
        new_entry = (c, -self.seq, a)

        if not self.active:
            heappush(self.active, new_entry)
            self.votes[a] += c
            self.keep_k = new_keep
            return

        if new_entry > self.active[0]:
            # New beats the worst active.
            if new_keep > self.keep_k:
                heappush(self.active, new_entry)
                self.votes[a] += c
            else:
                ev_c, ev_neg_seq, ev_a = heapreplace(self.active, new_entry)
                self.votes[ev_a] -= ev_c
                self.votes[a] += c
                heappush(self.inactive, (-ev_c, -ev_neg_seq, ev_a))
        else:
            # New does not beat the worst active (strictly worse, or tied
            # with later seq).
            if new_keep > self.keep_k:
                if self.inactive:
                    inact_neg_c, inact_seq, inact_a = self.inactive[0]
                    inact_c = -inact_neg_c
                    # On conf tie the inactive entry has an earlier seq
                    # (it was inserted earlier), so it wins the
                    # stable-sort tie-break: promote inactive when
                    # ``inact_c >= c``.
                    if inact_c >= c:
                        heappop(self.inactive)
                        heappush(self.active,
                                 (inact_c, -inact_seq, inact_a))
                        self.votes[inact_a] += inact_c
                        heappush(self.inactive, (-c, self.seq, a))
                    else:
                        heappush(self.active, new_entry)
                        self.votes[a] += c
                else:
                    heappush(self.active, new_entry)
                    self.votes[a] += c
            else:
                heappush(self.inactive, (-c, self.seq, a))

        self.keep_k = new_keep

    def winner(self):
        # Skip zero-vote entries (left over after eviction subtracted
        # back to 0) so they do not win on tie-break.
        best_a = None
        best_v = 0.0
        for ans, v in self.votes.items():
            if v > best_v:
                best_v = v
                best_a = ans
        return best_a

    def is_correct(self) -> bool:
        w = self.winner()
        return w is not None and w == self.gold


# =====================================================================
# Sweep voters: one core per (trial, problem) shared by N voters; each
# voter delegates ``add`` to the core, deduplicated via ``_n_local``.
# =====================================================================


class ACSweepCore:
    """AC (Aggarwal et al., EMNLP 2023) over an ascending threshold sweep."""

    def __init__(self, problem, conf_threshs):
        self.gold = problem["gold"]
        self.ans = problem["init_answers"]
        self.threshs = list(conf_threshs)
        n = len(self.threshs)
        self.preds = Counter()
        self.lock_idx = [None] * n
        self.locked_winner = [None] * n
        self.n_added = 0

    def add(self, idx):
        self.n_added += 1
        if self.lock_idx[-1] is not None:
            return
        self.preds[self.ans[idx]] += 1
        top = self.preds.most_common(2)
        a, b = top[0][1], (top[1][1] if len(top) > 1 else 0)
        prob = 1.0 - float(betainc(a + 1, b + 1, 0.5))
        winner = top[0][0]
        for i, t in enumerate(self.threshs):
            if self.lock_idx[i] is None and prob >= t:
                self.lock_idx[i] = self.n_added
                self.locked_winner[i] = winner

    def current_winner(self):
        return self.preds.most_common(1)[0][0] if self.preds else None


class ACSweepVoter:
    __slots__ = ("_core", "_i", "_n_local")

    def __init__(self, core, i):
        self._core, self._i, self._n_local = core, i, 0

    def add(self, idx):
        self._n_local += 1
        if self._core.n_added < self._n_local:
            self._core.add(idx)

    @property
    def lock_idx(self):
        return self._core.lock_idx[self._i]

    def winner(self):
        if self._core.lock_idx[self._i] is not None:
            return self._core.locked_winner[self._i]
        return self._core.current_winner()

    def is_correct(self):
        w = self.winner()
        return w is not None and w == self._core.gold


class ESCSweepCore:
    """ESC (Li et al., ICLR 2024) over a list of window sizes. Each
    window keeps its own non-overlapping buffer and intermediate
    ``Counter`` (the partial-last-window batch rule is omitted)."""

    def __init__(self, problem, window_sizes):
        self.gold = problem["gold"]
        self.ans = problem["init_answers"]
        self.windows = list(window_sizes)
        n = len(self.windows)
        self.buffers = [[] for _ in range(n)]
        self.preds = [Counter() for _ in range(n)]
        self.lock_idx = [None] * n
        self.locked_winner = [None] * n
        self.n_added = 0

    def add(self, idx):
        self.n_added += 1
        a = self.ans[idx]
        for i, w in enumerate(self.windows):
            if self.lock_idx[i] is not None:
                continue
            buf = self.buffers[i]
            buf.append(a)
            if len(buf) < w:
                continue
            if len(set(buf)) == 1:
                self.lock_idx[i] = self.n_added
                self.locked_winner[i] = buf[0]
                self.preds[i] = Counter([buf[0]])
            else:
                self.preds[i].update(buf)
            self.buffers[i] = []

    def current_winner(self, i):
        if self.lock_idx[i] is not None:
            return self.locked_winner[i]
        preds, buf = self.preds[i], self.buffers[i]
        if not preds and not buf:
            return None
        if not buf:
            return preds.most_common(1)[0][0]
        combined = Counter(preds)
        combined.update(buf)
        return combined.most_common(1)[0][0]


class ESCSweepVoter:
    __slots__ = ("_core", "_i", "_n_local")

    def __init__(self, core, i):
        self._core, self._i, self._n_local = core, i, 0

    def add(self, idx):
        self._n_local += 1
        if self._core.n_added < self._n_local:
            self._core.add(idx)

    @property
    def lock_idx(self):
        return self._core.lock_idx[self._i]

    def winner(self):
        return self._core.current_winner(self._i)

    def is_correct(self):
        w = self.winner()
        return w is not None and w == self._core.gold
