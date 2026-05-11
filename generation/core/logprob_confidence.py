"""
DeepConf baseline: token-level confidence and trace-level aggregation.

Computes confidence metrics from top-k logprobs saved as .npy files
during generation. Metrics include mean confidence, bottom-10%
confidence, tail confidence, first-token KL divergence, and
block-minimum confidence.

Library module: no CLI, no I/O beyond .npy loading. Used by convert_to_jsonl.py.

Paper: Fu et al., "Deep Think with Confidence" (ICLR 2026)
  https://arxiv.org/abs/2508.15260
  https://github.com/facebookresearch/deepconf

See also: Kang et al., "Self-Certainty" (NeurIPS 2025)
  https://arxiv.org/abs/2502.18581
"""

import math
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer

from .config import build_model_config

# Header keys written by enrich.py (step 2)
CONFIDENCE_HEADERS = {
    "mean_conf": "Mean Conf",
    "bottom10_conf": "Bottom10 Conf",
    "tail_conf": "Tail Conf",
    "first_token_conf": "First Token Conf",
    "block_min_conf": "Block Min Conf",
}

# =====================================================================
# Token-level confidence
# =====================================================================

def token_confidence(logprobs: List[float]) -> float:
    """DeepConf Token Confidence: C_i = -(1/k) * sum(log P_top-k).

    Parameters
    ----------
    logprobs : list of float
        Top-k log probabilities for a single token position.

    Returns
    -------
    float
        Token confidence (higher = more certain).
    """
    if not logprobs:
        return 0.0
    return -sum(logprobs) / len(logprobs)


def kl_from_uniform(logprobs: List[float]) -> float:
    """KL(P_normalized || U) where P is the normalized top-k distribution.

    Normalizes top-k probs to sum to 1, then computes KL(P || U).

    Parameters
    ----------
    logprobs : list of float
        Top-k log probabilities for a single token position.

    Returns
    -------
    float
        KL divergence from uniform (higher = more certain).
    """
    if not logprobs:
        return float("inf")
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total == 0:
        return float("inf")
    normalized = [p / total for p in probs]
    k = len(normalized)
    uniform = 1.0 / k
    eps = 1e-10
    kl = 0.0
    for p_i in normalized:
        p_i = max(p_i, eps)
        kl += p_i * math.log(p_i / max(uniform, eps))
    return kl


# =====================================================================
# Trace-level aggregation (from .npy files)
# =====================================================================
# .npy format: shape (T, k) where T = number of generated tokens,
#              k = number of top logprobs (typically 20).
# Each entry is a log probability (negative float).

def compute_mean_confidence(arr: np.ndarray) -> float:
    """Average trace confidence (= Self-Certainty / DeepConf avg).

    C_avg = (1/T) * sum_i C_i where C_i = -(1/k) * sum_j logprob_ij.
    Equivalent to -mean(arr), ignoring non-finite values (-inf from
    tokens whose logprob was not in the top-k).
    """
    if arr.size == 0:
        return float("nan")
    arr64 = arr.astype(np.float64)
    finite_mask = np.isfinite(arr64)
    count = int(np.sum(finite_mask))
    if count == 0:
        return float("nan")
    return -float(np.sum(np.where(finite_mask, arr64, 0.0)) / count)


def compute_bottom10_confidence(arr: np.ndarray, window_size: int = 1024) -> float:
    """Bottom-10% group confidence.

    1. Per-token row mean (nanmean) -> series of length T
    2. Sliding window (stride=1) mean over series (NaN-aware) -> group confidences
    3. Negate, take bottom 10% (ceil) of groups, return their mean.
    """
    if arr.ndim < 2 or arr.shape[0] == 0:
        return float("nan")
    arr64 = arr.astype(np.float64)
    row_values = np.nanmean(arr64, axis=1)
    n = len(row_values)

    if not np.any(np.isfinite(row_values)):
        return float("nan")

    # Truncate window_size for short answers
    ws = window_size
    if ws > n:
        ws = int(n / 2)
    if ws <= 0:
        return float("nan")

    # NaN-aware cumsum with count tracking
    finite_mask = np.isfinite(row_values)
    val_no_nan = np.where(finite_mask, row_values, 0.0)
    cnt = np.where(finite_mask, 1, 0).astype(np.int64)
    csum = np.concatenate(([0.0], np.cumsum(val_no_nan)))
    ccnt = np.concatenate(([0], np.cumsum(cnt)))

    window_means = []
    for start in range(0, n - ws + 1):
        end = start + ws
        s = csum[end] - csum[start]
        c = ccnt[end] - ccnt[start]
        if c <= 0:
            continue
        window_means.append(-float(s / c))

    if len(window_means) == 0:
        return float("nan")

    wm = np.asarray(window_means, dtype=np.float64)
    m = wm.size
    k = int(np.ceil(0.10 * m))
    if k <= 0:
        k = 1
    idx = np.argpartition(wm, k - 1)[:k]
    return float(np.mean(wm[idx]))


def compute_tail_confidence(arr: np.ndarray, tail_size: int = 2024) -> float:
    """Tail confidence: average confidence of the last tail_size tokens."""
    if arr.ndim < 2 or arr.shape[0] == 0:
        return float("nan")
    arr64 = arr.astype(np.float64)
    n = arr64.shape[0]
    start = max(0, n - tail_size)
    tail = arr64[start:]
    finite_mask = np.isfinite(tail)
    count = int(np.sum(finite_mask))
    if count == 0:
        return float("nan")
    return -float(np.sum(np.where(finite_mask, tail, 0.0)) / count)


def compute_first_token_confidence(arr: np.ndarray) -> float:
    """First-token confidence: KL divergence of the first generated token.

    Uses the top-k logprobs of the very first token as a proxy for the
    model's initial certainty about the reasoning direction.
    """
    if arr.ndim < 2 or arr.shape[0] == 0:
        return float("nan")
    row = arr[0].tolist()
    finite_row = [v for v in row if math.isfinite(v)]
    if not finite_row:
        return float("nan")
    return kl_from_uniform(finite_row)


def compute_block_min_confidence(
    arr: np.ndarray,
    answer_text: str,
    tokenizer,
) -> float:
    """Block-level minimum confidence.

    Split the answer into blocks by double-newline boundaries, compute
    the mean confidence per block, and return the minimum.

    Works from .npy arrays. The tokenizer is the same one used by vLLM,
    so tokenizer.encode(answer_text) produces token IDs that correspond
    1:1 with the rows of arr.
    """
    if arr.ndim < 2 or arr.shape[0] == 0:
        return float("nan")
    if not answer_text.strip():
        return float("nan")

    token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    T = arr.shape[0]
    L = min(T, len(token_ids))
    if L == 0:
        return float("nan")

    # Detect \n\n token ID and single \n token ID
    double_newline_id = None
    for tok_str in ["\n\n", "\n \n"]:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if len(ids) == 1:
            double_newline_id = ids[0]
            break
    single_nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    newline_id = single_nl_ids[0] if len(single_nl_ids) == 1 else None

    # Find block boundaries (token positions where a block ends)
    boundaries = []
    i = 0
    while i < L:
        if double_newline_id is not None and token_ids[i] == double_newline_id:
            boundaries.append(i)
            i += 1
        elif (newline_id is not None and i < L - 1
              and token_ids[i] == newline_id
              and token_ids[i + 1] == newline_id):
            boundaries.append(i + 1)
            i += 2
        else:
            i += 1

    # Split arr rows into blocks
    block_starts = [0]
    for b in boundaries:
        next_start = b + 1
        if next_start < L:
            block_starts.append(next_start)
    block_starts.append(L)

    block_confs = []
    for si in range(len(block_starts) - 1):
        s = block_starts[si]
        e = block_starts[si + 1]
        if e <= s:
            continue
        block_arr = arr[s:e].astype(np.float64)
        finite_mask = np.isfinite(block_arr)
        count = int(np.sum(finite_mask))
        if count == 0:
            continue
        block_mean = -float(np.sum(np.where(finite_mask, block_arr, 0.0)) / count)
        block_confs.append(block_mean)

    if not block_confs:
        return float("nan")
    return float(min(block_confs))


@lru_cache(maxsize=None)
def _get_tokenizer(model_name: str):
    model_info, _ = build_model_config(model_name)
    return AutoTokenizer.from_pretrained(model_info["default_model_path"])


def compute_all_confidences(
    npy_path: str,
    answer_text: str = "",
    model_name: Optional[str] = None,
) -> Dict[str, float]:
    """Load .npy and compute all confidence metrics.

    Returns dict with keys: mean_conf, bottom10_conf, tail_conf,
    first_token_conf, block_min_conf.

    *answer_text* and *model_name* are needed for block_min_conf.
    If not provided, block_min_conf will be NaN.
    """
    arr = np.load(npy_path, allow_pickle=False)
    result = {
        "mean_conf": compute_mean_confidence(arr),
        "bottom10_conf": compute_bottom10_confidence(arr),
        "tail_conf": compute_tail_confidence(arr),
        "first_token_conf": compute_first_token_confidence(arr),
    }
    if answer_text and model_name is not None:
        tokenizer = _get_tokenizer(model_name)
        result["block_min_conf"] = compute_block_min_confidence(
            arr, answer_text, tokenizer,
        )
    else:
        result["block_min_conf"] = float("nan")
    return result
