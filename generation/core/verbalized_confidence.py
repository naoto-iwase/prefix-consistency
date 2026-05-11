"""
Confidence extraction for CISC (Confidence-Informed Self-Consistency).

Implements 3 of the 4 CISC confidence methods. Each appends a prompt suffix
to (rendered_prompt + answer_text), calls the completions API, and extracts
a confidence score:

  - Verbal 0-100:    "rate 0-100" prompt, parse integer from text.
  - Verbal Binary:   "0 or 1" prompt, parse 0/1 from text.
  - P(True):         "0 or 1" prompt, extract P("1") from logprobs.

The 4th method, Response Probability, needs no API call. It is computed
from existing .tok_logprobs.npy files directly in convert_to_jsonl.py.

For reasoning models, the model reasons first (CoT), then outputs a
final answer containing the confidence. Both Verbal Binary and P(True)
extract from the final answer portion (after cot_to_final marker), not
from the CoT. For instruct/no-think models, the entire response is the
answer.

Library module: no CLI, no direct I/O. Used by enrich.py.

Reference:
  Taubenfeld et al., "Confidence Improves Self-Consistency in LLMs"
    (ACL 2025 Findings) https://arxiv.org/abs/2502.06233
"""

import re
from typing import Dict, Optional, Tuple
import math

from openai import OpenAI

from .config import TOP_LOGPROBS
from .api import call_completions
from .text import get_cot_to_final, split_cot_and_final

# =====================================================================
# Prompt suffixes
# =====================================================================

# Verbal 0-100 self-rating
VERBAL_0_100_PROMPT = (
    "\nNow I will rate my confidence in the proposed answer "
    "on a scale of 0-100.\nProposed confidence: ("
)

# Binary 0/1, shared by Verbal Binary (text parse) and P(True) (logprobs)
BINARY_PROMPT = (
    "\nNow I will rate my confidence in the proposed answer "
    "as either 0 or 1.\nProposed confidence: ("
)


def _prepare_confidence_prompt(
    rendered_prompt: str, answer_text: str, suffix: str, model_info: Dict,
) -> str:
    """Build the full prompt for a confidence query.

    If the answer's reasoning is unclosed (cot_to_final marker not found),
    inject cot_suffix + final_prefix to force-close the analysis channel
    and open the final channel before the confidence suffix.
    """
    cot_to_final = get_cot_to_final(model_info)
    if cot_to_final and cot_to_final not in answer_text:
        # Unclosed reasoning: close analysis, open final
        answer_text = answer_text + cot_to_final
    return rendered_prompt + answer_text + suffix


# =====================================================================
# JSONL field names and .txt header keys
# =====================================================================

VERBAL_0_100_CONF_KEY = "verbal_0_100_confidences"
VERBAL_0_100_CONF_HEADER = "Verbal 0-100 Confidence"
VERBAL_0_100_ACTUAL_TOKENS_KEY = "verbal_0_100_actual_tokens"
VERBAL_0_100_ACTUAL_TOKENS_HEADER = "Verbal 0-100 Actual Tokens"
VERBAL_0_100_MIN_TOKENS_KEY = "verbal_0_100_min_tokens"
VERBAL_0_100_MIN_TOKENS_HEADER = "Verbal 0-100 Min Tokens"

VERBAL_BINARY_CONF_KEY = "verbal_binary_confidences"
VERBAL_BINARY_CONF_HEADER = "Verbal Binary Confidence"
BINARY_QUERY_ACTUAL_TOKENS_KEY = "binary_query_actual_tokens"
BINARY_QUERY_ACTUAL_TOKENS_HEADER = "Binary Query Actual Tokens"
BINARY_QUERY_MIN_TOKENS_KEY = "binary_query_min_tokens"
BINARY_QUERY_MIN_TOKENS_HEADER = "Binary Query Min Tokens"

P_TRUE_CONF_KEY = "p_true_confidences"
P_TRUE_CONF_HEADER = "P(True)"

RESPONSE_PROB_KEY = "response_probabilities"
RESPONSE_PROB_HEADER = "Response Probability"


# =====================================================================
# Extraction: verbal (text parse)
# =====================================================================

def parse_verbal_0_100(text: str) -> Optional[float]:
    """Parse a confidence score (0-100) from the model's response.

    The model is expected to output something like "85)" or "85".
    We extract the first integer found before the first ")".
    """
    text = text.split(")")[0].strip()
    match = re.search(r"\d+", text)
    if match:
        val = int(match.group())
        return float(min(max(val, 0), 100))
    return None


def parse_verbal_binary(text: str) -> Optional[float]:
    """Parse a binary confidence (0 or 1) from the model's response.

    Same prompt as logit confidence ("0 or 1"), but extracts from text
    rather than logprobs. Returns 0.0 or 1.0.
    """
    text = text.split(")")[0].strip()
    match = re.search(r"[01]", text)
    if match:
        return float(int(match.group()))
    return None


# =====================================================================
# Extraction: P(True) (binary softmax from logprobs)
# =====================================================================

def extract_p_true_confidence(
    top_logprobs_at_pos: Optional[dict],
) -> Optional[float]:
    """Extract P("1") via softmax(logprob_0, logprob_1) from a single token's top_logprobs.

    P(True) = softmax([lp_0, lp_1])[1], i.e., the probability the model
    assigns to the "correct" token "1".
    Returns probability in [0, 1], or None if neither "0" nor "1" found.
    """
    if not top_logprobs_at_pos:
        return None

    logprob_0 = None
    logprob_1 = None
    for token_str, logprob in top_logprobs_at_pos.items():
        s = token_str.strip()
        if s == "0":
            if logprob_0 is None or logprob > logprob_0:
                logprob_0 = logprob
        elif s == "1":
            if logprob_1 is None or logprob > logprob_1:
                logprob_1 = logprob

    if logprob_0 is not None and logprob_1 is not None:
        max_lp = max(logprob_0, logprob_1)
        exp_0 = math.exp(logprob_0 - max_lp)
        exp_1 = math.exp(logprob_1 - max_lp)
        return exp_1 / (exp_0 + exp_1)
    if logprob_1 is not None:
        return float(math.exp(logprob_1))
    if logprob_0 is not None:
        return 1.0 - float(math.exp(logprob_0))
    return None


# =====================================================================
# Query functions
# =====================================================================

def _min_tokens_for_verbal_0_100(text: str, tokenizer) -> Optional[int]:
    """Count the minimum tokens needed to extract verbal 0-100 confidence.

    The parser splits on ")" and looks for the first integer, so the
    minimum text is everything up to and including the first ")".
    """
    paren_pos = text.find(")")
    if paren_pos == -1:
        return None
    prefix = text[:paren_pos + 1]
    return len(tokenizer.encode(prefix))


def query_verbal_0_100(
    client: OpenAI,
    model_name: str,
    rendered_prompt: str,
    answer_text: str,
    model_info: Dict,
    api_params: Dict,
    tokenizer,
) -> Tuple[Optional[float], str, int, Optional[int]]:
    """Query LLM for verbal confidence (0-100).

    Returns (confidence_score, raw_text, completion_tokens, min_tokens).
    """
    full_prompt = _prepare_confidence_prompt(
        rendered_prompt, answer_text, VERBAL_0_100_PROMPT, model_info)

    text, n_tokens, _, _ = call_completions(
        client, model_name, full_prompt,
        max_tokens=model_info["max_context_length"],
        max_context_length=model_info["max_context_length"],
        api_params=api_params, tokenizer=tokenizer, top_logprobs=0,
    )
    conf = parse_verbal_0_100(text)
    min_tokens = _min_tokens_for_verbal_0_100(text, tokenizer) if conf is not None else None
    return conf, text, n_tokens, min_tokens


def _find_confidence_token_idx(
    raw_top_logprobs: list, text: str, model_info: Dict,
) -> int:
    """Find the token index where the model outputs its confidence answer.

    For instruct/no-think models (no cot_to_final), this is token 0.
    For reasoning models, the model reasons first, then outputs a final
    answer. We find the cot_to_final marker, then scan the final portion
    for the first token whose top-k logprobs contain "0" or "1".

    Returns token index, or -1 if not found.
    """
    cot_to_final = get_cot_to_final(model_info)
    if cot_to_final is None:
        return 0

    marker_pos = text.find(cot_to_final)
    if marker_pos == -1:
        # CoT never finished (truncated). Fall back to token 0.
        return 0

    # Walk tokens to find where the final portion starts
    target_char = marker_pos + len(cot_to_final)
    char_count = 0
    final_start = -1
    for i, lp_dict in enumerate(raw_top_logprobs):
        tok_str = max(lp_dict, key=lp_dict.get)
        char_count += len(tok_str)
        if char_count >= target_char:
            final_start = i + 1
            break

    if final_start < 0 or final_start >= len(raw_top_logprobs):
        return 0

    # Scan final portion for the first token with "0" or "1" in top-k
    for i in range(final_start, len(raw_top_logprobs)):
        if any(k.strip() in ("0", "1") for k in raw_top_logprobs[i]):
            return i

    return 0  # fallback


def query_binary_confidence(
    client: OpenAI,
    model_name: str,
    rendered_prompt: str,
    answer_text: str,
    model_info: Dict,
    api_params: Dict,
    tokenizer,
) -> Tuple[Optional[float], Optional[float], str, int, Optional[int]]:
    """Single API call for both Verbal Binary and P(True).

    Uses the BINARY_PROMPT ("0 or 1"), generates full response
    with top_logprobs=TOP_LOGPROBS, then extracts:
      - Verbal Binary: parse 0/1 from the final answer text
      - P(True): softmax P("1") from logprobs at the confidence
        token in the final answer (after reasoning, not before)

    Returns (verbal_binary_conf, p_true_conf, raw_text, completion_tokens, min_tokens).
    """
    full_prompt = _prepare_confidence_prompt(
        rendered_prompt, answer_text, BINARY_PROMPT, model_info)

    text, n_tokens, raw_top_logprobs, _ = call_completions(
        client, model_name, full_prompt,
        max_tokens=model_info["max_context_length"],
        max_context_length=model_info["max_context_length"],
        api_params=api_params, tokenizer=tokenizer, top_logprobs=TOP_LOGPROBS,
    )

    # Verbal Binary: parse 0/1 from the final answer portion
    cot_to_final = get_cot_to_final(model_info)
    if cot_to_final and cot_to_final in text:
        _, final_text = split_cot_and_final(text, model_info)
        verbal_binary = parse_verbal_binary(final_text or text)
    else:
        verbal_binary = parse_verbal_binary(text)

    # P(True): find the confidence token in the final answer and
    # extract P("1") from its logprobs
    p_true = None
    min_tokens = None
    if raw_top_logprobs:
        idx = _find_confidence_token_idx(raw_top_logprobs, text, model_info)
        if 0 <= idx < len(raw_top_logprobs):
            p_true = extract_p_true_confidence(raw_top_logprobs[idx])
            min_tokens = idx + 1

    return verbal_binary, p_true, text, n_tokens, min_tokens
