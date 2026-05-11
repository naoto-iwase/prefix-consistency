"""
LLM-based pairwise answer comparison for evaluation.

Usage:
    import llm_judge
    llm_judge.init(enabled=True, dataset="frontierscience_olympiad", model_name="gpt-oss-20b")
    llm_judge.init(enabled=True, dataset="hmmt", model_name="gpt-oss-20b")
    llm_judge.init(enabled=False, dataset="aime2025", model_name="gpt-oss-20b")  # disabled
"""

import json
import os
import threading

from openai import OpenAI
from transformers import AutoTokenizer

from .config import BASE_URL, build_model_config
from .api import call_chat_completions

BACKENDS = ("local", "openai", "anthropic")
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"

MATH_JUDGE_PROMPT = """Determine whether two mathematical answers are numerically identical.

Answer 1: {answer1}
Answer 2: {answer2}

Criteria:
- If they represent exactly the same numerical value, respond "YES"
- If they represent different values or one is not numerical, respond "NO"
- Different formats (fractions, decimals, radicals, exponential notation) are acceptable if numerically equivalent

Examples:
- "1/2" and "0.5" → YES
- "√4" and "2" → YES
- "2^3" and "8" → YES
- "3.14" and "π" → NO (approximation vs exact value)
- "x+1" and "1+x" → YES (same expression)
- "x^2" and "2x" → NO (different expressions)

Answer only "YES" or "NO"."""

# Based on official FrontierScience Olympiad judge prompt (Appendix B of arxiv:2601.21165)
SCIENCE_JUDGE_PROMPT = """You are grading an attempted answer to a science olympiad problem. \
You will be given the attempted answer and the reference answer. \
Evaluate strictly, but fairly. \
The reference answer is either a single number or expression in latex formatting, \
a chemical formula, a compound name, or a phrase referring to a specific name, entity, or method. \
Mark the attempted answer as correct if it fully matches the reference answer or is otherwise \
equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal place \
rounding of the reference answer (e.g., 6.69≈6.7), an equivalent name for a compound/formula, \
equivalent when accounting for units, etc.). \
Mark it as incorrect if it is not equivalent to the reference answer.

Attempted answer: {answer1}
Reference answer: {answer2}

Answer only "YES" if correct or "NO" if incorrect."""


# Map dataset names to prompt templates
DATASET_PROMPTS = {
    "frontierscience_olympiad": SCIENCE_JUDGE_PROMPT,
    "hmmt": MATH_JUDGE_PROMPT,
    "brumo": MATH_JUDGE_PROMPT,
}

_client: OpenAI | None = None
_model: str | None = None
_model_info: dict | None = None
_api_params: dict | None = None
_tokenizer = None
_prompt_template: str = ""
_extra_body: dict | None = None
_cache: dict[tuple[str, str], bool] = {}
_cache_lock = threading.Lock()


def init(
    enabled: bool, dataset: str, model_name: str,
    backend: str = "local",
    reasoning_effort: str | None = None, no_think: bool = False,
    timeout: int = 1800,
) -> None:
    """Initialize the LLM judge. If disabled, normalize() becomes a pass-through.

    backend='local' uses vLLM at BASE_URL with the same model under eval.
    backend='openai' / 'anthropic' use cloud Chat Completions; reasoning_effort
    and no_think are ignored.
    """
    global _client, _model, _model_info, _api_params, _tokenizer, _prompt_template, _extra_body, _cache
    if not enabled:
        if dataset in DATASET_PROMPTS:
            print(f"Warning: LLM judge disabled for '{dataset}', but it has a registered judge prompt. "
                  f"Answer normalization will fall back to exact string matching.")
        _client = None
        _model = None
        _model_info = None
        _api_params = None
        _tokenizer = None
        _prompt_template = ""
        _extra_body = None
        _cache = {}
        return
    if backend not in BACKENDS:
        raise ValueError(f"Unknown judge backend: {backend!r} (choose from {BACKENDS})")
    _prompt_template = DATASET_PROMPTS[dataset]
    _model = model_name

    if backend == "local":
        _client = OpenAI(base_url=BASE_URL, api_key="dummy", timeout=timeout)
        _model_info, _api_params = build_model_config(model_name, no_think=no_think, reasoning_effort=reasoning_effort)
        _tokenizer = AutoTokenizer.from_pretrained(_model_info["default_model_path"])
        tkw = _model_info["template_kwargs"]
        _extra_body = {"chat_template_kwargs": tkw} if tkw else None
        print(f"LLM judge: {BASE_URL} (model: {_model}, max_context: {_model_info['max_context_length']})")
    else:
        env_var = "OPENAI_API_KEY" if backend == "openai" else "ANTHROPIC_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key:
            raise RuntimeError(f"{env_var} not set (required for backend={backend!r})")
        base_url = None if backend == "openai" else ANTHROPIC_BASE_URL
        _client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        # Cloud responses are short; skip tokenizer-based clamping in api.py.
        _model_info = {"max_context_length": 1024}
        _api_params = {}
        _tokenizer = None
        _extra_body = None
        print(f"LLM judge: {backend} (model: {_model})")


def _are_equivalent(answer1: str, answer2: str, question: str = "") -> bool:
    """Ask LLM whether two answers are equivalent (thread-safe cached)."""
    if not answer1.strip() or not answer2.strip():
        return False
    key = (answer1, answer2)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    # Release the lock during the API call to allow other (different) keys to proceed.
    # Same-key concurrent calls are prevented because the first caller will populate
    # the cache before others re-check. In the rare case of a race on the same key,
    # the worst outcome is a single duplicate API call (harmless).
    fmt = {"answer1": answer1, "answer2": answer2}
    if "{question}" in _prompt_template:
        if not question:
            print("Warning: judge prompt expects {question} but got empty string")
        fmt["question"] = question
    prompt = _prompt_template.format(**fmt)
    messages = [{"role": "user", "content": prompt}]
    try:
        content = call_chat_completions(
            _client, _model, messages, _model_info["max_context_length"],
            _model_info["max_context_length"], _api_params,
            _tokenizer, extra_body=_extra_body,
        )
        verdict = content.strip().upper()[-100:]
        result = "YES" in verdict and "NO" not in verdict
    except Exception as e:
        print(f"Warning: LLM judge failed: {e}, falling back to string comparison")
        result = answer1.strip() == answer2.strip()
    with _cache_lock:
        _cache[key] = result
    return result


def get_cache() -> dict:
    """Return cache as a JSON-serializable dict for persistence."""
    with _cache_lock:
        return {json.dumps([k[0], k[1]]): v for k, v in _cache.items()}


def load_cache(data: dict) -> int:
    """Populate cache from a JSON-serializable dict. Returns entries loaded."""
    loaded = 0
    with _cache_lock:
        for key_str, val in data.items():
            pair = json.loads(key_str)
            _cache[(pair[0], pair[1])] = val
            loaded += 1
    return loaded


def normalize(ans: str, gold: str, question: str = "") -> str:
    """If ans != gold and LLM judge says they match, return gold instead.

    If init() has not been called, returns ans unchanged.
    """
    if _client is None or ans == gold:
        return ans
    if _are_equivalent(ans, gold, question):
        return gold
    return ans
