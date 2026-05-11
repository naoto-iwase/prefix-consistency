"""
vLLM completions and chat API wrappers with retry logic.
"""

import time
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

_MAX_RETRIES = 3


def clamp_max_tokens(max_tokens: int, max_context_length: int, prompt: str, tokenizer) -> int:
    """Clamp max_tokens so that prompt + generation fits within the model context."""
    prompt_tokens = len(tokenizer.encode(prompt))
    return min(max_tokens, max_context_length - prompt_tokens - 256)


def call_completions(
    client: OpenAI, model_name: str, prompt: str, max_tokens: int,
    max_context_length: int, api_params: Dict,
    tokenizer, top_logprobs: int = 0,
) -> Tuple[str, int, Optional[list], Optional[list]]:
    """Call vLLM completions API.

    Returns (text, completion_tokens, top_logprobs_list, token_logprobs_list).
    top_logprobs_list is a list of dicts (one per token) when top_logprobs > 0,
    or None when top_logprobs == 0.
    token_logprobs_list is a list of floats (logprob of each generated token),
    or None when top_logprobs == 0.
    """
    max_tokens = clamp_max_tokens(max_tokens, max_context_length, prompt, tokenizer)
    kwargs = dict(model=model_name, prompt=prompt, max_tokens=max_tokens, **api_params)
    if top_logprobs > 0:
        kwargs["logprobs"] = top_logprobs
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.completions.create(**kwargs)
            break
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{_MAX_RETRIES} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    if not resp or not hasattr(resp, "choices") or not resp.choices:
        raise ValueError("Invalid API response")
    choice = resp.choices[0]
    if hasattr(choice, "text"):
        text = choice.text
    elif hasattr(choice, "message") and choice.message:
        text = choice.message.content
    else:
        raise ValueError(f"Cannot extract answer: {choice}")
    usage = getattr(resp, "usage", None)
    tokens = usage.completion_tokens if usage else 0

    raw_top_logprobs = None
    token_logprobs = None
    if top_logprobs > 0 and hasattr(choice, "logprobs") and choice.logprobs:
        raw_top_logprobs = choice.logprobs.top_logprobs
        if hasattr(choice.logprobs, "token_logprobs") and choice.logprobs.token_logprobs:
            token_logprobs = list(choice.logprobs.token_logprobs)
    return text, tokens, raw_top_logprobs, token_logprobs


def call_completions_echo(
    client: OpenAI, model_name: str, prompt: str, top_logprobs: int = 20,
):
    """Call vLLM completions API in echo mode (max_tokens=0, echo=True).

    Returns the logprobs object from the response choice.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=0,
                temperature=0,
                logprobs=top_logprobs,
                echo=True,
            )
            return resp.choices[0].logprobs
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{_MAX_RETRIES} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def call_chat_completions(
    client: OpenAI, model_name: str, messages: List[Dict],
    max_tokens: int,
    max_context_length: Optional[int] = None,
    api_params: Optional[Dict] = None,
    tokenizer=None,
    extra_body: Optional[Dict] = None,
) -> str:
    if tokenizer is not None and max_context_length is not None:
        prompt_text = " ".join(m["content"] for m in messages)
        max_tokens = clamp_max_tokens(max_tokens, max_context_length, prompt_text, tokenizer)
    api_params = api_params or {}
    kwargs = dict(model=model_name, messages=messages, max_tokens=max_tokens,
                  stream=False, **api_params)
    if extra_body:
        kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra_body}
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{_MAX_RETRIES} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    if not resp or not resp.choices:
        raise ValueError("Invalid chat API response")
    return resp.choices[0].message.content
