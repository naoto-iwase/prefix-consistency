"""
Naming helpers, prompt rendering, CoT parsing, and CLI helpers.
"""

from typing import Dict, List, Optional, Tuple


# =====================================================================
# Naming
# =====================================================================
def file_prefix(dataset: str, model_name: str) -> str:
    """Build filename prefix from dataset and model name."""
    return f"{dataset}_{model_name}"


# =====================================================================
# Prompt rendering
# =====================================================================
def create_prompt(problem: Dict, dataset_cfg: Dict) -> str:
    question_key = dataset_cfg["question_key"]
    return problem[question_key] + dataset_cfg["prompt_suffix"]


def build_rendered_prompt(
    problem: Dict, dataset_cfg: Dict, model_cfg: Dict, tokenizer,
) -> Tuple[str, str]:
    prompt_text = create_prompt(problem, dataset_cfg)
    messages = [
        {"role": "system", "content": dataset_cfg["system_prompt"]},
        {"role": "user", "content": prompt_text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **model_cfg["template_kwargs"],
    )
    return rendered, prompt_text


# =====================================================================
# CoT boundary helpers
# =====================================================================
def get_cot_to_final(model_cfg: Dict) -> Optional[str]:
    """Build the tag sequence that separates CoT from the final answer.

    For gpt-oss: cot_suffix (<|end|>) + final_prefix.
    For models with cot_suffix only (e.g. <think></think>, [THINK][/THINK]):
        cot_suffix.
    For no-think models (cot_suffix is None): returns None.
    """
    if model_cfg["cot_suffix"] is None:
        return None
    if model_cfg["final_prefix"] is not None:
        return model_cfg["cot_suffix"] + model_cfg["final_prefix"]
    return model_cfg["cot_suffix"]


def split_cot_and_final(
    text: str, model_cfg: Dict, start_from_cot: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """Split generated text into CoT content and Final content, excluding delimiters.

    Returns (cot_content, final_content):
      - Normal:    ("CoT text", "Final text")
      - Truncated: ("unfinished CoT text", None)   -- cot_suffix not found
      - No-think:  (None, "Final text")             -- model has no CoT phase

    *start_from_cot*: if False, treat the text as purely final answer when
    the CoT-to-final delimiter is not found (instead of assuming truncated CoT).
    Use this for continuations that are known to start from the final answer
    phase (e.g. truncation preserved the CoT boundary, or --insert-cot-closing was used).

    Works regardless of whether cot_prefix appears in the text (completions
    API may or may not include it).
    """
    cot_to_final = get_cot_to_final(model_cfg)

    # No-think model: entire text is final answer
    if cot_to_final is None:
        return None, text

    cot_prefix = model_cfg["cot_prefix"]

    # Skip cot_prefix if present, otherwise start from beginning
    prefix_pos = text.find(cot_prefix)
    after_prefix = text[prefix_pos + len(cot_prefix):] if prefix_pos != -1 else text

    # Find cot_to_final marker and split
    #   gpt-oss:
    #     <|channel|>analysis<|message|>...CoT...<|end|><|start|>assistant<|channel|>final<|message|>...
    #     cot_prefix + CoT + cot_to_final + Final
    #   models with cot_suffix only (e.g. <think>...</think>):
    #     <think>...CoT...</think>\n\nThe answer is \boxed{42}.
    #     cot_prefix + CoT + cot_to_final + Final
    marker_pos = after_prefix.find(cot_to_final)
    if marker_pos == -1:
        if start_from_cot:
            # Truncated: generation ended before cot_suffix
            return after_prefix, None
        else:
            # No CoT in this text, entire content is final answer
            return None, text

    cot_content = after_prefix[:marker_pos]
    final_content = after_prefix[marker_pos + len(cot_to_final):]
    return cot_content, final_content


def count_cot_and_final_tokens(
    text: str, model_cfg: Dict, tokenizer, start_from_cot: bool = True,
) -> Tuple[Optional[int], Optional[int]]:
    """Count CoT and Final tokens (delimiter-excluded) in generated text.

    Returns (cot_tokens, final_tokens). Both are None when cot_suffix is
    not found (e.g. truncated generation).

    See split_cot_and_final() for the meaning of *start_from_cot*.
    """
    cot_content, final_content = split_cot_and_final(text, model_cfg, start_from_cot)
    cot_tokens = None
    final_tokens = None
    if cot_content is not None:
        cot_tokens = len(tokenizer.encode(cot_content, add_special_tokens=False))
    if final_content is not None:
        final_tokens = len(tokenizer.encode(final_content, add_special_tokens=False))
    return cot_tokens, final_tokens


# =====================================================================
# CLI helpers
# =====================================================================
def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    result = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a.strip()), int(b.strip()) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))
