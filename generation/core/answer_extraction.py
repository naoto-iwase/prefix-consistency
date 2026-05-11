"""
Answer extraction and normalization from model output text.

Extracts answers from \\boxed{} expressions, normalizes mathematical
notation, and handles dataset-specific formatting (letter choices,
minimal normalization for free-form datasets, etc.).

Library module: no CLI, no I/O. Used by convert_to_jsonl.py and
core/answer_map.py.
"""

import re
from typing import Optional

EXTRACTED_ANSWER_HEADER = "Extracted Answer"
PARSE_FAILED = "PARSE_FAILED"


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{...} (supports nested braces).

    Falls back to the last number-like string if no \\boxed{} is found.
    """
    boxed_start = text.rfind("\\boxed{")
    if boxed_start != -1:
        start_pos = boxed_start + 7
        brace_count = 1
        end_pos = start_pos
        while end_pos < len(text) and brace_count > 0:
            if text[end_pos] == "{":
                brace_count += 1
            elif text[end_pos] == "}":
                brace_count -= 1
            end_pos += 1
        if brace_count == 0:
            content = text[start_pos : end_pos - 1].strip()
            if "=" in content:
                last_val = content.split("=")[-1].strip()
                if last_val:
                    content = last_val
            return content

    for pattern in [r"\b(\d+\.?\d*)\b", r"\b(\d+/\d+)\b", r"\b(\d+)\b"]:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    return None


def normalize_math_notation(text: str) -> str:
    """Normalize mathematical notation in an extracted answer."""
    if not text:
        return text
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\\[td]frac", r"\\frac", text)
    text = re.sub(r"\\text\{\\?\(?([A-Za-z])\\?\)?\}", r"\1", text)
    text = re.sub(r"(\\frac\{[^}]+\})(\d+)\\text\{degrees\}$", r"\1{\2}", text)
    text = re.sub(r"\\text\{degrees\}$", "", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", text)
    text = re.sub(r",\\;", ",", text)
    text = re.sub(r"\\%", "", text)
    text = re.sub(r",\\,", ",", text)
    text = re.sub(r",\\!", ",", text)
    text = re.sub(r"^[a-zA-Z]\s*\\in\s*", "", text)
    text = re.sub(r"^[a-zA-Z]\s*=\s*", "", text)

    def convert_pmatrix(match):
        elements = re.split(r"\\\\|\n", match.group(1))
        elements = [e.strip() for e in elements if e.strip()]
        return "(" + ",".join(elements) + ")"

    text = re.sub(
        r"\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}",
        convert_pmatrix, text, flags=re.DOTALL,
    )
    text = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = re.sub(r"°", "", text)
    text = re.sub(r"^\\\$", "", text)
    text = re.sub(r"\\mbox\{[^}]*\}(?:\^\d*)?$", "", text)
    text = re.sub(r"\^\{th\}(?:grade)?$", "", text)
    text = re.sub(
        r"\\text\{(?:cents?|dollars?|units?|cm|mm|km|m|inches?|feet|ft|yards?|yd)\}$", "", text,
    )
    text = re.sub(
        r"(?:cents?|dollars?|units?|cm|mm|km|m|inches?|feet|ft|yards?|yd)$", "", text,
    )
    text = re.sub(r"\[\d+pt\]", "", text)

    result = ""
    paren_depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            paren_depth += 1
            result += ch
        elif ch == ")":
            paren_depth -= 1
            result += ch
        elif ch == "," and paren_depth == 0:
            if i > 0 and i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
                continue
            result += ch
        else:
            result += ch
    text = result

    text = re.sub(r"\\frac(\d+)\{", r"\\frac{\1}{", text)
    text = re.sub(r"^\\,(.*)\\,$", r"\1", text)
    return text


def normalize_answer_format(text: str) -> str:
    """Normalize answer string: remove degree markers, unify fractions, trim."""
    if not isinstance(text, str):
        return str(text)
    if not text.strip():
        return text
    result = text
    result = re.sub(r"\^\{\\circ\}", "", result)
    result = re.sub(r"\^\\circ", "", result)
    result = re.sub(r"\\dfrac", r"\\frac", result)
    return result.strip()


def extract_answer(text: str, dataset: str) -> Optional[str]:
    """Extract and normalize an answer from model output text.

    For free-form science (FrontierScience): minimal normalization.
    For math datasets: full math normalization.
    """
    raw = extract_boxed_answer(text)
    if raw is None:
        return None
    dataset_lower = dataset.lower()
    if dataset_lower == "frontierscience_olympiad":
        return raw.strip()
    result = normalize_math_notation(raw)
    return normalize_answer_format(result) if result else None
