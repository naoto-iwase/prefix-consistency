"""
File I/O for answer files, headers, logprobs, and metadata.

Naming conventions:
  read_*   - load data from disk
  write_*  - save data to disk (with automatic retry on OSError)
  has_*    - check existence without full parsing
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# =====================================================================
# Constants
# =====================================================================
HEADER_DELIMITER = "=" * 50  # separates file header from answer body
REGEN_DELIMITER = "--- regen start " + "-" * 34  # separates kept text from continuation
TOP_LOGPROBS_SUFFIX = ".npy"  # top-K logprobs sidecar (T, K)
TOKEN_LOGPROBS_SUFFIX = ".tok_logprobs.npy"  # per-token logprobs sidecar (T,)

_HEADER_KEY_RE = re.compile(r"^[A-Z][A-Za-z0-9 ()_-]+$")

_RETRY_COUNT = 3
_RETRY_DELAY = 1.0


# =====================================================================
# Internal helpers
# =====================================================================
def _write_text_with_retry(path: Path, text: str) -> None:
    """Write text to a file, retrying on OSError (e.g. NV I/O stalls)."""
    for attempt in range(_RETRY_COUNT):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return
        except OSError:
            if attempt == _RETRY_COUNT - 1:
                raise
            time.sleep(_RETRY_DELAY * (attempt + 1))


def _write_npy_with_retry(path: Path, arr: np.ndarray) -> None:
    """Save numpy array to .npy, retrying on OSError."""
    for attempt in range(_RETRY_COUNT):
        try:
            np.save(str(path), arr)
            return
        except OSError:
            if attempt == _RETRY_COUNT - 1:
                raise
            time.sleep(_RETRY_DELAY * (attempt + 1))


def _logprobs_to_array(raw_top_logprobs: list, top_k: int) -> np.ndarray:
    """Convert vLLM top_logprobs (list of dicts) to numpy array (T, top_k)."""
    rows = []
    for top_lp_dict in raw_top_logprobs:
        if top_lp_dict is None:
            rows.append([float("-inf")] * top_k)
            continue
        d = dict(top_lp_dict)
        values = sorted(d.values(), reverse=True)
        if len(values) < top_k:
            values.extend([float("-inf")] * (top_k - len(values)))
        rows.append(values[:top_k])
    return np.array(rows, dtype=np.float32)


# =====================================================================
# Reading
# =====================================================================
def read_problems(data_file: str) -> List[Dict]:
    """Load problems from a JSONL file."""
    problems = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    print(f"  Loaded {data_file}: {len(problems)} problems")
    return problems


def read_answer_text(file_path: Path, regen_only: bool = False) -> str:
    """Read a saved answer file, stripping any regen delimiters.

    Returns clean model output suitable for truncation, answer extraction,
    and re-use as source for further regeneration.

    If regen_only=True, returns only the regenerated portion (after the
    regen delimiter). Returns the full text if no regen delimiter is found.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    pos = content.find(HEADER_DELIMITER)
    if pos == -1:
        raw = content
    else:
        start = pos + len(HEADER_DELIMITER)
        if start < len(content) and content[start] == "\n":
            start += 1
        raw = content[start:]

    delim = f"\n{REGEN_DELIMITER}\n"
    if regen_only:
        pos = raw.find(delim)
        if pos != -1:
            return raw[pos + len(delim):]
    return raw.replace(delim, "")


def read_header(file_path: Path) -> Dict[str, str]:
    """Parse all key-value pairs from a .txt file header (before === separator).

    Only lines where the part before the first ':' looks like a valid header
    key (alphanumeric with spaces) are treated as key-value pairs. Other lines
    (e.g., continuation of a multi-line answer value containing ':') are
    appended to the previous key's value.
    """
    header = {}
    last_key = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("=" * 10):
                break
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                if _HEADER_KEY_RE.match(key):
                    header[key] = val.strip()
                    last_key = key
                elif last_key is not None:
                    header[last_key] += "\n" + line
            elif last_key is not None and line.strip():
                header[last_key] += "\n" + line
    return header


def has_header_field(file_path: Path, key: str) -> bool:
    """Check if a .txt file header contains a given key.

    Returns False on I/O errors (file will be re-processed by the caller).
    """
    prefix = f"{key}:"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(prefix):
                    return True
                if line.startswith("=" * 10):
                    return False
    except OSError:
        pass
    return False


# =====================================================================
# Writing
# =====================================================================
def write_answer(
    file_path: Path,
    dataset: str,
    problem_index: int,
    answer_index: int,
    gold_answer: str,
    prompt: str,
    generated: str,
    completion_tokens: int,
    cot_tokens: Optional[int] = None,
    final_tokens: Optional[int] = None,
    extra_headers: Optional[Dict[str, object]] = None,
    regenerated: Optional[str] = None,
):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        f"Dataset: {dataset}",
        f"Problem Number: {problem_index}",
        f"Answer Index: {answer_index}",
        f"Gold Answer: {gold_answer}",
        f"Generated Tokens: {completion_tokens}",
    ]
    if cot_tokens is not None:
        header_lines.append(f"CoT Tokens: {cot_tokens}")
    if final_tokens is not None:
        header_lines.append(f"Final Tokens: {final_tokens}")
    if extra_headers:
        for key, val in extra_headers.items():
            header_lines.append(f"{key}: {val}")
    header_lines.append(f"Prompt: {prompt}")
    header = "\n".join(header_lines)
    body = generated
    if regenerated:
        body += f"\n{REGEN_DELIMITER}\n{regenerated}"
    content = f"{header}\n{HEADER_DELIMITER}\n{body}"
    _write_text_with_retry(file_path, content)


def write_header_fields(file_path: Path, fields: Dict[str, str]) -> None:
    """Insert new key-value pairs into a .txt file header.

    Fields are inserted before the "Prompt:" line if present (keeping Prompt
    as the last header before the === separator). Falls back to inserting
    before the === separator if no Prompt line exists.

    This is append-only: callers should check has_header_field() first to
    avoid duplicates.
    """
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []
    inserted = False
    for line in lines:
        if not inserted and (line.startswith("Prompt:") or line.startswith("=" * 10)):
            for key, value in fields.items():
                new_lines.append(f"{key}: {value}\n")
            inserted = True
        new_lines.append(line)
    _write_text_with_retry(file_path, "".join(new_lines))


def write_logprobs(npy_path: Path, raw_top_logprobs: list,
                   top_k: int) -> None:
    """Convert vLLM top_logprobs to numpy array and save as .npy."""
    arr = _logprobs_to_array(raw_top_logprobs, top_k)
    _write_npy_with_retry(npy_path, arr)


def write_token_logprobs(txt_path: Path, token_logprobs: list) -> None:
    """Save per-token logprobs (of actually generated tokens) as 1D .npy."""
    arr = np.array(token_logprobs, dtype=np.float64)
    _write_npy_with_retry(Path(str(txt_path) + TOKEN_LOGPROBS_SUFFIX), arr)


def write_metadata(out_dir: Path, dataset: str, model_name: str, **extra):
    """Write metadata.json to out_dir.

    If the file already exists, check that existing values are consistent
    with the new values. Raises ValueError on conflict.

    Pass any experiment-specific parameters via **extra so that runs are
    reproducible from metadata alone.  Examples:
        reasoning_effort, num_answers, insert_cot_closing, remove_pct, regen_count
    """
    meta_path = out_dir / "metadata.json"
    new_meta = {"dataset": dataset, "model": model_name}
    new_meta.update(extra)

    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        # regen_count naturally increases when extending experiments
        SKIP_KEYS = {"regen_count"}
        conflicts = {
            k: (existing[k], new_meta[k])
            for k in new_meta
            if k in existing and existing[k] != new_meta[k]
            and k not in SKIP_KEYS
        }
        if conflicts:
            detail = ", ".join(f"{k}: {old!r} vs {new!r}"
                               for k, (old, new) in conflicts.items())
            raise ValueError(
                f"metadata.json conflict in {out_dir}: {detail}. "
                f"Use a different output directory for different settings."
            )
        existing.update(new_meta)
        new_meta = existing

    _write_text_with_retry(meta_path,
                           json.dumps(new_meta, indent=2, ensure_ascii=False) + "\n")
