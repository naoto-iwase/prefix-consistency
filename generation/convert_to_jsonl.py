#!/usr/bin/env python3
"""
Convert regeneration experiment data into evaluation-ready JSONL format.
Enrichment data is read from pre-computed .txt file headers (written by
enrich.py).

No GPU needed. Pure file conversion.

Usage:
    uv run python convert_to_jsonl.py \
        --dataset aime2025 --model-name gpt-oss-20b \
        --init-dir data/init \
        --regen-dir data/regen_rm50pct_full \
        --marker-regen-dir data/regen_from_markers \
        --output ../analysis/jsonl/analysis_aime2025_gpt-oss-20b_rm50pct_full_x1.jsonl
"""

import argparse
import json
import re
import time
from typing import Dict, List, NamedTuple, Optional, Tuple
from collections import Counter, defaultdict
from pathlib import Path

from core.config import DATASET_NAMES, MODEL_NAMES
from core.io import read_header
from core.text import file_prefix, parse_int_list
from core.answer_extraction import EXTRACTED_ANSWER_HEADER, PARSE_FAILED
from core.answer_map import CANONICAL_HEADER_KEY
from core.logprob_confidence import CONFIDENCE_HEADERS
from core.verbalized_confidence import (
    RESPONSE_PROB_KEY, RESPONSE_PROB_HEADER,
    VERBAL_0_100_CONF_KEY, VERBAL_0_100_CONF_HEADER,
    VERBAL_0_100_ACTUAL_TOKENS_KEY, VERBAL_0_100_ACTUAL_TOKENS_HEADER,
    VERBAL_0_100_MIN_TOKENS_KEY, VERBAL_0_100_MIN_TOKENS_HEADER,
    VERBAL_BINARY_CONF_KEY, VERBAL_BINARY_CONF_HEADER,
    BINARY_QUERY_ACTUAL_TOKENS_KEY, BINARY_QUERY_ACTUAL_TOKENS_HEADER,
    BINARY_QUERY_MIN_TOKENS_KEY, BINARY_QUERY_MIN_TOKENS_HEADER,
    P_TRUE_CONF_KEY, P_TRUE_CONF_HEADER,
)


# =====================================================================
# NamedTuples
# =====================================================================

class InitAnswer(NamedTuple):
    ans_idx: int
    answer: Optional[str]
    # --- token counts (header order) ---
    tokens: int                                    # Generated Tokens
    cot_tokens: Optional[int]                      # CoT Tokens
    final_tokens: Optional[int]                    # Final Tokens
    # --- confidence (enrich step 2) ---
    logprob_conf: Optional[Dict[str, float]]       # DeepConf (5 metrics)
    resp_prob: Optional[float]                     # Response Probability
    # --- confidence (enrich steps 3-4) ---
    verbal_0_100_conf: Optional[int]               # Verbal 0-100
    verbal_0_100_actual_tokens: Optional[int]      # Verbal 0-100 query tokens
    verbal_0_100_min_tokens: Optional[int]         # Verbal 0-100 min tokens for extraction
    verbal_binary_conf: Optional[float]            # Verbal Binary
    p_true_conf: Optional[float]                   # P(True)
    binary_query_actual_tokens: Optional[int]      # Binary query tokens
    binary_query_min_tokens: Optional[int]         # Binary query min tokens for extraction


class RegenAnswer(NamedTuple):
    regen_idx: int
    answer: Optional[str]
    # --- token counts ---
    tokens: int                                    # Generated Tokens
    cot_tokens: Optional[int]                      # CoT Tokens
    final_tokens: Optional[int]                    # Final Tokens
    init_tokens: Optional[int]                     # Init Tokens
    kept_tokens: Optional[int]                     # Kept Tokens
    cut_tokens: Optional[int]                      # Cut Tokens
    # --- confidence ---
    logprob_conf: Optional[Dict[str, float]]       # DeepConf
    resp_prob: Optional[float]                     # Response Probability
    verbal_0_100_conf: Optional[int]               # Verbal 0-100
    verbal_0_100_actual_tokens: Optional[int]      # Verbal 0-100 query tokens
    verbal_0_100_min_tokens: Optional[int]         # Verbal 0-100 min tokens
    verbal_binary_conf: Optional[float]            # Verbal Binary
    p_true_conf: Optional[float]                   # P(True)
    binary_query_actual_tokens: Optional[int]      # Binary query tokens
    binary_query_min_tokens: Optional[int]         # Binary query min tokens


class MarkerRegenAnswer(NamedTuple):
    marker_idx: int
    regen_idx: int
    answer: Optional[str]
    # --- token counts ---
    tokens: int                                    # Generated Tokens
    cot_tokens: Optional[int]                      # CoT Tokens
    final_tokens: Optional[int]                    # Final Tokens
    init_tokens: Optional[int]                     # Init Tokens
    kept_tokens: Optional[int]                     # Kept Tokens
    cut_tokens: Optional[int]                      # Cut Tokens
    # --- marker-specific ---
    marker_position: Optional[int]                 # Marker Position (char offset)
    # --- confidence ---
    logprob_conf: Optional[Dict[str, float]]       # DeepConf
    resp_prob: Optional[float]                     # Response Probability
    verbal_0_100_conf: Optional[int]               # Verbal 0-100
    verbal_0_100_actual_tokens: Optional[int]      # Verbal 0-100 query tokens
    verbal_0_100_min_tokens: Optional[int]         # Verbal 0-100 min tokens
    verbal_binary_conf: Optional[float]            # Verbal Binary
    p_true_conf: Optional[float]                   # P(True)
    binary_query_actual_tokens: Optional[int]      # Binary query tokens
    binary_query_min_tokens: Optional[int]         # Binary query min tokens


# =====================================================================
# Header reading helpers
# =====================================================================

def _header_int(header: Dict[str, str], key: str) -> Optional[int]:
    val = header.get(key)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _header_float(header: Dict[str, str], key: str) -> Optional[float]:
    val = header.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _read_common_fields(header: Dict[str, str]):
    """Extract fields shared by init, regen, and marker regen files."""
    answer = (header.get(CANONICAL_HEADER_KEY)
              or header.get(EXTRACTED_ANSWER_HEADER)
              or PARSE_FAILED)
    tokens = _header_int(header, "Generated Tokens") or 0
    cot_tokens = _header_int(header, "CoT Tokens")
    final_tokens = _header_int(header, "Final Tokens")
    verbal_0_100_conf = _header_int(header, VERBAL_0_100_CONF_HEADER)
    verbal_0_100_actual_tokens = _header_int(header, VERBAL_0_100_ACTUAL_TOKENS_HEADER)
    verbal_0_100_min_tokens = _header_int(header, VERBAL_0_100_MIN_TOKENS_HEADER)
    binary_query_actual_tokens = _header_int(header, BINARY_QUERY_ACTUAL_TOKENS_HEADER)
    binary_query_min_tokens = _header_int(header, BINARY_QUERY_MIN_TOKENS_HEADER)

    confidence = None
    if any(k in header for k in CONFIDENCE_HEADERS.values()):
        confidence = {}
        for metric_key, header_key in CONFIDENCE_HEADERS.items():
            val = _header_float(header, header_key)
            confidence[metric_key] = val if val is not None else 0.0

    return (answer, tokens, cot_tokens, final_tokens, confidence,
            verbal_0_100_conf, verbal_0_100_actual_tokens, verbal_0_100_min_tokens,
            binary_query_actual_tokens, binary_query_min_tokens)


# =====================================================================
# Answer normalization via answer_map.json
# =====================================================================

def load_golds(init_dir: Path) -> Dict[int, str]:
    """Load normalized gold answers from answer_map.json."""
    path = init_dir / "answer_map.json"
    if not path.exists():
        raise FileNotFoundError(
            f"answer_map.json not found in {init_dir}. Run enrich.py first.")
    with open(path) as f:
        data = json.load(f)
    if "golds" not in data:
        raise ValueError(
            f"answer_map.json in {init_dir} is missing 'golds' key. "
            f"Regenerate it with the latest enrich.py.")
    return {int(k): v for k, v in data["golds"].items()}


# =====================================================================
# File loading
# =====================================================================

def load_init_answers(
    init_dir: Path, dataset: str, model_name: str,
) -> Dict[int, List[InitAnswer]]:
    """Load initial answers from .txt file headers.

    Returns {prob: [InitAnswer, ...]}, sorted by ans_idx.
    """
    prefix = file_prefix(dataset, model_name)
    pattern = re.compile(rf"{re.escape(prefix)}_prob(\d+)_answer(\d+)\.txt$")
    by_prob: Dict[int, List[InitAnswer]] = defaultdict(list)

    files = sorted(fp for fp in init_dir.glob(f"{prefix}_prob*_answer*.txt")
                   if "_regen" not in fp.name)
    t0 = time.time()
    errors = 0
    for i, fp in enumerate(files):
        m = pattern.match(fp.name)
        if not m:
            continue
        try:
            prob = int(m.group(1))
            ans_idx = int(m.group(2))
            header = read_header(fp)
            ans, tok, ct, ft, conf, v0100, v0100_tok, v0100_min, bq_tok, bq_min = _read_common_fields(header)

            by_prob[prob].append(InitAnswer(
                ans_idx=ans_idx, answer=ans, tokens=tok,
                cot_tokens=ct, final_tokens=ft, logprob_conf=conf,
                resp_prob=_header_float(header, RESPONSE_PROB_HEADER),
                verbal_0_100_conf=v0100,
                verbal_0_100_actual_tokens=v0100_tok,
                verbal_0_100_min_tokens=v0100_min,
                verbal_binary_conf=_header_float(header, VERBAL_BINARY_CONF_HEADER),
                p_true_conf=_header_float(header, P_TRUE_CONF_HEADER),
                binary_query_actual_tokens=bq_tok,
                binary_query_min_tokens=bq_min,
            ))
        except Exception as e:
            errors += 1
            print(f"  ERROR {fp.name}: {e}")
        if (i + 1) % 500 == 0 or i + 1 == len(files):
            print(f"  load_init: {i + 1}/{len(files)} ({time.time() - t0:.0f}s)")
    if errors:
        print(f"  load_init: {errors} files skipped due to errors")

    for prob in by_prob:
        by_prob[prob].sort(key=lambda a: a.ans_idx)
    return dict(by_prob)


def load_regen_answers(
    regen_dir: Path, dataset: str, model_name: str,
    regen_count: Optional[int] = None,
) -> Dict[Tuple[int, int], List[RegenAnswer]]:
    """Load regen answers from .txt file headers.

    Returns {(prob, ans_idx): [RegenAnswer, ...]}, sorted by regen_idx.
    """
    prefix = file_prefix(dataset, model_name)
    pattern = re.compile(rf"{re.escape(prefix)}_prob(\d+)_answer(\d+)_regen(\d+)\.txt$")
    by_key: Dict[Tuple[int, int], List[RegenAnswer]] = defaultdict(list)

    files = sorted(regen_dir.glob(f"{prefix}_prob*_answer*_regen*.txt"))
    t0 = time.time()
    errors = 0
    for i, fp in enumerate(files):
        m = pattern.match(fp.name)
        if not m:
            continue
        prob = int(m.group(1))
        ans_idx = int(m.group(2))
        it = int(m.group(3))
        if regen_count is not None and it >= regen_count:
            continue
        try:
            header = read_header(fp)
            ans, tok, ct, ft, conf, v0100, v0100_tok, v0100_min, bq_tok, bq_min = _read_common_fields(header)

            by_key[(prob, ans_idx)].append(RegenAnswer(
                regen_idx=it, answer=ans, tokens=tok,
                cot_tokens=ct, final_tokens=ft,
                init_tokens=_header_int(header, "Init Tokens"),
                kept_tokens=_header_int(header, "Kept Tokens"),
                cut_tokens=_header_int(header, "Cut Tokens"),
                logprob_conf=conf,
                resp_prob=_header_float(header, RESPONSE_PROB_HEADER),
                verbal_0_100_conf=v0100,
                verbal_0_100_actual_tokens=v0100_tok,
                verbal_0_100_min_tokens=v0100_min,
                verbal_binary_conf=_header_float(header, VERBAL_BINARY_CONF_HEADER),
                p_true_conf=_header_float(header, P_TRUE_CONF_HEADER),
                binary_query_actual_tokens=bq_tok,
                binary_query_min_tokens=bq_min,
            ))
        except Exception as e:
            errors += 1
            print(f"  ERROR {fp.name}: {e}")
        if (i + 1) % 500 == 0 or i + 1 == len(files):
            print(f"  load_regen: {i + 1}/{len(files)} ({time.time() - t0:.0f}s)")
    if errors:
        print(f"  load_regen: {errors} files skipped due to errors")

    for key in by_key:
        by_key[key].sort(key=lambda r: r.regen_idx)
    return dict(by_key)


def load_marker_regen_answers(
    marker_dir: Path, dataset: str, model_name: str,
) -> Dict[Tuple[int, int], List[MarkerRegenAnswer]]:
    """Load marker-based regen answers from .txt file headers.

    Returns {(prob, ans_idx): [MarkerRegenAnswer, ...]}.
    """
    prefix = file_prefix(dataset, model_name)
    pattern = re.compile(
        rf"{re.escape(prefix)}_prob(\d+)_answer(\d+)_marker(\d+)_regen(\d+)\.txt$")
    by_key: Dict[Tuple[int, int], List[MarkerRegenAnswer]] = defaultdict(list)

    files = sorted(marker_dir.glob(f"{prefix}_prob*_answer*_marker*_regen*.txt"))
    t0 = time.time()
    errors = 0
    for i, fp in enumerate(files):
        m = pattern.match(fp.name)
        if not m:
            continue
        prob = int(m.group(1))
        ans_idx = int(m.group(2))
        marker_idx = int(m.group(3))
        regen_idx = int(m.group(4))
        try:
            header = read_header(fp)
            ans, tok, ct, ft, conf, v0100, v0100_tok, v0100_min, bq_tok, bq_min = _read_common_fields(header)

            by_key[(prob, ans_idx)].append(MarkerRegenAnswer(
                marker_idx=marker_idx, regen_idx=regen_idx,
                answer=ans, tokens=tok, cot_tokens=ct, final_tokens=ft,
                init_tokens=_header_int(header, "Init Tokens"),
                kept_tokens=_header_int(header, "Kept Tokens"),
                cut_tokens=_header_int(header, "Cut Tokens"),
                marker_position=_header_int(header, "Marker Position"),
                logprob_conf=conf,
                resp_prob=_header_float(header, RESPONSE_PROB_HEADER),
                verbal_0_100_conf=v0100,
                verbal_0_100_actual_tokens=v0100_tok,
                verbal_0_100_min_tokens=v0100_min,
                verbal_binary_conf=_header_float(header, VERBAL_BINARY_CONF_HEADER),
                p_true_conf=_header_float(header, P_TRUE_CONF_HEADER),
                binary_query_actual_tokens=bq_tok,
                binary_query_min_tokens=bq_min,
            ))
        except Exception as e:
            errors += 1
            print(f"  ERROR {fp.name}: {e}")
        if (i + 1) % 500 == 0 or i + 1 == len(files):
            print(f"  load_marker_regen: {i + 1}/{len(files)} ({time.time() - t0:.0f}s)")
    if errors:
        print(f"  load_marker_regen: {errors} files skipped due to errors")

    for key in by_key:
        by_key[key].sort(key=lambda mr: (mr.marker_idx, mr.regen_idx))
    return dict(by_key)


# =====================================================================
# JSONL helpers
# =====================================================================

def write_jsonl_record(
    f, prob: int, gold: str, all_answers: list,
    dataset: str, model_name: str, **extra,
):
    """Write one JSONL record."""
    record = {
        "dataset": dataset,
        "model_name": model_name,
        "problem_num": prob,
        "gold_answer": gold,
        "all_answers": all_answers,
    }
    record.update(extra)
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _add_if_present(extra: dict, key: str, values: list):
    """Add values to extra dict if any element is not None."""
    if any(v is not None for v in values):
        extra[key] = values


def _add_if_present_nested(extra: dict, key: str, nested: list):
    """Add nested list to extra dict if any inner element is not None."""
    if any(v is not None for inner in nested for v in inner):
        extra[key] = nested


# =====================================================================
# Main conversion
# =====================================================================

def convert(
    dataset: str, model_name: str, init_dir: Path,
    regen_dir: Optional[Path] = None,
    marker_regen_dir: Optional[Path] = None,
    regen_count: Optional[int] = None,
    output: Optional[Path] = None,
    answer_filter: Optional[List[int]] = None,
    reasoning_effort: Optional[str] = None,
    no_think: bool = False,
):
    """Convert init + regen data into a single JSONL with regen_answers.

    Pure file conversion. All enrichment data is pre-computed by enrich.py
    and read from .txt headers.
    """
    prefix = file_prefix(dataset, model_name)
    golds = load_golds(init_dir)
    print(f"  Gold answers: from answer_map.json ({len(golds)} problems)")

    inits = load_init_answers(init_dir, dataset, model_name)
    if answer_filter is not None:
        answer_set = set(answer_filter)
        for prob in inits:
            inits[prob] = [a for a in inits[prob] if a.ans_idx in answer_set]
    regens = (
        load_regen_answers(regen_dir, dataset, model_name, regen_count=regen_count)
        if regen_dir else {}
    )
    marker_regens = (
        load_marker_regen_answers(marker_regen_dir, dataset, model_name)
        if marker_regen_dir else {}
    )

    problems = sorted(inits.keys())
    output.parent.mkdir(parents=True, exist_ok=True)

    # Trim to answers with consistent regen counts per problem.
    # When answer_filter is not set, auto-detect: keep only answers whose
    # regen count equals the mode (most common count) for that problem.
    if regens and answer_filter is None:
        for prob in problems:
            counts = [(a.ans_idx, len(regens.get((prob, a.ans_idx), [])))
                      for a in inits[prob]]
            unique = set(n for _, n in counts)
            if len(unique) > 1:
                # Find the most common regen count (mode)
                count_freq = Counter(n for _, n in counts)
                mode_count = count_freq.most_common(1)[0][0]
                kept = [a for a in inits[prob]
                        if len(regens.get((prob, a.ans_idx), [])) == mode_count]
                trimmed = len(inits[prob]) - len(kept)
                print(f"  Problem {prob}: trimmed {trimmed} answers with "
                      f"inconsistent regen counts (kept {len(kept)} with "
                      f"regen_count={mode_count})")
                inits[prob] = kept

    n_conf = 0
    n_verbal_0_100 = 0
    n_p_true = 0
    n_regen_verbal_0_100 = 0
    n_marker = 0
    with open(output, "w") as f:
        for prob in problems:
            gold = golds.get(prob, "")
            answers: List[InitAnswer] = inits[prob]

            all_answers = [[a.answer, a.tokens, a.cot_tokens, a.final_tokens]
                           for a in answers]

            # Regen data per init answer
            regen_lists = [regens.get((prob, a.ans_idx), []) for a in answers]

            extra = {}
            if regens:
                extra["regen_answers"] = [
                    [[r.answer, r.tokens, r.cot_tokens, r.final_tokens,
                      r.init_tokens, r.kept_tokens, r.cut_tokens]
                     for r in rl]
                    for rl in regen_lists
                ]
            if reasoning_effort is not None:
                extra["reasoning_effort"] = reasoning_effort
            if no_think:
                extra["no_think"] = True

            # DeepConf confidence
            confs = [a.logprob_conf for a in answers]
            if any(c is not None for c in confs):
                extra["confidences"] = confs
                n_conf += sum(1 for c in confs if c is not None)
                if regens:
                    extra["regen_confidences"] = [
                        [r.logprob_conf for r in rl] for rl in regen_lists]

            # Response probability (enrich step 2)
            _add_if_present(extra, RESPONSE_PROB_KEY,
                            [a.resp_prob for a in answers])
            if regens:
                _add_if_present_nested(
                    extra, "regen_response_probabilities",
                    [[r.resp_prob for r in rl] for rl in regen_lists])

            # Verbal 0-100 confidence (enrich step 3)
            verbal_0_100_vals = [a.verbal_0_100_conf for a in answers]
            if any(v is not None for v in verbal_0_100_vals):
                extra[VERBAL_0_100_CONF_KEY] = verbal_0_100_vals
                n_verbal_0_100 += sum(1 for v in verbal_0_100_vals if v is not None)
            _add_if_present(extra, VERBAL_0_100_ACTUAL_TOKENS_KEY,
                            [a.verbal_0_100_actual_tokens for a in answers])
            _add_if_present(extra, VERBAL_0_100_MIN_TOKENS_KEY,
                            [a.verbal_0_100_min_tokens for a in answers])
            if regens:
                regen_verbal_0_100 = [[r.verbal_0_100_conf for r in rl] for rl in regen_lists]
                if any(v is not None for rl in regen_verbal_0_100 for v in rl):
                    extra["regen_verbal_0_100_confidences"] = regen_verbal_0_100
                    n_regen_verbal_0_100 += sum(1 for rl in regen_verbal_0_100
                                       for v in rl if v is not None)

            # Verbal binary confidence (enrich step 4)
            _add_if_present(extra, VERBAL_BINARY_CONF_KEY,
                            [a.verbal_binary_conf for a in answers])
            if regens:
                _add_if_present_nested(
                    extra, "regen_verbal_binary_confidences",
                    [[r.verbal_binary_conf for r in rl] for rl in regen_lists])

            # Binary query tokens (enrich step 4, shared by verbal_binary and p_true)
            _add_if_present(extra, BINARY_QUERY_ACTUAL_TOKENS_KEY,
                            [a.binary_query_actual_tokens for a in answers])
            _add_if_present(extra, BINARY_QUERY_MIN_TOKENS_KEY,
                            [a.binary_query_min_tokens for a in answers])

            # P(True) confidence (enrich step 4)
            pt_vals = [a.p_true_conf for a in answers]
            if any(v is not None for v in pt_vals):
                extra[P_TRUE_CONF_KEY] = pt_vals
                n_p_true += sum(1 for v in pt_vals if v is not None)
            if regens:
                _add_if_present_nested(
                    extra, "regen_p_true_confidences",
                    [[r.p_true_conf for r in rl] for rl in regen_lists])

            # Marker-based regen answers (Hammoud et al., 2025)
            if marker_regens:
                mr_lists = [marker_regens.get((prob, a.ans_idx), [])
                            for a in answers]
                extra["marker_regen_answers"] = [
                    [[mr.answer, mr.tokens, mr.cot_tokens, mr.final_tokens,
                      mr.marker_idx, mr.regen_idx, mr.marker_position,
                      mr.init_tokens, mr.kept_tokens, mr.cut_tokens]
                     for mr in ml]
                    for ml in mr_lists
                ]
                _add_if_present_nested(
                    extra, "marker_regen_confidences",
                    [[mr.logprob_conf for mr in ml] for ml in mr_lists])
                _add_if_present_nested(
                    extra, "marker_regen_response_probabilities",
                    [[mr.resp_prob for mr in ml] for ml in mr_lists])
                _add_if_present_nested(
                    extra, "marker_regen_verbal_0_100_confidences",
                    [[mr.verbal_0_100_conf for mr in ml] for ml in mr_lists])
                _add_if_present_nested(
                    extra, "marker_regen_verbal_binary_confidences",
                    [[mr.verbal_binary_conf for mr in ml] for ml in mr_lists])
                _add_if_present_nested(
                    extra, "marker_regen_p_true_confidences",
                    [[mr.p_true_conf for mr in ml] for ml in mr_lists])
                n_marker += sum(len(ml) for ml in mr_lists)

            write_jsonl_record(f, prob, gold, all_answers,
                               dataset=dataset, model_name=model_name, **extra)

    total_answers = sum(len(inits[p]) for p in problems)
    if n_conf > 0:
        print(f"  DeepConf confidence: {n_conf}/{total_answers} answers")
    if n_verbal_0_100 > 0:
        print(f"  Verbal 0-100 confidence: {n_verbal_0_100}/{total_answers} answers")
    if n_p_true > 0:
        print(f"  P(True) confidence: {n_p_true}/{total_answers} answers")
    if n_regen_verbal_0_100 > 0:
        print(f"  Regen verbal 0-100 confidence: {n_regen_verbal_0_100} regen answers")
    if n_marker > 0:
        print(f"  Marker regen answers: {n_marker} total")

    n_with_regen = sum(1 for p in problems for a in inits[p]
                       if regens.get((p, a.ans_idx)))
    print(f"Written: {output}")
    print(f"  {len(problems)} problems, {total_answers} total answers")
    if regens:
        print(f"  {n_with_regen} answers have regen data")


# =====================================================================
# CLI
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description="Convert regeneration data to evaluation-ready JSONL format "
                    "(no GPU needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    p.add_argument("--model-name", required=True, choices=MODEL_NAMES)
    p.add_argument("--init-dir", required=True, type=Path,
                   help="Directory with initial answer files")
    p.add_argument("--regen-dir", type=Path, default=None,
                   help="Regen output directory (optional)")
    p.add_argument("--marker-regen-dir", type=Path, default=None,
                   help="Marker-based regen directory (Hammoud et al., 2025)")
    p.add_argument("--regen-count", type=int, default=None,
                   help="Number of regen iterations to include")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSONL path")
    p.add_argument("--answers", type=str, default=None,
                   help="Answer indices filter (e.g. 0-65 or 0,1,2)")
    p.add_argument("--reasoning-effort", type=str, default=None,
                   help="Reasoning effort level to record in JSONL")
    p.add_argument("--no-think", action="store_true",
                   help="Record no-think flag in JSONL")

    args = p.parse_args()
    convert(args.dataset, args.model_name, args.init_dir,
            regen_dir=args.regen_dir,
            marker_regen_dir=args.marker_regen_dir,
            regen_count=args.regen_count,
            output=args.output,
            answer_filter=parse_int_list(args.answers),
            reasoning_effort=args.reasoning_effort,
            no_think=args.no_think)


if __name__ == "__main__":
    main()
