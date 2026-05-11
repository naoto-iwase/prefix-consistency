#!/usr/bin/env python3
"""
Phase 3 enrichment: write metadata to .txt file headers and answer_map.json.

All enrichment results are stored in .txt headers (per-file, resume-safe) and
answer_map.json (golds + LLM judge cache).

Runs five steps sequentially:
  Step 1 (extracted answer): Extract answers from generated text via regex.
  Step 2 (logprob confidence): Read .npy files, write DeepConf metrics to headers.
  Step 3 (verbal 0-100): Query LLM for self-rated confidence 0-100.
  Step 4 (binary confidence): Single API call with "0 or 1" prompt, writes both
         Verbal Binary (text parse) and P(True) (logprobs) headers.
  Step 5 (answer map): Write Canonical Answer headers and build answer_map.json.

Steps 1, 2, and 5 need no GPU (except judge datasets for step 5).
Steps 3 and 4 require a running vLLM/SGLang server.

Usage:
    uv run python enrich.py \
        --dataset aime2025 --model-name gpt-oss-20b \
        --init-dir data/init \
        --data-dirs data/init data/regen_rm50pct_full \
        --regen-dirs data/regen_rm50pct_full \
        --parallel 32

    # Skip verbal 0-100 and binary confidence:
    uv run python enrich.py \
        --dataset aime2025 --model-name gpt-oss-20b \
        --init-dir data/init \
        --data-dirs data/init \
        --skip-verbal-0-100 --skip-binary --skip-answer-map
"""

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer

from core.config import (
    BASE_URL, DATASET_NAMES, MODEL_NAMES,
    get_dataset_config, build_model_config,
)
from core.answer_extraction import extract_answer, EXTRACTED_ANSWER_HEADER, PARSE_FAILED
from core.logprob_confidence import compute_all_confidences, CONFIDENCE_HEADERS
from core.verbalized_confidence import (
    RESPONSE_PROB_HEADER,
    VERBAL_0_100_CONF_HEADER, VERBAL_0_100_ACTUAL_TOKENS_HEADER, VERBAL_0_100_MIN_TOKENS_HEADER,
    VERBAL_BINARY_CONF_HEADER, BINARY_QUERY_ACTUAL_TOKENS_HEADER, BINARY_QUERY_MIN_TOKENS_HEADER,
    P_TRUE_CONF_HEADER,
    query_verbal_0_100, query_binary_confidence,
)
from core.io import (
    TOP_LOGPROBS_SUFFIX, TOKEN_LOGPROBS_SUFFIX, has_header_field,
    read_answer_text, read_problems, write_header_fields,
)
from core.text import build_rendered_prompt, file_prefix, parse_int_list
from core.answer_map import build_answer_map


# =====================================================================
# Step 1/5: Extracted answer
# =====================================================================

def run_extracted_answer(
    dataset: str,
    data_dirs: List[Path],
    force: bool = False,
):
    """Extract answers from generated text and write to .txt headers."""
    print("=== Step 1/5: Extracted answer ===")
    print(f"  dataset:   {dataset}")
    print(f"  data-dirs: {data_dirs}")

    tasks: List[Path] = []
    skipped = 0
    for d in data_dirs:
        for fp in sorted(d.glob("*.txt")):
            if not force and has_header_field(fp, EXTRACTED_ANSWER_HEADER):
                skipped += 1
                continue
            tasks.append(fp)

    print(f"  Found {len(tasks)} files to process, {skipped} already done")
    if not tasks:
        return

    t0 = time.time()
    updated, no_answer, error_count = 0, 0, 0
    for i, fp in enumerate(tasks):
        try:
            text = read_answer_text(fp)
            ans = extract_answer(text, dataset)
            if ans is None:
                no_answer += 1
                ans = PARSE_FAILED
            write_header_fields(fp, {EXTRACTED_ANSWER_HEADER: ans})
            updated += 1
        except Exception as e:
            error_count += 1
            print(f"  ERROR {fp.name}: {e}")

        if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
            elapsed = time.time() - t0
            print(f"  Progress: {i + 1}/{len(tasks)} "
                  f"(updated={updated}, no_answer={no_answer}, "
                  f"err={error_count}, {elapsed:.0f}s)")

    print(f"  Done: {updated} updated, {no_answer} no answer, "
          f"{error_count} errors, {skipped} skipped")


# =====================================================================
# Step 2/5: Logprob confidence (DeepConf)
# =====================================================================

# All 5 metrics are written atomically, so checking one is sufficient for resume.
_CONF_SENTINEL = CONFIDENCE_HEADERS["mean_conf"]


def run_logprob_confidence(
    model_name: str,
    data_dirs: List[Path],
    force: bool = False,
):
    """Read .npy files and write DeepConf confidence metrics to .txt headers."""
    print("=== Step 2/5: Logprob confidence (DeepConf) ===")
    print(f"  model-name: {model_name}")
    print(f"  data-dirs:  {data_dirs}")

    tasks: List[Path] = []
    skipped = 0
    for d in data_dirs:
        for fp in sorted(d.glob("*.txt")):
            npy_path = Path(str(fp) + TOP_LOGPROBS_SUFFIX)
            if not npy_path.exists():
                continue
            if not force and has_header_field(fp, _CONF_SENTINEL):
                skipped += 1
                continue
            tasks.append(fp)

    print(f"  Found {len(tasks)} files to process, {skipped} already done")
    if not tasks:
        return

    t0 = time.time()
    success_count, error_count = 0, 0

    for i, fp in enumerate(tasks):
        try:
            npy_path = str(fp) + TOP_LOGPROBS_SUFFIX
            text = read_answer_text(fp)
            metrics = compute_all_confidences(
                npy_path, text, model_name=model_name)

            fields = {}
            for metric_key, header_key in CONFIDENCE_HEADERS.items():
                val = metrics[metric_key]
                fields[header_key] = f"{val:.6f}"

            write_header_fields(fp, fields)
            success_count += 1
        except Exception as e:
            error_count += 1
            print(f"  ERROR {fp.name}: {e}")

        if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
            elapsed = time.time() - t0
            print(f"  Progress: {i + 1}/{len(tasks)} "
                  f"(ok={success_count}, err={error_count}, {elapsed:.0f}s)")

    print(f"  Done: {success_count} written, {error_count} errors, "
          f"{skipped} skipped")

    # Response Probability (from .tok_logprobs.npy, written at end of step 2
    # so the header appears right before the step 3/4 confidence headers)
    rp_tasks: List[Path] = []
    rp_skipped = 0
    for d in data_dirs:
        for fp in sorted(d.glob("*.txt")):
            tok_lp_path = Path(str(fp) + TOKEN_LOGPROBS_SUFFIX)
            if not tok_lp_path.exists():
                continue
            if not force and has_header_field(fp, RESPONSE_PROB_HEADER):
                rp_skipped += 1
                continue
            rp_tasks.append(fp)

    if rp_tasks:
        print(f"  Response Probability: {len(rp_tasks)} files, "
              f"{rp_skipped} already done")
        rp_ok, rp_err = 0, 0
        for fp in rp_tasks:
            try:
                arr = np.load(str(fp) + TOKEN_LOGPROBS_SUFFIX)
                resp_prob = float(np.exp(np.mean(arr)))
                write_header_fields(fp, {RESPONSE_PROB_HEADER: f"{resp_prob:.8f}"})
                rp_ok += 1
            except Exception as e:
                rp_err += 1
                print(f"  ERROR {fp.name}: {e}")
        print(f"  Response Probability done: {rp_ok} written, {rp_err} errors")
    elif rp_skipped > 0:
        print(f"  Response Probability: {rp_skipped} already done")


# =====================================================================
# Step 3/5: Verbal 0-100 confidence
# =====================================================================


def _collect_confidence_tasks(
    dirs: List[Path],
    prefix: str,
    pattern: re.Pattern,
    problem_indices: set,
    rendered_prompts: Dict[int, str],
    force: bool,
    header_to_check: str,
) -> Tuple[List[Tuple[Path, int, int]], int]:
    """Scan directories for .txt files needing confidence (verbalized or P(True)).

    Returns (tasks, skipped_count).
    """
    tasks: List[Tuple[Path, int, int]] = []
    skipped = 0
    for d in dirs:
        for fp in sorted(d.glob(f"{prefix}_prob*_answer*.txt")):
            m = pattern.match(fp.name)
            if not m:
                continue
            prob = int(m.group(1))
            ans_idx = int(m.group(2))
            if prob not in problem_indices:
                continue
            if prob not in rendered_prompts:
                continue
            if not force and has_header_field(fp, header_to_check):
                skipped += 1
                continue
            tasks.append((fp, prob, ans_idx))
    return tasks, skipped


def run_verbal_0_100_confidence(
    dataset: str, model_name: str,
    init_dir: Path, regen_dirs: List[Path],
    model_path: str, model_info: Dict, api_params: Dict,
    parallel: int = 32, timeout: int = 600,
    problem_filter: str = None, force: bool = False,
):
    """Query LLM for verbal 0-100 confidence and write to .txt headers.

    By default only processes init files. When regen_dirs is provided,
    also processes regen files (both _regen{I}.txt and _marker{B}_regen{R}.txt).
    """
    print("=== Step 3/5: Verbal 0-100 confidence ===")
    print(f"  dataset:    {dataset}")
    print(f"  model-name: {model_name}")
    print(f"  init-dir:   {init_dir}")
    if regen_dirs:
        print(f"  regen-dirs: {regen_dirs}")
    print(f"  parallel:   {parallel}")

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset_cfg = get_dataset_config(dataset)
    problems = read_problems(dataset_cfg["data_file"])
    prob_indices = parse_int_list(problem_filter)
    if prob_indices is not None:
        problem_indices = [i for i in prob_indices if 0 <= i < len(problems)]
    else:
        problem_indices = list(range(len(problems)))
    problem_indices_set = set(problem_indices)

    rendered_prompts: Dict[int, str] = {}
    for i in problem_indices:
        rendered, _ = build_rendered_prompt(
            problems[i], dataset_cfg, model_info, tokenizer)
        rendered_prompts[i] = rendered

    prefix = file_prefix(dataset, model_name)

    # Init files: match {prefix}_prob{P}_answer{A}.txt (exclude regen/marker)
    init_pattern = re.compile(rf"{re.escape(prefix)}_prob(\d+)_answer(\d+)\.txt$")
    tasks, skipped = _collect_confidence_tasks(
        [init_dir], prefix, init_pattern, problem_indices_set,
        rendered_prompts, force, VERBAL_0_100_CONF_HEADER)

    # Regen files: match both _regen{I}.txt and _marker{B}_regen{R}.txt
    if regen_dirs:
        regen_pattern = re.compile(
            rf"{re.escape(prefix)}_prob(\d+)_answer(\d+)(?:_regen\d+|_marker\d+_regen\d+)\.txt$"
        )
        regen_tasks, regen_skipped = _collect_confidence_tasks(
            regen_dirs, prefix, regen_pattern, problem_indices_set,
            rendered_prompts, force, VERBAL_0_100_CONF_HEADER)
        tasks.extend(regen_tasks)
        skipped += regen_skipped

    print(f"  Found {len(tasks)} files to process, {skipped} already done")
    if not tasks:
        return

    client = OpenAI(base_url=BASE_URL, api_key="dummy-key", timeout=timeout)
    success_count, error_count = 0, 0
    max_workers = min(parallel, len(tasks)) if tasks else 1

    def _query_verbal(fp, prob, ans_idx):
        answer_text = read_answer_text(fp)
        conf, _, n_tokens, min_tokens = query_verbal_0_100(
            client, model_name, rendered_prompts[prob], answer_text,
            model_info, api_params, tokenizer)
        return conf, n_tokens, min_tokens

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_query_verbal, fp, prob, ans_idx): (fp, prob, ans_idx)
            for fp, prob, ans_idx in tasks
        }
        t0 = time.time()
        done = 0
        for future in as_completed(futures):
            fp, prob, ans_idx = futures[future]
            done += 1
            try:
                conf, n_tokens, min_tokens = future.result()
                headers = {VERBAL_0_100_CONF_HEADER:
                            str(int(conf)) if conf is not None else PARSE_FAILED}
                if n_tokens:
                    headers[VERBAL_0_100_ACTUAL_TOKENS_HEADER] = str(n_tokens)
                if min_tokens is not None:
                    headers[VERBAL_0_100_MIN_TOKENS_HEADER] = str(min_tokens)
                write_header_fields(fp, headers)
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"  ERROR prob{prob}_ans{ans_idx}: {e}")
                continue
            if done % 200 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                print(f"  Progress: {done}/{len(tasks)} "
                      f"(ok={success_count}, err={error_count}, {elapsed:.0f}s)")

    print(f"  Done: {success_count} written, {error_count} errors, "
          f"{skipped} skipped")


# =====================================================================
# Step 4/5: Binary confidence (Verbal Binary + P(True) in one API call)
# =====================================================================


def run_binary_confidence(
    dataset: str, model_name: str,
    init_dir: Path, regen_dirs: List[Path],
    model_path: str, model_info: Dict, api_params: Dict,
    parallel: int = 32, timeout: int = 600,
    problem_filter: str = None, force: bool = False,
):
    """Query LLM once per answer with "0 or 1" prompt, write both headers.

    Single API call extracts:
      - Verbal Binary: parse 0/1 from generated text
      - P(True): softmax P("1") from logprobs at first final-answer token
    """
    print("=== Step 4/5: Binary confidence (Verbal Binary + P(True)) ===")
    print(f"  dataset:    {dataset}")
    print(f"  model-name: {model_name}")
    print(f"  init-dir:   {init_dir}")
    print(f"  parallel:   {parallel}")

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset_cfg = get_dataset_config(dataset)
    problems = read_problems(dataset_cfg["data_file"])
    prob_indices = parse_int_list(problem_filter)
    if prob_indices is not None:
        problem_indices = [i for i in prob_indices if 0 <= i < len(problems)]
    else:
        problem_indices = list(range(len(problems)))
    problem_indices_set = set(problem_indices)

    rendered_prompts: Dict[int, str] = {}
    for i in problem_indices:
        rendered, _ = build_rendered_prompt(
            problems[i], dataset_cfg, model_info, tokenizer)
        rendered_prompts[i] = rendered

    prefix = file_prefix(dataset, model_name)

    # Init files
    init_pattern = re.compile(rf"{re.escape(prefix)}_prob(\d+)_answer(\d+)\.txt$")
    tasks, skipped = _collect_confidence_tasks(
        [init_dir], prefix, init_pattern, problem_indices_set,
        rendered_prompts, force, VERBAL_BINARY_CONF_HEADER)

    # Regen files (if --regen-binary)
    if regen_dirs:
        regen_pattern = re.compile(
            rf"{re.escape(prefix)}_prob(\d+)_answer(\d+)(?:_regen\d+|_marker\d+_regen\d+)\.txt$"
        )
        regen_tasks, regen_skipped = _collect_confidence_tasks(
            regen_dirs, prefix, regen_pattern, problem_indices_set,
            rendered_prompts, force, VERBAL_BINARY_CONF_HEADER)
        tasks.extend(regen_tasks)
        skipped += regen_skipped

    print(f"  Found {len(tasks)} files to process, {skipped} already done")
    if not tasks:
        return

    client = OpenAI(base_url=BASE_URL, api_key="dummy-key", timeout=timeout)
    success_count, error_count = 0, 0
    max_workers = min(parallel, len(tasks)) if tasks else 1

    def _query_binary(fp, prob, ans_idx):
        answer_text = read_answer_text(fp)
        return query_binary_confidence(
            client, model_name, rendered_prompts[prob], answer_text,
            model_info, api_params, tokenizer)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_query_binary, fp, prob, ans_idx): (fp, prob, ans_idx)
            for fp, prob, ans_idx in tasks
        }
        t0 = time.time()
        done = 0
        for future in as_completed(futures):
            fp, prob, ans_idx = futures[future]
            done += 1
            try:
                vb_conf, pt_conf, _, n_tokens, min_tokens = future.result()
                headers = {
                    VERBAL_BINARY_CONF_HEADER:
                        f"{vb_conf:.0f}" if vb_conf is not None else PARSE_FAILED,
                    P_TRUE_CONF_HEADER:
                        f"{pt_conf:.6f}" if pt_conf is not None else PARSE_FAILED,
                }
                if n_tokens:
                    headers[BINARY_QUERY_ACTUAL_TOKENS_HEADER] = str(n_tokens)
                if min_tokens is not None:
                    headers[BINARY_QUERY_MIN_TOKENS_HEADER] = str(min_tokens)
                write_header_fields(fp, headers)
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"  ERROR prob{prob}_ans{ans_idx}: {e}")
                continue
            if done % 200 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                print(f"  Progress: {done}/{len(tasks)} "
                      f"(ok={success_count}, err={error_count}, {elapsed:.0f}s)")

    print(f"  Done: {success_count} written, {error_count} errors, "
          f"{skipped} skipped")


# =====================================================================
# CLI
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description="Phase 3 enrichment: extracted answer, logprob confidence, "
                    "verbal 0-100, verbal binary, P(True), and answer map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    p.add_argument("--model-name", required=True, choices=MODEL_NAMES)
    p.add_argument("--init-dir", required=True, type=Path,
                   help="Init answers directory")
    p.add_argument("--data-dirs", nargs="+", required=True, type=Path,
                   help="Directories with .txt/.npy files (for logprob confidence)")
    p.add_argument("--regen-dirs", nargs="*", type=Path, default=[],
                   help="Regen directories (for answer map scope)")
    p.add_argument("--parallel", type=int, default=32)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--model-path", default=None,
                   help="Tokenizer path override")
    p.add_argument("--problems", type=str, default=None,
                   help="Problem filter for verbalized confidence (e.g. 0-4)")
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                   default=None)
    p.add_argument("--force", action="store_true",
                   help="Re-compute even if already done")
    p.add_argument("--answer-map-output", type=Path, default=None,
                   help="Override answer_map.json output path")
    p.add_argument("--judge-backend", choices=["local", "openai", "anthropic"],
                   default="local",
                   help="Judge backend. openai/anthropic read OPENAI_API_KEY / "
                        "ANTHROPIC_API_KEY and require --judge-model.")
    p.add_argument("--judge-model", default=None,
                   help="Judge model id. Defaults to --model-name for local.")
    p.add_argument("--skip-extraction", action="store_true",
                   help="Skip extracted answer (step 1)")
    p.add_argument("--skip-confidence", action="store_true",
                   help="Skip logprob confidence (step 2)")
    p.add_argument("--skip-verbal-0-100", action="store_true",
                   help="Skip verbal 0-100 confidence (step 3)")
    p.add_argument("--skip-binary", action="store_true",
                   help="Skip binary confidence: Verbal Binary + P(True) (step 4)")
    p.add_argument("--skip-answer-map", action="store_true",
                   help="Skip answer map building (step 5)")
    p.add_argument("--regen-verbal-0-100", action="store_true",
                   help="Also run verbal 0-100 confidence on regen files "
                        "(in --regen-dirs)")
    p.add_argument("--regen-binary", action="store_true",
                   help="Also run binary confidence (Verbal Binary + P(True)) on regen files "
                        "(in --regen-dirs)")
    p.add_argument("--marker-regen-dirs", nargs="*", type=Path, default=[],
                   help="Marker-based regen directories (for verbal/binary confidence)")

    args = p.parse_args()

    model_info, api_params = build_model_config(
        args.model_name, no_think=args.no_think,
        reasoning_effort=args.reasoning_effort)
    model_path = args.model_path or model_info["default_model_path"]

    print("=== enrich.py: Phase 3 enrichment ===")
    print(f"  dataset:    {args.dataset}")
    print(f"  model-name: {args.model_name}")
    print(f"  init-dir:   {args.init_dir}")
    print(f"  data-dirs:  {args.data_dirs}")
    print(f"  regen-dirs: {args.regen_dirs}")
    print()

    # Step 1/5: Extracted answer
    if not args.skip_extraction:
        run_extracted_answer(
            dataset=args.dataset,
            data_dirs=args.data_dirs,
            force=args.force,
        )
    else:
        print("=== Step 1/5: Extracted answer — SKIPPED ===")
    print()

    # Step 2/5: Logprob confidence
    if not args.skip_confidence:
        run_logprob_confidence(
            model_name=args.model_name,
            data_dirs=args.data_dirs,
            force=args.force,
        )
    else:
        print("=== Step 2/5: Logprob confidence — SKIPPED ===")
    print()

    # Shared regen dirs for steps 3 and 4
    all_regen_dirs = list(args.regen_dirs) + list(args.marker_regen_dirs)

    # Step 3/5: Verbal 0-100 confidence
    if not args.skip_verbal_0_100:
        run_verbal_0_100_confidence(
            args.dataset, args.model_name,
            args.init_dir, all_regen_dirs if args.regen_verbal_0_100 else [],
            model_path, model_info, api_params,
            parallel=args.parallel, timeout=args.timeout,
            problem_filter=args.problems, force=args.force,
        )
    else:
        print("=== Step 3/5: Verbal 0-100 confidence — SKIPPED ===")
    print()

    # Step 4/5: Binary confidence (Verbal Binary + P(True) in one call)
    if not args.skip_binary:
        run_binary_confidence(
            args.dataset, args.model_name,
            args.init_dir, all_regen_dirs if args.regen_binary else [],
            model_path, model_info, api_params,
            parallel=args.parallel, timeout=args.timeout,
            problem_filter=args.problems, force=args.force,
        )
    else:
        print("=== Step 4/5: Binary confidence — SKIPPED ===")
    print()

    # Step 5/5: Answer map
    if not args.skip_answer_map:
        print("=== Step 5/5: Answer map ===")
        if args.judge_backend != "local" and args.judge_model is None:
            p.error(f"--judge-model is required when "
                    f"--judge-backend={args.judge_backend}")
        build_answer_map(
            dataset=args.dataset,
            model_name=args.model_name,
            init_dir=args.init_dir,
            regen_dirs=list(args.regen_dirs) + list(args.marker_regen_dirs),
            output=args.answer_map_output,
            judge_backend=args.judge_backend,
            judge_model=args.judge_model,
            reasoning_effort=args.reasoning_effort,
            no_think=args.no_think,
            timeout=args.timeout,
            force=args.force,
        )
    else:
        print("=== Step 5/5: Answer map — SKIPPED ===")
    print()

    print("=== Enrichment complete ===")


if __name__ == "__main__":
    main()
