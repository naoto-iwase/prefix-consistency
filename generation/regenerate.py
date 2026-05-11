#!/usr/bin/env python3
"""
Token-based percentage truncation + vLLM regeneration.
For each source answer file, truncate the CoT at a specified token percentage
and regenerate K times from the truncation point.

Usage:
    uv run python regenerate.py \
        --dataset aime2025 --model-name gpt-oss-20b \
        --init-dir generation/data/initial \
        --remove-pct 50 --regen-count 3 \
        --out-dir generation/data/regen_rm50pct_full

    # Remove by token count:
    uv run python regenerate.py \
        --dataset aime2025 --model-name gpt-oss-20b \
        --init-dir generation/data/initial \
        --remove-tokens 500 --regen-count 3 \
        --out-dir generation/data/regen_rm500tok_full

    # Apply removal relative to CoT only:
    uv run python regenerate.py \
        --dataset aime2025 --model-name gpt-oss-20b \
        --init-dir generation/data/initial \
        --remove-pct 50 --truncate-scope cot --regen-count 3 \
        --out-dir generation/data/regen_rm50pct_cot

    # Force conclusion at truncation point (insert CoT closing delimiter):
    uv run python regenerate.py \
        --dataset aime2025 --model-name gpt-oss-20b \
        --init-dir generation/data/initial \
        --remove-pct 50 --insert-cot-closing --regen-count 3 \
        --out-dir generation/data/regen_rm50pct_full_close
"""

import argparse
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from transformers import AutoTokenizer

from core.config import (
    BASE_URL, MIN_FILE_SIZE_BYTES, TOP_LOGPROBS,
    DATASET_NAMES, MODEL_NAMES,
    get_dataset_config, build_model_config,
)
from core.api import call_completions
from core.io import (
    TOP_LOGPROBS_SUFFIX,
    read_answer_text, read_problems,
    write_answer, write_logprobs, write_metadata, write_token_logprobs,
)
from core.text import (
    file_prefix,
    get_cot_to_final, split_cot_and_final,
    build_rendered_prompt, count_cot_and_final_tokens,
    parse_int_list,
)


# =====================================================================
# Token-based truncation
# =====================================================================
def truncate_generation(
    generated_text: str, model_cfg: Dict, tokenizer,
    remove_pct: Optional[int], remove_tokens: Optional[int],
    scope: str,
) -> Tuple[str, int]:
    """Truncate a generation and return (kept_text, cut_tokens).

    scope="cot":  split into CoT/final, apply removal to CoT token count.
    scope="full": strip cot_prefix, apply removal to total token count.

    The returned text excludes cot_prefix (caller prepends it to the prompt).
    For no-think models (cot_prefix is None), scope="full" works on the raw text.
    Raises ValueError if CoT extraction fails (scope="cot" only).
    """
    if scope == "full":
        cot_prefix = model_cfg["cot_prefix"]
        text = generated_text
        if cot_prefix and text.startswith(cot_prefix):
            text = text[len(cot_prefix):]
    elif scope == "cot":
        cot, _ = split_cot_and_final(generated_text, model_cfg)
        if cot is None:
            raise ValueError("CoT extraction failed")
        text = cot
    else:
        raise ValueError(f"Unknown truncate scope: {scope}")

    total_len = len(tokenizer.encode(text, add_special_tokens=False))
    if remove_pct is not None:
        remove_len = int(total_len * remove_pct / 100)
    else:
        remove_len = remove_tokens
    keep = max(0, total_len - remove_len)
    if keep >= total_len:
        kept_text = text
    elif keep <= 0:
        kept_text = ""
    else:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        kept_text = tokenizer.decode(tokens[:keep])
    return kept_text, total_len - keep


# =====================================================================
# Source file enumeration
# =====================================================================
def list_source_files(
    init_dir: Path, file_prefix: str, problem_filter: Optional[List[int]], answer_filter: Optional[List[int]]
) -> List[Tuple[Path, int, int]]:
    pattern = re.compile(rf"{re.escape(file_prefix)}_prob(\d+)_answer(\d+)\.txt$")
    results = []
    for p in init_dir.glob(f"{file_prefix}_prob*_answer*.txt"):
        m = pattern.match(p.name)
        if not m:
            continue
        prob_idx = int(m.group(1))
        ans_idx = int(m.group(2))
        if problem_filter is not None and prob_idx not in problem_filter:
            continue
        if answer_filter is not None and ans_idx not in answer_filter:
            continue
        results.append((p, prob_idx, ans_idx))
    results.sort(key=lambda x: (x[1], x[2]))
    return results


# =====================================================================
# Worker function
# =====================================================================
def do_regen(
    client: OpenAI,
    dataset: str,
    model_name: str,
    model_info: Dict,
    api_params: Dict,
    tokenizer,
    problems: List[Dict],
    rendered_prompts: Dict[int, Tuple[str, str]],
    source_file: Path,
    problem_index: int,
    answer_index: int,
    iteration: int,
    remove_pct: Optional[int],
    remove_tokens: Optional[int],
    truncate_scope: str,
    out_dir: Path,
    insert_cot_closing: bool = False,
    save_logprobs: bool = False,
) -> Tuple[bool, int, int, int, int, Optional[str]]:
    """Regenerate one answer from a truncation point. Returns (success, prob, ans, iter, tokens, error)."""
    try:
        problem = problems[problem_index]
        dataset_cfg = get_dataset_config(dataset)

        # Skip if file already exists and is large enough (resumability)
        prefix = file_prefix(dataset, model_name)
        fp = out_dir / f"{prefix}_prob{problem_index}_answer{answer_index}_regen{iteration}.txt"
        npy_path = Path(str(fp) + TOP_LOGPROBS_SUFFIX)
        if fp.exists() and fp.stat().st_size >= MIN_FILE_SIZE_BYTES:
            if not save_logprobs or npy_path.exists():
                return True, problem_index, answer_index, iteration, 0, None

        # Read source and truncate
        generated_text = read_answer_text(source_file)
        if not generated_text.strip():
            return False, problem_index, answer_index, iteration, 0, "Empty source file"

        init_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
        kept_text, cut_tokens = truncate_generation(
            generated_text, model_info, tokenizer,
            remove_pct, remove_tokens, truncate_scope,
        )
        kept_tokens = init_tokens - cut_tokens

        # Build prompt
        rendered, prompt_text = rendered_prompts[problem_index]
        cot_prefix = model_info["cot_prefix"] or ""
        cot_to_final = get_cot_to_final(model_info)

        # Truncation cut into the CoT region?
        cut_into_cot = cot_to_final is not None and cot_to_final not in kept_text
        # Insert CoT closing delimiter only when: --insert-cot-closing is set,
        # model has CoT delimiters, and truncation actually cut into the CoT.
        should_close = insert_cot_closing and cut_into_cot

        closing = cot_to_final if should_close else ""
        regen_prompt = f"{rendered}{cot_prefix}{kept_text}{closing}"
        adjusted_max_tokens = max(cut_tokens * 2, dataset_cfg["complete_max_tokens"] // 10)

        continuation, comp_tokens, raw_top_logprobs, token_logprobs = call_completions(
            client, model_name, regen_prompt, adjusted_max_tokens,
            model_info["max_context_length"], api_params,
            tokenizer, top_logprobs=TOP_LOGPROBS if save_logprobs else 0,
        )

        # Continuation starts from CoT only when: thinking model,
        # truncation cut into CoT, and we didn't close it.
        start_from_cot = cut_into_cot and not insert_cot_closing
        cot_tokens, final_tokens = count_cot_and_final_tokens(
            continuation, model_info, tokenizer, start_from_cot,
        )

        gold = problem[dataset_cfg["gold_key"]]
        kept_answer = f"{cot_prefix}{kept_text}{closing}"
        write_answer(fp, dataset, problem_index, answer_index, gold, prompt_text, kept_answer, comp_tokens,
                    cot_tokens=cot_tokens, final_tokens=final_tokens,
                    regenerated=continuation,
                    extra_headers={
                        "Init Tokens": init_tokens,
                        "Kept Tokens": kept_tokens,
                        "Cut Tokens": cut_tokens,
                    })

        if save_logprobs and raw_top_logprobs:
            write_logprobs(npy_path, raw_top_logprobs, TOP_LOGPROBS)
        if save_logprobs and token_logprobs:
            write_token_logprobs(fp, token_logprobs)

        return True, problem_index, answer_index, iteration, comp_tokens, None

    except Exception as e:
        error_msg = f"prob{problem_index}_ans{answer_index}_regen{iteration}: {e}"
        traceback.print_exc()
        return False, problem_index, answer_index, iteration, 0, error_msg


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Mid-regen: token-based truncation + vLLM regeneration")
    p.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    p.add_argument("--model-name", required=True, choices=MODEL_NAMES)
    p.add_argument("--model-path", default=None, help="Tokenizer path (default: auto-detected from config)")
    p.add_argument("--init-dir", required=True, type=Path, help="Directory with initial answer files")
    remove = p.add_mutually_exclusive_group(required=True)
    remove.add_argument("--remove-pct", type=int, help="Percentage of tokens to remove (0-100)")
    remove.add_argument("--remove-tokens", type=int, help="Number of tokens to remove")
    p.add_argument("--truncate-scope", choices=["cot", "full"], default="full",
                    help="Apply removal to full generation (default) or CoT only")
    p.add_argument("--regen-count", required=True, type=int, help="Number of regenerations per answer")
    p.add_argument("--insert-cot-closing", action="store_true",
                    help="Insert CoT closing delimiter at truncation point to force conclusion")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory for regenerated files")
    p.add_argument("--parallel", type=int, default=32, help="Max parallel workers")
    p.add_argument("--timeout", type=int, default=1800, help="OpenAI client timeout in seconds")
    p.add_argument("--problems", type=str, default=None, help="Problem indices filter (e.g. 0,1,2 or 0-4)")
    p.add_argument("--answers", type=str, default=None, help="Answer indices filter (e.g. 0,1,2,3,4)")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                   help="Reasoning effort level (gpt-oss); omit to use model default")
    p.add_argument("--no-think", action="store_true", help="Disable CoT (no-think mode)")
    p.add_argument("--save-logprobs", action="store_true",
                   help="Save top-20 logprobs as .npy alongside each regen file")
    return p.parse_args()


# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()
    dataset_cfg = get_dataset_config(args.dataset)
    if args.no_think:
        if args.truncate_scope == "cot":
            print("ERROR: --no-think with --truncate-scope cot is incompatible (no CoT to truncate)")
            return
        if args.insert_cot_closing:
            print("ERROR: --no-think with --insert-cot-closing is incompatible (no CoT delimiter to insert)")
            return

    model_info, api_params = build_model_config(args.model_name,
                                                no_think=args.no_think,
                                                reasoning_effort=args.reasoning_effort)
    model_path = args.model_path or model_info["default_model_path"]

    if args.remove_pct is not None:
        removal_desc = f"{args.remove_pct}%"
    else:
        removal_desc = f"{args.remove_tokens} tokens"

    print(f"  dataset:     {args.dataset}")
    print(f"  model-name:  {args.model_name}")
    print(f"  model-path:  {model_path}")
    print(f"  init-dir:    {args.init_dir}")
    print(f"  remove:      {removal_desc} (scope={args.truncate_scope})")
    print(f"  regen-count: {args.regen_count}")
    print(f"  insert-cot-closing: {args.insert_cot_closing}")
    print(f"  out-dir:     {args.out_dir}")
    print(f"  parallel:    {args.parallel}")

    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load problems
    problems = read_problems(dataset_cfg["data_file"])

    # Pre-build rendered prompts for each problem
    rendered_prompts: Dict[int, Tuple[str, str]] = {}
    for i, problem in enumerate(problems):
        rendered_prompts[i] = build_rendered_prompt(
            problem, dataset_cfg, model_info, tokenizer
        )

    # List source files
    problem_filter = parse_int_list(args.problems)
    answer_filter = parse_int_list(args.answers)
    prefix = file_prefix(args.dataset, args.model_name)
    src_files = list_source_files(args.init_dir, prefix, problem_filter, answer_filter)

    print(f"  Source files: {len(src_files)}")

    if not src_files:
        print("  No source files found.")
        return

    # Build task list
    tasks = []
    for fp, prob_idx, ans_idx in src_files:
        for it in range(args.regen_count):
            tasks.append((fp, prob_idx, ans_idx, it))

    total = len(tasks)
    close_desc = " +insert-cot-closing" if args.insert_cot_closing else ""
    print(f"\n{'='*60}")
    print(f"  Mid-regen: remove {removal_desc} scope={args.truncate_scope}{close_desc} ({total} tasks, max {args.parallel} parallel)")
    print(f"{'='*60}")

    # Write metadata
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(args.out_dir, args.dataset, args.model_name,
                   insert_cot_closing=args.insert_cot_closing,
                   remove_pct=args.remove_pct,
                   remove_tokens=args.remove_tokens,
                   truncate_scope=args.truncate_scope,
                   reasoning_effort=args.reasoning_effort,
                   no_think=args.no_think,
                   regen_count=args.regen_count)

    # Initialize vLLM client
    client = OpenAI(base_url=BASE_URL, api_key="dummy-key", timeout=args.timeout)

    # Run in parallel
    success_count = 0
    error_count = 0
    max_workers = min(args.parallel, total) if total > 0 else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fp, prob_idx, ans_idx, it in tasks:
            future = executor.submit(
                do_regen,
                client, args.dataset, args.model_name, model_info, api_params, tokenizer,
                problems, rendered_prompts,
                fp, prob_idx, ans_idx, it,
                args.remove_pct, args.remove_tokens, args.truncate_scope,
                args.out_dir,
                args.insert_cot_closing,
                args.save_logprobs,
            )
            futures[future] = (prob_idx, ans_idx, it)

        t0 = time.time()
        done = 0
        total_tokens = 0
        for future in as_completed(futures):
            ok, pi, ai, it, tokens, err = future.result()
            done += 1
            total_tokens += tokens
            if ok:
                success_count += 1
            else:
                error_count += 1
                if err:
                    print(f"  ERROR: {err}")
            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                tps = total_tokens / elapsed if elapsed > 0 else 0
                print(f"  Progress: {done}/{total} (success={success_count}, errors={error_count}, "
                      f"{total_tokens:,} tokens, {tps:.0f} tok/s, {elapsed:.0f}s)")

    print(f"\n  Done: success={success_count}, errors={error_count}, total={total}")


if __name__ == "__main__":
    main()
