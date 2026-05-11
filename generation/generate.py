#!/usr/bin/env python3
"""
Initial answer generation via vLLM completions API.
For each problem, generate N answers and save them as individual files.

Usage:
    uv run python generate.py \
        --dataset aime2025 --model-name gpt-oss-20b \
        --num-answers 100 --out-dir generation/data/initial \
        --parallel 32
"""

import argparse
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
    read_problems, write_answer, write_logprobs, write_metadata,
    write_token_logprobs,
)
from core.text import (
    build_rendered_prompt, count_cot_and_final_tokens,
    file_prefix, parse_int_list,
)


# =====================================================================
# Worker function
# =====================================================================
def do_generate(
    client: OpenAI,
    dataset: str,
    model_name: str,
    problems: List[Dict],
    rendered_prompts: Dict[int, Tuple[str, str]],
    problem_index: int,
    answer_index: int,
    out_dir: Path,
    model_info: Dict,
    api_params: Dict,
    tokenizer=None,
    save_logprobs: bool = False,
) -> Tuple[bool, int, int, Optional[str]]:
    """Generate one answer for a problem. Returns (success, prob, ans, error)."""
    try:
        problem = problems[problem_index]
        dataset_cfg = get_dataset_config(dataset)

        # Output path
        prefix = file_prefix(dataset, model_name)
        fp = out_dir / f"{prefix}_prob{problem_index}_answer{answer_index}.txt"
        npy_path = Path(str(fp) + TOP_LOGPROBS_SUFFIX)

        # Skip if file exists and is large enough (resumability)
        if fp.exists() and fp.stat().st_size >= MIN_FILE_SIZE_BYTES:
            if not save_logprobs or npy_path.exists():
                return True, problem_index, answer_index, 0, None

        rendered, prompt_text = rendered_prompts[problem_index]
        # Append cot_prefix when the chat template doesn't emit it (e.g. Ministral-3 Reasoning).
        cot_prefix = model_info["cot_prefix"]
        if cot_prefix and not rendered.rstrip().endswith(cot_prefix):
            rendered = rendered + cot_prefix

        # Call completions API (model generates natively from the rendered prompt)
        text, comp_tokens, raw_top_logprobs, token_logprobs = call_completions(
            client, model_name, rendered, dataset_cfg["complete_max_tokens"],
            model_info["max_context_length"], api_params,
            tokenizer, top_logprobs=TOP_LOGPROBS if save_logprobs else 0,
        )

        # Extract CoT / Final token counts (delimiter-excluded, for cost accounting)
        # gen_tokens >= cot_tokens + final_tokens (difference = delimiter overhead)
        cot_tokens, final_tokens = count_cot_and_final_tokens(text, model_info, tokenizer)

        gold = problem[dataset_cfg["gold_key"]]
        write_answer(fp, dataset, problem_index, answer_index, gold, prompt_text, text, comp_tokens,
                    cot_tokens=cot_tokens, final_tokens=final_tokens)

        if save_logprobs and raw_top_logprobs:
            write_logprobs(npy_path, raw_top_logprobs, TOP_LOGPROBS)
        if save_logprobs and token_logprobs:
            write_token_logprobs(fp, token_logprobs)

        return True, problem_index, answer_index, comp_tokens, None

    except Exception as e:
        error_msg = f"prob{problem_index}_ans{answer_index}: {e}"
        traceback.print_exc()
        return False, problem_index, answer_index, 0, error_msg


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Initial answer generation via vLLM completions API")
    p.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    p.add_argument("--model-name", required=True, choices=MODEL_NAMES)
    p.add_argument("--model-path", default=None, help="Tokenizer path (default: auto-detected from config)")
    p.add_argument("--num-answers", required=True, type=int, help="Number of answers to generate per problem")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    p.add_argument("--parallel", type=int, default=32, help="Max parallel workers")
    p.add_argument("--timeout", type=int, default=1800, help="OpenAI client timeout in seconds")
    p.add_argument("--problems", type=str, default=None, help="Problem indices filter (e.g. 0,1,2 or 0-4)")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                   help="Reasoning effort level (gpt-oss); omit to use model default")
    p.add_argument("--no-think", action="store_true", help="Disable CoT (no-think mode)")
    p.add_argument("--save-logprobs", action="store_true",
                   help="Save top-20 logprobs as .npy alongside each answer file")
    return p.parse_args()


# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()
    dataset_cfg = get_dataset_config(args.dataset)
    model_info, api_params = build_model_config(args.model_name,
                                                no_think=args.no_think,
                                                reasoning_effort=args.reasoning_effort)
    model_path = args.model_path or model_info["default_model_path"]

    print(f"  dataset:     {args.dataset}")
    print(f"  model-name:  {args.model_name}")
    print(f"  model-path:  {model_path}")
    print(f"  num-answers: {args.num_answers}")
    print(f"  out-dir:     {args.out_dir}")
    print(f"  parallel:    {args.parallel}")

    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load problems
    problems = read_problems(dataset_cfg["data_file"])

    # Filter problems
    problem_filter = parse_int_list(args.problems)
    if problem_filter is not None:
        problem_indices = [i for i in problem_filter if 0 <= i < len(problems)]
    else:
        problem_indices = list(range(len(problems)))

    # Pre-build rendered prompts
    rendered_prompts: Dict[int, Tuple[str, str]] = {}
    for i in problem_indices:
        rendered_prompts[i] = build_rendered_prompt(
            problems[i], dataset_cfg, model_info, tokenizer
        )

    # Build task list
    tasks = []
    for prob_idx in problem_indices:
        for ans_idx in range(args.num_answers):
            tasks.append((prob_idx, ans_idx))

    total = len(tasks)
    print(f"\n{'='*60}")
    print(f"  Initial generation: {len(problem_indices)} problems x {args.num_answers} answers = {total} tasks")
    print(f"{'='*60}")

    # Initialize vLLM client
    client = OpenAI(base_url=BASE_URL, api_key="dummy-key", timeout=args.timeout)

    # Run in parallel
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(args.out_dir, args.dataset, args.model_name,
                   reasoning_effort=args.reasoning_effort,
                   num_answers=args.num_answers,
                   no_think=args.no_think)
    success_count = 0
    error_count = 0
    max_workers = min(args.parallel, total) if total > 0 else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for prob_idx, ans_idx in tasks:
            future = executor.submit(
                do_generate,
                client, args.dataset, args.model_name,
                problems, rendered_prompts,
                prob_idx, ans_idx, args.out_dir,
                model_info, api_params, tokenizer, args.save_logprobs,
            )
            futures[future] = (prob_idx, ans_idx)

        t0 = time.time()
        done = 0
        total_tokens = 0
        for future in as_completed(futures):
            ok, pi, ai, tokens, err = future.result()
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
