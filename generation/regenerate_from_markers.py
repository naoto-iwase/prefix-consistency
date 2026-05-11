#!/usr/bin/env python3
"""
Regeneration from transition markers (Hammoud et al., 2025).
Segments the initial reasoning trace by linguistic transition markers,
then generates a completion from each marker boundary.

Output files: {prefix}_prob{P}_answer{A}_marker{B}_regen{R}.txt

Usage:
    uv run python regenerate_from_markers.py \
        --dataset aime2025 --model-name gpt-oss-20b \
        --init-dir data/initial \
        --regen-count 1 \
        --out-dir data/regen_from_markers
"""

import argparse
import time
import traceback
from collections import defaultdict
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
from core.transition_markers import get_marker_boundaries
from core.api import call_completions
from core.io import (
    TOP_LOGPROBS_SUFFIX,
    read_answer_text, read_header, read_problems,
    write_answer, write_logprobs, write_metadata, write_token_logprobs,
)
from core.text import (
    file_prefix,
    get_cot_to_final,
    build_rendered_prompt, count_cot_and_final_tokens,
    parse_int_list,
)
from regenerate import list_source_files


# =====================================================================
# Helpers
# =====================================================================
def truncate_at_marker(
    generated_text: str, model_cfg: Dict, boundary_char: int,
) -> str:
    """Return the kept text up to a marker boundary (character offset in the CoT).

    The returned text excludes cot_prefix (caller prepends it to the prompt).
    """
    cot_prefix = model_cfg["cot_prefix"]
    text = generated_text
    if cot_prefix and text.startswith(cot_prefix):
        text = text[len(cot_prefix):]

    cot_to_final = get_cot_to_final(model_cfg)
    if cot_to_final is not None:
        marker_pos = text.find(cot_to_final)
        if marker_pos != -1:
            cot_text = text[:marker_pos]
        else:
            cot_text = text
    else:
        cot_text = text

    return cot_text[:boundary_char]


def extract_cot_text(generated_text: str, model_cfg: Dict) -> str:
    """Extract the CoT portion from generated text.

    For thinking models: strips cot_prefix and everything after cot_to_final.
    For instruct models (no-think): returns the full text.
    """
    cot_prefix = model_cfg["cot_prefix"]
    text = generated_text
    if cot_prefix and text.startswith(cot_prefix):
        text = text[len(cot_prefix):]

    cot_to_final = get_cot_to_final(model_cfg)
    if cot_to_final is not None:
        marker_pos = text.find(cot_to_final)
        if marker_pos != -1:
            return text[:marker_pos]
    return text


def output_filename(prefix: str, prob: int, ans: int, marker: int, regen: int) -> str:
    return f"{prefix}_prob{prob}_answer{ans}_marker{marker}_regen{regen}.txt"


# =====================================================================
# Worker function
# =====================================================================
def do_regen_from_marker(
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
    marker_index: int,
    regen_index: int,
    boundary_char: int,
    out_dir: Path,
    save_logprobs: bool = False,
) -> Tuple[bool, int, int, int, int, int, Optional[str]]:
    """Regenerate one answer from a transition marker boundary.

    Returns (success, prob, ans, marker, regen, tokens, error).
    """
    try:
        problem = problems[problem_index]
        dataset_cfg = get_dataset_config(dataset)

        prefix = file_prefix(dataset, model_name)
        fp = out_dir / output_filename(prefix, problem_index, answer_index, marker_index, regen_index)
        npy_path = Path(str(fp) + TOP_LOGPROBS_SUFFIX)
        if fp.exists() and fp.stat().st_size >= MIN_FILE_SIZE_BYTES:
            if not save_logprobs or npy_path.exists():
                return True, problem_index, answer_index, marker_index, regen_index, 0, None

        generated_text = read_answer_text(source_file)
        if not generated_text.strip():
            return False, problem_index, answer_index, marker_index, regen_index, 0, "Empty source file"

        init_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
        kept_text = truncate_at_marker(generated_text, model_info, boundary_char)
        kept_tokens = len(tokenizer.encode(kept_text, add_special_tokens=False))
        cut_tokens = max(init_tokens - kept_tokens, 0)

        rendered, prompt_text = rendered_prompts[problem_index]
        cot_prefix = model_info["cot_prefix"] or ""
        regen_prompt = f"{rendered}{cot_prefix}{kept_text}"
        adjusted_max_tokens = max(cut_tokens * 2, dataset_cfg["complete_max_tokens"] // 10)

        continuation, comp_tokens, raw_top_logprobs, token_logprobs = call_completions(
            client, model_name, regen_prompt, adjusted_max_tokens,
            model_info["max_context_length"], api_params,
            tokenizer, top_logprobs=TOP_LOGPROBS if save_logprobs else 0,
        )

        cot_to_final = get_cot_to_final(model_info)
        start_from_cot = cot_to_final is not None and cot_to_final not in kept_text
        cot_tokens_count, final_tokens = count_cot_and_final_tokens(
            continuation, model_info, tokenizer, start_from_cot,
        )

        gold = problem[dataset_cfg["gold_key"]]
        kept_answer = f"{cot_prefix}{kept_text}"
        write_answer(fp, dataset, problem_index, answer_index, gold, prompt_text, kept_answer, comp_tokens,
                    cot_tokens=cot_tokens_count, final_tokens=final_tokens,
                    regenerated=continuation,
                    extra_headers={
                        "Marker Index": marker_index,
                        "Marker Position": boundary_char,
                        "Init Tokens": init_tokens,
                        "Kept Tokens": kept_tokens,
                        "Cut Tokens": cut_tokens,
                    })

        if save_logprobs and raw_top_logprobs:
            write_logprobs(npy_path, raw_top_logprobs, TOP_LOGPROBS)
        if save_logprobs and token_logprobs:
            write_token_logprobs(fp, token_logprobs)

        return True, problem_index, answer_index, marker_index, regen_index, comp_tokens, None

    except Exception as e:
        error_msg = f"prob{problem_index}_ans{answer_index}_marker{marker_index}_regen{regen_index}: {e}"
        traceback.print_exc()
        return False, problem_index, answer_index, marker_index, regen_index, 0, error_msg


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Regeneration from transition markers (Hammoud et al., 2025)")
    p.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    p.add_argument("--model-name", required=True, choices=MODEL_NAMES)
    p.add_argument("--model-path", default=None,
                    help="Tokenizer path (default: auto-detected from config)")
    p.add_argument("--init-dir", required=True, type=Path,
                    help="Directory with initial answer files")
    p.add_argument("--regen-count", required=True, type=int,
                    help="Number of regenerations per marker boundary")
    p.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory for regenerated files")
    p.add_argument("--parallel", type=int, default=32, help="Max parallel workers")
    p.add_argument("--timeout", type=int, default=1800, help="OpenAI client timeout in seconds")
    p.add_argument("--problems", type=str, default=None,
                    help="Problem indices filter (e.g. 0,1,2 or 0-4)")
    p.add_argument("--answers", type=str, default=None,
                    help="Answer indices filter (e.g. 0,1,2,3,4)")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                   help="Reasoning effort level (gpt-oss); omit to use model default")
    p.add_argument("--no-think", action="store_true",
                    help="Disable CoT (treat full output as segmentation target)")
    p.add_argument("--save-logprobs", action="store_true",
                    help="Save top-20 logprobs as .npy alongside each regen file")
    p.add_argument("--budget-stop", action="store_true",
                    help="Token-budget fair mode: for each problem, use the sum of "
                         "init Generated Tokens as the budget. Process answers in "
                         "order (0, 1, 2, ...), generating all marker regens per "
                         "answer. Stop when cumulative marker regen tokens exceed "
                         "the budget. Mutually exclusive with --answers.")
    return p.parse_args()


# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()
    if args.budget_stop and args.answers is not None:
        print("ERROR: --budget-stop and --answers are mutually exclusive")
        raise SystemExit(1)

    dataset_cfg = get_dataset_config(args.dataset)

    model_info, api_params = build_model_config(args.model_name,
                                                no_think=args.no_think,
                                                reasoning_effort=args.reasoning_effort)
    model_path = args.model_path or model_info["default_model_path"]

    print(f"  dataset:     {args.dataset}")
    print(f"  model-name:  {args.model_name}")
    print(f"  model-path:  {model_path}")
    print(f"  init-dir:    {args.init_dir}")
    print(f"  mode:        transition marker segmentation (Hammoud et al., 2025)")
    print(f"  regen-count: {args.regen_count}")
    print(f"  out-dir:     {args.out_dir}")
    print(f"  parallel:    {args.parallel}")

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    problems = read_problems(dataset_cfg["data_file"])

    rendered_prompts: Dict[int, Tuple[str, str]] = {}
    for i, problem in enumerate(problems):
        rendered_prompts[i] = build_rendered_prompt(
            problem, dataset_cfg, model_info, tokenizer
        )

    problem_filter = parse_int_list(args.problems)
    answer_filter = parse_int_list(args.answers)
    prefix = file_prefix(args.dataset, args.model_name)
    src_files = list_source_files(args.init_dir, prefix, problem_filter, answer_filter)

    # --budget-stop: budget = sum(init Generated Tokens) per problem.
    # Cost per marker group = init_tokens + regen_tokens (matches wmv_jsonl_loader).
    prob_budget: Dict[int, int] = {}
    init_answer_tokens: Dict[Tuple[int, int], int] = {}
    if args.budget_stop:
        for fp, prob_idx, ans_idx in src_files:
            header = read_header(fp)
            gen_tok = int(header.get("Generated Tokens", 0))
            prob_budget[prob_idx] = prob_budget.get(prob_idx, 0) + gen_tok
            init_answer_tokens[(prob_idx, ans_idx)] = gen_tok
        total_budget = sum(prob_budget.values())
        print(f"  --budget-stop: {len(prob_budget)} problems, "
              f"total init budget={total_budget:,} tokens")

    print(f"  Source files: {len(src_files)}")
    if not src_files:
        print("  No source files found.")
        return

    all_tasks_by_prob_ans: Dict[Tuple[int, int], List[Tuple[Path, int, int, int, int, int]]] = {}
    skipped_no_boundary = 0
    for fp, prob_idx, ans_idx in src_files:
        generated_text = read_answer_text(fp)
        if not generated_text.strip():
            continue

        cot_text = extract_cot_text(generated_text, model_info)
        boundaries = get_marker_boundaries(cot_text)
        if not boundaries:
            skipped_no_boundary += 1
            continue

        key = (prob_idx, ans_idx)
        all_tasks_by_prob_ans[key] = []
        for b_idx, boundary_char in enumerate(boundaries):
            for rc in range(args.regen_count):
                all_tasks_by_prob_ans[key].append(
                    (fp, prob_idx, ans_idx, b_idx, rc, boundary_char))

    if skipped_no_boundary:
        print(f"  Skipped {skipped_no_boundary} files with no marker boundaries")

    all_tasks = [t for group in all_tasks_by_prob_ans.values() for t in group]
    pending = []
    completed_tokens: Dict[Tuple[int, int], int] = {}  # (prob, ans) -> tokens already generated
    for task in all_tasks:
        _, prob_idx, ans_idx, marker_idx, regen_idx, _ = task
        out_fp = args.out_dir / output_filename(prefix, prob_idx, ans_idx, marker_idx, regen_idx)
        if out_fp.exists() and out_fp.stat().st_size >= MIN_FILE_SIZE_BYTES:
            # Track tokens from already-completed tasks for budget-stop budget
            if args.budget_stop:
                header = read_header(out_fp)
                tokens = int(header.get("Generated Tokens", 0))
                key = (prob_idx, ans_idx)
                completed_tokens[key] = completed_tokens.get(key, 0) + tokens
            continue
        pending.append(task)

    if len(pending) < len(all_tasks):
        print(f"  Resuming: {len(all_tasks) - len(pending)} already completed, "
              f"{len(pending)} remaining")

    prob_spent: Dict[int, int] = {}
    completed_answers: set = set()
    if args.budget_stop:
        for (pi, ai), regen_tokens in completed_tokens.items():
            init_tok = init_answer_tokens.get((pi, ai), 0)
            prob_spent[pi] = prob_spent.get(pi, 0) + init_tok + regen_tokens
            completed_answers.add((pi, ai))

    # For budget-stop: max answer index per problem from ALL source files
    prob_max_ans: Dict[int, int] = {}
    if args.budget_stop:
        for _, prob_idx, ans_idx in src_files:
            if prob_idx not in prob_max_ans or ans_idx > prob_max_ans[prob_idx]:
                prob_max_ans[prob_idx] = ans_idx

    total_pending = len(pending)
    n_probs = len(prob_max_ans) if args.budget_stop else 0
    probs_with_pending = set(t[1] for t in pending) if args.budget_stop else set()
    print(f"\n{'='*60}")
    if args.budget_stop:
        n_already_exhausted = sum(
            1 for pi in prob_max_ans
            if pi not in probs_with_pending
            or prob_spent.get(pi, 0) >= prob_budget.get(pi, 0))
        n_under_budget = n_probs - n_already_exhausted
        print(f"  Marker regen ({total_pending} candidate tasks, max {args.parallel} parallel)")
        print(f"  --budget-stop: {n_probs} problems, "
              f"{n_already_exhausted} already exhausted, "
              f"{n_under_budget} to process")
    else:
        print(f"  Marker regen ({total_pending} tasks, max {args.parallel} parallel)")
    print(f"{'='*60}")

    if total_pending == 0:
        print("  Nothing to do.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(args.out_dir, args.dataset, args.model_name,
                   mode="transition_markers", regen_count=args.regen_count,
                   reasoning_effort=args.reasoning_effort, no_think=args.no_think)

    client = OpenAI(base_url=BASE_URL, api_key="dummy-key", timeout=args.timeout)

    success_count = 0
    error_count = 0
    done = 0
    total_tokens = 0
    t0 = time.time()

    max_workers = min(args.parallel, total_pending)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        def _submit_task(task):
            fp, prob_idx, ans_idx, marker_idx, regen_idx, boundary_char = task
            future = executor.submit(
                do_regen_from_marker,
                client, args.dataset, args.model_name, model_info, api_params,
                tokenizer, problems, rendered_prompts,
                fp, prob_idx, ans_idx,
                marker_idx, regen_idx, boundary_char,
                args.out_dir, args.save_logprobs,
            )
            futures[future] = task

        if not args.budget_stop:
            # Standard mode: submit all tasks at once
            for task in pending:
                _submit_task(task)
        else:
            pending_by_prob_ans: Dict[Tuple[int, int], List] = defaultdict(list)
            for task in pending:
                pending_by_prob_ans[(task[1], task[2])].append(task)

            prob_next_ans: Dict[int, int] = {}
            prob_cur_remaining: Dict[int, int] = defaultdict(int)
            exhausted_probs: set = set()

            def _submit_answer(pi, ai):
                task_list = pending_by_prob_ans.get((pi, ai), [])
                if not task_list:
                    return False
                for task in task_list:
                    _submit_task(task)
                prob_cur_remaining[pi] = len(task_list)
                prob_next_ans[pi] = ai + 1
                if (pi, ai) not in completed_answers:
                    prob_spent[pi] = prob_spent.get(pi, 0) + init_answer_tokens.get((pi, ai), 0)
                    completed_answers.add((pi, ai))
                return True

            def _submit_next_available(pi, start_ai):
                ai = start_ai
                max_ai = prob_max_ans.get(pi, 0)
                while ai <= max_ai:
                    if _submit_answer(pi, ai):
                        return True
                    ai += 1
                return False

            # Mark problems with no pending tasks as exhausted
            for pi in prob_max_ans:
                if pi not in probs_with_pending:
                    exhausted_probs.add(pi)

            all_probs = sorted(probs_with_pending)
            for pi in all_probs:
                if prob_spent.get(pi, 0) >= prob_budget.get(pi, 0):
                    exhausted_probs.add(pi)
                else:
                    _submit_next_available(pi, 0)

        # as_completed() snapshots at call time, so loop to pick up
        # dynamically submitted futures in budget-stop mode.
        processed_futures: set = set()

        while True:
            batch = {f for f in futures if f not in processed_futures}
            if not batch:
                break
            for future in as_completed(batch):
                processed_futures.add(future)
                ok, pi, ai, mi, ri, tokens, err = future.result()
                done += 1
                total_tokens += tokens
                if ok:
                    success_count += 1
                else:
                    error_count += 1
                    if err:
                        print(f"  ERROR: {err}")

                if args.budget_stop:
                    prob_spent[pi] = prob_spent.get(pi, 0) + tokens
                    prob_cur_remaining[pi] -= 1
                    if prob_cur_remaining[pi] == 0 and pi not in exhausted_probs:
                        if prob_spent.get(pi, 0) >= prob_budget.get(pi, 0):
                            exhausted_probs.add(pi)
                        elif not _submit_next_available(pi, prob_next_ans.get(pi, ai + 1)):
                            exhausted_probs.add(pi)

                if done % 100 == 0 or done == total_pending:
                    elapsed = time.time() - t0
                    tps = total_tokens / elapsed if elapsed > 0 else 0
                    budget_info = ""
                    if args.budget_stop:
                        n_exhausted = len(exhausted_probs)
                        budget_info = f", exhausted={n_exhausted}/{n_probs}"
                    print(f"  Progress: {done}/{total_pending} "
                          f"(success={success_count}, errors={error_count}, "
                          f"{total_tokens:,} tokens, {tps:.0f} tok/s, "
                          f"{elapsed:.0f}s{budget_info})")

    if args.budget_stop:
        n_exhausted = len(exhausted_probs)
        n_generated = success_count + error_count
        print(f"\n  Done: {n_generated} generated (success={success_count}, "
              f"errors={error_count}), "
              f"{n_exhausted}/{n_probs} problems exhausted")
    else:
        print(f"\n  Done: success={success_count}, errors={error_count}, total={done}")


if __name__ == "__main__":
    main()
