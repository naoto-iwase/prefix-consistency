"""
Answer map construction: extract answers, normalize against gold via LLM judge.

Produces:
  - "Canonical Answer" header in each .txt file (per-file resume)
  - answer_map.json with sections:
    - "golds": {problem_num: normalized_gold}
    - "judge_cache": {serialized_pair: bool} (judge datasets only)

For exact-match datasets, canonical = the extracted string (identity).
For judge datasets, each answer is compared to gold only (O(N) per problem).
If equivalent to gold, canonical = gold. Otherwise canonical = the extracted
string itself. This replaces the old full pairwise clustering (O(N^2)).

Library module: no CLI, no argparse. Used by enrich.py.
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import get_dataset_config
from .io import has_header_field, write_header_fields
from .text import file_prefix
from .answer_extraction import normalize_answer_format, PARSE_FAILED
from . import llm_judge

CANONICAL_HEADER_KEY = "Canonical Answer"


def load_gold_answers(dataset: str) -> Dict[int, str]:
    """Load and normalize gold answers from the dataset JSONL."""
    dataset_cfg = get_dataset_config(dataset)
    golds = {}
    with open(dataset_cfg["data_file"], "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                raw = str(data[dataset_cfg["gold_key"]]).strip()
                golds[i] = normalize_answer_format(raw)
    return golds


def load_questions(dataset: str) -> Dict[int, str]:
    """Load question text from the dataset JSONL."""
    dataset_cfg = get_dataset_config(dataset)
    questions = {}
    with open(dataset_cfg["data_file"], "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                questions[i] = str(data[dataset_cfg["question_key"]]).strip()
    return questions


def _extract_header_value(fp: Path, key: str) -> Optional[str]:
    """Extract a string value from a .txt file header.

    Returns None on I/O errors.
    """
    prefix = f"{key}:"
    try:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()
                if line.startswith("=" * 10):
                    break
    except OSError:
        pass
    return None


def _collect_files_by_problem(
    dirs: List[Path], dataset: str, model_name: str,
) -> Dict[int, List[Tuple[Path, Optional[str]]]]:
    """Collect .txt files per problem with their extracted answers.

    Returns {prob: [(file_path, extracted_answer), ...]}.
    """
    pfx = file_prefix(dataset, model_name)
    pattern = re.compile(
        rf"{re.escape(pfx)}_prob(\d+)_answer\d+(?:_(?:regen|marker)\d+(?:_regen\d+)?)?\.txt$")
    by_prob: Dict[int, List[Tuple[Path, Optional[str]]]] = defaultdict(list)

    for d in dirs:
        for fp in sorted(d.glob(f"{pfx}_prob*.txt")):
            m = pattern.match(fp.name)
            if not m:
                continue
            prob = int(m.group(1))
            extracted = _extract_header_value(fp, "Extracted Answer")  # returns None on I/O error
            by_prob[prob].append((fp, extracted))

    return dict(by_prob)


def _normalize_against_gold(
    answers: List[str], gold: str, question: str,
) -> Dict[str, str]:
    """Normalize each unique answer against gold via LLM judge.

    For each answer: if it equals gold or is judged equivalent to gold,
    canonical = gold. Otherwise canonical = the answer itself.
    This is O(N) judge calls (one per unique answer), not O(N^2).
    """
    prob_map: Dict[str, str] = {}
    for ans in answers:
        if ans == gold:
            prob_map[ans] = gold
        elif llm_judge._are_equivalent(ans, gold, question):
            prob_map[ans] = gold
        else:
            prob_map[ans] = ans
    return prob_map


def build_answer_map(
    dataset: str,
    model_name: str,
    init_dir: Path,
    regen_dirs: Optional[List[Path]] = None,
    output: Optional[Path] = None,
    judge_backend: str = "local",
    judge_model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    no_think: bool = False,
    timeout: int = 1800,
    force: bool = False,
) -> Path:
    """Build answer_map.json and write Canonical Answer headers.

    For judge datasets, each answer is compared to gold only (O(N) per problem).
    For exact-match datasets, canonical = the extracted string (identity map).

    Resume-safe:
      - Problems where all .txt files have "Canonical Answer" are skipped.
      - For judge datasets, LLM judge cache is loaded from existing
        answer_map.json so completed comparisons are not repeated.

    Returns the output path.
    """
    dataset_cfg = get_dataset_config(dataset)
    use_judge = dataset_cfg["llm_judge"]
    regen_dirs = regen_dirs or []
    out = output or (init_dir / "answer_map.json")

    print("  === Answer map ===")
    print(f"    dataset:    {dataset}")
    print(f"    model-name: {model_name}")
    print(f"    init-dir:   {init_dir}")
    print(f"    regen-dirs: {regen_dirs}")
    print(f"    use_judge:  {use_judge}")

    # Load existing judge cache for resume
    if use_judge and out.exists() and not force:
        with open(out) as f:
            existing = json.load(f)
        if "judge_cache" in existing:
            n = llm_judge.load_cache(existing["judge_cache"])
            print(f"    Resumed: loaded {n} cached judge results")

    # Init judge if needed
    if use_judge:
        llm_judge.init(
            enabled=True, dataset=dataset,
            model_name=judge_model or model_name,
            backend=judge_backend,
            reasoning_effort=reasoning_effort,
            no_think=no_think, timeout=timeout,
        )
        questions = load_questions(dataset)

    # Collect files and gold answers
    all_dirs = [init_dir] + regen_dirs
    t0 = time.time()
    files_by_prob = _collect_files_by_problem(all_dirs, dataset, model_name)
    golds = load_gold_answers(dataset)
    n_files = sum(len(v) for v in files_by_prob.values())
    print(f"    Found {n_files} files across "
          f"{len(files_by_prob)} problems ({time.time() - t0:.1f}s)")

    # Process each problem
    answer_map: Dict[str, str] = {}
    skipped_probs = 0
    processed_probs = 0
    n_gold_matches = 0
    t0 = time.time()

    for prob in sorted(files_by_prob.keys()):
        files = files_by_prob[prob]

        # Separate files into done (have Canonical Answer) and pending
        done_files = []
        pending_files = []
        for fp, extracted in files:
            if not force and has_header_field(fp, CANONICAL_HEADER_KEY):
                done_files.append((fp, extracted))
            else:
                pending_files.append((fp, extracted))

        # Load cached canonical answers from done files
        for fp, extracted in done_files:
            canonical = _extract_header_value(fp, CANONICAL_HEADER_KEY)
            if extracted and canonical:
                answer_map[extracted] = canonical

        if not pending_files:
            skipped_probs += 1
            continue

        # Build canonical mapping from ALL files in this problem
        # (done + pending), so the judge cache covers the full set
        # Exclude PARSE_FAILED from judge comparison (never matches gold)
        unique_answers = sorted({ext for _, ext in files
                                 if ext and ext != PARSE_FAILED})

        # Build canonical mapping
        if use_judge:
            gold = golds.get(prob, "")
            question = questions.get(prob, "")
            prob_map = _normalize_against_gold(unique_answers, gold, question)
        else:
            prob_map = {a: a for a in unique_answers}

        # Count gold matches
        n_gold_matches += sum(1 for v in prob_map.values() if v == golds.get(prob, ""))

        # Write Canonical Answer header only to pending files
        # PARSE_FAILED maps to itself (never matches gold)
        prob_map[PARSE_FAILED] = PARSE_FAILED
        for fp, extracted in pending_files:
            if extracted and extracted in prob_map:
                try:
                    write_header_fields(fp, {CANONICAL_HEADER_KEY: prob_map[extracted]})
                except OSError as e:
                    print(f"    ERROR {fp.name}: {e}")

        answer_map.update(prob_map)
        processed_probs += 1

        if use_judge and processed_probs % 10 == 0:
            elapsed = time.time() - t0
            print(f"    Progress: {processed_probs + skipped_probs}/"
                  f"{len(files_by_prob)} problems ({elapsed:.0f}s)")

    # Save answer_map.json
    out.parent.mkdir(parents=True, exist_ok=True)
    data: Dict = {
        "golds": {str(k): v for k, v in golds.items()},
    }
    if use_judge:
        data["judge_cache"] = llm_judge.get_cache()

    with open(out, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"    Saved: {out}")
    print(f"    {len(golds)} gold answers")
    if use_judge:
        print(f"    {n_gold_matches} answers matched gold "
              f"(out of {len(answer_map)} unique answers)")
        cache = llm_judge.get_cache()
        print(f"    {len(cache)} judge results cached")
    if skipped_probs:
        print(f"    {skipped_probs} problems skipped (already done), "
              f"{processed_probs} processed")
    return out
