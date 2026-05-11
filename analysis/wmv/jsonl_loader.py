"""Data loading and preparation for weighted majority vote evaluation.

Loads init, regen, branching, and marker JSONL files into the
problem dict format expected by evaluate_multi_point().
Also provides attach helpers and data availability detection.
"""

import json
import re
from pathlib import Path
from typing import List

_REGEN_RE = re.compile(r"_(rm\d+(?:pct|tok)_(?:cot|full))_x(\d+)\.jsonl$")


def _extract_tokens(pair) -> int:
    """Extract total_tokens from an answer pair [answer, total_tokens, ...]."""
    if pair and len(pair) >= 2 and isinstance(pair[1], (int, float)):
        return int(pair[1])
    return 0


def _is_usable(pair) -> bool:
    """Drop PARSE_FAILED and 0-token entries from the sample pool."""
    return (pair is not None and len(pair) >= 1
            and pair[0] is not None
            and pair[0] != "PARSE_FAILED"
            and _extract_tokens(pair) > 0)


def _extract_per_answer(rec: dict, key: str, dtype=float, default=0):
    """Extract a per-answer list from a JSONL record, aligned with all_answers.

    Skips entries that are not usable (None answer or 0-token).
    """
    raw = rec.get(key) or []
    result = []
    for i, pair in enumerate(rec["all_answers"]):
        if _is_usable(pair):
            val = default
            if i < len(raw) and raw[i] is not None:
                val = dtype(raw[i])
            result.append(val)
    return result


def find_branching_jsonls(jsonl_dir: Path, regen_count: int) -> list[Path]:
    """Return one ``*_rmXXpct_full_xK.jsonl`` per (rm_pct, scope) usable
    at the requested ``regen_count``.

    Mirrors the K-fallback in ``_find_regen_jsonl``: prefers exact
    ``_x{regen_count}``, otherwise the smallest ``_xK'`` with K' > K.
    """
    by_tag: dict[str, tuple[int, Path]] = {}
    for p in sorted(jsonl_dir.glob("*_rm*pct_full_x*.jsonl")):
        m = _REGEN_RE.search(p.name)
        if not m:
            continue
        tag, rc = m.group(1), int(m.group(2))
        if rc < regen_count:
            continue
        prev = by_tag.get(tag)
        if prev is None or rc < prev[0]:
            by_tag[tag] = (rc, p)
    return [path for _, path in sorted(by_tag.values())]


def _find_regen_jsonl(jsonl_dir: Path, removal_tag: str,
                      regen_count: int):
    """Find the regen JSONL file to load.

    Try _x{regen_count}.jsonl first; if not found, use the smallest
    available file with regen count >= regen_count.
    Returns (path, file_regen_count).
    """
    exact = sorted(jsonl_dir.glob(f"*_{removal_tag}_x{regen_count}.jsonl"))
    if exact:
        return exact[0], regen_count

    candidates = []
    for p in jsonl_dir.glob(f"*_{removal_tag}_x*.jsonl"):
        m = re.search(r"_x(\d+)\.jsonl$", p.name)
        if m:
            fc = int(m.group(1))
            if fc >= regen_count:
                candidates.append((fc, p))
    if not candidates:
        raise FileNotFoundError(
            f"No file matching *_{removal_tag}_x{regen_count}.jsonl "
            f"(or higher) in {jsonl_dir}")
    candidates.sort()
    fc, path = candidates[0]
    print(f"NOTE: _x{regen_count}.jsonl not found, "
          f"using _x{fc}.jsonl (first {regen_count} regens per sample)")
    return path, fc


def load_init_data(jsonl_dir: Path):
    """Load init JSONL only. Return (init_data_dict, budget).

    Returns a dict keyed by problem_num, each containing init answers,
    tokens, confidences, and verbal data. No regen/group data.
    """
    semantic_init = sorted(jsonl_dir.glob("*_init_semantic.jsonl"))
    plain_init = sorted(jsonl_dir.glob("*_init.jsonl"))
    init_path = semantic_init[0] if semantic_init else plain_init[0]

    init_data = {}
    budget = 0
    with open(init_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            pnum = rec["problem_num"]
            answers = []
            init_tokens = []
            conf_lists = {}
            raw_confs = rec.get("confidences") or []
            for idx, pair in enumerate(rec["all_answers"]):
                if _is_usable(pair):
                    answers.append(str(pair[0]))
                    init_tokens.append(_extract_tokens(pair))
                    if idx < len(raw_confs) and raw_confs[idx] is not None:
                        for ck, val in raw_confs[idx].items():
                            conf_lists.setdefault(ck, []).append(float(val))
                    else:
                        for ck in conf_lists:
                            conf_lists[ck].append(None)
            budget = len(rec["all_answers"])

            verbal_0_100_list = _extract_per_answer(rec, "verbal_0_100_confidences", default=50)
            verb_bin_list = _extract_per_answer(rec, "verbal_binary_confidences", default=0.5)
            p_true_list = _extract_per_answer(rec, "p_true_confidences", default=0.5)
            resp_prob_list = _extract_per_answer(rec, "response_probabilities", default=None)
            v0100_tok_list = _extract_per_answer(rec, "verbal_0_100_actual_tokens", int, 0)
            v0100_min_tok_list = _extract_per_answer(rec, "verbal_0_100_min_tokens", int, 0)
            bq_tok_list = _extract_per_answer(rec, "binary_query_actual_tokens", int, 0)
            bq_min_tok_list = _extract_per_answer(rec, "binary_query_min_tokens", int, 0)

            init_data[pnum] = {
                "gold": rec["gold_answer"],
                "init_answers": answers,
                "init_confidences": conf_lists,
                "init_tokens": init_tokens,
                "init_verbal_0_100_confs": verbal_0_100_list,
                "init_verbal_binary_confs": verb_bin_list,
                "init_p_true_confs": p_true_list,
                "init_response_probs": resp_prob_list,
                "init_verbal_0_100_actual_tokens": v0100_tok_list,
                "init_verbal_0_100_min_tokens": v0100_min_tok_list,
                "init_binary_query_actual_tokens": bq_tok_list,
                "init_binary_query_min_tokens": bq_min_tok_list,
            }

    return init_data, budget



def load_data(
    jsonl_dir: Path, removal_tag: str, regen_count: int,
    init_data: dict = None,
):
    """Load init and regen JSONL. Return (problems, budget).

    Auto-detects the smallest available _x{N}.jsonl with N >= regen_count,
    using only the first regen_count regenerations per sample.

    If init_data is provided, skips loading init JSONL (avoids redundant I/O
    when called multiple times with different conditions).
    """
    if init_data is None:
        init_data, budget = load_init_data(jsonl_dir)
    else:
        budget = max(len(d["init_answers"]) for d in init_data.values())
    regen_path, _ = _find_regen_jsonl(jsonl_dir, removal_tag, regen_count)

    regen_by_pnum = {}
    with open(regen_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            regen_by_pnum[rec["problem_num"]] = rec

    problems = []
    total_inits = 0
    inits_without_regen = 0
    for pnum, idata in init_data.items():
        n_init = len(idata["init_answers"])
        total_inits += n_init

        groups = []
        rec = regen_by_pnum.get(pnum)
        if rec is None:
            inits_without_regen += n_init
        else:
            # Read inits from the regen jsonl directly so positional
            # alignment with regen_answers holds regardless of how
            # init.jsonl was filtered.
            regen_inits = rec["all_answers"]
            regen_answers = rec["regen_answers"]
            for i, init_pair in enumerate(regen_inits):
                if not _is_usable(init_pair):
                    continue
                init_ans = str(init_pair[0])
                init_tok = _extract_tokens(init_pair)
                regens = []
                regen_toks = []
                if i < len(regen_answers):
                    for r in regen_answers[i][:regen_count]:
                        if _is_usable(r):
                            regens.append(str(r[0]))
                            regen_toks.append(_extract_tokens(r))
                if not regens:
                    inits_without_regen += 1
                    continue

                groups.append({
                    "answers": [init_ans] + regens,
                    "group_tokens": init_tok + sum(regen_toks),
                    "answer_tokens": [init_tok] + regen_toks,
                })

        problems.append({
            "pnum": pnum,
            **idata,
            "groups": groups,
        })

    if inits_without_regen > 0:
        print(f"WARNING: {inits_without_regen}/{total_inits} init answers "
              f"have no regens and are excluded from prefix groups. "
              f"Re-run regeneration for all inits to maximize pool size.")

    return problems, budget


def load_branching_data(branching_jsonls, regen_count):
    """Load multiple regen JSONL files for multi-point branching MV.

    Each JSONL corresponds to a different truncation depth (branch point).
    For each problem and each init answer, we build ONE group containing
    the init answer plus all its regens from all branch points (symmetric
    with the regen group structure).

    Returns:
        branching_by_pnum: {pnum: {"gold": str, "init_answers": [...],
                           "groups": [{"answers": [...], "group_tokens": int,
                                       "answer_tokens": [int, ...]}, ...]}}
        branch_info: [(removal_tag_str, pct_fraction), ...] for each JSONL
    """
    branching_by_pnum = {}
    branch_info = []

    for bi, jsonl_path in enumerate(branching_jsonls):
        # Infer removal fraction from filename (e.g. rm25pct -> 0.25)
        fname = jsonl_path.stem
        m = re.search(r"rm(\d+)pct", fname)
        if m:
            pct_frac = int(m.group(1)) / 100
        else:
            # Default: assume evenly spaced branch points
            pct_frac = (bi + 1) / (len(branching_jsonls) + 1)
        branch_info.append((fname, pct_frac))

        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                pnum = rec["problem_num"]
                gold = rec["gold_answer"]
                regen_answers = rec["regen_answers"]

                if pnum not in branching_by_pnum:
                    init_ans = []
                    init_tokens = []
                    groups = []
                    for pair in rec["all_answers"]:
                        if not _is_usable(pair):
                            continue
                        ans = str(pair[0])
                        tok = _extract_tokens(pair)
                        init_ans.append(ans)
                        init_tokens.append(tok)
                        groups.append({
                            "answers": [ans],
                            "group_tokens": tok,
                            "answer_tokens": [tok],
                        })
                    branching_by_pnum[pnum] = {
                        "gold": gold,
                        "init_answers": init_ans,
                        "init_tokens": init_tokens,
                        "groups": groups,
                    }

                entry = branching_by_pnum[pnum]
                groups = entry["groups"]
                # filtered_pos: index into groups (post-filter).
                # orig_i: index into regen_answers (per-jsonl, pre-filter).
                filtered_pos = 0
                for orig_i, pair in enumerate(rec["all_answers"]):
                    if not _is_usable(pair):
                        continue
                    if filtered_pos >= len(groups):
                        break
                    group = groups[filtered_pos]
                    if orig_i < len(regen_answers):
                        for r in regen_answers[orig_i][:regen_count]:
                            if _is_usable(r):
                                ans = str(r[0])
                                tokens = _extract_tokens(r)
                                group["answers"].append(ans)
                                group["group_tokens"] += tokens
                                group["answer_tokens"].append(tokens)
                    filtered_pos += 1

    return branching_by_pnum, branch_info


def load_marker_regen_data(marker_jsonl: Path):
    """Load marker-based regen JSONL (from regenerate_from_markers.py).

    The JSONL has:
      all_answers[i] = [answer, tokens, cot_tokens, final_tokens]
      marker_regen_answers[i][j] = [answer, tokens, cot_tokens, final_tokens,
                                     marker_idx, removal_frac]

    For each problem and each init answer, we build ONE group containing
    the init answer plus all its regens from all marker positions (symmetric
    with the regen group structure).

    Returns:
        marker_by_pnum: {pnum: {"gold", "init_answers", "init_tokens",
                         "groups": [{"answers": [...], "group_tokens": int,
                                     "answer_tokens": [int, ...]}, ...]}}
    """
    marker_by_pnum = {}
    with open(marker_jsonl) as f:
        for line in f:
            rec = json.loads(line.strip())
            pnum = rec["problem_num"]
            gold = rec["gold_answer"]

            init_ans = []
            init_tokens = []
            kept_orig_indices = []
            for orig_i, pair in enumerate(rec["all_answers"]):
                if not _is_usable(pair):
                    continue
                init_ans.append(str(pair[0]))
                init_tokens.append(_extract_tokens(pair))
                kept_orig_indices.append(orig_i)

            # mr_data is keyed by original ans_idx, so look up via orig_i.
            groups = []
            mr_data = rec.get("marker_regen_answers") or []
            for filt_i, orig_i in enumerate(kept_orig_indices):
                group = {
                    "answers": [init_ans[filt_i]],
                    "group_tokens": init_tokens[filt_i],
                    "answer_tokens": [init_tokens[filt_i]],
                }
                if orig_i < len(mr_data):
                    for entry in mr_data[orig_i]:
                        if not _is_usable(entry):
                            continue
                        regen_ans = str(entry[0])
                        tokens = _extract_tokens(entry)
                        group["answers"].append(regen_ans)
                        group["group_tokens"] += tokens
                        group["answer_tokens"].append(tokens)
                groups.append(group)

            marker_by_pnum[pnum] = {
                "gold": gold,
                "init_answers": init_ans,
                "init_tokens": init_tokens,
                "groups": groups,
            }
    return marker_by_pnum


# =====================================================================
# Data attachment and availability detection
# =====================================================================

def attach_branching_data(problems: List[dict], branching_data: dict) -> None:
    """Attach branching groups to problems (keyed by pnum)."""
    for p in problems:
        pnum = p["pnum"]
        if pnum in branching_data:
            p["branching_groups"] = branching_data[pnum]["groups"]
        else:
            p["branching_groups"] = [
                {"answers": [ans], "group_tokens": tok,
                 "answer_tokens": [tok]}
                for ans, tok in zip(p["init_answers"], p["init_tokens"])
            ]


def attach_marker_data(problems: List[dict], marker_data: dict) -> None:
    """Attach marker groups to problems (keyed by pnum)."""
    for p in problems:
        pnum = p["pnum"]
        if pnum in marker_data:
            p["marker_groups"] = marker_data[pnum]["groups"]
        else:
            p["marker_groups"] = [
                {"answers": [ans], "group_tokens": tok,
                 "answer_tokens": [tok]}
                for ans, tok in zip(p["init_answers"], p["init_tokens"])
            ]


def detect_data_availability(problems: List[dict]) -> dict:
    """Check which confidence/verbal fields are present in the data.

    Returns dict with boolean flags. Also prints warnings for missing
    token counts when verbal confidence data is present.
    """
    has_confidence = any(
        v is not None and v > 0 for p in problems
        for vals in p["init_confidences"].values() for v in vals
    )
    has_verbal_0_100 = any(
        v is not None and v > 0 for p in problems for v in p["init_verbal_0_100_confs"]
    )
    has_verbal_binary = any(
        v is not None and v > 0 for p in problems for v in p["init_verbal_binary_confs"]
    )
    has_p_true = any(
        v is not None and v > 0 for p in problems for v in p["init_p_true_confs"]
    )
    has_response_prob = any(
        v is not None and v > 0 for p in problems for v in p["init_response_probs"]
    )
    has_v0100_tokens = any(
        t > 0 for p in problems for t in p["init_verbal_0_100_actual_tokens"]
    )
    has_binary_tokens = any(
        t > 0 for p in problems for t in p["init_binary_query_actual_tokens"]
    )

    if has_verbal_0_100 and not has_v0100_tokens:
        print("WARNING: verbal_0_100 confidence present but token counts missing. "
              "Verbal query cost will not be accounted for in token budget evaluation. "
              "Re-run enrich.py + convert_to_jsonl.py to populate token counts.")
    if (has_verbal_binary or has_p_true) and not has_binary_tokens:
        print("WARNING: verbal_binary/p_true confidence present but token counts missing. "
              "Verbal query cost will not be accounted for in token budget evaluation. "
              "Re-run enrich.py + convert_to_jsonl.py to populate token counts.")

    return {
        "has_confidence": has_confidence,
        "has_verbal_0_100": has_verbal_0_100,
        "has_verbal_binary": has_verbal_binary,
        "has_p_true": has_p_true,
        "has_response_prob": has_response_prob,
        "has_v0100_tokens": has_v0100_tokens,
        "has_binary_tokens": has_binary_tokens,
    }



def detect_conditions(jsonl_dir: Path, removal_tag=None, regen_count=None):
    """Auto-detect (removal_tag, regen_count) pairs from JSONL filenames.

    A ``_xK.jsonl`` file holds K regens per sample, so its data covers
    every K' <= K (downstream consumers truncate via ``regens[:K']``).
    Auto-detection therefore expands each tag's largest available K_max
    into K' = 1..K_max so smaller-K duplicates can be removed without
    losing reproducibility for analyses that range over K.
    """
    if removal_tag and regen_count:
        return [(removal_tag, regen_count)]
    max_k_by_tag: dict[str, int] = {}
    for p in sorted(jsonl_dir.glob("*.jsonl")):
        m = _REGEN_RE.search(p.name)
        if not m:
            continue
        tag, rc = m.group(1), int(m.group(2))
        if removal_tag and tag != removal_tag:
            continue
        if regen_count and rc < regen_count:
            continue
        if rc > max_k_by_tag.get(tag, 0):
            max_k_by_tag[tag] = rc
    conditions = []
    for tag, k_max in max_k_by_tag.items():
        if regen_count:
            conditions.append((tag, regen_count))
        else:
            for k in range(1, k_max + 1):
                conditions.append((tag, k))
    return sorted(conditions)
