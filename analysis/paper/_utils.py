"""Shared utilities for paper/ scripts.

Constants are defined in _defs.py; this module re-exports them
alongside file-discovery, data-loading, and formatting helpers.
"""

import json
import os
import re
from pathlib import Path

from _defs import (
    COND_RE, REGEN_RE,
    DATASET_TOTAL_PROBLEMS,
    BENCHMARK_LABELS, BENCHMARK_ORDER,
    MODEL_LABELS, MAIN_PAPER_MODELS,
    ALL_BASELINE_GROUPS, METHOD_DISPLAY, MAIN_METHODS,
    METHOD_ALIASES, METHOD_COLORS,
    SIGNAL_KEY_TO_METHOD, METHOD_TO_SIGNAL_KEY,
)


def signal_keys_in_order(subset):
    """Return signal keys ordered by ALL_BASELINE_GROUPS, filtered to *subset*."""
    sub = set(subset)
    return [
        METHOD_TO_SIGNAL_KEY[k]
        for _, methods in ALL_BASELINE_GROUPS
        for k, _ in methods
        if k in METHOD_TO_SIGNAL_KEY and METHOD_TO_SIGNAL_KEY[k] in sub
    ]


# ── Condition / sort-key helpers ──

def tau_from_rm_pct(rm_pct: int) -> float:
    """Convert removal percentage to truncation fraction tau.

    rm25pct means 25% of tokens are removed, so tau = 0.75.
    """
    return 1.0 - rm_pct / 100.0


def _index_sort_key(label: str, ordering: list[str]) -> tuple[int, str]:
    """Sort key by *ordering*. Unknown labels go to the end."""
    if label in ordering:
        return (ordering.index(label), label)
    return (len(ordering), label)


def bench_sort_key(label: str) -> tuple[int, str]:
    return _index_sort_key(label, BENCHMARK_ORDER)


def model_sort_key(label: str) -> tuple[int, str]:
    return _index_sort_key(label, MAIN_PAPER_MODELS)


def unwrap_wmv_data(data: dict, section: str = "token_budget") -> dict:
    """Unwrap wmv_result.json into the 3-tier structure.

    The current format nests results under ``token_budget`` and
    optionally ``sample_count``.  Old files without a ``token_budget``
    key are rejected with a clear message.
    """
    if "token_budget" not in data:
        raise SystemExit(
            "wmv_result.json is outdated (missing 'token_budget' key). "
            "Re-run wmv.py to regenerate.")
    if section not in data:
        raise SystemExit(
            f"wmv_result.json has no '{section}' section. "
            f"Re-run wmv.py with the appropriate flags.")
    return data[section]


def parse_condition(condition: str) -> tuple[float | None, int | None]:
    """Extract (tau, K) from condition label like 'rm50pct_full_x3'."""
    m = COND_RE.match(condition or "")
    if m:
        return tau_from_rm_pct(int(m.group(1))), int(m.group(3))
    return None, None


# ── File discovery ──

def find_init_file(jsonl_dir: Path) -> Path | None:
    """Find the first *_init.jsonl file in jsonl_dir."""
    files = sorted(jsonl_dir.glob("*_init.jsonl"))
    return files[0] if files else None


def find_regen_files(jsonl_dir: Path) -> list[tuple[int, str, int, Path]]:
    """Auto-detect regen JSONL files.

    Returns sorted list of (rm_pct, scope, K, path).
    """
    results = []
    for fp in sorted(jsonl_dir.glob("*.jsonl")):
        m = REGEN_RE.search(fp.name)
        if m:
            results.append((int(m.group(1)), m.group(2), int(m.group(3)), fp))
    return results


def find_regen_for_condition(jsonl_dir: Path, condition: str) -> Path | None:
    """Find the regen JSONL matching ``condition`` (e.g. ``rm25pct_full_x1``).

    Falls back to the smallest ``_xK'`` with K' > K and the same
    (rm_pct, scope), since that file is a superset of the requested K's regens.
    """
    m = COND_RE.match(condition or "")
    if not m:
        return None
    rm_want, scope_want, k_want = int(m.group(1)), m.group(2), int(m.group(3))
    files = find_regen_files(jsonl_dir)
    for rm, sc, K, p in files:
        if rm == rm_want and sc == scope_want and K == k_want:
            return p
    candidates = [(K, p) for rm, sc, K, p in files
                  if rm == rm_want and sc == scope_want and K > k_want]
    if candidates:
        return min(candidates, key=lambda x: x[0])[1]
    return None


# ── Data loading ──

def load_init_records(jsonl_dir: Path) -> list[dict]:
    """Load init JSONL, return list of per-problem records."""
    init_path = find_init_file(jsonl_dir)
    if not init_path:
        raise FileNotFoundError(f"No init JSONL in {jsonl_dir}")
    with open(init_path) as f:
        return [json.loads(line) for line in f]


def load_regen_records(regen_path: Path) -> dict[int, dict]:
    """Load regen JSONL, return dict[problem_num -> record]."""
    data = {}
    with open(regen_path) as f:
        for line in f:
            rec = json.loads(line)
            data[rec["problem_num"]] = rec
    return data


def is_usable_answer(pair) -> bool:
    """Whether ``pair`` (an ``all_answers[i]`` entry) participates in the
    PC-WMV / Standard MV pool. Mirrors ``wmv/jsonl_loader._is_usable``:
    drops missing answers, parser failures, and zero-token entries.
    Use this for any per-pool counting (Pass@1, per-problem rates, etc.)
    so the empirical pool matches the simulator's input pool.
    """
    return (pair is not None and len(pair) >= 1
            and pair[0] is not None
            and pair[0] != "PARSE_FAILED"
            and len(pair) >= 2
            and isinstance(pair[1], (int, float))
            and pair[1] > 0)


def get_pass_at_1(jsonl_dir: Path) -> float:
    """Closed-form aggregate Pass@1 over the initial answers."""
    init_path = find_init_file(jsonl_dir)
    if not init_path:
        raise FileNotFoundError(f"No init JSONL in {jsonl_dir}")
    total = 0.0
    n = 0
    with open(init_path) as f:
        for line in f:
            rec = json.loads(line)
            gold = str(rec["gold_answer"])
            valid = [str(p[0]) for p in rec["all_answers"]
                     if is_usable_answer(p)]
            if not valid:
                continue
            total += sum(1 for a in valid if a == gold) / len(valid)
            n += 1
    return total / n if n else 0.0


# ── Label inference ──

def detect_subset_footnote(wmv_path: Path, dir_name: str,
                           full_n: int = 128) -> str | None:
    """Return a footnote string if the experiment uses a problem subset or
    fewer than *full_n* initial generations. Returns None otherwise."""
    with open(wmv_path) as f:
        raw = json.load(f)
    meta = raw.get("meta", {})
    n_problems = meta.get("n_problems")
    n_init = meta.get("n_init_generations")

    dataset_key = None
    lower = dir_name.lower().replace("-", "_")
    for dk in DATASET_TOTAL_PROBLEMS:
        if dk in lower:
            dataset_key = dk
            break
    total_problems = DATASET_TOTAL_PROBLEMS.get(dataset_key)
    is_subset = (total_problems is not None and n_problems is not None
                 and n_problems < total_problems)

    parts = []
    if is_subset:
        parts.append(f"{n_problems} of {total_problems} problems")
    if n_init is not None and n_init < full_n:
        parts.append(f"$N$={n_init}")
    return ", ".join(parts) if parts else None


def infer_benchmark_label(dir_name: str) -> str:
    """Infer short benchmark label from directory name."""
    lower = dir_name.lower()
    for key, label in BENCHMARK_LABELS.items():
        if key in lower:
            return label
    return dir_name


def infer_model_label(dir_name: str) -> str:
    """Infer model name from directory name."""
    lower = dir_name.lower()
    for key, label in MODEL_LABELS.items():
        if key in lower:
            return label
    return ""


# ── WMV result loading ──

def format_tp(tp: int) -> str:
    """Format token point for display (e.g. 250000 -> '250k')."""
    if tp >= 1_000_000:
        return f"{tp / 1_000_000:g}M"
    if tp >= 1_000:
        return f"{tp / 1_000:g}k"
    return str(tp)


def format_acc(acc: float, ci: float, is_best: bool) -> str:
    """Format accuracy with CI, e.g. .526$_{\\pm.001}$"""
    if acc >= 0.9995:
        acc_s = "1.000"
    else:
        acc_s = f"{acc:.3f}"[1:]   # "0.526" -> ".526"
    ci_s = f"{ci:.3f}"[1:]         # "0.002" -> ".002"
    if is_best:
        return f"\\textbf{{{acc_s}}}$_{{\\pm{ci_s}}}$"
    return f"{acc_s}$_{{\\pm{ci_s}}}$"


def load_wmv_methods(json_path: Path, condition: str | None,
                     points: list[int],
                     section: str = "token_budget",
                     point_key: str = "token_point") -> dict:
    """Load all method results from wmv_result.json.

    Merges shared_methods, per_regen_count, and condition-specific
    methods into a single dict. Old method names are mapped to their
    current names via METHOD_ALIASES.

    *section* selects ``"token_budget"`` or ``"sample_count"``.
    *point_key* selects ``"token_point"`` or ``"sample_point"``.

    Returns {method_key: {point_value: (acc, ci)}}.
    """
    with open(json_path) as f:
        data = json.load(f)

    sec = unwrap_wmv_data(data, section)
    shared = sec.get("shared_methods", {})
    per_rc = sec.get("per_regen_count", {})
    conditions = sec.get("conditions", {})

    cond_data = conditions.get(condition, {})
    if not cond_data:
        for k, v in conditions.items():
            if condition and condition in k:
                cond_data = v
                break
    if not cond_data and conditions:
        cond_data = next(iter(conditions.values()))

    cond_methods = cond_data.get("methods", {})
    rc_key = str(cond_data.get("regen_count", ""))

    all_methods = {}
    all_methods.update(shared)
    all_methods.update(per_rc.get(rc_key, {}))
    all_methods.update(cond_methods)

    points_set = set(points)
    result = {}
    for mdata_key, mdata in all_methods.items():
        entries = {}
        for e in mdata.get("entries", []):
            pt = e.get(point_key)
            if pt in points_set:
                entries[pt] = (e["acc"], e["ci"])
        if entries:
            canonical = METHOD_ALIASES.get(mdata_key, mdata_key)
            result[canonical] = entries

    return result


# ── AUROC ──

_FAMILY_PREFIXES = (
    ("ac_t",                  "ac_sweep"),
    ("esc_w",                 "esc_sweep"),
)


def _natural_stopping_family(name: str) -> str | None:
    for prefix, family in _FAMILY_PREFIXES:
        if name.startswith(prefix):
            return family
    return None


def load_natural_stopping_curves(
    json_path: Path,
    families: tuple[str, ...] = ("ac_sweep", "esc_sweep"),
    require_min_lock_rate: float | None = None,
) -> dict[str, list[tuple[int, float, float, float]]]:
    """Synthesize per-family ``[(cost, acc, ci, cost_ci), ...]`` curves
    from the ``natural_stopping`` section of wmv_result.json.

    Groups stopping-rule entries into per-family sweeps via name
    prefixes (``ac_t*`` -> ``ac_sweep``, ``esc_w*`` -> ``esc_sweep``).
    Points are sorted by cost. *require_min_lock_rate* filters out
    points where the voter rarely locked, whose reported ``mean_cost``
    reflects pool exhaustion rather than natural stopping.
    """
    with open(json_path) as f:
        data = json.load(f)
    ns = data.get("natural_stopping", {})
    out: dict[str, list[tuple[int, float, float, float]]] = {
        f: [] for f in families}
    for name, entry in ns.items():
        if (require_min_lock_rate is not None
                and entry["lock_rate"] < require_min_lock_rate):
            continue
        family = _natural_stopping_family(name)
        if family is None or family not in out:
            continue
        cost = float(entry["mean_cost"])
        acc = float(entry["acc"])
        ci = float(entry["ci"])
        cost_ci = float(entry["cost_ci"])
        out[family].append((int(round(cost)), acc, ci, cost_ci))
    for f in out:
        out[f].sort(key=lambda x: x[0])
    return out


def compute_macro_auroc(
    y_per_problem: dict,
    s_per_problem: dict,
) -> float | None:
    """Macro-average per-problem AUROC across problems with both classes.

    Uses the standard pairwise definition (ties count 0.5). Problems
    lacking either class contribute nothing; returns None if no problem
    is usable.
    """
    aurocs = []
    for pnum, y in y_per_problem.items():
        s = s_per_problem.get(pnum, [])
        n_pos = sum(y)
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        pos_scores = [si for yi, si in zip(y, s) if yi == 1]
        neg_scores = [si for yi, si in zip(y, s) if yi == 0]
        gt = eq = 0
        for sp in pos_scores:
            for sn in neg_scores:
                if sp > sn:
                    gt += 1
                elif sp == sn:
                    eq += 1
        aurocs.append((gt + 0.5 * eq) / (n_pos * n_neg))
    if not aurocs:
        return None
    return sum(aurocs) / len(aurocs)


# ── Merge-base tex parsing ──

_EMPTY_CELL_TOKENS = {"", "--", "---", "—"}

_TABULAR_BEGIN_RE = re.compile(r"\\begin\{tabular[*x]?\}")
_TABULAR_END_RE = re.compile(r"\\end\{tabular[*x]?\}")


def is_empty_cell(cell: str) -> bool:
    """Return True if *cell* represents a missing value (``--``, ``---``).

    Whitespace and a trailing row terminator (``\\\\``) are ignored.
    """
    s = cell.strip()
    if s.endswith("\\\\"):
        s = s[:-2].strip()
    return s in _EMPTY_CELL_TOKENS


def _strip_line_comments(content: str) -> str:
    """Remove ``%``-initiated comments while preserving ``\\%`` escapes."""
    _PROTECT = "\x01"
    s = content.replace(r"\%", _PROTECT)
    out: list[str] = []
    in_comment = False
    for ch in s:
        if in_comment:
            if ch == "\n":
                in_comment = False
                out.append(ch)
        else:
            if ch == "%":
                in_comment = True
            else:
                out.append(ch)
    return "".join(out).replace(_PROTECT, r"\%")


def _skip_brace_group(s: str, i: int) -> int:
    """``s[i]`` must be ``{``. Return the index just after the matching ``}``."""
    assert s[i] == "{"
    depth = 1
    i += 1
    n = len(s)
    while i < n and depth > 0:
        if s[i] == "\\" and i + 1 < n:
            i += 2
            continue
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    return i


def load_base_cells_generic(
    path: Path,
    row_key_fn,
    group_key_fn=None,
) -> tuple[int | None, dict]:
    """General merge-base parser.

    For every data row in *path*, first call ``group_key_fn(cells)`` (if
    supplied); when it returns non-None, the returned value becomes the
    *current group* passed to later rows. Then call
    ``row_key_fn(cells, current_group)``; when it returns a row key
    (any hashable), store each non-empty data cell keyed by
    ``(row_key, col_idx)``. ``col_idx`` starts at 1 (col 0 is the label
    cell and is skipped).

    Returns ``(n_data_cols, cells)``. ``n_data_cols`` is the width of the
    first matched data row and lets callers refuse to merge when it
    differs from the run's expected column count.
    """
    content = path.read_text()
    out: dict = {}
    n_cols: int | None = None
    current_group = None
    for row in split_tex_rows(content):
        cells = [c.strip() for c in row.split("&")]
        # Single-cell rows are usually markup ("\\midrule") but can also be
        # group-header rows like "\\multicolumn{N}{...}{\\textit{Group}} \\\\".
        # Give group_key_fn a chance to read those and update the current
        # group before we skip.
        if len(cells) <= 1:
            if group_key_fn is not None:
                g = group_key_fn(cells)
                if g is not None:
                    current_group = g
            continue
        if group_key_fn is not None:
            g = group_key_fn(cells)
            if g is not None:
                current_group = g
        rk = row_key_fn(cells, current_group)
        if rk is None:
            continue
        if n_cols is None:
            n_cols = len(cells) - 1
        for idx, v in enumerate(cells[1:], start=1):
            if not is_empty_cell(v):
                out[(rk, idx)] = v
    return n_cols, out


def load_base_cells_checked(
    path: Path,
    row_key_fn_or_keys,
    expected_cols: int,
    label: str,
    group_key_fn=None,
) -> dict | None:
    """Load merge-base cells, returning None when the column count
    differs from *expected_cols* (in which case a warning names *label*).

    For convenience *row_key_fn_or_keys* may be either:

    * a list of string row keys (flat tables): matched as substrings of
      the row's first cell; or
    * a callable ``(cells, current_group) -> row_key | None``.
    """
    if callable(row_key_fn_or_keys):
        row_key_fn = row_key_fn_or_keys
    else:
        keys = list(row_key_fn_or_keys)

        def row_key_fn(cells, _group):
            first = cells[0]
            for rk in keys:
                if rk in first:
                    return rk
            return None

    n_cols, cells = load_base_cells_generic(
        path, row_key_fn, group_key_fn)
    if n_cols is None:
        print(f"  {label}: no matched rows in {path}; merge skipped")
        return None
    if n_cols != expected_cols:
        print(f"  {label}: column count mismatch "
              f"({n_cols} base vs {expected_cols} current); merge skipped. "
              f"Pass --models (and friends) to match the base's schema.")
        return None
    print(f"  {label}: loaded {len(cells)} cells from {path}")
    return cells


def fallback_cell(
    primary: str,
    key,
    base: dict | None,
) -> str:
    """Return *primary* unless it is empty/``--`` and *base* has a value
    at *key*."""
    if base is None or not is_empty_cell(primary):
        return primary
    return base.get(key, primary)


def split_tex_rows(content: str) -> list[str]:
    """Split every ``tabular`` body in *content* on ``\\\\`` at brace depth 0.

    ``%`` line comments are stripped first (preserving ``\\%``). The
    scanner looks for each ``\\begin{tabular[*x]?}{<cols>}`` environment,
    skips its column-spec argument, and splits the body at the inner
    ``\\\\`` row terminators. Brace depth tracked here is **relative to
    the tabular body**, so outer wrappers like ``\\resizebox{..}{..}{..}``
    do not disturb it.

    Each returned row still carries its leading markup
    (``\\toprule``, ``\\midrule``, ``\\multirow{..}{*}{..}``, ...) and
    inter-cell ``&`` separators.
    """
    s = _strip_line_comments(content)
    rows: list[str] = []

    pos = 0
    while True:
        m = _TABULAR_BEGIN_RE.search(s, pos)
        if m is None:
            break
        i = m.end()
        # Optional [pos-arg]
        while i < len(s) and s[i].isspace():
            i += 1
        if i < len(s) and s[i] == "[":
            while i < len(s) and s[i] != "]":
                i += 1
            if i < len(s):
                i += 1
        # Skip through any following brace groups (column spec, optional
        # width for tabular* / tabularx).
        while True:
            while i < len(s) and s[i].isspace():
                i += 1
            if i < len(s) and s[i] == "{":
                i = _skip_brace_group(s, i)
            else:
                break

        end_m = _TABULAR_END_RE.search(s, i)
        body_end = end_m.start() if end_m else len(s)
        body = s[i:body_end]

        depth = 0
        start = 0
        j = 0
        n = len(body)
        while j < n:
            c = body[j]
            if c == "\\" and j + 1 < n:
                if body[j + 1] == "\\" and depth == 0:
                    rows.append(body[start:j])
                    j += 2
                    # Optional vspace arg on the terminator, e.g. \\[2pt].
                    while j < n and body[j].isspace():
                        j += 1
                    if j < n and body[j] == "[":
                        while j < n and body[j] != "]":
                            j += 1
                        if j < n:
                            j += 1
                    start = j
                    continue
                # Escaped char (``\{``, ``\%``, ``\&``, command name char).
                j += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1

        if end_m is None:
            break
        pos = end_m.end()

    return rows


# ── matplotlib / TeX Live setup ──

def setup_tex_rendering():
    """Configure matplotlib for LaTeX-quality rendering.

    Reads TEXLIVE_BIN from environment.
    Falls back to serif font without usetex if unavailable.
    Call this before any plt.subplots().
    """
    import matplotlib.pyplot as plt

    texlive_bin = os.environ.get("TEXLIVE_BIN", "")
    use_tex = bool(texlive_bin) and os.path.isdir(texlive_bin)
    if use_tex:
        os.environ["PATH"] = texlive_bin + ":" + os.environ.get("PATH", "")

    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.size": 11,
    })
    print(f"[setup_tex_rendering] text.usetex={use_tex} "
          f"(TEXLIVE_BIN={texlive_bin or 'unset'})")
    return use_tex
