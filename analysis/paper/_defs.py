"""Shared constant definitions for paper/ scripts."""

import re

# ── Regex patterns ──

COND_RE = re.compile(r"^rm(\d+)pct_(\w+)_x(\d+)$")
REGEN_RE = re.compile(r"_rm(\d+)pct_(\w+)_x(\d+)\.jsonl$")


# ── Datasets ──

DATASET_TOTAL_PROBLEMS = {
    "aime2025": 30,
    "frontierscience": 100,
    "hmmt": 33,
    "brumo": 30,
}

# Tildes are LaTeX non-breaking spaces, kept consistent with main.tex.
BENCHMARK_LABELS = {
    "frontierscience": "FrontierScience-Olympiad",
    "hmmt": "HMMT Feb~2026",
    "aime2025": "AIME~2025",
    "brumo": "Brumo~2025",
}

BENCHMARK_ORDER = list(BENCHMARK_LABELS.values())

BENCHMARK_ABBREV = {
    "FrontierScience-Olympiad": "FSci",
    "HMMT Feb~2026": "HMMT",
    "AIME~2025": "AIME",
    "Brumo~2025": "Brumo",
}

BENCHMARK_DOMAIN = {
    "FrontierScience-Olympiad": "Science",
    "HMMT Feb~2026": "Math",
    "AIME~2025": "Math",
    "Brumo~2025": "Math",
}

DOMAIN_COLOR = {"Science": "tab:blue", "Math": "tab:red"}

# Math benchmarks share a red ramp so they read as one group.
BENCHMARK_COLOR = {
    "FrontierScience-Olympiad": "tab:blue",
    "HMMT Feb~2026": "#a83232",
    "AIME~2025":     "#d62728",
    "Brumo~2025":    "#f08080",
}


# ── Models ──

# Longer keys must come before shorter prefixes for substring matching.
MODEL_LABELS = {
    "gpt-oss-120b": "GPT-OSS-120B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "nemotron-3-nano-30b-a3b": "Nemotron3-30B",
    "nemotron-nano-9b-v2": "Nemotron2-9B",
    "ministral-3-14b-reasoning-2512": "Ministral3-14B",
}

MAIN_PAPER_MODELS = list(MODEL_LABELS.values())

MODEL_STEMS = {label: label.lower().replace("-", "_")
               for label in MODEL_LABELS.values()}


# ── Method definitions ──

ALL_BASELINE_GROUPS = [
    ("Baseline", [
        ("standard_mv", "Standard MV"),
    ]),
    ("DeepConf", [
        ("deepconf_first_token", "DeepConf first-token"),
        ("deepconf_mean", "Self-certainty"),
        ("deepconf_bottom10", "DeepConf bottom-10\\%"),
        ("deepconf_block_min", "DeepConf block-min"),
        ("deepconf_tail", "DeepConf tail"),
    ]),
    ("DeepConf (filtered)", [
        ("deepconf_bottom10_top10pct", "DeepConf bottom-10\\% (top-10\\%)"),
        ("deepconf_bottom10_top90pct", "DeepConf bottom-10\\% (top-90\\%)"),
        ("deepconf_tail_top10pct", "DeepConf tail (top-10\\%)"),
        ("deepconf_tail_top90pct", "DeepConf tail (top-90\\%)"),
    ]),
    ("CISC", [
        ("response_prob_raw", "Response probability"),
        ("verbal_binary_raw", "Verbal binary"),
        ("verbal_0_100_raw", "Verbal 0--100"),
        ("p_true_raw", "P(True)"),
    ]),
    ("SubthoughtReasoner", [
        ("markers", "SubthoughtReasoner"),
    ]),
    ("Adaptive stopping", [
        ("ac_sweep", "AC sweep"),
        ("esc_sweep", "ESC sweep"),
    ]),
    ("Prefix consistency", [
        ("prefix_linear", "PC-linear"),
        ("prefix_quadratic", "PC-quadratic"),
        ("prefix_cubic", "PC-cubic"),
    ]),
]

METHOD_DISPLAY = {
    key: display
    for _, methods in ALL_BASELINE_GROUPS
    for key, display in methods
}

MAIN_METHOD_KEYS = [
    "standard_mv",
    "deepconf_mean",
    "deepconf_tail",
    "p_true_raw",
    "prefix_linear",
    "prefix_quadratic",
    "prefix_cubic",
]

MAIN_METHODS = [(k, METHOD_DISPLAY[k]) for k in MAIN_METHOD_KEYS]

HEADLINE_METHOD_KEYS = [
    "prefix_cubic",
    "standard_mv",
    "deepconf_tail",
    "p_true_raw",
]

NATURAL_FAMILIES = ("ac_sweep", "esc_sweep")
NATURAL_MARKERS = {"ac_sweep": "o", "esc_sweep": "o"}

ORACLE_KEYS = {"oracle_init", "oracle_prefix"}

# Old wmv_result.json key -> current key (backward compatibility).
METHOD_ALIASES = {
    "verbal_0_100_linear": "verbal_0_100_raw",
    "verbal_binary_weighted": "verbal_binary_raw",
    "p_true_linear": "p_true_raw",
    "response_prob_linear": "response_prob_raw",
    "verbal_0_100_linear_cisc_T1": "verbal_0_100_linear_T1",
}

# JSONL ``*_conf`` / ``*_confidences`` field name -> canonical method key.
SIGNAL_KEY_TO_METHOD = {
    "first_token_conf": "deepconf_first_token",
    "mean_conf":        "deepconf_mean",
    "bottom10_conf":    "deepconf_bottom10",
    "block_min_conf":   "deepconf_block_min",
    "tail_conf":        "deepconf_tail",
    "response_prob":    "response_prob_raw",
    "verbal_binary":    "verbal_binary_raw",
    "verbal_0_100":     "verbal_0_100_raw",
    "p_true":           "p_true_raw",
}

METHOD_TO_SIGNAL_KEY = {v: k for k, v in SIGNAL_KEY_TO_METHOD.items()}


# ── Plot colors ──
METHOD_COLORS = {
    "standard_mv":          "#000000",
    # PC
    "prefix_linear":        "#74c476",
    "prefix_quadratic":     "#31a354",
    "prefix_cubic":         "#006d2c",
    # DeepConf
    "deepconf_first_token": "#c6dbef",
    "deepconf_mean":        "#9ecae1",
    "deepconf_bottom10":    "#6baed6",
    "deepconf_block_min":   "#4292c6",
    "deepconf_tail":        "#2171b5",
    # DeepConf filtered
    "deepconf_bottom10_top10pct": "#dadaeb",
    "deepconf_bottom10_top90pct": "#bcbddc",
    "deepconf_tail_top10pct":     "#807dba",
    "deepconf_tail_top90pct":     "#4a1486",
    # CISC
    "response_prob_raw":    "#fee391",
    "verbal_binary_raw":    "#fec44f",
    "verbal_0_100_raw":     "#fe9929",
    "p_true_raw":           "#ec7014",
    "markers":              "#17becf",
    "ac_sweep":             "#cb181d",
    "esc_sweep":            "#cc79a7",
    "oracle_init":          "#bdbdbd",
    "oracle_prefix":        "#a1d99b",
}


# ── Method citation tags ──
# Plain text (no \citep) so labels render with or without text.usetex.
METHOD_CITATIONS = {
    "prefix_linear":        "(ours)",
    "prefix_quadratic":     "(ours)",
    "prefix_cubic":         "(ours)",
    "standard_mv":          "(Wang et al. 2023)",
    "deepconf_mean":        "(Kang et al. 2025)",
    "deepconf_tail":        "(Fu et al. 2026)",
    "deepconf_bottom10":    "(Fu et al. 2026)",
    "deepconf_first_token": "(Fu et al. 2026)",
    "deepconf_block_min":   "(Fu et al. 2026)",
    "p_true_raw":           "(Kadavath et al. 2022)",
    "verbal_0_100_raw":     "(Lin et al. 2022)",
    "verbal_binary_raw":    "(Lin et al. 2022)",
    "response_prob_raw":    "(Wang et al. 2023)",
    "ac_sweep":             "(Aggarwal et al. 2023)",
    "esc_sweep":            "(Li et al. 2024)",
}
