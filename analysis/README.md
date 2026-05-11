# analysis

Runs weighted majority voting (WMV) evaluation and produces the figures
and tables for the paper from per-cell JSONL files.

## Bundled per-cell data

Per-cell JSONL inputs live at the top of the release (sibling of `analysis/`), split by the equivalence judge used to score them:

- `../data-self-judge/<model>_<benchmark>_jsonl/` — 20 cells, scored with each model as its own equivalence judge (the paper's primary protocol).
- `../data-external-judge/<model>_<benchmark>_jsonl/` — 12 cells (4 models x 3 LLM-judged benchmarks), the same generation pool re-scored with Claude Sonnet 4.6, used for the judge-robustness section (Appendix D.6).

Each `<model>_<benchmark>_jsonl/` contains:

- `*_init.jsonl`: N initial samples per problem (with per-sample log-prob features and verbal/p_true outputs).
- `*_rm25pct_full_x1.jsonl`: regenerations from CoT truncated at 75% (default tau=0.75, K=1).
- `wmv_result.json`: aggregated WMV results at fixed token budgets, sample counts, and a dense token grid, for every evaluated method (Standard MV, DeepConf variants, CISC, P(True), Response probability, AC, ESC, PC-linear/quadratic/cubic), plus natural-stopping operating points for AC and ESC.

GPT-OSS-20B cells under `data-self-judge/` additionally bundle `*_rm{25,50,75}pct_full_x{1,2,3}.jsonl` for the (tau, K) sensitivity tables and `*_markers_x1.jsonl` for the marker-based regeneration baseline.

`wmv_detail.json` (per-problem arrays) is omitted to keep the bundle small. Regenerate with `uv run python wmv.py <cell_dir> --dense` if needed.

## Quick start

```bash
bash run_paper_all.sh
```

Reads `../data-self-judge/` and `../data-external-judge/` and writes to `../paper-out/{tables,figures}/`. Override with `DATA_ROOT=...`, `EXTERNAL_DATA_ROOT=...`, and `OUT_DIR=...`. To run a single script, see `paper/<script>.py --help`.

## Methods

`wmv.py` evaluates every method below in a single pass. All baselines
vote over the same pre-generated pool (`init` JSONL); prefix consistency
additionally consumes the regen JSONL. The exact confidence score and
weighting rule for each method, including any deviations from the
original release, are documented in `wmv/voters.py` and `wmv.py`.

| Group | `wmv_result.json` keys | Reference |
| --- | --- | --- |
| Standard MV | `standard_mv` | Wang et al. 2023 (self-consistency) |
| Prefix consistency (ours) | `prefix_linear`, `prefix_quadratic`, `prefix_cubic`, `prefix_unanimous` | This paper |
| DeepConf, unfiltered | `deepconf_mean` (= self-certainty), `deepconf_bottom10`, `deepconf_tail`, `deepconf_first_token`, `deepconf_block_min` | Fu et al. 2025 |
| DeepConf, filtered | `deepconf_tail_top{10,90}pct`, `deepconf_bottom10_top{10,90}pct` | Fu et al. 2025 |
| CISC (verbal / probability) | `verbal_0_100_raw`, `verbal_binary_raw`, `p_true_raw`, `response_prob_raw`, plus `*_softmax_T1` variants (omitted from the paper) | Taubenfeld et al. 2025 |
| Adaptive stopping | `ac_t{0500..0999}` (sweep over `C_thresh`), `esc_w{2..10}` (sweep over window) | AC: Aggarwal et al. 2023; ESC: Li et al. 2024 |
| Marker-based regen | `markers` | Hammoud et al. 2025 |
| Oracles | `oracle_init`, `oracle_prefix`, `oracle_branching`, `oracle_marker` | Internal upper bounds |
| Branching ablation | `multi_cut_points` | Internal ablation |

The `MAIN_METHOD_KEYS` list in `paper/_defs.py` selects the subset shown
in the main paper table (`table_wmv`): Standard MV, DeepConf-mean,
DeepConf-tail, P(True), Response probability, and the three PC weights.
The full set, grouped via `ALL_BASELINE_GROUPS`, is reported in the
appendix tables (`table_wmv_all`, `table_token_savings_all`).

## Directory structure

```
analysis/
├── run_paper_all.sh          # regenerate every paper table + figure
├── analyze_jsonls.sh         # one-line wrapper: wmv.py --dense
├── wmv.py                    # WMV evaluation -> wmv_result.json + wmv_detail.json
├── wmv/                      # internal modules used by wmv.py
│   ├── eval.py               # evaluate_curves engine
│   ├── jsonl_loader.py       # JSONL loading and group construction
│   └── voters.py             # one voter class per WMV method
├── paper/                    # publication-ready figures and tables
│   ├── _defs.py              # method / benchmark / model definitions
│   ├── _utils.py             # file discovery, label inference, TeX setup
│   ├── figure_cost_accuracy.py            # main Fig 1 cost-accuracy panels
│   ├── figure_cost_accuracy_all.py        # appendix: 5x4 cost-accuracy grid
│   ├── figure_cost_accuracy_full.py       # appendix: full-baseline 5x4 grid
│   ├── figure_cost_accuracy_oracle.py     # appendix: 5x4 oracle decomposition grid
│   ├── figure_pool_correctness_scatter.py # appendix: per-problem init-vs-regen Pass@1
│   ├── figure_roc_all.py                  # appendix: 5x4 macro ROC grid
│   ├── figure_pass1_vs_rates.py           # pass@1 vs r_C, r_W panels
│   ├── figure_pass1_vs_signals.py         # pass@1 vs baseline signals
│   ├── figure_signal_violin.py            # signal distribution (violin)
│   ├── table_wmv.py                       # main WMV accuracy table
│   ├── table_wmv_all.py                   # appendix: per-model WMV
│   ├── table_token_savings.py             # main token-efficiency table
│   ├── table_token_savings_all.py         # appendix: per-model token-eff.
│   ├── table_auroc.py                     # AUROC + signal (r_C, r_W, D)
│   ├── table_format_failure.py            # appendix: extraction-failure rates
│   ├── table_glm_beta_per_benchmark.py    # appendix: per-benchmark GLM slopes
│   ├── table_transition.py                # transition rates and tau scan
│   ├── table_sensitivity.py               # PC weight x (tau, K)
│   ├── table_token_stats.py               # init/regen token counts
│   └── table_assumption.py                # Theorem assumption check on Q'
```

Problem numbers in JSONL files and internal data structures are
0-indexed (`problem_num=0` is the first problem). Plot scripts display
them as 1-indexed (Problem 1, Problem 2, ...) in figure titles.

## Output

`wmv.py` writes two files into the JSONL directory.

```
data-self-judge/<model>_<benchmark>_jsonl/   # (or data-external-judge/...)
├── *.jsonl                   # init, regen, marker regen (input from generation/)
├── wmv_result.json           # WMV evaluation summary (all conditions)
└── wmv_detail.json           # WMV per-problem arrays (separate for size)
```

`paper/` scripts write `.tex` and `.pdf` outputs to a path passed via
`--out` or `--output-dir`; nothing is written to the JSONL directory.

## wmv.py

Evaluates every WMV method at matched token budgets and at fixed sample
counts, then writes the result. See `--help` for the full CLI.

When `--removal-tag` and `--regen-count` are omitted, `wmv.py`
auto-detects every condition present in the JSONL directory. Methods
that share data are evaluated once and reused: shared families (init,
verbal, AC, ESC, marker) are evaluated once for the whole directory;
branching is evaluated once per regen-count `K`; prefix is evaluated
once per (`tau`, `K`) condition.

Each evaluation point (token budget or sample count) draws an
independent random pool, following the standard self-consistency
evaluation protocol.

**Three evaluation modes** are produced when `analyze_jsonls.sh` is
used:

- Token budget (`--token-points`): samples with replacement until the
  cumulative token cost reaches the budget. Default: 13-point
  geometric grid from 1 K to 10 M.
- Fixed sample count (`--sample-points`): draws exactly N samples
  regardless of token cost. Default: `[1]`.
- Dense token-budget grid (`--dense`): draws one sample sequence per
  trial up to the largest budget and snapshots the voters in
  `wmv/voters.py` at a dense grid (default 100 points per decade,
  10^3..10^7). Required by `table_token_savings.py` and
  `figure_cost_accuracy.py`.

### Method storage: shared vs per-condition

Method results are split based on whether they depend on the regen
condition (`tau`, `K`). Methods that read only init data produce
identical results across conditions and are stored once in
`shared_methods`.

```
  Method                         Varies with           Stored in
 ──────────────────────────────────────────────────────────────────────
  standard_mv                    (nothing)             shared_methods
  deepconf_*                     (nothing)             shared_methods
  verbal_*, p_true_*             (nothing)             shared_methods
  response_prob_*                (nothing)             shared_methods
  ac_*, esc_*                    (nothing)             shared_methods
  markers                        (nothing)             shared_methods
  oracle_init, oracle_marker     (nothing)             shared_methods
 ──────────────────────────────────────────────────────────────────────
  multi_cut_points               K (uses all taus)     per_regen_count
  oracle_branching               K                     per_regen_count
 ──────────────────────────────────────────────────────────────────────
  prefix_*                       tau, K                conditions (per tau x K)
  oracle_prefix                  tau, K                conditions (per tau x K)
```

`multi_cut_points` consumes regen data from all `tau` values combined,
so its result only changes with `K`. Readers should merge all three
tiers: `shared_methods` + `per_regen_count[K]` +
`conditions[cond].methods`.

For AC and ESC, `wmv.py` also records each natural-stopping point
(one per swept hyperparameter) under a `natural_stopping` block. The
paper scripts treat the swept points as a cost-accuracy curve and
interpolate it (see `paper/table_token_savings.py` for the
interpolation rule).

<details>
<summary>wmv_result.json schema</summary>

Results are split into `token_budget` and `sample_count` sections, plus
an optional `token_budget_dense` section when `wmv.py --dense` was
passed. Each section contains the same 3-tier method split (see table
above). Per-problem arrays are in the separate `wmv_detail.json`.

Old files without a `token_budget` key are rejected by downstream
scripts with a message to re-run `wmv.py`.

```json
{
  "meta": {
    "n_problems": 100,
    "n_init_generations": 128,
    "n_trials": 500
  },
  "token_budget": {
    "shared_methods": {
      "standard_mv": {
        "family": "init",
        "cost_type": "init",
        "entries": [
          {
            "token_point": 25000,
            "acc": 0.312, "ci": 0.003,
            "mean_tokens": 23439, "n_answers": 3, "n_samples": 3
          }
        ]
      }
    },
    "per_regen_count": { "3": { "...": "..." } },
    "conditions": {
      "rm50pct_full_x3": {
        "removal_tag": "rm50pct_full",
        "regen_count": 3,
        "problem_nums": [0, 1, "..."],
        "n_available": {"...": "..."},
        "methods": {
          "prefix_quadratic": { "...": "..." }
        }
      }
    }
  },
  "sample_count": { "...": "(same shape with sample_point keys)" },
  "token_budget_dense": { "...": "(same shape with ~401 dense points)" },
  "natural_stopping": {
    "ac_t0950": {
      "mean_cost": 386000, "cost_ci": 12000,
      "acc": 0.717, "ci": 0.04,
      "lock_rate": 0.90
    },
    "esc_w5": { "...": "(same fields)" }
  }
}
```

`cost_ci` is the half-width (2 sigma) of `mean_cost` aggregated across
problems. `paper/table_token_savings.py` perturbs operating-point
accuracy by `N(0, (ci/2)^2)` and cost by `N(0, (cost_ci/2)^2)` in its
parametric trial-MC bootstrap.

</details>

<details>
<summary>wmv_detail.json schema</summary>

Same top-level structure as `wmv_result.json` (`token_budget` /
`sample_count`, plus `token_budget_dense` when `--dense` was passed).
Each section mirrors the 3-tier split. Each method stores per-problem
accuracy arrays and `pp_tokens` deduplicated by `n_samples`.

```json
{
  "meta": { "...": "..." },
  "token_budget": {
    "shared_methods": {
      "standard_mv": {
        "pp_tokens_by_n_samples": {
          "3": [8200.1, 7500.3, "...(one per problem)"]
        },
        "entries": [
          {
            "token_point": 25000,
            "n_samples": 3,
            "per_problem": [0.62, 0.0, "...(one per problem)"]
          }
        ]
      }
    },
    "per_regen_count": { "...": "..." },
    "conditions": { "...": "..." }
  },
  "sample_count": { "...": "(same structure with sample_point keys)" },
  "natural_stopping": {
    "ac_t0950": {
      "per_problem_cost":      [380000, "..."],
      "per_problem_acc":       [0.91,   "..."],
      "per_problem_lock_rate": [1.0,    "..."]
    }
  }
}
```

</details>
