#!/usr/bin/env bash
# Regenerate every paper-derived table and figure from the bundled
# wmv_result.json files.
#
#   DATA_ROOT            Per-cell directory root scored under self-judge.
#                        Default: ../data-self-judge.
#   EXTERNAL_DATA_ROOT   Per-cell directory root scored under an external LLM
#                        judge; when this directory exists, the script also
#                        emits the judge-robustness appendix figure and table.
#                        Default: ../data-external-judge.
#   OUT_DIR              Output dir for tables/ and figures/. Default: ../paper-out.
#
#   bash run_paper_all.sh
#   DATA_ROOT=../data-self-judge EXTERNAL_DATA_ROOT=../data-external-judge OUT_DIR=../../overleaf bash run_paper_all.sh

set -uo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-../data-self-judge}"
EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-../data-external-judge}"
OUT_DIR="${OUT_DIR:-../paper-out}"
TABLES="$OUT_DIR/tables"
FIGURES="$OUT_DIR/figures"
mkdir -p "$TABLES" "$FIGURES"

CONDITION="rm25pct_full_x1"

GPT120B="$DATA_ROOT/gpt-oss-120b_aime2025_jsonl $DATA_ROOT/gpt-oss-120b_brumo_jsonl $DATA_ROOT/gpt-oss-120b_frontierscience_olympiad_jsonl $DATA_ROOT/gpt-oss-120b_hmmt_feb_2026_jsonl"
GPT20B="$DATA_ROOT/gpt-oss-20b_aime2025_jsonl $DATA_ROOT/gpt-oss-20b_brumo_jsonl $DATA_ROOT/gpt-oss-20b_frontierscience_olympiad_jsonl $DATA_ROOT/gpt-oss-20b_hmmt_feb_2026_jsonl"
NEMO30B="$DATA_ROOT/Nemotron-3-Nano-30B-A3B_aime2025_jsonl $DATA_ROOT/Nemotron-3-Nano-30B-A3B_brumo_jsonl $DATA_ROOT/Nemotron-3-Nano-30B-A3B_frontierscience_olympiad_jsonl $DATA_ROOT/Nemotron-3-Nano-30B-A3B_hmmt_feb_2026_jsonl"
NEMO9B="$DATA_ROOT/Nemotron-Nano-9B-v2_aime2025_jsonl $DATA_ROOT/Nemotron-Nano-9B-v2_brumo_jsonl $DATA_ROOT/Nemotron-Nano-9B-v2_frontierscience_olympiad_jsonl $DATA_ROOT/Nemotron-Nano-9B-v2_hmmt_feb_2026_jsonl"
MINI14B="$DATA_ROOT/Ministral-3-14B-Reasoning-2512_aime2025_jsonl $DATA_ROOT/Ministral-3-14B-Reasoning-2512_brumo_jsonl $DATA_ROOT/Ministral-3-14B-Reasoning-2512_frontierscience_olympiad_jsonl $DATA_ROOT/Ministral-3-14B-Reasoning-2512_hmmt_feb_2026_jsonl"

ALL_DIRS="$GPT120B $GPT20B $NEMO30B $NEMO9B $MINI14B"

SUPPRESS_FLAGS="--suppress-condition --suppress-subset-footnote"

echo "=== [1/20] table_wmv (canonical, 3 models) ==="
uv run python paper/table_wmv.py --emit-canonical $SUPPRESS_FLAGS \
    --condition "$CONDITION" \
    --token-points 250000 1000000 5000000 \
    --label tab:wmv \
    -o "$TABLES/table_wmv.tex" \
    --model "GPT-OSS-120B"   $GPT120B \
    --model "Nemotron3-30B"  $NEMO30B \
    --model "Ministral3-14B" $MINI14B

echo "=== [2/20] table_wmv_all (per-model, 5 models) ==="
uv run python paper/table_wmv_all.py $SUPPRESS_FLAGS \
    --condition "$CONDITION" \
    --output-dir "$TABLES" \
    --model "GPT-OSS-120B"   $GPT120B \
    --model "GPT-OSS-20B"    $GPT20B \
    --model "Nemotron3-30B"  $NEMO30B \
    --model "Nemotron2-9B"   $NEMO9B \
    --model "Ministral3-14B" $MINI14B

echo "=== [3/20] table_token_savings (canonical, 3 models) ==="
uv run python paper/table_token_savings.py $SUPPRESS_FLAGS \
    --condition "$CONDITION" \
    -o "$TABLES/table_token_savings.tex" \
    --model "GPT-OSS-120B"   $GPT120B \
    --model "Nemotron3-30B"  $NEMO30B \
    --model "Ministral3-14B" $MINI14B

echo "=== [4/20] table_token_savings_all (per-model, 5 models) ==="
uv run python paper/table_token_savings_all.py $SUPPRESS_FLAGS \
    --condition "$CONDITION" \
    --output-dir "$TABLES" \
    --model "GPT-OSS-120B"   $GPT120B \
    --model "GPT-OSS-20B"    $GPT20B \
    --model "Nemotron3-30B"  $NEMO30B \
    --model "Nemotron2-9B"   $NEMO9B \
    --model "Ministral3-14B" $MINI14B

echo "=== [5/20] table_signal + table_auroc ==="
uv run python paper/table_auroc.py \
    --condition "$CONDITION" \
    --suppress-condition \
    --auroc-out "$TABLES/table_auroc.tex" \
    --signal-out "$TABLES/table_signal.tex" \
    --model "GPT-OSS-120B"   $GPT120B \
    --model "GPT-OSS-20B"    $GPT20B \
    --model "Nemotron3-30B"  $NEMO30B \
    --model "Nemotron2-9B"   $NEMO9B \
    --model "Ministral3-14B" $MINI14B

echo "=== [6/20] table_auroc_verbal_sensitivity ==="
uv run python paper/table_auroc_verbal_sensitivity.py \
    --condition "$CONDITION" \
    -o "$TABLES/table_auroc_verbal_sensitivity.tex" \
    --model "GPT-OSS-120B"   $GPT120B \
    --model "GPT-OSS-20B"    $GPT20B \
    --model "Nemotron3-30B"  $NEMO30B \
    --model "Nemotron2-9B"   $NEMO9B \
    --model "Ministral3-14B" $MINI14B

echo "=== [7/20] table_assumption ==="
uv run --script paper/table_assumption.py \
    $ALL_DIRS \
    --condition "$CONDITION" \
    -o "$TABLES/table_assumption.tex"

echo "=== [8/20] table_token_stats ==="
uv run --script paper/table_token_stats.py \
    $ALL_DIRS \
    --suppress-subset-footnote \
    -o "$TABLES/table_token_stats.tex"

echo "=== [9/20] table_transition_tau (gpt-oss-20b, 4 benchmarks, multi-tau) ==="
uv run python paper/table_transition.py \
    $GPT20B \
    --conditions rm75pct_full_x1 rm50pct_full_x1 rm25pct_full_x1 \
    --mode tau \
    -o "$TABLES/table_transition_tau.tex"

echo "=== [10/20] figure_cost_accuracy (Fig 1 panels + shared legend) ==="
uv run --script paper/figure_cost_accuracy.py \
    "$DATA_ROOT/gpt-oss-120b_frontierscience_olympiad_jsonl/" \
    --condition "$CONDITION" --headline-alpha 0.99 \
    --xlim 10000 3000000 --ylim 41 51 \
    -o "$FIGURES" --stem fig1_ca_120b_fsci --formats pdf \
    --legend-out "$FIGURES/cost_accuracy_legend" \
    --title "GPT-OSS-120B / FrontierScience-Olympiad"
uv run --script paper/figure_cost_accuracy.py \
    "$DATA_ROOT/Ministral-3-14B-Reasoning-2512_aime2025_jsonl/" \
    --condition "$CONDITION" --headline-alpha 0.99 \
    --xlim 30000 10000000 --ylim 45 62 \
    -o "$FIGURES" --stem fig1_ca_14b_aime --formats pdf \
    --no-legend \
    --title "Ministral3-14B / AIME 2025"

echo "=== [11/20] figure_cost_accuracy_all (5x4 grid) ==="
uv run --script paper/figure_cost_accuracy_all.py \
    --root "$DATA_ROOT" \
    --out-dir "$FIGURES" \
    --condition "$CONDITION" \
    --formats pdf

echo "=== [12/20] figure_cost_accuracy_full (5x4 grid, full baseline + oracle) ==="
uv run --script paper/figure_cost_accuracy_full.py \
    --root "$DATA_ROOT" \
    --out-dir "$FIGURES" \
    --condition "$CONDITION" \
    --formats pdf

echo "=== [13/20] figure_cost_accuracy_oracle (5x4 oracle decomposition grid) ==="
uv run --script paper/figure_cost_accuracy_oracle.py \
    --root "$DATA_ROOT" \
    --out-dir "$FIGURES" \
    --condition "$CONDITION" \
    --formats pdf

echo "=== [14/20] figure_pool_correctness_scatter (5x4 per-problem init-vs-regen Pass@1) ==="
uv run --script paper/figure_pool_correctness_scatter.py \
    --root "$DATA_ROOT" \
    --out-dir "$FIGURES" \
    --formats pdf

echo "=== [15/20] figure_roc_all (5x4 macro ROC grid) ==="
uv run --script paper/figure_roc_all.py \
    --root "$DATA_ROOT" \
    --out-dir "$FIGURES" \
    --regen-k 1 \
    --formats pdf

echo "=== [16/20] figure_pass1_vs_rates_all (5 models x 4 panels + 2 GLM tables) ==="
uv run --script paper/figure_pass1_vs_rates_all.py \
    --root "$DATA_ROOT" \
    --out-dir "$FIGURES" \
    --tables-dir "$TABLES" \
    --condition "$CONDITION"

echo "=== [17/20] figure_pass1_vs_signals_all (5 models x 5 baselines) ==="
uv run --script paper/figure_pass1_vs_signals_all.py \
    --root "$DATA_ROOT" \
    --out-dir "$FIGURES"

echo "=== [18/20] figure_signal_violin (120B / FrontierScience-Olympiad) ==="
uv run --script paper/figure_signal_violin.py \
    "$DATA_ROOT/gpt-oss-120b_frontierscience_olympiad_jsonl" \
    --stem signal_violin_120b_fsci --formats pdf \
    -o "$FIGURES"

echo "=== [19/20] table_sensitivity (gpt-oss-20b only) ==="
uv run --script paper/table_sensitivity.py \
    $GPT20B \
    -o "$TABLES/table_wmv_sensitivity.tex"

echo "=== [20/21] table_format_failure ==="
uv run --script paper/table_format_failure.py \
    $ALL_DIRS \
    --suppress-subset-footnote \
    -o "$TABLES/table_format_failure.tex"

if [ -d "$EXTERNAL_DATA_ROOT" ]; then
    echo "=== [21/21] figure_judge_robustness (4 models x 3 benchmarks subset) ==="
    uv run --script paper/figure_judge_robustness.py \
        --self-root "$DATA_ROOT" \
        --external-root "$EXTERNAL_DATA_ROOT" \
        --out-fig-lead "$FIGURES/fig_judge_robustness_lead.pdf" \
        --out-fig-grid "$FIGURES/fig_judge_robustness_grid.pdf" \
        --out-table    "$TABLES/table_judge_robustness.tex"
else
    echo "=== [21/21] figure_judge_robustness SKIPPED (EXTERNAL_DATA_ROOT=$EXTERNAL_DATA_ROOT not found) ==="
fi

echo "=== ALL DONE ==="
echo "Tables:  $TABLES"
echo "Figures: $FIGURES"
