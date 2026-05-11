#!/usr/bin/env bash
#
# Regen-consistency pipeline: init generation -> regeneration -> JSONL -> signal analysis.
#
# Regenerates all init answers by default (use --n-regen-gens to limit).
# Cost-equivalent comparison is handled at analysis time by
# weighted_sampling_batch.py.
#
# Requires a running vLLM server (see generation/start_vllm_server.sh).
#
# Per-cell JSONLs are written to ${JSONL_BASE}/${TAG}_jsonl/. JSONL_BASE
# defaults to <repo-root>/data-self-judge; override via env var if you
# want a different output location (e.g., JSONL_BASE=../data-external-judge).
#
# Usage:
#   bash generation/run_regen.sh \
#     --tag gpt-oss-20b_aime2025 \
#     --dataset aime2025 --model gpt-oss-20b \
#     --n-init-gens 128 \
#     --remove-pcts 50 --regen-count 3
#
#   # Pct sweep with fewer regen answers
#   bash generation/run_regen.sh \
#     --tag gpt-oss-20b_aime2025 \
#     --dataset aime2025 --model gpt-oss-20b \
#     --n-init-gens 128 \
#     --remove-pcts 25,50,75 --regen-count 1 --n-regen-gens 64
#
#   # Init generation only (no regen)
#   bash generation/run_regen.sh \
#     --tag gpt-oss-20b_aime2025 \
#     --dataset aime2025 --model gpt-oss-20b \
#     --n-init-gens 10 --skip-regen
#
#   # With marker-based regen
#   bash generation/run_regen.sh \
#     --tag gpt-oss-20b_aime2025 \
#     --dataset aime2025 --model gpt-oss-20b \
#     --n-init-gens 128 \
#     --remove-pcts 50 --regen-count 3 \
#     --marker-regen --marker-regen-count 1
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# =====================================================================
# Defaults
# =====================================================================

# --- Experiment identity ---
TAG=""
DATASET=""
MODEL=""
N_INIT_GENS=""

# --- Regeneration (not needed with --skip-regen) ---
REMOVE_PCTS=""
REMOVE_TOKENS=""
TRUNCATE_SCOPE="full"
REGEN_COUNT=""
N_REGEN_GENS=""
INSERT_COT_CLOSING=""
MARKER_REGEN=""
MARKER_REGEN_COUNT=""
N_MARKER_REGEN_GENS=""
NO_MARKER_BUDGET_STOP=""

# --- Execution ---
PROBLEMS=""
PARALLEL=""
TIMEOUT=3600

# --- Model behavior ---
REASONING_EFFORT=""
NO_THINK=""

# --- Phase control ---
SKIP_REGEN=""
SKIP_ANALYSIS=""
ENRICH_EXTRA=()

# =====================================================================
# Help
# =====================================================================

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Run the full pipeline: init generation, regeneration, JSONL conversion,"
    echo "and regen signal analysis."
    echo ""
    echo "Required:"
    echo "  --tag TAG                        Experiment tag (used for output dirs)"
    echo "  --dataset NAME                   Dataset name (e.g. aime2025, hmmt, brumo, frontierscience_olympiad)"
    echo "  --model NAME                     Model name (e.g. gpt-oss-20b, gpt-oss-120b)"
    echo "  --n-init-gens N                  Number of init answers per problem"
    echo ""
    echo "Regeneration (not needed with --skip-regen):"
    echo "  --remove-pcts P1,P2,...          Remove by percentage (e.g. 50 or 25,50,75)"
    echo "  --remove-tokens N                Remove by token count (mutually exclusive with --remove-pcts)"
    echo "  --regen-count N                  Regens per truncation point per answer (e.g. 1, 3)"
    echo "  --truncate-scope SCOPE           cot or full (default: full)"
    echo "  --n-regen-gens N                 Regen only the first N init answers (default: all)"
    echo "  --insert-cot-closing             Insert CoT closing delimiter at truncation point"
    echo "  --marker-regen                   Also run marker-based regen"
    echo "  --marker-regen-count N           Regens per marker boundary (default: 1)"
    echo "  --n-marker-regen-gens N          Marker-regen the first N init answers (default: --n-regen-gens)"
    echo "  --no-marker-budget-stop          Disable per-problem token budget early stopping (on by default)"
    echo ""
    echo "Execution:"
    echo "  --problems P1,P2,...             Problem indices filter (e.g. 13,14,29)"
    echo "  --parallel N                     Parallel requests (default: N_GPUs x 64)"
    echo "  --timeout SEC                    Request timeout in seconds (default: 3600)"
    echo ""
    echo "Model behavior:"
    echo "  --reasoning-effort LEVEL         Set reasoning effort: low|medium|high"
    echo "  --no-think                       Disable CoT (no-think mode)"
    echo ""
    echo "Phase control:"
    echo "  --skip-regen                     Skip regeneration (Phase 2); generate + enrich + convert only"
    echo "  --skip-analysis                  Skip Phase 5 analysis (signal + wmv eval)"
    echo ""
    echo "Enrich options (forwarded to enrich.py):"
    echo "  --skip-extraction                Skip extracted answer (step 1)"
    echo "  --skip-confidence                Skip logprob confidence (step 2)"
    echo "  --skip-verbal-0-100              Skip verbal 0-100 confidence (step 3)"
    echo "  --skip-binary                    Skip binary confidence (step 4)"
    echo "  --skip-answer-map                Skip answer map building (step 5)"
    echo "  --regen-verbal-0-100             Also run verbal 0-100 on regen files"
    echo "  --regen-binary                   Also run binary confidence on regen files"
    echo "  --force                          Re-compute enrichment even if already done"
    echo ""
    echo "  -h, --help                       Show this help"
}

# =====================================================================
# Argument parsing
# =====================================================================

while [ $# -gt 0 ]; do
    case "$1" in
        # Experiment identity
        --tag)                   TAG="$2"; shift 2 ;;
        --dataset)               DATASET="$2"; shift 2 ;;
        --model)                 MODEL="$2"; shift 2 ;;
        --n-init-gens)           N_INIT_GENS="$2"; shift 2 ;;
        # Regeneration
        --remove-pcts)           REMOVE_PCTS="$2"; shift 2 ;;
        --remove-tokens)         REMOVE_TOKENS="$2"; shift 2 ;;
        --regen-count)           REGEN_COUNT="$2"; shift 2 ;;
        --truncate-scope)        TRUNCATE_SCOPE="$2"; shift 2 ;;
        --n-regen-gens)          N_REGEN_GENS="$2"; shift 2 ;;
        --insert-cot-closing)    INSERT_COT_CLOSING="1"; shift ;;
        --marker-regen)          MARKER_REGEN="1"; shift ;;
        --marker-regen-count)    MARKER_REGEN_COUNT="$2"; shift 2 ;;
        --n-marker-regen-gens)   N_MARKER_REGEN_GENS="$2"; shift 2 ;;
        --no-marker-budget-stop) NO_MARKER_BUDGET_STOP="1"; shift ;;
        # Execution
        --problems)              PROBLEMS="$2"; shift 2 ;;
        --parallel)              PARALLEL="$2"; shift 2 ;;
        --timeout)               TIMEOUT="$2"; shift 2 ;;
        # Model behavior
        --reasoning-effort)      REASONING_EFFORT="$2"; shift 2 ;;
        --no-think)              NO_THINK="1"; shift ;;
        # Phase control
        --skip-regen)            SKIP_REGEN="1"; shift ;;
        --skip-analysis)         SKIP_ANALYSIS="1"; shift ;;
        # Enrich flags (forwarded to enrich.py as-is)
        --skip-extraction|--skip-confidence|--skip-verbal-0-100|\
        --skip-binary|--skip-answer-map|\
        --regen-verbal-0-100|--regen-binary|--force)
            ENRICH_EXTRA+=("$1"); shift ;;
        # Help
        -h|--help)               show_help; exit 0 ;;
        *)
            echo "ERROR: unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

# =====================================================================
# Validation
# =====================================================================

[ -z "$TAG" ]        && { echo "ERROR: --tag is required"; exit 1; }
[ -z "$DATASET" ]    && { echo "ERROR: --dataset is required"; exit 1; }
[ -z "$MODEL" ]      && { echo "ERROR: --model is required"; exit 1; }
[ -z "$N_INIT_GENS" ] && { echo "ERROR: --n-init-gens is required"; exit 1; }

# Auto-detect parallel from GPU count if not specified
if [ -z "$PARALLEL" ]; then
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    N_GPUS=${N_GPUS:-1}
    PARALLEL=$((N_GPUS * 64))
    echo "Auto-detected ${N_GPUS} GPU(s), setting --parallel ${PARALLEL}"
fi

if [ -z "$SKIP_REGEN" ]; then
    [ -z "$REGEN_COUNT" ] && { echo "ERROR: --regen-count is required (or use --skip-regen)"; exit 1; }
    if [ -z "$REMOVE_PCTS" ] && [ -z "$REMOVE_TOKENS" ] && [ -z "$MARKER_REGEN" ]; then
        echo "ERROR: at least one of --remove-pcts, --remove-tokens, or --marker-regen is required (or use --skip-regen)"
        exit 1
    fi
fi
if [ -n "$REMOVE_PCTS" ] && [ -n "$REMOVE_TOKENS" ]; then
    echo "ERROR: --remove-pcts and --remove-tokens are mutually exclusive"
    exit 1
fi

# =====================================================================
# Derived variables
# =====================================================================

# Build removal tags for fixed-% regen
REMOVAL_TAGS=()
REMOVE_PCTS_ARRAY=()
if [ -n "$REMOVE_PCTS" ]; then
    IFS=',' read -ra REMOVE_PCTS_ARRAY <<< "$REMOVE_PCTS"
    for PCT in "${REMOVE_PCTS_ARRAY[@]}"; do
        REMOVAL_TAGS+=("rm${PCT}pct_${TRUNCATE_SCOPE}")
    done
elif [ -n "$REMOVE_TOKENS" ]; then
    REMOVAL_TAGS+=("rm${REMOVE_TOKENS}tok_${TRUNCATE_SCOPE}")
fi

# Marker regen defaults: fall back to shared values if not specified
if [ -n "$MARKER_REGEN" ]; then
    : "${MARKER_REGEN_COUNT:=1}"
    : "${N_MARKER_REGEN_GENS:=$N_REGEN_GENS}"
fi

# Directories
DATA_DIR="${PROJECT_ROOT}/generation/${TAG}_data"
JSONL_BASE="${JSONL_BASE:-${PROJECT_ROOT}/data-self-judge}"
JSONL_DIR="${JSONL_BASE}/${TAG}_jsonl"
INIT_DIR="${DATA_DIR}/init"
MARKER_DIR="${DATA_DIR}/regen_from_markers"
mkdir -p "${PROJECT_ROOT}/logs" "$DATA_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Build shared flags (ordered to match .py conventions)
EFFORT_FLAG=()
[ -n "$REASONING_EFFORT" ] && EFFORT_FLAG=(--reasoning-effort "$REASONING_EFFORT")

NO_THINK_FLAG=()
[ -n "$NO_THINK" ] && NO_THINK_FLAG=(--no-think)

INSERT_COT_CLOSING_FLAG=()
[ -n "$INSERT_COT_CLOSING" ] && INSERT_COT_CLOSING_FLAG=(--insert-cot-closing)

PROBLEMS_FLAG=()
[ -n "$PROBLEMS" ] && PROBLEMS_FLAG=(--problems "$PROBLEMS")

ANSWERS_FLAG=()
[ -n "$N_REGEN_GENS" ] && ANSWERS_FLAG=(--answers "$(seq -s, 0 $((N_REGEN_GENS - 1)))")

MARKER_ANSWERS_FLAG=()
if [ -n "$MARKER_REGEN" ] && [ -n "$N_MARKER_REGEN_GENS" ]; then
    MARKER_ANSWERS_FLAG=(--answers "$(seq -s, 0 $((N_MARKER_REGEN_GENS - 1)))")
fi

# =====================================================================
# Print config
# =====================================================================

echo "=== Regen-consistency pipeline ==="
echo "Tag:             $TAG"
echo "Dataset:         $DATASET"
echo "Model:           $MODEL"
echo "N init gens:     $N_INIT_GENS"
[ -n "$PROBLEMS" ] && echo "Problems:        $PROBLEMS"
if [ -n "$SKIP_REGEN" ]; then
    echo "Skip regen:      yes"
else
    if [ -n "$REMOVE_PCTS" ]; then
        echo "Remove pcts:     $REMOVE_PCTS"
    elif [ -n "$REMOVE_TOKENS" ]; then
        echo "Remove tokens:   $REMOVE_TOKENS"
    fi
    if [ ${#REMOVAL_TAGS[@]} -gt 0 ]; then
        echo "Scope:           $TRUNCATE_SCOPE"
        echo "Removal tags:    ${REMOVAL_TAGS[*]}"
    fi
    echo "Regen count:     $REGEN_COUNT"
    echo "N regen gens:    ${N_REGEN_GENS:-all}"
    echo "Insert CoT closing: ${INSERT_COT_CLOSING:-no}"
    echo "Marker regen:    ${MARKER_REGEN:-no}"
    if [ -n "$MARKER_REGEN" ]; then
        echo "  regen-count:   $MARKER_REGEN_COUNT"
        echo "  n-regen-gens:  ${N_MARKER_REGEN_GENS:-all}"
        [ -z "$NO_MARKER_BUDGET_STOP" ] && echo "  budget-stop:   yes"
    fi
fi
echo "Parallel:        $PARALLEL"
echo "Timeout:         ${TIMEOUT}s"
echo "Reasoning:       ${REASONING_EFFORT:-(not set)}"
echo "No-think:        ${NO_THINK:-no}"
echo "PID:             $$"
echo ""

log "Experiment started"

# ============================================================
# Preflight: wait for vLLM server
# ============================================================
log "Waiting for vLLM server on port 8100..."
while true; do
    HEALTH=$(curl -sf http://localhost:8100/health 2>/dev/null) && break
    sleep 10
done
log "vLLM server OK (health: ${HEALTH})"

# ============================================================
# Phase 1: Init generation
# ============================================================
log "=== Phase 1: Init generation (n=${N_INIT_GENS}) ==="

uv run python generate.py \
    --dataset "$DATASET" \
    --model-name "$MODEL" \
    --num-answers "$N_INIT_GENS" \
    --out-dir "$INIT_DIR" \
    --parallel "$PARALLEL" \
    --timeout "$TIMEOUT" \
    "${PROBLEMS_FLAG[@]}" \
    "${EFFORT_FLAG[@]}" \
    "${NO_THINK_FLAG[@]}" \
    --save-logprobs

log "=== Phase 1 complete ==="

# ============================================================
# Phase 2: Regeneration
# ============================================================
if [ -n "$SKIP_REGEN" ]; then
    log "=== Phase 2: Regeneration skipped (--skip-regen) ==="
fi

if [ -z "$SKIP_REGEN" ] && [ ${#REMOVAL_TAGS[@]} -gt 0 ]; then
    log "=== Phase 2a: Fixed-% regeneration ==="
    for i in "${!REMOVAL_TAGS[@]}"; do
        RTAG="${REMOVAL_TAGS[$i]}"
        OUT_DIR="${DATA_DIR}/regen_${RTAG}"
        mkdir -p "$OUT_DIR"

        REMOVE_FLAG=()
        if [ -n "$REMOVE_PCTS" ]; then
            REMOVE_FLAG=(--remove-pct "${REMOVE_PCTS_ARRAY[$i]}")
        else
            REMOVE_FLAG=(--remove-tokens "$REMOVE_TOKENS")
        fi

        log "--- ${RTAG} regen-count=${REGEN_COUNT} ---"
        uv run python regenerate.py \
            --dataset "$DATASET" \
            --model-name "$MODEL" \
            --init-dir "$INIT_DIR" \
            "${REMOVE_FLAG[@]}" \
            --truncate-scope "$TRUNCATE_SCOPE" \
            --regen-count "$REGEN_COUNT" \
            "${INSERT_COT_CLOSING_FLAG[@]}" \
            --out-dir "$OUT_DIR" \
            --parallel "$PARALLEL" \
            --timeout "$TIMEOUT" \
            "${PROBLEMS_FLAG[@]}" \
            "${ANSWERS_FLAG[@]}" \
            "${EFFORT_FLAG[@]}" \
            "${NO_THINK_FLAG[@]}" \
            --save-logprobs \
            && log "--- ${RTAG} done ---" \
            || { log "ERROR: ${RTAG} failed"; bash "$PROJECT_ROOT/utils/notify.sh" "[${TAG}] ${RTAG} failed"; }

        COUNT=$(find "$OUT_DIR" -name "*.txt" | wc -l)
        log "    ${RTAG}: ${COUNT} files"
    done
fi

if [ -z "$SKIP_REGEN" ] && [ -n "$MARKER_REGEN" ]; then
    log "=== Phase 2b: Marker-based regeneration (regen-count=${MARKER_REGEN_COUNT}) ==="
    mkdir -p "$MARKER_DIR"

    # budget-stop is on by default to avoid generating unused marker regens
    if [ -z "$NO_MARKER_BUDGET_STOP" ]; then
        MARKER_LIMIT_FLAG=(--budget-stop)
    else
        MARKER_LIMIT_FLAG=("${MARKER_ANSWERS_FLAG[@]}")
    fi

    uv run python regenerate_from_markers.py \
        --dataset "$DATASET" \
        --model-name "$MODEL" \
        --init-dir "$INIT_DIR" \
        --regen-count "$MARKER_REGEN_COUNT" \
        --out-dir "$MARKER_DIR" \
        --parallel "$PARALLEL" \
        --timeout "$TIMEOUT" \
        "${PROBLEMS_FLAG[@]}" \
        "${MARKER_LIMIT_FLAG[@]}" \
        "${EFFORT_FLAG[@]}" \
        "${NO_THINK_FLAG[@]}" \
        --save-logprobs \
        && log "--- marker regen done ---" \
        || { log "ERROR: marker regen failed"; bash "$PROJECT_ROOT/utils/notify.sh" "[${TAG}] marker regen failed"; }

    COUNT=$(find "$MARKER_DIR" -name "*.txt" | wc -l)
    log "    marker regen: ${COUNT} files"
fi

if [ -z "$SKIP_REGEN" ]; then
    bash "$PROJECT_ROOT/utils/notify.sh" "[${TAG}] Phase 2 done"
fi

# ============================================================
# Phase 3: Enrichment (confidence + verbalized + answer map)
# ============================================================
log "=== Phase 3: Enrichment ==="

ENRICH_DIRS=("$INIT_DIR")
REGEN_DIR_ARGS=()
for RTAG in "${REMOVAL_TAGS[@]}"; do
    ENRICH_DIRS+=("${DATA_DIR}/regen_${RTAG}")
    REGEN_DIR_ARGS+=("${DATA_DIR}/regen_${RTAG}")
done

MARKER_REGEN_DIR_ARGS=()
if [ -n "$MARKER_REGEN" ]; then
    ENRICH_DIRS+=("$MARKER_DIR")
    REGEN_DIR_ARGS+=("$MARKER_DIR")
    MARKER_REGEN_DIR_ARGS=(--marker-regen-dirs "$MARKER_DIR")
fi

uv run python enrich.py \
    --dataset "$DATASET" \
    --model-name "$MODEL" \
    --init-dir "$INIT_DIR" \
    --data-dirs "${ENRICH_DIRS[@]}" \
    --regen-dirs "${REGEN_DIR_ARGS[@]}" \
    "${MARKER_REGEN_DIR_ARGS[@]}" \
    --parallel "$PARALLEL" \
    --timeout "$TIMEOUT" \
    "${PROBLEMS_FLAG[@]}" \
    "${EFFORT_FLAG[@]}" \
    "${NO_THINK_FLAG[@]}" \
    "${ENRICH_EXTRA[@]}"

log "=== Phase 3 complete ==="

# ============================================================
# Phase 4: JSONL conversion
# ============================================================
log "=== Phase 4: JSONL conversion ==="

# Baseline (init only)
BASELINE_FILE="$JSONL_DIR/analysis_${DATASET}_${MODEL}_init.jsonl"
log "--- baseline: $BASELINE_FILE ---"
uv run python convert_to_jsonl.py \
    --dataset "$DATASET" \
    --model-name "$MODEL" \
    --init-dir "$INIT_DIR" \
    --output "$BASELINE_FILE" \
    "${EFFORT_FLAG[@]}" \
    2>&1

# Fixed-% regen JSONLs
for RTAG in "${REMOVAL_TAGS[@]}"; do
    REGEN_DIR="${DATA_DIR}/regen_${RTAG}"
    JSONL_FILE="$JSONL_DIR/analysis_${DATASET}_${MODEL}_${RTAG}_x${REGEN_COUNT}.jsonl"
    log "--- ${RTAG}: $JSONL_FILE ---"

    uv run python convert_to_jsonl.py \
        --dataset "$DATASET" \
        --model-name "$MODEL" \
        --init-dir "$INIT_DIR" \
        --regen-dir "$REGEN_DIR" \
        --regen-count "$REGEN_COUNT" \
        --output "$JSONL_FILE" \
        "${ANSWERS_FLAG[@]}" \
        "${EFFORT_FLAG[@]}" \
        2>&1
done

# Marker regen JSONL
# With --budget-stop, include all init answers (budget controls regen volume,
# not the answer count in the JSONL). Without it, use MARKER_ANSWERS_FLAG.
if [ -n "$MARKER_REGEN" ]; then
    MARKER_JSONL="$JSONL_DIR/analysis_${DATASET}_${MODEL}_markers_x${MARKER_REGEN_COUNT}.jsonl"
    log "--- markers: $MARKER_JSONL ---"

    MARKER_JSONL_ANSWERS_FLAG=()
    if [ -n "$NO_MARKER_BUDGET_STOP" ]; then
        MARKER_JSONL_ANSWERS_FLAG=("${MARKER_ANSWERS_FLAG[@]}")
    fi

    uv run python convert_to_jsonl.py \
        --dataset "$DATASET" \
        --model-name "$MODEL" \
        --init-dir "$INIT_DIR" \
        --marker-regen-dir "$MARKER_DIR" \
        --output "$MARKER_JSONL" \
        "${MARKER_JSONL_ANSWERS_FLAG[@]}" \
        "${EFFORT_FLAG[@]}" \
        2>&1
fi

log "=== Phase 4 complete ==="

# ============================================================
# Phase 5: Analysis (signal + batch eval)
# ============================================================
if [ -n "$SKIP_ANALYSIS" ]; then
    log "=== Phase 5: Analysis skipped (--skip-analysis) ==="
else
    log "=== Phase 5: Analysis ==="
    bash "$PROJECT_ROOT/analysis/analyze_jsonls.sh" "$JSONL_DIR"
fi

log "=== All phases complete ==="
if [ -n "$SKIP_REGEN" ]; then
    bash "$PROJECT_ROOT/utils/notify.sh" "[${TAG}] Pipeline done (init-only)"
else
    bash "$PROJECT_ROOT/utils/notify.sh" "[${TAG}] Pipeline done" "tags=${REMOVAL_TAGS[*]:-markers} regen-count=${REGEN_COUNT}"
fi
