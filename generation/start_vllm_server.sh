#!/bin/bash
#
# Start a vLLM server in the foreground.
#
# Wrapper args (--dp, --venv, --stop) are handled by this
# script.  Everything else is forwarded to vllm as-is, so you can
# pass any vllm-native flag directly.
#
# Usage:
#   bash start_vllm_server.sh gpt-oss-20b                              # defaults
#   bash start_vllm_server.sh gpt-oss-20b --max-model-len 65536        # override default
#   bash start_vllm_server.sh gpt-oss-20b --gpu-memory-utilization 0.95
#   bash start_vllm_server.sh gpt-oss-20b --dp                         # data parallel (TP=1, DP=N)
#   bash start_vllm_server.sh gpt-oss-20b &                            # background
#   bash start_vllm_server.sh --stop                                   # stop running server
#
# Environment variables:
#   MODELS_DIR      Model directory (default: auto-detected from script location)
#   VLLM_MODEL_DIR  Override for MODELS_DIR (default: $MODELS_DIR)
#   VLLM_PORT       Port number (default: 8100)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
_DEFAULT_PARENT="$(cd "$PROJECT_ROOT/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$_DEFAULT_PARENT/models}"
VLLM_MODEL_DIR="${VLLM_MODEL_DIR:-$MODELS_DIR}"
VLLM_PORT="${VLLM_PORT:-8100}"
VLLM_VENV=""
USE_DP=false
PASSTHROUGH=()

show_help() {
    echo "Usage: $0 <model_name> [wrapper options] [-- vllm options]"
    echo ""
    echo "Start a vLLM server in the foreground."
    echo "Models are loaded from ${VLLM_MODEL_DIR}/<model_name>."
    echo ""
    echo "Arguments:"
    echo "  model_name    Model name (e.g. gpt-oss-20b)"
    echo ""
    echo "Wrapper options (handled by this script):"
    echo "  -h, --help                       Show this help"
    echo "  --stop                           Stop vLLM process on port ${VLLM_PORT}"
    echo "  --dp                             Use data parallelism (TP=1, DP=num_gpus)"
    echo "  --venv DIR                       Path to venv with vllm installed (required)"
    echo ""
    echo "All other options are forwarded to vllm (e.g. --max-model-len,"
    echo "--gpu-memory-utilization, --trust-remote-code, --language-model-only, etc.)."
    echo ""
    echo "Defaults applied unless overridden:"
    echo "  --max-model-len 131072"
    echo "  --gpu-memory-utilization 0.9"
    echo "  --enable-prefix-caching"
    echo "  --async-scheduling"
    echo ""
    echo "Environment variables:"
    echo "  MODELS_DIR       Model directory (default: auto-detected)"
    echo "  VLLM_MODEL_DIR   Override for MODELS_DIR (default: \$MODELS_DIR)"
    echo "  VLLM_PORT       Port number (default: 8100)"
}

stop_vllm() {
    local pids
    pids=$(lsof -ti :"$VLLM_PORT" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        echo "No process found on port ${VLLM_PORT}"
        return 0
    fi
    echo "Stopping processes on port ${VLLM_PORT} (PID: $pids)"
    echo "$pids" | xargs kill 2>/dev/null || true
    echo "Stop signal sent"
}

MODEL_NAME="${1:-}"
if [ "$MODEL_NAME" = "-h" ] || [ "$MODEL_NAME" = "--help" ]; then
    show_help
    exit 0
fi
if [ "$MODEL_NAME" = "--stop" ]; then
    stop_vllm
    exit 0
fi
if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: model name required"
    echo ""
    show_help
    exit 1
fi
shift

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)                 show_help; exit 0 ;;
        --stop)                    stop_vllm; exit 0 ;;
        --venv)                    VLLM_VENV="$2"; shift 2 ;;
        --dp)                      USE_DP=true; shift ;;
        *)                         PASSTHROUGH+=("$1"); shift ;;
    esac
done

VLLM_MODEL_PATH="${VLLM_MODEL_DIR}/${MODEL_NAME}"

if [ ! -d "$VLLM_MODEL_PATH" ]; then
    echo "ERROR: model not found: $VLLM_MODEL_PATH"
    exit 1
fi

cd "$PROJECT_ROOT"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: no GPU detected"
    exit 1
fi

if [ "$USE_DP" = true ]; then
    TP_SIZE=1
    DP_SIZE="$NUM_GPUS"
else
    TP_SIZE="$NUM_GPUS"
    DP_SIZE=1
fi

echo "=== Starting vLLM server ==="
echo "Model:           $VLLM_MODEL_PATH"
echo "Port:            $VLLM_PORT"
echo "GPUs:            $NUM_GPUS (TP=$TP_SIZE, DP=$DP_SIZE)"
echo "venv:            ${VLLM_VENV:-(default)}"
if [ ${#PASSTHROUGH[@]} -gt 0 ]; then
    echo "extra args:      ${PASSTHROUGH[*]}"
fi
echo "PID:             $$"
echo ""

SERVE_ARGS=(serve "$VLLM_MODEL_PATH"
    --port "$VLLM_PORT"
    --max-model-len 131072
    --gpu-memory-utilization 0.9
    --served-model-name "$MODEL_NAME"
    --tensor-parallel-size "$TP_SIZE"
    --data-parallel-size "$DP_SIZE"
    --enable-prefix-caching
    --async-scheduling
)

# Passthrough args come last so they can override defaults above
if [ ${#PASSTHROUGH[@]} -gt 0 ]; then
    SERVE_ARGS+=("${PASSTHROUGH[@]}")
fi

if [ -z "$VLLM_VENV" ]; then
    echo "ERROR: --venv is required (vllm is not in the project venv)."
    echo ""
    echo "  Example:  bash start_vllm_server.sh $MODEL_NAME --venv .venv-vllm"
    echo ""
    echo "  To create a vllm venv:"
    echo "    uv venv .venv-vllm --python 3.12 --seed --managed-python"
    echo "    uv pip install --python .venv-vllm/bin/python vllm --torch-backend=auto"
    exit 1
fi

exec "$VLLM_VENV/bin/vllm" "${SERVE_ARGS[@]}"
