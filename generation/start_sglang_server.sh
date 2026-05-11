#!/bin/bash
#
# Start an SGLang server in the foreground.
#
# Wrapper args (--dp, --venv, --stop) are handled by this
# script.  Everything else is forwarded to sglang as-is, so you can
# pass any sglang-native flag directly.
#
# Usage:
#   bash start_sglang_server.sh gpt-oss-20b                              # defaults
#   bash start_sglang_server.sh gpt-oss-20b --context-length 65536       # override default
#   bash start_sglang_server.sh gpt-oss-20b --mem-fraction-static 0.95
#   bash start_sglang_server.sh gpt-oss-20b --dp                         # data parallel (TP=1, DP=N)
#   bash start_sglang_server.sh gpt-oss-20b &                            # background
#   bash start_sglang_server.sh --stop                                   # stop running server
#
# Environment variables:
#   MODELS_DIR        Model directory (default: auto-detected from script location)
#   SGLANG_MODEL_DIR  Override for MODELS_DIR (default: $MODELS_DIR)
#   SGLANG_PORT       Port number (default: 8100)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
_DEFAULT_PARENT="$(cd "$PROJECT_ROOT/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$_DEFAULT_PARENT/models}"
SGLANG_MODEL_DIR="${SGLANG_MODEL_DIR:-$MODELS_DIR}"
SGLANG_PORT="${SGLANG_PORT:-8100}"
SGLANG_VENV=""
USE_DP=false
PASSTHROUGH=()

show_help() {
    echo "Usage: $0 <model_name> [wrapper options] [-- sglang options]"
    echo ""
    echo "Start an SGLang server in the foreground."
    echo "Models are loaded from ${SGLANG_MODEL_DIR}/<model_name>."
    echo ""
    echo "Arguments:"
    echo "  model_name    Model name (e.g. gpt-oss-20b)"
    echo ""
    echo "Wrapper options (handled by this script):"
    echo "  -h, --help                       Show this help"
    echo "  --stop                           Stop SGLang process on port ${SGLANG_PORT}"
    echo "  --dp                             Use data parallelism (TP=1, DP=num_gpus)"
    echo "  --venv DIR                       Path to venv with sglang installed (required)"
    echo ""
    echo "All other options are forwarded to sglang (e.g. --context-length,"
    echo "--mem-fraction-static, --trust-remote-code, --language-only, etc.)."
    echo ""
    echo "Defaults applied unless overridden:"
    echo "  --context-length 131072"
    echo "  --mem-fraction-static 0.8"
    echo "  --chunked-prefill-size 4096"
    echo ""
    echo "Environment variables:"
    echo "  MODELS_DIR        Model directory (default: auto-detected)"
    echo "  SGLANG_MODEL_DIR  Override for MODELS_DIR (default: \$MODELS_DIR)"
    echo "  SGLANG_PORT       Port number (default: 8100)"
}

stop_sglang() {
    local pids
    pids=$(lsof -ti :"$SGLANG_PORT" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        echo "No process found on port ${SGLANG_PORT}"
        return 0
    fi
    echo "Stopping processes on port ${SGLANG_PORT} (PID: $pids)"
    echo "$pids" | xargs kill 2>/dev/null || true
    echo "Stop signal sent"
}

MODEL_NAME="${1:-}"
if [ "$MODEL_NAME" = "-h" ] || [ "$MODEL_NAME" = "--help" ]; then
    show_help
    exit 0
fi
if [ "$MODEL_NAME" = "--stop" ]; then
    stop_sglang
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
        --stop)                    stop_sglang; exit 0 ;;
        --venv)                    SGLANG_VENV="$2"; shift 2 ;;
        --dp)                      USE_DP=true; shift ;;
        *)                         PASSTHROUGH+=("$1"); shift ;;
    esac
done

MODEL_PATH="${SGLANG_MODEL_DIR}/${MODEL_NAME}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model not found: $MODEL_PATH"
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

echo "=== Starting SGLang server ==="
echo "Model:           $MODEL_PATH"
echo "Port:            $SGLANG_PORT"
echo "GPUs:            $NUM_GPUS (TP=$TP_SIZE, DP=$DP_SIZE)"
echo "venv:            ${SGLANG_VENV:-(default)}"
if [ ${#PASSTHROUGH[@]} -gt 0 ]; then
    echo "extra args:      ${PASSTHROUGH[*]}"
fi
echo "PID:             $$"
echo ""

SERVE_ARGS=(
    --model-path "$MODEL_PATH"
    --port "$SGLANG_PORT"
    --host 0.0.0.0
    --context-length 131072
    --mem-fraction-static 0.8
    --served-model-name "$MODEL_NAME"
    --tp-size "$TP_SIZE"
    --dp-size "$DP_SIZE"
    --chunked-prefill-size 4096
)

# Passthrough args come last so they can override defaults above
if [ ${#PASSTHROUGH[@]} -gt 0 ]; then
    SERVE_ARGS+=("${PASSTHROUGH[@]}")
fi

if [ -z "$SGLANG_VENV" ]; then
    echo "ERROR: --venv is required (sglang is not in the project venv)."
    echo ""
    echo "  Example:  bash start_sglang_server.sh $MODEL_NAME --venv .venv-sglang"
    exit 1
fi

exec "$SGLANG_VENV/bin/python" -m sglang.launch_server "${SERVE_ARGS[@]}"
