#!/usr/bin/env bash
#
# Run weighted majority voting (WMV) evaluation with the dense token-budget
# grid required by paper/table_token_savings.py.
#
# Usage:
#   bash analysis/analyze_jsonls.sh data-self-judge/gpt-oss-20b_aime2025_jsonl/
#
set -euo pipefail

cd "$(dirname "$0")"

JSONL_DIR="${1:?Usage: $0 <jsonl-dir>}"
[ -d "$JSONL_DIR" ] || { echo "ERROR: directory not found: $JSONL_DIR" >&2; exit 1; }

uv run --script wmv.py -- "$JSONL_DIR" --dense
