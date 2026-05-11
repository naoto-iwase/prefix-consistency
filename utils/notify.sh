#!/bin/bash
#
# Send a push notification via ntfy.sh
#
# Usage:
#   bash notify.sh "title" "message body"
#   bash notify.sh "title"     # no body
#
# Environment:
#   NTFY_TOPIC  Topic name (required, skips silently if unset)
#
set -euo pipefail

if [ -z "${NTFY_TOPIC:-}" ]; then
    exit 0
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <title> [message]"
    exit 1
fi

TITLE="$1"
MSG="${2:-}"

curl -s -H "Title: $TITLE" -d "$MSG" "ntfy.sh/${NTFY_TOPIC}" > /dev/null 2>&1
