#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
WAIT_PID="${2:-}"
shift 2 || true

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
fi

if [[ "$#" -eq 0 ]]; then
  set -- semantic_factorized semantic_flatten shuffled_credit
fi

for mode in "$@"; do
  bash scripts/run_wikibigedit_credit_worker.sh "$GPU_ID" "$mode" 42
done

