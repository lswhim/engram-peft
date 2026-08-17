#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
SEED="$2"
ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_collision}"
cd "$ROOT"

if [[ -n "${WAIT_PID:-}" ]]; then
  echo "waiting for existing queue pid=$WAIT_PID" >&2
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi

for MODE in loadmatched_flatten loadmatched_specificity; do
  bash scripts/run_wikibigedit_collision_scaling_worker.sh "$GPU_ID" "$MODE" "$SEED" 50000
done
