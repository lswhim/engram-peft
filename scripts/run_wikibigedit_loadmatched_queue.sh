#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
SEED="$2"
ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram}"
PY=${PY:-/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python}
cd "$ROOT"

is_complete() {
  local mode="$1"
  local point
  for point in 1000 5000 10000 50000; do
    local result="outputs/semantic_memory/wikibigedit_collision_scaling_50000/$mode/seed_${SEED}/at_${point}.json"
    [[ -f "$result" ]] || return 1
    "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1])).get("status") == "complete"' "$result" \
      >/dev/null 2>&1 || return 1
  done
}

wait_for_same_worker() {
  local mode="$1"
  local pattern="scripts/run_wikibigedit_collision_scaling_worker.sh [0-9]+ ${mode} ${SEED} 50000"
  local found=0
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    if [[ "$found" -eq 0 ]]; then
      echo "WAIT active $mode seed=$SEED from another GPU" >&2
      found=1
    fi
    sleep 30
  done
}

if [[ -n "${WAIT_PID:-}" ]]; then
  echo "waiting for existing queue pid=$WAIT_PID" >&2
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi

for MODE in loadmatched_flatten loadmatched_semantic_keyed; do
  wait_for_same_worker "$MODE"
  if is_complete "$MODE"; then
    echo "SKIP complete $MODE seed=$SEED"
    continue
  fi
  bash scripts/run_wikibigedit_collision_scaling_worker.sh "$GPU_ID" "$MODE" "$SEED" 50000
done
