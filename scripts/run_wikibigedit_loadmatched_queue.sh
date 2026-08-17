#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
SEED="$2"
ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_collision}"
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

if [[ -n "${WAIT_PID:-}" ]]; then
  echo "waiting for existing queue pid=$WAIT_PID" >&2
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi

for MODE in loadmatched_flatten loadmatched_specificity; do
  if is_complete "$MODE"; then
    echo "SKIP complete $MODE seed=$SEED"
    continue
  fi
  bash scripts/run_wikibigedit_collision_scaling_worker.sh "$GPU_ID" "$MODE" "$SEED" 50000
done
