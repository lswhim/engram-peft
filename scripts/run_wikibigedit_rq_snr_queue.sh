#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
WAIT_PID="$2"
shift 2

ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_collision}"
GAIN_ROOT="$ROOT/rq_tables/wikibigedit50k_qwen3emb4b_M8K16"
SHUFFLED_GAIN_ROOT="${GAIN_ROOT}_runtime_shuffled_seed42"
LOADMATCHED_GAIN_ROOT="${GAIN_ROOT}_loadmatched_seed42"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 GPU_ID WAIT_PID MODE:SEED [MODE:SEED ...]" >&2
  exit 2
fi

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

# The backfill writes both arrays atomically.  Do not let a formal run silently
# fall back to a different routing signal if either n-gram order is incomplete.
until [[ -s "$GAIN_ROOT/residual_gains_2.npy" && -s "$GAIN_ROOT/residual_gains_3.npy" &&
         -s "$SHUFFLED_GAIN_ROOT/residual_gains_2.npy" && -s "$SHUFFLED_GAIN_ROOT/residual_gains_3.npy" &&
         -s "$LOADMATCHED_GAIN_ROOT/residual_gains_2.npy" && -s "$LOADMATCHED_GAIN_ROOT/residual_gains_3.npy" ]]; do
  sleep 30
done

for SPEC in "$@"; do
  MODE="${SPEC%%:*}"
  SEED="${SPEC##*:}"
  bash "$ROOT/scripts/run_wikibigedit_collision_scaling_worker.sh" \
    "$GPU_ID" "$MODE" "$SEED" 50000
done
