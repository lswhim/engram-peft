#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
SEED="$2"
ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram}"
cd "$ROOT"

for MODE in semantic_flatten shuffled_flatten shuffled_specificity semantic_specificity; do
  bash scripts/run_wikibigedit_collision_scaling_worker.sh "$GPU_ID" "$MODE" "$SEED" 50000
done
