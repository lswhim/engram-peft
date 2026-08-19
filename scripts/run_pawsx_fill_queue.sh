#!/usr/bin/env bash
# Fill the remaining PAWS-X methods onto whichever GPU frees first.
# One background process per method, so a method only starts when a card is idle.
set -uo pipefail

ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram
cd "$ROOT"

find_free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 800 {print $1; exit}'
}

for MODE in arithmetic semantic_flatten; do
  while true; do
    GPU="$(find_free_gpu)"
    if [ -n "$GPU" ]; then
      bash scripts/run_pawsx_semantic_keyed_worker.sh "$GPU" "$MODE" \
        > "run_logs/pawsx_${MODE}_gpu${GPU}.log" 2>&1
      break
    fi
    sleep 20
  done
done

echo "PAWSX_FILL_QUEUE_DONE"
