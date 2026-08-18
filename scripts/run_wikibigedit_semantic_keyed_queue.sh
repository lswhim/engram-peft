#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
SEED="$2"
ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_collision}"

cd "$ROOT"
for mode in semantic_keyed loadmatched_semantic_keyed shuffled_semantic_keyed; do
  result="outputs/semantic_memory/wikibigedit_collision_scaling_50000/$mode/seed_${SEED}/at_50000.json"
  if [[ -s "$result" ]] && python - "$result" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    raise SystemExit(0 if json.load(handle).get("status") == "complete" else 1)
PY
  then
    echo "[skip] complete: $result"
    continue
  fi
  bash scripts/run_wikibigedit_collision_scaling_worker.sh \
    "$GPU_ID" "$mode" "$SEED" 50000
done
