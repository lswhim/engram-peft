#!/usr/bin/env bash
set -euo pipefail

WAIT_PID=${WAIT_PID:?set WAIT_PID to the arithmetic worker PID}
ROOT=${ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_collision}
PY=${PY:-/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python}
OUT_ROOT=${OUT_ROOT:-outputs/semantic_memory/wikibigedit_collision_scaling_50000}

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 20
done

cd "$ROOT"
export PYTHONPATH=.
bash scripts/run_oov_slice_existing_results.sh
"$PY" scripts/analyze_collision_scaling.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/summary_live.json" \
  --bootstrap-replicates 5000
"$PY" scripts/compare_wikibigedit_oov_dose.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/oov_dose_comparison.json" \
  --bootstrap-replicates 5000
echo "ARITHMETIC_ANALYSIS_COMPLETE"
