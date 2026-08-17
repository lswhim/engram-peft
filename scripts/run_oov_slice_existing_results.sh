#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_collision}
PY=${PY:-/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python}
MODEL=${MODEL:-/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base}
MANIFEST=${MANIFEST:-data/semantic_memory/wikibigedit_eval_50000.jsonl}
TABLE=${TABLE:-rq_tables/wikibigedit50k_qwen3emb4b_M8K16}
RESULT_ROOT=${RESULT_ROOT:-outputs/semantic_memory/wikibigedit_collision_scaling_50000}

cd "$ROOT"
export PYTHONPATH=.
for spec in semantic_flatten:42 semantic_flatten:123 semantic_flatten:456 arithmetic:42 arithmetic:123 arithmetic:456; do
  method=${spec%%:*}
  seed=${spec##*:}
  input="$RESULT_ROOT/$method/seed_$seed/at_50000.json"
  output="$RESULT_ROOT/$method/seed_$seed/at_50000_oov_slices.json"
  if [[ ! -f "$input" ]]; then
    echo "WAITING $method seed=$seed"
    continue
  fi
  if [[ -f "$output" ]]; then
    echo "EXISTS $method seed=$seed"
    continue
  fi
  "$PY" scripts/analyze_wikibigedit_oov_slices.py \
    --result "$input" \
    --manifest "$MANIFEST" \
    --table-dir "$TABLE" \
    --model "$MODEL" \
    --output "$output"
  echo "DONE $method seed=$seed"
done
