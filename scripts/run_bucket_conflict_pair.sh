#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram}
PY=${PY:-/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python}
MODEL=${MODEL:-/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base}
MANIFEST=${MANIFEST:-data/semantic_memory/wikibigedit_chronological.jsonl}
OUT=${OUT:-outputs/semantic_memory/wikibigedit_collision_scaling_50000}

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for spec in \
  "semantic:rq_tables/wikibigedit50k_qwen3emb4b_M8K16" \
  "loadmatched:rq_tables/wikibigedit50k_qwen3emb4b_M8K16_loadmatched_seed42"; do
  name=${spec%%:*}
  table=${spec#*:}
  "$PY" -u scripts/analyze_bucket_target_conflict.py \
    --manifest "$MANIFEST" \
    --table-dir "$table" \
    --base-tokenizer "$MODEL" \
    --output "$OUT/${name}_bucket_target_conflict.json" \
    --prefix-cases 40000 --max-cases 50000 --max-length 128 \
    --top-k 4 --bootstrap-replicates 5000
done

"$PY" scripts/compare_bucket_target_conflict.py \
  --semantic "$OUT/semantic_bucket_target_conflict.jsonl" \
  --control "$OUT/loadmatched_bucket_target_conflict.jsonl" \
  --output "$OUT/bucket_target_conflict_interaction.json" \
  --bootstrap-replicates 5000
