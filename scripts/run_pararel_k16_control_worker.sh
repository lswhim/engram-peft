#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
MODE="$2"
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram_multilingual
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
BASE=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT/src:$ROOT"
mkdir -p run_logs outputs/semantic_memory/pararel_k16_controls

if [[ "$MODE" == shuffled ]]; then
  SOURCE=rq_tables/pararel5k_M8K16
  TABLE=rq_tables/pararel5k_M8K16_runtime_shuffled_seed42
  if [[ ! -f "$TABLE/meta.json" ]]; then
    "$PY" scripts/make_runtime_shuffled_rq.py --source "$SOURCE" --output "$TABLE" --seed 42
  fi
  RUN=pararel5k_k16_shuffled_seed42
  METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/runtime_cache_seed42,n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
  CKPT="outputs/benchmarks/ckpt_Qwen3-1.7B-Base_rq_h8_seed42_pararel5k_M8K16_runtime_shuffled_seed42${RUN}"
  RESULT=outputs/semantic_memory/pararel_k16_controls/shuffled_seed42.json
elif [[ "$MODE" == arithmetic ]]; then
  RUN=pararel5k_k16_arithmetic_fixed_seed42
  METHOD="engram:hash_backend=arithmetic_fixed,engram_vocab_size_per_ngram=[128,128],n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
  CKPT="outputs/benchmarks/ckpt_Qwen3-1.7B-Base_arithmetic_fixed_h8_seed42${RUN}"
  RESULT=outputs/semantic_memory/pararel_k16_controls/arithmetic_fixed_seed42.json
else
  echo "unknown mode: $MODE" >&2
  exit 2
fi

"$PY" -u examples/compare_engram_lora.py \
  --model_name "$BASE" --dataset semantic_manifest \
  --manifest_path data/semantic_memory/pararel.jsonl --subset 5000 \
  --max_steps 782 --batch_size 8 --grad_accum 4 --max_length 128 \
  --num_workers 0 --disable_early_stopping --seed 42 \
  --methods "$METHOD" --run_suffix "$RUN" --skip_plot --skip_inference

"$PY" -u examples/evaluate_semantic_memory.py \
  --model "$BASE" --manifest data/semantic_memory/pararel5k_eval4.jsonl \
  --engram-weights "$CKPT" --output "$RESULT" \
  --geometry-cache outputs/semantic_memory/pararel/geometry_cache.json \
  --batch-size 16 --embed-batch-size 64

"$PY" -u examples/render_semantic_memory_results.py
