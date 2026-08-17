#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
MODE="$2"
EMBED_SIZE="${3:-0.6b}"
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram_multilingual
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
BASE=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base

case "$EMBED_SIZE" in
  0.6b)
    EMBEDDER=Qwen/Qwen3-Embedding-0.6B
    EMBED_TAG=qwen3emb06b
    SOURCE=rq_tables/fineweb_M8K16_300k_strict
    ;;
  4b)
    EMBEDDER=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-Embedding-4B
    EMBED_TAG=qwen3emb4b
    SOURCE=rq_tables/fineweb_qwen3emb4b_M8K16_300k_strict
    ;;
  8b)
    EMBEDDER=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-Embedding-8B
    EMBED_TAG=qwen3emb8b
    SOURCE=rq_tables/fineweb_qwen3emb8b_M8K16_300k_strict
    ;;
  *)
    echo "unknown embedder size: $EMBED_SIZE" >&2
    exit 2
    ;;
esac

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT/src:$ROOT"
export https_proxy=http://star-proxy.oa.com:3128
export http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,.tencentcos.cn,localhost,127.0.0.1
export HF_HUB_DISABLE_XET=1
OUTDIR="outputs/semantic_memory/pararel_fineweb_k16/${EMBED_TAG}"
mkdir -p run_logs "$OUTDIR"

if [[ "$MODE" == semantic ]]; then
  "$PY" -u scripts/build_rq_table.py \
    --dataset HuggingFaceFW/fineweb-edu --dataset_config sample-10BT \
    --split train --text_column text --num_docs 12000 \
    --base_tokenizer "$BASE" --embedder "$EMBEDDER" \
    --num_levels 8 --codebook_size 16 --projection_dim 0 \
    --max_ngrams_per_size 300000 --min_count 2 --output_dir "$SOURCE"
  TABLE="$SOURCE"
  RUN=pararel5k_fineweb_${EMBED_TAG}_k16_semantic_seed42
  RESULT="$OUTDIR/semantic_seed42.json"
elif [[ "$MODE" == shuffled ]]; then
  while [[ ! -f "$SOURCE/meta.json" ]]; do sleep 15; done
  TABLE="${SOURCE}_runtime_shuffled_seed42"
  if [[ ! -f "$TABLE/meta.json" ]]; then
    "$PY" scripts/make_runtime_shuffled_rq.py --source "$SOURCE" --output "$TABLE" --seed 42
  fi
  RUN=pararel5k_fineweb_${EMBED_TAG}_k16_shuffled_seed42
  RESULT="$OUTDIR/shuffled_seed42.json"
else
  echo "unknown mode: $MODE" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/runtime_cache_seed42,n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
"$PY" -u examples/compare_engram_lora.py \
  --model_name "$BASE" --dataset semantic_manifest \
  --manifest_path data/semantic_memory/pararel.jsonl --subset 5000 \
  --max_steps 782 --batch_size 8 --grad_accum 4 --max_length 128 \
  --num_workers 0 --disable_early_stopping --seed 42 \
  --methods "$METHOD" --run_suffix "$RUN" --skip_plot --skip_inference

CKPT="outputs/benchmarks/ckpt_Qwen3-1.7B-Base_rq_h8_seed42_$(basename "$TABLE")${RUN}"
"$PY" -u examples/evaluate_semantic_memory.py \
  --model "$BASE" --manifest data/semantic_memory/pararel5k_eval4.jsonl \
  --engram-weights "$CKPT" --output "$RESULT" \
  --geometry-cache outputs/semantic_memory/pararel/geometry_cache.json \
  --batch-size 16 --embed-batch-size 64

"$PY" -u examples/render_semantic_memory_results.py
