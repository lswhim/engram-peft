#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
CODEBOOK_SIZE="$2"
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram_multilingual
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
BASE=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
TABLE="rq_tables/pararel5k_M8K${CODEBOOK_SIZE}"
RUN="pararel5k_indomain_rq_k${CODEBOOK_SIZE}_seed42"
CKPT="outputs/benchmarks/ckpt_Qwen3-1.7B-Base_rq_h8_seed42_pararel5k_M8K${CODEBOOK_SIZE}${RUN}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT/src:$ROOT"
export https_proxy=http://star-proxy.oa.com:3128
export http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,.tencentcos.cn,localhost,127.0.0.1
export HF_HUB_DISABLE_XET=1

mkdir -p run_logs "$TABLE" outputs/semantic_memory/pararel_k_sweep

"$PY" -u scripts/build_rq_table.py \
  --data_files data/semantic_memory/pararel.jsonl --text_columns prompt target \
  --num_docs 5000 --base_tokenizer "$BASE" \
  --embedder Qwen/Qwen3-Embedding-0.6B --num_levels 8 \
  --codebook_size "$CODEBOOK_SIZE" --projection_dim 0 \
  --max_ngrams_per_size 300000 --min_count 1 --output_dir "$TABLE"

export HF_HUB_OFFLINE=1
"$PY" -u examples/compare_engram_lora.py \
  --model_name "$BASE" --dataset semantic_manifest \
  --manifest_path data/semantic_memory/pararel.jsonl --subset 5000 \
  --max_steps 782 --batch_size 8 --grad_accum 4 --max_length 128 \
  --num_workers 0 --disable_early_stopping --seed 42 \
  --methods "engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/runtime_cache_seed42,n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]" \
  --run_suffix "$RUN" --skip_plot --skip_inference

"$PY" -u examples/evaluate_semantic_memory.py \
  --model "$BASE" --manifest data/semantic_memory/pararel5k_eval4.jsonl \
  --engram-weights "$CKPT" \
  --output "outputs/semantic_memory/pararel_k_sweep/k${CODEBOOK_SIZE}_seed42.json" \
  --geometry-cache outputs/semantic_memory/pararel/geometry_cache.json \
  --batch-size 16 --embed-batch-size 64

"$PY" -u examples/render_semantic_memory_results.py
