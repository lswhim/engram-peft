#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
MODE="$2"
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram_multilingual
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
BASE=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
EMBEDDER=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-Embedding-4B
SOURCE=rq_tables/wikibigedit50k_qwen3emb4b_M8K16
MANIFEST=data/semantic_memory/wikibigedit_chronological.jsonl

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT/src:$ROOT"
export https_proxy=http://star-proxy.oa.com:3128
export http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,.tencentcos.cn,localhost,127.0.0.1
export HF_HUB_DISABLE_XET=1
mkdir -p run_logs "outputs/semantic_memory/wikibigedit_4b_k16/$MODE"

if [[ "$MODE" == semantic ]]; then
  if [[ ! -f "$SOURCE/meta.json" ]]; then
    "$PY" -u scripts/build_rq_table.py \
      --data_files "$MANIFEST" --text_columns prompt target --num_docs 50000 \
      --base_tokenizer "$BASE" --embedder "$EMBEDDER" \
      --num_levels 8 --codebook_size 16 --projection_dim 0 \
      --max_ngrams_per_size 300000 --min_count 1 --output_dir "$SOURCE"
  fi
  TABLE="$SOURCE"
  RUN=wikibigedit50k_qwen3emb4b_k16_semantic_seed42
  METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_cache_seed42,n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
elif [[ "$MODE" == shuffled ]]; then
  while [[ ! -f "$SOURCE/meta.json" ]]; do sleep 20; done
  TABLE="${SOURCE}_runtime_shuffled_seed42"
  if [[ ! -f "$TABLE/meta.json" ]]; then
    "$PY" scripts/make_runtime_shuffled_rq.py --source "$SOURCE" --output "$TABLE" --seed 42
  fi
  RUN=wikibigedit50k_qwen3emb4b_k16_shuffled_seed42
  METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_cache_seed42,n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
elif [[ "$MODE" == arithmetic ]]; then
  RUN=wikibigedit50k_k16_arithmetic_fixed_seed42
  METHOD="engram:hash_backend=arithmetic_fixed,engram_vocab_size_per_ngram=[128,128],n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
else
  echo "unknown mode: $MODE" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
"$PY" -u examples/compare_engram_lora.py \
  --model_name "$BASE" --dataset semantic_manifest \
  --manifest_path "$MANIFEST" --subset 50000 --chronological \
  --milestone_examples 1000 5000 10000 50000 \
  --max_steps 2500 --batch_size 4 --grad_accum 5 --max_length 128 \
  --num_workers 4 --disable_early_stopping --seed 42 \
  --methods "$METHOD" --run_suffix "$RUN" --skip_plot --skip_inference

MILESTONE_ROOT="$(find outputs/benchmarks/milestones -mindepth 1 -maxdepth 1 -type d -name "*${RUN}" | head -n 1)"
if [[ -z "$MILESTONE_ROOT" ]]; then
  echo "milestone directory not found for $RUN" >&2
  exit 3
fi

for POINT in 1000 5000 10000 50000; do
  "$PY" -u examples/evaluate_semantic_memory.py \
    --model "$BASE" --manifest "data/semantic_memory/wikibigedit_eval_${POINT}.jsonl" \
    --engram-weights "$MILESTONE_ROOT/writes_${POINT}" \
    --output "outputs/semantic_memory/wikibigedit_4b_k16/${MODE}/at_${POINT}.json" \
    --batch-size 16 --embed-batch-size 64
done
