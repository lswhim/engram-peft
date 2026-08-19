#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?gpu}
MODE=${2:?mode: semantic_keyed|semantic_flatten|arithmetic}
SEED=${3:-42}
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
MODEL=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
SEM_TABLE=rq_tables/fineweb_qwen3emb06b_M8K256_300k_strict

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_DISABLE_XET=1
export https_proxy=http://star-proxy.oa.com:3128 http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,localhost,127.0.0.1
mkdir -p run_logs outputs/standard_lm

COMMON="n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
case "$MODE" in
  semantic_keyed)
    SPEC="engram:hash_backend=rq,rq_table_dir=${SEM_TABLE},rq_cache_dir=${SEM_TABLE}/lm_cache_${MODE}_seed${SEED},${COMMON},memory_fusion=head_factorized,head_router_selection=semantic_keyed"
    ;;
  semantic_flatten)
    SPEC="engram:hash_backend=rq,rq_table_dir=${SEM_TABLE},rq_cache_dir=${SEM_TABLE}/lm_cache_${MODE}_seed${SEED},${COMMON},memory_fusion=flatten"
    ;;
  arithmetic)
    # Match RQ per-head capacity: K=256 per head x 8 heads = 2048 total per n-gram order.
    SPEC="engram:hash_backend=arithmetic_fixed,engram_vocab_size_per_ngram=[2048,2048],${COMMON},memory_fusion=flatten"
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

SUFFIX="_lm_semantic_keyed_${MODE}_seed${SEED}"
LOG="run_logs/lm_${MODE}_seed${SEED}.log"

"$PY" -u examples/compare_engram_lora.py \
  --dataset fineweb --model_name "$MODEL" \
  --subset 10000 --max_steps 400 --batch_size 4 --grad_accum 8 \
  --max_length 128 --num_workers 0 --seed "$SEED" \
  --disable_early_stopping --skip_plot --skip_inference \
  --run_suffix "$SUFFIX" --methods "${SPEC},save_steps=400,eval_steps=400" \
  > "$LOG" 2>&1

CKPT="$(find outputs/benchmarks -mindepth 1 -maxdepth 1 -type d -name "*${SUFFIX}" | head -n 1)"
test -n "$CKPT" && test -s "$CKPT/engram_adapters.safetensors"

"$PY" -u examples/evaluate_standard_lm.py \
  --model "$MODEL" --tasks wikitext,lambada_openai \
  --engram-weights "$CKPT" --method "$MODE" --seed "$SEED" \
  --output "outputs/standard_lm/${MODE}_seed${SEED}.json" \
  >> "$LOG" 2>&1

echo "LM_WORKER_DONE mode=$MODE seed=$SEED"
