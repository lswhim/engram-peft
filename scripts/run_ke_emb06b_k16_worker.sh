#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?gpu}
MODEL_SLUG=${2:?model slug}
MODEL=${3:?model path}
MODE=${4:?mode: semantic_keyed|semantic_flatten|arithmetic}
SEED=${5:-42}
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
SEM_TABLE=rq_tables/fineweb_qwen3emb06b_M8K16_300k_strict
OUT_ROOT=outputs/base_model_sweep_emb06b/ke

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_DISABLE_XET=1
export https_proxy=http://star-proxy.oa.com:3128 http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,localhost,127.0.0.1
mkdir -p run_logs "$OUT_ROOT/$MODEL_SLUG"

COMMON="n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
case "$MODE" in
  semantic_keyed)
    SPEC="engram:hash_backend=rq,rq_table_dir=${SEM_TABLE},rq_cache_dir=${SEM_TABLE}/ke_emb06b_k16_${MODEL_SLUG}_${MODE}_cache_seed${SEED},${COMMON},memory_fusion=head_factorized,head_router_selection=semantic_keyed"
    ;;
  semantic_flatten)
    SPEC="engram:hash_backend=rq,rq_table_dir=${SEM_TABLE},rq_cache_dir=${SEM_TABLE}/ke_emb06b_k16_${MODEL_SLUG}_${MODE}_cache_seed${SEED},${COMMON},memory_fusion=flatten"
    ;;
  arithmetic)
    # M=8, K=16 RQ has 128 rows per n-gram order in total.
    SPEC="engram:hash_backend=arithmetic_fixed,engram_vocab_size_per_ngram=[128,128],${COMMON},memory_fusion=flatten"
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

if [[ "$MODEL_SLUG" == qwen3_8b_base ]]; then
  BATCH=2
  GRAD_ACCUM=16
else
  BATCH=4
  GRAD_ACCUM=8
fi

run_one() {
  local dataset=$1 steps=$2
  local short=cf
  [[ "$dataset" == zsre ]] && short=zsre
  local suffix="_ke_emb06b_k16_${MODEL_SLUG}_${short}_${MODE}_seed${SEED}"
  local log="run_logs/ke_emb06b_k16_${MODEL_SLUG}_${short}_${MODE}_seed${SEED}.log"

  "$PY" -u examples/compare_engram_lora.py \
    --dataset "${dataset}_canonical" --model_name "$MODEL" --max_steps "$steps" \
    --subset 999999 --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" --max_length 64 --num_workers 0 \
    --seed "$SEED" --disable_early_stopping --skip_plot --skip_inference \
    --run_suffix "$suffix" --methods "${SPEC},save_steps=${steps},eval_steps=${steps}" \
    > "$log" 2>&1

  local ckpt_dir
  ckpt_dir="$(find outputs/benchmarks -mindepth 1 -maxdepth 1 -type d -name "*${suffix}" | head -n 1)"
  test -n "$ckpt_dir" && test -s "$ckpt_dir/engram_adapters.safetensors"

  "$PY" -u examples/evaluate_standard_ke.py \
    --dataset "$dataset" --model "$MODEL" --engram-weights "$ckpt_dir" \
    --output "$OUT_ROOT/$MODEL_SLUG/${dataset}_${MODE}_seed${SEED}.json" \
    --limit 0 --batch-size 24 --case-chunk-size 32 \
    >> "$log" 2>&1
}

run_one counterfact 345
run_one zsre 205
echo "KE_EMB06B_K16_WORKER_DONE model=$MODEL_SLUG mode=$MODE seed=$SEED gpu=$GPU"
