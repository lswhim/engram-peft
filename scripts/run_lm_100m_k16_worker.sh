#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?gpu}
MODE=${2:?mode: arithmetic|semantic_flatten|semantic_keyed|shuffled_keyed}
SEED=${3:-42}
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
MODEL=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
SEM_TABLE=rq_tables/fineweb_qwen3emb06b_M8K16_300k_strict
SHUF_TABLE=${SEM_TABLE}_runtime_shuffled_seed42
OUT=outputs/standard_lm_k16_emb06b

MAX_STEPS=763
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_DISABLE_XET=1
export https_proxy=http://star-proxy.oa.com:3128 http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,localhost,127.0.0.1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TORCH_NUM_THREADS=8
mkdir -p run_logs "$OUT"

COMMON="n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
case "$MODE" in
  arithmetic)
    SPEC="engram:hash_backend=arithmetic_fixed,engram_vocab_size_per_ngram=[128,128],${COMMON},memory_fusion=flatten"
    ;;
  semantic_flatten)
    SPEC="engram:hash_backend=rq,rq_table_dir=${SEM_TABLE},rq_cache_dir=${SEM_TABLE}/lm100m_k16_emb06b_${MODE}_seed${SEED},${COMMON},memory_fusion=flatten"
    ;;
  semantic_keyed)
    SPEC="engram:hash_backend=rq,rq_table_dir=${SEM_TABLE},rq_cache_dir=${SEM_TABLE}/lm100m_k16_emb06b_${MODE}_seed${SEED},${COMMON},memory_fusion=head_factorized,head_router_selection=semantic_keyed"
    ;;
  shuffled_keyed)
    SPEC="engram:hash_backend=rq,rq_table_dir=${SHUF_TABLE},rq_cache_dir=${SHUF_TABLE}/lm100m_k16_emb06b_${MODE}_seed${SEED},${COMMON},memory_fusion=head_factorized,head_router_selection=semantic_keyed"
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

if [[ "$MODE" == shuffled_keyed ]]; then
  test -f "$SHUF_TABLE/meta.json"
fi

SUFFIX="_lm100m_k16_emb06b_${MODE}_seed${SEED}"
LOG="run_logs/lm100m_k16_emb06b_${MODE}_seed${SEED}.log"
"$PY" -u examples/compare_engram_lora.py \
  --dataset fineweb --model_name "$MODEL" --subset 100000 --max_steps "$MAX_STEPS" \
  --batch_size 4 --grad_accum 32 --max_length 1024 --num_workers 4 --seed "$SEED" \
  --disable_early_stopping --skip_plot --skip_inference \
  --run_suffix "$SUFFIX" --methods "${SPEC},save_steps=${MAX_STEPS},eval_steps=${MAX_STEPS}" \
  > "$LOG" 2>&1

CKPT="$(find outputs/benchmarks -mindepth 1 -maxdepth 1 -type d -name "*${SUFFIX}" | head -n 1)"
test -n "$CKPT" && test -s "$CKPT/engram_adapters.safetensors"
"$PY" -u examples/evaluate_standard_lm.py \
  --model "$MODEL" --tasks wikitext,lambada_openai --engram-weights "$CKPT" \
  --method "k16_${MODE}" --seed "$SEED" --output "$OUT/100m_${MODE}_seed${SEED}.json" \
  >> "$LOG" 2>&1
echo "LM100M_K16_EMB06B_WORKER_DONE mode=$MODE seed=$SEED gpu=$GPU"
