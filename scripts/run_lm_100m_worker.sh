#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?gpu}
MODE=${2:?mode: arithmetic|semantic_flatten|semantic_keyed|shuffled_keyed}
SEED=${3:-42}
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
MODEL=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
SEM_TABLE=rq_tables/fineweb_qwen3emb06b_M8K256_300k_strict
SHUF_TABLE=${SEM_TABLE}_runtime_shuffled_seed42

# 100M-token post-hoc adaptation budget.
# effective batch = 4 x 32 = 128 seq; 128 x 1024 = 131072 token slots/step.
# 763 steps x 131072 = 100,038,656 token slots ~= 100M.
MAX_STEPS=763
BATCH=4
GRAD_ACCUM=32
SEQ=1024
SUBSET=100000

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_DISABLE_XET=1
export https_proxy=http://star-proxy.oa.com:3128 http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,localhost,127.0.0.1
# Bound per-process CPU threads so concurrent workers on one shared pod do not
# oversubscribe the host CPUs and starve each other's dataloader/encoder path.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TORCH_NUM_THREADS=8
mkdir -p run_logs outputs/standard_lm

COMMON="n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
case "$MODE" in
  arithmetic)
    SPEC="engram:hash_backend=arithmetic_fixed,engram_vocab_size_per_ngram=[2048,2048],${COMMON},memory_fusion=flatten"
    ;;
  semantic_flatten)
    SPEC="engram:hash_backend=rq,rq_table_dir=${SEM_TABLE},rq_cache_dir=${SEM_TABLE}/lm100m_cache_${MODE}_seed${SEED},${COMMON},memory_fusion=flatten"
    ;;
  semantic_keyed)
    SPEC="engram:hash_backend=rq,rq_table_dir=${SEM_TABLE},rq_cache_dir=${SEM_TABLE}/lm100m_cache_${MODE}_seed${SEED},${COMMON},memory_fusion=head_factorized,head_router_selection=semantic_keyed"
    ;;
  shuffled_keyed)
    SPEC="engram:hash_backend=rq,rq_table_dir=${SHUF_TABLE},rq_cache_dir=${SHUF_TABLE}/lm100m_cache_${MODE}_seed${SEED},${COMMON},memory_fusion=head_factorized,head_router_selection=semantic_keyed"
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

SUFFIX="_lm100m_${MODE}_seed${SEED}"
LOG="run_logs/lm100m_${MODE}_seed${SEED}.log"

"$PY" -u examples/compare_engram_lora.py \
  --dataset fineweb --model_name "$MODEL" \
  --subset "$SUBSET" --max_steps "$MAX_STEPS" --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" \
  --max_length "$SEQ" --num_workers 4 --seed "$SEED" \
  --disable_early_stopping --skip_plot --skip_inference \
  --run_suffix "$SUFFIX" --methods "${SPEC},save_steps=${MAX_STEPS},eval_steps=${MAX_STEPS}" \
  > "$LOG" 2>&1

CKPT="$(find outputs/benchmarks -mindepth 1 -maxdepth 1 -type d -name "*${SUFFIX}" | head -n 1)"
test -n "$CKPT" && test -s "$CKPT/engram_adapters.safetensors"

"$PY" -u examples/evaluate_standard_lm.py \
  --model "$MODEL" --tasks wikitext,lambada_openai \
  --engram-weights "$CKPT" --method "$MODE" --seed "$SEED" \
  --output "outputs/standard_lm/100m_${MODE}_seed${SEED}.json" \
  >> "$LOG" 2>&1

echo "LM100M_WORKER_DONE mode=$MODE seed=$SEED gpu=$GPU"
