#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?gpu}
MODE=${2:?mode: semantic_keyed|semantic_flatten|arithmetic}
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
mkdir -p run_logs outputs/xnli_semantic_keyed

case "$MODE" in
  semantic_keyed)
    METHOD=rq
    ROUTER=semantic_keyed
    TABLE="$SEM_TABLE"
    ;;
  semantic_flatten)
    METHOD=rq
    ROUTER=flatten
    TABLE="$SEM_TABLE"
    ;;
  arithmetic)
    METHOD=arithmetic_matched
    ROUTER=flatten
    TABLE="$SEM_TABLE"
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

LOG="run_logs/xnli_${MODE}.log"
"$PY" -u examples/run_xtreme_xnli.py \
  --method "$METHOD" --model "$MODEL" \
  --rq_table_dir "$TABLE" --rq_cache_dir "$TABLE/xnli_cache_${MODE}_seed42" \
  --rq_router "$ROUTER" \
  --output_dir "outputs/xnli_semantic_keyed/${MODE}" \
  --epochs 1 --batch_size 4 --grad_accum 8 --eval_batch_size 8 \
  --max_length 256 --num_workers 4 --seed 42 \
  > "$LOG" 2>&1

echo "XNLI_WORKER_DONE mode=$MODE"
