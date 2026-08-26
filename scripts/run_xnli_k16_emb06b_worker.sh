#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?gpu}
MODE=${2:?mode: arithmetic|semantic_flatten|semantic_keyed}
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
MODEL=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
TABLE=rq_tables/fineweb_qwen3emb06b_M8K16_300k_strict
OUT=outputs/xnli_k16_emb06b

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_DISABLE_XET=1
export https_proxy=http://star-proxy.oa.com:3128 http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,localhost,127.0.0.1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TORCH_NUM_THREADS=8
mkdir -p run_logs "$OUT"

case "$MODE" in
  semantic_keyed) METHOD=rq; ROUTER=semantic_keyed ;;
  semantic_flatten) METHOD=rq; ROUTER=flatten ;;
  arithmetic) METHOD=arithmetic_matched; ROUTER=flatten ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

LOG="run_logs/xnli_k16_emb06b_${MODE}.log"
"$PY" -u examples/run_xtreme_xnli.py \
  --method "$METHOD" --model "$MODEL" --rq_table_dir "$TABLE" \
  --rq_cache_dir "$TABLE/xnli_k16_emb06b_cache_${MODE}_seed42" --rq_router "$ROUTER" \
  --output_dir "$OUT/$MODE" --epochs 1 --batch_size 4 --grad_accum 8 --eval_batch_size 8 \
  --max_length 256 --num_workers 4 --seed 42 \
  > "$LOG" 2>&1
echo "XNLI_K16_EMB06B_WORKER_DONE mode=$MODE gpu=$GPU"
