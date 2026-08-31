#!/usr/bin/env bash
set -euo pipefail

ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
TORCHRUN=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/torchrun
MODEL=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
DATA=/tmp/engram_scratch_pt_1b_data_fast
OUT=$ROOT/outputs/qwen3_base_cpt_1b/runs
TABLE=$ROOT/rq_tables/fineweb_qwen3emb06b_M8K16_300k_strict
RQ_CACHE=/tmp/engram_scratch_pt_1b_rq_runtime_cache_k16

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TORCH_NUM_THREADS=2
export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT"
test -s "$MODEL/config.json"
test -s "$DATA/train.bin"
test -s "$DATA/eval.bin"
test -s "$RQ_CACHE/preencode_complete.json"

for mode in base arithmetic semantic_flatten semantic_keyed; do
  mode_out="$OUT/$mode"
  log="$OUT/${mode}.log"
  if [[ -s "$mode_out/metrics.json" ]]; then
    echo "[qwen3 cpt queue] already complete: $mode" | tee -a "$OUT/queue.log"
    continue
  fi
  mkdir -p "$mode_out"
  extra_args=()
  if [[ "$mode" == semantic_flatten || "$mode" == semantic_keyed ]]; then
    for rank in 0 1 2 3 4 5 6 7; do
      mkdir -p "$RQ_CACHE/rank$rank"
      ln -sfn "$RQ_CACHE/semantic_codes.sqlite3" "$RQ_CACHE/rank$rank/semantic_codes.sqlite3"
    done
    extra_args+=(--rq-table-dir "$TABLE" --rq-cache-dir "$RQ_CACHE")
  fi
  echo "[qwen3 cpt queue] starting mode=$mode" | tee -a "$OUT/queue.log" "$log"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$TORCHRUN" \
    --standalone --nproc_per_node=8 \
    examples/run_qwen3_pretrained_cpt_1b.py \
    --mode "$mode" \
    --data-dir "$DATA" \
    --tokenizer "$MODEL" \
    --output-dir "$mode_out" \
    --train-tokens 1000000000 \
    --sequence-length 2048 \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 64 \
    --learning-rate 3e-5 \
    --eval-steps 95 \
    --checkpoint-steps 239 \
    --num-workers 2 \
    --attn-implementation eager \
    "${extra_args[@]}" \
    >> "$log" 2>&1
  echo "[qwen3 cpt queue] finished mode=$mode" | tee -a "$OUT/queue.log"
done

echo "QWEN3_PRETRAINED_CPT_1B_QUEUE_DONE" | tee -a "$OUT/queue.log"
