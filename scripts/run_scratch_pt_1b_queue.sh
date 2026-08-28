#!/usr/bin/env bash
set -euo pipefail

ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
TORCHRUN=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/torchrun
TOKENIZER=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
DATA=$ROOT/outputs/scratch_pt_1b/data_fast
OUT=$ROOT/outputs/scratch_pt_1b/runs

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TORCH_NUM_THREADS=2
export NCCL_TIMEOUT=1800

mkdir -p "$OUT"
while [[ ! -s "$DATA/metadata.json" ]]; do
  sleep 20
done

for mode in base arithmetic; do
  mode_out="$OUT/$mode"
  log="$OUT/${mode}.log"
  if [[ -s "$mode_out/metrics.json" ]]; then
    echo "[scratch queue] already complete: $mode"
    continue
  fi
  mkdir -p "$mode_out"
  echo "[scratch queue] starting mode=$mode" | tee -a "$log"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$TORCHRUN" \
    --standalone --nproc_per_node=8 \
    examples/run_scratch_pt.py \
    --mode "$mode" \
    --data-dir "$DATA" \
    --tokenizer "$TOKENIZER" \
    --output-dir "$mode_out" \
    --train-tokens 1000000000 \
    --sequence-length 2048 \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 64 \
    --learning-rate 1e-4 \
    --eval-steps 95 \
    --checkpoint-steps 239 \
    --num-workers 2 \
    --fp32 \
    >> "$log" 2>&1
  echo "[scratch queue] finished mode=$mode" | tee -a "$log"
done

echo "SCRATCH_PT_1B_QUEUE_DONE" | tee -a "$OUT/queue.log"
