#!/usr/bin/env bash
set -uo pipefail

ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram
cd "$ROOT"
mkdir -p run_logs

MODEL17=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
MODEL4=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-4B-Base
MODEL8=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-8B-Base

JOBS=(
  "bash scripts/run_ke_emb06b_k16_worker.sh 0 qwen3_1p7b_base $MODEL17 semantic_keyed 42"
  "bash scripts/run_ke_emb06b_k16_worker.sh 1 qwen3_1p7b_base $MODEL17 semantic_flatten 42"
  "bash scripts/run_ke_emb06b_k16_worker.sh 2 qwen3_1p7b_base $MODEL17 arithmetic 42"
  "bash scripts/run_ke_emb06b_k16_worker.sh 3 qwen3_4b_base $MODEL4 semantic_keyed 42"
  "bash scripts/run_ke_emb06b_k16_worker.sh 4 qwen3_4b_base $MODEL4 semantic_flatten 42"
  "bash scripts/run_ke_emb06b_k16_worker.sh 5 qwen3_4b_base $MODEL4 arithmetic 42"
  "bash scripts/run_ke_emb06b_k16_worker.sh 6 qwen3_8b_base $MODEL8 semantic_keyed 42"
  "bash scripts/run_ke_emb06b_k16_worker.sh 7 qwen3_8b_base $MODEL8 semantic_flatten 42"
  "bash scripts/run_ke_emb06b_k16_worker.sh 0 qwen3_8b_base $MODEL8 arithmetic 42"
  "bash scripts/run_lm_100m_k16_worker.sh 1 arithmetic 42"
  "bash scripts/run_lm_100m_k16_worker.sh 2 semantic_flatten 42"
  "bash scripts/run_lm_100m_k16_worker.sh 3 semantic_keyed 42"
  "bash scripts/run_lm_100m_k16_worker.sh 4 shuffled_keyed 42"
  "bash scripts/run_xnli_k16_emb06b_worker.sh 5 arithmetic"
  "bash scripts/run_xnli_k16_emb06b_worker.sh 6 semantic_flatten"
  "bash scripts/run_xnli_k16_emb06b_worker.sh 7 semantic_keyed"
  "bash scripts/run_pawsx_k16_emb06b_worker.sh 0 arithmetic"
  "bash scripts/run_pawsx_k16_emb06b_worker.sh 1 semantic_flatten"
  "bash scripts/run_pawsx_k16_emb06b_worker.sh 2 semantic_keyed"
)

GPUS=(0 1 2 3 4 5 6 7)
declare -A PID_GPU=() PID_JOB=()
next=0
total=${#JOBS[@]}
echo "K16_QUEUE_START total=$total time=$(date -Is)"

while (( next < total || ${#PID_GPU[@]} > 0 )); do
  for gpu in "${GPUS[@]}"; do
    occupied=0
    for pid in "${!PID_GPU[@]}"; do
      [[ "${PID_GPU[$pid]}" == "$gpu" ]] && occupied=1
    done
    if (( occupied == 0 && next < total )); then
      log="run_logs/k16_queue_job${next}_gpu${gpu}.log"
      echo "K16_QUEUE_LAUNCH job=$next gpu=$gpu cmd=${JOBS[$next]} time=$(date -Is)"
      nohup bash -c "${JOBS[$next]}" > "$log" 2>&1 &
      pid=$!
      PID_GPU[$pid]=$gpu
      PID_JOB[$pid]=$next
      ((next += 1))
    fi
  done

  for pid in "${!PID_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rc=0
      wait "$pid" || rc=$?
      echo "K16_QUEUE_DONE job=${PID_JOB[$pid]} gpu=${PID_GPU[$pid]} rc=$rc time=$(date -Is)"
      unset 'PID_GPU[$pid]' 'PID_JOB[$pid]'
    fi
  done
  (( ${#PID_GPU[@]} > 0 )) && sleep 20
done
echo "K16_QUEUE_FINISH time=$(date -Is)"
