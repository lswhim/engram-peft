#!/usr/bin/env bash
set -euo pipefail

cd /anguszhang-cfs-nj/seokliu_workspace/engram
mkdir -p run_logs

export HF_HUB_DISABLE_XET=1
export https_proxy=http://star-proxy.oa.com:3128
export http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,.tencentcos.cn,localhost,127.0.0.1
export WANDB_INIT_TIMEOUT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
COMMON_TRAIN=(--max_steps 1000 --subset 3000 --batch_size 8 --grad_accum 4 --methods)
METHOD_PREFIX="engram:hash_backend=mixed_v2,n_rq_levels_used=4,n_arith_heads_per_ngram=4,n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[5,7,13,17],rq_table_dir="
COMMON_EVAL=(--limit 0)
read -r -a GPU_IDS <<<"${RUN_GPU_IDS:-0 1 2 3}"

models=(
  "Qwen/Qwen3-0.6B|06b"
  "Qwen/Qwen3-1.7B|17b"
  "Qwen/Qwen3-4B|4b"
  "Qwen/Qwen3-8B|8b"
)

datasets=(
  "counterfact|cfF|counterfact_qwen3_06b"
  "zsre|zsreF|zsre_qwen3_06b"
  "mquake|mquakeF|mquake_qwen3_06b"
  "wiki_cf|wcfF|wiki_cf_qwen3_06b"
  "wiki_recent|wrF|wiki_recent_qwen3_06b"
)

wait_for_wiki_lora() {
  while pgrep -af "compare_engram_lora.py.*--dataset wiki_.*--methods lora" >/dev/null; do
    echo "[mixed_v2] waiting for Wiki LoRA jobs to finish..."
    sleep 60
  done
}

run_wave() {
  local -n cmds=$1
  local pids=()
  local num_gpus=${#GPU_IDS[@]}
  if (( num_gpus == 0 )); then
    echo "[mixed_v2] RUN_GPU_IDS is empty" >&2
    return 1
  fi
  for i in "${!cmds[@]}"; do
    local gpu=${GPU_IDS[$((i % num_gpus))]}
    echo "[mixed_v2] GPU ${gpu}: ${cmds[$i]}"
    CUDA_VISIBLE_DEVICES=$gpu bash -lc "${cmds[$i]}" &
    pids+=("$!")
    if (( ${#pids[@]} == num_gpus )); then
      wait "${pids[@]}"
      pids=()
    fi
  done
  if (( ${#pids[@]} > 0 )); then
    wait "${pids[@]}"
  fi
}

wait_for_wiki_lora

lora_eval_cmds=()
for ds_entry in "wiki_cf|wcfF|unused" "wiki_recent|wrF|unused"; do
  IFS='|' read -r dataset suffix _rq_name <<<"$ds_entry"
  for model_entry in "${models[@]}"; do
    IFS='|' read -r model_name model_short <<<"$model_entry"
    model_base=${model_name##*/}
    ckpt="outputs/benchmarks/ckpt_${model_base}_lora_r16_seed2024_${suffix}"
    log="run_logs/eval_ke_${model_short}_${dataset}_lora_${suffix}.log"
    lora_eval_cmds+=("$PY -u examples/eval_ke.py --dataset $dataset --model_name $model_name --lora_weights $ckpt ${COMMON_EVAL[*]} > $log 2>&1")
  done
done

run_wave lora_eval_cmds

train_cmds=()
for ds_entry in "${datasets[@]}"; do
  IFS='|' read -r dataset suffix rq_name <<<"$ds_entry"
  for model_entry in "${models[@]}"; do
    IFS='|' read -r model_name model_short <<<"$model_entry"
    rq_dir="rq_tables/${rq_name}"
    method="${METHOD_PREFIX}${rq_dir}"
    log="run_logs/train_ke_${model_short}_${dataset}_mixed_v2_${suffix}.log"
    train_cmds+=("$PY -u examples/compare_engram_lora.py --dataset $dataset --model_name $model_name ${COMMON_TRAIN[*]} '$method' --seed 2024 --wandb --wandb_project engram-rq-matrix --run_suffix _${suffix}_mixv2 > $log 2>&1")
  done
done

run_wave train_cmds

eval_cmds=()
for ds_entry in "${datasets[@]}"; do
  IFS='|' read -r dataset suffix _rq_name <<<"$ds_entry"
  for model_entry in "${models[@]}"; do
    IFS='|' read -r model_name model_short <<<"$model_entry"
    model_base=${model_name##*/}
    ckpt="outputs/benchmarks/ckpt_${model_base}_mixed_v2_h8_seed2024_${suffix}_mixv2"
    log="run_logs/eval_ke_${model_short}_${dataset}_mixed_v2_${suffix}.log"
    eval_cmds+=("$PY -u examples/eval_ke.py --dataset $dataset --model_name $model_name --engram_weights $ckpt ${COMMON_EVAL[*]} > $log 2>&1")
  done
done

run_wave eval_cmds
echo "MIXED_V2_MAIN_DONE"
