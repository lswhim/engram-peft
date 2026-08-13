#!/usr/bin/env bash
set -euo pipefail

METHOD=${1:?method: arithmetic|lora|rq}
SEED=${2:?seed}
GPU=${3:?gpu}
ROOT=${ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_multilingual}
PY=${PY:-/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python}
MODEL=${MODEL:-/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base}
RQ_TABLE=${RQ_TABLE:-rq_tables/fineweb_M8K1024_300k_strict}

cd "$ROOT"
mkdir -p logs outputs/standard_ke_v3
export PYTHONPATH="$ROOT/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU"

case "$METHOD" in
  arithmetic)
    METHOD_SPEC='engram:hash_backend=arithmetic_fixed,n_head_per_ngram=8,engram_vocab_size_per_ngram=[8192,8192],use_sparse_embeddings=False,target_layers=[11,21]'
    CKPT_PREFIX="ckpt_Qwen3-1.7B-Base_arithmetic_fixed_h8_seed${SEED}"
    EVAL_FLAG=--engram-weights
    ;;
  lora)
    METHOD_SPEC=lora
    CKPT_PREFIX="ckpt_Qwen3-1.7B-Base_lora_r16_seed${SEED}"
    EVAL_FLAG=--lora-weights
    ;;
  rq)
    METHOD_SPEC="engram:hash_backend=rq,rq_table_dir=${RQ_TABLE},rq_cache_dir=${RQ_TABLE}/runtime_cache_v3_seed${SEED},n_head_per_ngram=8,engram_vocab_size_per_ngram=[8192,8192],use_sparse_embeddings=False,target_layers=[11,21]"
    CKPT_PREFIX="ckpt_Qwen3-1.7B-Base_rq_h8_seed${SEED}_fineweb_M8K1024_300k_strict"
    EVAL_FLAG=--engram-weights
    ;;
  *) echo "unknown method: $METHOD" >&2; exit 2 ;;
esac

run_one() {
  local dataset=$1 steps=$2
  local short=cf
  [[ "$dataset" == zsre ]] && short=zsre
  local suffix="_formal_v3_${short}_5ep_k1024_seed${SEED}"
  local ckpt="outputs/benchmarks/${CKPT_PREFIX}${suffix}"
  local train_log="logs/formal_v3_${short}_${METHOD}_seed${SEED}.log"
  local eval_log="logs/formal_v3_eval_${short}_${METHOD}_seed${SEED}.log"
  local result="outputs/standard_ke_v3/${dataset}_${METHOD}_seed${SEED}.json"
  local method_with_budget
  if [[ "$METHOD_SPEC" == *:* ]]; then
    method_with_budget="${METHOD_SPEC},save_steps=${steps},eval_steps=${steps}"
  else
    method_with_budget="${METHOD_SPEC}:save_steps=${steps},eval_steps=${steps}"
  fi

  "$PY" -u examples/compare_engram_lora.py \
    --dataset "${dataset}_canonical" --model_name "$MODEL" --max_steps "$steps" \
    --subset 999999 --batch_size 4 --grad_accum 8 --max_length 64 --num_workers 0 \
    --seed "$SEED" --disable_early_stopping --skip_plot --skip_inference \
    --run_suffix "$suffix" --methods "$method_with_budget" \
    > "$train_log" 2>&1

  if [[ "$METHOD" == lora ]]; then
    test -s "$ckpt/adapter_model.safetensors"
  else
    test -s "$ckpt/engram_adapters.safetensors"
  fi
  "$PY" -u examples/evaluate_standard_ke.py \
    --dataset "$dataset" --model "$MODEL" "$EVAL_FLAG" "$ckpt" \
    --output "$result" --limit 0 --batch-size 24 --case-chunk-size 32 \
    > "$eval_log" 2>&1
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="complete" and d["complete_official_split"] is True' "$result"
}

run_one counterfact 345
run_one zsre 205
echo "FORMAL_WORKER_DONE method=$METHOD seed=$SEED gpu=$GPU"
