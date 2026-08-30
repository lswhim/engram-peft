#!/usr/bin/env bash
set -euo pipefail

# Evaluate the four completed 1B scratch-PT models in two four-GPU waves.
ROOT=${ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram}
PY=${PY:-/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python}
TOKENIZER=${TOKENIZER:-/tmp/Qwen3-1.7B-Base}
RUNS="$ROOT/outputs/scratch_pt_1b/runs"
OUT="$ROOT/outputs/scratch_pt_1b/eval"
LOG="$ROOT/outputs/scratch_pt_1b/eval_logs"

mkdir -p "$OUT" "$LOG"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_DISABLE_XET=1
export https_proxy=http://star-proxy.oa.com:3128
export http_proxy=http://star-proxy.oa.com:3128
export no_proxy=".woa.com,.oa.com,.tencentcos.cn,localhost,127.0.0.1"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

STANDARD_TASKS="wikitext,c4,lambada_openai"
FINEWEB_TASKS="commonsense_qa,hellaswag,openbookqa,piqa,social_iqa,winogrande,arc,mmlu"
SUITE_ONLY=${SUITE_ONLY:-all}

# The evaluator rejects a dummy --engram-weights path for base, so use two
# small wrappers to keep the eight jobs explicit and independently resumable.
run_base() {
  local gpu=$1
  local suite=$2
  local tasks=$3
  local output="$OUT/base_${suite}.json"
  [[ -s "$output" ]] && return
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u examples/evaluate_standard_lm.py \
    --model "$RUNS/base/final" --tokenizer "$TOKENIZER" \
    --method base --seed 42 --tasks "$tasks" --batch-size auto \
    --output "$output" > "$LOG/base_${suite}.log" 2>&1
}

run_engram() {
  local gpu=$1
  local method=$2
  local suite=$3
  local tasks=$4
  local output="$OUT/${method}_${suite}.json"
  [[ -s "$output" ]] && return
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u examples/evaluate_standard_lm.py \
    --model "$RUNS/$method/final/base_model" \
    --tokenizer "$TOKENIZER" \
    --engram-weights "$RUNS/$method/final/engram" \
    --method "$method" --seed 42 --tasks "$tasks" --batch-size auto \
    --output "$output" > "$LOG/${method}_${suite}.log" 2>&1
}

if [[ "$SUITE_ONLY" == all || "$SUITE_ONLY" == standard ]]; then
  run_base 0 standard "$STANDARD_TASKS" &
  run_engram 1 arithmetic standard "$STANDARD_TASKS" &
  run_engram 2 semantic_flatten standard "$STANDARD_TASKS" &
  run_engram 3 semantic_keyed standard "$STANDARD_TASKS" &
fi
if [[ "$SUITE_ONLY" == all || "$SUITE_ONLY" == fineweb ]]; then
  run_base 4 fineweb "$FINEWEB_TASKS" &
  run_engram 5 arithmetic fineweb "$FINEWEB_TASKS" &
  run_engram 6 semantic_flatten fineweb "$FINEWEB_TASKS" &
  run_engram 7 semantic_keyed fineweb "$FINEWEB_TASKS" &
fi
wait

echo "SCRATCH_PT_1B_EVAL_QUEUE_DONE"
