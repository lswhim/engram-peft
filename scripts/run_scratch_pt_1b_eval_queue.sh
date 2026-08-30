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
# lm-eval 0.4.12's mmlu group omits the dataset config name. Expand it to
# the registered subject tasks so each one loads cais/mmlu/<subject> correctly.
MMLU_TASKS="mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy,mmlu_business_ethics,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_chemistry,mmlu_college_computer_science,mmlu_college_mathematics,mmlu_college_medicine,mmlu_college_physics,mmlu_computer_security,mmlu_conceptual_physics,mmlu_econometrics,mmlu_electrical_engineering,mmlu_elementary_mathematics,mmlu_formal_logic,mmlu_global_facts,mmlu_high_school_biology,mmlu_high_school_chemistry,mmlu_high_school_computer_science,mmlu_high_school_european_history,mmlu_high_school_geography,mmlu_high_school_government_and_politics,mmlu_high_school_macroeconomics,mmlu_high_school_mathematics,mmlu_high_school_microeconomics,mmlu_high_school_physics,mmlu_high_school_psychology,mmlu_high_school_statistics,mmlu_high_school_us_history,mmlu_high_school_world_history,mmlu_human_aging,mmlu_human_sexuality,mmlu_international_law,mmlu_jurisprudence,mmlu_logical_fallacies,mmlu_machine_learning,mmlu_management,mmlu_marketing,mmlu_medical_genetics,mmlu_miscellaneous,mmlu_moral_disputes,mmlu_moral_scenarios,mmlu_nutrition,mmlu_philosophy,mmlu_prehistory,mmlu_professional_accounting,mmlu_professional_law,mmlu_professional_medicine,mmlu_professional_psychology,mmlu_public_relations,mmlu_security_studies,mmlu_sociology,mmlu_us_foreign_policy,mmlu_virology,mmlu_world_religions"
FINEWEB_TASKS="commonsense_qa,hellaswag,openbookqa,piqa,social_iqa,winogrande,arc_challenge,$MMLU_TASKS"
SUITE_ONLY=${SUITE_ONLY:-all}
METHODS_ONLY=${METHODS_ONLY:-all}
BATCH_SIZE=${BATCH_SIZE:-auto}

method_enabled() {
  [[ "$METHODS_ONLY" == all || ",$METHODS_ONLY," == *,"$1",* ]]
}

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
    --method base --seed 42 --tasks "$tasks" --batch-size "$BATCH_SIZE" \
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
    --method "$method" --seed 42 --tasks "$tasks" --batch-size "$BATCH_SIZE" \
    --output "$output" > "$LOG/${method}_${suite}.log" 2>&1
}

if [[ "$SUITE_ONLY" == all || "$SUITE_ONLY" == standard ]]; then
  method_enabled base && run_base 0 standard "$STANDARD_TASKS" &
  method_enabled arithmetic && run_engram 1 arithmetic standard "$STANDARD_TASKS" &
  method_enabled semantic_flatten && run_engram 2 semantic_flatten standard "$STANDARD_TASKS" &
  method_enabled semantic_keyed && run_engram 3 semantic_keyed standard "$STANDARD_TASKS" &
fi
if [[ "$SUITE_ONLY" == all || "$SUITE_ONLY" == fineweb ]]; then
  method_enabled base && run_base 4 fineweb "$FINEWEB_TASKS" &
  method_enabled arithmetic && run_engram 5 arithmetic fineweb "$FINEWEB_TASKS" &
  method_enabled semantic_flatten && run_engram 6 semantic_flatten fineweb "$FINEWEB_TASKS" &
  method_enabled semantic_keyed && run_engram 7 semantic_keyed fineweb "$FINEWEB_TASKS" &
fi
wait

echo "SCRATCH_PT_1B_EVAL_QUEUE_DONE"
