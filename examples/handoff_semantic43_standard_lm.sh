#!/usr/bin/env bash
set -u
cd /anguszhang-cfs-nj/seokliu_workspace/engram
suffix="_paper_gate1_fineweb_100m_fixedsteps_semantic_rq_seed43"
output_dir="outputs/semantic_hash_paper/standard_lm"
log_dir="outputs/semantic_hash_paper/logs"

gate_b_complete() {
  /anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python - "$1" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1]))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "complete" and payload.get("paper_eligible") is True else 1)
PY
}

gate_b_running() {
  /anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python - "$1" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1]))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "training" else 1)
PY
}

while ! /anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python - "$suffix" <<'PY'
import glob, json, sys
for path in glob.glob("outputs/benchmarks/*.json"):
    try:
        payload = json.load(open(path))
    except (OSError, json.JSONDecodeError):
        continue
    metrics = payload.get("metrics", {})
    if payload.get("params", {}).get("run_suffix") == sys.argv[1] and (
        metrics.get("fixed_steps_complete") is True
        and metrics.get("completed_steps") == metrics.get("planned_steps") == 12_208
    ):
        raise SystemExit(0)
raise SystemExit(1)
PY
do
  sleep 20
done

for task in wikitext lambada; do
  output="$output_dir/semantic_rq_seed43_${task}.json"
  log="$log_dir/standard_lm_semantic_rq_seed43_${task}.log"
  [[ -s "$output" ]] && continue
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HTTPS_PROXY=http://star-proxy.oa.com:3128 HTTP_PROXY=http://star-proxy.oa.com:3128 \
  /anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python -u \
    examples/evaluate_standard_lm.py \
    --model /anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base \
    --tasks "$task" --method semantic_rq --seed 43 --batch-size 1 \
    --result-suffix "$suffix" --output "$output" > "$log" 2>&1
done

gate_b_output="outputs/semantic_hash_paper/qqp_paws/semantic_rq_seed43.json"
gate_b_log="outputs/semantic_hash_paper/logs/qqp_paws_semantic_rq_seed43.log"
while gate_b_running "$gate_b_output"; do sleep 30; done
if ! gate_b_complete "$gate_b_output"; then
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/anguszhang-cfs-nj/seokliu_workspace/engram/src:/anguszhang-cfs-nj/seokliu_workspace/engram \
  /anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python -u \
    examples/run_qqp_paws_frozen.py \
    --model /anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base \
    --method semantic_rq --seed 43 --result-suffix "$suffix" \
    --batch-size 16 --eval-batch-size 16 --num-workers 4 \
    --output "$gate_b_output" > "$gate_b_log" 2>&1
fi

for method_seed_suffix in \
  "base 43 none" \
  "arithmetic_matched 43 _paper_gate1_fineweb_100m_fixedsteps_arithmetic_matched_seed43"
do
  read -r gate_method gate_seed gate_suffix <<< "$method_seed_suffix"
  gate_b_output="outputs/semantic_hash_paper/qqp_paws/${gate_method}_seed${gate_seed}.json"
  while gate_b_running "$gate_b_output"; do sleep 30; done
  gate_b_complete "$gate_b_output" && continue
  suffix_args=()
  [[ "$gate_suffix" != none ]] && suffix_args=(--result-suffix "$gate_suffix")
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/anguszhang-cfs-nj/seokliu_workspace/engram/src:/anguszhang-cfs-nj/seokliu_workspace/engram \
  /anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python -u \
    examples/run_qqp_paws_frozen.py \
    --model /anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base \
    --method "$gate_method" --seed "$gate_seed" "${suffix_args[@]}" \
    --batch-size 16 --eval-batch-size 16 --num-workers 4 \
    --output "$gate_b_output" \
    > "$log_dir/qqp_paws_${gate_method}_seed${gate_seed}.log" 2>&1
done

for c4_spec in \
  "semantic_rq 42 _paper_gate1_fineweb_100m_fixedsteps_semantic_rq_seed42" \
  "arithmetic_matched 42 _paper_gate1_fineweb_100m_fixedsteps_arithmetic_matched_seed42" \
  "arithmetic_matched 43 _paper_gate1_fineweb_100m_fixedsteps_arithmetic_matched_seed43"
do
  read -r c4_method c4_seed c4_suffix <<< "$c4_spec"
  c4_output="$output_dir/${c4_method}_seed${c4_seed}_c4.json"
  [[ -s "$c4_output" ]] && continue
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HTTPS_PROXY=http://star-proxy.oa.com:3128 HTTP_PROXY=http://star-proxy.oa.com:3128 \
  /anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python -u \
    examples/evaluate_standard_lm.py \
    --model /anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base \
    --tasks c4 --method "$c4_method" --seed "$c4_seed" --batch-size 1 \
    --result-suffix "$c4_suffix" --output "$c4_output" \
    > "$log_dir/standard_lm_${c4_method}_seed${c4_seed}_c4.log" 2>&1
done

c4_output="$output_dir/semantic_rq_seed43_c4.json"
if [[ ! -s "$c4_output" ]]; then
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HTTPS_PROXY=http://star-proxy.oa.com:3128 HTTP_PROXY=http://star-proxy.oa.com:3128 \
  /anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python -u \
    examples/evaluate_standard_lm.py \
    --model /anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base \
    --tasks c4 --method semantic_rq --seed 43 --batch-size 1 \
    --result-suffix "$suffix" --output "$c4_output" \
    > "$log_dir/standard_lm_semantic_rq_seed43_c4.log" 2>&1
fi
