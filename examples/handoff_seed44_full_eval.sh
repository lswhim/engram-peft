#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 METHOD GPU RESULT_SUFFIX" >&2
  exit 2
fi

method="$1"
gpu="$2"
suffix="$3"
cd /anguszhang-cfs-nj/seokliu_workspace/engram
python=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
model=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
output_dir=outputs/semantic_hash_paper/standard_lm
log_dir=outputs/semantic_hash_paper/logs

gate_b_complete() {
  "$python" - "$1" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1]))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "complete" and payload.get("paper_eligible") is True else 1)
PY
}

gate_b_running() {
  "$python" - "$1" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1]))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "training" else 1)
PY
}

while ! "$python" - "$suffix" <<'PY'
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
  output="$output_dir/${method}_seed44_${task}.json"
  [[ -s "$output" ]] && continue
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HTTPS_PROXY=http://star-proxy.oa.com:3128 HTTP_PROXY=http://star-proxy.oa.com:3128 \
  "$python" -u examples/evaluate_standard_lm.py \
    --model "$model" --tasks "$task" --method "$method" --seed 44 --batch-size 1 \
    --result-suffix "$suffix" --output "$output" \
    > "$log_dir/standard_lm_${method}_seed44_${task}.log" 2>&1
done

gate_b_output="outputs/semantic_hash_paper/qqp_paws/${method}_seed44.json"
while gate_b_running "$gate_b_output"; do sleep 30; done
if ! gate_b_complete "$gate_b_output"; then
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/anguszhang-cfs-nj/seokliu_workspace/engram/src:/anguszhang-cfs-nj/seokliu_workspace/engram \
  "$python" -u examples/run_qqp_paws_frozen.py \
    --model "$model" --method "$method" --seed 44 --result-suffix "$suffix" \
    --batch-size 16 --eval-batch-size 16 --num-workers 4 \
    --output "$gate_b_output" > "$log_dir/qqp_paws_${method}_seed44.log" 2>&1
fi

if [[ "$method" == "rq_shuffled" ]]; then
  base_output="outputs/semantic_hash_paper/qqp_paws/base_seed44.json"
  while gate_b_running "$base_output"; do sleep 30; done
  if ! gate_b_complete "$base_output"; then
    CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/anguszhang-cfs-nj/seokliu_workspace/engram/src:/anguszhang-cfs-nj/seokliu_workspace/engram \
    "$python" -u examples/run_qqp_paws_frozen.py \
      --model "$model" --method base --seed 44 \
      --batch-size 16 --eval-batch-size 16 --num-workers 4 \
      --output "$base_output" > "$log_dir/qqp_paws_base_seed44.log" 2>&1
  fi
fi

# C4 has long examples and previously OOMed only while sharing a training GPU.
# It is intentionally last, after this card's training and Gate-B jobs release memory.
c4_output="$output_dir/${method}_seed44_c4.json"
if [[ ! -s "$c4_output" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HTTPS_PROXY=http://star-proxy.oa.com:3128 HTTP_PROXY=http://star-proxy.oa.com:3128 \
  "$python" -u examples/evaluate_standard_lm.py \
    --model "$model" --tasks c4 --method "$method" --seed 44 --batch-size 1 \
    --result-suffix "$suffix" --output "$c4_output" \
    > "$log_dir/standard_lm_${method}_seed44_c4.log" 2>&1
fi
