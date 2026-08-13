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
if [[ ! -s "$gate_b_output" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/anguszhang-cfs-nj/seokliu_workspace/engram/src:/anguszhang-cfs-nj/seokliu_workspace/engram \
  "$python" -u examples/run_qqp_paws_frozen.py \
    --model "$model" --method "$method" --seed 44 --result-suffix "$suffix" \
    --batch-size 16 --eval-batch-size 16 --num-workers 4 \
    --output "$gate_b_output" > "$log_dir/qqp_paws_${method}_seed44.log" 2>&1
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
