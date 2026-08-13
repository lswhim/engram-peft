#!/usr/bin/env bash
set -u
cd /anguszhang-cfs-nj/seokliu_workspace/engram
suffix="_paper_gate1_fineweb_100m_fixedsteps_rq_shuffled_seed43"
output_dir="outputs/semantic_hash_paper/standard_lm"
log_dir="outputs/semantic_hash_paper/logs"

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

for gate_seed in 42 43; do
  gate_suffix="_paper_gate1_fineweb_100m_fixedsteps_rq_shuffled_seed${gate_seed}"
  gate_b_output="outputs/semantic_hash_paper/qqp_paws/rq_shuffled_seed${gate_seed}.json"
  gate_b_log="outputs/semantic_hash_paper/logs/qqp_paws_rq_shuffled_seed${gate_seed}.log"
  [[ -s "$gate_b_output" ]] && continue
  CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/anguszhang-cfs-nj/seokliu_workspace/engram/src:/anguszhang-cfs-nj/seokliu_workspace/engram \
    /anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python -u \
      examples/run_qqp_paws_frozen.py \
      --model /anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base \
      --method rq_shuffled --seed "$gate_seed" --result-suffix "$gate_suffix" \
      --batch-size 16 --eval-batch-size 16 --num-workers 4 \
      --output "$gate_b_output" > "$gate_b_log" 2>&1
done

for task in wikitext lambada; do
  output="$output_dir/rq_shuffled_seed43_${task}.json"
  log="$log_dir/standard_lm_rq_shuffled_seed43_${task}.log"
  [[ -s "$output" ]] && continue
  CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HTTPS_PROXY=http://star-proxy.oa.com:3128 HTTP_PROXY=http://star-proxy.oa.com:3128 \
  /anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python -u \
    examples/evaluate_standard_lm.py \
    --model /anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base \
    --tasks "$task" --method rq_shuffled --seed 43 --batch-size 1 \
    --result-suffix "$suffix" --output "$output" > "$log" 2>&1
done
