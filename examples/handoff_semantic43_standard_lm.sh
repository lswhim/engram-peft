#!/usr/bin/env bash
set -u
cd /anguszhang-cfs-nj/seokliu_workspace/engram
suffix="_paper_gate1_fineweb_100m_fixedsteps_semantic_rq_seed43"
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
