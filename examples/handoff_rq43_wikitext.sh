#!/usr/bin/env bash
set -u
cd /anguszhang-cfs-nj/seokliu_workspace/engram
suffix="_paper_gate1_fineweb_100m_fixedsteps_rq_shuffled_seed43"
output="outputs/semantic_hash_paper/standard_lm/rq_shuffled_seed43_wikitext.json"
log="outputs/semantic_hash_paper/logs/standard_lm_rq_shuffled_seed43_wikitext.log"

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

[[ -s "$output" ]] && exit 0
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HTTPS_PROXY=http://star-proxy.oa.com:3128 HTTP_PROXY=http://star-proxy.oa.com:3128 \
/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python -u \
  examples/evaluate_standard_lm.py \
  --model /anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base \
  --tasks wikitext --method rq_shuffled --seed 43 --batch-size 1 \
  --result-suffix "$suffix" --output "$output" > "$log" 2>&1
