#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
SHARD="$2"
ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_learned}"
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
MODEL=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
MULTILINGUAL_TABLE=/anguszhang-cfs-nj/seokliu_workspace/engram/rq_tables/wiki15_qwen3_06b_M8K256_500k
OUT="$ROOT/outputs/learned_router_formal"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/logs"

is_complete() {
  local result="$1"
  [[ -s "$result" ]] && "$PY" - "$result" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    raise SystemExit(0 if json.load(handle).get("status") == "complete" else 1)
PY
}

run_crosslingual() {
  local benchmark="$1"
  local router="$2"
  local seed="$3"
  local runner
  local result="$OUT/$benchmark/rq_${router}_seed${seed}/metrics.json"
  local log="$OUT/logs/${benchmark}_rq_${router}_seed${seed}.log"
  if is_complete "$result"; then
    echo "[skip] complete: $result"
    return
  fi
  if [[ "$benchmark" == "xnli" ]]; then
    runner=examples/run_xtreme_xnli.py
  elif [[ "$benchmark" == "pawsx" ]]; then
    runner=examples/run_xtreme_pawsx.py
  else
    echo "unknown benchmark: $benchmark" >&2
    exit 2
  fi
  "$PY" -u "$runner" \
    --method rq --rq_router "$router" \
    --model "$MODEL" \
    --rq_table_dir "$MULTILINGUAL_TABLE" \
    --rq_cache_dir "$MULTILINGUAL_TABLE/learned_router_cache/${benchmark}_${router}_seed${seed}" \
    --output_dir "$OUT/$benchmark" \
    --epochs 1 --batch_size 4 --eval_batch_size 8 --grad_accum 8 \
    --max_length 256 --num_workers 4 --seed "$seed" \
    2>&1 | tee "$log"
}

case "$SHARD" in
  0)
    bash scripts/run_wikibigedit_collision_scaling_worker.sh "$GPU_ID" semantic_learned 42 50000
    run_crosslingual xnli learned 42
    run_crosslingual pawsx learned 42
    ;;
  1)
    bash scripts/run_wikibigedit_collision_scaling_worker.sh "$GPU_ID" semantic_learned 123 50000
    run_crosslingual xnli learned 43
    run_crosslingual pawsx learned 43
    ;;
  2)
    bash scripts/run_wikibigedit_collision_scaling_worker.sh "$GPU_ID" semantic_learned 456 50000
    run_crosslingual xnli learned 44
    run_crosslingual pawsx learned 44
    ;;
  3)
    for seed in 42 43 44; do
      run_crosslingual xnli collision "$seed"
      run_crosslingual pawsx collision "$seed"
    done
    # Existing flatten results cover seeds 42/43; seed 44 completes the
    # three-seed formal comparison without rerunning identical finished jobs.
    run_crosslingual xnli flatten 44
    run_crosslingual pawsx flatten 44
    ;;
  *)
    echo "SHARD must be 0, 1, 2, or 3" >&2
    exit 2
    ;;
esac
