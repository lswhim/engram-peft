#!/usr/bin/env bash
set -euo pipefail

ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram
cd "$ROOT"

SEM_CACHE=rq_tables/fineweb_qwen3emb06b_M8K16_300k_strict/lm100m_k16_emb06b_semantic_keyed_seed42
FLAT_CACHE=rq_tables/fineweb_qwen3emb06b_M8K16_300k_strict/lm100m_k16_emb06b_semantic_flatten_seed42
SHUF_CACHE=rq_tables/fineweb_qwen3emb06b_M8K16_300k_strict_runtime_shuffled_seed42/lm100m_k16_emb06b_shuffled_keyed_seed42

echo "WAITING_FOR_LM_PREENCODE time=$(date -Is)"
while [[ ! -s "$SEM_CACHE/preencode_complete.json" || ! -s "$SHUF_CACHE/preencode_complete.json" ]]; do
  if ! pgrep -f "preencode_xnli_lm.py --mode lm" >/dev/null 2>&1; then
    echo "LM_PREENCODE_ABORTED time=$(date -Is)" >&2
    exit 4
  fi
  sleep 60
done

# Flatten and keyed use the same semantic codes; only the shuffled table has a
# distinct runtime address mapping and therefore needs its own pre-encode.
mkdir -p "$FLAT_CACHE"
cp -f "$SEM_CACHE/semantic_codes.sqlite3" "$FLAT_CACHE/semantic_codes.sqlite3"
cp -f "$SEM_CACHE/preencode_complete.json" "$FLAT_CACHE/preencode_complete.json"
echo "LM_PREENCODE_READY time=$(date -Is)"

declare -A PID_MODE=()
for spec in "0 semantic_flatten" "1 semantic_keyed" "2 shuffled_keyed"; do
  read -r gpu mode <<< "$spec"
  log="run_logs/lm100m_k16_emb06b_${mode}_seed42.log"
  echo "LM_LAUNCH gpu=$gpu mode=$mode time=$(date -Is)"
  bash scripts/run_lm_100m_k16_worker.sh "$gpu" "$mode" 42 > "$log" 2>&1 &
  PID_MODE[$!]="$mode"
done

for pid in "${!PID_MODE[@]}"; do
  rc=0
  wait "$pid" || rc=$?
  echo "LM_DONE mode=${PID_MODE[$pid]} rc=$rc time=$(date -Is)"
done
echo "LM_K16_EMB06B_SUITE_DONE time=$(date -Is)"
