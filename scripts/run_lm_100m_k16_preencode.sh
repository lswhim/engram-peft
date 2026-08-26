#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?gpu}
MODE=${2:?mode: semantic_keyed|shuffled_keyed}
SEED=${3:-42}
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
SEM_TABLE=rq_tables/fineweb_qwen3emb06b_M8K16_300k_strict
SHUF_TABLE=${SEM_TABLE}_runtime_shuffled_seed42

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_DISABLE_XET=1
export https_proxy=http://star-proxy.oa.com:3128 http_proxy=http://star-proxy.oa.com:3128
export no_proxy=.woa.com,.oa.com,localhost,127.0.0.1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TORCH_NUM_THREADS=8
export PREENCODE_FLUSH_ROWS=${PREENCODE_FLUSH_ROWS:-2000}
mkdir -p run_logs

case "$MODE" in
  semantic_keyed)
    TABLE="$SEM_TABLE"
    FINAL_CACHE="$SEM_TABLE/lm100m_k16_emb06b_semantic_keyed_seed${SEED}"
    ;;
  shuffled_keyed)
    TABLE="$SHUF_TABLE"
    FINAL_CACHE="$SHUF_TABLE/lm100m_k16_emb06b_shuffled_keyed_seed${SEED}"
    test -f "$TABLE/meta.json"
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

LOG="run_logs/preencode_lm100m_k16_emb06b_${MODE}_seed${SEED}.log"
SCRATCH_ROOT=${PREENCODE_CACHE_ROOT:-/tmp/engram_k16_lm_preencode}
CACHE="$SCRATCH_ROOT/${MODE}_seed${SEED}"
mkdir -p "$CACHE"

# SQLite is unreliable on the shared CFS mount under this write volume. Keep
# the hot write path on the pod-local overlay and publish one finished file.
if [[ ! -s "$CACHE/preencode_complete.json" || ! -s "$CACHE/semantic_codes.sqlite3" ]]; then
  rm -f "$CACHE/preencode_complete.json"
  "$PY" -u scripts/preencode_xnli_lm.py \
    --mode lm --seed "$SEED" --lm-rows 100000 --max-length 1024 --batch-size 256 \
    --table-dir "$TABLE" --cache-dir "$CACHE" \
    > "$LOG" 2>&1
fi
mkdir -p "$FINAL_CACHE"
cp -f "$CACHE/semantic_codes.sqlite3" "$FINAL_CACHE/semantic_codes.sqlite3"
cp -f "$CACHE/preencode_complete.json" "$FINAL_CACHE/preencode_complete.json"
echo "PREENCODE_LM100M_K16_EMB06B_DONE mode=$MODE seed=$SEED gpu=$GPU"
