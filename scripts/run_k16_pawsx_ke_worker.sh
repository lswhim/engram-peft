#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?gpu}
METHOD=${2:?semantic|shuffled|arithmetic}
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram_multilingual
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
MODEL=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
SEM_TABLE=rq_tables/fineweb_qwen3emb4b_M8K16_300k_strict
SHUF_TABLE=${SEM_TABLE}_runtime_shuffled_seed42
OUT=outputs/k16_4b_rerun

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p run_logs "$OUT/pawsx" "$OUT/standard_ke"

case "$METHOD" in
  semantic)
    TABLE="$SEM_TABLE"
    PAWS_METHOD=rq
    SPEC="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/k16_suite_cache_seed42,n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
    ;;
  shuffled)
    TABLE="$SHUF_TABLE"
    PAWS_METHOD=rq
    SPEC="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/k16_suite_cache_seed42,n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
    ;;
  arithmetic)
    TABLE="$SEM_TABLE"
    PAWS_METHOD=arithmetic_matched
    SPEC="engram:hash_backend=arithmetic_fixed,engram_vocab_size_per_ngram=[128,128],n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
    ;;
  *) echo "unknown method: $METHOD" >&2; exit 2 ;;
esac

if [[ "$METHOD" == shuffled ]]; then
  while [[ ! -f "$SHUF_TABLE/meta.json" ]]; do sleep 20; done
fi

"$PY" -u examples/run_xtreme_pawsx.py \
  --method "$PAWS_METHOD" --model "$MODEL" \
  --rq_table_dir "$TABLE" --rq_cache_dir "$TABLE/pawsx_cache_seed42" \
  --output_dir "$OUT/pawsx/$METHOD" --epochs 1 --batch_size 4 \
  --grad_accum 8 --eval_batch_size 8 --max_length 256 --num_workers 4 --seed 42 \
  > "run_logs/k16_4b_pawsx_${METHOD}_seed42.log" 2>&1

run_standard_ke() {
  local dataset=$1 steps=$2
  local suffix="k16_4b_${dataset}_${METHOD}_seed42"
  local method_budget="${SPEC},save_steps=${steps},eval_steps=${steps}"
  "$PY" -u examples/compare_engram_lora.py \
    --dataset "${dataset}_canonical" --model_name "$MODEL" --max_steps "$steps" \
    --subset 999999 --batch_size 4 --grad_accum 8 --max_length 64 --num_workers 0 \
    --seed 42 --disable_early_stopping --skip_plot --skip_inference \
    --run_suffix "$suffix" --methods "$method_budget" \
    > "run_logs/k16_4b_${dataset}_${METHOD}_train_seed42.log" 2>&1
  local ckpt
  ckpt="$(find outputs/benchmarks -mindepth 1 -maxdepth 1 -type d -name "ckpt_*${suffix}" | head -n 1)"
  test -s "$ckpt/engram_adapters.safetensors"
  "$PY" -u examples/evaluate_standard_ke.py \
    --dataset "$dataset" --model "$MODEL" --engram-weights "$ckpt" \
    --output "$OUT/standard_ke/${dataset}_${METHOD}_seed42.json" \
    --limit 0 --batch-size 24 --case-chunk-size 32 \
    > "run_logs/k16_4b_${dataset}_${METHOD}_eval_seed42.log" 2>&1
}

run_standard_ke counterfact 345
run_standard_ke zsre 205

# MQuAKE keeps the repository's established 3K/1K-step protocol; it is
# reported separately from the canonical-only CounterFact/ZsRE table.
MQ_SUFFIX="k16_4b_mquake_${METHOD}_seed42"
"$PY" -u examples/compare_engram_lora.py \
  --dataset mquake --model_name "$MODEL" --max_steps 1000 --subset 3000 \
  --batch_size 8 --grad_accum 4 --max_length 128 --num_workers 0 \
  --seed 42 --disable_early_stopping --skip_plot --skip_inference \
  --run_suffix "$MQ_SUFFIX" --methods "${SPEC},save_steps=1000,eval_steps=1000" \
  > "run_logs/k16_4b_mquake_${METHOD}_train_seed42.log" 2>&1
MQ_CKPT="$(find outputs/benchmarks -mindepth 1 -maxdepth 1 -type d -name "ckpt_*${MQ_SUFFIX}" | head -n 1)"
test -s "$MQ_CKPT/engram_adapters.safetensors"
"$PY" -u examples/eval_ke.py --dataset mquake --model_name "$MODEL" \
  --engram_weights "$MQ_CKPT" --limit 0 \
  > "run_logs/k16_4b_mquake_${METHOD}_eval_seed42.log" 2>&1

echo "K16_4B_SUITE_DONE method=$METHOD gpu=$GPU"
