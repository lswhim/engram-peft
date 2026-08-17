#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
MODE="$2"
SEED="${3:-42}"
ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_collision}"
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
BASE=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
TIMESTEP_DIR=data/semantic_memory/wikibigedit_official_timesteps
FULL_MANIFEST=data/semantic_memory/wikibigedit_official_chronological.jsonl
SEMANTIC_TABLE=rq_tables/wikibigedit_official502k_qwen3emb4b_M8K16
SHUFFLED_TABLE=rq_tables/wikibigedit_official502k_qwen3emb4b_M8K16_runtime_shuffled_seed42
OUT_ROOT=outputs/semantic_memory/wikibigedit_official

TIMESTEPS=(
  wiki_big_edit_20240201_20240220
  wiki_big_edit_20240220_20240301
  wiki_big_edit_20240301_20240320
  wiki_big_edit_20240320_20240401
  wiki_big_edit_20240401_20240501
  wiki_big_edit_20240501_20240601
  wiki_big_edit_20240601_20240620
  wiki_big_edit_20240620_20240701
)
EXPECTED_COUNTS=(26922 29835 54504 43443 121116 101728 69403 55431)

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p run_logs "$OUT_ROOT/$MODE/seed_${SEED}"

COMMON="n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
case "$MODE" in
  semantic_specificity)
    TABLE="$SEMANTIC_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=specificity,head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/official_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  semantic_flatten)
    TABLE="$SEMANTIC_TABLE"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/official_cache_seed${SEED},${COMMON},memory_fusion=flatten"
    ;;
  shuffled_specificity)
    TABLE="$SHUFFLED_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=specificity,head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/official_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  shuffled_flatten)
    TABLE="$SHUFFLED_TABLE"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/official_cache_seed${SEED},${COMMON},memory_fusion=flatten"
    ;;
  arithmetic)
    METHOD="engram:hash_backend=arithmetic_fixed,engram_vocab_size_per_ngram=[128,128],${COMMON},memory_fusion=flatten"
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

if [[ "$MODE" != arithmetic && ! -f "$TABLE/meta.json" ]]; then
  echo "missing official train-only RQ table: $TABLE" >&2
  exit 3
fi

previous=""
cumulative=0
checkpoints=()
for index in "${!TIMESTEPS[@]}"; do
  name="${TIMESTEPS[$index]}"
  expected="${EXPECTED_COUNTS[$index]}"
  manifest="$TIMESTEP_DIR/$name.jsonl"
  actual="$(wc -l < "$manifest")"
  if [[ "$actual" -ne "$expected" ]]; then
    echo "timestep size mismatch for $name: expected=$expected actual=$actual" >&2
    exit 4
  fi
  steps=$(( (actual + 19) / 20 ))
  cumulative=$(( cumulative + actual ))
  suffix="_wikibigedit_official_${MODE}_seed${SEED}_t${index}"
  resume_args=()
  if [[ -n "$previous" ]]; then
    resume_args=(--resume_engram_weights "$previous")
  fi
  "$PY" -u examples/compare_engram_lora.py \
    --model_name "$BASE" --dataset semantic_manifest \
    --manifest_path "$manifest" --subset "$actual" --chronological \
    --max_steps "$steps" --batch_size 4 --grad_accum 5 --max_length 128 \
    --prompt_format qa --num_workers 4 --disable_early_stopping --seed "$SEED" \
    --methods "$METHOD" --run_suffix "$suffix" --skip_plot --skip_inference \
    "${resume_args[@]}"
  previous="$(find outputs/benchmarks -mindepth 1 -maxdepth 1 -type d -name "ckpt_*${suffix}" | head -n 1)"
  if [[ -z "$previous" ]]; then
    echo "checkpoint not found after timestep $index" >&2
    exit 5
  fi
  checkpoints+=("$previous")
  printf '%s\n' "$previous" > "$OUT_ROOT/$MODE/seed_${SEED}/checkpoint_t${index}.txt"
done

cumulative=0
for index in "${!TIMESTEPS[@]}"; do
  cumulative=$(( cumulative + EXPECTED_COUNTS[index] ))
  "$PY" -u examples/evaluate_semantic_memory.py \
    --model "$BASE" --manifest "$FULL_MANIFEST" --limit "$cumulative" \
    --engram-weights "${checkpoints[$index]}" \
    --output "$OUT_ROOT/$MODE/seed_${SEED}/t${index}_at_${cumulative}.json" \
    --prompt-format qa --locality-mode pre_post_preservation \
    --batch-size 16 --embed-batch-size 64
done
