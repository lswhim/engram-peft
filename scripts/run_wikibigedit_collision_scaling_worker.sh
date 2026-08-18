#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
MODE="$2"
SEED="${3:-42}"
SCALE="${4:-50000}"
ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_collision}"
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
BASE=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
MANIFEST=data/semantic_memory/wikibigedit_chronological.jsonl
SEMANTIC_TABLE=rq_tables/wikibigedit50k_qwen3emb4b_M8K16
SHUFFLED_TABLE=rq_tables/wikibigedit50k_qwen3emb4b_M8K16_runtime_shuffled_seed42
LOADMATCHED_TABLE=rq_tables/wikibigedit50k_qwen3emb4b_M8K16_loadmatched_seed42
OUT_ROOT="outputs/semantic_memory/wikibigedit_collision_scaling_${SCALE}"

if [[ "$SCALE" != "50000" ]]; then
  echo "Only SCALE=50000 is currently valid: the address statistics and eval manifests are frozen at 50K." >&2
  exit 2
fi

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
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  semantic_rq_snr)
    TABLE="$SEMANTIC_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=rq_snr,head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  semantic_rq_signal)
    TABLE="$SEMANTIC_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=rq_signal,head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  semantic_rq_level_prior|semantic_context_topk)
    TABLE="$SEMANTIC_TABLE"
    if [[ "$MODE" == "semantic_rq_level_prior" ]]; then SELECTION="rq_level_prior"; else SELECTION="context"; fi
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=${SELECTION},head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  semantic_flatten)
    TABLE="$SEMANTIC_TABLE"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},memory_fusion=flatten"
    ;;
  shuffled_specificity)
    TABLE="$SHUFFLED_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=specificity,head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  shuffled_rq_snr)
    TABLE="$SHUFFLED_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=rq_snr,head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  shuffled_rq_signal)
    TABLE="$SHUFFLED_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=rq_signal,head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  shuffled_rq_level_prior|shuffled_context_topk)
    TABLE="$SHUFFLED_TABLE"
    if [[ "$MODE" == "shuffled_rq_level_prior" ]]; then SELECTION="rq_level_prior"; else SELECTION="context"; fi
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=${SELECTION},head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  shuffled_flatten)
    TABLE="$SHUFFLED_TABLE"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},memory_fusion=flatten"
    ;;
  loadmatched_specificity)
    TABLE="$LOADMATCHED_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=specificity,head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  loadmatched_rq_snr)
    TABLE="$LOADMATCHED_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=rq_snr,head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  loadmatched_rq_signal)
    TABLE="$LOADMATCHED_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=rq_signal,head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  loadmatched_rq_level_prior|loadmatched_context_topk)
    TABLE="$LOADMATCHED_TABLE"
    if [[ "$MODE" == "loadmatched_rq_level_prior" ]]; then SELECTION="rq_level_prior"; else SELECTION="context"; fi
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=${SELECTION},head_router_use_null=False,credit_loss_weight=0.0"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},${EXTRA}"
    ;;
  loadmatched_flatten)
    TABLE="$LOADMATCHED_TABLE"
    METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_collision50k_cache_seed${SEED},${COMMON},memory_fusion=flatten"
    ;;
  arithmetic)
    METHOD="engram:hash_backend=arithmetic_fixed,engram_vocab_size_per_ngram=[128,128],${COMMON},memory_fusion=flatten"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 3
    ;;
esac

if [[ "$MODE" != "arithmetic" && ! -f "$TABLE/meta.json" ]]; then
  echo "missing RQ table: $TABLE" >&2
  exit 4
fi

RUN_SUFFIX="_wikibigedit_collision${SCALE}_${MODE}_seed${SEED}"
LOG="run_logs/wikibigedit_collision${SCALE}_${MODE}_seed${SEED}.log"

"$PY" -u examples/compare_engram_lora.py \
  --model_name "$BASE" --dataset semantic_manifest \
  --manifest_path "$MANIFEST" --subset "$SCALE" --chronological \
  --milestone_examples 1000 5000 10000 50000 \
  --max_steps 2500 --batch_size 4 --grad_accum 5 --max_length 128 \
  --num_workers 4 --disable_early_stopping --seed "$SEED" \
  --methods "$METHOD" --run_suffix "$RUN_SUFFIX" --skip_plot --skip_inference \
  2>&1 | tee "$LOG"

MILESTONE_ROOT="$(find outputs/benchmarks/milestones -mindepth 1 -maxdepth 1 -type d -name "*${RUN_SUFFIX}" | head -n 1)"
if [[ -z "$MILESTONE_ROOT" ]]; then
  echo "milestone directory not found for $RUN_SUFFIX" >&2
  exit 5
fi

for POINT in 1000 5000 10000 50000; do
  "$PY" -u examples/evaluate_semantic_memory.py \
    --model "$BASE" --manifest "data/semantic_memory/wikibigedit_eval_${POINT}.jsonl" \
    --engram-weights "$MILESTONE_ROOT/writes_${POINT}" \
    --output "$OUT_ROOT/$MODE/seed_${SEED}/at_${POINT}.json" \
    --batch-size 16 --embed-batch-size 64 \
    2>&1 | tee -a "$LOG"
done
