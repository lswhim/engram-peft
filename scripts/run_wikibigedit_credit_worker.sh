#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
MODE="$2"
SEED="${3:-42}"
ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_multilingual}"
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
BASE=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
MANIFEST=data/semantic_memory/wikibigedit_chronological.jsonl
SEMANTIC_TABLE=rq_tables/wikibigedit50k_qwen3emb4b_M8K16
SHUFFLED_TABLE=rq_tables/wikibigedit50k_qwen3emb4b_M8K16_runtime_shuffled_seed42
OUT_ROOT=outputs/semantic_memory/wikibigedit_credit_10k

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p run_logs "$OUT_ROOT/$MODE/seed_${SEED}"

COMMON="n_head_per_ngram=8,use_sparse_embeddings=False,target_layers=[11,21]"
case "$MODE" in
  semantic_flatten)
    TABLE="$SEMANTIC_TABLE"
    EXTRA="memory_fusion=flatten"
    ;;
  semantic_factorized)
    TABLE="$SEMANTIC_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_use_null=False,credit_loss_weight=0.0"
    ;;
  semantic_credit)
    TABLE="$SEMANTIC_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_use_null=False,credit_loss_weight=0.5,credit_pair_fraction=0.5,credit_route_k=4,credit_temperature=1.0"
    ;;
  semantic_factorized_mass)
    TABLE="$SEMANTIC_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_use_null=False,credit_loss_weight=0.0"
    ;;
  semantic_credit_mass)
    TABLE="$SEMANTIC_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_use_null=False,credit_loss_weight=0.5,credit_pair_fraction=0.5,credit_route_k=4,credit_temperature=1.0"
    ;;
  shuffled_credit_mass)
    TABLE="$SHUFFLED_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_use_null=False,credit_loss_weight=0.5,credit_pair_fraction=0.5,credit_route_k=4,credit_temperature=1.0"
    ;;
  shuffled_credit)
    TABLE="$SHUFFLED_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_use_null=False,credit_loss_weight=0.5,credit_pair_fraction=0.5,credit_route_k=4,credit_temperature=1.0"
    ;;
  semantic_specificity)
    TABLE="$SEMANTIC_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=specificity,head_router_use_null=False,credit_loss_weight=0.0"
    ;;
  semantic_specificity_k8)
    TABLE="$SEMANTIC_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=8,head_router_preserve_mass=True,head_router_selection=specificity,head_router_use_null=False,credit_loss_weight=0.0"
    ;;
  shuffled_specificity_k8)
    TABLE="$SHUFFLED_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=8,head_router_preserve_mass=True,head_router_selection=specificity,head_router_use_null=False,credit_loss_weight=0.0"
    ;;
  shuffled_specificity)
    TABLE="$SHUFFLED_TABLE"
    EXTRA="memory_fusion=head_factorized,head_router_top_k=4,head_router_preserve_mass=True,head_router_selection=specificity,head_router_use_null=False,credit_loss_weight=0.0"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

if [[ ! -f "$TABLE/meta.json" ]]; then
  echo "missing RQ table: $TABLE" >&2
  exit 3
fi

RUN_SUFFIX="_wikibigedit_credit10k_${MODE}_seed${SEED}"
METHOD="engram:hash_backend=rq,rq_table_dir=${TABLE},rq_cache_dir=${TABLE}/wikibigedit_credit10k_cache_seed${SEED},${COMMON},${EXTRA}"

"$PY" -u examples/compare_engram_lora.py \
  --model_name "$BASE" --dataset semantic_manifest \
  --manifest_path "$MANIFEST" --subset 10000 --chronological \
  --milestone_examples 10000 \
  --max_steps 500 --batch_size 4 --grad_accum 5 --max_length 128 \
  --num_workers 4 --disable_early_stopping --seed "$SEED" \
  --methods "$METHOD" --run_suffix "$RUN_SUFFIX" --skip_plot --skip_inference

MILESTONE_ROOT="$(find outputs/benchmarks/milestones -mindepth 1 -maxdepth 1 -type d -name "*${RUN_SUFFIX}" | head -n 1)"
if [[ -z "$MILESTONE_ROOT" ]]; then
  echo "milestone directory not found for $RUN_SUFFIX" >&2
  exit 4
fi

"$PY" -u examples/evaluate_semantic_memory.py \
  --model "$BASE" --manifest data/semantic_memory/wikibigedit_eval_10000.jsonl \
  --engram-weights "$MILESTONE_ROOT/writes_10000" \
  --output "$OUT_ROOT/$MODE/seed_${SEED}/at_10000.json" \
  --batch-size 16 --embed-batch-size 64
