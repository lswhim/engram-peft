#!/usr/bin/env bash
set -euo pipefail

ROOT=/tmp/engram_scratch_pt_code
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
TORCHRUN=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/torchrun
TOKENIZER=/tmp/Qwen3-1.7B-Base
DATA=/tmp/engram_scratch_pt_1b_data_fast
OUT=/anguszhang-cfs-nj/seokliu_workspace/engram/outputs/scratch_pt_1b/runs
TABLE=/anguszhang-cfs-nj/seokliu_workspace/engram/rq_tables/fineweb_qwen3emb06b_M8K16_300k_strict
RQ_CACHE=/anguszhang-cfs-nj/seokliu_workspace/engram/outputs/scratch_pt_1b/rq_cache_k16
RQ_WORK_CACHE=/tmp/engram_scratch_pt_1b_rq_shards_k16_v2

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TORCH_NUM_THREADS=2
export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT"
while [[ ! -s "$DATA/metadata.json" ]]; do
  sleep 20
done

# This queue can be launched while the original Base -> arithmetic queue is
# still running. It waits for arithmetic to finish before adding semantic PT.
if [[ -s "$OUT/base/metrics.json" && ! -s "$OUT/arithmetic/metrics.json" ]]; then
  echo "[scratch queue] waiting for arithmetic to finish" | tee -a "$OUT/queue.log"
  while [[ ! -s "$OUT/arithmetic/metrics.json" ]]; do
    sleep 60
  done
fi

for mode in base arithmetic semantic_flatten semantic_keyed; do
  mode_out="$OUT/$mode"
  log="$OUT/${mode}.log"
  if [[ -s "$mode_out/metrics.json" ]]; then
    echo "[scratch queue] already complete: $mode"
    continue
  fi
  mkdir -p "$mode_out"
  echo "[scratch queue] starting mode=$mode" | tee -a "$log"
  extra_args=()
  if [[ "$mode" == semantic_flatten || "$mode" == semantic_keyed ]]; then
    if [[ ! -s "$RQ_CACHE/preencode_complete.json" ]]; then
      echo "[scratch queue] building globally deduplicated exact RQ cache" | tee -a "$OUT/queue.log"
      CAND="$RQ_WORK_CACHE/candidates"
      UNIQUE="$RQ_WORK_CACHE/unique"
      ENCODED="$RQ_WORK_CACHE/encoded"
      SEED="$RQ_WORK_CACHE/seed.sqlite3"
      mkdir -p "$CAND" "$UNIQUE" "$ENCODED"
      train_tokens=$(( $(stat -c %s "$DATA/train.bin") / 4 ))
      train_rows=$(( (train_tokens - 1) / 2048 ))
      shard_rows=$(( (train_rows + 7) / 8 ))
      if [[ ! -s "$CAND/collect_complete" ]]; then
        collect_pids=()
        for gpu in 0 1 2 3 4 5 6 7; do
          start_row=$(( gpu * shard_rows ))
          remaining=$(( train_rows - start_row ))
          count_rows=$(( remaining > shard_rows ? shard_rows : remaining ))
          if (( count_rows <= 0 )); then continue; fi
          CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/collect_rq_candidates.py \
            --table-dir "$TABLE" --data-dir "$DATA" --tokenizer "$TOKENIZER" \
            --output-dir "$CAND" --sequence-length 2048 --batch-rows 512 \
            --start-row "$start_row" --max-rows "$count_rows" --partitions 64 \
            --stream train > "$CAND/train_rank$gpu.log" 2>&1 &
          collect_pids+=("$!")
        done
        collect_status=0
        for pid in "${collect_pids[@]}"; do wait "$pid" || collect_status=1; done
        if (( collect_status != 0 )); then
          echo "[scratch queue] RQ candidate collection failed" | tee -a "$OUT/queue.log"
          exit 1
        fi
        "$PY" -u scripts/collect_rq_candidates.py \
          --table-dir "$TABLE" --data-dir "$DATA" --tokenizer "$TOKENIZER" \
          --output-dir "$CAND" --sequence-length 2048 --batch-rows 512 \
          --start-row 0 --max-rows 0 --partitions 64 --stream eval \
          > "$CAND/eval.log" 2>&1
        printf 'complete\n' > "$CAND/collect_complete"
      fi
      if [[ ! -s "$UNIQUE/dedup_complete" ]]; then
        "$PY" -u scripts/dedup_rq_candidates.py \
          --input-dir "$CAND" --output-dir "$UNIQUE" --partitions 64 \
          >> "$OUT/rq_preencode.log" 2>&1
        printf 'complete\n' > "$UNIQUE/dedup_complete"
      fi
      if [[ ! -s "$ENCODED/encode_complete" ]]; then
        if [[ ! -s "$SEED" ]]; then
          seed_inputs=( )
          for f in "$RQ_WORK_CACHE"/train_rank*/semantic_codes.sqlite3; do
            [[ -s "$f" ]] && seed_inputs+=("$f")
          done
          if (( ${#seed_inputs[@]} )); then
            "$PY" -u scripts/merge_rq_cache.py --output "$SEED" "${seed_inputs[@]}" \
              >> "$OUT/rq_preencode.log" 2>&1
          else
            "$PY" -c "import sqlite3; c=sqlite3.connect('$SEED'); c.execute(\"CREATE TABLE codes (n INTEGER NOT NULL, key INTEGER NOT NULL, code BLOB NOT NULL, PRIMARY KEY (n,key))\"); c.commit(); c.close()"
          fi
        fi
        encode_pids=()
        for gpu in 0 1 2 3 4 5 6 7; do
          mkdir -p "$ENCODED/rank$gpu"
          CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/encode_rq_candidates.py \
            --table-dir "$TABLE" --candidate-dir "$UNIQUE" \
            --output-dir "$ENCODED/rank$gpu" --seed-db "$SEED" \
            --gpu "$gpu" --device cuda:0 --world-size 8 \
            --embed-batch-size 1024 --chunk-size 16384 \
            > "$ENCODED/rank$gpu.log" 2>&1 &
          encode_pids+=("$!")
        done
        encode_status=0
        for pid in "${encode_pids[@]}"; do wait "$pid" || encode_status=1; done
        if (( encode_status != 0 )); then
          echo "[scratch queue] RQ candidate encoding failed" | tee -a "$OUT/queue.log"
          exit 1
        fi
        encoded_inputs=("$SEED")
        for f in "$ENCODED"/rank*/semantic_codes.sqlite3; do
          [[ -s "$f" ]] && encoded_inputs+=("$f")
        done
        "$PY" -u scripts/merge_rq_cache.py \
          --output "$RQ_CACHE/semantic_codes.sqlite3" "${encoded_inputs[@]}" \
          >> "$OUT/rq_preencode.log" 2>&1
        printf '{"status":"complete","stream":"both","sequence_length":2048}\n' \
          > "$RQ_CACHE/preencode_complete.json"
        printf 'complete\n' > "$ENCODED/encode_complete"
      fi
    fi
    mkdir -p "$RQ_CACHE/rank0" "$RQ_CACHE/rank1" "$RQ_CACHE/rank2" "$RQ_CACHE/rank3" \
      "$RQ_CACHE/rank4" "$RQ_CACHE/rank5" "$RQ_CACHE/rank6" "$RQ_CACHE/rank7"
    for rank in 0 1 2 3 4 5 6 7; do
      ln -sfn "$RQ_CACHE/semantic_codes.sqlite3" "$RQ_CACHE/rank$rank/semantic_codes.sqlite3"
    done
    extra_args+=(--rq-table-dir "$TABLE" --rq-cache-dir "$RQ_CACHE")
  fi
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$TORCHRUN" \
    --standalone --nproc_per_node=8 \
    examples/run_scratch_pt.py \
    --mode "$mode" \
    --data-dir "$DATA" \
    --tokenizer "$TOKENIZER" \
    --output-dir "$mode_out" \
    --train-tokens 1000000000 \
    --sequence-length 2048 \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 64 \
    --learning-rate 3e-5 \
    --eval-steps 95 \
    --checkpoint-steps 239 \
    --num-workers 2 \
    --attn-implementation eager \
    "${extra_args[@]}" \
    >> "$log" 2>&1
  echo "[scratch queue] finished mode=$mode" | tee -a "$log"
done

echo "SCRATCH_PT_1B_QUEUE_DONE" | tee -a "$OUT/queue.log"
