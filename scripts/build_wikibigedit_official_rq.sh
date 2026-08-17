#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_collision}"
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
BASE=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
EMBEDDER=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-Embedding-4B
MANIFEST=data/semantic_memory/wikibigedit_official_chronological.jsonl
TABLE=rq_tables/wikibigedit_official502k_qwen3emb4b_M8K16
SHUFFLED=${TABLE}_runtime_shuffled_seed42

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT/src:$ROOT"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ "$(wc -l < "$MANIFEST")" -ne 502382 ]]; then
  echo "official manifest must contain exactly 502382 train edits" >&2
  exit 2
fi

if [[ ! -f "$TABLE/meta.json" ]]; then
  "$PY" -u scripts/build_rq_table.py \
    --data_files "$MANIFEST" --text_columns prompt target --num_docs 502382 \
    --base_tokenizer "$BASE" --embedder "$EMBEDDER" \
    --num_levels 8 --codebook_size 16 --projection_dim 0 \
    --max_doc_tokens 128 --max_ngrams_per_size 500000 --min_count 1 \
    --embed_batch_size 256 --rq_train_threads 32 --seed 0 \
    --output_dir "$TABLE"
fi

if [[ ! -f "$SHUFFLED/meta.json" ]]; then
  "$PY" scripts/make_runtime_shuffled_rq.py \
    --source "$TABLE" --output "$SHUFFLED" --seed 42
fi

"$PY" - <<'PY'
import json
from pathlib import Path

import numpy as np

for directory in (
    Path("rq_tables/wikibigedit_official502k_qwen3emb4b_M8K16"),
    Path("rq_tables/wikibigedit_official502k_qwen3emb4b_M8K16_runtime_shuffled_seed42"),
):
    meta = json.loads((directory / "meta.json").read_text())
    print(directory, meta)
    for order in (2, 3):
        keys = np.load(directory / f"keys_{order}.npy", mmap_mode="r")
        codes = np.load(directory / f"codes_{order}.npy", mmap_mode="r")
        if len(keys) != len(codes) or codes.shape[1] != 8:
            raise RuntimeError(f"invalid table arrays for {directory}, order={order}")
        print(order, keys.shape, codes.shape)
PY
