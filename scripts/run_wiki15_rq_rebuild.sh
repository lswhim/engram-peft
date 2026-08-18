#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-3}"
ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_learned}"
PY=/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python
MODEL=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-1.7B-Base
EMBEDDER=/anguszhang-cfs-nj/seokliu_workspace/models/Qwen3-Embedding-0.6B
CORPUS="$ROOT/data/wiki15_wikipedia_1500.jsonl"
TABLE="$ROOT/rq_tables/wiki15_qwen3_06b_M8K256_500k_v2"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT/src:$ROOT"
export TOKENIZERS_PARALLELISM=false
export https_proxy=http://star-proxy.oa.com:3128
export http_proxy=http://star-proxy.oa.com:3128
export no_proxy=".woa.com,.oa.com,.tencentcos.cn,localhost,127.0.0.1"

if [[ ! -s "$CORPUS" ]]; then
  "$PY" -u scripts/prepare_wiki15_corpus.py \
    --output "$CORPUS" --docs-per-language 1500
fi

if [[ ! -s "$TABLE/rq_2.faiss" || ! -s "$TABLE/rq_3.faiss" ]]; then
  "$PY" -u scripts/build_rq_table.py \
    --data_files "$CORPUS" --split train --text_column text \
    --num_docs 22500 --max_doc_tokens 512 \
    --base_tokenizer "$MODEL" --embedder "$EMBEDDER" \
    --ngram_sizes 2 3 --num_levels 8 --codebook_size 256 \
    --max_ngrams_per_size 500000 --min_count 1 \
    --embed_batch_size 256 --rq_train_threads 32 \
    --output_dir "$TABLE"
fi

"$PY" - "$TABLE" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

table = Path(sys.argv[1])
meta = json.loads((table / "meta.json").read_text())
assert meta["ngram_sizes"] == [2, 3]
assert meta["num_levels"] == 8
assert meta["codebook_size"] == 256
for n in (2, 3):
    keys = np.load(table / f"keys_{n}.npy", mmap_mode="r")
    codes = np.load(table / f"codes_{n}.npy", mmap_mode="r")
    assert keys.ndim == 1 and codes.shape == (len(keys), 8)
    assert (table / f"rq_{n}.faiss").stat().st_size > 0
print(json.dumps({"status": "complete", "table": str(table), "rows": {
    str(n): int(len(np.load(table / f"keys_{n}.npy", mmap_mode="r"))) for n in (2, 3)
}}, indent=2))
PY
