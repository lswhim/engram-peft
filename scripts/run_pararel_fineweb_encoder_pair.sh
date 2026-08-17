#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
EMBED_SIZE="$2"
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram_multilingual

cd "$ROOT"
bash scripts/run_pararel_fineweb_k16_worker.sh "$GPU_ID" semantic "$EMBED_SIZE"
bash scripts/run_pararel_fineweb_k16_worker.sh "$GPU_ID" shuffled "$EMBED_SIZE"
