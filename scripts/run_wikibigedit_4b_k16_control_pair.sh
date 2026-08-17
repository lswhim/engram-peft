#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
ROOT=/anguszhang-cfs-nj/seokliu_workspace/engram_multilingual

cd "$ROOT"
bash scripts/run_wikibigedit_4b_k16_worker.sh "$GPU_ID" arithmetic
bash scripts/run_wikibigedit_4b_k16_worker.sh "$GPU_ID" shuffled
