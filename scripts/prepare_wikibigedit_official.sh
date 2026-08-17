#!/usr/bin/env bash
set -euo pipefail

ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_collision}"
PY="${ENGRAM_PYTHON:-/anguszhang-cfs-nj/seokliu_workspace/miniconda3/envs/engram/bin/python}"
DATA_DIR="$ROOT/data/wikibigedit_official"
OUTPUT="$ROOT/data/semantic_memory/wikibigedit_official_chronological.jsonl"
BASE_URL=https://huggingface.co/datasets/lukasthede/WikiBigEdit/resolve/main

FILES=(
  wiki_big_edit_20240201_20240220.json
  wiki_big_edit_20240220_20240301.json
  wiki_big_edit_20240301_20240320.json
  wiki_big_edit_20240320_20240401.json
  wiki_big_edit_20240401_20240501.json
  wiki_big_edit_20240501_20240601.json
  wiki_big_edit_20240601_20240620.json
  wiki_big_edit_20240620_20240701.json
)

mkdir -p "$DATA_DIR" "$(dirname "$OUTPUT")"
paths=()
for name in "${FILES[@]}"; do
  target="$DATA_DIR/$name"
  if [[ ! -s "$target" ]]; then
    echo "DOWNLOAD $name"
    curl -L --fail --retry 5 --silent --show-error \
      "$BASE_URL/$name?download=true" -o "$target.tmp"
    mv "$target.tmp" "$target"
  fi
  echo "READY $name $(wc -c < "$target") bytes"
  paths+=("$target")
done

cd "$ROOT"
"$PY" examples/prepare_wikibigedit_official.py \
  --files "${paths[@]}" \
  --output "$OUTPUT"
wc -l "$OUTPUT"
