#!/usr/bin/env bash
set -euo pipefail

ROOT="${ENGRAM_ROOT:-/anguszhang-cfs-nj/seokliu_workspace/engram_learned}"
V2="$ROOT/rq_tables/wiki15_qwen3_06b_M8K256_500k_v2"
LEGACY=/anguszhang-cfs-nj/seokliu_workspace/engram/rq_tables/wiki15_qwen3_06b_M8K256_500k
BACKUP=/anguszhang-cfs-nj/seokliu_workspace/engram/rq_tables/wiki15_qwen3_06b_M8K256_500k_legacy_static_20260818

until [[ -s "$V2/meta.json" && -s "$V2/rq_2.faiss" && -s "$V2/rq_3.faiss" ]]; do
  if ! pgrep -f "run_wiki15_rq_rebuild.sh 3" >/dev/null; then
    echo "wiki15 V2 builder exited before producing a complete table" >&2
    exit 3
  fi
  sleep 30
done

if [[ -L "$LEGACY" ]]; then
  current="$(readlink -f "$LEGACY")"
  if [[ "$current" != "$(readlink -f "$V2")" ]]; then
    echo "legacy table path already points elsewhere: $current" >&2
    exit 4
  fi
elif [[ -d "$LEGACY" ]]; then
  if [[ -e "$BACKUP" ]]; then
    echo "backup already exists; refusing to move $LEGACY" >&2
    exit 5
  fi
  mv "$LEGACY" "$BACKUP"
  ln -s "$V2" "$LEGACY"
else
  ln -s "$V2" "$LEGACY"
fi

echo "[activate] $LEGACY -> $(readlink -f "$LEGACY")"
echo "[backup] $BACKUP"
exec bash "$ROOT/scripts/run_learned_router_formal_queue.sh" 3 3
