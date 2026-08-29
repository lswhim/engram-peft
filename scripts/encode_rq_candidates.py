#!/usr/bin/env python3
"""Encode globally deduplicated RQ candidates on one GPU."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from engram_peft.rq_hashing import RQNgramMapping


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--table-dir", type=Path, required=True)
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed-db", type=Path)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--embed-batch-size", type=int, default=1024)
    p.add_argument("--chunk-size", type=int, default=16384)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    known: dict[int, np.ndarray] = {}
    if a.seed_db and a.seed_db.exists():
        conn = sqlite3.connect(f"file:{a.seed_db}?mode=ro", uri=True)
        for n in (2, 3):
            known[n] = np.fromiter(
                (int(row[0]) for row in conn.execute("SELECT key FROM codes WHERE n=? ORDER BY key", (n,))),
                dtype=np.int64,
            )
        conn.close()

    mapping = RQNgramMapping(
        table_dir=str(a.table_dir),
        cache_dir=str(a.output_dir),
        embed_device=a.device,
        embed_batch_size=a.embed_batch_size,
    )
    for path in sorted(a.candidate_dir.glob("n*_p*.npz")):
        partition = int(path.stem.rsplit("_p", 1)[1])
        if partition % a.world_size != a.gpu:
            continue
        n = int(path.stem.split("_", 1)[0][1:])
        payload = np.load(path)
        keys = np.asarray(payload["keys"], dtype=np.int64)
        windows = np.asarray(payload["windows"], dtype=np.int64)
        if n in known and len(known[n]):
            pos = np.searchsorted(known[n], keys)
            keep = (pos == len(known[n])) | (known[n][np.clip(pos, 0, len(known[n]) - 1)] != keys)
            keys, windows = keys[keep], windows[keep]
        if len(keys) == 0:
            continue
        for start in range(0, len(keys), a.chunk_size):
            stop = min(start + a.chunk_size, len(keys))
            mapping._encode_missing(n, keys[start:stop], windows[start:stop])
            print(
                f"[rq encode] gpu={a.gpu} n={n} partition={partition} "
                f"{stop:,}/{len(keys):,}",
                flush=True,
            )


if __name__ == "__main__":
    main()
