#!/usr/bin/env python3
"""Globally deduplicate partitioned RQ candidates by compressed n-gram key."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--partitions", type=int, default=64)
    p.add_argument("--ngram-sizes", type=int, nargs="+", default=[2, 3])
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for n in a.ngram_sizes:
        dtype = np.dtype([("key", "<i8"), ("window", "<i8", (n,))])
        for part in range(a.partitions):
            paths = sorted(a.input_dir.glob(f"*_n{n}_p{part}.bin"))
            if not paths:
                continue
            chunks = [np.fromfile(path, dtype=dtype) for path in paths]
            records = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            if len(records) == 0:
                continue
            order = np.argsort(records["key"], kind="mergesort")
            records = records[order]
            keep = np.empty(len(records), dtype=bool)
            keep[0] = True
            keep[1:] = records["key"][1:] != records["key"][:-1]
            records = records[keep]
            np.savez(
                a.output_dir / f"n{n}_p{part}.npz",
                keys=records["key"],
                windows=records["window"],
            )
            print(
                f"[rq dedup] n={n} partition={part} "
                f"raw={len(order):,} unique={len(records):,}",
                flush=True,
            )


if __name__ == "__main__":
    main()
