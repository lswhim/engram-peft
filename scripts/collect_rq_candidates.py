#!/usr/bin/env python3
"""Collect unique-per-batch OOV RQ candidates into partitioned binary files.

The files are intentionally append-only.  A later sort/dedup pass performs the
global key deduplication without keeping the full 1B-token key set in RAM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from engram_peft.compression import CompressedTokenizer


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--table-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sequence-length", type=int, default=2048)
    p.add_argument("--batch-rows", type=int, default=512)
    p.add_argument("--start-row", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--partitions", type=int, default=64)
    p.add_argument("--stream", choices=("train", "eval"), required=True)
    return p.parse_args()


def poly_keys(windows: np.ndarray, base: int) -> np.ndarray:
    keys = np.zeros(windows.shape[:-1], dtype=np.int64)
    for j in range(windows.shape[-1]):
        keys = keys * base + windows[..., j].astype(np.int64)
    return keys


def record_dtype(n: int) -> np.dtype:
    return np.dtype([("key", "<i8"), ("window", "<i8", (n,))])


def main() -> None:
    a = args()
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    compressor = CompressedTokenizer(tokenizer=tokenizer)
    meta = __import__("json").loads((a.table_dir / "meta.json").read_text())
    base = int(meta["base"])
    ngram_sizes = [int(n) for n in meta["ngram_sizes"]]
    static_keys = {
        n: np.load(a.table_dir / f"keys_{n}.npy", mmap_mode="r") for n in ngram_sizes
    }
    data_path = a.data_dir / f"{a.stream}.bin"
    tokens = np.memmap(data_path, mode="r", dtype=np.uint32)
    total_rows = (len(tokens) - 1) // a.sequence_length
    first_row = min(max(a.start_row, 0), total_rows)
    stop_row = total_rows if a.max_rows <= 0 else min(total_rows, first_row + a.max_rows)
    starts = list(range(first_row, stop_row))
    rank_label = a.start_row // max(a.max_rows, 1)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[tuple[int, int], object] = {}
    for n in ngram_sizes:
        dtype = record_dtype(n)
        for part in range(a.partitions):
            path = a.output_dir / f"{a.stream}_rank{rank_label}_n{n}_p{part}.bin"
            handles[n, part] = path.open("ab")

    try:
        for batch_start in range(0, len(starts), a.batch_rows):
            row_ids = starts[batch_start : batch_start + a.batch_rows]
            begin = row_ids[0] * a.sequence_length
            end = (row_ids[-1] + 1) * a.sequence_length
            original = np.asarray(tokens[begin:end], dtype=np.int64).reshape(
                len(row_ids), a.sequence_length
            )
            compressed = compressor.map_ids(original)
            for n in ngram_sizes:
                pad = np.full((len(row_ids), n - 1), compressor.map_id(int(tokenizer.pad_token_id)), dtype=np.int64)
                padded = np.concatenate([pad, compressed.astype(np.int64)], axis=1)
                windows = np.stack(
                    [padded[:, i : i + a.sequence_length] for i in range(n)], axis=-1
                )
                keys = poly_keys(windows, base)
                pos = np.searchsorted(static_keys[n], keys)
                hit = static_keys[n][np.clip(pos, 0, static_keys[n].size - 1)] == keys
                missing = ~hit
                if not missing.any():
                    continue
                original_pad = np.full(
                    (len(row_ids), n - 1), int(meta.get("pad_token_id", 0)), dtype=np.int64
                )
                original_padded = np.concatenate([original_pad, original], axis=1)
                original_windows = np.stack(
                    [original_padded[:, i : i + a.sequence_length] for i in range(n)], axis=-1
                )
                flat_keys = keys[missing].reshape(-1)
                flat_windows = original_windows[missing].reshape(-1, n)
                unique_keys, first = np.unique(flat_keys, return_index=True)
                records = np.empty(len(unique_keys), dtype=record_dtype(n))
                records["key"] = unique_keys
                records["window"] = flat_windows[first]
                for part in np.unique(unique_keys % a.partitions):
                    selected = records[records["key"] % a.partitions == part]
                    if len(selected):
                        selected.tofile(handles[n, int(part)])
            if (batch_start // a.batch_rows + 1) % 10 == 0 or batch_start + a.batch_rows >= len(starts):
                print(
                    f"[rq collect] stream={a.stream} rank={rank_label} rows="
                    f"{min(batch_start + a.batch_rows, len(starts)):,}/{len(starts):,}",
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    main()
