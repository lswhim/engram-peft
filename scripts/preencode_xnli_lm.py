"""Pre-encode XNLI and FineWeb LM n-grams into RQ cache SQLite files.

Training still performs O(1) table lookup for offline-covered n-grams, but any
unseen n-gram triggers a frozen Qwen3-Embedding -> FAISS RQ forward on the fly.
XNLI (English MNLI) and FineWeb LM training text share little vocabulary with the
FineWeb-built RQ table, so the dynamic path dominates step time. This script
walks the exact training rows used by the benchmark workers and writes all
missing codes ahead of time so training starts with a warm cache.

It intentionally pre-encodes *train* text only; evaluation remains a genuine
dynamic-OOV path, which is the paper claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from engram_peft import RQNgramMapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--mode", choices=("xnli", "lm"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lm-rows", type=int, default=100_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def poly_keys(compressed: np.ndarray, order: int, base: int) -> np.ndarray:
    """Return suffix polynomial keys for every n-gram ending at position t."""
    batch, length = compressed.shape
    result = np.zeros((batch, length), dtype=np.int64)
    if length < order:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(
        compressed, window_shape=order, axis=1
    )
    values = np.zeros(windows.shape[:2], dtype=np.int64)
    for offset in range(order):
        values = values * base + windows[..., offset].astype(np.int64)
    result[:, order - 1 :] = values
    return result


def valid_mask(attention: np.ndarray, order: int) -> np.ndarray:
    mask = np.zeros_like(attention, dtype=bool)
    if attention.shape[1] <= order:
        return mask
    mask[:, order - 1 : -1] = (
        attention[:, order - 1 : -1].astype(bool)
        & attention[:, order:].astype(bool)
    )
    return mask


def iter_xnli_rows(seed: int):
    train = load_dataset("facebook/xnli", "en", split="train")
    # Worker calls encode_train_example(example, tokenizer, max_length); reproduce
    # the same prompt text used there.
    from examples.benchmarks.xtreme_xnli import format_xnli_prompt

    for example in train:
        yield format_xnli_prompt(str(example["premise"]), str(example["hypothesis"]))


def iter_lm_rows(seed: int, rows: int):
    address_reserve_rows = 6_000
    raw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    ).skip(address_reserve_rows).shuffle(
        seed=seed, buffer_size=max(rows + 200, 20_000)
    )
    for example in islice(raw, rows + 200):
        text = example.get("text")
        if text:
            yield str(text)


def main() -> None:
    args = parse_args()
    meta = json.loads((Path(args.table_dir) / "meta.json").read_text(encoding="utf-8"))
    orders = [int(value) for value in meta["ngram_sizes"]]
    table_keys = {
        order: np.load(Path(args.table_dir) / f"keys_{order}.npy", allow_pickle=False)
        for order in orders
    }

    tokenizer = AutoTokenizer.from_pretrained(
        meta["base_tokenizer"], trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    from engram_peft.compression import CompressedTokenizer

    compressor = CompressedTokenizer(tokenizer=tokenizer)
    base = int(compressor.compressed_vocab_size) + 1

    if args.mode == "xnli":
        rows = iter_xnli_rows(args.seed)
    else:
        rows = iter_lm_rows(args.seed, args.lm_rows)

    mapping = RQNgramMapping(
        table_dir=args.table_dir,
        cache_dir=args.cache_dir,
        embed_device=args.device,
        embed_batch_size=args.batch_size,
    )
    mapping._load_encoder()

    cache_path = Path(args.cache_dir) / "semantic_codes.sqlite3"
    conn = sqlite3.connect(cache_path)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS codes "
        "(n INTEGER NOT NULL, key INTEGER NOT NULL, code BLOB NOT NULL, "
        "PRIMARY KEY (n, key))"
    )

    pending_keys: dict[int, list[int]] = {order: [] for order in orders}
    pending_windows: dict[int, list[np.ndarray]] = {order: [] for order in orders}
    total = 0
    for text in rows:
        ids = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
        )
        input_ids = np.asarray(ids["input_ids"], dtype=np.int64)
        attention = np.asarray(ids["attention_mask"], dtype=np.uint8)
        compressed = np.asarray(compressor.map_ids(input_ids), dtype=np.int64)
        for order in orders:
            keys = poly_keys(compressed, order, base)
            valid = valid_mask(attention, order)
            seen: set[int] = set()
            for row in range(keys.shape[0]):
                for col in np.flatnonzero(valid[row]):
                    key = int(keys[row, col])
                    if key in seen:
                        continue
                    seen.add(key)
                    pos = np.searchsorted(table_keys[order], key)
                    if pos < len(table_keys[order]) and table_keys[order][pos] == key:
                        continue  # offline covered
                    pending_keys[order].append(key)
                    window = compressed[row, col - order + 1 : col + 1]
                    pending_windows[order].append(window)
        total += 1
        if total % 2000 == 0:
            print(f"[preencode] rows={total}", flush=True)

    print(f"[preencode] collecting {len(pending_keys[2]) + len(pending_keys[3])} misses", flush=True)
    for order in orders:
        keys = np.asarray(pending_keys[order], dtype=np.int64)
        if keys.size == 0:
            continue
        windows = np.stack(pending_windows[order])
        codes = mapping._encode_missing(order, keys, windows)
        conn.executemany(
            "INSERT OR IGNORE INTO codes(n, key, code) VALUES (?, ?, ?)",
            [
                (order, int(key), sqlite3.Binary(code.astype(np.uint16).tobytes()))
                for key, code in zip(keys, codes, strict=True)
            ],
        )
        conn.commit()
        print(f"[preencode] order={order} wrote={len(codes)}", flush=True)
    print("[preencode] done", flush=True)


if __name__ == "__main__":
    main()
