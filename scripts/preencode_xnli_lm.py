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

from engram_peft.rq_hashing import RQNgramMapping


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


def collect_misses(
    texts: list[str],
    tokenizer: Any,
    compressor: Any,
    table_keys: dict[int, np.ndarray],
    orders: list[int],
    max_length: int,
    base: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return per-order (unique_miss_keys, first_original_window) vectors.

    Vectorized replacement for the old row/col Python loop: every context
    position in a text batch is expanded to all n-gram windows, matched against
    the offline table with searchsorted, and only genuinely unseen keys survive.
    """
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="np",
    )
    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    attention = np.asarray(encoded["attention_mask"], dtype=np.uint8)
    compressed = np.asarray(compressor.map_ids(input_ids), dtype=np.int64)

    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for order in orders:
        batch, length = compressed.shape
        if length < order:
            result[order] = (
                np.empty(0, dtype=np.int64),
                np.empty((0, order), dtype=np.int64),
            )
            continue
        windows = np.lib.stride_tricks.sliding_window_view(
            compressed, window_shape=order, axis=1
        )  # [batch, length-order+1, order]
        flat_windows = windows.reshape(-1, order)
        keys = np.zeros(flat_windows.shape[0], dtype=np.int64)
        for offset in range(order):
            keys = keys * base + flat_windows[..., offset].astype(np.int64)
        # valid context positions: token t has a real next token (attention[t+1])
        valid = np.zeros((batch, length), dtype=bool)
        valid[:, order - 1 : -1] = (
            attention[:, order - 1 : -1].astype(bool)
            & attention[:, order:].astype(bool)
        )
        flat_valid = valid[:, order - 1 :].reshape(-1)
        keys = keys[flat_valid]
        flat_windows = flat_windows[flat_valid]

        pos = np.searchsorted(table_keys[order], keys)
        hit = (pos < len(table_keys[order])) & (
            table_keys[order][np.clip(pos, 0, len(table_keys[order]) - 1)] == keys
        )
        keys = keys[~hit]
        flat_windows = flat_windows[~hit]
        if keys.size == 0:
            result[order] = (
                np.empty(0, dtype=np.int64),
                np.empty((0, order), dtype=np.int64),
            )
            continue

        # Keep the first window per unique key by sorting on (key, position).
        order_ids = np.arange(keys.size)
        sort_idx = np.lexsort((order_ids, keys))
        sorted_keys = keys[sort_idx]
        sorted_windows = flat_windows[sort_idx]
        first_mask = np.ones(keys.size, dtype=bool)
        first_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
        unique_keys = sorted_keys[first_mask]
        first_windows = sorted_windows[first_mask]
        result[order] = (unique_keys, first_windows)
    return result


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

    from engram_peft.compression import CompressedTokenizer

    compressor = CompressedTokenizer(tokenizer=tokenizer)
    base = int(compressor.compressed_vocab_size) + 1

    if args.mode == "xnli":
        rows = iter_xnli_rows(args.seed)
    else:
        rows = iter_lm_rows(args.seed, args.lm_rows)

    # Batch texts and flush misses every N rows so GPU memory and the SQLite
    # write set stay bounded while still vectorizing tokenization/quantization.
    text_batch: list[str] = []
    rows_seen = 0
    # Large flush batches amortize the embedder forward and SQLite writes;
    # 20k rows keeps peak GPU/memory reasonable for XNLI (392k rows).
    flush_every = 20_000

    def flush() -> None:
        if not text_batch:
            return
        misses = collect_misses(
            text_batch,
            tokenizer,
            compressor,
            table_keys,
            orders,
            args.max_length,
            base,
        )
        for order in orders:
            keys, windows = misses[order]
            if keys.size == 0:
                continue
            codes = mapping._encode_missing(order, keys, windows)
            conn.executemany(
                "INSERT OR IGNORE INTO codes(n, key, code) VALUES (?, ?, ?)",
                [
                    (order, int(key), sqlite3.Binary(code.astype(np.uint16).tobytes()))
                    for key, code in zip(keys, codes, strict=True)
                ],
            )
            conn.commit()
            print(
                f"[preencode] rows={rows_seen} order={order} wrote={len(codes)}",
                flush=True,
            )

    for text in rows:
        text_batch.append(text)
        rows_seen += 1
        if rows_seen % flush_every == 0:
            flush()
            text_batch.clear()
    flush()
    print("[preencode] done", flush=True)


if __name__ == "__main__":
    main()
