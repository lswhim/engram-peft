#!/usr/bin/env python
"""Small end-to-end validator for cold and warm lazy semantic RQ lookup."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from transformers import AutoTokenizer

from engram_peft.compression import CompressedTokenizer
from engram_peft.rq_hashing import RQNgramMapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    compressor = CompressedTokenizer(tokenizer=tokenizer)
    texts = [
        "This completely novel phrase validates the lazy semantic cache.",
        "Another unseen multilingual sequence 北京 validates first access encoding.",
    ]
    original = tokenizer(texts, padding=True, return_tensors="np")["input_ids"]
    compressed = compressor.map_ids(original)
    mapping = RQNgramMapping(
        table_dir=args.table,
        pad_id=compressor.map_id(tokenizer.pad_token_id or tokenizer.eos_token_id),
        cache_dir=args.cache,
        embed_device=args.device,
        embed_batch_size=64,
    )
    started = time.time()
    cold_codes = mapping.hash(compressed, original_ids=original)[0]
    cold_seconds = time.time() - started
    cached_rows = mapping._cache.execute("SELECT COUNT(*) FROM codes").fetchone()[0]
    started = time.time()
    warm_codes = mapping.hash(compressed, original_ids=original)[0]
    warm_seconds = time.time() - started
    print(
        json.dumps(
            {
                "shape": list(cold_codes.shape),
                "cold_warm_bitwise_equal": bool(np.array_equal(cold_codes, warm_codes)),
                "cached_rows": cached_rows,
                "cold_seconds": cold_seconds,
                "warm_seconds": warm_seconds,
                "min_code": int(cold_codes.min()),
                "max_code": int(cold_codes.max()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
