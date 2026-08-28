#!/usr/bin/env python3
"""Materialize a deterministic FineWeb-Edu token stream for scratch PT.

The train and validation streams are carved from one shuffled stream before
tokenization, so every comparison condition sees exactly the same examples.
"""

from __future__ import annotations

import argparse
import array
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--eval-tokens", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--shuffle-buffer", type=int, default=100_000)
    parser.add_argument("--flush-tokens", type=int, default=1_000_000)
    return parser.parse_args()


class TokenWriter:
    def __init__(self, path: Path, flush_tokens: int) -> None:
        self.path = path
        self.file = path.open("wb")
        self.buffer = array.array("I")
        self.flush_tokens = flush_tokens
        self.count = 0

    def append(self, token_ids: list[int]) -> None:
        self.buffer.extend(token_ids)
        self.count += len(token_ids)
        if len(self.buffer) >= self.flush_tokens:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.buffer.tofile(self.file)
            self.buffer = array.array("I")
            self.file.flush()

    def close(self) -> None:
        self.flush()
        self.file.close()


def main() -> None:
    args = parse_args()
    if args.train_tokens <= 0 or args.eval_tokens <= 0:
        raise ValueError("train/eval token budgets must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("the tokenizer must define eos_token_id")

    train_writer = TokenWriter(args.output_dir / "train.bin", args.flush_tokens)
    eval_writer = TokenWriter(args.output_dir / "eval.bin", args.flush_tokens)
    train_remaining = args.train_tokens
    eval_remaining = args.eval_tokens
    documents = 0
    train_documents = 0
    eval_documents = 0

    print(
        f"Loading {args.dataset}/{args.dataset_config}; target="
        f"{args.train_tokens:,} train + {args.eval_tokens:,} eval tokens",
        flush=True,
    )
    stream = load_dataset(
        args.dataset,
        args.dataset_config,
        split="train",
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    try:
        for example in stream:
            text = str(example.get("text", ""))
            if not text:
                continue
            ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
            ids.append(int(eos_id))
            if not ids:
                continue
            documents += 1

            if train_remaining > 0:
                take = min(train_remaining, len(ids))
                train_writer.append(ids[:take])
                train_remaining -= take
                if take == len(ids):
                    train_documents += 1
                else:
                    # A document crossing the split boundary is never reused.
                    continue
            elif eval_remaining > 0:
                take = min(eval_remaining, len(ids))
                eval_writer.append(ids[:take])
                eval_remaining -= take
                if take == len(ids):
                    eval_documents += 1

            if documents % 1000 == 0:
                print(
                    f"docs={documents:,} train={train_writer.count:,}/"
                    f"{args.train_tokens:,} eval={eval_writer.count:,}/"
                    f"{args.eval_tokens:,}",
                    flush=True,
                )
            if train_remaining == 0 and eval_remaining == 0:
                break
    finally:
        train_writer.close()
        eval_writer.close()

    if train_remaining or eval_remaining:
        raise RuntimeError(
            f"stream ended early: train_remaining={train_remaining}, "
            f"eval_remaining={eval_remaining}"
        )

    metadata: dict[str, Any] = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "train_tokens": train_writer.count,
        "eval_tokens": eval_writer.count,
        "train_documents": train_documents,
        "eval_documents": eval_documents,
        "source_documents_seen": documents,
        "dtype": "uint32",
        "format": "contiguous_token_stream_v1",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
