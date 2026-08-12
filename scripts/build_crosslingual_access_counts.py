#!/usr/bin/env python
"""Count exact RQ-table accesses under XNLI or PAWS-X English training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_lm_slice_manifest import poly_keys, table_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("xnli", "pawsx"), required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def valid_access_mask(attention: np.ndarray, order: int) -> np.ndarray:
    """Positions whose Engram value can contribute to a later supervised token."""
    mask = np.zeros_like(attention, dtype=bool)
    if attention.shape[1] <= order:
        return mask
    mask[:, order - 1 : -1] = (
        attention[:, order - 1 : -1].astype(bool)
        & attention[:, order:].astype(bool)
    )
    return mask


def accumulate_access_counts(
    compressed: np.ndarray,
    attention: np.ndarray,
    *,
    base: int,
    table_keys: dict[int, np.ndarray],
    counts: dict[int, np.ndarray],
) -> None:
    for order, keys_table in table_keys.items():
        keys = poly_keys(compressed, order, base)
        positions, covered = table_positions(keys, keys_table)
        covered &= valid_access_mask(attention, order)
        np.add.at(counts[order], positions[covered], 1)


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def main() -> None:
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from engram_peft.compression import CompressedTokenizer
    from examples.benchmarks.xtreme_xnli import CausalLabelCollator

    args = parse_args()
    meta = json.loads((args.table_dir / "meta.json").read_text(encoding="utf-8"))
    orders = tuple(int(value) for value in meta["ngram_sizes"])
    table_keys = {
        order: np.load(args.table_dir / f"keys_{order}.npy", allow_pickle=False)
        for order in orders
    }
    counts = {
        order: np.zeros(len(table_keys[order]), dtype=np.int64)
        for order in orders
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    compressor = CompressedTokenizer(tokenizer=tokenizer)
    base = int(compressor.compressed_vocab_size) + 1
    if int(meta["base"]) != base:
        raise ValueError(f"table base={meta['base']} does not match tokenizer base={base}")

    if args.benchmark == "xnli":
        from examples.benchmarks.xtreme_xnli import tokenize_xnli_train

        raw = load_dataset("facebook/xnli", "en", split="train")
        tokenized = tokenize_xnli_train(
            raw, tokenizer, args.max_length, args.num_proc
        )
    else:
        from examples.benchmarks.xtreme_pawsx import tokenize_pawsx_train

        raw = load_dataset(
            "google-research-datasets/paws-x", "en", split="train"
        )
        tokenized = tokenize_pawsx_train(
            raw, tokenizer, args.max_length, args.num_proc
        )

    loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=CausalLabelCollator(tokenizer),
    )
    examples = 0
    for batch_index, batch in enumerate(loader):
        ids = np.asarray(batch["input_ids"].numpy(), dtype=np.int64)
        attention = np.asarray(batch["attention_mask"].numpy(), dtype=np.uint8)
        compressed = np.asarray(compressor.map_ids(ids), dtype=np.int64)
        accumulate_access_counts(
            compressed,
            attention,
            base=base,
            table_keys=table_keys,
            counts=counts,
        )
        examples += len(ids)
        if batch_index % 100 == 0:
            print(
                f"[{args.benchmark} access] {examples}/{len(tokenized)}",
                flush=True,
            )

    arrays = {f"train_access_count_{order}": counts[order] for order in orders}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)

    payload: dict[str, Any] = {
        "status": "complete",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "benchmark": args.benchmark,
        "protocol": "full English source train, one epoch, all valid causal context positions",
        "examples": examples,
        "max_length": args.max_length,
        "table_dir": str(args.table_dir),
        "manifest": str(args.output),
        "orders": {
            str(order): {
                "table_rows": len(counts[order]),
                "accessed_rows": int(np.count_nonzero(counts[order])),
                "total_accesses": int(counts[order].sum()),
                "sha256": digest(counts[order]),
            }
            for order in orders
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary_tmp = args.summary.with_suffix(args.summary.suffix + ".tmp")
    summary_tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(summary_tmp, args.summary)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
