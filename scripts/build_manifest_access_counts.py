#!/usr/bin/env python
"""Count exact frozen-table accesses for a canonical-write JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.build_crosslingual_access_counts import (
    accumulate_access_counts,
    valid_access_mask,
)
from scripts.build_lm_slice_manifest import poly_keys, table_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prompt-format", choices=("plain", "qa"), default="plain")
    parser.add_argument(
        "--source",
        choices=("canonical", "queries"),
        default="canonical",
        help="Count canonical training pairs or every scorable evaluation query.",
    )
    return parser.parse_args()


def canonical_token_ids(
    tokenizer: object,
    prompt: str,
    target: str,
    max_length: int,
    prompt_format: str,
) -> list[int]:
    formatted = f"Q: {prompt} A:" if prompt_format == "qa" else prompt
    prompt_ids = tokenizer(formatted, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(" " + target.strip(), add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_ids[: max(0, max_length - len(target_ids))]
    target_ids = target_ids[: max_length - len(prompt_ids)]
    return list(prompt_ids) + list(target_ids)


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def main() -> None:
    from transformers import AutoTokenizer
    from engram_peft.compression import CompressedTokenizer

    args = parse_args()
    meta = json.loads((args.table_dir / "meta.json").read_text(encoding="utf-8"))
    orders = tuple(int(value) for value in meta["ngram_sizes"])
    keys = {
        order: np.load(args.table_dir / f"keys_{order}.npy", allow_pickle=False)
        for order in orders
    }
    counts = {order: np.zeros(len(keys[order]), dtype=np.int64) for order in orders}
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    compressor = CompressedTokenizer(tokenizer=tokenizer)
    base = int(compressor.compressed_vocab_size) + 1
    if base != int(meta["base"]):
        raise ValueError(f"table base={meta['base']} does not match tokenizer base={base}")

    rows: list[list[int]] = []
    cases = 0
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            cases += 1
            pairs = [(str(case["prompt"]), str(case["target"]))]
            if args.source == "queries":
                pairs = [
                    (str(query["prompt"]), str(query["answers"][0]))
                    for query in case.get("queries", [])
                    if query.get("answers") and str(query["answers"][0]).strip()
                ]
            rows.extend(
                canonical_token_ids(
                    tokenizer, prompt, target, args.max_length, args.prompt_format
                )
                for prompt, target in pairs
            )
            if args.subset and cases >= args.subset:
                break
    pad_id = int(tokenizer.pad_token_id)
    candidate_accesses = {order: 0 for order in orders}
    covered_accesses = {order: 0 for order in orders}
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        width = args.max_length
        ids = np.full((len(batch_rows), width), pad_id, dtype=np.int64)
        attention = np.zeros((len(batch_rows), width), dtype=np.uint8)
        for row_index, token_ids in enumerate(batch_rows):
            length = min(len(token_ids), width)
            ids[row_index, :length] = token_ids[:length]
            attention[row_index, :length] = 1
        compressed = np.asarray(compressor.map_ids(ids), dtype=np.int64)
        for order, table_keys in keys.items():
            batch_keys = poly_keys(compressed, order, base)
            _, covered = table_positions(batch_keys, table_keys)
            valid = valid_access_mask(attention, order)
            candidate_accesses[order] += int(valid.sum())
            covered_accesses[order] += int((covered & valid).sum())
        accumulate_access_counts(
            compressed,
            attention,
            base=base,
            table_keys=keys,
            counts=counts,
        )
        if start == 0 or start + len(batch_rows) == len(rows) or start % 10000 == 0:
            print(f"[manifest access] {start + len(batch_rows)}/{len(rows)}", flush=True)

    arrays = {f"train_access_count_{order}": counts[order] for order in orders}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)
    payload = {
        "status": "complete",
        "manifest": str(args.manifest),
        "cases": cases,
        "pairs": len(rows),
        "source": args.source,
        "max_length": args.max_length,
        "prompt_format": args.prompt_format,
        "table_dir": str(args.table_dir),
        "orders": {
            str(order): {
                "table_rows": len(counts[order]),
                "accessed_rows": int(np.count_nonzero(counts[order])),
                "total_accesses": int(counts[order].sum()),
                "candidate_accesses": candidate_accesses[order],
                "covered_accesses": covered_accesses[order],
                "offline_coverage": (
                    covered_accesses[order] / candidate_accesses[order]
                    if candidate_accesses[order]
                    else None
                ),
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
