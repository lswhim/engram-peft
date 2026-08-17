#!/usr/bin/env python
"""Slice WikiBigEdit query accuracy by static-table hits versus dynamic RQ OOVs.

This is a post-hoc analysis over the auditable JSONL emitted by
``examples/evaluate_semantic_memory.py``.  It does not rerun the language model.
The address slice is computed from the exact prompt/answer token sequence used by
the scorer, so a query is ``offline_all`` only when every valid 2/3-gram context
is present in the frozen RQ table.  Otherwise it is ``dynamic_any``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from examples.evaluate_semantic_memory import first_nonempty, formatted_pair
from scripts.build_crosslingual_access_counts import valid_access_mask
from scripts.build_lm_slice_manifest import poly_keys, table_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def query_answers(manifest: Path) -> dict[tuple[str, str], str]:
    answers: dict[tuple[str, str], str] = {}
    for case in read_jsonl(manifest):
        for query in case.get("queries", []):
            answer = first_nonempty(query.get("answers", []))
            if answer is not None:
                answers[(str(case["case_id"]), str(query["prompt"]))] = answer
    return answers


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not row.get("eligible", False):
            continue
        axis = str(row["axis"])
        slice_name = str(row["address_slice"])
        accuracy = float(row["accuracy"])
        groups[f"slice/{slice_name}"].append(accuracy)
        groups[f"axis/{axis}/slice/{slice_name}"].append(accuracy)
    return {
        key: {"mean": float(np.mean(values)), "n": len(values)}
        for key, values in sorted(groups.items())
    }


def main() -> None:
    from transformers import AutoTokenizer

    from engram_peft.compression import CompressedTokenizer

    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    samples_path = Path(result.get("samples") or args.result.with_suffix(".jsonl"))
    if not samples_path.is_absolute():
        candidate = args.result.parent / samples_path
        samples_path = candidate if candidate.is_file() else samples_path
    rows = read_jsonl(samples_path)
    answers = query_answers(args.manifest)
    prompt_format = str(result.get("protocol", {}).get("prompt_format", "plain"))

    meta = json.loads((args.table_dir / "meta.json").read_text(encoding="utf-8"))
    orders = tuple(int(order) for order in meta["ngram_sizes"])
    table_keys = {
        order: np.load(args.table_dir / f"keys_{order}.npy", allow_pickle=False)
        for order in orders
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    compressor = CompressedTokenizer(tokenizer=tokenizer)
    base = int(compressor.compressed_vocab_size) + 1
    if base != int(meta["base"]):
        raise ValueError(f"table base={meta['base']} does not match tokenizer base={base}")

    for row in rows:
        key = (str(row["case_id"]), str(row["prompt"]))
        if key not in answers:
            raise KeyError(f"scored query missing from manifest: {key}")
        prompt_ids, answer_ids = formatted_pair(
            tokenizer, row["prompt"], answers[key], prompt_format
        )
        token_ids = np.asarray([prompt_ids + answer_ids], dtype=np.int64)
        attention = np.ones_like(token_ids, dtype=np.uint8)
        compressed = np.asarray(compressor.map_ids(token_ids), dtype=np.int64)
        per_order: dict[str, dict[str, float | int]] = {}
        total_candidates = 0
        total_hits = 0
        for order in orders:
            keys = poly_keys(compressed, order, base)
            _, hit = table_positions(keys, table_keys[order])
            valid = valid_access_mask(attention, order)
            candidates = int(valid.sum())
            hits = int((hit & valid).sum())
            total_candidates += candidates
            total_hits += hits
            per_order[str(order)] = {
                "candidate": candidates,
                "offline_hit": hits,
                "offline_rate": hits / candidates if candidates else 1.0,
            }
        row["address_coverage"] = per_order
        row["address_slice"] = (
            "offline_all" if total_hits == total_candidates else "dynamic_any"
        )

    payload = {
        "status": "complete",
        "result": str(args.result),
        "manifest": str(args.manifest),
        "table_dir": str(args.table_dir),
        "queries": len(rows),
        "eligible_queries": sum(bool(row.get("eligible")) for row in rows),
        "metrics": summarize(rows),
        "protocol": {
            "unit": "query",
            "offline_all": "every valid context n-gram is in the frozen RQ table",
            "dynamic_any": "at least one valid context n-gram requires online embedding-to-RQ",
            "score_reused_from": str(samples_path),
            "prompt_format": prompt_format,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples_output = args.output.with_suffix(".jsonl")
    with samples_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload["samples"] = str(samples_output)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
