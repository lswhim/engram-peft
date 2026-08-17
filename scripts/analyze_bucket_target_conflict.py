#!/usr/bin/env python
"""Measure held-out target conflict in specificity-selected RQ buckets.

The chronological stream is split once: prefix edits estimate each physical
bucket's next-target-token distribution, while suffix edits only query those
distributions.  This tests whether low-load heads selected by the proposed
router carry more mutually compatible updates than bottom-load or random heads.
No language-model checkpoint or evaluation label is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_lm_slice_manifest import table_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-cases", type=int, default=40_000)
    parser.add_argument("--max-cases", type=int, default=50_000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--prior-strength", type=float, default=10.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def polynomial_key(values: list[int], base: int) -> int:
    key = 0
    for value in values:
        key = key * base + int(value)
    return key


def choose_heads(loads: list[int], top_k: int, seed: int) -> dict[str, list[int]]:
    if not 0 < top_k <= len(loads):
        raise ValueError("top_k must be between one and the number of available heads")
    ascending = sorted(range(len(loads)), key=lambda index: (loads[index], index))
    rng = random.Random(seed)
    return {
        "top_specific": ascending[:top_k],
        "bottom_specific": ascending[-top_k:],
        "random_k": sorted(rng.sample(range(len(loads)), top_k)),
    }


def posterior_probability(
    target_count: int,
    bucket_total: int,
    global_probability: float,
    prior_strength: float,
) -> float:
    return (target_count + prior_strength * global_probability) / (
        bucket_total + prior_strength
    )


def bootstrap_mean(
    values: dict[str, float], replicates: int, rng: random.Random
) -> dict[str, Any]:
    xs = list(values.values())
    if not xs:
        return {"mean": None, "ci95": None, "cases": 0}
    draws = [statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(replicates)]
    draws.sort()
    return {
        "mean": statistics.mean(xs),
        "ci95": [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]],
        "cases": len(xs),
    }


def main() -> None:
    from transformers import AutoTokenizer

    from engram_peft.compression import CompressedTokenizer

    args = parse_args()
    meta = json.loads((args.table_dir / "meta.json").read_text(encoding="utf-8"))
    orders = [int(value) for value in meta["ngram_sizes"]]
    levels = int(meta["num_levels"])
    codebook_size = int(meta["codebook_size"])
    keys = {
        order: np.load(args.table_dir / f"keys_{order}.npy", allow_pickle=False)
        for order in orders
    }
    codes = {
        order: np.load(args.table_dir / f"codes_{order}.npy", allow_pickle=False).astype(np.int64)
        for order in orders
    }
    bucket_loads: list[np.ndarray] = []
    for order in orders:
        bucket_loads.extend(
            np.bincount(codes[order][:, level], minlength=codebook_size)
            for level in range(levels)
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    compressor = CompressedTokenizer(tokenizer=tokenizer)
    base = int(compressor.compressed_vocab_size) + 1
    if base != int(meta["base"]):
        raise ValueError(f"table base={meta['base']} does not match tokenizer base={base}")

    rows: list[tuple[str, list[int], list[int], int]] = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            prompt_ids = tokenizer(str(case["prompt"]), add_special_tokens=True)["input_ids"]
            target_ids = tokenizer(
                " " + str(case["target"]).strip(), add_special_tokens=False
            )["input_ids"]
            prompt_ids = prompt_ids[: max(0, args.max_length - len(target_ids))]
            target_ids = target_ids[: args.max_length - len(prompt_ids)]
            token_ids = list(prompt_ids) + list(target_ids)
            compressed_ids = list(
                map(
                    int,
                    compressor.map_ids(np.asarray(token_ids, dtype=np.int64)),
                )
            )
            rows.append(
                (
                    str(case.get("case_id", len(rows))),
                    token_ids,
                    compressed_ids,
                    len(prompt_ids),
                )
            )
            if len(rows) % 5_000 == 0:
                print(f"[tokenize] {len(rows)}/{args.max_cases}", flush=True)
            if len(rows) >= args.max_cases:
                break
    if not args.prefix_cases < len(rows):
        raise ValueError("prefix_cases must leave at least one held-out case")

    bucket_targets: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    global_targets: Counter[int] = Counter()
    skipped_contexts = 0

    def accesses(compressed: list[int], target_position: int) -> tuple[list[int], list[int]] | None:
        context_end = target_position - 1
        head_codes: list[int] = []
        head_loads: list[int] = []
        head_offset = 0
        for order in orders:
            start = context_end - order + 1
            if start < 0:
                return None
            key = polynomial_key(compressed[start : context_end + 1], base)
            _, hit = table_positions(np.asarray([key], dtype=np.int64), keys[order])
            if not bool(hit[0]):
                return None
            position = int(np.searchsorted(keys[order], key))
            for level, code in enumerate(codes[order][position]):
                head_codes.append(int(code))
                head_loads.append(int(bucket_loads[head_offset + level][int(code)]))
            head_offset += levels
        return head_codes, head_loads

    for row_index, (_, token_ids, compressed, prompt_length) in enumerate(
        rows[: args.prefix_cases], start=1
    ):
        for target_position in range(prompt_length, len(token_ids)):
            access = accesses(compressed, target_position)
            if access is None:
                skipped_contexts += 1
                continue
            head_codes, _ = access
            target = int(token_ids[target_position])
            global_targets[target] += 1
            for head, code in enumerate(head_codes):
                bucket_targets[(head, code)][target] += 1
        if row_index % 5_000 == 0:
            print(f"[prefix buckets] {row_index}/{args.prefix_cases}", flush=True)

    global_total = sum(global_targets.values())
    vocabulary = int(tokenizer.vocab_size)
    per_case: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    group_hits: Counter[str] = Counter()
    group_events: Counter[str] = Counter()
    for case_index, (case_id, token_ids, compressed, prompt_length) in enumerate(
        rows[args.prefix_cases :], start=args.prefix_cases
    ):
        for target_position in range(prompt_length, len(token_ids)):
            access = accesses(compressed, target_position)
            if access is None:
                skipped_contexts += 1
                continue
            head_codes, loads = access
            groups = choose_heads(loads, args.top_k, args.seed + case_index * 1009 + target_position)
            target = int(token_ids[target_position])
            global_probability = (global_targets[target] + 1) / (global_total + vocabulary)
            for group, selected in groups.items():
                surprises: list[float] = []
                for head in selected:
                    counts = bucket_targets[(head, head_codes[head])]
                    probability = posterior_probability(
                        counts[target], sum(counts.values()), global_probability, args.prior_strength
                    )
                    surprises.append(-math.log(max(probability, 1e-30)))
                    group_hits[group] += int(counts[target] > 0)
                    group_events[group] += 1
                per_case[case_id][group].append(statistics.mean(surprises))
        if (case_index + 1 - args.prefix_cases) % 2_000 == 0:
            print(
                f"[heldout conflict] {case_index + 1 - args.prefix_cases}/{len(rows) - args.prefix_cases}",
                flush=True,
            )

    case_means = {
        group: {
            case_id: statistics.mean(values[group])
            for case_id, values in per_case.items()
            if values.get(group)
        }
        for group in ("top_specific", "bottom_specific", "random_k")
    }
    rng = random.Random(args.seed)
    group_metrics = {
        group: {
            **bootstrap_mean(values, args.bootstrap_replicates, rng),
            "target_seen_rate": group_hits[group] / group_events[group] if group_events[group] else None,
            "head_events": group_events[group],
        }
        for group, values in case_means.items()
    }
    comparisons: dict[str, Any] = {}
    for right in ("bottom_specific", "random_k"):
        common = sorted(case_means["top_specific"].keys() & case_means[right].keys())
        deltas = {
            case_id: case_means["top_specific"][case_id] - case_means[right][case_id]
            for case_id in common
        }
        comparisons[f"top_specific_minus_{right}"] = bootstrap_mean(
            deltas, args.bootstrap_replicates, rng
        )

    payload = {
        "status": "complete",
        "table_dir": str(args.table_dir),
        "protocol": {
            "prefix_cases": args.prefix_cases,
            "heldout_cases": len(rows) - args.prefix_cases,
            "top_k": args.top_k,
            "prior_strength": args.prior_strength,
            "metric": "held-out target-token posterior surprisal; lower is better",
            "selection": "distinct frozen-table row load only",
            "bootstrap": "case-cluster bootstrap",
            "bootstrap_replicates": args.bootstrap_replicates,
            "offline_complete_contexts_only": True,
        },
        "skipped_oov_or_short_contexts": skipped_contexts,
        "groups": group_metrics,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples_output = args.output.with_suffix(".jsonl")
    samples_tmp = samples_output.with_suffix(samples_output.suffix + ".tmp")
    with samples_tmp.open("w", encoding="utf-8") as handle:
        for case_id in sorted(per_case):
            values = {
                group: statistics.mean(per_case[case_id][group])
                for group in ("top_specific", "bottom_specific", "random_k")
                if per_case[case_id].get(group)
            }
            handle.write(json.dumps({"case_id": case_id, **values}) + "\n")
    os.replace(samples_tmp, samples_output)
    payload["samples"] = str(samples_output)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
