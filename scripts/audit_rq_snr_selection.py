#!/usr/bin/env python
"""Audit whether RQ-SNR routing is conditional or static on the train address stream.

Every known 3-gram determines its suffix 2-gram exactly under the polynomial key
scheme.  Pairing those rows reconstructs the 16 heads seen at a token without
loading the language model.  Train access counts weight the audit by actual
address frequency rather than treating rare dictionary rows as equally common.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from engram_peft.rq_hashing import RQNgramMapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="+", type=Path, required=True)
    parser.add_argument("--access-counts", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def weighted_entropy(weights: dict[tuple[int, ...], float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return -sum(
        (weight / total) * math.log(weight / total)
        for weight in weights.values()
        if weight > 0
    )


def audit_table(
    table_dir: Path, access_counts: np.lib.npyio.NpzFile, top_k: int
) -> dict[str, Any]:
    mapping = RQNgramMapping(str(table_dir))
    if mapping.ngram_sizes != [2, 3]:
        raise ValueError(f"expected [2, 3] n-grams, got {mapping.ngram_sizes}")
    if not 0 < top_k < mapping.total_heads:
        raise ValueError("top-k must be between zero and total heads")

    keys_2 = mapping.sorted_keys[2]
    keys_3 = mapping.sorted_keys[3]
    suffix_2 = keys_3 % (mapping.base**2)
    row_2 = np.searchsorted(keys_2, suffix_2)
    hit = (row_2 < len(keys_2)) & (keys_2[np.minimum(row_2, len(keys_2) - 1)] == suffix_2)
    if not np.any(hit):
        raise ValueError("no 3-gram suffixes matched the 2-gram table")

    row_2 = row_2[hit]
    row_3 = np.flatnonzero(hit)
    codes = np.concatenate(
        [mapping.codes[2][row_2], mapping.codes[3][row_3]], axis=1
    )
    score_table = mapping.signal_to_interference_table()
    scores = np.take_along_axis(score_table[None, :, :], codes[:, :, None], axis=2)[
        :, :, 0
    ]
    selected = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]

    field = "train_access_count_3"
    if field not in access_counts:
        raise KeyError(f"{field} missing from access-count artifact")
    weights = np.asarray(access_counts[field], dtype=np.float64)[row_3]
    positive = weights > 0
    selected = selected[positive]
    scores = scores[positive]
    weights = weights[positive]
    if not len(weights):
        raise ValueError("no positive-frequency matched rows")

    head_mass = np.zeros(mapping.total_heads, dtype=np.float64)
    set_mass: dict[tuple[int, ...], float] = defaultdict(float)
    for heads, weight in zip(selected, weights, strict=True):
        canonical = tuple(sorted(int(head) for head in heads))
        set_mass[canonical] += float(weight)
        head_mass[list(canonical)] += weight
    total_weight = float(weights.sum())
    selection_rate = head_mass / total_weight
    top_sets = sorted(set_mass.items(), key=lambda item: item[1], reverse=True)[:20]

    ordered = np.sort(scores, axis=1)
    boundary_gap = ordered[:, -top_k] - ordered[:, -(top_k + 1)]
    normalized_entropy = weighted_entropy(set_mass) / math.log(max(len(set_mass), 2))
    return {
        "table": str(table_dir),
        "matched_known_3gram_rows": int(hit.sum()),
        "positive_access_rows": int(positive.sum()),
        "weighted_accesses": total_weight,
        "top_k": top_k,
        "distinct_selected_head_sets": len(set_mass),
        "selected_set_entropy_nats": weighted_entropy(set_mass),
        "selected_set_entropy_normalized": normalized_entropy,
        "mean_topk_boundary_gap": float(np.average(boundary_gap, weights=weights)),
        "head_selection_rate": {
            str(index): float(rate) for index, rate in enumerate(selection_rate)
        },
        "coarse_anchor_rate": {
            "2gram_level0": float(selection_rate[0]),
            "3gram_level0": float(selection_rate[mapping.num_levels]),
        },
        "top_selected_sets": [
            {"heads": list(heads), "access_fraction": weight / total_weight}
            for heads, weight in top_sets
        ],
    }


def main() -> None:
    args = parse_args()
    with np.load(args.access_counts, allow_pickle=False) as access_counts:
        payload = {
            "protocol": {
                "unit": "known train 3-gram paired with exact suffix 2-gram",
                "weight": "train 3-gram access frequency",
                "top_k": args.top_k,
                "no_model_forward": True,
            },
            "tables": [
                audit_table(table, access_counts, args.top_k) for table in args.tables
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
