#!/usr/bin/env python
"""Analyze Semantic-RQ shared-head masking against a matched random mask."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-root", type=Path, required=True)
    parser.add_argument("--intervention-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=260813)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_losses(path: Path, expected_mask: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("head_mask") != expected_mask:
        raise RuntimeError(f"invalid intervention result: {path}")
    return np.asarray(
        np.load(payload["losses"], allow_pickle=False)["token_loss"],
        dtype=np.float64,
    )


def document_statistics(
    left: np.ndarray, right: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(left) & np.isfinite(right)
    difference = np.where(valid, left - right, 0.0)
    return difference.sum(axis=1), valid.sum(axis=1, dtype=np.int64)


def bootstrap(
    values: list[tuple[np.ndarray, np.ndarray]],
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    observed_sum = sum(float(delta.sum()) for delta, _ in values)
    observed_count = sum(int(count.sum()) for _, count in values)
    if not observed_count:
        return {"tokens": 0, "delta_nll": None, "ci95": [None, None]}
    draws = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        total, count = 0.0, 0
        for seed_index in rng.integers(0, len(values), size=len(values)):
            delta, token_count = values[int(seed_index)]
            documents = rng.integers(0, len(delta), size=len(delta))
            total += float(delta[documents].sum())
            count += int(token_count[documents].sum())
        draws[draw] = total / count if count else np.nan
    finite = draws[np.isfinite(draws)]
    ci = np.quantile(finite, (0.025, 0.975))
    return {
        "tokens": observed_count,
        "delta_nll": observed_sum / observed_count,
        "ci95": [float(ci[0]), float(ci[1])],
        "replicates": replicates,
        "bootstrap_unit": "document nested within resampled seed",
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    stores: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    per_seed: dict[str, Any] = {}
    contrasts = {
        "shared_minus_none": ("shared", "none"),
        "random_minus_none": ("random-matched", "none"),
        "shared_minus_random": ("shared", "random-matched"),
    }

    for seed in (42, 43, 44):
        manifest = np.load(
            args.slice_root / f"manifest_eval2000_seed{seed}.npz",
            allow_pickle=False,
        )
        losses = {
            "none": load_losses(
                args.slice_root / f"semantic_rq_seed{seed}.json", "none"
            ),
            "shared": load_losses(
                args.intervention_root / f"shared_seed{seed}.json", "shared"
            ),
            "random-matched": load_losses(
                args.intervention_root / f"random_matched_seed{seed}.json",
                "random-matched",
            ),
        }
        attention = np.asarray(manifest["attention_mask"], dtype=bool)
        valid = attention[:, :-1] & attention[:, 1:]
        masks: dict[str, np.ndarray] = {}
        combined = np.zeros_like(valid)
        for order in (2, 3):
            category = np.asarray(manifest[f"category_{order}"])[:, :-1]
            lexical = np.asarray(
                manifest[f"lexical_jaccard_{order}"], dtype=np.float32
            )[:, :-1]
            shared_bits = np.asarray(
                manifest[f"rq_shared_head_mask_{order}"]
            )[:, :-1]
            shared = valid & (category == 2) & (shared_bits > 0)
            masks[f"{order}gram_semantic_neighbor_shared_code"] = shared
            masks[f"{order}gram_low_lexical_shared_code"] = shared & (lexical <= 0.10)
            combined |= shared
        masks["semantic_neighbor_shared_code_union"] = combined

        per_seed[str(seed)] = {}
        for contrast, (left, right) in contrasts.items():
            per_seed[str(seed)][contrast] = {}
            for slice_name, mask in masks.items():
                delta, counts = document_statistics(losses[left], losses[right], mask)
                stores.setdefault((contrast, slice_name), []).append((delta, counts))
                tokens = int(counts.sum())
                per_seed[str(seed)][contrast][slice_name] = {
                    "tokens": tokens,
                    "delta_nll": float(delta.sum() / tokens) if tokens else None,
                }

    aggregate: dict[str, Any] = {name: {} for name in contrasts}
    for (contrast, slice_name), values in stores.items():
        aggregate[contrast][slice_name] = bootstrap(
            values, args.replicates, rng
        )

    payload = {
        "status": "complete",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "interpretation": (
            "Positive shared_minus_random means masking actually shared RQ heads "
            "causes more NLL damage than masking an equal number of random heads."
        ),
        "seeds": [42, 43, 44],
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
