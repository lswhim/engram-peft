#!/usr/bin/env python
"""Paired document-cluster bootstrap for the Semantic-RQ LM slices."""

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
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_losses(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete slice result: {path}")
    return np.asarray(np.load(payload["losses"], allow_pickle=False)["token_loss"])


def document_statistics(
    semantic: np.ndarray, control: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(semantic) & np.isfinite(control)
    difference = np.where(valid, semantic - control, 0.0)
    return difference.sum(axis=1, dtype=np.float64), valid.sum(axis=1, dtype=np.int64)


def bootstrap(
    per_seed: list[tuple[np.ndarray, np.ndarray]],
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    observed_sum = sum(float(values.sum()) for values, _ in per_seed)
    observed_count = sum(int(counts.sum()) for _, counts in per_seed)
    if observed_count == 0:
        return {"tokens": 0, "delta_nll": None, "ci95": [None, None]}
    draws = np.empty(replicates, dtype=np.float64)
    seed_count = len(per_seed)
    for draw in range(replicates):
        total, count = 0.0, 0
        for seed_index in rng.integers(0, seed_count, size=seed_count):
            values, counts = per_seed[int(seed_index)]
            documents = rng.integers(0, len(values), size=len(values))
            total += float(values[documents].sum())
            count += int(counts[documents].sum())
        draws[draw] = total / count if count else np.nan
    finite = draws[np.isfinite(draws)]
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {
        "tokens": observed_count,
        "delta_nll": observed_sum / observed_count,
        "ci95": [float(lower), float(upper)],
        "bootstrap_unit": "document nested within resampled seed",
        "replicates": replicates,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    seeds = (42, 43, 44)
    comparisons = ("arithmetic_matched", "rq_shuffled")
    stores: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    per_seed_results: dict[str, Any] = {}

    for seed in seeds:
        manifest = np.load(
            args.slice_root / f"manifest_eval2000_seed{seed}.npz",
            allow_pickle=False,
        )
        semantic = load_losses(args.slice_root / f"semantic_rq_seed{seed}.json")
        attention = np.asarray(manifest["attention_mask"], dtype=bool)
        valid_loss = attention[:, :-1] & attention[:, 1:]
        per_seed_results[str(seed)] = {}
        masks: dict[str, np.ndarray] = {"overall": valid_loss}
        for order in (2, 3):
            category = np.asarray(manifest[f"category_{order}"])[:, :-1]
            lexical = np.asarray(
                manifest[f"lexical_jaccard_{order}"], dtype=np.float32
            )[:, :-1]
            masks[f"{order}gram_exact_seen"] = valid_loss & (category == 1)
            masks[f"{order}gram_semantic_neighbor"] = valid_loss & (category == 2)
            masks[f"{order}gram_semantic_neighbor_low_lexical"] = (
                valid_loss & (category == 2) & (lexical <= 0.10)
            )
            overlap = np.asarray(
                manifest[f"rq_code_overlap_{order}"], dtype=np.float32
            )[:, :-1]
            masks[f"{order}gram_semantic_neighbor_shared_code"] = (
                valid_loss & (category == 2) & (overlap > 0)
            )
            masks[f"{order}gram_semantic_neighbor_no_shared_code"] = (
                valid_loss & (category == 2) & (overlap == 0)
            )
            masks[f"{order}gram_semantic_neighbor_low_lexical_shared_code"] = (
                valid_loss
                & (category == 2)
                & (lexical <= 0.10)
                & (overlap > 0)
            )
            masks[f"{order}gram_covered_no_neighbor"] = valid_loss & (category == 3)
            masks[f"{order}gram_covered_no_neighbor_high_lexical"] = (
                valid_loss & (category == 3) & (lexical >= 0.25)
            )
            masks[f"{order}gram_address_oov"] = valid_loss & (category == 4)

        for control_name in comparisons:
            control = load_losses(args.slice_root / f"{control_name}_seed{seed}.json")
            per_seed_results[str(seed)][control_name] = {}
            for slice_name, mask in masks.items():
                values, counts = document_statistics(semantic, control, mask)
                key = (control_name, slice_name)
                stores.setdefault(key, []).append((values, counts))
                tokens = int(counts.sum())
                per_seed_results[str(seed)][control_name][slice_name] = {
                    "tokens": tokens,
                    "delta_nll": float(values.sum() / tokens) if tokens else None,
                }

    aggregate: dict[str, Any] = {name: {} for name in comparisons}
    for (control_name, slice_name), values in stores.items():
        aggregate[control_name][slice_name] = bootstrap(
            values, args.replicates, rng
        )

    payload = {
        "status": "complete",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "definition": "delta_nll = Semantic-RQ minus control; negative is better",
        "seeds": list(seeds),
        "per_seed": per_seed_results,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
