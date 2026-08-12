#!/usr/bin/env python
"""Analyze the preregistered single-seed finite-memory capacity sweep."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


CONTROLS = ("arithmetic_matched", "rq_shuffled")
SLICES = ("overall", "3gram_semantic_neighbor_shared_code")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-root", type=Path, required=True)
    parser.add_argument("--gate1-slice-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_losses(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete slice result: {path}")
    losses = payload.get("losses")
    if not isinstance(losses, str):
        raise RuntimeError(f"missing token-loss path: {path}")
    return np.asarray(np.load(losses, allow_pickle=False)["token_loss"])


def masks(manifest: Any) -> dict[str, np.ndarray]:
    attention = np.asarray(manifest["attention_mask"], dtype=bool)
    valid = attention[:, :-1] & attention[:, 1:]
    category = np.asarray(manifest["category_3"], dtype=np.uint8)[:, :-1]
    overlap = np.asarray(manifest["rq_code_overlap_3"], dtype=np.float32)[:, :-1]
    return {
        "overall": valid,
        "3gram_semantic_neighbor_shared_code": (
            valid & (category == 2) & (overlap > 0)
        ),
    }


def paired_bootstrap(
    semantic: np.ndarray,
    control: np.ndarray,
    mask: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    valid = mask & np.isfinite(semantic) & np.isfinite(control)
    values = np.where(valid, semantic - control, 0.0).sum(axis=1, dtype=np.float64)
    counts = valid.sum(axis=1, dtype=np.int64)
    token_count = int(counts.sum())
    if token_count == 0:
        return {"tokens": 0, "delta_nll": None, "ci95": [None, None]}
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        documents = rng.integers(0, len(values), size=len(values))
        count = int(counts[documents].sum())
        draws[index] = float(values[documents].sum()) / count if count else np.nan
    finite = draws[np.isfinite(draws)]
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {
        "tokens": token_count,
        "delta_nll": float(values.sum()) / token_count,
        "ci95": [float(lower), float(upper)],
        "bootstrap_unit": "held-out document",
        "replicates": replicates,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    results: dict[str, Any] = {}
    for capacity in (64, 256, 1024):
        root = (
            args.gate1_slice_root
            if capacity == 256
            else args.capacity_root / f"k{capacity}" / "lm_slices"
        )
        manifest = np.load(root / "manifest_eval2000_seed42.npz", allow_pickle=False)
        semantic = load_losses(root / "semantic_rq_seed42.json")
        slice_masks = masks(manifest)
        results[str(capacity)] = {}
        for control_name in CONTROLS:
            control = load_losses(root / f"{control_name}_seed42.json")
            results[str(capacity)][control_name] = {
                slice_name: paired_bootstrap(
                    semantic,
                    control,
                    slice_masks[slice_name],
                    args.replicates,
                    rng,
                )
                for slice_name in SLICES
            }
    payload = {
        "status": "complete",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "definition": "delta_nll = Semantic-RQ minus control; negative is better",
        "seed": 42,
        "capacities_per_head": [64, 256, 1024],
        "results": results,
        "scope": "single-seed curve; center and decisive endpoints require replication",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
