#!/usr/bin/env python
"""Paired difference-in-differences for semantic bucket reliability."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from scripts.analyze_bucket_target_conflict import bootstrap_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def load_samples(path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            values[str(row["case_id"])] = {
                key: float(row[key])
                for key in ("top_specific", "bottom_specific", "random_k")
            }
    return values


def paired_interactions(
    semantic: dict[str, dict[str, float]],
    control: dict[str, dict[str, float]],
    reference: str,
) -> dict[str, float]:
    common = sorted(semantic.keys() & control.keys())
    return {
        case_id: (
            semantic[case_id]["top_specific"] - semantic[case_id][reference]
        ) - (
            control[case_id]["top_specific"] - control[case_id][reference]
        )
        for case_id in common
    }


def main() -> None:
    args = parse_args()
    semantic = load_samples(args.semantic)
    control = load_samples(args.control)
    rng = random.Random(args.seed)
    results: dict[str, Any] = {}
    for reference in ("random_k", "bottom_specific"):
        deltas = paired_interactions(semantic, control, reference)
        results[f"semantic_minus_control_top_vs_{reference}"] = bootstrap_mean(
            deltas, args.bootstrap_replicates, rng
        )
    payload = {
        "status": "complete",
        "semantic": str(args.semantic),
        "control": str(args.control),
        "metric": "difference-in-differences of held-out surprisal; negative favors semantic specificity",
        "bootstrap": "paired case-cluster bootstrap",
        "bootstrap_replicates": args.bootstrap_replicates,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
