#!/usr/bin/env python
"""Paired hierarchical bootstrap for WikiBigEdit dynamic-RQ burden slices."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.analyze_collision_scaling import hierarchical_cluster_bootstrap


AXES = ("efficacy", "generalization", "locality", "multi_hop")
BINS = (
    "00_exact",
    "01_0-10pct",
    "02_10-25pct",
    "03_25-50pct",
    "04_50-100pct",
)
GROUPS = BINS + ("dynamic_any",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--left", default="semantic_flatten")
    parser.add_argument("--right", default="arithmetic")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    return parser.parse_args()


def load_case_scores(path: Path, axis: str, burden: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("eligible")
                and row.get("axis") == axis
                and (
                    row.get("dynamic_oov_bin") == burden
                    or (
                        burden == "dynamic_any"
                        and row.get("dynamic_oov_bin") != "00_exact"
                    )
                )
            ):
                grouped[str(row["case_id"])].append(float(row["accuracy"]))
    return {
        case_id: statistics.mean(values) for case_id, values in grouped.items()
    }


def paired_deltas(left: Path, right: Path, axis: str, burden: str) -> dict[str, float]:
    left_scores = load_case_scores(left, axis, burden)
    right_scores = load_case_scores(right, axis, burden)
    if set(left_scores) != set(right_scores):
        raise ValueError(f"unmatched query slice for {axis}/{burden}")
    return {
        case_id: left_scores[case_id] - right_scores[case_id]
        for case_id in sorted(left_scores)
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.bootstrap_seed)
    expected = set(args.seeds)
    results: dict[str, Any] = {}
    for axis in AXES:
        results[axis] = {}
        for burden in GROUPS:
            by_seed: dict[int, dict[str, float]] = {}
            for seed in args.seeds:
                left = args.root / args.left / f"seed_{seed}" / "at_50000_oov_dose.jsonl"
                right = args.root / args.right / f"seed_{seed}" / "at_50000_oov_dose.jsonl"
                if left.is_file() and right.is_file():
                    deltas = paired_deltas(left, right, axis, burden)
                    if deltas:
                        by_seed[seed] = deltas
            cell: dict[str, Any] = {
                "complete": set(by_seed) == expected,
                "seeds": sorted(by_seed),
                "case_counts": {str(seed): len(rows) for seed, rows in by_seed.items()},
            }
            if by_seed:
                mean, low, high, observations = hierarchical_cluster_bootstrap(
                    by_seed, args.bootstrap_replicates, rng
                )
                seed_means = [statistics.mean(rows.values()) for rows in by_seed.values()]
                cell.update(
                    mean=mean,
                    sample_std=(statistics.stdev(seed_means) if len(seed_means) > 1 else None),
                    ci95=[low, high],
                    paired_seed_cases=observations,
                )
            results[axis][burden] = cell
    return {
        "status": "complete" if all(
            cell["complete"] for axis in results.values() for cell in axis.values()
            if cell["seeds"]
        ) else "partial",
        "comparison": f"{args.left}_minus_{args.right}",
        "protocol": {
            "required_seeds": args.seeds,
            "bootstrap": "seed-outer, paired case-cluster inner",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "scale": "fraction",
        },
        "results": results,
    }


def main() -> None:
    args = parse_args()
    payload = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
