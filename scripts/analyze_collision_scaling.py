#!/usr/bin/env python
"""Summarize the complete collision-scaling matrix without manual transcription.

The script accepts partial directories for live auditing, but marks every aggregate
as incomplete until all requested seeds are present. Paired confidence intervals
resample cases within seed so multiple queries from one edit stay clustered.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


AXES = ("efficacy", "generalization", "locality", "multi_hop")
DEFAULT_METHODS = (
    "arithmetic",
    "semantic_flatten",
    "shuffled_flatten",
    "shuffled_specificity",
    "loadmatched_flatten",
    "loadmatched_specificity",
    "semantic_specificity",
    "semantic_rq_snr",
    "shuffled_rq_snr",
    "loadmatched_rq_snr",
)
DEFAULT_PAIRS = (
    ("semantic_specificity", "semantic_flatten"),
    ("shuffled_specificity", "shuffled_flatten"),
    ("loadmatched_specificity", "loadmatched_flatten"),
    ("semantic_specificity", "shuffled_specificity"),
    ("semantic_specificity", "loadmatched_specificity"),
    ("semantic_flatten", "shuffled_flatten"),
    ("semantic_flatten", "loadmatched_flatten"),
    ("semantic_flatten", "arithmetic"),
    ("semantic_rq_snr", "semantic_flatten"),
    ("shuffled_rq_snr", "shuffled_flatten"),
    ("loadmatched_rq_snr", "loadmatched_flatten"),
    ("semantic_rq_snr", "shuffled_rq_snr"),
    ("semantic_rq_snr", "loadmatched_rq_snr"),
)
DEFAULT_INTERACTIONS = (
    (
        "semantic_specificity",
        "semantic_flatten",
        "shuffled_specificity",
        "shuffled_flatten",
    ),
    (
        "semantic_specificity",
        "semantic_flatten",
        "loadmatched_specificity",
        "loadmatched_flatten",
    ),
    (
        "semantic_rq_snr",
        "semantic_flatten",
        "shuffled_rq_snr",
        "shuffled_flatten",
    ),
    (
        "semantic_rq_snr",
        "semantic_flatten",
        "loadmatched_rq_snr",
        "loadmatched_flatten",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--milestones", nargs="+", type=int, default=[1000, 5000, 10000, 50000])
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    return parser.parse_args()


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def load_metric(path: Path, axis: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"non-complete result: {path}")
    return float(payload["metrics"][f"axis/{axis}"]["mean"])


def load_case_scores(path: Path, axis: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("eligible") and row.get("axis") == axis:
                grouped[str(row["case_id"])].append(float(row["accuracy"]))
    return {
        case_id: statistics.mean(values) for case_id, values in grouped.items()
    }


def paired_case_deltas(left: Path, right: Path, axis: str) -> dict[str, float]:
    left_scores = load_case_scores(left, axis)
    right_scores = load_case_scores(right, axis)
    common = sorted(left_scores.keys() & right_scores.keys())
    if not common:
        raise ValueError(f"no paired {axis} cases for {left} and {right}")
    return {case_id: left_scores[case_id] - right_scores[case_id] for case_id in common}


def interaction_case_deltas(
    semantic_aware: Path,
    semantic_flat: Path,
    control_aware: Path,
    control_flat: Path,
    axis: str,
) -> dict[str, float]:
    scores = [
        load_case_scores(path, axis)
        for path in (semantic_aware, semantic_flat, control_aware, control_flat)
    ]
    common = set(scores[0]).intersection(*(set(values) for values in scores[1:]))
    if not common:
        raise ValueError(f"no four-way paired {axis} cases")
    return {
        case_id: (scores[0][case_id] - scores[1][case_id])
        - (scores[2][case_id] - scores[3][case_id])
        for case_id in sorted(common)
    }


def hierarchical_cluster_bootstrap(
    by_seed: dict[int, dict[str, float]], replicates: int, rng: random.Random
) -> tuple[float, float, float, int]:
    seed_ids = sorted(by_seed)
    observed = statistics.mean(
        statistics.mean(case_deltas.values()) for case_deltas in by_seed.values()
    )
    draws: list[float] = []
    for _ in range(replicates):
        sampled_seed_means: list[float] = []
        for seed in rng.choices(seed_ids, k=len(seed_ids)):
            values = list(by_seed[seed].values())
            sampled_seed_means.append(statistics.mean(rng.choices(values, k=len(values))))
        draws.append(statistics.mean(sampled_seed_means))
    draws.sort()
    lower = draws[int(0.025 * (len(draws) - 1))]
    upper = draws[int(0.975 * (len(draws) - 1))]
    return observed, lower, upper, sum(len(values) for values in by_seed.values())


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    expected_seeds = set(args.seeds)
    aggregates: dict[str, Any] = {}
    for method in args.methods:
        aggregates[method] = {}
        for milestone in args.milestones:
            axis_payload: dict[str, Any] = {}
            for axis in AXES:
                values: dict[int, float] = {}
                for seed in args.seeds:
                    result = args.root / method / f"seed_{seed}" / f"at_{milestone}.json"
                    if result.is_file():
                        values[seed] = load_metric(result, axis)
                xs = list(values.values())
                axis_payload[axis] = {
                    "complete": set(values) == expected_seeds,
                    "seeds": values,
                    "mean": statistics.mean(xs) if xs else None,
                    "sample_std": sample_std(xs),
                }
            aggregates[method][str(milestone)] = axis_payload

    rng = random.Random(args.bootstrap_seed)
    comparisons: dict[str, Any] = {}
    active_methods = set(args.methods)
    for left, right in DEFAULT_PAIRS:
        if left not in active_methods or right not in active_methods:
            continue
        name = f"{left}_minus_{right}"
        comparisons[name] = {}
        for milestone in args.milestones:
            comparisons[name][str(milestone)] = {}
            for axis in AXES:
                by_seed: dict[int, dict[str, float]] = {}
                for seed in args.seeds:
                    left_samples = args.root / left / f"seed_{seed}" / f"at_{milestone}.jsonl"
                    right_samples = args.root / right / f"seed_{seed}" / f"at_{milestone}.jsonl"
                    if left_samples.is_file() and right_samples.is_file():
                        by_seed[seed] = paired_case_deltas(left_samples, right_samples, axis)
                payload: dict[str, Any] = {
                    "complete": set(by_seed) == expected_seeds,
                    "seeds": sorted(by_seed),
                }
                if by_seed:
                    mean, low, high, observations = hierarchical_cluster_bootstrap(
                        by_seed, args.bootstrap_replicates, rng
                    )
                    payload.update(
                        mean=mean,
                        ci95=[low, high],
                        paired_seed_cases=observations,
                    )
                comparisons[name][str(milestone)][axis] = payload

    interactions: dict[str, Any] = {}
    for semantic_aware, semantic_flat, control_aware, control_flat in DEFAULT_INTERACTIONS:
        if not {
            semantic_aware,
            semantic_flat,
            control_aware,
            control_flat,
        }.issubset(active_methods):
            continue
        name = (
            f"{semantic_aware}_minus_{semantic_flat}__minus__"
            f"{control_aware}_minus_{control_flat}"
        )
        interactions[name] = {}
        for milestone in args.milestones:
            interactions[name][str(milestone)] = {}
            for axis in AXES:
                by_seed: dict[int, dict[str, float]] = {}
                for seed in args.seeds:
                    paths = [
                        args.root / method / f"seed_{seed}" / f"at_{milestone}.jsonl"
                        for method in (
                            semantic_aware,
                            semantic_flat,
                            control_aware,
                            control_flat,
                        )
                    ]
                    if all(path.is_file() for path in paths):
                        by_seed[seed] = interaction_case_deltas(*paths, axis)
                payload: dict[str, Any] = {
                    "complete": set(by_seed) == expected_seeds,
                    "seeds": sorted(by_seed),
                }
                if by_seed:
                    mean, low, high, observations = hierarchical_cluster_bootstrap(
                        by_seed, args.bootstrap_replicates, rng
                    )
                    payload.update(
                        mean=mean,
                        ci95=[low, high],
                        paired_seed_cases=observations,
                    )
                interactions[name][str(milestone)][axis] = payload

    return {
        "protocol": {
            "required_methods": args.methods,
            "required_seeds": args.seeds,
            "required_milestones": args.milestones,
            "mean_scale": "fraction",
            "std": "sample standard deviation across seeds",
            "ci": "hierarchical paired case-cluster bootstrap",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "aggregates": aggregates,
        "comparisons": comparisons,
        "interactions": interactions,
    }


def main() -> None:
    args = parse_args()
    payload = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    complete_cells = sum(
        metric["complete"]
        for method in payload["aggregates"].values()
        for milestone in method.values()
        for metric in milestone.values()
    )
    total_cells = len(args.methods) * len(args.milestones) * len(AXES)
    print(f"aggregate completeness: {complete_cells}/{total_cells}")
    print(args.output)


if __name__ == "__main__":
    main()
