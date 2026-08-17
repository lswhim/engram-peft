#!/usr/bin/env python
"""Build an auditable WikiBigEdit lifelong retention/forgetting summary."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


TIMESTEPS = (
    "wiki_big_edit_20240201_20240220",
    "wiki_big_edit_20240220_20240301",
    "wiki_big_edit_20240301_20240320",
    "wiki_big_edit_20240320_20240401",
    "wiki_big_edit_20240401_20240501",
    "wiki_big_edit_20240501_20240601",
    "wiki_big_edit_20240601_20240620",
    "wiki_big_edit_20240620_20240701",
)
COUNTS = (26922, 29835, 54504, 43443, 121116, 101728, 69403, 55431)
AXES = ("efficacy", "generalization", "personas", "multi_hop", "locality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    return parser.parse_args()


def result_paths(root: Path, method: str, seed: int) -> list[Path]:
    cumulative = 0
    paths = []
    for index, count in enumerate(COUNTS):
        cumulative += count
        paths.append(root / method / f"seed_{seed}" / f"t{index}_at_{cumulative}.json")
    return paths


def load_trajectory(paths: list[Path]) -> dict[str, Any] | None:
    if not all(path.is_file() for path in paths):
        return None
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(payload.get("status") != "complete" for payload in payloads):
        raise ValueError("official trajectory contains a non-complete result")
    retention: dict[str, dict[str, list[float | None]]] = {}
    for origin_index, origin in enumerate(TIMESTEPS):
        retention[origin] = {}
        for axis in AXES:
            values: list[float | None] = []
            for evaluation_index, payload in enumerate(payloads):
                key = f"cohort/{origin}/{axis}"
                metric = payload["metrics"].get(key)
                if evaluation_index < origin_index:
                    values.append(None)
                else:
                    values.append(float(metric["mean"]) if metric else None)
            retention[origin][axis] = values
    final_axes = {
        axis: float(payloads[-1]["metrics"][f"axis/{axis}"]["mean"])
        for axis in AXES
        if f"axis/{axis}" in payloads[-1]["metrics"]
    }
    forgetting: dict[str, float] = {}
    for axis in AXES:
        cohort_forgetting = []
        for origin in TIMESTEPS:
            observed = [value for value in retention[origin][axis] if value is not None]
            if observed:
                cohort_forgetting.append(max(observed) - observed[-1])
        if cohort_forgetting:
            forgetting[axis] = statistics.mean(cohort_forgetting)
    return {"final_axes": final_axes, "forgetting": forgetting, "retention": retention}


def mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
    }


def summarize(root: Path, methods: list[str], seeds: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in methods:
        runs: dict[int, dict[str, Any]] = {}
        for seed in seeds:
            trajectory = load_trajectory(result_paths(root, method, seed))
            if trajectory is not None:
                runs[seed] = trajectory
        final_axes = {
            axis: mean_std([
                run["final_axes"][axis]
                for run in runs.values()
                if axis in run["final_axes"]
            ])
            for axis in AXES
            if any(axis in run["final_axes"] for run in runs.values())
        }
        forgetting = {
            axis: mean_std([
                run["forgetting"][axis]
                for run in runs.values()
                if axis in run["forgetting"]
            ])
            for axis in AXES
            if any(axis in run["forgetting"] for run in runs.values())
        }
        output[method] = {
            "complete": set(runs) == set(seeds),
            "completed_seeds": sorted(runs),
            "final_axes": final_axes,
            "average_forgetting": forgetting,
            "runs": runs,
        }
    return {
        "protocol": {
            "timesteps": list(TIMESTEPS),
            "counts": list(COUNTS),
            "axes": list(AXES),
            "required_seeds": seeds,
            "forgetting": "mean_over_cohorts(max_score_since_origin - final_score)",
        },
        "methods": output,
    }


def main() -> None:
    args = parse_args()
    payload = summarize(args.root, args.methods, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    complete = sum(method["complete"] for method in payload["methods"].values())
    print(f"complete official methods: {complete}/{len(args.methods)}")


if __name__ == "__main__":
    main()
