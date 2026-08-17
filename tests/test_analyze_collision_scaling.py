import json
import random
from pathlib import Path

from scripts.analyze_collision_scaling import (
    hierarchical_cluster_bootstrap,
    paired_case_deltas,
)


def write_rows(path: Path, scores: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for case_id, accuracy in scores.items():
            handle.write(json.dumps({
                "case_id": case_id,
                "axis": "generalization",
                "accuracy": accuracy,
                "eligible": True,
            }) + "\n")


def test_paired_case_delta_uses_common_case_ids(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    write_rows(left, {"a": 0.8, "b": 0.5})
    write_rows(right, {"a": 0.3, "c": 0.2})

    assert paired_case_deltas(left, right, "generalization") == {"a": 0.5}


def test_hierarchical_bootstrap_reports_observed_paired_mean() -> None:
    by_seed = {
        42: {"a": 0.1, "b": 0.3},
        123: {"a": 0.2, "b": 0.4},
    }

    mean, low, high, observations = hierarchical_cluster_bootstrap(
        by_seed, 500, random.Random(7)
    )

    assert mean == 0.25
    assert low <= mean <= high
    assert observations == 4
