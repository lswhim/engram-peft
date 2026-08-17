import json
from pathlib import Path

from scripts.compare_wikibigedit_oov_dose import paired_deltas


def write_rows(path: Path, values: list[float]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, value in enumerate(values):
            row = {
                "eligible": True,
                "axis": "generalization",
                "dynamic_oov_bin": "02_10-25pct",
                "case_id": f"c{index}",
                "accuracy": value,
            }
            handle.write(json.dumps(row) + "\n")


def test_paired_deltas_keep_case_clusters(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    write_rows(left, [0.8, 0.4])
    write_rows(right, [0.5, 0.3])

    deltas = paired_deltas(left, right, "generalization", "02_10-25pct")

    assert set(deltas) == {"c0", "c1"}
    assert abs(deltas["c0"] - 0.3) < 1e-12
    assert abs(deltas["c1"] - 0.1) < 1e-12

    dynamic = paired_deltas(left, right, "generalization", "dynamic_any")
    assert dynamic == deltas
