import json
from pathlib import Path

from scripts.analyze_wikibigedit_official import COUNTS, TIMESTEPS, load_trajectory


def test_load_trajectory_builds_retention_and_forgetting(tmp_path: Path) -> None:
    paths = []
    cumulative = 0
    for evaluation_index, count in enumerate(COUNTS):
        cumulative += count
        path = tmp_path / f"t{evaluation_index}_at_{cumulative}.json"
        metrics = {"axis/efficacy": {"mean": 0.5, "n": 1}}
        for origin_index, origin in enumerate(TIMESTEPS[: evaluation_index + 1]):
            score = 0.8 - 0.1 * max(0, evaluation_index - origin_index)
            metrics[f"cohort/{origin}/efficacy"] = {"mean": score, "n": 1}
        path.write_text(json.dumps({"status": "complete", "metrics": metrics}))
        paths.append(path)

    trajectory = load_trajectory(paths)

    assert trajectory is not None
    assert trajectory["retention"][TIMESTEPS[0]]["efficacy"][0] == 0.8
    assert trajectory["retention"][TIMESTEPS[1]]["efficacy"][0] is None
    assert trajectory["forgetting"]["efficacy"] > 0
