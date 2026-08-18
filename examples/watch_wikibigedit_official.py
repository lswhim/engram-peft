#!/usr/bin/env python
"""Live WikiBigEdit official dashboard for the semantic-keyed matrix.

Scans ``outputs/semantic_memory/wikibigedit_official`` and renders one HTML
file with per-method/seed progress plus the latest five-axis metrics.  It never
blocks on long commands; run it as a background watcher and poll its HTML.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
from datetime import datetime
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
AXIS_LABELS = {
    "efficacy": "Update",
    "generalization": "Rephrase",
    "personas": "Personas",
    "multi_hop": "Mhop",
    "locality": "Locality",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/semantic_memory/wikibigedit_official"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/semantic_hash_paper/dashboard.html"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "arithmetic",
            "semantic_flatten",
            "semantic_keyed",
            "shuffled_flatten",
            "shuffled_semantic_keyed",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def scan(root: Path, methods: list[str], seeds: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in methods:
        entries: dict[int, dict[str, Any]] = {}
        for seed in seeds:
            base = root / method / f"seed_{seed}"
            cumulative = 0
            completed = 0
            points: list[dict[str, Any]] = []
            for index, count in enumerate(COUNTS):
                cumulative += count
                payload = read_json(base / f"t{index}_at_{cumulative}.json")
                if payload.get("status") == "complete":
                    completed += 1
                    points.append(payload)
            entries[seed] = {
                "completed_timesteps": completed,
                "points": points,
            }
        # Latest completed axes across completed seeds.
        final_axes: dict[str, dict[str, float | None]] = {}
        values_by_axis: dict[str, list[float]] = {axis: [] for axis in AXES}
        for seed in seeds:
            points = entries[seed]["points"]
            if not points:
                continue
            metrics = points[-1].get("metrics", {})
            for axis in AXES:
                metric = metrics.get(f"axis/{axis}")
                if isinstance(metric, dict) and isinstance(metric.get("mean"), (int, float)):
                    values_by_axis[axis].append(float(metric["mean"]))
        for axis in AXES:
            values = values_by_axis[axis]
            final_axes[axis] = {
                "mean": statistics.mean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
                "n_seeds": len(values),
            }
        summary[method] = {
            "seeds": entries,
            "final_axes": final_axes,
            "completed_seeds": [
                seed for seed in seeds if entries[seed]["completed_timesteps"] == len(COUNTS)
            ],
        }
    return summary


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}"


def render(summary: dict[str, Any], now: str) -> str:
    method_rows: list[str] = []
    for method, payload in summary.items():
        axes = payload["final_axes"]
        cells = "".join(
            f"<td>{fmt(axes[axis]['mean'])} ± {fmt(axes[axis]['sample_std'], 3)}</td>"
            for axis in AXES
        )
        done = len(payload["completed_seeds"])
        method_rows.append(
            f"<tr><th>{html.escape(method)}</th>"
            f"<td>{done}</td>{cells}</tr>"
        )

    seed_rows: list[str] = []
    for method, payload in summary.items():
        for seed, entry in payload["seeds"].items():
            seed_rows.append(
                f"<tr><td>{html.escape(method)}</td><td>{seed}</td>"
                f"<td>{entry['completed_timesteps']}/{len(TIMESTEPS)}</td></tr>"
            )

    head = "".join(f"<th>{html.escape(AXIS_LABELS[a])}</th>" for a in AXES)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WikiBigEdit Semantic-Keyed · Live</title>
<style>
:root{{--bg:#f4f1e9;--ink:#17201d;--muted:#68736d;--line:#d8d2c5;--green:#176b4d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"PingFang SC",sans-serif}}
main{{max-width:1200px;margin:auto;padding:28px 20px 80px}}h1{{font-size:30px;margin:4px 0 6px}}
.muted{{color:var(--muted)}}table{{border-collapse:collapse;width:100%;background:#fffdf7;border:1px solid var(--line)}}
th,td{{padding:8px 10px;border-bottom:1px solid #e6e0d5;text-align:left}}thead th{{background:#f7f3ea}}
section{{margin:20px 0}}h2{{font-size:20px;margin:0 0 8px}}
</style></head><body><main>
<h1>WikiBigEdit Official · Semantic-Keyed Matrix</h1>
<div class="muted">更新于 {now} · 完成列 = 完整 8 时间步的 seed 数</div>
<section><h2>最新五轴（mean ± sample std，%）</h2>
<table><thead><tr><th>方法</th><th>完成</th>{head}</tr></thead>
<tbody>{''.join(method_rows)}</tbody></table></section>
<section><h2>逐 Seed 进度</h2>
<table><thead><tr><th>方法</th><th>Seed</th><th>时间步</th></tr></thead>
<tbody>{''.join(seed_rows)}</tbody></table></section>
</main></body></html>
"""


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        summary = scan(args.root, args.methods, args.seeds)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temporary = args.output.with_suffix(".tmp.html")
        temporary.write_text(render(summary, now), encoding="utf-8")
        temporary.replace(args.output)
        if args.once:
            break
        import time

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
