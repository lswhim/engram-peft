#!/usr/bin/env python
"""Render KE semantic-keyed results into one live HTML table."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DATASETS = ("counterfact", "zsre")
METHODS = ("arithmetic", "semantic_keyed")
METRICS = ("efficacy", "paraphrase", "specificity", "harmonic_score")
LABELS = {
    "efficacy": "Efficacy",
    "paraphrase": "Paraphrase",
    "specificity": "Specificity",
    "harmonic_score": "Harmonic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/ke_semantic_keyed"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/semantic_hash_paper/ke_dashboard.html"),
    )
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}"


def render(root: Path, now: str) -> str:
    rows: list[str] = []
    for dataset in DATASETS:
        for method in METHODS:
            payload = read_json(root / f"{dataset}_{method}_seed42.json")
            if payload.get("status") != "complete":
                continue
            metrics = payload.get("metrics", {})
            cells = "".join(
                f"<td>{pct(metrics.get(m))}</td>" for m in METRICS
            )
            rows.append(
                f"<tr><th>{dataset}</th><th>{method}</th>{cells}</tr>"
            )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>KE Semantic-Keyed · Live</title>
<style>
:root{{--bg:#f4f1e9;--ink:#17201d;--muted:#68736d;--line:#d8d2c5}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"PingFang SC",sans-serif}}
main{{max-width:900px;margin:auto;padding:28px 20px}}h1{{font-size:26px}}
table{{border-collapse:collapse;width:100%;background:#fffdf7;border:1px solid var(--line)}}
th,td{{padding:8px 10px;border-bottom:1px solid #e6e0d5;text-align:left}}thead th{{background:#f7f3ea}}
</style></head><body><main>
<h1>Knowledge Editing · Semantic-Keyed</h1>
<div style="color:var(--muted)">更新于 {now} · seed 42</div>
<table><thead><tr><th>数据集</th><th>方法</th>{''.join(f'<th>{LABELS[m]}</th>' for m in METRICS)}</tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</main></body></html>
"""


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temporary = args.output.with_suffix(".tmp.html")
        temporary.write_text(render(args.root, now), encoding="utf-8")
        temporary.replace(args.output)
        if args.once:
            break
        import time
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
