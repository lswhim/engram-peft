#!/usr/bin/env python
"""Unified tabbed dashboard for all Semantic-Keyed benchmark families."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


KE_DATASETS = ("counterfact", "zsre")
KE_METHODS = ("arithmetic", "semantic_flatten", "semantic_keyed")
KE_METRICS = ("efficacy", "paraphrase", "specificity", "harmonic_score")

XTREME_METHODS = ("arithmetic", "semantic_flatten", "semantic_keyed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/semantic_hash_paper/dashboard.html"),
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


def ke_rows(root: Path) -> list[str]:
    rows: list[str] = []
    for dataset in KE_DATASETS:
        for method in KE_METHODS:
            payload = read_json(
                root / "ke_semantic_keyed" / f"{dataset}_{method}_seed42.json"
            )
            if payload.get("status") != "complete":
                continue
            metrics = payload.get("metrics", {})
            cells = "".join(
                f"<td>{pct(metrics.get(m))}</td>" for m in KE_METRICS
            )
            rows.append(
                f"<tr><th>{html.escape(dataset)}</th><th>{html.escape(method)}</th>{cells}</tr>"
            )
    return rows


def xnli_rows(root: Path) -> list[str]:
    rows: list[str] = []
    for method in XTREME_METHODS:
        payload = read_json(
            root / "xnli_semantic_keyed" / method / f"rq_{method}_seed42" / "metrics.json"
        ) if method != "arithmetic" else read_json(
            root / "xnli_semantic_keyed" / method / "arithmetic_matched_seed42" / "metrics.json"
        )
        if payload.get("status") != "complete":
            rows.append(
                f"<tr><th>{method}</th><td>running</td></tr>"
            )
            continue
        languages = payload.get("languages", {})
        macro = (
            sum(float(v) for v in languages.values()) / len(languages)
            if languages
            else None
        )
        rows.append(
            f"<tr><th>{html.escape(method)}</th>"
            f"<td>{pct(macro)}</td><td>{len(languages)} langs</td></tr>"
        )
    return rows


def pawsx_rows(root: Path) -> list[str]:
    rows: list[str] = []
    for method in XTREME_METHODS:
        payload = read_json(
            root / "pawsx_semantic_keyed" / method / f"rq_{method}_seed42" / "metrics.json"
        ) if method != "arithmetic" else read_json(
            root / "pawsx_semantic_keyed" / method / "arithmetic_matched_seed42" / "metrics.json"
        )
        if payload.get("status") != "complete":
            rows.append(f"<tr><th>{method}</th><td>running</td></tr>")
            continue
        languages = payload.get("languages", {})
        macro = (
            sum(float(v) for v in languages.values()) / len(languages)
            if languages
            else None
        )
        rows.append(
            f"<tr><th>{html.escape(method)}</th>"
            f"<td>{pct(macro)}</td><td>{len(languages)} langs</td></tr>"
        )
    return rows


def render(root: Path, now: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic-Keyed · Unified Dashboard</title>
<style>
:root{{--bg:#f4f1e9;--ink:#17201d;--muted:#68736d;--line:#d8d2c5;--green:#176b4d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"PingFang SC",sans-serif}}
main{{max-width:1100px;margin:auto;padding:28px 20px 80px}}h1{{font-size:28px;margin:4px 0 8px}}
.muted{{color:var(--muted)}}.tabs{{display:flex;gap:6px;margin:16px 0;flex-wrap:wrap}}
button.tab{{border:1px solid var(--line);background:#fffdf7;padding:8px 14px;border-radius:999px;cursor:pointer;font-weight:700}}
button.tab.active{{background:var(--ink);color:#fff;border-color:var(--ink)}}
.panel{{display:none}}.panel.active{{display:block}}table{{border-collapse:collapse;width:100%;background:#fffdf7;border:1px solid var(--line)}}
th,td{{padding:8px 10px;border-bottom:1px solid #e6e0d5;text-align:left}}thead th{{background:#f7f3ea}}
</style></head><body><main>
<h1>Semantic-Keyed · Unified Dashboard</h1>
<div class="muted">更新于 {now}</div>
<nav class="tabs">
<button class="tab active" data-tab="ke">知识编辑</button>
<button class="tab" data-tab="xnli">XNLI</button>
<button class="tab" data-tab="pawsx">PAWS-X</button>
<button class="tab" data-tab="lm">语言建模</button>
</nav>
<section id="ke" class="panel active">
<h2>知识编辑（seed 42）</h2>
<table><thead><tr><th>数据集</th><th>方法</th><th>Efficacy</th><th>Paraphrase</th><th>Specificity</th><th>Harmonic</th></tr></thead>
<tbody>{''.join(ke_rows(root))}</tbody></table></section>
<section id="xnli" class="panel">
<h2>XNLI 跨语言（seed 42）</h2>
<table><thead><tr><th>方法</th><th>Macro acc</th><th>语言数</th></tr></thead>
<tbody>{''.join(xnli_rows(root))}</tbody></table></section>
<section id="pawsx" class="panel">
<h2>PAWS-X 跨语言（seed 42）</h2>
<table><thead><tr><th>方法</th><th>Macro acc</th><th>语言数</th></tr></thead>
<tbody>{''.join(pawsx_rows(root))}</tbody></table></section>
<section id="lm" class="panel"><h2>语言建模</h2><p class="muted">待训练 checkpoint 后评测</p></section>
</main>
<script>
document.querySelectorAll('button.tab').forEach(b=>b.onclick=()=>{{
document.querySelectorAll('button.tab').forEach(x=>x.classList.remove('active'));
document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active');}});
</script></body></html>
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
