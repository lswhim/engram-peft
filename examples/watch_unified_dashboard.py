#!/usr/bin/env python
"""Unified tabbed dashboard for all Semantic-Keyed benchmark families.

Keeps one HTML file fresh as workers drop JSON results on the shared CFS tree.
It never invents numbers: a tab cell is only filled from a result file whose
``status == "complete"``.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


KE_DATASETS = ("counterfact", "zsre")
KE_METHODS = ("arithmetic", "semantic_flatten", "semantic_keyed")
KE_METRICS = ("efficacy", "paraphrase", "specificity", "harmonic_score")

XTREME_METHODS = ("arithmetic", "semantic_flatten", "semantic_keyed")
LM_METHODS = ("arithmetic", "semantic_flatten", "semantic_keyed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/semantic_hash_paper/dashboard.html"),
    )
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--logdir", type=Path, default=Path("run_logs"))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


_STEP_RE = re.compile(r"(\d+)/(\d+)\s+\[[0-9:]+<([0-9:]+),")


def tail_text(path: Path, max_bytes: int = 20000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


def last_step_progress(path: Path) -> tuple[str, str | None] | None:
    matches = _STEP_RE.findall(tail_text(path))
    if not matches:
        return None
    current, total, eta = matches[-1]
    return f"{current}/{total}", eta


def step_cell(logdir: Path, log_name: str) -> str:
    progress = last_step_progress(logdir / log_name)
    if progress is None:
        return "—"
    step, eta = progress
    if eta:
        return f"{step} · ETA {eta}"
    return step


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}"


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def xtreme_run_name(method: str) -> str:
    """Map a dashboard method label to the on-disk run directory name."""
    if method == "arithmetic":
        return "arithmetic_matched_seed42"
    if method == "semantic_flatten":
        return "rq_flatten_seed42"
    return f"rq_{method}_seed42"


def xtreme_payload(root: Path, family: str, method: str) -> tuple[str, dict[str, Any]]:
    """Return (state, payload) where state is complete|running|queued."""
    run_name = xtreme_run_name(method)
    path = root / f"{family}_semantic_keyed" / method / run_name / "metrics.json"
    payload = read_json(path)
    if not payload:
        return "queued", {}
    if payload.get("status") == "complete":
        return "complete", payload
    return "running", payload


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
                f"<tr><th>{html.escape(dataset)}</th>"
                f"<th>{html.escape(method)}</th>{cells}</tr>"
            )
    return rows


def language_details(languages: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(str(lang))}</td><td>{pct(float(v))}</td></tr>"
        for lang, v in languages.items()
        if lang != "macro"
    )
    macro = languages.get("macro")
    summary = f"{pct(macro)} / {len([k for k in languages if k != 'macro'])} langs"
    return (
        f"<details><summary>{summary}</summary><table>"
        f"<thead><tr><th>语言</th><th>acc</th></tr></thead><tbody>{rows}</tbody>"
        f"</table></details>"
    )


def xtreme_rows(root: Path, family: str, logdir: Path) -> list[str]:
    rows: list[str] = []
    for method in XTREME_METHODS:
        state, payload = xtreme_payload(root, family, method)
        if state == "complete":
            detail = language_details(payload.get("languages", {}))
            rows.append(
                f"<tr><th>{html.escape(method)}</th>"
                f"<td><span class='ok'>完成</span></td><td>{detail}</td></tr>"
            )
        elif state == "running":
            status = str(payload.get("status", "running"))
            if status == "loading_data":
                status = "训练中"
            elif status == "evaluating":
                status = "评测中"
            log_name = f"{family}_{method}.log"
            step = step_cell(logdir, log_name)
            rows.append(
                f"<tr><th>{html.escape(method)}</th>"
                f"<td><span class='run'>运行中 · {html.escape(status)}</span>"
                f" <span class='step'>{html.escape(step)}</span></td>"
                f"<td>—</td></tr>"
            )
        else:
            rows.append(
                f"<tr><th>{html.escape(method)}</th>"
                f"<td><span class='queue'>排队中</span></td><td>—</td></tr>"
            )
    return rows


def lm_rows(root: Path, logdir: Path) -> list[str]:
    rows: list[str] = []
    for method in LM_METHODS:
        payload = read_json(root / "standard_lm" / f"{method}_seed42.json")
        if payload.get("status") != "complete":
            step = step_cell(logdir, f"lm_{method}_seed42.log")
            rows.append(
                f"<tr><th>{html.escape(method)}</th>"
                f"<td><span class='run'>运行中</span>"
                f" <span class='step'>{html.escape(step)}</span></td>"
                f"<td>—</td><td>—</td><td>—</td></tr>"
            )
            continue
        results = payload.get("results", {})
        wikitext = results.get("wikitext", {})
        lambada = results.get("lambada_openai", {})
        wikitext = {key.split(",")[0]: value for key, value in wikitext.items()}
        lambada = {key.split(",")[0]: value for key, value in lambada.items()}
        rows.append(
            f"<tr><th>{html.escape(method)}</th>"
            f"<td><span class='ok'>完成</span></td>"
            f"<td>{fmt(wikitext.get('word_perplexity'))}</td>"
            f"<td>{fmt(wikitext.get('bits_per_byte'))}</td>"
            f"<td>{pct(lambada.get('acc'))}</td></tr>"
        )
    return rows


def render(root: Path, logdir: Path, now: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic-Keyed · Unified Dashboard</title>
<style>
:root{{--bg:#f4f1e9;--ink:#17201d;--muted:#68736d;--line:#d8d2c5;--green:#176b4d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"PingFang SC",sans-serif}}
main{{max-width:1100px;margin:auto;padding:28px 20px 80px}}h1{{font-size:28px;margin:4px 0 8px}}
h2{{font-size:18px;margin:18px 0 8px}}.muted{{color:var(--muted)}}
.tabs{{display:flex;gap:6px;margin:16px 0;flex-wrap:wrap}}
button.tab{{border:1px solid var(--line);background:#fffdf7;padding:8px 14px;border-radius:999px;cursor:pointer;font-weight:700}}
button.tab.active{{background:var(--ink);color:#fff;border-color:var(--ink)}}
.panel{{display:none}}.panel.active{{display:block}}
table{{border-collapse:collapse;width:100%;background:#fffdf7;border:1px solid var(--line)}}
th,td{{padding:8px 10px;border-bottom:1px solid #e6e0d5;text-align:left;vertical-align:top}}
thead th{{background:#f7f3ea}}details summary{{cursor:pointer;font-weight:600}}
.ok{{color:var(--green);font-weight:700}}.run{{color:#8a5a00;font-weight:700}}.queue{{color:var(--muted);font-weight:700}}
.step{{color:var(--muted);font-variant-numeric:tabular-nums}}
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
<h2>XNLI 跨语言（seed 42，英文 MNLI 训练 → 15 语零样本）</h2>
<table><thead><tr><th>方法</th><th>状态</th><th>Macro acc / 语言</th></tr></thead>
<tbody>{''.join(xtreme_rows(root, 'xnli', logdir))}</tbody></table></section>
<section id="pawsx" class="panel">
<h2>PAWS-X 跨语言（seed 42，英文训练 → 7 语零样本）</h2>
<table><thead><tr><th>方法</th><th>状态</th><th>Macro acc / 语言</th></tr></thead>
<tbody>{''.join(xtreme_rows(root, 'pawsx', logdir))}</tbody></table></section>
<section id="lm" class="panel">
<h2>语言建模（seed 42，FineWeb 400 步训练 → WikiText / LAMBADA）</h2>
<table><thead><tr><th>方法</th><th>状态</th><th>WikiText PPL</th><th>bits/byte</th><th>LAMBADA acc</th></tr></thead>
<tbody>{''.join(lm_rows(root, logdir))}</tbody></table></section>
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
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            temporary = args.output.with_suffix(".tmp.html")
            temporary.write_text(render(args.root, args.logdir, now), encoding="utf-8")
            temporary.replace(args.output)
        except Exception:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
