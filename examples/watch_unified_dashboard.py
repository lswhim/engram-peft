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
LM_METHODS = (
    "arithmetic",
    "semantic_flatten",
    "semantic_keyed",
    "shuffled_keyed",
)
# LM 100M worker writes standard_lm/100m_<method>_seed<seed>.json and logs to
# run_logs/lm100m_<method>_seed<seed>.log.
LM_RESULT_NAMES = {
    "arithmetic": "100m_arithmetic",
    "semantic_flatten": "100m_semantic_flatten",
    "semantic_keyed": "100m_semantic_keyed",
    "shuffled_keyed": "100m_shuffled_keyed",
}

WIKI_METHODS = (
    "arithmetic",
    "semantic_flatten",
    "semantic_keyed",
    "shuffled_flatten",
    "shuffled_semantic_keyed",
)
WIKI_AXES = ("efficacy", "generalization", "personas", "multi_hop", "locality")
WIKI_AXIS_LABELS = {
    "efficacy": "Update",
    "generalization": "Rephrase",
    "personas": "Personas",
    "multi_hop": "Mhop",
    "locality": "Locality",
}
WIKI_COUNTS = (26922, 29835, 54504, 43443, 121116, 101728, 69403, 55431)
WIKI_METHOD_LABELS = {
    "arithmetic": "Arithmetic",
    "semantic_flatten": "Semantic-flat",
    "semantic_keyed": "Semantic-keyed",
    "shuffled_flatten": "Shuffled-flat",
    "shuffled_semantic_keyed": "Shuffled-keyed",
}

# Machine / task inventory. run_logs live on the shared CFS, so the watcher on
# any one machine can read every worker's tqdm progress regardless of which
# host the worker actually runs on. ``complete_path`` is an ``outputs``-relative
# file that exists only once that task has finished.
MACHINE_TASKS = (
    {
        "machine": "1号机 (4×A100)",
        "gpu": "GPU1",
        "label": "XNLI · semantic_keyed",
        "log": "xnli_semantic_keyed.log",
        "expected_total": 12272,
        "complete": "xnli_semantic_keyed/semantic_keyed/rq_semantic_keyed_seed42/metrics.json",
    },
    {
        "machine": "3号机 (8×A100)",
        "gpu": "GPU1",
        "label": "XNLI · semantic_flatten",
        "log": "xnli_semantic_flatten.log",
        "expected_total": 12272,
        "complete": "xnli_semantic_keyed/semantic_flatten/rq_flatten_seed42/metrics.json",
    },
    {
        "machine": "1号机 (4×A100)",
        "gpu": "GPU2",
        "label": "WikiBigEdit · semantic_keyed",
        "log": "wikibigedit_official_semantic_keyed_seed42_resume.log",
        "expected_total": 2726,
        "complete": "semantic_memory/wikibigedit_official/semantic_keyed/seed_42/t7_at_502382.json",
    },
    {
        "machine": "1号机 (4×A100)",
        "gpu": "GPU3",
        "label": "WikiBigEdit · semantic_flatten",
        "log": "wikibigedit_official_semantic_flatten_seed42_resume.log",
        "expected_total": 2726,
        "complete": "semantic_memory/wikibigedit_official/semantic_flatten/seed_42/t7_at_502382.json",
    },
    {
        "machine": "3号机 (8×A100)",
        "gpu": "GPU0",
        "label": "WikiBigEdit · arithmetic",
        "log": "wikibigedit_official_arithmetic_seed42.log",
        "expected_total": 2726,
        "complete": "semantic_memory/wikibigedit_official/arithmetic/seed_42/t7_at_502382.json",
    },
    {
        "machine": "3号机 (8×A100)",
        "gpu": "GPU1",
        "label": "WikiBigEdit · shuffled_flatten",
        "log": "wikibigedit_official_shuffled_flatten_seed42_resume.log",
        "expected_total": 2726,
        "complete": "semantic_memory/wikibigedit_official/shuffled_flatten/seed_42/t7_at_502382.json",
    },
    {
        "machine": "3号机 (8×A100)",
        "gpu": "GPU2",
        "label": "WikiBigEdit · shuffled_semantic_keyed",
        "log": "wikibigedit_official_shuffled_semantic_keyed_seed42_resume.log",
        "expected_total": 2726,
        "complete": "semantic_memory/wikibigedit_official/shuffled_semantic_keyed/seed_42/t7_at_502382.json",
    },
    {
        "machine": "3号机 (8×A100)",
        "gpu": "GPU3",
        "label": "LM 100M · arithmetic",
        "log": "lm100m_arithmetic_seed42.log",
        "expected_total": 763,
        "complete": "standard_lm/100m_arithmetic_seed42.json",
    },
    {
        "machine": "3号机 (8×A100)",
        "gpu": "GPU7",
        "label": "LM 100M · semantic_keyed",
        "log": "lm100m_semantic_keyed_seed42.log",
        "expected_total": 763,
        "complete": "standard_lm/100m_semantic_keyed_seed42.json",
    },
)


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


def last_step_progress(
    path: Path, expected_total: int | None = None
) -> tuple[str, str | None] | None:
    matches = _STEP_RE.findall(tail_text(path))
    if not matches:
        return None
    if expected_total is not None:
        matches = [m for m in matches if int(m[1]) == expected_total]
        if not matches:
            return None
    current, total, eta = matches[-1]
    return f"{current}/{total}", eta


def log_is_stale(path: Path, max_age: float = 600.0) -> bool:
    """True when a log exists but has not been written for ``max_age`` seconds."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    return (time.time() - mtime) > max_age


def step_cell(logdir: Path, log_name: str, expected_total: int | None = None) -> str:
    progress = last_step_progress(logdir / log_name, expected_total)
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


def best_in(values: dict[Any, float], *, higher: bool) -> Any | None:
    """Return the key whose value is optimal in a non-empty numeric dict."""
    numeric = {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}
    if not numeric:
        return None
    return min(numeric, key=numeric.get) if not higher else max(numeric, key=numeric.get)


def cell(value: float | None, best: bool) -> str:
    """Render a metric cell, bolding the column's optimum."""
    text = pct(value)
    return f"<td class='best'>{text}</td>" if best else f"<td>{text}</td>"


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
        by_metric = {
            metric: {
                method: read_json(
                    root / "ke_semantic_keyed" / f"{dataset}_{method}_seed42.json"
                ).get("metrics", {}).get(metric)
                for method in KE_METHODS
            }
            for metric in KE_METRICS
        }
        winners = {
            metric: best_in(by_metric[metric], higher=True) for metric in KE_METRICS
        }
        for method in KE_METHODS:
            payload = read_json(
                root / "ke_semantic_keyed" / f"{dataset}_{method}_seed42.json"
            )
            if payload.get("status") != "complete":
                continue
            metrics = payload.get("metrics", {})
            cells = "".join(
                cell(metrics.get(m), winners[m] == method) for m in KE_METRICS
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
            log_path = logdir / log_name
            if log_is_stale(log_path):
                rows.append(
                    f"<tr><th>{html.escape(method)}</th>"
                    f"<td><span class='queue'>已停止</span>"
                    f" <span class='step'>{html.escape(step)}</span></td>"
                    f"<td>—</td></tr>"
                )
            else:
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
    parsed: dict[str, dict[str, float | None]] = {}
    for method in LM_METHODS:
        result_name = LM_RESULT_NAMES[method]
        payload = read_json(root / "standard_lm" / f"{result_name}_seed42.json")
        if payload.get("status") != "complete":
            parsed[method] = {}
            continue
        results = payload.get("results", {})
        wikitext = {k.split(",")[0]: v for k, v in results.get("wikitext", {}).items()}
        lambada = {k.split(",")[0]: v for k, v in results.get("lambada_openai", {}).items()}
        parsed[method] = {
            "ppl": wikitext.get("word_perplexity"),
            "bpb": wikitext.get("bits_per_byte"),
            "acc": lambada.get("acc"),
        }

    winners = {
        "ppl": best_in({m: parsed[m].get("ppl") for m in LM_METHODS}, higher=False),
        "bpb": best_in({m: parsed[m].get("bpb") for m in LM_METHODS}, higher=False),
        "acc": best_in({m: parsed[m].get("acc") for m in LM_METHODS}, higher=True),
    }

    rows: list[str] = []
    for method in LM_METHODS:
        if not parsed.get(method):
            step = step_cell(logdir, f"lm100m_{method}_seed42.log")
            rows.append(
                f"<tr><th>{html.escape(method)}</th>"
                f"<td><span class='run'>运行中</span>"
                f" <span class='step'>{html.escape(step)}</span></td>"
                f"<td>—</td><td>—</td><td>—</td></tr>"
            )
            continue
        values = parsed[method]
        cls = lambda key: "best" if winners[key] == method else ""
        rows.append(
            f"<tr><th>{html.escape(method)}</th>"
            f"<td><span class='ok'>完成</span></td>"
            f"<td class='{cls('ppl')}'>"
            f"{fmt(values.get('ppl'))}</td>"
            f"<td class='{cls('bpb')}'>"
            f"{fmt(values.get('bpb'))}</td>"
            f"<td class='{cls('acc')}'>"
            f"{pct(values.get('acc'))}</td></tr>"
        )
    return rows


def wiki_cumulative_counts() -> list[int]:
    """Cumulative edit counts at the end of each of the 8 official timesteps."""
    cumulative: list[int] = []
    total = 0
    for count in WIKI_COUNTS:
        total += count
        cumulative.append(total)
    return cumulative


def wiki_metric_mean(payload: dict[str, Any], axis: str) -> float | None:
    metric = payload.get("metrics", {}).get(f"axis/{axis}")
    if isinstance(metric, dict) and isinstance(metric.get("mean"), (int, float)):
        return float(metric["mean"])
    return None


def wiki_payloads(root: Path, method: str) -> dict[str, dict[str, Any]]:
    """Return a map of timestep label -> result payload for one method."""
    base = root / "semantic_memory" / "wikibigedit_official" / method / "seed_42"
    out: dict[str, dict[str, Any]] = {}
    for index, cumulative in enumerate(wiki_cumulative_counts()):
        payload = read_json(base / f"t{index}_at_{cumulative}.json")
        out[f"T{index}"] = payload
    return out


def wiki_progress(logdir: Path, method: str) -> str:
    log_path = logdir / f"wikibigedit_official_{method}_seed42_resume.log"
    progress = last_step_progress(log_path)
    if progress is None:
        return "—"
    step, eta = progress
    if eta:
        return f"{step} · ETA {eta}"
    return step


def wiki_table_rows(root: Path, logdir: Path) -> list[str]:
    timesteps = [f"T{i}" for i in range(len(WIKI_COUNTS))]
    rows: list[str] = []
    for method in WIKI_METHODS:
        payloads = wiki_payloads(root, method)
        step = wiki_progress(logdir, method)
        live = step != "—"
        cells: list[str] = []
        for timestep in timesteps:
            payload = payloads.get(timestep, {})
            axis_means = {
                axis: wiki_metric_mean(payload, axis) for axis in WIKI_AXES
            }
            if payload.get("status") == "complete" and any(
                value is not None for value in axis_means.values()
            ):
                parts = " ".join(
                    f"<span>{html.escape(WIKI_AXIS_LABELS[a])} "
                    f"<b>{pct(axis_means[a])}</b></span>"
                    for a in WIKI_AXES
                )
                cells.append(
                    f"<td><div class='wiki-axes'>{parts}</div></td>"
                )
            elif payload.get("status") == "evaluating":
                cells.append(
                    f"<td><span class='run'>评测中</span></td>"
                )
            elif live:
                cells.append(
                    f"<td><span class='run'>训练中</span></td>"
                )
            else:
                cells.append(
                    f"<td><span class='queue'>排队中</span></td>"
                )
        rows.append(
            f"<tr><th>{html.escape(WIKI_METHOD_LABELS[method])}</th>"
            f"<td><span class='step'>{html.escape(step)}</span></td>"
            f"{''.join(cells)}</tr>"
        )
    return rows


def wiki_axis_table_rows(root: Path) -> list[str]:
    """One row per method with each of the 5 axes as its own column (final T7)."""
    rows: list[str] = []
    final = f"T{len(WIKI_COUNTS) - 1}"
    winners: dict[str, str | None] = {}
    values: dict[str, dict[str, float | None]] = {}
    for method in WIKI_METHODS:
        payload = wiki_payloads(root, method).get(final, {})
        values[method] = {
            axis: wiki_metric_mean(payload, axis) for axis in WIKI_AXES
        }
    for axis in WIKI_AXES:
        winners[axis] = best_in(
            {m: values[m].get(axis) for m in WIKI_METHODS}, higher=True
        )
    for method in WIKI_METHODS:
        cells = "".join(
            f"<td class='{'best' if winners[axis] == method else ''}'>"
            f"{pct(values[method].get(axis))}</td>"
            for axis in WIKI_AXES
        )
        rows.append(
            f"<tr><th>{html.escape(WIKI_METHOD_LABELS[method])}</th>{cells}</tr>"
        )
    return rows


def machine_rows(root: Path, logdir: Path) -> list[str]:
    """Render one row per task with its latest step/ETA or completion state."""
    rows: list[str] = []
    current_machine: str | None = None
    for task in MACHINE_TASKS:
        if task["machine"] != current_machine:
            current_machine = task["machine"]
            rows.append(
                "<tr class='machine-head'>"
                f"<th colspan='4'>{html.escape(str(current_machine))}</th></tr>"
            )

        complete_path = root / str(task["complete"])
        complete = False
        if complete_path.is_file():
            payload = read_json(complete_path)
            # ``metrics.json`` exists as soon as a run starts with
            # ``status=loading_data``, so file existence is not enough.
            complete = payload.get("status") == "complete"
        progress = last_step_progress(
            logdir / str(task["log"]),
            expected_total=task.get("expected_total"),
        )
        log_path = logdir / str(task["log"])
        if complete:
            state = "<span class='ok'>完成</span>"
            step = "—"
        elif progress is None:
            if log_is_stale(log_path):
                state = "<span class='queue'>已停止</span>"
                step = "—"
            else:
                state = "<span class='queue'>排队中</span>"
                step = "—"
        else:
            step, eta = progress
            if log_is_stale(log_path):
                state = "<span class='queue'>已停止</span>"
            else:
                state = "<span class='run'>运行中</span>"
                if eta:
                    step = f"{step} · ETA {eta}"

        rows.append(
            "<tr>"
            f"<td>{html.escape(str(task['gpu']))}</td>"
            f"<th>{html.escape(str(task['label']))}</th>"
            f"<td>{state}</td>"
            f"<td class='step'>{html.escape(step)}</td>"
            "</tr>"
        )
    return rows


def render(root: Path, logdir: Path, now: str) -> str:
    timestep_heads = "".join(
        f"<th>T{i}<span class='sub'>{(WIKI_COUNTS[i] if i == 0 else sum(WIKI_COUNTS[:i+1])):,}</span></th>"
        for i in range(len(WIKI_COUNTS))
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic-Keyed · Unified Dashboard</title>
<style>
:root{{--bg:#f4f6f9;--surface:#ffffff;--ink:#17201d;--muted:#6b7280;--line:#e5e7eb;--green:#176b4d;--amber:#b45309;--blue:#245ca4;--accent:#0f766e}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 ui-sans-serif,system-ui,"PingFang SC","Microsoft YaHei",sans-serif;-webkit-font-smoothing:antialiased}}
main{{max-width:1280px;margin:auto;padding:36px 24px 96px}}
.masthead{{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;border-bottom:2px solid var(--ink);padding-bottom:18px}}
.masthead h1{{font-size:30px;margin:0;letter-spacing:-.01em;font-weight:800}}
.masthead .meta{{color:var(--muted);font-size:12px;text-align:right}}
.tabs{{display:flex;gap:8px;margin:22px 0 24px;flex-wrap:wrap}}
button.tab{{appearance:none;border:1px solid var(--line);background:var(--surface);padding:9px 16px;border-radius:8px;cursor:pointer;font-weight:600;font-size:13px;color:var(--ink);transition:.15s}}
button.tab:hover{{border-color:var(--accent);color:var(--accent)}}
button.tab.active{{background:var(--ink);color:#fff;border-color:var(--ink)}}
.panel{{display:none}}.panel.active{{display:block}}
.panel h2{{font-size:18px;margin:0 0 14px;font-weight:700}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:auto;box-shadow:0 1px 2px rgba(16,24,40,.04)}}
.table-wrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;background:var(--surface)}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}}
thead th{{background:#f9fafb;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em;font-weight:600}}
tbody th{{font-weight:600;color:var(--ink)}}
tbody tr:last-child th,tbody tr:last-child td{{border-bottom:none}}
.metric-num{{font-variant-numeric:tabular-nums;font-weight:600}}
.sub{{display:block;font-size:10px;color:var(--muted);font-weight:400}}
.tag{{display:inline-flex;align-items:center;gap:6px;padding:3px 9px;border-radius:999px;font-size:11px;font-weight:600}}
.ok{{background:#ecfdf3;color:var(--green)}}
.run{{background:#fff7ed;color:var(--amber)}}
.queue{{background:#eef2f7;color:var(--muted)}}
.step{{color:var(--muted);font-variant-numeric:tabular-nums;font-size:12px}}
.best{{font-weight:800;color:var(--green);background:#f2f7f3}}
.dir{{font-weight:600;color:var(--muted);font-size:11px}}
.wiki-axes{{display:grid;grid-template-columns:repeat(5,auto);gap:6px 12px;min-width:360px}}
.wiki-axes span{{display:inline-flex;gap:5px;align-items:center;font-size:12px}}
.wiki-axes b{{font-variant-numeric:tabular-nums;font-weight:700}}
.machine-head th{{background:#f3f4f6;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em}}
details summary{{cursor:pointer;font-weight:600}}details table{{margin-top:6px}}
@media(max-width:760px){{.masthead{{flex-direction:column;align-items:flex-start}}.masthead .meta{{text-align:left}}.wiki-axes{{grid-template-columns:repeat(3,auto)}}}}
</style></head><body><main>
<div class="masthead"><h1>Semantic-Keyed · Unified Dashboard</h1><div class="meta">更新于 {now}<br>Qwen3-1.7B-Base · seed 42</div></div>
<nav class="tabs">
<button class="tab active" data-tab="ke">知识编辑</button>
<button class="tab" data-tab="xnli">XNLI</button>
<button class="tab" data-tab="pawsx">PAWS-X</button>
<button class="tab" data-tab="lm">语言建模</button>
<button class="tab" data-tab="wiki">WikiBigEdit</button>
<button class="tab" data-tab="machines">机器</button>
</nav>
<section id="ke" class="panel active">
<h2>知识编辑（seed 42）</h2>
<div class="card table-wrap">
<table><thead><tr><th>数据集</th><th>方法</th><th>Efficacy <span class="dir">↑</span></th><th>Paraphrase <span class="dir">↑</span></th><th>Specificity <span class="dir">↑</span></th><th>Harmonic <span class="dir">↑</span></th></tr></thead>
<tbody>{''.join(ke_rows(root))}</tbody></table></div></section>
<section id="xnli" class="panel">
<h2>XNLI 跨语言（seed 42，英文 MNLI 训练 → 15 语零样本）</h2>
<div class="card table-wrap">
<table><thead><tr><th>方法</th><th>状态</th><th>Macro acc <span class="dir">↑</span> / 语言</th></tr></thead>
<tbody>{''.join(xtreme_rows(root, 'xnli', logdir))}</tbody></table></div></section>
<section id="pawsx" class="panel">
<h2>PAWS-X 跨语言（seed 42，英文训练 → 7 语零样本）</h2>
<div class="card table-wrap">
<table><thead><tr><th>方法</th><th>状态</th><th>Macro acc <span class="dir">↑</span> / 语言</th></tr></thead>
<tbody>{''.join(xtreme_rows(root, 'pawsx', logdir))}</tbody></table></div></section>
<section id="lm" class="panel">
<h2>语言建模（seed 42，FineWeb 100M tokens → WikiText / LAMBADA）</h2>
<div class="card table-wrap">
<table><thead><tr><th>方法</th><th>状态</th><th>WikiText PPL <span class="dir">↓</span></th><th>bits/byte <span class="dir">↓</span></th><th>LAMBADA acc <span class="dir">↑</span></th></tr></thead>
<tbody>{''.join(lm_rows(root, logdir))}</tbody></table></div></section>
<section id="wiki" class="panel">
<h2>WikiBigEdit · 官方 8 timestep（seed 42）</h2>
<div class="card table-wrap">
<table><thead><tr><th>方法</th><th>当前进度</th>{timestep_heads}</tr></thead>
<tbody>{''.join(wiki_table_rows(root, logdir))}</tbody></table>
</div>
<h2 style="margin-top:22px">最终 T7 · 五轴（%）</h2>
<div class="card table-wrap">
<table><thead><tr><th>方法</th><th>Update <span class="dir">↑</span></th><th>Rephrase <span class="dir">↑</span></th><th>Personas <span class="dir">↑</span></th><th>Mhop <span class="dir">↑</span></th><th>Locality <span class="dir">↑</span></th></tr></thead>
<tbody>{''.join(wiki_axis_table_rows(root))}</tbody></table>
</div></section>
<section id="machines" class="panel">
<h2>机器与任务进度</h2>
<div class="card table-wrap">
<table><thead><tr><th>GPU</th><th>任务</th><th>状态</th><th>进度</th></tr></thead>
<tbody>{''.join(machine_rows(root, logdir))}</tbody></table>
</div></section>
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
