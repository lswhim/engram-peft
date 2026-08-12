"""Render one live HTML dashboard for XNLI and PAWS-X experiment matrices."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

METHODS = ["base", "full_ft", "lora", "arithmetic", "arithmetic_matched", "rq"]
METHOD_LABELS = {
    "base": "Base",
    "full_ft": "Full FT",
    "lora": "LoRA",
    "arithmetic": "Arithmetic Engram",
    "arithmetic_matched": "Arithmetic (matched)",
    "rq": "Semantic-RQ Engram",
}
BENCHMARKS = {
    "xnli": {
        "title": "XNLI",
        "subtitle": "MNLI English full train (392,702) → 15-language XNLI test",
        "languages": [
            "ar", "bg", "de", "el", "en", "es", "fr", "hi",
            "ru", "sw", "th", "tr", "ur", "vi", "zh",
        ],
        "runner": "run_xtreme_xnli.py",
        "logs": {
            "base": "gpu3_base.log",
            "full_ft": "gpu2_full_ft.log",
            "lora": "gpu0_lora.log",
            "arithmetic": "gpu1_arithmetic.log",
            "arithmetic_matched": "gpu1_arithmetic_matched.log",
            "rq": "gpu3_wiki_rq.log",
        },
    },
    "pawsx": {
        "title": "PAWS-X",
        "subtitle": "PAWS-X English full train (49,401) → 7-language PAWS-X test",
        "languages": ["en", "de", "es", "fr", "ja", "ko", "zh"],
        "runner": "run_xtreme_pawsx.py",
        "logs": {method: f"pawsx_{method}.log" for method in METHODS},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xnli-results-dir", type=Path, required=True)
    parser.add_argument("--pawsx-results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def active_runs() -> set[tuple[str, str]]:
    active: set[tuple[str, str]] = set()
    proc = Path("/proc")
    if not proc.exists():
        return active
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="ignore"
            )
        except OSError:
            continue
        method_match = re.search(r"--method\s+([a-z_]+)", command)
        if not method_match:
            continue
        for benchmark, config in BENCHMARKS.items():
            if config["runner"] in command:
                active.add((benchmark, method_match.group(1)))
    return active


def log_progress(path: Path) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    matches = list(re.finditer(r"(\d{1,3})%\|[^\r\n]*?\|\s*(\d+)/(\d+)", content))
    losses = list(re.finditer(r"['\"]loss['\"]:\s*([0-9.eE+-]+)", content))
    result: dict[str, Any] = {}
    if matches:
        match = matches[-1]
        result.update(
            percent=int(match.group(1)),
            step=int(match.group(2)),
            total_steps=int(match.group(3)),
        )
    if losses:
        result["loss"] = float(losses[-1].group(1))
    if "Traceback (most recent call last)" in content[-12000:]:
        result["has_traceback"] = True
    return result


def collect_benchmark(
    benchmark: str, results_dir: Path, active: set[tuple[str, str]]
) -> dict[str, Any]:
    config = BENCHMARKS[benchmark]
    languages = list(config["languages"])
    rows: dict[str, Any] = {}
    for method in METHODS:
        metrics = read_json(results_dir / f"{method}_seed42" / "metrics.json")
        progress = log_progress(results_dir / "logs" / config["logs"][method])
        values = metrics.get("languages", {})
        values = values if isinstance(values, dict) else {}
        raw_status = str(metrics.get("status", "pending"))
        is_active = (benchmark, method) in active
        if raw_status == "complete":
            status = "complete"
        elif raw_status == "evaluating":
            status = "evaluating"
        elif is_active:
            status = "training"
        elif raw_status == "loading_data" and progress.get("has_traceback"):
            status = "failed"
        elif raw_status == "loading_data":
            status = "waiting"
        else:
            status = "queued" if benchmark == "pawsx" else "pending"
        rows[method] = {
            "status": status,
            "raw_status": raw_status,
            "languages": values,
            "completed_languages": sum(lang in values for lang in languages),
            "train_metrics": metrics.get("train_metrics", {}),
            "progress": progress,
            "active": is_active,
        }
    return {
        "title": config["title"],
        "subtitle": config["subtitle"],
        "languages": languages,
        "methods": rows,
        "complete": all(row["status"] == "complete" for row in rows.values()),
    }


def collect(xnli_dir: Path, pawsx_dir: Path) -> dict[str, Any]:
    active = active_runs()
    benchmarks = {
        "xnli": collect_benchmark("xnli", xnli_dir, active),
        "pawsx": collect_benchmark("pawsx", pawsx_dir, active),
    }
    return {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "benchmarks": benchmarks,
        "all_complete": all(item["complete"] for item in benchmarks.values()),
    }


def pct(value: Any) -> str:
    return "—" if not isinstance(value, int | float) else f"{100 * value:.2f}"


def status_html(row: dict[str, Any], total_languages: int) -> str:
    status = row["status"]
    label = {
        "complete": "已完成",
        "evaluating": f"评测中 {row['completed_languages']}/{total_languages}",
        "training": "训练中",
        "waiting": "等待启动",
        "queued": "已排队",
        "failed": "需要重启",
        "pending": "未开始",
    }[status]
    progress = row["progress"]
    if status == "training" and "percent" in progress:
        label += f" {progress['percent']}% ({progress['step']}/{progress['total_steps']})"
    return f'<span class="pill {status}">{html.escape(label)}</span>'


def benchmark_section(benchmark: dict[str, Any]) -> str:
    methods = benchmark["methods"]
    languages = benchmark["languages"]
    base_macro = methods["base"]["languages"].get("macro")
    rows: list[str] = []
    for method in METHODS:
        row = methods[method]
        values = row["languages"]
        macro = values.get("macro")
        delta = "—"
        if isinstance(macro, int | float) and isinstance(base_macro, int | float):
            delta = f"{100 * (macro - base_macro):+.2f}"
        loss = row["train_metrics"].get("train_loss", row["progress"].get("loss"))
        cells = "".join(f"<td>{pct(values.get(lang))}</td>" for lang in languages)
        rows.append(
            f'<tr><th>{html.escape(METHOD_LABELS[method])}</th>'
            f'<td class="status-cell">{status_html(row, len(languages))}</td>'
            f'<td class="strong">{pct(macro)}</td><td>{delta}</td>'
            f'<td>{"—" if loss is None else f"{loss:.4f}"}</td>{cells}</tr>'
        )
    completed = sum(row["status"] == "complete" for row in methods.values())
    return f"""<section class="benchmark">
<div class="section-title"><div><h2>{html.escape(benchmark['title'])}</h2>
<p>{html.escape(benchmark['subtitle'])} · Seed 42 · Zero target-language updates</p></div>
<span class="count">{completed}/{len(METHODS)} methods</span></div>
<div class="table-box"><table><thead><tr><th>Method</th><th>Status</th><th>Macro ↑</th>
<th>Δ vs Base</th><th>Train loss</th>{''.join(f'<th>{lang}</th>' for lang in languages)}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>"""


def render(snapshot: dict[str, Any]) -> str:
    refresh = "" if snapshot["all_complete"] else '<meta http-equiv="refresh" content="15">'
    total_complete = sum(
        row["status"] == "complete"
        for benchmark in snapshot["benchmarks"].values()
        for row in benchmark["methods"].values()
    )
    banner = (
        "XNLI 与 PAWS-X 全部完成，自动刷新已停止。"
        if snapshot["all_complete"]
        else "XNLI 完成后按 GPU 自动接力 PAWS-X；页面每 15 秒刷新，新语言结果即时入表。"
    )
    sections = "".join(
        benchmark_section(snapshot["benchmarks"][name]) for name in ("xnli", "pawsx")
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">{refresh}
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Cross-lingual Semantic Hash Benchmarks</title>
<style>
:root{{--bg:#07111f;--panel:#0e1b2d;--text:#edf5ff;--muted:#8fa6c2;--line:#203752;--cyan:#4bd6ff;--green:#44d19d;--amber:#ffbf69;--red:#ff6b7a}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 20% 0,#102b48 0,transparent 34%),var(--bg);color:var(--text);font:14px/1.5 Inter,system-ui,-apple-system,"PingFang SC",sans-serif}}
.wrap{{max-width:1780px;margin:auto;padding:34px 26px 60px}}h1{{font-size:34px;margin:0}}.lead{{color:var(--muted);margin:7px 0 20px}}.note{{background:#0b192a;border:1px solid var(--line);border-radius:12px;padding:13px 16px;margin-bottom:26px}}.summary{{display:flex;gap:12px;margin:18px 0}}.chip,.count{{background:#13243a;border:1px solid var(--line);border-radius:999px;padding:6px 12px;color:#c9d8e9}}.benchmark{{margin:30px 0 44px}}.section-title{{display:flex;align-items:end;justify-content:space-between;margin-bottom:12px}}h2{{font-size:25px;margin:0}}.section-title p{{color:var(--muted);margin:3px 0 0}}.table-box{{overflow:auto;background:rgba(12,26,43,.92);border:1px solid var(--line);border-radius:14px}}table{{border-collapse:collapse;min-width:1120px;width:100%}}th,td{{padding:12px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}thead th{{background:#10233a;color:#9db4cf;font-size:12px}}tbody th{{text-align:left;background:#0e1d30;position:sticky;left:0}}tr:last-child td,tr:last-child th{{border-bottom:0}}.status-cell{{text-align:left}}.strong{{font-weight:800;color:var(--cyan)}}.pill{{display:inline-block;padding:4px 9px;border-radius:999px;font-size:12px;font-weight:700}}.complete{{background:#44d19d24;color:var(--green)}}.training,.evaluating{{background:#4bd6ff21;color:var(--cyan)}}.queued,.waiting,.pending{{background:#ffbf691f;color:var(--amber)}}.failed{{background:#ff6b7a21;color:var(--red)}}
</style></head><body><main class="wrap"><h1>Cross-lingual Semantic Hash Benchmarks</h1>
<div class="lead">English-source training → multilingual zero-shot evaluation</div>
<div class="summary"><span class="chip">Completed {total_complete}/12 method-runs</span><span class="chip">Updated {html.escape(snapshot['updated_at'])}</span></div>
<div class="note">{banner}</div>{sections}</main></body></html>"""


def main() -> None:
    args = parse_args()
    while True:
        snapshot = collect(args.xnli_results_dir, args.pawsx_results_dir)
        atomic_write(args.output, render(snapshot))
        atomic_write(args.output.with_suffix(".json"), json.dumps(snapshot, indent=2))
        print(f"[{snapshot['updated_at']}] cross-lingual dashboard updated", flush=True)
        if args.once or snapshot["all_complete"]:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
