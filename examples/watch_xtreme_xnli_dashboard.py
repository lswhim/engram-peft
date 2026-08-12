"""Continuously render a self-contained HTML dashboard for XTREME-XNLI runs."""

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


LANGUAGES = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi",
    "ru", "sw", "th", "tr", "ur", "vi", "zh",
]
METHODS = ["base", "full_ft", "lora", "arithmetic", "arithmetic_matched", "rq"]
LABELS = {
    "base": "Base",
    "full_ft": "Full FT",
    "lora": "LoRA",
    "arithmetic": "Arithmetic Engram",
    "arithmetic_matched": "Arithmetic (matched)",
    "rq": "Semantic-RQ Engram",
}
LOG_NAMES = {
    "base": "gpu3_base.log",
    "full_ft": "gpu2_full_ft.log",
    "lora": "gpu0_lora.log",
    "arithmetic": "gpu1_arithmetic.log",
    "arithmetic_matched": "gpu1_arithmetic_matched.log",
    "rq": "gpu3_wiki_rq.log",
}
TOTAL_LANGUAGES = len(LANGUAGES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
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


def running_methods() -> set[str]:
    running: set[str] = set()
    proc = Path("/proc")
    if not proc.exists():
        return running
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="ignore"
            )
        except OSError:
            continue
        if "run_xtreme_xnli.py" not in command:
            continue
        match = re.search(r"--method\s+([a-z_]+)", command)
        if match:
            running.add(match.group(1))
    return running


def log_progress(path: Path) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    progress_matches = list(
        re.finditer(r"(\d{1,3})%\|[^\r\n]*?\|\s*(\d+)/(\d+)", content)
    )
    losses = list(re.finditer(r"['\"]loss['\"]:\s*([0-9.eE+-]+)", content))
    result: dict[str, Any] = {}
    if progress_matches:
        match = progress_matches[-1]
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


def collect(results_dir: Path) -> dict[str, Any]:
    active = running_methods()
    methods: dict[str, Any] = {}
    for method in METHODS:
        metrics_path = results_dir / f"{method}_seed42" / "metrics.json"
        metrics = read_json(metrics_path)
        log = log_progress(results_dir / "logs" / LOG_NAMES[method])
        raw_status = str(metrics.get("status", "pending"))
        languages = metrics.get("languages", {})
        languages = languages if isinstance(languages, dict) else {}
        completed_languages = sum(lang in languages for lang in LANGUAGES)
        if raw_status == "complete":
            status = "complete"
        elif raw_status == "evaluating":
            status = "evaluating"
        elif method in active:
            status = "training"
        elif raw_status == "loading_data" and log.get("has_traceback"):
            status = "failed"
        elif raw_status == "loading_data":
            status = "waiting"
        else:
            status = "pending"
        methods[method] = {
            "status": status,
            "raw_status": raw_status,
            "languages": languages,
            "completed_languages": completed_languages,
            "train_metrics": metrics.get("train_metrics", {}),
            "wall_time_seconds": metrics.get("wall_time_seconds"),
            "peak_memory_gb": metrics.get("peak_memory_gb"),
            "progress": log,
            "active": method in active,
        }
    return {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "methods": methods,
        "all_complete": all(row["status"] == "complete" for row in methods.values()),
    }


def pct(value: Any) -> str:
    return "—" if not isinstance(value, int | float) else f"{100 * value:.2f}"


def status_html(row: dict[str, Any]) -> str:
    status = row["status"]
    names = {
        "complete": "已完成",
        "evaluating": f"评测中 {row['completed_languages']}/{TOTAL_LANGUAGES}",
        "training": "训练中",
        "waiting": "等待启动",
        "failed": "需要重启",
        "pending": "未开始",
    }
    extra = ""
    progress = row["progress"]
    if status == "training" and "percent" in progress:
        extra = f" {progress['percent']}% ({progress['step']}/{progress['total_steps']})"
    return f'<span class="pill {status}">{html.escape(names[status] + extra)}</span>'


def render(snapshot: dict[str, Any]) -> str:
    methods = snapshot["methods"]
    base_macro = methods["base"]["languages"].get("macro")
    complete_count = sum(row["status"] == "complete" for row in methods.values())
    refresh = "" if snapshot["all_complete"] else '<meta http-equiv="refresh" content="15">'
    rows: list[str] = []
    for method in METHODS:
        row = methods[method]
        values = row["languages"]
        macro = values.get("macro")
        delta = "—"
        if isinstance(macro, int | float) and isinstance(base_macro, int | float):
            delta = f"{100 * (macro - base_macro):+.2f}"
        language_cells = "".join(f"<td>{pct(values.get(lang))}</td>" for lang in LANGUAGES)
        loss = row["train_metrics"].get("train_loss")
        if loss is None:
            loss = row["progress"].get("loss")
        rows.append(
            "<tr>"
            f"<th>{html.escape(LABELS[method])}</th>"
            f"<td class=\"status-cell\">{status_html(row)}</td>"
            f"<td class=\"strong\">{pct(macro)}</td>"
            f"<td class=\"delta\">{delta}</td>"
            f"<td>{'—' if loss is None else f'{loss:.4f}'}</td>"
            f"{language_cells}</tr>"
        )
    complete_banner = (
        "全部实验与 15 语言评测均已完成，自动刷新已停止。"
        if snapshot["all_complete"]
        else "页面每 15 秒自动刷新；新语言结果写入后会自动出现。"
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">{refresh}
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic-Hash Engram · XTREME-XNLI</title>
<style>
:root{{--bg:#07111f;--panel:#0e1b2d;--panel2:#13243a;--text:#edf5ff;--muted:#8fa6c2;--line:#203752;--cyan:#4bd6ff;--green:#44d19d;--amber:#ffbf69;--red:#ff6b7a}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 20% 0,#102b48 0,transparent 34%),var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"PingFang SC",sans-serif}}
.wrap{{max-width:1780px;margin:auto;padding:38px 28px 60px}} h1{{font-size:34px;margin:0 0 7px;letter-spacing:-.7px}} .sub{{color:var(--muted);margin-bottom:26px}} .cards{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:14px;margin-bottom:20px}} .card{{background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:18px}} .label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}} .value{{font-size:26px;font-weight:750;margin-top:4px}} .note{{background:#0b192a;border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin-bottom:18px;color:#c9d8e9}}
.table-box{{overflow:auto;background:rgba(12,26,43,.92);border:1px solid var(--line);border-radius:14px}} table{{border-collapse:collapse;min-width:1540px;width:100%}} th,td{{padding:12px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} thead th{{position:sticky;top:0;background:#10233a;color:#9db4cf;font-size:12px}} tbody th{{text-align:left;background:#0e1d30;position:sticky;left:0;z-index:1}} tr:last-child td,tr:last-child th{{border-bottom:0}} .status-cell{{text-align:left}} .strong{{font-weight:800;color:var(--cyan)}} .delta{{font-weight:700}} .pill{{display:inline-block;padding:4px 9px;border-radius:999px;font-size:12px;font-weight:700}} .complete{{background:rgba(68,209,157,.14);color:var(--green)}} .training,.evaluating{{background:rgba(75,214,255,.13);color:var(--cyan)}} .waiting,.pending{{background:rgba(255,191,105,.12);color:var(--amber)}} .failed{{background:rgba(255,107,122,.13);color:var(--red)}}
.foot{{color:var(--muted);font-size:12px;margin-top:13px}} @media(max-width:850px){{.cards{{grid-template-columns:1fr 1fr}}.wrap{{padding:24px 14px}}}}
</style></head><body><main class="wrap">
<h1>Semantic-Hash Engram · XTREME-XNLI</h1>
<div class="sub">MNLI English full train (392,702) → XNLI 15-language test · Seed 42 · Zero target-language updates</div>
<section class="cards">
<div class="card"><div class="label">Completed methods</div><div class="value">{complete_count} / {len(METHODS)}</div></div>
<div class="card"><div class="label">Base macro</div><div class="value">{pct(base_macro)}%</div></div>
<div class="card"><div class="label">Best completed macro</div><div class="value">{max([float(r['languages'].get('macro', 0)) for r in methods.values()] or [0])*100:.2f}%</div></div>
<div class="card"><div class="label">Last update</div><div class="value" style="font-size:17px">{html.escape(snapshot['updated_at'])}</div></div>
</section><div class="note">{complete_banner}</div>
<div class="table-box"><table><thead><tr><th>Method</th><th>Status</th><th>Macro ↑</th><th>Δ vs Base</th><th>Train loss</th>{''.join(f'<th>{lang}</th>' for lang in LANGUAGES)}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<div class="foot">所有准确率均为百分数。Arithmetic (matched) 使用与 Semantic-RQ 相同的桶规模，用于参数公平对照。</div>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    while True:
        snapshot = collect(args.results_dir)
        atomic_write(args.output, render(snapshot))
        atomic_write(args.output.with_suffix(".json"), json.dumps(snapshot, indent=2))
        print(f"[{snapshot['updated_at']}] dashboard updated", flush=True)
        if args.once or snapshot["all_complete"]:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
