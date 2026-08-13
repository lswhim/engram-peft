"""Render the strict PAWS-X multilingual transfer experiment dashboard."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


LANGUAGES = ("en", "de", "es", "fr", "ja", "ko", "zh")
METHODS = (
    ("base", "Base"),
    ("arithmetic_matched", "Arithmetic-fixed"),
    ("rq", "Semantic-RQ (strict dynamic)"),
    ("lora", "LoRA"),
    ("rq_shuffled", "RQ-Shuffled (pending)"),
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def pct(value: Any) -> str:
    return f"{100 * float(value):.2f}" if isinstance(value, (int, float)) else "—"


def progress(log: Path) -> tuple[str, str]:
    try:
        text = log.read_bytes()[-200_000:].decode(errors="ignore")
    except OSError:
        return "—", "—"
    steps = re.findall(r"(\d+)/(1544)", text)
    losses = re.findall(r"['\"]loss['\"]:\s*([0-9.eE+-]+)", text)
    return ("/".join(steps[-1]) if steps else "—", losses[-1] if losses else "—")


def jobs() -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            parts = [x.decode(errors="ignore") for x in (entry / "cmdline").read_bytes().split(b"\0") if x]
            env = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        command = " ".join(parts)
        if "run_xtreme_pawsx.py" not in command:
            continue
        match = re.search(r"--method\s+(\S+)", command)
        if not match:
            continue
        gpu = next((x.split(b"=", 1)[1].decode() for x in env if x.startswith(b"CUDA_VISIBLE_DEVICES=")), "?")
        found[match.group(1)] = {"pid": entry.name, "gpu": gpu}
    return found


def gpu_rows() -> str:
    try:
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True)
    except Exception:
        return ""
    cards = []
    for line in raw.splitlines():
        index, used, total, util = [x.strip() for x in line.split(",")]
        cards.append(f'<div class="gpu"><b>GPU {index}</b><strong>{util}%</strong><span>{used}/{total} MiB</span></div>')
    return "".join(cards)


def render(root: Path) -> str:
    active = jobs()
    rows = []
    reuse_rows = []
    completed = 0
    for method, label in METHODS:
        payload = read_json(root / f"{method}_seed42" / "metrics.json")
        status = str(payload.get("status", "等待"))
        if status == "complete":
            completed += 1
        job = active.get(method, {})
        step, loss = progress(Path("logs") / f"pawsx_{'arithmetic' if method == 'arithmetic_matched' else method}_seed42.log")
        languages = payload.get("languages", {}) if isinstance(payload.get("languages"), dict) else {}
        cells = "".join(f"<td>{pct(languages.get(lang))}</td>" for lang in LANGUAGES)
        macro = languages.get("macro")
        rows.append(f"<tr><th>{label}</th><td>{status}</td><td>{job.get('gpu','—')}</td><td>{job.get('pid','—')}</td><td>{step}</td><td>{loss}</td>{cells}<td>{pct(macro)}</td></tr>")
        reuse = payload.get("row_reuse", {}) if isinstance(payload.get("row_reuse"), dict) else {}
        if method == "rq":
            for lang in LANGUAGES:
                value = reuse.get(lang, {}) if isinstance(reuse.get(lang), dict) else {}
                reuse_rows.append(
                    f"<tr><th>{lang}</th><td>{value.get('shared_rows','—')}</td>"
                    f"<td>{pct(value.get('test_row_reuse_rate'))}</td>"
                    f"<td>{pct(value.get('test_access_mass_on_train_rows'))}</td>"
                    f"<td>{pct(value.get('frequency_histogram_intersection'))}</td></tr>"
                )
    updated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="refresh" content="15"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic Hash · Multilingual Transfer</title><style>:root{{--bg:#f4f1e9;--card:#fffdf7;--ink:#17201d;--muted:#66716b;--line:#d8d2c5;--green:#176b4d;--blue:#245ca4}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,"PingFang SC",sans-serif}}main{{max-width:1380px;margin:auto;padding:36px 24px 80px}}header{{border-bottom:2px solid var(--ink);padding-bottom:18px;display:flex;justify-content:space-between;align-items:end}}h1{{font:38px/1 Georgia,serif;margin:4px 0}}h2{{margin:30px 0 10px}}.kicker{{color:var(--green);font-weight:800;letter-spacing:.12em}}.muted{{color:var(--muted)}}.cards,.gpus{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}}.card,.gpu{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}.card strong{{display:block;font:22px/1.15 Georgia,serif;margin:8px 0}}.gpu{{display:grid;grid-template-columns:1fr auto}}.gpu strong{{font-size:22px;color:var(--green)}}.gpu span{{grid-column:1/-1;color:var(--muted)}}.box{{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:1080px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}}th:first-child{{text-align:left}}thead th{{color:var(--muted);font-size:12px;background:#f8f5ee}}code{{color:var(--blue)}}@media(max-width:850px){{.cards,.gpus{{grid-template-columns:repeat(2,1fr)}}}}</style></head><body><main><header><div><div class="kicker">STRICT MULTILINGUAL TRANSFER · LIVE</div><h1>Semantic Hash × PAWS-X</h1><div class="muted">英语训练 → 七语言零更新测试；修正后的动态 RQ，fallback=0</div></div><div class="muted">{updated}<br>15 秒自动刷新</div></header><section class="cards"><div class="card"><b>当前主问题</b><strong>英语 memory 能否跨语言读取？</strong><span class="muted">准确率和 memory-row reuse 必须同时成立。</span></div><div class="card"><b>有效实现</b><strong>动态 encode → RQ → cache</strong><span class="muted">任意新 n-gram 均取得语义地址，不回退 Arithmetic。</span></div><div class="card"><b>旧结果</b><strong>全部作废</strong><span class="muted">旧 PAWS-X 使用错误 packed-code 解码与表外 fallback。</span></div><div class="card"><b>当前进度</b><strong>{completed}/5 方法完成</strong><span class="muted">Seed 42 第一轮；正向后再补三 seed。</span></div></section><h2>四卡状态</h2><div class="gpus">{gpu_rows()}</div><h2>实验协议</h2><div class="box"><table><tbody><tr><th>训练数据</th><td>PAWS-X English train，49,401 对</td><th>测试数据</th><td>en/de/es/fr/ja/ko/zh test，各 2,000 对</td></tr><tr><th>底座</th><td>Qwen3-1.7B-Base</td><th>更新参数</th><td>Engram-only；LoRA 为外部对照</td></tr><tr><th>公平对照</th><td>Arithmetic-fixed：8 heads × 256 rows</td><th>Semantic-RQ</th><td>M=8, K=256；首次 miss 在线编码并 cache</td></tr></tbody></table></div><h2>训练进度与七语言 Accuracy（%）</h2><div class="box"><table><thead><tr><th>方法</th><th>状态</th><th>GPU</th><th>PID</th><th>Step</th><th>Loss</th>{''.join(f'<th>{x}</th>' for x in LANGUAGES)}<th>Macro</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><h2>Semantic-RQ 的英语训练 Row 跨语言复用</h2><p class="muted">集合 overlap 容易因小表容量饱和，因此同时报告测试访问质量落在英语已训练 rows 的比例和频率直方图交集。仅 accuracy 提升、没有 row reuse 证据，不能称为 Engram memory transfer。</p><div class="box"><table><thead><tr><th>语言</th><th>共享 rows</th><th>测试 unique-row reuse %</th><th>测试访问质量 reuse %</th><th>频率交集 %</th></tr></thead><tbody>{''.join(reuse_rows)}</tbody></table></div><h2>判定标准</h2><div class="cards"><div class="card"><b>效果</b><strong>RQ > Arithmetic</strong><span class="muted">非英语 macro 为主，英语 accuracy 必报。</span></div><div class="card"><b>机制</b><strong>跨语言 row reuse</strong><span class="muted">复用率与逐语言 gain 方向一致。</span></div><div class="card"><b>反事实</b><strong>还需 RQ-Shuffled</strong><span class="muted">动态 strict shuffled 完成前不归因于语义几何。</span></div><div class="card"><b>统计</b><strong>正向后补 3 seeds</strong><span class="muted">单 seed 只做 go/no-go，不形成论文结论。</span></div></section><p class="muted">当前不能声称 Semantic-RQ 提升跨语言泛化；页面只展示修正实现后的运行状态与可验证结果。</p></main></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/pawsx_strict"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    while True:
        content = render(args.root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(content)
        os.replace(temporary, args.output)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
