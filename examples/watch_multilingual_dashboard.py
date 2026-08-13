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
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="refresh" content="15"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic Hash · Multilingual Transfer</title><style>:root{{--bg:#f4f1e9;--card:#fffdf7;--ink:#17201d;--muted:#66716b;--line:#d8d2c5;--green:#176b4d;--blue:#245ca4}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,"PingFang SC",sans-serif}}main{{max-width:1380px;margin:auto;padding:36px 24px 80px}}header{{border-bottom:2px solid var(--ink);padding-bottom:18px;display:flex;justify-content:space-between;align-items:end}}h1{{font:38px/1 Georgia,serif;margin:4px 0}}h2{{margin:30px 0 10px}}.kicker{{color:var(--green);font-weight:800;letter-spacing:.12em}}.muted{{color:var(--muted)}}.cards,.gpus{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}}.card,.gpu{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}.card strong{{display:block;font:22px/1.15 Georgia,serif;margin:8px 0}}.gpu{{display:grid;grid-template-columns:1fr auto}}.gpu strong{{font-size:22px;color:var(--green)}}.gpu span{{grid-column:1/-1;color:var(--muted)}}.box{{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:1080px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}}th:first-child{{text-align:left}}thead th{{color:var(--muted);font-size:12px;background:#f8f5ee}}code{{color:var(--blue)}}@media(max-width:850px){{.cards,.gpus{{grid-template-columns:repeat(2,1fr)}}}}</style></head><body><main><header><div><div class="kicker">STRICT MULTILINGUAL TRANSFER · LIVE</div><h1>Semantic Hash × PAWS-X</h1><div class="muted">英语训练 → 七语言零更新测试；修正后的动态 RQ，fallback=0</div></div><div class="muted">{updated}<br>15 秒自动刷新</div></header><section class="cards"><div class="card"><b>论文定位</b><strong>跨语言知识写入/读取</strong><span class="muted">BabelEdits / MzsRE 为主任务；PAWS-X 只是实现诊断。</span></div><div class="card"><b>Engram 优势</b><strong>冻结底座 · 稀疏写 memory</strong><span class="muted">连续更新、少遗忘、可扩容和 offload 必须单独验证。</span></div><div class="card"><b>Semantic Hash 增量</b><strong>跨语言共享已写 rows</strong><span class="muted">必须胜过 Arithmetic 和 RQ-Shuffled，并通过 reset/gate 干预。</span></div><div class="card"><b>当前进度</b><strong>{completed}/5 方法完成</strong><span class="muted">PAWS-X seed42 完成后只决定是否进入知识写入 pilot。</span></div></section><h2>当前实验的准确定位</h2><div class="box"><table><tbody><tr><th>它能验证</th><td>strict dynamic RQ 可训练、七语言行为、row trace、冷/热 cache 成本</td></tr><tr><th>它不能验证</th><td>新知识写入、连续记忆、少遗忘，也不能单独支撑论文 claim</td></tr><tr><th>下一阶段</th><td>BabelEdits / MzsRE：只用源语言写入事实，跨语言零更新读取，并测 locality</td></tr><tr><th>最终因果链</th><td>语义相似 → aligned code overlap → source-updated row reuse → target propagation → shuffle/reset 后消失</td></tr></tbody></table></div><h2>四卡状态</h2><div class="gpus">{gpu_rows()}</div><h2>PAWS-X 诊断协议</h2><div class="box"><table><tbody><tr><th>训练数据</th><td>PAWS-X English train，49,401 对</td><th>测试数据</th><td>en/de/es/fr/ja/ko/zh test，各 2,000 对</td></tr><tr><th>底座</th><td>Qwen3-1.7B-Base</td><th>更新参数</th><td>Engram-only；LoRA 为外部对照</td></tr><tr><th>公平对照</th><td>Arithmetic-fixed：8 heads × 256 rows</td><th>Semantic-RQ</th><td>M=8, K=256；首次 miss 在线编码并 cache</td></tr></tbody></table></div><h2>训练进度与七语言 Accuracy（%）</h2><div class="box"><table><thead><tr><th>方法</th><th>状态</th><th>GPU</th><th>PID</th><th>Step</th><th>Loss</th>{''.join(f'<th>{x}</th>' for x in LANGUAGES)}<th>Macro</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><h2>Semantic-RQ 的英语训练 Row 跨语言复用</h2><p class="muted">集合 overlap 容易因小表容量饱和，因此同时报告测试访问质量落在英语已训练 rows 的比例和频率直方图交集。正式 BabelEdits/MzsRE 将使用同一事实配对的 aligned code overlap，而非只用全局集合重合。</p><div class="box"><table><thead><tr><th>语言</th><th>共享 rows</th><th>测试 unique-row reuse %</th><th>测试访问质量 reuse %</th><th>频率交集 %</th></tr></thead><tbody>{''.join(reuse_rows)}</tbody></table></div><h2>正式论文实验框架</h2><div class="cards"><div class="card"><b>Table 1</b><strong>跨语言知识迁移</strong><span class="muted">BabelEdits + MzsRE：efficacy、propagation、aliases、locality。</span></div><div class="card"><b>Table 2</b><strong>持续知识写入</strong><span class="muted">WikiBigEdit：1/10/100/1k/10k edits retention 与遗忘。</span></div><div class="card"><b>Figure 1</b><strong>容量—迁移—干扰</strong><span class="muted">K=64/256/1024/4096，展示 Semantic sharing 何时 work。</span></div><div class="card"><b>Figure 2</b><strong>因果机制</strong><span class="muted">RQ-Shuffled、shared-row reset、gate intervention。</span></div></section><p class="muted">当前不能声称 Semantic-RQ 提升跨语言泛化；PAWS-X 结果仅作为诊断。正式结论由 BabelEdits/MzsRE 与 WikiBigEdit 决定。</p></main></body></html>'''


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
