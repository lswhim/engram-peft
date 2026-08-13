#!/usr/bin/env python
"""Live dashboard for the full, canonical-only knowledge-editing suite."""

from __future__ import annotations

import argparse, html, json, os, re, subprocess, time
from datetime import datetime
from pathlib import Path

RUNS = [
    ("CounterFact", "Arithmetic-fixed", "logs/standard_cf_arithmetic_seed42.log", "running · source path verified"),
    ("CounterFact", "LoRA", "logs/standard_cf_lora_seed42.log", "running · source path verified"),
    ("ZsRE", "Arithmetic-fixed", "logs/standard_zsre_arithmetic_seed42_v2.log", "diagnostic only · 73.8 epochs"),
    ("CounterFact", "Semantic-RQ", "logs/standard_cf_rq_seed42.log", "queued after RQ table"),
    ("ZsRE", "Arithmetic-fixed · matched", "logs/standard_zsre_arithmetic_matched_seed42.log", "queued · 204 steps ≈ 5 epochs"),
    ("ZsRE", "Semantic-RQ · matched", "logs/standard_zsre_rq_matched_seed42.log", "queued · 204 steps ≈ 5 epochs"),
    ("ZsRE", "LoRA · matched", "logs/standard_zsre_lora_matched_seed42.log", "queued · 204 steps ≈ 5 epochs"),
]

def read(path: Path) -> str:
    try: return path.read_bytes()[-240_000:].decode(errors="ignore")
    except OSError: return ""

def progress(path: Path) -> tuple[str, str]:
    text = read(path)
    steps = re.findall(r"(\d+)/3000", text)
    losses = re.findall(r"['\"]loss['\"]:\s*([0-9.eE+-]+)", text)
    if "Traceback (most recent call last)" in text: return "failed", losses[-1] if losses else "—"
    return (steps[-1] + "/3000" if steps else "—", losses[-1] if losses else "—")

def processes() -> list[tuple[str, str, str]]:
    out=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            cmd=' '.join(x.decode(errors='ignore') for x in (p/'cmdline').read_bytes().split(b'\0') if x)
            env=(p/'environ').read_bytes().split(b'\0')
        except OSError: continue
        if not any(x in cmd for x in ('compare_engram_lora.py','build_rq_table.py','evaluate_standard_ke.py')): continue
        gpu=next((x.split(b'=',1)[1].decode() for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES=')), '—')
        out.append((p.name,gpu,cmd[-170:]))
    return out

def gpus() -> str:
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    except Exception: return '<div>不可用</div>'
    return ''.join(f'<div class="card"><b>GPU {a.strip()}</b><strong>{c.strip()}%</strong><span>{b.strip()} MiB</span></div>' for a,b,c in (x.split(',') for x in raw.splitlines()))

def result_rows(root: Path) -> str:
    rows=[]
    for path in sorted(root.glob('*.json')):
        try: d=json.loads(path.read_text())
        except Exception: continue
        m=d.get('metrics',{})
        rows.append('<tr><th>%s</th><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
            html.escape(path.stem), d.get('examples','—'), *(f"{100*m.get(k):.2f}" if isinstance(m.get(k),(int,float)) else '—' for k in ('efficacy','paraphrase','specificity','harmonic_score'))))
    return ''.join(rows) or '<tr><td colspan="6">正式全集评测尚未完成；这里不会填入 smoke-test 数字。</td></tr>'

def render(root: Path) -> str:
    run_rows=''.join(f'<tr><th>{d}</th><td>{m}</td><td>{s}</td><td>{progress(Path(l))[0]}</td><td>{progress(Path(l))[1]}</td><td><code>{l}</code></td></tr>' for d,m,l,s in RUNS)
    proc_rows=''.join(f'<tr><th>{p}</th><td>{g}</td><td class="cmd">{html.escape(c)}</td></tr>' for p,g,c in processes())
    build=read(Path('logs/build_rq_M8K1024_300k.log'))
    stage='complete' if 'meta.json' in build or '[done]' in build.lower() else ('RQ fitting / writing' if '[rq]' in build else 'embedding')
    now=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="refresh" content="15"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic Hash · Formal Benchmarks</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f1e9;color:#17201d;font:14px/1.55 system-ui,"PingFang SC",sans-serif}}main{{max-width:1400px;margin:auto;padding:34px 24px 80px}}header{{display:flex;justify-content:space-between;border-bottom:2px solid;padding-bottom:16px}}h1{{font:38px Georgia;margin:4px 0}}h2{{margin:28px 0 10px}}.kicker{{color:#176b4d;font-weight:800;letter-spacing:.12em}}.muted{{color:#66716b}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}}.card,.box{{background:#fffdf7;border:1px solid #d8d2c5;border-radius:12px}}.card{{padding:15px}}.card strong{{display:block;font:23px Georgia;color:#176b4d}}.box{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:900px}}th,td{{padding:10px 12px;border-bottom:1px solid #ddd6c8;text-align:left}}thead th{{font-size:12px;color:#66716b;background:#f8f5ee}}.bad{{color:#9b2c2c}}code{{color:#245ca4}}.cmd{{font:12px ui-monospace;max-width:900px}}@media(max-width:850px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}</style></head><body><main><header><div><div class="kicker">FULL OFFICIAL BENCHMARKS · LIVE</div><h1>Semantic-Hash Engram</h1><div class="muted">canonical-only 写入；dataset-specific official scoring；evaluation-only paraphrase/locality</div></div><div class="muted">{now}<br>15 秒刷新</div></header><section class="cards"><div class="card"><b>CounterFact</b><strong>21,919 cases</strong><span>ES / PS / NS / Harmonic</span></div><div class="card"><b>ZsRE</b><strong>1,301 edits</strong><span>EA / PA / NA / Harmonic</span></div><div class="card"><b>RQ 表</b><strong>{stage}</strong><span>M=8, K=1024, FineWeb-Edu 300k/order</span></div><div class="card"><b>结果门槛</b><strong>No smoke tests</strong><span>只有完整官方 split 才进入正式结果表</span></div></section><h2>实验协议</h2><div class="box"><table><tbody><tr><th>写入 setting</th><td>完整 benchmark canonical edits 批量写入同一 memory；不是逐样本重建模型</td></tr><tr><th>训练隔离</th><td>仅 canonical prompt → new target；prompt token 全 mask；paraphrase/locality evaluation-only</td></tr><tr><th>训练预算</th><td>统一约 5 次完整数据遍历：CounterFact 3000 steps≈4.38 epochs；ZsRE 204 steps≈5.02 epochs</td></tr><tr><th>CounterFact scoring</th><td>官方完整答案逐 token 平均 NLL：new 优于 true；报告 ES / PS / NS</td></tr><tr><th>ZsRE scoring</th><td>官方 teacher-forced 逐 token accuracy；报告 EA / PA / NA</td></tr><tr><th>比较</th><td>Base / LoRA / Arithmetic-fixed / Semantic-RQ；同底座、层、容量、数据遍历数与 seed</td></tr><tr><th>有效性审计</th><td class="bad">旧静态 100k+fallback、首 token、训练混入 paraphrase、ZsRE 3000-step(73.8 epochs) 均不进主表</td></tr></tbody></table></div><h2>GPU</h2><div class="cards">{gpus()}</div><h2>训练与队列</h2><div class="box"><table><thead><tr><th>Benchmark</th><th>方法</th><th>状态</th><th>Step</th><th>Loss</th><th>日志</th></tr></thead><tbody>{run_rows}</tbody></table></div><h2>正式结果（%）</h2><div class="box"><table><thead><tr><th>Run</th><th>N</th><th>Efficacy</th><th>Paraphrase</th><th>Specificity</th><th>Harmonic</th></tr></thead><tbody>{result_rows(root)}</tbody></table></div><h2>真实进程</h2><div class="box"><table><thead><tr><th>PID</th><th>GPU</th><th>Command</th></tr></thead><tbody>{proc_rows}</tbody></table></div></main></body></html>'''

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('outputs/standard_ke')); p.add_argument('--output',type=Path,required=True); p.add_argument('--interval',type=float,default=15); a=p.parse_args()
    while True:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        tmp=a.output.with_suffix('.tmp'); tmp.write_text(render(a.root)); os.replace(tmp,a.output)
        snapshot={"updated_at":datetime.now().astimezone().isoformat(),"all_complete":False,"kind":"formal_standard_ke"}
        json_tmp=a.output.with_suffix('.json.tmp'); json_tmp.write_text(json.dumps(snapshot)); os.replace(json_tmp,a.output.with_suffix('.json'))
        time.sleep(a.interval)
if __name__=='__main__': main()
