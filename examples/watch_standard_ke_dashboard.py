#!/usr/bin/env python
"""Live dashboard for the full, canonical-only knowledge-editing suite."""

from __future__ import annotations

import argparse, html, json, os, re, sqlite3, statistics, subprocess, time
from datetime import datetime
from pathlib import Path

RUNS = [
    (dataset, method, seed, f'logs/formal_v3_{short}_{slug}_seed{seed}.log')
    for dataset,short in (("CounterFact","cf"),("ZsRE","zsre"))
    for method,slug in (("Arithmetic-fixed","arithmetic"),("LoRA","lora"),("Semantic-RQ","rq"))
    for seed in (42,43,44)
]

METHOD_LABELS = {
    "base": "Base",
    "lora": "LoRA",
    "arithmetic": "Arithmetic-fixed",
    "rq": "Semantic-RQ",
}

SCORING_LABELS = {
    "full_target_mean_token_log_likelihood": "完整答案 mean token log-likelihood",
    "teacher_forced_token_accuracy": "teacher-forced token accuracy",
}

def read(path: Path) -> str:
    try: return path.read_bytes()[-240_000:].decode(errors="ignore")
    except OSError: return ""

def progress(path: Path, eval_path: Path | None = None, result_path: Path | None = None) -> tuple[str, str]:
    text = read(path)
    steps = re.findall(r"(\d+)/(205|345|3000)", text)
    losses = re.findall(r"['\"]loss['\"]:\s*([0-9.eE+-]+)", text)
    if "Traceback (most recent call last)" in text: return "failed", losses[-1] if losses else "—"
    if result_path is not None:
        try: result=json.loads(result_path.read_text())
        except Exception: result={}
        if result.get('status') == 'complete' and result.get('complete_official_split'):
            return "official eval complete", losses[-1] if losses else "—"
    eval_text=read(eval_path) if eval_path is not None else ""
    eval_steps=re.findall(r"\[(\d+)/(1301|2191)\]",eval_text)
    if "Traceback (most recent call last)" in eval_text:
        return "eval failed", losses[-1] if losses else "—"
    if eval_steps:
        return "eval " + "/".join(eval_steps[-1]), losses[-1] if losses else "—"
    if "Summary Table" in text and steps:
        return f"train {steps[-1][1]}/{steps[-1][1]} · eval queued", losses[-1] if losses else "—"
    return ("/".join(steps[-1]) if steps else "—", losses[-1] if losses else "—")

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

def rq_cache_status() -> str:
    base=Path('rq_tables/fineweb_M8K1024_300k_strict')
    parts=[]
    for seed,name in ((42,'runtime_cache_v3.sqlite'),(43,'runtime_cache_v3_seed43'),(44,'runtime_cache_v3_seed44')):
        db=base/name/'semantic_codes.sqlite3'
        counts={2:0,3:0}
        if db.exists():
            try:
                con=sqlite3.connect(f'file:{db}?mode=ro',uri=True,timeout=.2)
                counts.update({int(n):int(c) for n,c in con.execute('SELECT n, COUNT(*) FROM codes GROUP BY n')})
                con.close()
            except Exception: pass
        parts.append(f'seed{seed}: 2g={counts[2]:,}, 3g={counts[3]:,}')
    return ' · '.join(parts)

def result_rows(root: Path) -> str:
    rows=[]
    for path in sorted(root.glob('*.json')):
        try: d=json.loads(path.read_text())
        except Exception: continue
        if d.get('status') != 'complete' or not d.get('complete_official_split'):
            continue
        m=d.get('metrics',{})
        dataset=d.get('dataset', path.stem.split('_',1)[0]).lower()
        method=next((name for key,name in METHOD_LABELS.items() if f'_{key}_' in f'_{path.stem}_'), path.stem)
        seed_match=re.search(r'_seed(\d+)$',path.stem); seed=seed_match.group(1) if seed_match else '—'
        metric_names=('ES','PS','NS') if dataset == 'counterfact' else ('EA','PA','NA')
        den=d.get('denominators',{})
        denominator='<br>'.join(f'{name}: {den.get(key,"—"):,}' if isinstance(den.get(key),(int,float)) else f'{name}: —' for name,key in zip(metric_names,('efficacy','paraphrase','specificity')))
        score=SCORING_LABELS.get(d.get('scoring'), d.get('scoring','—'))
        rows.append('<tr><th>%s</th><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td class="small">%s</td></tr>' % (
            'CounterFact' if dataset == 'counterfact' else 'ZsRE', html.escape(method), seed, d.get('examples','—'),
            *(f"{100*m.get(k):.2f}" if isinstance(m.get(k),(int,float)) else '—' for k in ('efficacy','paraphrase','specificity','harmonic_score')),
            denominator, html.escape(score)))
    return ''.join(rows) or '<tr><td colspan="10">正式全集评测尚未完成；这里不会填入 smoke-test 数字。</td></tr>'

def aggregate_rows(root: Path) -> str:
    rows=[]
    for dataset in ('counterfact','zsre'):
        for method in ('arithmetic','lora','rq'):
            runs=[]
            for seed in (42,43,44):
                try: d=json.loads((root/f'{dataset}_{method}_seed{seed}.json').read_text())
                except Exception: continue
                if d.get('status') == 'complete' and d.get('complete_official_split'):
                    runs.append(d['metrics'])
            values=[]
            for key in ('efficacy','paraphrase','specificity','harmonic_score'):
                xs=[100*r[key] for r in runs]
                values.append(f'{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f}' if len(xs) == 3 else '等待 3 seeds')
            rows.append(f'<tr><th>{"CounterFact" if dataset=="counterfact" else "ZsRE"}</th><td>{METHOD_LABELS[method]}</td><td>{len(runs)}/3</td>'+''.join(f'<td>{v}</td>' for v in values)+'</tr>')
    return ''.join(rows)

def conclusions(root: Path) -> str:
    results={}
    for path in root.glob('*.json'):
        try: d=json.loads(path.read_text())
        except Exception: continue
        if d.get('status') == 'complete' and d.get('complete_official_split'):
            results[path.stem]=d
    items=[]
    for ds,label in [('counterfact','CounterFact'),('zsre','ZsRE')]:
        triples={}
        for method in ('arithmetic','lora'):
            runs=[]
            for seed in (42,43,44):
                d=results.get(f'{ds}_{method}_seed{seed}')
                if d: runs.append(d['metrics'])
            if len(runs) == 3:
                triples[method]={k:(100*statistics.mean(xs:=[r[k] for r in runs]),100*statistics.stdev(xs)) for k in ('efficacy','paraphrase','specificity','harmonic_score')}
        if len(triples) == 2:
            a,l=triples['arithmetic'],triples['lora']
            fmt=lambda x:f'{x[0]:.2f}±{x[1]:.2f}'
            items.append(f'<li><b>{label} 三 seed：</b>Arithmetic→LoRA 的 efficacy 为 {fmt(a["efficacy"])}→{fmt(l["efficacy"])}，paraphrase 为 {fmt(a["paraphrase"])}→{fmt(l["paraphrase"])}，specificity 为 {fmt(a["specificity"])}→{fmt(l["specificity"])}，harmonic 为 {fmt(a["harmonic_score"])}→{fmt(l["harmonic_score"])}。</li>')
            continue
        b=results.get(f'{ds}_base_seed42'); a=results.get(f'{ds}_arithmetic_seed42'); l=results.get(f'{ds}_lora_seed42')
        if b and a and l:
            bm,am,lm=b['metrics'],a['metrics'],l['metrics']
            items.append(f'<li><b>{label}：</b>Base→Arithmetic→LoRA 的 efficacy 为 {100*bm["efficacy"]:.2f}→{100*am["efficacy"]:.2f}→{100*lm["efficacy"]:.2f}，paraphrase 为 {100*bm["paraphrase"]:.2f}→{100*am["paraphrase"]:.2f}→{100*lm["paraphrase"]:.2f}，specificity 为 {100*bm["specificity"]:.2f}→{100*am["specificity"]:.2f}→{100*lm["specificity"]:.2f}。当前信号是写入/泛化增强伴随局部保持代价。</li>')
    rq_done=all((root/f'{ds}_rq_seed42.json').exists() for ds in ('counterfact','zsre'))
    if not rq_done:
        items.append('<li><b>Semantic-RQ 主结论尚未形成：</b>RQ 的完整官方评测未全部完成，当前不能据 Arithmetic/LoRA 推断 semantic hash 有效或无效。</li>')
    items.append('<li><b>效率口径：</b>RQ seeds 42/43 在进程启动时使用逐 key SQLite 热查询实现，只用于效果；seed44 起启用等价的进程内地址 cache。最终 latency/throughput 将用同一代码、同一已预热 cache 单独复测，不混用当前训练 wall-clock。</li>')
    complete_seed_counts=[]
    for ds in ('counterfact','zsre'):
        for method in ('arithmetic','lora','rq'):
            n=sum(f'{ds}_{method}_seed{seed}' in results for seed in (42,43,44))
            complete_seed_counts.append(n)
    if min(complete_seed_counts, default=0) < 3:
        items.append('<li><b>统计边界：</b>三 seed 矩阵仍不完整；逐 run 数字可用于运行审计，但方法排序只在对应方法 3/3 seeds 到齐后报告 mean ± sample std。</li>')
    else:
        items.append('<li><b>统计状态：</b>核心方法三 seed 已齐，汇总表报告 mean ± sample std；Base 仅作为固定预训练参照，不参与随机种子方差。</li>')
    return ''.join(items)

def completion(root: Path) -> str:
    complete,total=completion_counts(root)
    return f'{complete}/{total} complete'

def completion_counts(root: Path) -> tuple[int, int]:
    expected={f'{dataset}_{method}_seed{seed}.json' for dataset in ('counterfact','zsre') for method in ('arithmetic','lora','rq') for seed in (42,43,44)} | {f'{dataset}_base_seed42.json' for dataset in ('counterfact','zsre')}
    complete=0
    for name in expected:
        try: d=json.loads((root/name).read_text())
        except Exception: continue
        complete += d.get('status') == 'complete' and d.get('complete_official_split') is True
    return complete,len(expected)

def render(root: Path) -> str:
    run_rows=''
    for d,m,s,l in RUNS:
        dataset='counterfact' if d == 'CounterFact' else 'zsre'
        slug=next(key for key,name in METHOD_LABELS.items() if name == m)
        eval_log=Path(f'logs/formal_v3_eval_{"cf" if dataset=="counterfact" else "zsre"}_{slug}_seed{s}.log')
        result=root/f'{dataset}_{slug}_seed{s}.json'
        state,loss=progress(Path(l),eval_log,result)
        run_rows+=f'<tr><th>{d}</th><td>{m}</td><td>{s}</td><td>{state}</td><td>{loss}</td><td><code>{l}</code><br><code>{eval_log}</code></td></tr>'
    proc_rows=''.join(f'<tr><th>{p}</th><td>{g}</td><td class="cmd">{html.escape(c)}</td></tr>' for p,g,c in processes())
    build=read(Path('logs/build_rq_M8K1024_300k.log'))
    stage='complete' if 'meta.json' in build or '[done]' in build.lower() else ('RQ fitting / writing' if '[rq]' in build else 'embedding')
    now=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="refresh" content="15"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic Hash · Formal Benchmarks</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f1e9;color:#17201d;font:14px/1.55 system-ui,"PingFang SC",sans-serif}}main{{max-width:1400px;margin:auto;padding:34px 24px 80px}}header{{display:flex;justify-content:space-between;border-bottom:2px solid;padding-bottom:16px}}h1{{font:38px Georgia;margin:4px 0}}h2{{margin:28px 0 10px}}.kicker{{color:#176b4d;font-weight:800;letter-spacing:.12em}}.muted{{color:#66716b}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}}.card,.box{{background:#fffdf7;border:1px solid #d8d2c5;border-radius:12px}}.card{{padding:15px}}.card strong{{display:block;font:23px Georgia;color:#176b4d}}.box{{overflow:auto;padding:0}}.notes{{padding:12px 22px}}table{{border-collapse:collapse;width:100%;min-width:900px}}th,td{{padding:10px 12px;border-bottom:1px solid #ddd6c8;text-align:left;vertical-align:top}}thead th{{font-size:12px;color:#66716b;background:#f8f5ee}}.bad{{color:#9b2c2c}}code{{color:#245ca4}}.cmd{{font:12px ui-monospace;max-width:900px}}.small{{font-size:12px;color:#59635e}}a{{color:#176b4d}}@media(max-width:850px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}</style></head><body><main><header><div><div class="kicker">FULL OFFICIAL BENCHMARKS · LIVE</div><h1>Semantic-Hash Engram</h1><div class="muted">canonical-only target supervision；dataset-specific official scoring；evaluation-only paraphrase/locality</div></div><div class="muted">{now}<br>15 秒刷新</div></header><section class="cards"><div class="card"><b>CounterFact</b><strong>2,191 cases</strong><span>ES / PS / NS / Harmonic</span></div><div class="card"><b>ZsRE</b><strong>1,301 edits</strong><span>EA / PA / NA / Harmonic</span></div><div class="card"><b>正式矩阵</b><strong>{completion(root)}</strong><span>3 seeds × core methods；Base seed42</span></div><div class="card"><b>结果门槛</b><strong>No smoke tests</strong><span>只有完整官方 split 才进入正式结果表</span></div></section><h2>实验协议</h2><div class="box"><table><tbody><tr><th>写入 setting</th><td>完整 benchmark canonical edits 批量写入同一 memory；不是逐样本重建模型</td></tr><tr><th>训练隔离</th><td>仅 canonical prompt → new target；prompt labels=-100；paraphrase/locality evaluation-only</td></tr><tr><th>训练预算</th><td>统一 5 次数据遍历：CounterFact 345 optimizer steps；ZsRE 205 optimizer steps；effective batch=32</td></tr><tr><th>CounterFact scoring</th><td>沿用 ROME/CounterFact 的 ES / PS / NS 指标语义；为正确覆盖多 token 答案，比较完整 target 的 mean token log-likelihood，而非旧实现的首 token shortcut</td></tr><tr><th>ZsRE scoring</th><td>主流 locate-then-edit 口径：teacher-forced 逐 target-token accuracy；EA / PA / NA 的分母是 target tokens</td></tr><tr><th>RQ 地址</th><td>{stage}；FineWeb-Edu 2/3-gram 各 300k，M=8/K=1024；unseen n-gram 在线 embedding→RQ，首次写入持久 cache，后续直接命中</td></tr><tr><th>运行时语义地址</th><td>{rq_cache_status()}；这些是离线表外、经 encoder→RQ 后持久化的唯一 n-gram 数，不是 Arithmetic fallback</td></tr><tr><th>比较</th><td>Base / LoRA / Arithmetic-fixed / Semantic-RQ；同底座、层、容量、数据遍历数；核心方法 seeds 42/43/44</td></tr><tr><th>有效性审计</th><td class="bad">v1/v2：静态 fallback、首 token、prompt-label 覆盖、legacy hash 覆盖、错误数据规模/预算，全部排除</td></tr></tbody></table></div><h2>文献与官方实现依据</h2><div class="box notes"><ul><li><a href="https://arxiv.org/abs/2202.05262">ROME / CounterFact paper</a>：以 efficacy、paraphrase generalization 与 neighborhood specificity 评价是否真正改写事实而非复述训练句。</li><li><a href="https://github.com/kmeng01/rome/blob/main/experiments/py/eval_utils_counterfact.py">ROME CounterFact evaluator</a>：ES / PS / NS 的官方参考实现。</li><li><a href="https://github.com/kmeng01/rome/blob/main/experiments/py/eval_utils_zsre.py">ROME ZsRE evaluator</a>：teacher-forced target-token accuracy 的官方参考实现。</li><li><a href="https://github.com/zjunlp/EasyEdit">EasyEdit</a>：当前知识编辑文献常用的 reliability / generalization / locality 统一框架。这里保留 benchmark 原生名称和分母，不混用生成式 EM。</li></ul><p class="small">“官方”指指标定义与 benchmark prompt 集合；本实验为 Qwen3 和多 token target 做了透明的完整答案扩展，不声称逐字复刻只针对旧 GPT 架构的脚本。</p></div><h2>当前可支持的结论</h2><div class="box notes"><ul>{conclusions(root)}</ul></div><h2>GPU</h2><div class="cards">{gpus()}</div><h2>三 Seed 正式汇总（mean ± sample std，%）</h2><div class="box"><table><thead><tr><th>Benchmark</th><th>方法</th><th>Seeds</th><th>Efficacy</th><th>Paraphrase</th><th>Specificity</th><th>Harmonic</th></tr></thead><tbody>{aggregate_rows(root)}</tbody></table></div><h2>训练与队列</h2><div class="box"><table><thead><tr><th>Benchmark</th><th>方法</th><th>Seed</th><th>Step</th><th>Loss</th><th>日志</th></tr></thead><tbody>{run_rows}</tbody></table></div><h2>正式完整结果（%）</h2><div class="box"><table><thead><tr><th>Benchmark</th><th>方法</th><th>Seed</th><th>Cases</th><th>Efficacy<br>ES / EA</th><th>Paraphrase<br>PS / PA</th><th>Specificity<br>NS / NA</th><th>Harmonic</th><th>实际分母</th><th>核心评分口径</th></tr></thead><tbody>{result_rows(root)}</tbody></table></div><h2>真实进程</h2><div class="box"><table><thead><tr><th>PID</th><th>GPU</th><th>Command</th></tr></thead><tbody>{proc_rows}</tbody></table></div></main></body></html>'''

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('outputs/standard_ke_v3')); p.add_argument('--output',type=Path,required=True); p.add_argument('--interval',type=float,default=15); a=p.parse_args()
    while True:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        tmp=a.output.with_suffix('.tmp'); tmp.write_text(render(a.root)); os.replace(tmp,a.output)
        complete,total=completion_counts(a.root)
        snapshot={"updated_at":datetime.now().astimezone().isoformat(),"all_complete":complete==total,"completed":complete,"expected":total,"kind":"formal_standard_ke"}
        json_tmp=a.output.with_suffix('.json.tmp'); json_tmp.write_text(json.dumps(snapshot)); os.replace(json_tmp,a.output.with_suffix('.json'))
        time.sleep(a.interval)
if __name__=='__main__': main()
