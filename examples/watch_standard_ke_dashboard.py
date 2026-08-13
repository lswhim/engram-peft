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
    for seed,name in ((42,'runtime_cache_v3_seed42'),(43,'runtime_cache_v3_seed43'),(44,'runtime_cache_v3_seed44')):
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

def geometry_analysis(root: Path) -> str:
    path=root/'rq_address_geometry.json'
    try: d=json.loads(path.read_text())
    except Exception:
        log=read(Path('logs/formal_v3_rq_address_geometry.log'))
        embedded=re.findall(r'\[diagnostic embed\] (\d+)/(8000)',log)
        state=f'embedding {"/".join(embedded[-1])}' if embedded else 'surface reconstruction / queued'
        return f'<p><b>状态：</b>{state}。正式规模：2/3-gram 各 4,000；RQ-Shuffled 保持 bucket histogram。</p>'
    rows=[]
    for order,payload in d.get('orders',{}).items():
        quadrant=payload.get('quadrants',{}).get('high_semantic_low_lexical',{})
        rq_overlap=quadrant.get('rq_code_overlap')
        shuffled_overlap=quadrant.get('shuffled_code_overlap')
        contrast=(f'{100*rq_overlap:.2f}% / {100*shuffled_overlap:.2f}%' if isinstance(rq_overlap,(int,float)) and isinstance(shuffled_overlap,(int,float)) else '—')
        rows.append(f'<tr><th>{order}-gram</th><td>{payload.get("sampled_ngrams","—"):,}</td><td>{payload.get("candidate_pairs","—"):,}</td><td>{payload.get("spearman_semantic_vs_rq_overlap",float("nan")):.4f}</td><td>{payload.get("spearman_semantic_vs_shuffled_overlap",float("nan")):.4f}</td><td>{contrast}</td></tr>')
    coverage=d.get('coverage',{})
    cov=' · '.join(f'{order}-gram={100*item.get("coverage",0):.2f}%' for order,item in coverage.get('per_order',{}).items()) or coverage.get('status','—')
    reconstruction=d.get('protocol',{}).get('surface_reconstruction','—')
    return f'<p><b>状态：</b>{d.get("status","—")}；held-out coverage：{cov}；surface={reconstruction}</p><table><thead><tr><th>Order</th><th>N-grams</th><th>Candidate pairs</th><th>Spearman(semantic, RQ overlap)</th><th>Spearman(semantic, shuffled overlap)</th><th>High-semantic / low-lexical overlap<br>RQ / Shuffled</th></tr></thead><tbody>{"".join(rows)}</tbody></table><p class="small">RQ-Shuffled 对完整 code vector 做行置换，保持各 level bucket histogram 与联合 code 分布。High-semantic / low-lexical 象限按各 order 内 semantic cosine ≥ Q75 且 char-trigram Jaccard ≤ Q25 定义。当前 surface 由 compressed key 的确定性 canonical representative 反解；因此可解释为地址几何验证，不宣称覆盖原始语料的全部真实表面变体。Coverage 因 FineWeb streaming split 在 offline 环境不可用而明确留空。</p>'

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
        row_class='method-rq' if method == 'Semantic-RQ' else ('method-lora' if method == 'LoRA' else ('method-arithmetic' if method == 'Arithmetic-fixed' else ''))
        if dataset == 'zsre' and method == 'Arithmetic-fixed' and seed == '42': row_class += ' dataset-start'
        rows.append('<tr class="%s"><th>%s</th><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td class="small">%s</td></tr>' % (row_class,
            'CounterFact' if dataset == 'counterfact' else 'ZsRE', html.escape(method), seed, d.get('examples','—'),
            *(f"{100*m.get(k):.2f}" if isinstance(m.get(k),(int,float)) else '—' for k in ('efficacy','paraphrase','specificity','harmonic_score')),
            denominator, html.escape(score)))
    return ''.join(rows) or '<tr><td colspan="10">正式全集评测尚未完成；这里不会填入 smoke-test 数字。</td></tr>'

def aggregate_rows(root: Path) -> str:
    rows=[]
    for dataset in ('counterfact','zsre'):
        for method in ('base','arithmetic','rq','lora'):
            runs=[]
            seeds=(42,) if method == 'base' else (42,43,44)
            for seed in seeds:
                try: d=json.loads((root/f'{dataset}_{method}_seed{seed}.json').read_text())
                except Exception: continue
                if d.get('status') == 'complete' and d.get('complete_official_split'):
                    runs.append(d['metrics'])
            values=[]
            for key in ('efficacy','paraphrase','specificity','harmonic_score'):
                xs=[100*r[key] for r in runs]
                if method == 'base' and len(xs) == 1:
                    values.append(f'{xs[0]:.2f}')
                else:
                    values.append(f'{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f}' if len(xs) == 3 else '等待 3 seeds')
            row_class=f'method-{method}' + (' dataset-start' if dataset == 'zsre' and method == 'base' else '')
            seed_label='fixed reference' if method == 'base' else f'{len(runs)}/3'
            rows.append(f'<tr class="{row_class}"><th>{"CounterFact" if dataset=="counterfact" else "ZsRE"}</th><td>{METHOD_LABELS[method]}</td><td>{seed_label}</td>'+''.join(f'<td>{v}</td>' for v in values)+'</tr>')
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
    for ds,label in [('counterfact','CounterFact'),('zsre','ZsRE')]:
        rq_runs=[results.get(f'{ds}_rq_seed{s}') for s in (42,43,44)]
        rq_runs=[d['metrics'] for d in rq_runs if d]
        arith_runs=[results.get(f'{ds}_arithmetic_seed{s}') for s in (42,43,44)]
        arith_runs=[d['metrics'] for d in arith_runs if d]
        if rq_runs and len(arith_runs) == 3:
            rm={k:100*statistics.mean(r[k] for r in rq_runs) for k in ('efficacy','paraphrase','specificity','harmonic_score')}
            am={k:100*statistics.mean(r[k] for r in arith_runs) for k in ('efficacy','paraphrase','specificity','harmonic_score')}
            if len(rq_runs) == 3:
                rs={k:100*statistics.stdev(r[k] for r in rq_runs) for k in ('efficacy','paraphrase','specificity','harmonic_score')}
                items.append(f'<li><b>{label} Semantic-RQ 最终三 seed：</b>RQ={rm["efficacy"]:.2f}±{rs["efficacy"]:.2f} / {rm["paraphrase"]:.2f}±{rs["paraphrase"]:.2f} / {rm["specificity"]:.2f}±{rs["specificity"]:.2f} / {rm["harmonic_score"]:.2f}±{rs["harmonic_score"]:.2f}（efficacy / paraphrase / specificity / harmonic）；相对 Arithmetic 的 Δ={rm["efficacy"]-am["efficacy"]:+.2f} / {rm["paraphrase"]-am["paraphrase"]:+.2f} / {rm["specificity"]-am["specificity"]:+.2f} / {rm["harmonic_score"]-am["harmonic_score"]:+.2f} 个百分点。</li>')
            else:
                items.append(f'<li><b>{label} Semantic-RQ 预备信号（{len(rq_runs)}/3 seeds）：</b>相对 Arithmetic 三-seed均值，Δ efficacy={rm["efficacy"]-am["efficacy"]:+.2f}，Δ paraphrase={rm["paraphrase"]-am["paraphrase"]:+.2f}，Δ specificity={rm["specificity"]-am["specificity"]:+.2f}，Δ harmonic={rm["harmonic_score"]-am["harmonic_score"]:+.2f} 个百分点。仅作运行中趋势，不作为最终方法排序。</li>')
    rq_done=all(f'{ds}_rq_seed{s}' in results for ds in ('counterfact','zsre') for s in (42,43,44))
    if not rq_done:
        items.append('<li><b>Semantic-RQ 主结论尚未形成：</b>修复后三 seed 完整官方评测尚未全部完成；当前只报告已完成 run 的预备趋势。</li>')
    items.append('<li><b>修复后重跑：</b>RQ seeds 42/43/44 均从空的独立持久 cache 重新开始，离线表外 n-gram 执行 encoder→RQ，随后同时写入 SQLite 与进程内热 cache；旧协议中途产物不进入结果表。</li>')
    items.append('<li><b>效率口径：</b>当前训练 wall-clock 包含首次遇到 n-gram 的 embedding/RQ 成本，不作为最终推理延迟。latency/throughput 将在三方法同代码、同 batch、同序列及已预热 cache 下单独复测。</li>')
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
        row_class='method-rq' if m == 'Semantic-RQ' else ('method-lora' if m == 'LoRA' else 'method-arithmetic')
        if d == 'ZsRE' and m == 'Arithmetic-fixed' and s == 42: row_class += ' dataset-start'
        state_html=f'<span class="status-ok">{state}</span>' if state == 'official eval complete' else state
        run_rows+=f'<tr class="{row_class}"><th>{d}</th><td>{m}</td><td>{s}</td><td>{state_html}</td><td>{loss}</td><td><code>{l}</code><br><code>{eval_log}</code></td></tr>'
    proc_rows=''.join(f'<tr><th>{p}</th><td>{g}</td><td class="cmd">{html.escape(c)}</td></tr>' for p,g,c in processes())
    build=read(Path('logs/build_rq_M8K1024_300k.log'))
    stage='complete' if 'meta.json' in build or '[done]' in build.lower() else ('RQ fitting / writing' if '[rq]' in build else 'embedding')
    now=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="refresh" content="15"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic Hash · Formal Benchmarks</title><style>
:root{{--ink:#17211d;--muted:#6d7772;--line:#e5e8e4;--paper:#fff;--bg:#f3f6f4;--green:#126747;--green2:#e9f5ef;--blue:#285c86;--amber:#9a6508;--red:#a13a35;--shadow:0 12px 32px rgba(20,45,35,.07)}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:radial-gradient(circle at 12% 0,#e8f3ed 0,transparent 32rem),var(--bg);color:var(--ink);font:14px/1.6 Inter,ui-sans-serif,system-ui,"PingFang SC",sans-serif;font-variant-numeric:tabular-nums}}main{{max-width:1500px;margin:auto;padding:38px 28px 96px}}header{{display:flex;align-items:flex-end;justify-content:space-between;gap:32px;padding:24px 28px;background:linear-gradient(125deg,#113e30,#176247 62%,#217556);color:#fff;border-radius:22px;box-shadow:var(--shadow)}}h1{{font:700 42px/1.05 Georgia,serif;letter-spacing:-.025em;margin:7px 0 10px}}h2{{display:flex;align-items:center;gap:10px;margin:38px 0 12px;font-size:19px;letter-spacing:-.01em}}h2:before{{content:"";width:5px;height:20px;border-radius:5px;background:var(--green)}}.kicker{{color:#bfe8d4;font-size:11px;font-weight:800;letter-spacing:.18em}}header .muted{{color:#d4e8df}}header>div:last-child{{text-align:right;white-space:nowrap;background:rgba(255,255,255,.1);padding:10px 13px;border:1px solid rgba(255,255,255,.15);border-radius:12px}}.muted{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:18px 0}}.card,.box{{background:rgba(255,255,255,.94);border:1px solid var(--line);border-radius:16px;box-shadow:0 4px 18px rgba(20,45,35,.04)}}.card{{position:relative;overflow:hidden;padding:18px 19px;min-height:112px}}.card:after{{content:"";position:absolute;right:-24px;bottom:-30px;width:82px;height:82px;border-radius:50%;background:var(--green2)}}.card b{{position:relative;z-index:1;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}}.card strong{{position:relative;z-index:1;display:block;margin:6px 0 2px;color:var(--green);font-size:25px;line-height:1.2;letter-spacing:-.02em}}.card span{{position:relative;z-index:1;color:var(--muted);font-size:12px}}.box{{overflow:auto}}.notes{{padding:20px 24px}}.notes ul{{display:grid;gap:9px;margin:0;padding-left:20px}}.notes li{{padding-left:4px}}.notes b{{color:#22352d}}table{{border-collapse:separate;border-spacing:0;width:100%;min-width:880px}}th,td{{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}thead th{{position:sticky;top:0;z-index:2;padding-top:11px;padding-bottom:11px;background:#f7f9f8;color:#5f6d66;font-size:11px;text-transform:uppercase;letter-spacing:.055em;white-space:nowrap}}tbody tr:last-child>*{{border-bottom:0}}tbody tr:hover>*{{background:#f7fbf9}}tbody th{{font-weight:700;white-space:nowrap}}td:nth-child(n+3){{font-variant-numeric:tabular-nums}}h2+div.box>table>tbody>tr:has(td:nth-child(2)){{transition:background .15s}}tr:has(td:nth-child(2):is(:not(:empty))){{}}
/* Method emphasis in generated benchmark tables. */
tr:has(td:nth-child(2)) td:nth-child(2){{font-weight:650}}tr:has(td:nth-child(2)) td:nth-child(2):where(td){{white-space:nowrap}}tr:has(td:nth-child(2)) td:nth-child(2){{color:#34463e}}tr:has(td:nth-child(2)) td:nth-child(2){{}}
tr:has(td:nth-child(2)){{}}
td:has(>code){{color:var(--muted)}}code{{display:inline-block;color:#386288;background:#f1f5f8;border-radius:5px;padding:1px 5px;font:11px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}}.cmd{{max-width:820px;color:#718078;font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}}.small{{font-size:12px;color:var(--muted)}}.bad{{color:var(--red);font-weight:650}}a{{color:var(--green);font-weight:650;text-decoration:none;border-bottom:1px solid #b9d8ca}}a:hover{{color:#094f34;border-color:currentColor}}
/* Protocol table reads as key/value metadata. */
h2:first-of-type+.box table{{min-width:720px}}h2:first-of-type+.box th{{width:190px;color:#506159;background:#fafcfb;border-right:1px solid var(--line)}}h2:first-of-type+.box td{{line-height:1.65}}
/* Highlight our method and separate benchmark blocks. */
tbody tr:has(td:nth-child(2)) td:nth-child(2){{border-left:3px solid transparent}}tbody tr:has(td:nth-child(2)) td:nth-child(2){{}}
tbody tr:has(td:nth-child(2)) td:nth-child(2){{}}
tbody tr:has(td:nth-child(2)) td:nth-child(2){{}}
tr:has(td:nth-child(2)) td:nth-child(2){{}}
tr:has(td:nth-child(2)) td:nth-child(2){{}}
tr:has(td:nth-child(2)) td:nth-child(2){{}}
/* CSS text matching is unavailable; generated rows receive semantic classes below. */
.method-rq>*{{background:#edf8f2!important}}.method-rq td:nth-child(2){{color:var(--green);border-left:3px solid #35a46e}}.method-lora td:nth-child(2){{color:var(--blue)}}.method-arithmetic td:nth-child(2){{color:var(--amber)}}.dataset-start>*{{border-top:2px solid #cbd5d0}}.status-ok{{display:inline-flex;align-items:center;gap:6px;color:var(--green);font-weight:700}}.status-ok:before{{content:"✓";display:grid;place-items:center;width:18px;height:18px;border-radius:50%;color:#fff;background:var(--green);font-size:11px}}
details.audit{{margin-top:38px;border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.72);box-shadow:0 4px 18px rgba(20,45,35,.03)}}details.audit>summary{{cursor:pointer;list-style:none;padding:17px 21px;color:#526159;font-weight:750}}details.audit>summary::-webkit-details-marker{{display:none}}details.audit>summary:before{{content:"＋";display:inline-grid;place-items:center;width:24px;height:24px;margin-right:9px;border-radius:50%;background:var(--green2);color:var(--green)}}details.audit[open]>summary:before{{content:"−"}}details.audit>div{{border-top:1px solid var(--line)}}.method-base td:nth-child(2){{color:#66716b}}.method-base>*{{background:#fafbfa}}.table-note{{margin:0;padding:11px 15px;border-bottom:1px solid var(--line);background:#f7f9f8;color:var(--muted);font-size:12px}}
@media(max-width:980px){{main{{padding:20px 16px 72px}}header{{align-items:flex-start;flex-direction:column}}header>div:last-child{{text-align:left}}.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:34px}}}}@media(max-width:580px){{.cards{{grid-template-columns:1fr}}.card{{min-height:95px}}h1{{font-size:30px}}.notes{{padding:16px 18px}}}}
</style></head><body><main><header><div><div class="kicker">FULL OFFICIAL BENCHMARKS · LIVE</div><h1>Semantic-Hash Engram</h1><div class="muted">canonical-only target supervision · dataset-specific official scoring · evaluation-only paraphrase/locality</div></div><div class="muted">{now}<br>每 15 秒自动刷新</div></header><section class="cards"><div class="card"><b>CounterFact</b><strong>2,191 cases</strong><span>ES · PS · NS · Harmonic</span></div><div class="card"><b>ZsRE</b><strong>1,301 edits</strong><span>EA · PA · NA · Harmonic</span></div><div class="card"><b>正式矩阵</b><strong>{completion(root)}</strong><span>3 seeds × core methods · Base seed42</span></div><div class="card"><b>结果门槛</b><strong>No smoke tests</strong><span>仅完整官方 split 进入结果表</span></div></section><h2>实验协议</h2><div class="box"><table><tbody><tr><th>写入 setting</th><td>完整 benchmark canonical edits 批量写入同一 memory；不是逐样本重建模型</td></tr><tr><th>训练隔离</th><td>仅 canonical prompt → new target；prompt labels=-100；paraphrase/locality evaluation-only</td></tr><tr><th>训练预算</th><td>统一 5 次数据遍历：CounterFact 345 optimizer steps；ZsRE 205 optimizer steps；effective batch=32</td></tr><tr><th>CounterFact scoring</th><td>沿用 ROME/CounterFact 的 ES / PS / NS 指标语义；为正确覆盖多 token 答案，比较完整 target 的 mean token log-likelihood，而非旧实现的首 token shortcut</td></tr><tr><th>ZsRE scoring</th><td>主流 locate-then-edit 口径：teacher-forced 逐 target-token accuracy；EA / PA / NA 的分母是 target tokens</td></tr><tr><th>RQ 地址</th><td>{stage}；FineWeb-Edu 2/3-gram 各 300k，M=8/K=1024；unseen n-gram 在线 embedding→RQ，首次写入持久 cache，后续直接命中</td></tr><tr><th>运行时语义地址</th><td>{rq_cache_status()}；累计覆盖训练与只读评测期间遇到的离线表外唯一 n-gram。评测会扩充地址 cache，但冻结模型与 memory value 不更新，因此不是训练泄漏，也不是 Arithmetic fallback</td></tr><tr><th>比较</th><td>Base / LoRA / Arithmetic-fixed / Semantic-RQ；同底座、层、容量、数据遍历数；核心方法 seeds 42/43/44</td></tr><tr><th>有效性审计</th><td class="bad">v1/v2：静态 fallback、首 token、prompt-label 覆盖、legacy hash 覆盖、错误数据规模/预算，全部排除</td></tr></tbody></table></div><h2>文献与官方实现依据</h2><div class="box notes"><ul><li><a href="https://arxiv.org/abs/2202.05262">ROME / CounterFact paper</a>：以 efficacy、paraphrase generalization 与 neighborhood specificity 评价是否真正改写事实而非复述训练句。</li><li><a href="https://github.com/kmeng01/rome/blob/main/experiments/py/eval_utils_counterfact.py">ROME CounterFact evaluator</a>：ES / PS / NS 的官方参考实现。</li><li><a href="https://github.com/kmeng01/rome/blob/main/experiments/py/eval_utils_zsre.py">ROME ZsRE evaluator</a>：teacher-forced target-token accuracy 的官方参考实现。</li><li><a href="https://github.com/zjunlp/EasyEdit">EasyEdit</a>：当前知识编辑文献常用的 reliability / generalization / locality 统一框架。这里保留 benchmark 原生名称和分母，不混用生成式 EM。</li></ul><p class="small">“官方”指指标定义与 benchmark prompt 集合；本实验为 Qwen3 和多 token target 做了透明的完整答案扩展，不声称逐字复刻只针对旧 GPT 架构的脚本。</p></div><h2>当前可支持的结论</h2><div class="box notes"><ul>{conclusions(root)}</ul></div><h2>地址几何机制分析</h2><div class="box notes">{geometry_analysis(root)}</div><h2>GPU</h2><div class="cards">{gpus()}</div><h2>正式主结果（mean ± sample std，%）</h2><div class="box"><p class="table-note">Arithmetic-fixed、Semantic-RQ、LoRA 均汇总 seeds 42/43/44；Base 是同一个冻结预训练模型，仅作为 fixed reference，不报告伪造的方差。</p><table><thead><tr><th>Benchmark</th><th>方法</th><th>统计口径</th><th>Efficacy</th><th>Paraphrase</th><th>Specificity</th><th>Harmonic</th></tr></thead><tbody>{aggregate_rows(root)}</tbody></table></div><details class="audit"><summary>展开逐 Seed 运行审计与日志</summary><div><h2>训练与队列</h2><div class="box"><table><thead><tr><th>Benchmark</th><th>方法</th><th>Seed</th><th>Step</th><th>Loss</th><th>日志</th></tr></thead><tbody>{run_rows}</tbody></table></div><h2>逐 Seed 完整结果（仅审计，%）</h2><div class="box"><table><thead><tr><th>Benchmark</th><th>方法</th><th>Seed</th><th>Cases</th><th>Efficacy<br>ES / EA</th><th>Paraphrase<br>PS / PA</th><th>Specificity<br>NS / NA</th><th>Harmonic</th><th>实际分母</th><th>核心评分口径</th></tr></thead><tbody>{result_rows(root)}</tbody></table></div><h2>真实进程</h2><div class="box"><table><thead><tr><th>PID</th><th>GPU</th><th>Command</th></tr></thead><tbody>{proc_rows or '<tr><td colspan="3" class="muted">当前没有训练或评测进程</td></tr>'}</tbody></table></div></div></details></main></body></html>'''

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
