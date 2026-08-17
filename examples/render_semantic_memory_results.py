#!/usr/bin/env python
"""Render paper-ready summaries for the three semantic-memory experiments."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS=("arithmetic","rq","rq_shuffled")
LABELS={"arithmetic":"Arithmetic","rq":"Semantic-RQ","rq_shuffled":"RQ-Shuffled"}
POINTS=(1000,5000,10000,50000)
BINS=("low","mid_low","mid_high","high")


def load(path: Path): return json.loads(path.read_text())


def render_pararel(root: Path, output: Path) -> dict:
    payload={m:load(root/f"{m}_seed42.json") for m in METHODS}
    matrices={}
    for method in METHODS:
        matrix=np.full((4,4),np.nan)
        for i,semantic in enumerate(BINS):
            for j,lexical in enumerate(BINS):
                item=payload[method]["metrics"].get(f"geometry/{semantic}_{lexical}")
                if item: matrix[i,j]=100*item["mean"]
        matrices[method]=matrix
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,(title,matrix) in zip(axes,[("Semantic-RQ − RQ-Shuffled",matrices["rq"]-matrices["rq_shuffled"]),("Semantic-RQ − Arithmetic",matrices["rq"]-matrices["arithmetic"])],strict=True):
        vmax=max(1,np.nanmax(np.abs(matrix))); image=ax.imshow(matrix,cmap="RdBu_r",vmin=-vmax,vmax=vmax)
        for i in range(4):
            for j in range(4):
                if np.isfinite(matrix[i,j]): ax.text(j,i,f"{matrix[i,j]:+.1f}",ha="center",va="center",fontsize=8)
        ax.set(title=title,xlabel="Lexical similarity",ylabel="Semantic similarity",xticks=range(4),yticks=range(4),xticklabels=BINS,yticklabels=BINS)
        fig.colorbar(image,ax=ax,label="Accuracy difference (pp)")
    fig.savefig(output/"pararel_geometry_heatmap.png",dpi=220); plt.close(fig)
    return {m:100*payload[m]["metrics"]["role/should_propagate"]["mean"] for m in METHODS}


def common_ripple(root: Path) -> dict:
    rows={m:{(r["case_id"],r["prompt"]):r for r in map(json.loads,(root/f"{m}_seed42.jsonl").open())} for m in METHODS}
    keys=set.intersection(*(set(value) for value in rows.values()))
    keys={key for key in keys if all(rows[m][key]["eligible"] for m in METHODS)}
    output={"common_eligible":len(keys),"roles":{},"axes":{}}
    for field,target in (("role","roles"),("axis","axes")):
        for label in sorted({rows["arithmetic"][key][field] for key in keys}):
            subset=[key for key in keys if rows["arithmetic"][key][field]==label]
            output[target][label]={"n":len(subset),**{m:100*sum(rows[m][key]["accuracy"] for key in subset)/len(subset) for m in METHODS}}
    return output


def render_wiki(root: Path, output: Path) -> dict:
    summary={}
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True,constrained_layout=True)
    for method in METHODS:
        available=[]
        for point in POINTS:
            path=root/method/f"at_{point}.json"
            if path.is_file(): available.append((point,load(path)))
        summary[method]={str(point):{axis:100*data["metrics"][f"axis/{axis}"]["mean"] for axis in ("efficacy","generalization","locality")} for point,data in available}
        for ax,axis in zip(axes,("efficacy","generalization"),strict=True):
            ax.plot([p for p,_ in available],[100*d["metrics"][f"axis/{axis}"]["mean"] for _,d in available],marker="o",label=LABELS[method])
            ax.set(title=axis.title(),xlabel="Cumulative writes",ylabel="Target-token accuracy (%)",xscale="log",xticks=POINTS)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()); ax.grid(alpha=.25)
    axes[0].legend(); fig.savefig(output/"wikibigedit_scaling_curves.png",dpi=220); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True,constrained_layout=True)
    for method in METHODS:
        for ax,axis in zip(axes,("efficacy","generalization"),strict=True):
            xs=[]; ys=[]
            for point in POINTS:
                path=root/method/f"at_{point}.json"
                if not path.is_file(): continue
                metric=load(path)["metrics"].get(f"cohort/1000/{axis}")
                if metric: xs.append(point); ys.append(100*metric["mean"])
            ax.plot(xs,ys,marker="o",label=LABELS[method])
            ax.set(title=f"Oldest 1K cohort: {axis}",xlabel="Cumulative writes",ylabel="Target-token accuracy (%)",xscale="log",xticks=POINTS)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()); ax.grid(alpha=.25)
    axes[0].legend(); fig.savefig(output/"wikibigedit_retention_curves.png",dpi=220); plt.close(fig)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--root",type=Path,default=Path("outputs/semantic_memory")); parser.add_argument("--output",type=Path,default=Path("outputs/semantic_memory/paper_results")); args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    summary={"pararel":render_pararel(args.root/"pararel",args.output),"ripple":common_ripple(args.root/"ripple"),"wikibigedit":render_wiki(args.root/"wikibigedit",args.output)}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    wiki_rows="".join(f"<tr><th>{point:,}</th>"+"".join(f"<td>{summary['wikibigedit'][m][str(point)]['efficacy']:.2f} / {summary['wikibigedit'][m][str(point)]['generalization']:.2f}</td>" for m in METHODS)+"</tr>" for point in POINTS)
    ripple_rows="".join(f"<tr><th>{html.escape(role)}</th><td>{values['n']:,}</td>"+"".join(f"<td>{values[m]:.2f}</td>" for m in METHODS)+"</tr>" for role,values in summary["ripple"]["roles"].items())
    k_results={}
    for k in (8,16,32,64,1024):
        path=(args.root/"pararel_k_sweep"/f"k{k}_seed42.json") if k != 1024 else (args.root/"pararel_indomain_rq"/"seed42.json")
        if path.is_file():
            k_results[k]=100*load(path)["metrics"]["role/should_propagate"]["mean"]
    k_rows="".join(
        f"<tr><th>{k}</th><td>8</td><td>{2*k:,}</td><td>{score:.2f}</td>"
        f"<td>{score-summary['pararel']['rq']:+.2f}</td><td>{score-summary['pararel']['arithmetic']:+.2f}</td></tr>"
        for k,score in sorted(k_results.items())
    ) or "<tr><td colspan='6'>K sweep 正在运行</td></tr>"
    k_section=f"""<section><h2>ParaRel in-domain RQ · Codebook-size sweep</h2><p>固定 M=8、5K canonical writes、seed 42；每种 K 在独立 GPU 上完成建表、782-step 训练和 16,024-query unseen-template 评测。地址行数为 2/3-gram 两组 memory heads 每层的合计，不表示组合空间。</p><table><thead><tr><th>K / level</th><th>RQ levels</th><th>Rows / level</th><th>Accuracy (%)</th><th>Δ FineWeb-RQ</th><th>Δ Arithmetic</th></tr></thead><tbody>{k_rows}</tbody></table></section>"""
    page=f"""<!doctype html><html><head><meta charset='utf-8'><title>Semantic-Hash Engram Results</title><style>body{{font:15px system-ui;background:#f4f6fb;color:#182033;margin:0}}main{{max-width:1180px;margin:auto;padding:32px}}h1{{font-size:34px}}section{{background:white;border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 4px 18px #18203312}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #e4e8f0;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{width:100%;border:1px solid #e4e8f0;border-radius:10px}}.bad{{color:#a63131}}.good{{color:#18794e}}code{{background:#eef1f7;padding:2px 5px;border-radius:4px}}</style></head><body><main><h1>Semantic-Hash Engram · Formal Results</h1><p>Qwen3-1.7B Base · seed 42 · complete official manifests · online OOV embedding→RQ with persistent cache</p><section><h2>结论</h2><ul><li class='bad'>ParaRel overall：Arithmetic {summary['pararel']['arithmetic']:.2f}，RQ {summary['pararel']['rq']:.2f}，Shuffled {summary['pararel']['rq_shuffled']:.2f}。语义几何存在局部效应，但总体不胜 Arithmetic。</li><li class='good'>WikiBigEdit@50K：RQ efficacy/generalization {summary['wikibigedit']['rq']['50000']['efficacy']:.2f}/{summary['wikibigedit']['rq']['50000']['generalization']:.2f}，比 Arithmetic 高 {summary['wikibigedit']['rq']['50000']['efficacy']-summary['wikibigedit']['arithmetic']['50000']['efficacy']:+.2f}/{summary['wikibigedit']['rq']['50000']['generalization']-summary['wikibigedit']['arithmetic']['50000']['generalization']:+.2f} pp；但与 Shuffled 基本相同，规模收益主要来自 RQ 结构。</li><li>Ripple common-eligible={summary['ripple']['common_eligible']:,}；RQ 对 Shuffled 的传播/不传播优势很小，边界未恶化但 semantic ordering 贡献有限。</li></ul></section>{k_section}<section><h2>WikiBigEdit scaling（Efficacy / Generalization, %）</h2><table><thead><tr><th>Writes</th><th>Arithmetic</th><th>Semantic-RQ</th><th>RQ-Shuffled</th></tr></thead><tbody>{wiki_rows}</tbody></table><img src='paper_results/wikibigedit_scaling_curves.png'><img src='paper_results/wikibigedit_retention_curves.png'></section><section><h2>ParaRel causal geometry</h2><img src='paper_results/pararel_geometry_heatmap.png'></section><section><h2>RippleEdits common-eligible paired set (%)</h2><table><thead><tr><th>Role</th><th>N</th><th>Arithmetic</th><th>Semantic-RQ</th><th>RQ-Shuffled</th></tr></thead><tbody>{ripple_rows}</tbody></table></section><section><h2>Protocol audit</h2><p>ParaRel: 5,000 cases / 16,024 queries / 16 geometry bins. Ripple: 3,780 cases / 36,570 scorable queries; paired table uses intersection eligible queries. WikiBigEdit: one chronological adapter, milestones 1K/5K/10K/50K, disjoint fixed 500-case cohorts.</p></section></main></body></html>"""
    (args.output.parent/"dashboard_formal.html").write_text(page,encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
