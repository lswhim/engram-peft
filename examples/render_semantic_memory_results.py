#!/usr/bin/env python
"""Render paper-ready summaries for the three semantic-memory experiments."""

from __future__ import annotations

import argparse
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
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--root",type=Path,default=Path("outputs/semantic_memory")); parser.add_argument("--output",type=Path,default=Path("outputs/semantic_memory/paper_results")); args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    summary={"pararel":render_pararel(args.root/"pararel",args.output),"ripple":common_ripple(args.root/"ripple"),"wikibigedit":render_wiki(args.root/"wikibigedit",args.output)}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
