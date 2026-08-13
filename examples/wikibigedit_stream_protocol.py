#!/usr/bin/env python
"""Prepare auditable chronological WikiBigEdit training/evaluation cohorts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.semantic_memory_benchmarks import stream_checkpoints


def read_cases(path: Path, limit: int) -> list[dict[str, Any]]:
    cases=[]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): cases.append(json.loads(line))
            if len(cases) >= limit: break
    return cases


def cohort_indices(point: int, size: int, seed: int, start: int = 0) -> list[int]:
    """Deterministic evenly spread sample from the half-open interval [start, point)."""
    width=max(0,point-start)
    if size <= 0 or width <= 0: return []
    count=min(size,width)
    stride=max(1,width//count)
    offset=seed % stride
    return [min(point-1,start+offset+(i*width)//count) for i in range(count)]


def build_protocol(cases: list[dict[str, Any]], points: tuple[int,...], cohort_size: int, seed: int) -> dict[str,Any]:
    cohorts={}
    start=0
    for point in points:
        cohorts[str(point)]=cohort_indices(point,cohort_size,seed+point,start)
        start=point
    return {
        "protocol":"single_adapter_chronological_one_pass",
        "points":list(points),
        "cases":len(cases),
        "cohort_size":cohort_size,
        "cohorts":cohorts,
        "retention":{
            str(point):{str(origin):cohorts[str(origin)] for origin in points if origin <= point}
            for point in points
        },
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--points",type=int,nargs="+",default=[1000,5000,10000,50000])
    parser.add_argument("--cohort-size",type=int,default=500)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    maximum=max(args.points); cases=read_cases(args.manifest,maximum)
    points=stream_checkpoints(len(cases),args.points)
    payload=build_protocol(cases,points,args.cohort_size,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k!="retention" and k!="cohorts"},indent=2))


if __name__ == "__main__": main()
