#!/usr/bin/env python
"""Build fixed-cohort WikiBigEdit evaluation manifests for one write milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_eval_cases(cases: list[dict[str, Any]], protocol: dict[str, Any], milestone: int) -> list[dict[str, Any]]:
    retained=protocol["retention"].get(str(milestone))
    if retained is None:
        raise ValueError(f"milestone {milestone} is absent from protocol")
    output=[]
    for origin,indices in retained.items():
        for index in indices:
            source=cases[index]
            metadata=dict(source.get("metadata",{}),cohort_origin=int(origin),evaluated_at=milestone,source_index=index)
            canonical={"prompt":source["prompt"],"answers":[source["target"]],"role":"should_propagate","axis":"efficacy","condition_prompts":[],"condition_answers":[],"condition":"OR","lexical_similarity":None,"geometry_text":None}
            output.append({**source,"queries":[canonical,*source.get("queries",[])],"metadata":metadata})
    return output


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--milestone",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    cases=read_cases(args.manifest); protocol=json.loads(args.protocol.read_text())
    output=build_eval_cases(cases,protocol,args.milestone)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8") as handle:
        for case in output: handle.write(json.dumps(case,ensure_ascii=False)+"\n")
    print(json.dumps({"milestone":args.milestone,"cases":len(output),"origins":sorted({c["metadata"]["cohort_origin"] for c in output})}))


if __name__ == "__main__": main()
