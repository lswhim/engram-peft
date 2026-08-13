#!/usr/bin/env python
"""Create a lightweight RQ-Shuffled table covering offline and runtime OOV codes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((args.source/"meta.json").read_text())
    meta["runtime_shuffle_seed"]=args.seed
    meta["shuffle_protocol"]="blake2b_joint_code_vector_v1"
    for path in args.source.iterdir():
        if path.name=="meta.json" or path.is_dir(): continue
        shutil.copy2(path,args.output/path.name)
    (args.output/"meta.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps({"source":str(args.source),"output":str(args.output),"seed":args.seed,"protocol":meta["shuffle_protocol"]},indent=2))


if __name__ == "__main__": main()
