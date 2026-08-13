#!/usr/bin/env python
"""Atomically mirror the Taiji formal dashboard to the local workspace."""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--instance", default="8b1d815e9eef2052019ef80778f8098a")
    parser.add_argument(
        "--remote",
        default="/anguszhang-cfs-nj/seokliu_workspace/engram/outputs/semantic_hash_paper/dashboard_formal_v3.html",
    )
    args = parser.parse_args()
    remote_cmd = (
        "export TAIJI_API_TOKEN=SHdzTqUCxVnt7b7gxI3i7Q && "
        f"taiji_cli exec {args.instance} 'base64 -w0 {args.remote}'"
    )
    while True:
        try:
            proc = subprocess.run(
                ["ssh", "seokliu-any2.devcloud.woa.com", remote_cmd],
                check=True,
                capture_output=True,
                text=True,
                timeout=45,
            )
            payload = proc.stdout.rsplit("\n", 1)[-1].strip()
            data = base64.b64decode(payload, validate=True)
            if b"FULL OFFICIAL BENCHMARKS" not in data:
                raise ValueError("unexpected dashboard payload")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.output.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, args.output)
        except Exception as exc:
            print(f"sync failed: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
