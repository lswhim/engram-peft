"""Mirror a dashboard from a Taiji instance to a local file until it is complete."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--remote-html", required=True)
    parser.add_argument("--local-html", type=Path, required=True)
    parser.add_argument("--host", default="seokliu-any2.devcloud.woa.com")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--html-only", action="store_true", help="sync HTML even when the remote watcher has no JSON snapshot")
    return parser.parse_args()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def fetch(host: str, instance: str, remote_path: str) -> bytes:
    token = os.environ.get("TAIJI_API_TOKEN", "")
    token_prefix = f"export TAIJI_API_TOKEN={token} && " if token else ""
    command = [
        "ssh",
        host,
        f"{token_prefix}taiji_cli exec {instance} 'base64 -w0 {remote_path}'",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=35, check=True)
    candidates = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    encoded = max(candidates, key=len)
    return base64.b64decode(encoded, validate=True)


def main() -> None:
    args = parse_args()
    remote_json = str(Path(args.remote_html).with_suffix(".json"))
    local_json = args.local_html.with_suffix(".json")
    failures = 0
    while True:
        try:
            page = fetch(args.host, args.instance, args.remote_html)
            if not page.lstrip().startswith(b"<!doctype html>"):
                raise ValueError("remote dashboard is not valid HTML")
            atomic_write(args.local_html, page)
            if args.html_only:
                snapshot = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "all_complete": False}
            else:
                snapshot_bytes = fetch(args.host, args.instance, remote_json)
                snapshot = json.loads(snapshot_bytes)
                atomic_write(local_json, snapshot_bytes)
            failures = 0
            print(f"[{snapshot['updated_at']}] local dashboard synchronized", flush=True)
            if args.once or snapshot.get("all_complete") is True:
                break
        except Exception as error:  # A transient gateway failure should not stop syncing.
            failures += 1
            print(f"sync attempt failed ({failures}): {error}", flush=True)
            if args.once:
                raise
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
