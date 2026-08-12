"""Queue PAWS-X methods on each GPU immediately after its XNLI work finishes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

SCHEDULE = {
    0: ("lora", ["lora"]),
    1: ("arithmetic_matched", ["arithmetic_matched"]),
    2: ("full_ft", ["full_ft"]),
    # XNLI arithmetic is already complete by the time this relay starts, so
    # keep GPU 3 busy after the short base/RQ jobs instead of waiting for the
    # much longer arithmetic_matched gate on GPU 1.
    3: ("rq", ["base", "rq", "arithmetic"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xnli-dir", type=Path, required=True)
    parser.add_argument("--pawsx-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rq-table-dir", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def status(path: Path) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", ""))
    except (OSError, json.JSONDecodeError):
        return ""


def wait_complete(path: Path, poll_seconds: float) -> None:
    while status(path) != "complete":
        time.sleep(poll_seconds)


def run_method(args: argparse.Namespace, gpu: int, method: str) -> None:
    result = args.pawsx_dir / f"{method}_seed42" / "metrics.json"
    if status(result) == "complete":
        print(f"[GPU{gpu}] PAWS-X {method} already complete; skipping", flush=True)
        return
    log_dir = args.pawsx_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "examples/run_xtreme_pawsx.py",
        "--method", method,
        "--model", args.model,
        "--rq_table_dir", args.rq_table_dir,
        "--output_dir", str(args.pawsx_dir),
        "--epochs", "1",
        "--batch_size", "4",
        "--eval_batch_size", "8",
        "--grad_accum", "8",
        "--max_length", "256",
        "--num_workers", "4",
        "--seed", "42",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = log_dir / f"pawsx_{method}.log"
    print(f"[GPU{gpu}] starting PAWS-X {method}; log={log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"PAWS-X {method} failed with exit code {completed.returncode}")
    print(f"[GPU{gpu}] PAWS-X {method} complete", flush=True)


def worker(args: argparse.Namespace, gpu: int, gate: str, methods: list[str]) -> None:
    gate_path = args.xnli_dir / f"{gate}_seed42" / "metrics.json"
    print(f"[GPU{gpu}] waiting for XNLI {gate}", flush=True)
    wait_complete(gate_path, args.poll_seconds)
    for method in methods:
        run_method(args, gpu, method)


def main() -> None:
    args = parse_args()
    args.pawsx_dir.mkdir(parents=True, exist_ok=True)
    threads = [
        threading.Thread(target=worker, args=(args, gpu, gate, list(methods)))
        for gpu, (gate, methods) in SCHEDULE.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
