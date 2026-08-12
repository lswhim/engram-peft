#!/usr/bin/env python
"""Run the frequency-matched three-seed XNLI/PAWS-X external triads."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from examples.run_semantic_hash_pipeline import (
    Task,
    active_command_matches,
    complete_metric,
    gpu_memory,
    reserved_process_gpus,
    write_state,
)


SEEDS = (42, 43, 44)
BENCHMARKS = ("xnli", "pawsx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--table-dir", type=Path,
        default=Path("rq_tables/wiki15_qwen3_06b_M8K256_500k"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/semantic_hash_paper/external_matched"),
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--free-memory-threshold-mib", type=int, default=1500)
    return parser.parse_args()


def result_path(benchmark: str, variant: str, seed: int) -> Path:
    roots = {
        "semantic_rq": Path(f"outputs/xtreme_{benchmark}_v1"),
        "arithmetic_matched": Path(f"outputs/xtreme_{benchmark}_matched_exact"),
        "rq_shuffled": Path(f"outputs/xtreme_{benchmark}_rq_shuffled_freqmatched"),
    }
    method_dir = "rq" if variant in {"semantic_rq", "rq_shuffled"} else variant
    return roots[variant] / f"{method_dir}_seed{seed}" / "metrics.json"


def benchmark_command(
    benchmark: str,
    model: str,
    table_dir: Path,
    variant: str,
    seed: int,
) -> tuple[str, ...]:
    method = "rq" if variant in {"semantic_rq", "rq_shuffled"} else variant
    output = {
        "semantic_rq": f"outputs/xtreme_{benchmark}_v1",
        "arithmetic_matched": f"outputs/xtreme_{benchmark}_matched_exact",
        "rq_shuffled": f"outputs/xtreme_{benchmark}_rq_shuffled_freqmatched",
    }[variant]
    return (
        sys.executable, "-u", f"examples/run_xtreme_{benchmark}.py",
        "--method", method,
        "--model", model,
        "--rq_table_dir", str(table_dir),
        "--output_dir", output,
        "--epochs", "1",
        "--batch_size", "4",
        "--eval_batch_size", "8",
        "--grad_accum", "8",
        "--max_length", "256",
        "--num_workers", "4",
        "--seed", str(seed),
    )


def make_tasks(
    args: argparse.Namespace,
    shuffled: dict[tuple[str, int], Path],
) -> list[Task]:
    direct_training: list[Task] = []
    access_tasks: list[Task] = []
    shuffled_training: list[Task] = []
    for benchmark in BENCHMARKS:
        manifest = args.output_dir / benchmark / "access_counts.npz"
        summary = args.output_dir / benchmark / "access_counts.json"
        access_tasks.append(Task(
            name=f"external_access_{benchmark}",
            command=(
                sys.executable, "-u", "scripts/build_crosslingual_access_counts.py",
                "--benchmark", benchmark,
                "--table-dir", str(args.table_dir),
                "--model", args.model,
                "--max-length", "256",
                "--batch-size", "256",
                "--num-proc", "4",
                "--output", str(manifest),
                "--summary", str(summary),
            ),
            log_name=f"external_access_{benchmark}.log",
            done=lambda summary=summary: complete_metric(summary),
        ))
        for variant in ("arithmetic_matched", "semantic_rq"):
            for seed in SEEDS:
                output = result_path(benchmark, variant, seed)
                direct_training.append(Task(
                    name=f"external_{benchmark}_{variant}_seed{seed}",
                    command=benchmark_command(
                        benchmark, args.model, args.table_dir, variant, seed
                    ),
                    log_name=f"external_{benchmark}_{variant}_seed{seed}.log",
                    done=lambda output=output: complete_metric(output),
                ))
        for seed in SEEDS:
            table = shuffled[(benchmark, seed)]
            output = result_path(benchmark, "rq_shuffled", seed)
            shuffled_training.append(Task(
                name=f"external_{benchmark}_rq_shuffled_seed{seed}",
                command=benchmark_command(
                    benchmark, args.model, table, "rq_shuffled", seed
                ),
                log_name=f"external_{benchmark}_rq_shuffled_seed{seed}.log",
                done=lambda output=output: complete_metric(output),
                ready=lambda table=table: (
                    table / "shuffle_manifest.json"
                ).is_file(),
            ))
    # Existing seed42/43 direct runs are already complete.  The four missing
    # seed44 direct runs therefore occupy all GPUs first.  Access counting then
    # prepares the decisive shuffled controls while those cards turn over.
    return [*direct_training, *access_tasks, *shuffled_training]


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    args.table_dir = args.table_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "pipeline_state.json"
    lock_handle = (args.output_dir / "pipeline_state.json.lock").open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("external scheduler already active", flush=True)
        return

    shuffled = {
        (benchmark, seed): Path(
            str(args.table_dir) + f"_{benchmark}_shuffled_freqmatched_seed{seed}"
        )
        for benchmark in BENCHMARKS for seed in SEEDS
    }
    tasks = make_tasks(args, shuffled)
    children: dict[str, subprocess.Popen[bytes]] = {}
    running: dict[str, dict[str, Any]] = {}
    while True:
        for name, process in list(children.items()):
            return_code = process.poll()
            if return_code is not None:
                running[name]["return_code"] = return_code
                running[name]["finished_at"] = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )
                del children[name]

        for benchmark in BENCHMARKS:
            access = args.output_dir / benchmark / "access_counts.npz"
            if not access.is_file():
                continue
            for seed in SEEDS:
                directory = shuffled[(benchmark, seed)]
                if (directory / "shuffle_manifest.json").is_file():
                    continue
                subprocess.run((
                    sys.executable, "scripts/shuffle_rq_table.py",
                    "--source-dir", str(args.table_dir),
                    "--output-dir", str(directory),
                    "--seed", str(seed),
                    "--access-counts", str(access),
                ), check=True)

        memory = gpu_memory()
        reserved = reserved_process_gpus() | {
            int(running[name]["gpu"]) for name in children
        }
        free = [
            gpu for gpu, used in memory.items()
            if used < args.free_memory_threshold_mib and gpu not in reserved
        ]
        for task in tasks:
            if not free:
                break
            if task.done() or task.name in children or active_command_matches(task.command):
                continue
            if not task.ready():
                continue
            gpu = free.pop(0)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["PYTHONPATH"] = f"{repo / 'src'}:{repo}"
            environment["TOKENIZERS_PARALLELISM"] = "false"
            environment.setdefault("http_proxy", "http://star-proxy.oa.com:3128")
            environment.setdefault("https_proxy", "http://star-proxy.oa.com:3128")
            environment.setdefault(
                "no_proxy", ".woa.com,.oa.com,.tencentcos.cn,localhost,127.0.0.1"
            )
            environment.setdefault("HF_HUB_DISABLE_XET", "1")
            log_path = logs / task.log_name
            handle = log_path.open("ab")
            process = subprocess.Popen(
                task.command, cwd=repo, env=environment,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
            handle.close()
            children[task.name] = process
            running[task.name] = {
                "pid": process.pid,
                "gpu": gpu,
                "log": str(log_path),
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "command": list(task.command),
            }
        finished = all(task.done() for task in tasks)
        write_state(
            state_path,
            running,
            "all external triads complete" if finished else "external scheduler active",
        )
        if finished:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
