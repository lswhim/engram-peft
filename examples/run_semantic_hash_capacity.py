#!/usr/bin/env python
"""Run the gated K={64,256,1024} finite-memory capacity sweep.

The K=256 center point reuses Gate-1 seed 42.  K=64 and K=1024 rebuild the
semantic address table, construct a frequency-matched shuffled control, train
the exact-capacity causal triad, and evaluate the same frozen token slices.
This scheduler waits until both earlier schedulers have exited, preventing GPU
allocation races across independent queues.
"""

from __future__ import annotations

import argparse
import fcntl
import json
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
    read_json,
    reserved_process_gpus,
    result_with_suffix,
    slice_eval_command,
    write_state,
)


CAPACITIES = (64, 1024)
METHODS = ("arithmetic_matched", "rq_shuffled", "semantic_rq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/semantic_hash_paper/capacity_sweep"),
    )
    parser.add_argument(
        "--gate1-slice-root", type=Path,
        default=Path("outputs/semantic_hash_paper/lm_slices"),
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--free-memory-threshold-mib", type=int, default=1500)
    return parser.parse_args()


def process_active(script_name: str) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            parts = [
                item.decode(errors="ignore")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except OSError:
            continue
        if any(Path(item).name == script_name for item in parts):
            return True
    return False


def table_complete(path: Path, capacity: int) -> bool:
    meta = read_json(path / "meta.json")
    return (
        meta.get("num_levels") == 8
        and meta.get("codebook_size") == capacity
        and (path / "keys_2.npy").is_file()
        and (path / "codes_2.npy").is_file()
        and (path / "keys_3.npy").is_file()
        and (path / "codes_3.npy").is_file()
    )


def benchmark_command(
    model: str, method: str, suffix: str
) -> tuple[str, ...]:
    return (
        sys.executable, "-u", "examples/compare_engram_lora.py",
        "--model_name", model,
        "--dataset", "fineweb",
        "--methods", method,
        "--max_steps", "12208",
        "--batch_size", "4",
        "--grad_accum", "8",
        "--max_length", "256",
        "--subset", "48832",
        "--num_workers", "4",
        "--seed", "42",
        "--run_suffix", suffix,
        "--disable_early_stopping",
        "--skip_plot",
        "--skip_inference",
    )


def make_tasks(args: argparse.Namespace) -> tuple[list[Task], dict[int, Path]]:
    tasks: list[Task] = []
    shuffled_dirs: dict[int, Path] = {}
    for capacity in CAPACITIES:
        table = Path(f"rq_tables/fineweb_paper_M8K{capacity}_100k").resolve()
        shuffled = Path(str(table) + "_shuffled_freqmatched_seed42")
        shuffled_dirs[capacity] = shuffled
        root = args.output_dir / f"k{capacity}"
        slices = root / "lm_slices"
        tasks.append(Task(
            name=f"capacity_build_k{capacity}",
            command=(
                sys.executable, "-u", "scripts/build_rq_table.py",
                "--dataset", "HuggingFaceFW/fineweb-edu",
                "--dataset_config", "sample-10BT",
                "--num_docs", "5000",
                "--max_doc_tokens", "512",
                "--base_tokenizer", args.model,
                "--embedder", args.embedder,
                "--num_levels", "8",
                "--codebook_size", str(capacity),
                "--max_ngrams_per_size", "100000",
                "--min_count", "2",
                "--output_dir", str(table),
                "--seed", "42",
            ),
            log_name=f"capacity_build_k{capacity}.log",
            done=lambda table=table, capacity=capacity: table_complete(table, capacity),
        ))
        access = slices / "manifest_seed42.npz"
        access_summary = slices / "manifest_seed42.json"
        tasks.append(Task(
            name=f"capacity_access_manifest_k{capacity}",
            command=(
                sys.executable, "-u", "scripts/build_lm_slice_manifest.py",
                "--table-dir", str(table),
                "--base-tokenizer", args.model,
                "--embedder", args.embedder,
                "--seed", "42",
                "--train-rows", "48832",
                "--eval-rows", "200",
                "--shuffle-buffer-size", "49032",
                "--output", str(access),
                "--summary", str(access_summary),
            ),
            log_name=f"capacity_access_manifest_k{capacity}.log",
            done=lambda path=access_summary: complete_metric(path),
            ready=lambda table=table, capacity=capacity: table_complete(table, capacity),
        ))
        manifest = slices / "manifest_eval2000_seed42.npz"
        manifest_summary = slices / "manifest_eval2000_seed42.json"
        tasks.append(Task(
            name=f"capacity_eval_manifest_k{capacity}",
            command=(
                sys.executable, "-u", "scripts/build_lm_slice_manifest.py",
                "--table-dir", str(table),
                "--base-tokenizer", args.model,
                "--embedder", args.embedder,
                "--seed", "42",
                "--train-rows", "48832",
                "--eval-rows", "2000",
                "--shuffle-buffer-size", "49032",
                "--output", str(manifest),
                "--summary", str(manifest_summary),
            ),
            log_name=f"capacity_eval_manifest_k{capacity}.log",
            done=lambda path=manifest_summary: complete_metric(path),
            ready=lambda table=table, capacity=capacity: table_complete(table, capacity),
        ))

        common = "n_head_per_ngram=8,use_sparse_embeddings=False,save_steps=1000,eval_steps=500"
        total = 8 * capacity
        specs = {
            "arithmetic_matched": (
                f"engram:hash_backend=arithmetic_fixed,{common},"
                f"engram_vocab_size_per_ngram=[{total},{total}]"
            ),
            "rq_shuffled": (
                f"engram:hash_backend=rq,rq_table_dir={shuffled},{common},"
                f"engram_vocab_size_per_ngram=[{total},{total}]"
            ),
            "semantic_rq": (
                f"engram:hash_backend=rq,rq_table_dir={table},{common},"
                f"engram_vocab_size_per_ngram=[{total},{total}]"
            ),
        }
        for method_name, method in specs.items():
            suffix = f"_paper_capacity_k{capacity}_fixedsteps_{method_name}_seed42"
            tasks.append(Task(
                name=f"capacity_train_k{capacity}_{method_name}",
                command=benchmark_command(args.model, method, suffix),
                log_name=f"capacity_train_k{capacity}_{method_name}.log",
                done=lambda suffix=suffix: result_with_suffix(suffix),
                ready=lambda shuffled=shuffled: (
                    shuffled / "shuffle_manifest.json"
                ).is_file(),
            ))
            output = slices / f"{method_name}_seed42.json"
            tasks.append(Task(
                name=f"capacity_slice_k{capacity}_{method_name}",
                command=slice_eval_command(
                    args.model, manifest, output, method_name, 42, suffix
                ),
                log_name=f"capacity_slice_k{capacity}_{method_name}.log",
                done=lambda output=output: complete_metric(output),
                ready=lambda summary=manifest_summary, suffix=suffix: (
                    complete_metric(summary) and result_with_suffix(suffix)
                ),
            ))

    analysis = args.output_dir / "comparison.json"
    tasks.append(Task(
        name="capacity_paired_bootstrap",
        command=(
            sys.executable, "-u", "scripts/analyze_capacity_sweep.py",
            "--capacity-root", str(args.output_dir),
            "--gate1-slice-root", str(args.gate1_slice_root),
            "--replicates", "10000",
            "--output", str(analysis),
        ),
        log_name="capacity_paired_bootstrap.log",
        done=lambda: complete_metric(analysis),
        ready=lambda: all(
            complete_metric(
                args.output_dir / f"k{capacity}" / "lm_slices"
                / f"{method}_seed42.json"
            )
            for capacity in CAPACITIES for method in METHODS
        ) and all(
            complete_metric(args.gate1_slice_root / f"{method}_seed42.json")
            for method in METHODS
        ),
    ))
    return tasks, shuffled_dirs


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "pipeline_state.json"
    lock_handle = (args.output_dir / "pipeline_state.json.lock").open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("capacity scheduler already active", flush=True)
        return

    while process_active("run_semantic_hash_pipeline.py") or process_active(
        "run_semantic_hash_phase2.py"
    ):
        write_state(state_path, {}, "waiting for earlier paper schedulers")
        time.sleep(args.poll_seconds)

    tasks, shuffled_dirs = make_tasks(args)
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

        for capacity, shuffled in shuffled_dirs.items():
            if (shuffled / "shuffle_manifest.json").is_file():
                continue
            table = Path(f"rq_tables/fineweb_paper_M8K{capacity}_100k").resolve()
            access = args.output_dir / f"k{capacity}" / "lm_slices" / "manifest_seed42.npz"
            if not table_complete(table, capacity) or not access.is_file():
                continue
            subprocess.run((
                sys.executable, "scripts/shuffle_rq_table.py",
                "--source-dir", str(table),
                "--output-dir", str(shuffled),
                "--seed", "42",
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
            "all capacity experiments complete" if finished else "capacity scheduler active",
        )
        if finished:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
