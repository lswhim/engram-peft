"""Keep four GPUs busy with the staged Semantic-RQ paper experiment queue.

The scheduler never kills an existing process.  It waits for a genuinely free GPU,
builds the FineWeb-Edu address table, creates the frequency-identical RQ-Shuffled
control, and then launches the matched Gate-1 LM matrix.  Corrected cross-lingual
matched runs fill otherwise idle GPUs while the address table is being built.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Task:
    name: str
    command: tuple[str, ...]
    log_name: str
    done: Callable[[], bool]
    requires_table: bool = False
    ready: Callable[[], bool] = lambda: True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--control-rq-table",
        type=Path,
        default=Path("rq_tables/wiki15_qwen3_06b_M8K256_500k"),
        help="Existing RQ metadata used only to size corrected external controls.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_hash_paper"))
    parser.add_argument("--table-dir", type=Path, default=Path("rq_tables/fineweb_paper_M8K256_100k"))
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--state-name", default="pipeline_state.json", help="State JSON inside output-dir."
    )
    parser.add_argument("--free-memory-threshold-mib", type=int, default=1500)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def complete_metric(path: Path) -> bool:
    return read_json(path).get("status") == "complete"


def result_with_suffix(suffix: str) -> bool:
    for path in Path("outputs/benchmarks").glob("*.json"):
        payload = read_json(path)
        params = payload.get("params")
        if isinstance(params, dict) and params.get("run_suffix") == suffix:
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict) or not isinstance(
                metrics.get("eval_loss"), int | float
            ):
                return False
            if "_fixedsteps_" in suffix:
                return (
                    metrics.get("fixed_steps_complete") is True
                    and metrics.get("completed_steps") == 12_208
                    and metrics.get("planned_steps") == 12_208
                )
            return True
    return False


def gpu_memory() -> dict[int, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    result: dict[int, int] = {}
    for line in output.splitlines():
        index, used = (part.strip() for part in line.split(",", 1))
        result[int(index)] = int(used)
    return result


def active_command_contains(marker: str) -> bool:
    proc = Path("/proc")
    if not proc.exists():
        return False
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if marker in command and "run_semantic_hash_pipeline.py" not in command:
            return True
    return False


def active_command_matches(command: tuple[str, ...]) -> bool:
    """Return true when a live process argv exactly contains the task argv.

    Task names are scheduler metadata and do not appear in child argv.  Exact
    argv matching prevents a restarted scheduler from duplicating a manually
    adopted or predecessor-owned run on another free GPU.
    """
    proc = Path("/proc")
    if not proc.exists():
        return False
    expected = list(command)
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parts = [
                value.decode(errors="ignore")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except OSError:
            continue
        if len(parts) < len(expected):
            continue
        for start in range(len(parts) - len(expected) + 1):
            if parts[start : start + len(expected)] == expected:
                return True
    return False


def another_fixedsteps_scheduler_active() -> bool:
    """Prevent a legacy handoff supervisor from starting a duplicate scheduler."""
    proc = Path("/proc")
    if not proc.exists():
        return False
    current_pid = os.getpid()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == current_pid:
            continue
        try:
            parts = [
                value.decode(errors="ignore")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except OSError:
            continue
        if not any(Path(value).name == "run_semantic_hash_pipeline.py" for value in parts):
            continue
        if "--state-name" in parts:
            index = parts.index("--state-name")
            if index + 1 < len(parts) and parts[index + 1] == "pipeline_state_fixedsteps.json":
                return True
    return False


def reserved_process_gpus() -> set[int]:
    """Return CUDA devices claimed by live experiment processes.

    CUDA allocation is delayed during dataset/tokenizer loading, so memory usage
    alone is not a sufficient reservation signal.  Reading the process environment
    also makes a restarted scheduler respect jobs launched by its predecessor.
    """
    reserved: set[int] = set()
    proc = Path("/proc")
    if not proc.exists():
        return reserved
    experiment_markers = (
        "build_rq_table.py",
        "build_lm_slice_manifest.py",
        "evaluate_lm_slices.py",
        "run_xtreme_xnli.py",
        "run_xtreme_pawsx.py",
        "compare_engram_lora.py",
    )
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().decode(errors="ignore")
            if not any(marker in command for marker in experiment_markers):
                continue
            environment = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        for item in environment:
            if not item.startswith(b"CUDA_VISIBLE_DEVICES="):
                continue
            value = item.split(b"=", 1)[1].decode(errors="ignore")
            if value.isdigit():
                reserved.add(int(value))
    return reserved


def write_state(path: Path, running: dict[str, dict[str, object]], note: str) -> None:
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": note,
        "running": running,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def benchmark_command(
    model: str, method: str, suffix: str, seed: int
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-u",
        "examples/compare_engram_lora.py",
        "--model_name", model,
        "--dataset", "fineweb",
        "--methods", method,
        "--max_steps", "12208",
        "--batch_size", "4",
        "--grad_accum", "8",
        "--max_length", "256",
        # 48,832 / (batch 4 * grad_accum 8) = 1,526 optimizer
        # steps/epoch; 12,208 steps are exactly 8 full epochs.  This makes the
        # single-pass access counts used by the frequency-matched control exact
        # up to a constant factor over the entire training trajectory.
        "--subset", "48832",
        "--num_workers", "4",
        "--seed", str(seed),
        "--run_suffix", suffix,
        "--disable_early_stopping",
        "--skip_plot",
        "--skip_inference",
    )


def crosslingual_command(
    runner: str, model: str, output_dir: str, seed: int, table_dir: Path
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-u",
        f"examples/{runner}",
        "--method", "arithmetic_matched",
        "--model", model,
        "--rq_table_dir", str(table_dir),
        "--output_dir", output_dir,
        "--epochs", "1",
        "--batch_size", "4",
        "--eval_batch_size", "8",
        "--grad_accum", "8",
        "--max_length", "256",
        "--num_workers", "4",
        "--seed", str(seed),
    )


def slice_eval_command(
    model: str,
    manifest: Path,
    output: Path,
    method: str,
    seed: int,
    suffix: str | None = None,
    head_mask: str = "none",
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-u",
        "examples/evaluate_lm_slices.py",
        "--model", model,
        "--manifest", str(manifest),
        "--method", method,
        "--seed", str(seed),
        "--head-mask", head_mask,
        "--output", str(output),
    ]
    if suffix is not None:
        command.extend(("--result-suffix", suffix))
    return tuple(command)


def standard_lm_command(
    model: str,
    output: Path,
    method: str,
    seed: int,
    task: str,
    suffix: str | None = None,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-u",
        "examples/evaluate_standard_lm.py",
        "--model", model,
        "--method", method,
        "--seed", str(seed),
        "--tasks", task,
        "--batch-size", "1",
        "--output", str(output),
    ]
    if suffix is not None:
        command.extend(("--result-suffix", suffix))
    return tuple(command)


def make_tasks(args: argparse.Namespace, shuffled_dirs: dict[int, Path]) -> list[Task]:
    table_dir = args.table_dir
    build_marker = f"--output_dir {table_dir}"
    build = Task(
        name="gate0_build_fineweb_rq",
        command=(
            sys.executable,
            "-u",
            "scripts/build_rq_table.py",
            "--dataset", "HuggingFaceFW/fineweb-edu",
            "--dataset_config", "sample-10BT",
            "--num_docs", "5000",
            "--max_doc_tokens", "512",
            "--base_tokenizer", args.model,
            "--embedder", args.embedder,
            "--num_levels", "8",
            "--codebook_size", "256",
            "--max_ngrams_per_size", "100000",
            "--min_count", "2",
            "--output_dir", str(table_dir),
            "--seed", "42",
        ),
        log_name="gate0_build_fineweb_rq.log",
        done=lambda: (table_dir / "meta.json").is_file(),
    )

    fillers: list[Task] = []
    for benchmark, runner in (("xnli", "run_xtreme_xnli.py"), ("pawsx", "run_xtreme_pawsx.py")):
        output = Path(f"outputs/xtreme_{benchmark}_matched_exact")
        for seed in (42, 43):
            result = output / f"arithmetic_matched_seed{seed}" / "metrics.json"
            fillers.append(
                Task(
                    name=f"{benchmark}_matched_exact_seed{seed}",
                    command=crosslingual_command(
                        runner, args.model, str(output), seed, args.control_rq_table
                    ),
                    log_name=f"{benchmark}_matched_exact_seed{seed}.log",
                    done=lambda result=result: complete_metric(result),
                )
            )

    common = (
        "n_head_per_ngram=8,use_sparse_embeddings=False,"
        "save_steps=1000,eval_steps=500"
    )
    gate1: list[Task] = []
    suffixes: dict[tuple[str, int], str] = {}
    for seed in (42, 43, 44):
        specs = {
            "arithmetic_matched": (
                f"engram:hash_backend=arithmetic_fixed,{common},"
                "engram_vocab_size_per_ngram=[2048,2048]"
            ),
            "rq_shuffled": (
                f"engram:hash_backend=rq,rq_table_dir={shuffled_dirs[seed]},{common},"
                "engram_vocab_size_per_ngram=[2048,2048]"
            ),
            "semantic_rq": (
                f"engram:hash_backend=rq,rq_table_dir={table_dir},{common},"
                "engram_vocab_size_per_ngram=[2048,2048]"
            ),
        }
        for name, method in specs.items():
            suffix = f"_paper_gate1_fineweb_100m_fixedsteps_{name}_seed{seed}"
            suffixes[(name, seed)] = suffix
            task_name = f"gate1_{name}" if seed == 42 else f"gate1_{name}_seed{seed}"
            gate1.append(
                Task(
                    name=task_name,
                    command=benchmark_command(args.model, method, suffix, seed),
                    log_name=f"gate1_{name}_seed{seed}.log",
                    done=lambda suffix=suffix: result_with_suffix(suffix),
                    requires_table=True,
                )
            )
    slice_root = args.output_dir / "lm_slices"
    manifests: list[Task] = []
    evaluation_manifests: list[Task] = []
    slice_evaluations: list[Task] = []
    for seed in (42, 43, 44):
        access_manifest = slice_root / f"manifest_seed{seed}.npz"
        access_summary = slice_root / f"manifest_seed{seed}.json"
        manifests.append(
            Task(
                name=f"lm_slice_manifest_seed{seed}",
                command=(
                    sys.executable,
                    "-u",
                    "scripts/build_lm_slice_manifest.py",
                    "--table-dir", str(table_dir),
                    "--base-tokenizer", args.model,
                    "--embedder", args.embedder,
                    "--seed", str(seed),
                    "--train-rows", "48832",
                    "--eval-rows", "200",
                    "--shuffle-buffer-size", "49032",
                    "--output", str(access_manifest),
                    "--summary", str(access_summary),
                ),
                log_name=f"lm_slice_manifest_seed{seed}.log",
                done=lambda summary=access_summary: complete_metric(summary),
                ready=lambda table_dir=table_dir: (table_dir / "meta.json").is_file(),
            )
        )
        manifest = slice_root / f"manifest_eval2000_seed{seed}.npz"
        summary = slice_root / f"manifest_eval2000_seed{seed}.json"
        evaluation_manifests.append(
            Task(
                name=f"lm_slice_manifest_eval2000_seed{seed}",
                command=(
                    sys.executable,
                    "-u",
                    "scripts/build_lm_slice_manifest.py",
                    "--table-dir", str(table_dir),
                    "--base-tokenizer", args.model,
                    "--embedder", args.embedder,
                    "--seed", str(seed),
                    "--train-rows", "48832",
                    "--eval-rows", "2000",
                    "--shuffle-buffer-size", "49032",
                    "--output", str(manifest),
                    "--summary", str(summary),
                ),
                log_name=f"lm_slice_manifest_eval2000_seed{seed}.log",
                done=lambda summary=summary: complete_metric(summary),
                ready=lambda table_dir=table_dir: (table_dir / "meta.json").is_file(),
            )
        )
        base_output = slice_root / f"base_seed{seed}.json"
        slice_evaluations.append(
            Task(
                name=f"lm_slice_base_seed{seed}",
                command=slice_eval_command(
                    args.model, manifest, base_output, "base", seed
                ),
                log_name=f"lm_slice_base_seed{seed}.log",
                done=lambda output=base_output: complete_metric(output),
                ready=lambda summary=summary: complete_metric(summary),
            )
        )
        for name in ("arithmetic_matched", "rq_shuffled", "semantic_rq"):
            suffix = suffixes[(name, seed)]
            output = slice_root / f"{name}_seed{seed}.json"
            slice_evaluations.append(
                Task(
                    name=f"lm_slice_{name}_seed{seed}",
                    command=slice_eval_command(
                        args.model, manifest, output, name, seed, suffix
                    ),
                    log_name=f"lm_slice_{name}_seed{seed}.log",
                    done=lambda output=output: complete_metric(output),
                    requires_table=True,
                    ready=lambda summary=summary, suffix=suffix: (
                        complete_metric(summary) and result_with_suffix(suffix)
                    ),
                )
            )

    manifest_audit_output = slice_root / "manifest_audit.json"
    manifest_audit = Task(
        name="lm_slice_manifest_consistency_audit",
        command=(
            sys.executable,
            "-u",
            "scripts/audit_lm_manifests.py",
            "--slice-root", str(slice_root),
            "--output", str(manifest_audit_output),
        ),
        log_name="lm_slice_manifest_consistency_audit.log",
        done=lambda: (
            complete_metric(manifest_audit_output)
            and read_json(manifest_audit_output).get(
                "all_train_access_counts_identical"
            )
            is True
        ),
        ready=lambda: all(
            complete_metric(slice_root / f"manifest_seed{seed}.json")
            and complete_metric(
                slice_root / f"manifest_eval2000_seed{seed}.json"
            )
            for seed in (42, 43, 44)
        ),
    )

    comparison_output = slice_root / "comparison.json"
    slice_comparison = Task(
        name="lm_slice_paired_bootstrap",
        command=(
            sys.executable,
            "-u",
            "scripts/analyze_lm_slice_results.py",
            "--slice-root", str(slice_root),
            "--replicates", "10000",
            "--output", str(comparison_output),
        ),
        log_name="lm_slice_paired_bootstrap.log",
        done=lambda: complete_metric(comparison_output),
        requires_table=True,
        ready=lambda: all(
            complete_metric(slice_root / f"{method}_seed{seed}.json")
            for seed in (42, 43, 44)
            for method in ("arithmetic_matched", "rq_shuffled", "semantic_rq")
        ),
    )

    intervention_root = args.output_dir / "head_intervention"
    intervention_evaluations: list[Task] = []
    for seed in (42, 43, 44):
        manifest = slice_root / f"manifest_eval2000_seed{seed}.npz"
        summary = slice_root / f"manifest_eval2000_seed{seed}.json"
        suffix = suffixes[("semantic_rq", seed)]
        for head_mask, filename in (
            ("shared", "shared"),
            ("random-matched", "random_matched"),
        ):
            output = intervention_root / f"{filename}_seed{seed}.json"
            intervention_evaluations.append(
                Task(
                    name=f"head_intervention_{filename}_seed{seed}",
                    command=slice_eval_command(
                        args.model,
                        manifest,
                        output,
                        "semantic_rq",
                        seed,
                        suffix,
                        head_mask,
                    ),
                    log_name=f"head_intervention_{filename}_seed{seed}.log",
                    done=lambda output=output, head_mask=head_mask: (
                        complete_metric(output)
                        and read_json(output).get("head_mask") == head_mask
                    ),
                    requires_table=True,
                    ready=lambda summary=summary, suffix=suffix: (
                        complete_metric(summary) and result_with_suffix(suffix)
                    ),
                )
            )

    intervention_output = intervention_root / "comparison.json"
    intervention_comparison = Task(
        name="head_intervention_paired_bootstrap",
        command=(
            sys.executable,
            "-u",
            "scripts/analyze_shared_head_intervention.py",
            "--slice-root", str(slice_root),
            "--intervention-root", str(intervention_root),
            "--replicates", "10000",
            "--output", str(intervention_output),
        ),
        log_name="head_intervention_paired_bootstrap.log",
        done=lambda: complete_metric(intervention_output),
        requires_table=True,
        ready=lambda: all(
            complete_metric(intervention_root / f"{name}_seed{seed}.json")
            for seed in (42, 43, 44)
            for name in ("shared", "random_matched")
        )
        and all(
            complete_metric(slice_root / f"semantic_rq_seed{seed}.json")
            for seed in (42, 43, 44)
        ),
    )

    standard_root = args.output_dir / "standard_lm"
    standard_evaluations: list[Task] = []
    public_tasks = {
        "wikitext": "wikitext",
        "lambada": "lambada_openai",
        "c4": "c4",
    }
    for seed in (42, 43, 44):
        for name in ("arithmetic_matched", "rq_shuffled", "semantic_rq"):
            suffix = suffixes[(name, seed)]
            for label, task in public_tasks.items():
                output = standard_root / f"{name}_seed{seed}_{label}.json"
                standard_evaluations.append(
                    Task(
                        name=f"standard_lm_{name}_seed{seed}_{label}",
                        command=standard_lm_command(
                            args.model, output, name, seed, task, suffix
                        ),
                        log_name=f"standard_lm_{name}_seed{seed}_{label}.log",
                        done=lambda output=output: (
                            complete_metric(output)
                            and read_json(output).get("paper_eligible") is True
                        ),
                        requires_table=True,
                        ready=lambda suffix=suffix: result_with_suffix(suffix),
                    )
                )

    # Benchmark-first paper protocol: completed matched checkpoints are sent
    # directly to public LM evaluation.  Bespoke token slices and masking are
    # intentionally not scheduled here; they become conditional follow-ups
    # only if the public benchmark gate is positive.
    return [
        build,
        *standard_evaluations,
        *gate1,
        *manifests,
        *evaluation_manifests,
        manifest_audit,
        *fillers,
    ]


def main() -> None:
    args = parse_args()
    if args.state_name != "pipeline_state_fixedsteps.json" and another_fixedsteps_scheduler_active():
        print("fixedsteps scheduler already active; legacy handoff exits", flush=True)
        return
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    args.table_dir = args.table_dir.resolve()
    shuffled_dirs = {
        seed: Path(str(args.table_dir) + f"_shuffled_freqmatched_seed{seed}")
        for seed in (42, 43, 44)
    }
    state_path = args.output_dir / args.state_name
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"scheduler already owns {lock_path}; exiting", flush=True)
        return
    tasks = make_tasks(args, shuffled_dirs)
    children: dict[str, subprocess.Popen[bytes]] = {}
    running: dict[str, dict[str, object]] = {}

    while True:
        for name, process in list(children.items()):
            return_code = process.poll()
            if return_code is not None:
                running[name]["return_code"] = return_code
                running[name]["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                del children[name]

        table_ready = (args.table_dir / "meta.json").is_file()
        if table_ready:
            for seed, shuffled_dir in shuffled_dirs.items():
                if (shuffled_dir / "shuffle_manifest.json").is_file():
                    continue
                access_counts = (
                    args.output_dir / "lm_slices" / f"manifest_seed{seed}.npz"
                )
                if not access_counts.is_file():
                    continue
                if shuffled_dir.exists() and not any(shuffled_dir.iterdir()):
                    shuffled_dir.rmdir()
                command = [
                    sys.executable,
                    "scripts/shuffle_rq_table.py",
                    "--source-dir", str(args.table_dir),
                    "--output-dir", str(shuffled_dir),
                    "--seed", str(seed),
                    "--access-counts", str(access_counts),
                ]
                subprocess.run(command, check=True)

        table_controls_ready = all(
            (directory / "shuffle_manifest.json").is_file()
            for directory in shuffled_dirs.values()
        )
        memory = gpu_memory()
        # nvidia-smi can report 0 MiB for tens of seconds while a newly spawned
        # process downloads data or loads tokenizer/model weights.  Treat every
        # GPU assigned to a live child as reserved even before CUDA is initialized.
        reserved_gpus = reserved_process_gpus() | {
            int(running[name]["gpu"])
            for name in children
            if name in running and isinstance(running[name].get("gpu"), int)
        }
        free = [
            gpu
            for gpu, used in memory.items()
            if used < args.free_memory_threshold_mib and gpu not in reserved_gpus
        ]

        for task in tasks:
            if not free:
                break
            if task.done() or task.name in children or active_command_matches(task.command):
                continue
            if task.requires_table and not table_controls_ready:
                continue
            if not task.ready():
                continue
            # The build is the only table-independent dependency and gets first priority.
            gpu = free.pop(0)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["PYTHONPATH"] = f"{root / 'src'}:{root}"
            environment["TOKENIZERS_PARALLELISM"] = "false"
            environment.setdefault("http_proxy", "http://star-proxy.oa.com:3128")
            environment.setdefault("https_proxy", "http://star-proxy.oa.com:3128")
            environment.setdefault(
                "no_proxy", ".woa.com,.oa.com,.tencentcos.cn,localhost,127.0.0.1"
            )
            environment.setdefault("HF_HUB_DISABLE_XET", "1")
            log_path = logs / task.log_name
            log_handle = log_path.open("ab")
            process = subprocess.Popen(
                task.command,
                cwd=root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log_handle.close()
            children[task.name] = process
            running[task.name] = {
                "pid": process.pid,
                "gpu": gpu,
                "log": str(log_path),
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "command": list(task.command),
            }

        finished = all(task.done() for task in tasks)
        note = "all queued experiments complete" if finished else "scheduler active"
        write_state(state_path, running, note)
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {note}; gpu_mib={memory}", flush=True)
        if finished:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
