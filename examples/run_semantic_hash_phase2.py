#!/usr/bin/env python
"""Gate-controlled one-pass LM replication for the Semantic-RQ paper.

This scheduler waits for the three-seed Gate-1 paired bootstrap.  It starts the
one-pass replication only when the predeclared shared-code endpoint beats both
matched controls and overall NLL shows no significant regression.
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
    standard_lm_command,
    write_state,
)


SEEDS = (42, 43, 44)
METHODS = ("arithmetic_matched", "rq_shuffled", "semantic_rq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--table-dir", type=Path,
        default=Path("rq_tables/fineweb_paper_M8K256_100k"),
    )
    parser.add_argument(
        "--gate1-comparison", type=Path,
        default=Path("outputs/semantic_hash_paper/lm_slices/comparison.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/semantic_hash_paper/phase2_onepass"),
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--free-memory-threshold-mib", type=int, default=1500)
    return parser.parse_args()


def ci(payload: dict[str, Any], control: str, slice_name: str) -> list[float] | None:
    value = payload.get("aggregate", {}).get(control, {}).get(slice_name, {}).get("ci95")
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int | float) for item in value)
    ):
        return [float(value[0]), float(value[1])]
    return None


def gate_decision(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "complete":
        return {"status": "waiting", "reason": "Gate-1 bootstrap incomplete"}
    primary_name = "3gram_semantic_neighbor_shared_code"
    primary = {control: ci(payload, control, primary_name) for control in METHODS[:2]}
    overall = {control: ci(payload, control, "overall") for control in METHODS[:2]}
    interactions = {
        control: payload.get("interactions", {})
        .get(control, {})
        .get("3gram_shared_minus_no_shared", {})
        .get("ci95")
        for control in METHODS[:2]
    }
    if any(
        not (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(item, int | float) for item in value)
        )
        for value in interactions.values()
    ):
        return {"status": "failed", "reason": "required shared/no-shared interaction CI is missing"}
    if any(value is None for value in (*primary.values(), *overall.values())):
        return {"status": "failed", "reason": "required Gate-1 CI is missing"}
    primary_pass = all(value is not None and value[1] < 0 for value in primary.values())
    interaction_pass = all(value[1] < 0 for value in interactions.values())
    # Delta is Semantic-RQ minus control; a lower CI bound > 0 would prove regression.
    overall_safe = all(value is not None and value[0] <= 0 for value in overall.values())
    seed_directions: dict[str, list[bool]] = {}
    for control in METHODS[:2]:
        seed_directions[control] = [
            bool(
                payload.get("per_seed", {})
                .get(str(seed), {})
                .get(control, {})
                .get(primary_name, {})
                .get("delta_nll", float("inf"))
                < 0
            )
            for seed in SEEDS
        ]
    seed_consistent = all(sum(values) >= 2 for values in seed_directions.values())
    false_sharing = {
        control: {
            name: payload.get("aggregate", {}).get(control, {}).get(name, {})
            for name in (
                "2gram_covered_no_neighbor_high_lexical",
                "3gram_covered_no_neighbor_high_lexical",
            )
        }
        for control in METHODS[:2]
    }
    return {
        "status": (
            "pass"
            if primary_pass and interaction_pass and overall_safe and seed_consistent
            else "no_go"
        ),
        "primary": primary,
        "overall": overall,
        "interactions": interactions,
        "seed_directions": seed_directions,
        "false_sharing_proxy": false_sharing,
        "false_sharing_status": (
            "underpowered in Gate-1; mandatory PAWS-X matched triad remains queued"
        ),
        "criteria": {
            "primary": "both 95% CI upper bounds < 0",
            "interaction": "shared-minus-no-shared 95% CI upper bound < 0 for both controls",
            "overall_safety": "neither 95% CI lower bound > 0",
            "seed_consistency": "negative primary delta in at least 2/3 seeds for both controls",
            "false_sharing": "not inferred from tiny proxy slices; resolved by PAWS-X",
        },
        "primary_pass": primary_pass,
        "interaction_pass": interaction_pass,
        "overall_safe": overall_safe,
        "seed_consistent": seed_consistent,
    }


def benchmark_command(
    model: str, method: str, suffix: str, seed: int
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
        # 390,656 / 32 = 12,208: exactly one complete epoch.
        "--subset", "390656",
        "--num_workers", "4",
        "--seed", str(seed),
        "--run_suffix", suffix,
        "--disable_early_stopping",
        "--skip_plot",
        "--skip_inference",
    )


def make_tasks(
    args: argparse.Namespace, shuffled_dirs: dict[int, Path]
) -> tuple[list[Task], dict[tuple[str, int], str]]:
    root = args.output_dir / "lm_slices"
    access_tasks: list[Task] = []
    eval_manifest_tasks: list[Task] = []
    suffixes: dict[tuple[str, int], str] = {}
    for seed in SEEDS:
        access = root / f"manifest_seed{seed}.npz"
        access_summary = root / f"manifest_seed{seed}.json"
        access_tasks.append(Task(
            name=f"phase2_access_manifest_seed{seed}",
            command=(
                sys.executable, "-u", "scripts/build_lm_slice_manifest.py",
                "--table-dir", str(args.table_dir),
                "--base-tokenizer", args.model,
                "--embedder", args.embedder,
                "--seed", str(seed),
                "--train-rows", "390656",
                "--eval-rows", "200",
                "--shuffle-buffer-size", "390856",
                "--output", str(access),
                "--summary", str(access_summary),
            ),
            log_name=f"phase2_access_manifest_seed{seed}.log",
            done=lambda path=access_summary: complete_metric(path),
        ))
        evaluation = root / f"manifest_eval2000_seed{seed}.npz"
        evaluation_summary = root / f"manifest_eval2000_seed{seed}.json"
        eval_manifest_tasks.append(Task(
            name=f"phase2_eval_manifest_seed{seed}",
            command=(
                sys.executable, "-u", "scripts/build_lm_slice_manifest.py",
                "--table-dir", str(args.table_dir),
                "--base-tokenizer", args.model,
                "--embedder", args.embedder,
                "--seed", str(seed),
                "--train-rows", "390656",
                "--eval-rows", "2000",
                "--shuffle-buffer-size", "390856",
                "--output", str(evaluation),
                "--summary", str(evaluation_summary),
            ),
            log_name=f"phase2_eval_manifest_seed{seed}.log",
            done=lambda path=evaluation_summary: complete_metric(path),
        ))

    audit_output = root / "manifest_audit.json"
    audit = Task(
        name="phase2_manifest_audit",
        command=(
            sys.executable, "-u", "scripts/audit_lm_manifests.py",
            "--slice-root", str(root), "--output", str(audit_output),
        ),
        log_name="phase2_manifest_audit.log",
        done=lambda: complete_metric(audit_output)
        and read_json(audit_output).get("all_train_access_counts_identical") is True,
        ready=lambda: all(
            complete_metric(root / f"manifest_seed{seed}.json")
            and complete_metric(root / f"manifest_eval2000_seed{seed}.json")
            for seed in SEEDS
        ),
    )

    common = "n_head_per_ngram=8,use_sparse_embeddings=False,save_steps=1000,eval_steps=500"
    training: list[Task] = []
    for seed in SEEDS:
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
                f"engram:hash_backend=rq,rq_table_dir={args.table_dir},{common},"
                "engram_vocab_size_per_ngram=[2048,2048]"
            ),
        }
        for name, method in specs.items():
            suffix = f"_paper_phase2_fineweb_onepass_fixedsteps_{name}_seed{seed}"
            suffixes[(name, seed)] = suffix
            training.append(Task(
                name=f"phase2_train_{name}_seed{seed}",
                command=benchmark_command(args.model, method, suffix, seed),
                log_name=f"phase2_train_{name}_seed{seed}.log",
                done=lambda suffix=suffix: result_with_suffix(suffix),
                requires_table=True,
            ))

    evaluations: list[Task] = []
    for seed in SEEDS:
        manifest = root / f"manifest_eval2000_seed{seed}.npz"
        summary = root / f"manifest_eval2000_seed{seed}.json"
        base_output = root / f"base_seed{seed}.json"
        evaluations.append(Task(
            name=f"phase2_slice_base_seed{seed}",
            command=slice_eval_command(args.model, manifest, base_output, "base", seed),
            log_name=f"phase2_slice_base_seed{seed}.log",
            done=lambda output=base_output: complete_metric(output),
            ready=lambda summary=summary: complete_metric(summary),
        ))
        for name in METHODS:
            suffix = suffixes[(name, seed)]
            output = root / f"{name}_seed{seed}.json"
            evaluations.append(Task(
                name=f"phase2_slice_{name}_seed{seed}",
                command=slice_eval_command(
                    args.model, manifest, output, name, seed, suffix
                ),
                log_name=f"phase2_slice_{name}_seed{seed}.log",
                done=lambda output=output: complete_metric(output),
                requires_table=True,
                ready=lambda summary=summary, suffix=suffix: (
                    complete_metric(summary) and result_with_suffix(suffix)
                ),
            ))
    comparison = root / "comparison.json"
    bootstrap = Task(
        name="phase2_slice_bootstrap",
        command=(
            sys.executable, "-u", "scripts/analyze_lm_slice_results.py",
            "--slice-root", str(root), "--replicates", "10000",
            "--output", str(comparison),
        ),
        log_name="phase2_slice_bootstrap.log",
        done=lambda: complete_metric(comparison),
        requires_table=True,
        ready=lambda: all(
            complete_metric(root / f"{method}_seed{seed}.json")
            for seed in SEEDS for method in METHODS
        ),
    )

    standard_root = args.output_dir / "standard_lm"
    standards: list[Task] = []
    base_standard = standard_root / "base_seed42.json"
    standards.append(Task(
        name="phase2_standard_base_seed42",
        command=standard_lm_command(args.model, base_standard, "base", 42),
        log_name="phase2_standard_base_seed42.log",
        done=lambda: complete_metric(base_standard),
        ready=lambda: all(result_with_suffix(suffixes[(name, 42)]) for name in METHODS),
    ))
    for seed in SEEDS:
        for name in METHODS:
            suffix = suffixes[(name, seed)]
            output = standard_root / f"{name}_seed{seed}.json"
            standards.append(Task(
                name=f"phase2_standard_{name}_seed{seed}",
                command=standard_lm_command(args.model, output, name, seed, suffix),
                log_name=f"phase2_standard_{name}_seed{seed}.log",
                done=lambda output=output: complete_metric(output),
                requires_table=True,
                ready=lambda suffix=suffix: result_with_suffix(suffix),
            ))
    return (
        [*access_tasks, *eval_manifest_tasks, audit, *training, *evaluations, bootstrap, *standards],
        suffixes,
    )


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    args.table_dir = args.table_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "pipeline_state.json"
    lock = (args.output_dir / "pipeline_state.json.lock").open("w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("phase2 scheduler already active", flush=True)
        return

    while True:
        decision = gate_decision(read_json(args.gate1_comparison))
        decision_path = args.output_dir / "gate_decision.json"
        temporary = decision_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        os.replace(temporary, decision_path)
        if decision["status"] == "pass":
            break
        write_state(state_path, {}, f"phase2 {decision['status']}: {decision.get('reason', '')}")
        if decision["status"] in {"no_go", "failed"}:
            return
        time.sleep(args.poll_seconds)

    shuffled = {
        seed: Path(str(args.table_dir) + f"_onepass_freqmatched_seed{seed}")
        for seed in SEEDS
    }
    tasks, _ = make_tasks(args, shuffled)
    children: dict[str, subprocess.Popen[bytes]] = {}
    running: dict[str, dict[str, Any]] = {}
    while True:
        for name, process in list(children.items()):
            return_code = process.poll()
            if return_code is not None:
                running[name]["return_code"] = return_code
                running[name]["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                del children[name]
        for seed, directory in shuffled.items():
            if (directory / "shuffle_manifest.json").is_file():
                continue
            access = args.output_dir / "lm_slices" / f"manifest_seed{seed}.npz"
            if not access.is_file():
                continue
            subprocess.run((
                sys.executable, "scripts/shuffle_rq_table.py",
                "--source-dir", str(args.table_dir),
                "--output-dir", str(directory),
                "--seed", str(seed),
                "--access-counts", str(access),
            ), check=True)
        controls_ready = all(
            (directory / "shuffle_manifest.json").is_file()
            for directory in shuffled.values()
        )
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
            if task.requires_table and not controls_ready:
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
                "pid": process.pid, "gpu": gpu, "log": str(log_path),
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "command": list(task.command),
            }
        finished = all(task.done() for task in tasks)
        write_state(
            state_path, running,
            "all one-pass experiments complete" if finished else "phase2 scheduler active",
        )
        if finished:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
