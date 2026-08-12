"""Render the live Semantic-RQ paper status, results, and cautious conclusions."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


GATE1 = ("arithmetic_matched", "rq_shuffled", "semantic_rq", "mixed_4_4")
GATE1_SEEDS = (42, 43, 44)
STANDARD_METHODS = ("arithmetic_matched", "rq_shuffled", "semantic_rq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_hash_paper"))
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def log_progress(path: Path) -> dict[str, Any]:
    """Parse the last Trainer progress/loss without rereading a multi-hour log."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 262_144))
            content = handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return {}
    steps = list(re.finditer(r"(\d{1,3})%\|[^\r\n]*?\|\s*(\d+)/(\d+)", content))
    losses = list(re.finditer(r"['\"]loss['\"]:\s*([0-9.eE+-]+)", content))
    eval_losses = list(
        re.finditer(r"['\"]eval_loss['\"]:\s*([0-9.eE+-]+)", content)
    )
    progress: dict[str, Any] = {}
    if steps:
        match = steps[-1]
        progress.update(
            percent=int(match.group(1)),
            step=int(match.group(2)),
            total_steps=int(match.group(3)),
        )
    if losses:
        progress["loss"] = float(losses[-1].group(1))
    if eval_losses:
        progress["eval_loss"] = float(eval_losses[-1].group(1))
    return progress


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in output.splitlines():
        index, name, used, total, util = (part.strip() for part in line.split(","))
        rows.append({"index": int(index), "name": name, "used_mib": int(used), "total_mib": int(total), "util": int(util)})
    return rows


def active_experiments() -> list[dict[str, Any]]:
    """Discover one row per live experiment, including jobs from older relays."""
    proc = Path("/proc")
    if not proc.exists():
        return []
    target_scripts = {
        "build_rq_table.py",
        "build_lm_slice_manifest.py",
        "build_crosslingual_access_counts.py",
        "shuffle_rq_table.py",
        "analyze_rq_addresses.py",
        "analyze_lm_slice_results.py",
        "analyze_capacity_sweep.py",
        "evaluate_lm_slices.py",
        "evaluate_standard_lm.py",
        "run_xtreme_xnli.py",
        "run_xtreme_pawsx.py",
        "compare_engram_lora.py",
    }
    found: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parts = [part.decode(errors="ignore") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
            command = " ".join(parts)
            script_arg = next((value for value in parts if Path(value).name in target_scripts), None)
            # A relay shell may contain a complete command as one `bash -c`
            # argument.  It is not itself an experiment and has no trustworthy
            # CUDA_VISIBLE_DEVICES assignment, so require an exact argv item.
            if script_arg is None:
                continue
            environment = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        gpu = "?"
        for item in environment:
            if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                gpu = item.split(b"=", 1)[1].decode(errors="ignore")
                break
        script = Path(script_arg).name
        method_match = re.search(r"--method(?:s)?\s+([^\s]+)", command)
        seed_match = re.search(r"--seed\s+(\d+)", command)
        suffix_match = re.search(r"--run_suffix\s+([^\s]+)", command)
        output_match = re.search(r"--output_dir\s+([^\s]+)", command)
        method = method_match.group(1).split(":", 1)[0] if method_match else (
            "semantic-rq table"
            if script == "build_rq_table.py"
            else "slice manifest"
            if script == "build_lm_slice_manifest.py"
            else "crosslingual access audit"
            if script == "build_crosslingual_access_counts.py"
            else "frequency-matched shuffle"
            if script == "shuffle_rq_table.py"
            else "paired bootstrap"
            if script in {"analyze_lm_slice_results.py", "analyze_capacity_sweep.py"}
            else "address diagnostics"
            if script == "analyze_rq_addresses.py"
            else "—"
        )
        run_suffix = suffix_match.group(1) if suffix_match else None
        if run_suffix:
            for pattern in (
                r"_paper_gate1_fineweb_100m_fixedsteps_(.+)_seed\d+",
                r"_paper_phase2_fineweb_onepass_fixedsteps_(.+)_seed\d+",
                r"_paper_capacity_k(?:64|1024)_fixedsteps_(.+)_seed42",
            ):
                suffix_method = re.fullmatch(pattern, run_suffix)
                if suffix_method:
                    method = suffix_method.group(1)
                    break
        seed = seed_match.group(1) if seed_match else "—"
        phase = "运行中"
        if script in {"run_xtreme_xnli.py", "run_xtreme_pawsx.py"}:
            expected_languages = 15 if script == "run_xtreme_xnli.py" else 7
            metrics_path = None
            if output_match and seed != "—":
                metrics_path = (
                    Path(output_match.group(1))
                    / f"{method}_seed{seed}"
                    / "metrics.json"
                )
            metrics = read_json(metrics_path) if metrics_path is not None else {}
            status = metrics.get("status")
            languages = metrics.get("languages")
            completed_languages = (
                sum(key != "macro" for key in languages)
                if isinstance(languages, dict)
                else 0
            )
            latest_result = ""
            if isinstance(languages, dict):
                language_keys = [key for key in languages if key != "macro"]
                if language_keys:
                    latest_language = language_keys[-1]
                    latest_accuracy = languages.get(latest_language)
                    if isinstance(latest_accuracy, int | float):
                        latest_result = (
                            f"；{latest_language}={100.0 * latest_accuracy:.2f}"
                        )
            if status == "complete":
                phase = f"完成（{completed_languages}/{expected_languages} 语言）"
            elif status == "evaluating":
                phase = (
                    f"评测中（{completed_languages}/{expected_languages} "
                    f"语言{latest_result}）"
                )
            else:
                phase = "训练中"
        key = (gpu, script, method, seed)
        pid = int(entry.name)
        progress: dict[str, Any] = {}
        log_path: str | None = None
        try:
            stdout_path = (entry / "fd" / "1").resolve(strict=True)
            if stdout_path.is_file():
                log_path = str(stdout_path)
                progress = log_progress(stdout_path)
        except OSError:
            pass
        previous = found.get(key)
        if previous is None or pid < previous["pid"]:
            found[key] = {
                "pid": pid,
                "gpu": gpu,
                "script": script,
                "method": method,
                "seed": seed,
                "run_suffix": run_suffix,
                "phase": phase,
                "log": log_path,
                "progress": progress,
            }
    return sorted(found.values(), key=lambda row: (str(row["gpu"]), row["pid"]))


def benchmark_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in Path("outputs/benchmarks").glob("*.json"):
        payload = read_json(path)
        params = payload.get("params")
        if not isinstance(params, dict):
            continue
        suffix = str(params.get("run_suffix", ""))
        match = re.fullmatch(
            r"_paper_gate1_fineweb_100m_fixedsteps_(.+)_seed(\d+)", suffix
        )
        if not match:
            continue
        name = match.group(1)
        seed = int(match.group(2))
        metrics = payload.get("metrics")
        if not (
            isinstance(metrics, dict)
            and metrics.get("fixed_steps_complete") is True
            and metrics.get("completed_steps") == 12_208
            and metrics.get("planned_steps") == 12_208
        ):
            continue
        payload["file"] = str(path)
        results[f"{name}_seed{seed}"] = payload
    return results


def legacy_early_stop_results() -> dict[str, dict[str, Any]]:
    """Read the pre-fixedsteps pilot only for a visibly invalid diagnostic table."""
    results: dict[str, dict[str, Any]] = {}
    for path in Path("outputs/benchmarks").glob("*.json"):
        payload = read_json(path)
        params = payload.get("params")
        if not isinstance(params, dict):
            continue
        suffix = str(params.get("run_suffix", ""))
        match = re.fullmatch(
            r"_paper_gate1_fineweb_100m_(?!fixedsteps_)(.+)_seed(\d+)", suffix
        )
        if match:
            results[f"{match.group(1)}_seed{match.group(2)}"] = payload
    return results


def phase2_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in Path("outputs/benchmarks").glob("*.json"):
        payload = read_json(path)
        params = payload.get("params")
        metrics = payload.get("metrics")
        if not isinstance(params, dict) or not isinstance(metrics, dict):
            continue
        suffix = str(params.get("run_suffix", ""))
        match = re.fullmatch(
            r"_paper_phase2_fineweb_onepass_fixedsteps_(.+)_seed(\d+)", suffix
        )
        if not match or not (
            metrics.get("fixed_steps_complete") is True
            and metrics.get("completed_steps") == metrics.get("planned_steps") == 12_208
        ):
            continue
        results[f"{match.group(1)}_seed{match.group(2)}"] = payload
    return results


def capacity_results() -> dict[str, dict[str, Any]]:
    """Read only fixed-step, paper-eligible K64/K1024 sweep checkpoints."""
    results: dict[str, dict[str, Any]] = {}
    for path in Path("outputs/benchmarks").glob("*.json"):
        payload = read_json(path)
        params = payload.get("params")
        metrics = payload.get("metrics")
        if not isinstance(params, dict) or not isinstance(metrics, dict):
            continue
        suffix = str(params.get("run_suffix", ""))
        match = re.fullmatch(
            r"_paper_capacity_k(64|1024)_fixedsteps_(.+)_seed42", suffix
        )
        if not match or not (
            metrics.get("fixed_steps_complete") is True
            and metrics.get("completed_steps") == metrics.get("planned_steps") == 12_208
        ):
            continue
        results[f"k{match.group(1)}_{match.group(2)}"] = payload
    return results


def external_result(benchmark: str, method: str, seed: int, corrected: bool = False) -> dict[str, Any]:
    if corrected:
        root = Path(f"outputs/xtreme_{benchmark}_matched_exact")
    else:
        root = Path(f"outputs/xtreme_{benchmark}_v1")
    return read_json(root / f"{method}_seed{seed}" / "metrics.json")


def external_triad_result(
    benchmark: str, variant: str, seed: int
) -> dict[str, Any]:
    if variant == "semantic_rq":
        root = Path(f"outputs/xtreme_{benchmark}_v1")
        method = "rq"
    elif variant == "arithmetic_matched":
        root = Path(f"outputs/xtreme_{benchmark}_matched_exact")
        method = variant
    elif variant == "rq_shuffled":
        root = Path(f"outputs/xtreme_{benchmark}_rq_shuffled_freqmatched")
        method = "rq"
    else:
        raise ValueError(f"unknown external variant: {variant}")
    return read_json(root / f"{method}_seed{seed}" / "metrics.json")


def macro(payload: dict[str, Any]) -> float | None:
    languages = payload.get("languages")
    value = languages.get("macro") if isinstance(languages, dict) else None
    return float(value) if isinstance(value, int | float) else None


def collect(output_dir: Path) -> dict[str, Any]:
    gate1_all = benchmark_results()
    legacy_gate1 = legacy_early_stop_results()
    gate1 = {
        name: gate1_all[f"{name}_seed42"]
        for name in GATE1
        if f"{name}_seed42" in gate1_all
    }
    legacy_pipeline = read_json(output_dir / "pipeline_state.json")
    fixed_pipeline = read_json(output_dir / "pipeline_state_fixedsteps.json")
    pipeline = fixed_pipeline or legacy_pipeline
    running = pipeline.get("running")
    if isinstance(running, dict):
        for entry in running.values():
            if isinstance(entry, dict) and isinstance(entry.get("log"), str):
                entry["progress"] = log_progress(Path(entry["log"]))
    shuffles = {
        str(seed): read_json(
            Path(
                f"rq_tables/fineweb_paper_M8K256_100k_"
                f"shuffled_freqmatched_seed{seed}/shuffle_manifest.json"
            )
        )
        for seed in GATE1_SEEDS
    }
    shuffle = shuffles["42"]
    address_diagnostics = read_json(output_dir / "address_diagnostics.json")
    slice_root = output_dir / "lm_slices"
    slice_manifests = {
        str(seed): read_json(slice_root / f"manifest_eval2000_seed{seed}.json")
        for seed in GATE1_SEEDS
    }
    access_manifests = {
        str(seed): read_json(slice_root / f"manifest_seed{seed}.json")
        for seed in GATE1_SEEDS
    }
    manifest_audit = read_json(slice_root / "manifest_audit.json")
    lm_slices = {
        f"{method}_seed{seed}": read_json(slice_root / f"{method}_seed{seed}.json")
        for seed in GATE1_SEEDS
        for method in ("base", *GATE1)
    }
    slice_comparison = read_json(slice_root / "comparison.json")
    intervention_root = output_dir / "head_intervention"
    head_interventions = {
        f"{name}_seed{seed}": read_json(
            intervention_root / f"{name}_seed{seed}.json"
        )
        for seed in GATE1_SEEDS
        for name in ("shared", "random_matched")
    }
    head_intervention_comparison = read_json(
        intervention_root / "comparison.json"
    )
    standard_root = output_dir / "standard_lm"
    standard_lm = {
        "base_seed42": read_json(standard_root / "base_seed42.json"),
        **{
            f"{method}_seed{seed}": read_json(
                standard_root / f"{method}_seed{seed}.json"
            )
            for seed in GATE1_SEEDS
            for method in STANDARD_METHODS
        },
    }
    phase2_root = output_dir / "phase2_onepass"
    phase2_decision = read_json(phase2_root / "gate_decision.json")
    phase2_pipeline = read_json(phase2_root / "pipeline_state.json")
    phase2_gate1 = phase2_results()
    phase2_slices_root = phase2_root / "lm_slices"
    phase2_slice_comparison = read_json(phase2_slices_root / "comparison.json")
    phase2_standard = {
        f"{method}_seed{seed}": read_json(
            phase2_root / "standard_lm" / f"{method}_seed{seed}.json"
        )
        for seed in GATE1_SEEDS
        for method in STANDARD_METHODS
    }
    capacity_root = output_dir / "capacity_sweep"
    capacity_pipeline = read_json(capacity_root / "pipeline_state.json")
    capacity_comparison = read_json(capacity_root / "comparison.json")
    capacity_gate1 = capacity_results()
    external: dict[str, Any] = {}
    for benchmark in ("xnli", "pawsx"):
        external[benchmark] = {
            "rq_seed42": macro(external_result(benchmark, "rq", 42)),
            "arithmetic_seed42": macro(external_result(benchmark, "arithmetic", 42)),
            "old_matched_seed42": macro(external_result(benchmark, "arithmetic_matched", 42)),
            "corrected_matched_seed42": macro(external_result(benchmark, "arithmetic_matched", 42, corrected=True)),
            "rq_seed43": macro(external_result(benchmark, "rq", 43)),
            "corrected_matched_seed43": macro(external_result(benchmark, "arithmetic_matched", 43, corrected=True)),
            "triad": {
                variant: {
                    str(seed): macro(external_triad_result(benchmark, variant, seed))
                    for seed in GATE1_SEEDS
                }
                for variant in STANDARD_METHODS
            },
        }
    external_root = output_dir / "external_matched"
    external_pipeline = read_json(external_root / "pipeline_state.json")
    external_access = {
        benchmark: read_json(external_root / benchmark / "access_counts.json")
        for benchmark in ("xnli", "pawsx")
    }
    external_shuffles = {
        f"{benchmark}_seed{seed}": read_json(
            Path(
                "rq_tables/wiki15_qwen3_06b_M8K256_500k_"
                f"{benchmark}_shuffled_freqmatched_seed{seed}/shuffle_manifest.json"
            )
        )
        for benchmark in ("xnli", "pawsx")
        for seed in GATE1_SEEDS
    }
    complete_gate1 = sum(name in gate1 for name in GATE1)
    complete_gate1_total = sum(
        f"{name}_seed{seed}" in gate1_all
        for seed in GATE1_SEEDS
        for name in GATE1
    )
    complete_slice_manifests = sum(
        payload.get("status") == "complete" for payload in slice_manifests.values()
    )
    complete_lm_slices = sum(
        payload.get("status") == "complete" for payload in lm_slices.values()
    )
    complete_standard_lm = sum(
        payload.get("status") == "complete"
        and payload.get("paper_eligible") is True
        for payload in standard_lm.values()
    )
    complete_head_interventions = sum(
        payload.get("status") == "complete"
        for payload in head_interventions.values()
    )
    queue_complete = pipeline.get("note") == "all queued experiments complete"
    phase2_status = phase2_decision.get("status")
    phase2_complete = (
        phase2_status in {"no_go", "failed"}
        or (
            phase2_status == "pass"
            and phase2_pipeline.get("note") == "all one-pass experiments complete"
            and len(phase2_gate1) == len(STANDARD_METHODS) * len(GATE1_SEEDS)
            and phase2_slice_comparison.get("status") == "complete"
            and all(
                payload.get("status") == "complete"
                and payload.get("paper_eligible") is True
                for payload in phase2_standard.values()
            )
        )
    )
    capacity_complete = (
        capacity_pipeline.get("note") == "all capacity experiments complete"
        and capacity_comparison.get("status") == "complete"
        and len(capacity_gate1) == 2 * len(STANDARD_METHODS)
    )
    external_complete = (
        external_pipeline.get("note") == "all external triads complete"
        and all(
            external_access[benchmark].get("status") == "complete"
            for benchmark in ("xnli", "pawsx")
        )
        and all(
            payload.get("status") == "complete"
            for benchmark in ("xnli", "pawsx")
            for variant in STANDARD_METHODS
            for seed in GATE1_SEEDS
            for payload in [external_triad_result(benchmark, variant, seed)]
        )
        and all(
            payload.get("status") == "complete"
            for payload in external_shuffles.values()
        )
    )
    return {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpus": gpu_snapshot(),
        "active_jobs": active_experiments(),
        "pipeline": pipeline,
        "legacy_pipeline": legacy_pipeline,
        "shuffle": shuffle,
        "shuffles": shuffles,
        "address_diagnostics": address_diagnostics,
        "slice_manifests": slice_manifests,
        "access_manifests": access_manifests,
        "manifest_audit": manifest_audit,
        "lm_slices": lm_slices,
        "slice_comparison": slice_comparison,
        "head_interventions": head_interventions,
        "head_intervention_comparison": head_intervention_comparison,
        "standard_lm": standard_lm,
        "phase2_decision": phase2_decision,
        "phase2_pipeline": phase2_pipeline,
        "phase2_gate1": phase2_gate1,
        "phase2_slice_comparison": phase2_slice_comparison,
        "phase2_standard": phase2_standard,
        "capacity_pipeline": capacity_pipeline,
        "capacity_comparison": capacity_comparison,
        "capacity_gate1": capacity_gate1,
        "gate1": gate1,
        "gate1_all": gate1_all,
        "legacy_gate1": legacy_gate1,
        "external": external,
        "external_pipeline": external_pipeline,
        "external_access": external_access,
        "external_shuffles": external_shuffles,
        "complete_gate1": complete_gate1,
        "complete_gate1_total": complete_gate1_total,
        "complete_slice_manifests": complete_slice_manifests,
        "complete_lm_slices": complete_lm_slices,
        "complete_standard_lm": complete_standard_lm,
        "complete_head_interventions": complete_head_interventions,
        "phase2_complete": phase2_complete,
        "capacity_complete": capacity_complete,
        "external_complete": external_complete,
        "all_complete": (
            complete_gate1_total == len(GATE1) * len(GATE1_SEEDS)
            and complete_slice_manifests == len(GATE1_SEEDS)
            and manifest_audit.get("all_train_access_counts_identical") is True
            and complete_lm_slices == (len(GATE1) + 1) * len(GATE1_SEEDS)
            and slice_comparison.get("status") == "complete"
            and complete_standard_lm == 10
            and complete_head_interventions == 6
            and head_intervention_comparison.get("status") == "complete"
            and phase2_complete
            and capacity_complete
            and external_complete
            and queue_complete
            and address_diagnostics.get("status") == "complete"
        ),
    }


def fmt(value: object, digits: int = 4) -> str:
    return "—" if not isinstance(value, int | float) else f"{value:.{digits}f}"


def standard_metric(
    payload: dict[str, Any], task: str, metric_names: tuple[str, ...]
) -> float | None:
    values = payload.get("results", {}).get(task, {})
    if not isinstance(values, dict):
        return None
    for name in metric_names:
        value = values.get(name)
        if isinstance(value, int | float):
            return float(value)
    return None


def task_key(name: str, seed: int) -> str:
    return f"gate1_{name}" if seed == 42 else f"gate1_{name}_seed{seed}"


def active_gate_job(
    name: str, snapshot: dict[str, Any], seed: int
) -> dict[str, Any] | None:
    suffix = f"_paper_gate1_fineweb_100m_fixedsteps_{name}_seed{seed}"
    for job in snapshot.get("active_jobs", []):
        if isinstance(job, dict) and job.get("run_suffix") == suffix:
            return job
    return None


def gate_progress(
    name: str, snapshot: dict[str, Any], seed: int
) -> dict[str, Any]:
    running = snapshot.get("pipeline", {}).get("running", {})
    entry = running.get(task_key(name, seed)) if isinstance(running, dict) else None
    if is_fixedstep_entry(entry) and "return_code" not in entry:
        progress = entry.get("progress")
        if isinstance(progress, dict):
            return progress
    job = active_gate_job(name, snapshot, seed)
    progress = job.get("progress") if isinstance(job, dict) else None
    return progress if isinstance(progress, dict) else {}


def is_fixedstep_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    command = entry.get("command")
    return isinstance(command, list) and any(
        "_paper_gate1_fineweb_100m_fixedsteps_" in str(value)
        for value in command
    )


def status_for(name: str, snapshot: dict[str, Any], seed: int = 42) -> str:
    if f"{name}_seed{seed}" in snapshot["gate1_all"]:
        return '<span class="pill done">完成</span>'
    running = snapshot["pipeline"].get("running", {})
    entry = running.get(task_key(name, seed)) if isinstance(running, dict) else None
    if is_fixedstep_entry(entry) and "return_code" not in entry:
        progress = entry.get("progress", {})
        suffix = ""
        if isinstance(progress, dict) and isinstance(progress.get("step"), int):
            suffix = f' · {progress["percent"]}% ({progress["step"]}/{progress["total_steps"]})'
        return f'<span class="pill run">GPU {entry.get("gpu", "?")} 训练中{suffix}</span>'
    if is_fixedstep_entry(entry) and entry.get("return_code") not in (None, 0):
        return '<span class="pill fail">失败，等待诊断</span>'
    job = active_gate_job(name, snapshot, seed)
    if job is not None:
        progress = job.get("progress", {})
        suffix = ""
        if isinstance(progress, dict) and isinstance(progress.get("step"), int):
            suffix = (
                f' · {progress["percent"]}% '
                f'({progress["step"]}/{progress["total_steps"]})'
            )
        return (
            f'<span class="pill run">GPU {job.get("gpu", "?")} '
            f'训练中（外部接管）{suffix}</span>'
        )
    return '<span class="pill wait">等待依赖/空卡</span>'


def conclusions(snapshot: dict[str, Any]) -> list[str]:
    gate1 = snapshot["gate1"]
    items: list[str] = []
    if not snapshot["shuffle"]:
        items.append("RQ-Shuffled 尚未生成，因此目前不能把任何下游领先归因于语义地址结构。")
    else:
        preserved = all(
            value.get("level_histograms_preserved") is True
            and value.get("access_weighted_histograms_preserved") is True
            for value in snapshot["shuffle"].get("ngram_sizes", {}).values()
            if isinstance(value, dict)
        )
        items.append(
            "Frequency-matched RQ-Shuffled 已生成；未加权行数与 LM-train "
            f"access-weighted 桶负载均逐层保持：{'是' if preserved else '待核验'}。"
        )
    diagnostics = snapshot.get("address_diagnostics", {})
    if diagnostics.get("status") in {"complete", "partial"}:
        summaries = []
        for order, values in diagnostics.get("orders", {}).items():
            rq_corr = values.get("spearman_semantic_vs_rq_overlap")
            shuffled_corr = values.get("spearman_semantic_vs_shuffled_overlap")
            quadrant = values.get("quadrants", {}).get("high_semantic_low_lexical", {})
            rq_overlap = quadrant.get("rq_code_overlap")
            shuffled_overlap = quadrant.get("shuffled_code_overlap")
            if all(isinstance(value, int | float) for value in (rq_corr, shuffled_corr, rq_overlap, shuffled_overlap)):
                summaries.append(
                    f"{order}-gram ρ={rq_corr:.3f} vs shuffled {shuffled_corr:.3f}，"
                    f"低词面高语义 overlap={rq_overlap:.3f} vs {shuffled_overlap:.3f}"
                )
        coverage_status = diagnostics.get("coverage", {}).get("status")
        suffix = "held-out coverage 已完成" if coverage_status == "complete" else "held-out coverage 因数据源限流待补"
        items.append("Gate 0 地址结构已计算：" + "；".join(summaries) + f"；{suffix}。")
    else:
        items.append("Gate 0 的语义/词面四象限与 held-out coverage 正在计算，当前仅有桶频率不变量。")
    triad = ("arithmetic_matched", "rq_shuffled", "semantic_rq")
    all_results = snapshot["gate1_all"]
    if all(f"{name}_seed{seed}" in all_results for seed in GATE1_SEEDS for name in triad):
        per_seed = {
            seed: {
                name: all_results[f"{name}_seed{seed}"].get("metrics", {}).get("eval_loss")
                for name in triad
            }
            for seed in GATE1_SEEDS
        }
        means = {
            name: sum(float(per_seed[seed][name]) for seed in GATE1_SEEDS) / len(GATE1_SEEDS)
            for name in triad
        }
        wins = sum(
            per_seed[seed]["semantic_rq"] < per_seed[seed]["rq_shuffled"]
            and per_seed[seed]["semantic_rq"] < per_seed[seed]["arithmetic_matched"]
            for seed in GATE1_SEEDS
        )
        items.append(
            f"Gate 1 三 seed 已完成：Semantic-RQ 平均 loss={means['semantic_rq']:.4f}，"
            f"Arithmetic-fixed={means['arithmetic_matched']:.4f}，RQ-Shuffled={means['rq_shuffled']:.4f}；"
            f"Semantic-RQ 同时胜过两对照的 seed 数为 {wins}/3。"
        )
    elif all(name in gate1 for name in triad):
        losses = {name: gate1[name].get("metrics", {}).get("eval_loss") for name in GATE1 if name in gate1}
        semantic = losses.get("semantic_rq")
        shuffled = losses.get("rq_shuffled")
        arithmetic = losses.get("arithmetic_matched")
        if all(isinstance(value, int | float) for value in (semantic, shuffled, arithmetic)):
            if semantic < shuffled and semantic < arithmetic:
                items.append("Gate 1 单 seed 支持结构化共享假设：Semantic-RQ 的 held-out loss 同时低于 matched arithmetic 与 RQ-Shuffled；仍需多 seed 和切片证据。")
            else:
                items.append("Gate 1 单 seed 尚不支持核心假设：Semantic-RQ 未同时击败两个决定性对照；不得扩写正向结论。")
    else:
        items.append("Gate 1 尚未形成三角对照，当前状态只说明实验在运行，不构成论文结论。")
    slices = snapshot.get("lm_slices", {})
    slice_triad = {
        name: slices.get(f"{name}_seed42", {})
        for name in ("arithmetic_matched", "rq_shuffled", "semantic_rq")
    }
    if all(payload.get("status") == "complete" for payload in slice_triad.values()):
        values = {
            name: payload.get("metrics", {}).get("3", {}).get("semantic_neighbor", {}).get("nll")
            for name, payload in slice_triad.items()
        }
        if all(isinstance(value, int | float) for value in values.values()):
            verdict = (
                "同时低于两个对照"
                if values["semantic_rq"] < values["arithmetic_matched"]
                and values["semantic_rq"] < values["rq_shuffled"]
                else "未同时低于两个对照"
            )
            items.append(
                "Seed42 的 3-gram semantic-neighbor NLL："
                f"Semantic-RQ={values['semantic_rq']:.4f}，"
                f"Arithmetic-fixed={values['arithmetic_matched']:.4f}，"
                f"RQ-Shuffled={values['rq_shuffled']:.4f}；Semantic-RQ {verdict}。"
            )
    else:
        items.append("逐 token LM 切片评测已进入自动队列；完成前 overall loss 不能单独证明语义迁移。")
    manifest_audit = snapshot.get("manifest_audit", {})
    if manifest_audit.get("all_train_access_counts_identical") is True:
        shared_counts = [
            snapshot.get("slice_manifests", {}).get(str(seed), {})
            .get("orders", {}).get("3", {})
            .get("semantic_neighbor_shared_code")
            for seed in GATE1_SEEDS
        ]
        items.append(
            "2,000-row 正式切片与 200-row frequency-control manifest 的 train-access "
            "数组在 3 seeds × 2 orders 上逐元素一致；3-gram semantic/shared-code "
            f"token 数为 {shared_counts}。"
        )
    comparison = snapshot.get("slice_comparison", {})
    if comparison.get("status") == "complete":
        primary = comparison.get("aggregate", {}).get("rq_shuffled", {}).get(
            "3gram_semantic_neighbor_shared_code", {}
        )
        delta = primary.get("delta_nll")
        ci = primary.get("ci95")
        if isinstance(delta, int | float) and isinstance(ci, list) and len(ci) == 2:
            significant = isinstance(ci[1], int | float) and ci[1] < 0
            items.append(
                "预注册主切片（3-gram semantic-neighbor/shared-code）的 paired cluster bootstrap："
                f"Semantic-RQ − RQ-Shuffled ΔNLL={delta:.4f}，"
                f"95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]；"
                + ("支持显著改善。" if significant else "尚不足以宣称显著改善。")
            )
        interaction_fragments = []
        for control, display in (
            ("arithmetic_matched", "Arithmetic-fixed"),
            ("rq_shuffled", "RQ-Shuffled"),
        ):
            interaction = comparison.get("interactions", {}).get(control, {}).get(
                "3gram_shared_minus_no_shared", {}
            )
            value = interaction.get("delta_nll_interaction")
            interval = interaction.get("ci95")
            if (
                isinstance(value, int | float)
                and isinstance(interval, list)
                and len(interval) == 2
            ):
                interaction_fragments.append(
                    f"vs {display}: {value:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"
                )
        if interaction_fragments:
            items.append(
                "3-gram shared-code − no-shared-code 机制交互 ΔNLL："
                + "；".join(interaction_fragments)
                + "；仅当两个 CI 上界均低于 0，才证明收益集中于真正共享 rows。"
            )
    standard = snapshot.get("standard_lm", {})
    if all(
        standard.get(f"{method}_seed{seed}", {}).get("status") == "complete"
        for method in STANDARD_METHODS
        for seed in GATE1_SEEDS
    ):
        means: dict[str, float] = {}
        for method in STANDARD_METHODS:
            values = [
                standard_metric(
                    standard[f"{method}_seed{seed}"],
                    "paloma_wikitext_103",
                    ("word_perplexity,none", "word_perplexity"),
                )
                for seed in GATE1_SEEDS
            ]
            if all(value is not None for value in values):
                means[method] = sum(float(value) for value in values) / len(values)
        if len(means) == len(STANDARD_METHODS):
            items.append(
                "标准 Paloma WikiText-103 三 seed word PPL："
                f"Semantic-RQ={means['semantic_rq']:.3f}，"
                f"Arithmetic-fixed={means['arithmetic_matched']:.3f}，"
                f"RQ-Shuffled={means['rq_shuffled']:.3f}。"
            )
    else:
        items.append("标准 LM 评测已排队：Paloma WikiText-103、Paloma C4 与 LAMBADA；完成前不把 200-row FineWeb slice 外推为标准 LM 结论。")
    intervention = snapshot.get("head_intervention_comparison", {})
    if intervention.get("status") == "complete":
        primary = intervention.get("aggregate", {}).get(
            "shared_minus_random", {}
        ).get("3gram_semantic_neighbor_shared_code", {})
        delta = primary.get("delta_nll")
        ci = primary.get("ci95")
        if isinstance(delta, int | float) and isinstance(ci, list) and len(ci) == 2:
            supported = isinstance(ci[0], int | float) and ci[0] > 0
            items.append(
                "Shared-head 因果干预：共享 head mask − 等数量随机 head mask "
                f"ΔNLL={delta:.4f}，95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]；"
                + ("共享 rows 的特异作用得到支持。" if supported else "尚不足以证明共享 rows 的特异作用。")
            )
    else:
        items.append("Shared-head masking 与 equal-count random masking 已进入队列；该干预完成前，shared-row exposure 仍只是相关证据。")
    phase2 = snapshot.get("phase2_decision", {})
    if phase2.get("status") == "waiting":
        items.append("One-pass Phase 2 正由预注册 Gate 等待：Gate-1 bootstrap 完成前不会提前扩规模。")
    elif phase2.get("status") == "pass":
        items.append(
            "Gate-1 已通过主切片、shared/no-shared interaction、overall safety "
            "与 2/3-seed consistency 条件；390,656-row、单 epoch one-pass replication 已启动。"
        )
    elif phase2.get("status") in {"no_go", "failed"}:
        failed = [
            name
            for name in (
                "primary_pass",
                "interaction_pass",
                "overall_safe",
                "seed_consistent",
            )
            if phase2.get(name) is False
        ]
        items.append(
            "Gate-1 未通过预注册条件"
            + (f"（失败项：{', '.join(failed)}）" if failed else "")
            + "；one-pass 扩规模已自动停止，当前结果不得包装成正向方法结论。"
        )
    capacity = snapshot.get("capacity_comparison", {})
    if capacity.get("status") == "complete":
        fragments = []
        for value in (64, 256, 1024):
            result = (
                capacity.get("results", {})
                .get(str(value), {})
                .get("rq_shuffled", {})
                .get("3gram_semantic_neighbor_shared_code", {})
            )
            delta = result.get("delta_nll") if isinstance(result, dict) else None
            ci = result.get("ci95") if isinstance(result, dict) else None
            if isinstance(delta, int | float) and isinstance(ci, list) and len(ci) == 2:
                fragments.append(
                    f"K={value}: ΔNLL={delta:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
                )
        items.append(
            "有限容量曲线（Semantic-RQ − RQ-Shuffled，3g semantic/shared）："
            + "；".join(fragments)
            + "。该曲线首轮为单 seed，关键端点仍需复现。"
        )
    else:
        state = snapshot.get("capacity_pipeline", {}).get("note")
        items.append(
            "K={64,256,1024} finite-memory capacity sweep 已串行排队；"
            + (f"当前状态：{state}。" if state else "将在 one-pass 队列结束后自动启动。")
        )
    external = snapshot.get("external", {})
    if snapshot.get("external_complete") is True:
        for benchmark in ("xnli", "pawsx"):
            means = {}
            for variant in STANDARD_METHODS:
                values = list(
                    external.get(benchmark, {}).get("triad", {}).get(variant, {}).values()
                )
                if all(isinstance(value, int | float) for value in values):
                    means[variant] = statistics.mean(float(value) for value in values)
            if len(means) == len(STANDARD_METHODS):
                items.append(
                    f"{benchmark.upper()} 三 seed Macro accuracy："
                    f"Semantic-RQ={means['semantic_rq']:.4f}，"
                    f"Arithmetic-fixed={means['arithmetic_matched']:.4f}，"
                    f"RQ-Shuffled={means['rq_shuffled']:.4f}。"
                )
    else:
        state = snapshot.get("external_pipeline", {}).get("note")
        items.append(
            "XNLI/PAWS-X 的 benchmark-specific frequency-matched 三角对照已排队；"
            + (f"当前状态：{state}。" if state else "将在 capacity sweep 后自动启动。")
        )
    items.append("旧 Arithmetic (matched) 实际仅约 32 行/头；中间 v2 又因原实现使用递增质数桶而非固定 256 行/头。两者均不进入主表，主表只读取 arithmetic_fixed 的 exact 目录。")
    items.append("早期 Gate-1 run 被 EarlyStoppingCallback 在约 1,600/12,208 step 截停，只覆盖约 13M tokens；这些结果仅作诊断。主表只读取禁用 early stopping、严格跑满 12,208 steps 的 fixedsteps run。")
    items.append("正式 Gate-1 的 100,007,936 指 processed token slots：48,832 条序列重复 8 个完整 epoch，不是 100M 个独立 token；它是机制 stress test，不能替代后续 one-pass LM replication。")
    return items


def render(snapshot: dict[str, Any]) -> str:
    gpu_cards = "".join(
        f'<div class="card"><b>GPU {gpu["index"]}</b><span>{gpu["used_mib"]}/{gpu["total_mib"]} MiB</span><strong>{gpu["util"]}%</strong></div>'
        for gpu in snapshot["gpus"]
    )
    labels = {
        "base": "Base / No memory",
        "arithmetic_matched": "Arithmetic-matched",
        "rq_shuffled": "RQ-Shuffled",
        "semantic_rq": "Semantic-RQ",
        "mixed_4_4": "Mixed 4+4（消融）",
    }
    gate_rows = []
    for name in GATE1:
        payload = snapshot["gate1"].get(name, {})
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        progress = gate_progress(name, snapshot, 42)
        latest_loss = progress.get("loss") if isinstance(progress, dict) else None
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None and isinstance(progress, dict):
            eval_loss = progress.get("eval_loss")
        gate_rows.append(
            f"<tr><th>{labels[name]}</th><td>{status_for(name, snapshot)}</td>"
            f"<td>{fmt(latest_loss)}</td><td>{fmt(eval_loss)}</td><td>{fmt(metrics.get('peak_memory_gb'), 2)}</td>"
            f"<td>{fmt(metrics.get('avg_time_per_step'), 3)}</td>"
            f"<td>{fmt(metrics.get('causal_target_token_presentations'), 0)}</td></tr>"
        )
    replication_rows = []
    for name in GATE1:
        cells = []
        for seed in GATE1_SEEDS:
            payload = snapshot["gate1_all"].get(f"{name}_seed{seed}", {})
            metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
            eval_loss = metrics.get("eval_loss") if isinstance(metrics, dict) else None
            progress = gate_progress(name, snapshot, seed)
            if eval_loss is None:
                eval_loss = progress.get("eval_loss")
            completed_steps = metrics.get("completed_steps")
            if completed_steps is None:
                completed_steps = progress.get("step", "—")
            cells.append(
                f"<td>{status_for(name, snapshot, seed)}<br><span class='lead'>eval {fmt(eval_loss)} · "
                f"steps {completed_steps}/12208</span></td>"
            )
        replication_rows.append(f"<tr><th>{labels[name]}</th>{''.join(cells)}</tr>")
    legacy_rows = []
    running = snapshot.get("legacy_pipeline", {}).get("running", {})
    for name in GATE1:
        payload = snapshot.get("legacy_gate1", {}).get(f"{name}_seed42", {})
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        history = metrics.get("log_history", []) if isinstance(metrics, dict) else []
        completed = max(
            (int(item.get("step", 0)) for item in history if isinstance(item, dict)),
            default=0,
        )
        entry = running.get(f"gate1_{name}", {}) if isinstance(running, dict) else {}
        progress = entry.get("progress", {}) if isinstance(entry, dict) else {}
        if not is_fixedstep_entry(entry) and isinstance(progress, dict):
            completed = max(completed, int(progress.get("step", 0)))
        state = "完成（无效）" if metrics else ("运行中（无效）" if completed else "等待")
        legacy_rows.append(
            f"<tr><th>{labels[name]}</th><td>{state}</td><td>{completed or '—'}/12208</td>"
            f"<td>{fmt(metrics.get('eval_loss'))}</td></tr>"
        )
    diagnostic_rows = []
    diagnostics = snapshot.get("address_diagnostics", {})
    for order in ("2", "3"):
        values = diagnostics.get("orders", {}).get(order, {}) if isinstance(diagnostics, dict) else {}
        quadrant = values.get("quadrants", {}).get("high_semantic_low_lexical", {}) if isinstance(values, dict) else {}
        coverage = diagnostics.get("coverage", {}).get("per_order", {}).get(order, {}) if isinstance(diagnostics, dict) else {}
        diagnostic_rows.append(
            f"<tr><th>{order}-gram</th>"
            f"<td>{fmt(values.get('spearman_semantic_vs_rq_overlap'), 3)}</td>"
            f"<td>{fmt(values.get('spearman_semantic_vs_shuffled_overlap'), 3)}</td>"
            f"<td>{fmt(quadrant.get('rq_code_overlap'), 3)}</td>"
            f"<td>{fmt(quadrant.get('shuffled_code_overlap'), 3)}</td>"
            f"<td>{fmt(coverage.get('coverage'), 3)}</td>"
            f"<td>{quadrant.get('pairs', '—')}</td></tr>"
        )
    shuffle_rows = []
    for seed in GATE1_SEEDS:
        payload = snapshot.get("shuffles", {}).get(str(seed), {})
        sizes = payload.get("ngram_sizes", {}) if isinstance(payload, dict) else {}
        for order in ("2", "3"):
            values = sizes.get(order, {}) if isinstance(sizes, dict) else {}
            shuffle_rows.append(
                f"<tr><th>Seed {seed} · {order}-gram</th>"
                f"<td>{fmt(values.get('moved_fraction'), 4)}</td>"
                f"<td>{fmt(values.get('accessed_rows_moved_fraction'), 4)}</td>"
                f"<td>{'是' if values.get('level_histograms_preserved') is True else '—'}</td>"
                f"<td>{'是' if values.get('access_weighted_histograms_preserved') is True else '—'}</td>"
                f"<td>{values.get('singleton_frequency_rows', '—')}</td></tr>"
            )
    manifest_rows = []
    for seed in GATE1_SEEDS:
        payload = snapshot.get("slice_manifests", {}).get(str(seed), {})
        orders = payload.get("orders", {}) if isinstance(payload, dict) else {}
        cells = []
        for order in ("2", "3"):
            values = orders.get(order, {}) if isinstance(orders, dict) else {}
            counts = values.get("counts", {}) if isinstance(values, dict) else {}
            cells.extend(
                (
                    str(counts.get("exact_seen", "—")),
                    str(counts.get("semantic_neighbor", "—")),
                    str(counts.get("covered_no_neighbor", "—")),
                    str(counts.get("address_oov", "—")),
                )
            )
        status = payload.get("status") if isinstance(payload, dict) else None
        manifest_rows.append(
            f"<tr><th>Seed {seed}</th><td>{'完成' if status == 'complete' else '等待/运行中'}</td>"
            + "".join(f"<td>{html.escape(value)}</td>" for value in cells)
            + "</tr>"
        )
    slice_rows = []
    for method in ("base", *GATE1):
        cells = []
        for seed in GATE1_SEEDS:
            payload = snapshot.get("lm_slices", {}).get(f"{method}_seed{seed}", {})
            metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
            overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
            order2 = metrics.get("2", {}) if isinstance(metrics, dict) else {}
            order3 = metrics.get("3", {}) if isinstance(metrics, dict) else {}
            if payload.get("status") == "complete":
                cells.append(
                    "<td>"
                    f"overall {fmt(overall.get('nll'))}<br>"
                    f"2g-sem {fmt(order2.get('semantic_neighbor', {}).get('nll'))}<br>"
                    f"3g-sem {fmt(order3.get('semantic_neighbor', {}).get('nll'))}"
                    "</td>"
                )
            else:
                cells.append('<td><span class="pill wait">等待/运行中</span></td>')
        slice_rows.append(f"<tr><th>{labels[method]}</th>{''.join(cells)}</tr>")
    comparison_rows = []
    comparison = snapshot.get("slice_comparison", {})
    aggregate = comparison.get("aggregate", {}) if isinstance(comparison, dict) else {}
    comparison_labels = {
        "arithmetic_matched": "vs Arithmetic-fixed",
        "rq_shuffled": "vs RQ-Shuffled",
    }
    for control, control_label in comparison_labels.items():
        values = aggregate.get(control, {}) if isinstance(aggregate, dict) else {}
        for slice_name, display in (
            ("overall", "Overall"),
            ("2gram_semantic_neighbor", "2g semantic-neighbor"),
            ("2gram_semantic_neighbor_shared_code", "2g semantic + shared code"),
            ("2gram_semantic_neighbor_no_shared_code", "2g semantic + no shared code"),
            ("2gram_semantic_neighbor_low_lexical", "2g semantic / low lexical"),
            ("2gram_semantic_neighbor_low_lexical_shared_code", "2g semantic / low lexical / shared"),
            ("3gram_semantic_neighbor", "3g semantic-neighbor"),
            ("3gram_semantic_neighbor_shared_code", "3g semantic + shared code"),
            ("3gram_semantic_neighbor_no_shared_code", "3g semantic + no shared code"),
            ("3gram_semantic_neighbor_low_lexical", "3g semantic / low lexical"),
            ("3gram_semantic_neighbor_low_lexical_shared_code", "3g semantic / low lexical / shared"),
            ("2gram_covered_no_neighbor_high_lexical", "2g covered/no-neighbor / high lexical"),
            ("3gram_covered_no_neighbor_high_lexical", "3g covered/no-neighbor / high lexical"),
            ("3gram_address_oov", "3g address-OOV"),
        ):
            metric_value = values.get(slice_name, {}) if isinstance(values, dict) else {}
            ci = metric_value.get("ci95") if isinstance(metric_value, dict) else None
            ci_text = (
                f"[{fmt(ci[0])}, {fmt(ci[1])}]"
                if isinstance(ci, list) and len(ci) == 2
                else "—"
            )
            comparison_rows.append(
                f"<tr><th>{control_label}</th><td>{display}</td>"
                f"<td>{fmt(metric_value.get('delta_nll'))}</td><td>{ci_text}</td>"
                f"<td>{metric_value.get('tokens', '—')}</td></tr>"
            )
        interaction = (
            comparison.get("interactions", {})
            .get(control, {})
            .get("3gram_shared_minus_no_shared", {})
        )
        interaction_ci = interaction.get("ci95") if isinstance(interaction, dict) else None
        interaction_ci_text = (
            f"[{fmt(interaction_ci[0])}, {fmt(interaction_ci[1])}]"
            if isinstance(interaction_ci, list) and len(interaction_ci) == 2
            else "—"
        )
        comparison_rows.append(
            f"<tr><th>{control_label}</th><td>3g shared − no-shared interaction</td>"
            f"<td>{fmt(interaction.get('delta_nll_interaction'))}</td>"
            f"<td>{interaction_ci_text}</td>"
            f"<td>{interaction.get('left_tokens', '—')} / {interaction.get('right_tokens', '—')}</td></tr>"
        )
    standard_rows = []
    for method in ("base", *STANDARD_METHODS):
        seeds = (42,) if method == "base" else GATE1_SEEDS
        for seed in seeds:
            payload = snapshot.get("standard_lm", {}).get(
                f"{method}_seed{seed}", {}
            )
            wikitext = standard_metric(
                payload,
                "paloma_wikitext_103",
                ("word_perplexity,none", "word_perplexity"),
            )
            c4 = standard_metric(
                payload,
                "paloma_c4_en",
                ("word_perplexity,none", "word_perplexity"),
            )
            lambada_acc = standard_metric(
                payload,
                "lambada_openai",
                ("acc,none", "acc"),
            )
            lambada_ppl = standard_metric(
                payload,
                "lambada_openai",
                ("perplexity,none", "perplexity"),
            )
            state = "完成" if payload.get("status") == "complete" else "等待/运行中"
            standard_rows.append(
                f"<tr><th>{labels[method]}</th><td>{seed}</td><td>{state}</td>"
                f"<td>{fmt(wikitext, 3)}</td><td>{fmt(c4, 3)}</td>"
                f"<td>{fmt(lambada_acc, 4)}</td><td>{fmt(lambada_ppl, 3)}</td></tr>"
            )
    intervention_rows = []
    intervention = snapshot.get("head_intervention_comparison", {})
    aggregate_intervention = (
        intervention.get("aggregate", {}) if isinstance(intervention, dict) else {}
    )
    intervention_labels = {
        "shared_minus_none": "Shared mask − No mask",
        "random_minus_none": "Random-matched mask − No mask",
        "shared_minus_random": "Shared mask − Random-matched mask",
    }
    for contrast, display in intervention_labels.items():
        values = aggregate_intervention.get(contrast, {})
        for slice_name, slice_display in (
            ("2gram_semantic_neighbor_shared_code", "2g semantic/shared"),
            ("3gram_semantic_neighbor_shared_code", "3g semantic/shared"),
            ("semantic_neighbor_shared_code_union", "2g∪3g semantic/shared"),
        ):
            metric_value = values.get(slice_name, {}) if isinstance(values, dict) else {}
            ci = metric_value.get("ci95") if isinstance(metric_value, dict) else None
            ci_text = (
                f"[{fmt(ci[0])}, {fmt(ci[1])}]"
                if isinstance(ci, list) and len(ci) == 2
                else "—"
            )
            intervention_rows.append(
                f"<tr><th>{display}</th><td>{slice_display}</td>"
                f"<td>{fmt(metric_value.get('delta_nll'))}</td><td>{ci_text}</td>"
                f"<td>{metric_value.get('tokens', '—')}</td></tr>"
            )
    phase2_rows = []
    phase2_results_snapshot = snapshot.get("phase2_gate1", {})
    for method in STANDARD_METHODS:
        cells = []
        for seed in GATE1_SEEDS:
            payload = phase2_results_snapshot.get(f"{method}_seed{seed}", {})
            metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
            cells.append(
                f"<td>{'完成' if metrics else '等待/运行中'}<br>"
                f"eval {fmt(metrics.get('eval_loss'))} · "
                f"steps {metrics.get('completed_steps', '—')}/12208</td>"
            )
        phase2_rows.append(f"<tr><th>{labels[method]}</th>{''.join(cells)}</tr>")
    phase2_decision = snapshot.get("phase2_decision", {})
    phase2_status = html.escape(str(phase2_decision.get("status", "waiting")))
    capacity_rows = []
    capacity_runs = snapshot.get("capacity_gate1", {})
    for capacity in (64, 256, 1024):
        for method in STANDARD_METHODS:
            payload = (
                snapshot.get("gate1_all", {}).get(f"{method}_seed42", {})
                if capacity == 256
                else capacity_runs.get(f"k{capacity}_{method}", {})
            )
            metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
            capacity_rows.append(
                f"<tr><th>K={capacity}</th><td>{labels[method]}</td>"
                f"<td>{'完成' if metrics else '等待/运行中'}</td>"
                f"<td>{fmt(metrics.get('eval_loss'))}</td>"
                f"<td>{fmt(metrics.get('peak_memory_gb'), 2)}</td></tr>"
            )
    capacity_comparison_rows = []
    capacity_comparison = snapshot.get("capacity_comparison", {})
    capacity_values = (
        capacity_comparison.get("results", {})
        if isinstance(capacity_comparison, dict)
        else {}
    )
    for capacity in (64, 256, 1024):
        for control, display in (
            ("arithmetic_matched", "vs Arithmetic-fixed"),
            ("rq_shuffled", "vs RQ-Shuffled"),
        ):
            values = capacity_values.get(str(capacity), {}).get(control, {})
            overall = values.get("overall", {}) if isinstance(values, dict) else {}
            shared = (
                values.get("3gram_semantic_neighbor_shared_code", {})
                if isinstance(values, dict)
                else {}
            )
            overall_ci = overall.get("ci95") if isinstance(overall, dict) else None
            shared_ci = shared.get("ci95") if isinstance(shared, dict) else None
            capacity_comparison_rows.append(
                f"<tr><th>K={capacity}</th><td>{display}</td>"
                f"<td>{fmt(overall.get('delta_nll'))}</td>"
                f"<td>{'[' + fmt(overall_ci[0]) + ', ' + fmt(overall_ci[1]) + ']' if isinstance(overall_ci, list) and len(overall_ci) == 2 else '—'}</td>"
                f"<td>{fmt(shared.get('delta_nll'))}</td>"
                f"<td>{'[' + fmt(shared_ci[0]) + ', ' + fmt(shared_ci[1]) + ']' if isinstance(shared_ci, list) and len(shared_ci) == 2 else '—'}</td></tr>"
            )
    ext_rows = []
    for benchmark, row in snapshot["external"].items():
        ext_rows.append(
            f"<tr><th>{benchmark.upper()}</th><td>{fmt(row['arithmetic_seed42'])}</td>"
            f"<td>{fmt(row['rq_seed42'])}</td><td>{fmt(row['rq_seed43'])}</td>"
            f"<td class='invalid'>{fmt(row['old_matched_seed42'])}</td>"
            f"<td>{fmt(row['corrected_matched_seed42'])}</td><td>{fmt(row['corrected_matched_seed43'])}</td></tr>"
        )
    external_triad_rows = []
    external_variant_labels = {
        "arithmetic_matched": "Arithmetic-fixed",
        "rq_shuffled": "RQ-Shuffled (frequency-matched)",
        "semantic_rq": "Semantic-RQ",
    }
    for benchmark in ("xnli", "pawsx"):
        triad = snapshot.get("external", {}).get(benchmark, {}).get("triad", {})
        for variant in STANDARD_METHODS:
            per_seed = triad.get(variant, {}) if isinstance(triad, dict) else {}
            values = [per_seed.get(str(seed)) for seed in GATE1_SEEDS]
            complete_values = [
                float(value) for value in values if isinstance(value, int | float)
            ]
            mean = statistics.mean(complete_values) if len(complete_values) == 3 else None
            std = statistics.stdev(complete_values) if len(complete_values) == 3 else None
            external_triad_rows.append(
                f"<tr><th>{benchmark.upper()}</th>"
                f"<td>{external_variant_labels[variant]}</td>"
                + "".join(f"<td>{fmt(value)}</td>" for value in values)
                + f"<td>{fmt(mean)}</td><td>{fmt(std)}</td></tr>"
            )
    external_access_rows = []
    for benchmark in ("xnli", "pawsx"):
        access = snapshot.get("external_access", {}).get(benchmark, {})
        orders = access.get("orders", {}) if isinstance(access, dict) else {}
        shuffles = snapshot.get("external_shuffles", {})
        preserved = all(
            all(
                value.get("access_weighted_histograms_preserved") is True
                for value in shuffles.get(f"{benchmark}_seed{seed}", {})
                .get("ngram_sizes", {}).values()
                if isinstance(value, dict)
            )
            and shuffles.get(f"{benchmark}_seed{seed}", {}).get("status") == "complete"
            for seed in GATE1_SEEDS
        )
        external_access_rows.append(
            f"<tr><th>{benchmark.upper()}</th>"
            f"<td>{'完成' if access.get('status') == 'complete' else '等待/运行中'}</td>"
            f"<td>{orders.get('2', {}).get('total_accesses', '—')}</td>"
            f"<td>{orders.get('3', {}).get('total_accesses', '—')}</td>"
            f"<td>{'是' if preserved else '等待'}</td></tr>"
        )
    conclusion_html = "".join(f"<li>{html.escape(item)}</li>" for item in conclusions(snapshot))
    active_rows = "".join(
        f'<tr><th>GPU {html.escape(str(job["gpu"]))}</th>'
        f'<td>{html.escape(job["script"])}</td>'
        f'<td>{html.escape(job["method"])}</td>'
        f'<td>{html.escape(str(job["seed"]))}</td>'
        f'<td>{html.escape(str(job.get("phase", "运行中")))}</td>'
        f'<td>{fmt(job.get("progress", {}).get("step"), 0)}/'
        f'{fmt(job.get("progress", {}).get("total_steps"), 0)}</td>'
        f'<td>{fmt(job.get("progress", {}).get("loss"))}</td>'
        f'<td>{job["pid"]}</td></tr>'
        for job in snapshot["active_jobs"]
    ) or '<tr><td colspan="8">没有检测到实验进程</td></tr>'
    refresh = "" if snapshot["all_complete"] else '<meta http-equiv="refresh" content="15">'
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">{refresh}
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic-RQ Paper Lab</title>
<style>:root{{--bg:#071018;--panel:#0d1c28;--line:#20394a;--text:#edf7fb;--muted:#8eabb9;--cyan:#52d6df;--green:#55d69e;--amber:#ffc66d;--red:#ff7185}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% 0,#11364b,transparent 32%),var(--bg);color:var(--text);font:14px/1.5 system-ui,"PingFang SC",sans-serif}}main{{max-width:1450px;margin:auto;padding:34px 24px 70px}}h1{{margin:0;font-size:34px}}.lead{{color:var(--muted);margin:5px 0 22px}}h2{{margin:32px 0 12px}}.gpus{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;display:flex;flex-direction:column}}.card span{{color:var(--muted)}}.card strong{{font-size:25px;color:var(--cyan)}}.box{{background:rgba(13,28,40,.94);border:1px solid var(--line);border-radius:14px;overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:820px}}th,td{{padding:12px 14px;border-bottom:1px solid var(--line);text-align:right}}th:first-child{{text-align:left}}thead th{{color:var(--muted)}}.pill{{padding:4px 9px;border-radius:999px;font-size:12px;font-weight:700}}.done{{color:var(--green);background:#55d69e22}}.run{{color:var(--cyan);background:#52d6df22}}.wait{{color:var(--amber);background:#ffc66d20}}.fail{{color:var(--red);background:#ff718522}}.invalid{{color:#748b96;text-decoration:line-through}}.conclusions{{background:#0c2030;border-left:4px solid var(--cyan);padding:14px 20px;border-radius:8px}}li{{margin:8px 0}}code{{color:var(--cyan)}}@media(max-width:800px){{.gpus{{grid-template-columns:repeat(2,1fr)}}}}</style></head>
<body><main><h1>Semantic-RQ Paper Lab</h1><div class="lead">Gate 0 地址诊断 → Gate 1 受控语言建模 → One-pass → Capacity → 外部验证 · 更新于 {html.escape(snapshot['updated_at'])}</div>
<div class="gpus">{gpu_cards}</div><h2>当前运行任务</h2><div class="box"><table><thead><tr><th>设备</th><th>Runner</th><th>方法</th><th>Seed</th><th>阶段</th><th>Step</th><th>Train loss</th><th>主 PID</th></tr></thead><tbody>{active_rows}</tbody></table></div>
<h2>Gate 0 · 地址结构诊断</h2><div class="box"><table><thead><tr><th>Order</th><th>ρ(语义, RQ overlap)</th><th>ρ(语义, Shuffled)</th><th>低词面高语义 RQ overlap</th><th>Shuffled overlap</th><th>Held-out coverage</th><th>切片 pairs</th></tr></thead><tbody>{''.join(diagnostic_rows)}</tbody></table></div>
<h2>主对照 · Frequency-matched RQ-Shuffled 审计</h2><div class="box"><table><thead><tr><th>Seed / Order</th><th>全表 moved</th><th>已访问 rows moved</th><th>行数 histogram</th><th>访问加权 histogram</th><th>同频 singleton rows</th></tr></thead><tbody>{''.join(shuffle_rows)}</tbody></table></div>
<h2>公平容量审计</h2><div class="box"><table><thead><tr><th>地址方法</th><th>Layer 11 · 2gram</th><th>Layer 11 · 3gram</th><th>Layer 21 · 2gram</th><th>Layer 21 · 3gram</th><th>可训练参数</th><th>主表资格</th></tr></thead><tbody>
<tr><th>Semantic-RQ (M8,K256)</th><td>2048</td><td>2048</td><td>2048</td><td>2048</td><td>26,984,448</td><td><span class="pill done">是</span></td></tr>
<tr><th>Arithmetic-fixed (8×256)</th><td>2048</td><td>2048</td><td>2048</td><td>2048</td><td>26,984,448</td><td><span class="pill done">是</span></td></tr>
<tr><th>Legacy prime arithmetic</th><td>2194</td><td>2612</td><td>3000</td><td>3396</td><td>不匹配</td><td><span class="pill fail">否</span></td></tr>
</tbody></table></div><h2>Gate 1 · FineWeb-Edu 100M processed-token-slot fixedsteps（Seed 42）</h2><div class="box"><table><thead><tr><th>方法</th><th>状态</th><th>最近 train loss</th><th>中间/最终 Eval loss ↓</th><th>峰值显存 GB</th><th>秒/step</th><th>实际 causal targets</th></tr></thead><tbody>{''.join(gate_rows)}</tbody></table></div>
<h2>旧 Early-stop 诊断（不进入主表）</h2><div class="box"><table><thead><tr><th>方法</th><th>状态</th><th>实际 steps</th><th>Eval loss</th></tr></thead><tbody>{''.join(legacy_rows)}</tbody></table></div>
<h2>Gate 1 · 三 Seed 复现矩阵（完成 {snapshot['complete_gate1_total']}/12）</h2><div class="box"><table><thead><tr><th>方法</th><th>Seed 42</th><th>Seed 43</th><th>Seed 44</th></tr></thead><tbody>{''.join(replication_rows)}</tbody></table></div>
<h2>LM 因果切片 Manifest · 2,000 held-out rows（完成 {snapshot['complete_slice_manifests']}/3）</h2><div class="box"><table><thead><tr><th>数据切分</th><th>状态</th><th>2g exact</th><th>2g semantic</th><th>2g covered/no-neighbor</th><th>2g OOV</th><th>3g exact</th><th>3g semantic</th><th>3g covered/no-neighbor</th><th>3g OOV</th></tr></thead><tbody>{''.join(manifest_rows)}</tbody></table></div>
<h2>逐 Token NLL（完成 {snapshot['complete_lm_slices']}/15）</h2><div class="box"><table><thead><tr><th>方法</th><th>Seed 42</th><th>Seed 43</th><th>Seed 44</th></tr></thead><tbody>{''.join(slice_rows)}</tbody></table></div>
<h2>Paired Cluster Bootstrap（ΔNLL = Semantic-RQ − Control）</h2><div class="box"><table><thead><tr><th>比较</th><th>切片</th><th>ΔNLL ↓</th><th>95% CI</th><th>Tokens</th></tr></thead><tbody>{''.join(comparison_rows) or '<tr><td colspan="5">等待三 seed 逐 token loss</td></tr>'}</tbody></table></div>
<h2>Shared-head 因果干预（完成 {snapshot['complete_head_interventions']}/6）</h2><div class="box"><table><thead><tr><th>比较</th><th>切片</th><th>ΔNLL</th><th>95% CI</th><th>Tokens</th></tr></thead><tbody>{''.join(intervention_rows) or '<tr><td colspan="5">等待 shared/random-matched masking</td></tr>'}</tbody></table></div>
<h2>标准 LM Benchmark（完成 {snapshot['complete_standard_lm']}/10）</h2><div class="box"><table><thead><tr><th>方法</th><th>Seed</th><th>状态</th><th>Paloma WikiText-103 word PPL ↓</th><th>Paloma C4 word PPL ↓</th><th>LAMBADA Acc ↑</th><th>LAMBADA PPL ↓</th></tr></thead><tbody>{''.join(standard_rows)}</tbody></table></div>
<h2>Phase 2 · One-pass Replication（Gate: {phase2_status}）</h2><div class="box"><table><thead><tr><th>方法</th><th>Seed 42</th><th>Seed 43</th><th>Seed 44</th></tr></thead><tbody>{''.join(phase2_rows)}</tbody></table></div>
<h2>Finite-memory Capacity Sweep · 训练矩阵</h2><div class="box"><table><thead><tr><th>每 head 容量</th><th>方法</th><th>状态</th><th>Eval loss ↓</th><th>峰值显存 GB</th></tr></thead><tbody>{''.join(capacity_rows)}</tbody></table></div>
<h2>Finite-memory Capacity Sweep · Paired ΔNLL</h2><div class="box"><table><thead><tr><th>每 head 容量</th><th>比较</th><th>Overall ΔNLL ↓</th><th>95% CI</th><th>3g semantic/shared ΔNLL ↓</th><th>95% CI</th></tr></thead><tbody>{''.join(capacity_comparison_rows)}</tbody></table></div>
<h2>外部验证 · Benchmark-specific Frequency Audit</h2><div class="box"><table><thead><tr><th>Benchmark</th><th>访问统计</th><th>2g accesses</th><th>3g accesses</th><th>3 seeds 加权 histogram 保持</th></tr></thead><tbody>{''.join(external_access_rows)}</tbody></table></div>
<h2>外部验证 · 三 Seed Macro Accuracy</h2><div class="box"><table><thead><tr><th>Benchmark</th><th>方法</th><th>Seed 42</th><th>Seed 43</th><th>Seed 44</th><th>Mean</th><th>Std</th></tr></thead><tbody>{''.join(external_triad_rows)}</tbody></table></div>
<h2>外部验证 · 历史诊断（不承担主结论）</h2><div class="box"><table><thead><tr><th>Benchmark</th><th>Arithmetic s42</th><th>RQ s42</th><th>RQ s43</th><th>旧 matched s42（无效）</th><th>修正 matched s42</th><th>修正 matched s43</th></tr></thead><tbody>{''.join(ext_rows)}</tbody></table></div>
<h2>当前可写结论</h2><ul class="conclusions">{conclusion_html}</ul>
<p class="lead">公平口径：每个 n-gram order 为 8 heads × 256 rows/head = 2048 rows；正式 RQ-Shuffled 只在 LM-train 精确同访问频率组内置换完整 code vector，因此同时保持逐层行数和实际 access-weighted 桶负载，只破坏 n-gram↔语义地址对应。</p>
<p class="lead">数据隔离：FineWeb-Edu 流的前 5,000 篇只用于离线建表；LM train/eval 固定跳过前 6,000 行后才 shuffle 和切分，地址语料与评测语料不重叠。</p></main></body></html>'''


def main() -> None:
    args = parse_args()
    output = args.output_dir / "dashboard.html"
    while True:
        snapshot = collect(args.output_dir)
        atomic_write(output, render(snapshot))
        atomic_write(output.with_suffix(".json"), json.dumps(snapshot, indent=2))
        print(f"[{snapshot['updated_at']}] paper dashboard updated", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
