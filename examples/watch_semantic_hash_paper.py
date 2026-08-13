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
from collections.abc import Mapping
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


def valid_gate_b_result(payload: Mapping[str, Any]) -> bool:
    if payload.get("status") != "complete" or payload.get("paper_eligible") is not True:
        return False
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    for endpoint, expected_examples in (
        ("qqp_validation", 40_430),
        ("paws_wiki_test", 8_000),
    ):
        values = metrics.get(endpoint)
        if not isinstance(values, Mapping) or values.get("examples") != expected_examples:
            return False
        if not all(
            isinstance(values.get(metric), int | float)
            for metric in ("accuracy", "f1", "auroc")
        ):
            return False
    return True


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
    gate_b = list(
        re.finditer(r"step=(\d+)/(\d+)\s+loss=([0-9.eE+-]+)", content)
    )
    if gate_b:
        match = gate_b[-1]
        step = int(match.group(1))
        total = int(match.group(2))
        progress.update(
            percent=round(100 * step / total),
            step=step,
            total_steps=total,
            loss=float(match.group(3)),
        )
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
        "run_qqp_paws_frozen.py",
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
    base_standard = read_json(standard_root / "base_seed42.json")
    # Public benchmarks may be run separately so one unavailable dataset does
    # not discard completed results from the others. Merge their lm-eval
    # payloads into the row consumed by the dashboard.
    for split_path in sorted(standard_root.glob("base_seed42_*.json")):
        split_payload = read_json(split_path)
        if split_payload.get("status") != "complete":
            continue
        if not base_standard:
            base_standard = {"status": "complete", "results": {}}
        base_standard.setdefault("results", {}).update(
            split_payload.get("results", {})
        )
        base_standard.setdefault("benchmark_files", []).append(str(split_path))
    if base_standard.get("results"):
        base_standard["status"] = "complete"
    standard_lm = {"base_seed42": base_standard}
    for seed in GATE1_SEEDS:
        for method in STANDARD_METHODS:
            key = f"{method}_seed{seed}"
            combined = read_json(standard_root / f"{key}.json")
            for split_path in sorted(standard_root.glob(f"{key}_*.json")):
                split_payload = read_json(split_path)
                if split_payload.get("status") != "complete":
                    continue
                if not combined:
                    combined = {"status": "complete", "results": {}}
                combined.setdefault("results", {}).update(
                    split_payload.get("results", {})
                )
                combined.setdefault("benchmark_files", []).append(str(split_path))
            if combined.get("results"):
                combined["status"] = "complete"
            standard_lm[key] = combined
    qqp_paws_root = output_dir / "qqp_paws"
    qqp_paws = {
        f"{method}_seed{seed}": read_json(
            qqp_paws_root / f"{method}_seed{seed}.json"
        )
        for method in ("base", *STANDARD_METHODS)
        for seed in GATE1_SEEDS
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
        "qqp_paws": qqp_paws,
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
        items.append(
            "Gate 1 尚未形成三角对照；表中运行时 eval loss 只用于健康监控。"
            "各 run 当前 step 不同，不能横向比较，也不构成论文结论。"
        )
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
                    "wikitext",
                    ("word_perplexity,none", "word_perplexity"),
                )
                for seed in GATE1_SEEDS
            ]
            if all(value is not None for value in values):
                means[method] = sum(float(value) for value in values) / len(values)
        if len(means) == len(STANDARD_METHODS):
            items.append(
                "标准 WikiText-103 三 seed word PPL："
                f"Semantic-RQ={means['semantic_rq']:.3f}，"
                f"Arithmetic-fixed={means['arithmetic_matched']:.3f}，"
                f"RQ-Shuffled={means['rq_shuffled']:.3f}。"
            )
    else:
        items.append("标准 LM 评测进行中：WikiText-103、C4 validation 与 LAMBADA；三方法三 seed 完成前不提前外推结论。")
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
    diagnostic_fragments = []
    for benchmark in ("xnli", "pawsx"):
        row = external.get(benchmark, {})
        arithmetic_by_seed = [
            row.get("corrected_matched_seed42"),
            row.get("corrected_matched_seed43"),
        ]
        semantic_by_seed = [row.get("rq_seed42"), row.get("rq_seed43")]
        paired = [
            (float(arithmetic), float(semantic))
            for arithmetic, semantic in zip(
                arithmetic_by_seed, semantic_by_seed, strict=True
            )
            if isinstance(arithmetic, int | float)
            and isinstance(semantic, int | float)
        ]
        if len(paired) == 2:
            arithmetic_mean = statistics.mean(value[0] for value in paired)
            semantic_mean = statistics.mean(value[1] for value in paired)
            deltas = [100 * (semantic - arithmetic) for arithmetic, semantic in paired]
            sign_note = (
                "逐 seed 差值变号，目前更支持持平"
                if deltas[0] * deltas[1] < 0
                else "逐 seed 方向一致，但仍不足以形成主结论"
            )
            diagnostic_fragments.append(
                f"{benchmark.upper()} 两 seed：Arithmetic-fixed={arithmetic_mean:.4f}，"
                f"Semantic-RQ={semantic_mean:.4f}（均值 Δ={100 * (semantic_mean - arithmetic_mean):+.3f} pp；"
                f"seed42/43 Δ={deltas[0]:+.3f}/{deltas[1]:+.3f} pp；{sign_note}）"
            )
        elif len(paired) == 1:
            arithmetic, semantic = paired[0]
            diagnostic_fragments.append(
                f"{benchmark.upper()} 单 seed：Arithmetic-fixed={arithmetic:.4f}，"
                f"Semantic-RQ={semantic:.4f}（Δ={100 * (semantic - arithmetic):+.2f} pp）"
            )
    if diagnostic_fragments:
        items.append(
            "外部任务的历史诊断："
            + "；".join(diagnostic_fragments)
            + "。该比较尚缺 benchmark-specific RQ-Shuffled 与三 seed，不能承担主结论。"
        )
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
    completed_checkpoint_count = 0
    for name in STANDARD_METHODS:
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
            if (
                metrics.get("fixed_steps_complete") is True
                and completed_steps == 12_208
            ):
                completed_checkpoint_count += 1
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
        for seed in GATE1_SEEDS:
            payload = snapshot.get("standard_lm", {}).get(
                f"{method}_seed{seed}", {}
            )
            wikitext = standard_metric(
                payload,
                "wikitext",
                ("word_perplexity,none", "word_perplexity"),
            )
            c4 = standard_metric(
                payload,
                "c4",
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
            completed_tasks = sum(
                task in payload.get("results", {})
                for task in ("wikitext", "c4", "lambada_openai")
            )
            state = "完成" if completed_tasks == 3 else f"{completed_tasks}/3"
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
    completed_standard_rows = []
    for method in ("base", *STANDARD_METHODS):
        seeds = (42,) if method == "base" else GATE1_SEEDS
        for seed in seeds:
            payload = snapshot.get("standard_lm", {}).get(f"{method}_seed{seed}", {})
            if payload.get("status") != "complete":
                continue
            completed_standard_rows.append(
                f"<tr><th>{labels[method]}</th><td>{seed}</td>"
                f"<td>{fmt(standard_metric(payload, 'wikitext', ('word_perplexity,none', 'word_perplexity')), 3)}</td>"
                f"<td>{fmt(standard_metric(payload, 'c4', ('word_perplexity,none', 'word_perplexity', 'perplexity,none', 'perplexity')), 3)}</td>"
                f"<td>{fmt(standard_metric(payload, 'lambada_openai', ('acc,none', 'acc')), 4)}</td>"
                f"<td>{fmt(standard_metric(payload, 'lambada_openai', ('perplexity,none', 'perplexity')), 3)}</td></tr>"
            )
    standard_body = "".join(completed_standard_rows) or (
        '<tr><td colspan="6" class="empty">等待公平 matched checkpoints 完成；完成后立即运行公开 LM benchmark。</td></tr>'
    )
    qqp_paws_rows = []
    active_gate_b = {
        (str(job.get("method")), str(job.get("seed"))): job
        for job in snapshot.get("active_jobs", [])
        if job.get("script") == "run_qqp_paws_frozen.py"
    }
    for method in ("base", *STANDARD_METHODS):
        for seed in GATE1_SEEDS:
            payload = snapshot.get("qqp_paws", {}).get(f"{method}_seed{seed}", {})
            metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
            qqp = metrics.get("qqp_validation", {}) if isinstance(metrics, dict) else {}
            wiki = metrics.get("paws_wiki_test", {}) if isinstance(metrics, dict) else {}
            paws_qqp = (
                metrics.get("paws_qqp_dev_and_test", {})
                if isinstance(metrics, dict)
                else {}
            )
            if valid_gate_b_result(payload):
                status = '<span class="pill done">完成</span>'
            elif payload.get("status") == "complete":
                status = '<span class="pill fail">结果不完整</span>'
            elif payload.get("status") == "training":
                completed = payload.get("completed_steps", 0)
                total = payload.get("total_steps", 0)
                status = f'<span class="pill run">{completed}/{total}</span>'
            elif (method, str(seed)) in active_gate_b:
                progress = active_gate_b[(method, str(seed))].get("progress", {})
                completed = progress.get("step", "—")
                total = progress.get("total_steps", "—")
                status = f'<span class="pill run">{completed}/{total}</span>'
            else:
                status = '<span class="pill wait">等待/运行中</span>'
            qqp_paws_rows.append(
                f"<tr><th>{labels[method]}</th><td>{seed}</td><td>{status}</td>"
                f"<td>{fmt(qqp.get('accuracy'), 4)}</td><td>{fmt(qqp.get('f1'), 4)}</td>"
                f"<td>{fmt(wiki.get('accuracy'), 4)}</td><td>{fmt(wiki.get('f1'), 4)}</td>"
                f"<td>{fmt(paws_qqp.get('accuracy'), 4)}</td><td>{fmt(paws_qqp.get('f1'), 4)}</td></tr>"
            )
    qqp_paws_body = "".join(qqp_paws_rows)
    gate_b_seed42: dict[str, dict[str, float]] = {}
    for method in ("base", *STANDARD_METHODS):
        payload = snapshot.get("qqp_paws", {}).get(f"{method}_seed42", {})
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        qqp = metrics.get("qqp_validation", {}) if isinstance(metrics, dict) else {}
        wiki = metrics.get("paws_wiki_test", {}) if isinstance(metrics, dict) else {}
        if valid_gate_b_result(payload) and all(
            isinstance(value, int | float)
            for value in (qqp.get("accuracy"), wiki.get("accuracy"))
        ):
            gate_b_seed42[method] = {
                "qqp": float(qqp["accuracy"]),
                "wiki": float(wiki["accuracy"]),
            }
    if len(gate_b_seed42) == 4:
        semantic_gate_b = gate_b_seed42["semantic_rq"]
        controls_gate_b = [
            gate_b_seed42[name] for name in ("base", "arithmetic_matched", "rq_shuffled")
        ]
        beats_all_wiki = all(
            semantic_gate_b["wiki"] > control["wiki"] for control in controls_gate_b
        )
        no_qqp_regression = semantic_gate_b["qqp"] >= min(
            control["qqp"] for control in controls_gate_b
        )
        gate_b_verdict = (
            "Seed42 暂时通过" if beats_all_wiki and no_qqp_regression else "Seed42 未通过"
        )
        gate_b_detail = (
            f"PAWS-Wiki Acc：Base {gate_b_seed42['base']['wiki']:.2%} / "
            f"Semantic {semantic_gate_b['wiki']:.2%} / Arithmetic "
            f"{gate_b_seed42['arithmetic_matched']['wiki']:.2%} / Shuffled "
            f"{gate_b_seed42['rq_shuffled']['wiki']:.2%}；QQP Acc：Semantic "
            f"{semantic_gate_b['qqp']:.2%}。仍需 seed43/44 与 PAWS-QQP 主 endpoint。"
        )
    else:
        gate_b_verdict = f"Seed42 {len(gate_b_seed42)}/4 完成"
        if "base" in gate_b_seed42 and "semantic_rq" in gate_b_seed42:
            base_gate_b = gate_b_seed42["base"]
            semantic_gate_b = gate_b_seed42["semantic_rq"]
            qqp_delta = semantic_gate_b["qqp"] - base_gate_b["qqp"]
            wiki_delta = semantic_gate_b["wiki"] - base_gate_b["wiki"]
            gate_b_verdict = "Seed42 暂不支持 OOD 泛化"
            comparison = (
                f"Semantic vs Base：QQP Acc {semantic_gate_b['qqp']:.2%} vs "
                f"{base_gate_b['qqp']:.2%}（{100 * qqp_delta:+.2f} pp），PAWS-Wiki Acc "
                f"{semantic_gate_b['wiki']:.2%} vs {base_gate_b['wiki']:.2%}"
                f"（{100 * wiki_delta:+.2f} pp）。"
            )
            if "arithmetic_matched" in gate_b_seed42:
                arithmetic_gate_b = gate_b_seed42["arithmetic_matched"]
                comparison += (
                    f" Semantic vs Arithmetic：QQP "
                    f"{100 * (semantic_gate_b['qqp'] - arithmetic_gate_b['qqp']):+.2f} pp，"
                    f"PAWS-Wiki {100 * (semantic_gate_b['wiki'] - arithmetic_gate_b['wiki']):+.2f} pp。"
                )
            gate_b_detail = (
                comparison
                + "内域略升但 OOD 下降；仍需 Shuffled 与 seed43/44 判断原因和稳定性。"
            )
        elif "base" in gate_b_seed42:
            gate_b_detail = (
                f"Base：QQP Acc {gate_b_seed42['base']['qqp']:.2%}，"
                f"PAWS-Wiki Acc {gate_b_seed42['base']['wiki']:.2%}。"
                "其余方法全量运行中；四个方法齐后自动进行三角判断。"
            )
        else:
            gate_b_detail = "全量 GLUE QQP 训练中；四个方法齐后自动进行三角判断。"
    base_payload = snapshot.get("standard_lm", {}).get("base_seed42", {})
    base_lambada_acc = standard_metric(
        base_payload, "lambada_openai", ("acc,none", "acc")
    )
    base_lambada_ppl = standard_metric(
        base_payload,
        "lambada_openai",
        ("perplexity,none", "perplexity"),
    )
    if base_lambada_acc is not None:
        base_wikitext = standard_metric(
            base_payload, "wikitext", ("word_perplexity,none", "word_perplexity")
        )
        base_c4 = standard_metric(
            base_payload,
            "c4",
            (
                "word_perplexity,none",
                "word_perplexity",
                "perplexity,none",
                "perplexity",
            ),
        )
        completed_base_tasks = (
            1 + int(base_wikitext is not None) + int(base_c4 is not None)
        )
        base_status_badge = (
            f'<span class="pill done">{completed_base_tasks}/3 已完成</span>'
        )
        base_status_text = (
            (f"WikiText-103 PPL {base_wikitext:.3f}；" if base_wikitext is not None else "")
            + f"LAMBADA Accuracy {base_lambada_acc:.2%}，PPL {base_lambada_ppl:.3f}；"
            + (f"C4 validation PPL {base_c4:.3f}。" if base_c4 is not None else "C4 正在运行。")
            + "结果均自动进入 Gate A 主表。"
        )
    else:
        base_status_badge = '<span class="pill run">LAMBADA 运行中</span>'
        base_status_text = "当前先跑 LAMBADA；完成后结果自动进入 Gate A 主表。"
    arithmetic42 = snapshot.get("standard_lm", {}).get("arithmetic_matched_seed42", {})
    arithmetic42_wiki = standard_metric(
        arithmetic42, "wikitext", ("word_perplexity,none", "word_perplexity")
    )
    arithmetic42_lambada_acc = standard_metric(
        arithmetic42, "lambada_openai", ("acc,none", "acc")
    )
    arithmetic43 = snapshot.get("standard_lm", {}).get("arithmetic_matched_seed43", {})
    arithmetic43_wiki = standard_metric(
        arithmetic43, "wikitext", ("word_perplexity,none", "word_perplexity")
    )
    arithmetic43_lambada_acc = standard_metric(
        arithmetic43, "lambada_openai", ("acc,none", "acc")
    )
    shuffled42 = snapshot.get("standard_lm", {}).get("rq_shuffled_seed42", {})
    shuffled42_wiki = standard_metric(
        shuffled42, "wikitext", ("word_perplexity,none", "word_perplexity")
    )
    semantic42 = snapshot.get("standard_lm", {}).get("semantic_rq_seed42", {})
    semantic42_wiki = standard_metric(
        semantic42, "wikitext", ("word_perplexity,none", "word_perplexity")
    )
    shuffled42_lambada_acc = standard_metric(
        shuffled42, "lambada_openai", ("acc,none", "acc")
    )
    semantic42_lambada_acc = standard_metric(
        semantic42, "lambada_openai", ("acc,none", "acc")
    )
    shuffled43 = snapshot.get("standard_lm", {}).get("rq_shuffled_seed43", {})
    shuffled43_wiki = standard_metric(
        shuffled43, "wikitext", ("word_perplexity,none", "word_perplexity")
    )
    shuffled43_lambada_acc = standard_metric(
        shuffled43, "lambada_openai", ("acc,none", "acc")
    )
    semantic43 = snapshot.get("standard_lm", {}).get("semantic_rq_seed43", {})
    semantic43_wiki = standard_metric(
        semantic43, "wikitext", ("word_perplexity,none", "word_perplexity")
    )
    semantic43_lambada_acc = standard_metric(
        semantic43, "lambada_openai", ("acc,none", "acc")
    )
    if all(
        value is not None
        for value in (
            arithmetic42_wiki,
            shuffled42_wiki,
            semantic42_wiki,
            arithmetic42_lambada_acc,
            shuffled42_lambada_acc,
            semantic42_lambada_acc,
        )
    ):
        public_verdict = "Seed42 两个公开任务均否定核心假设"
        public_detail = (
            f"WikiText PPL：Base {base_wikitext:.3f} / Semantic {semantic42_wiki:.3f} / "
            f"Arithmetic {arithmetic42_wiki:.3f} / Shuffled {shuffled42_wiki:.3f}；"
            f"LAMBADA Acc：Base {base_lambada_acc:.2%} / Semantic "
            f"{semantic42_lambada_acc:.2%} / Arithmetic {arithmetic42_lambada_acc:.2%} / "
            f"Shuffled {shuffled42_lambada_acc:.2%}。Semantic 未同时击败两个对照。"
        )
        if all(
            value is not None
            for value in (
                arithmetic43_wiki,
                shuffled43_wiki,
                semantic43_wiki,
                arithmetic43_lambada_acc,
                shuffled43_lambada_acc,
                semantic43_lambada_acc,
            )
        ):
            public_verdict = "两个 seed、两个公开任务均否定核心假设"
            public_detail += (
                f" Seed43 再现：WikiText Semantic {semantic43_wiki:.3f} / Arithmetic "
                f"{arithmetic43_wiki:.3f} / Shuffled {shuffled43_wiki:.3f}；LAMBADA "
                f"Semantic {semantic43_lambada_acc:.2%} / Arithmetic "
                f"{arithmetic43_lambada_acc:.2%} / Shuffled {shuffled43_lambada_acc:.2%}，"
                "三者仍均劣于 Base。"
            )
        elif shuffled43_wiki is not None:
            public_detail += (
                f" Shuffled WikiText 已跨 seed 复现：seed42 {shuffled42_wiki:.3f} / "
                f"seed43 {shuffled43_wiki:.3f}，均劣于 Base。"
            )
    elif all(
        value is not None
        for value in (arithmetic42_wiki, shuffled42_wiki, semantic42_wiki)
    ):
        public_verdict = "Seed42 WikiText 三角对照否定核心假设"
        public_detail = (
            f"Base {base_wikitext:.3f}；Semantic {semantic42_wiki:.3f}；"
            f"Arithmetic {arithmetic42_wiki:.3f}；Shuffled {shuffled42_wiki:.3f}。"
            "Semantic 未击败任一公平对照且仍退化；等待 LAMBADA 与多 seed 复核。"
        )
    elif (
        arithmetic42_wiki is not None
        and arithmetic43_wiki is not None
        and arithmetic42_lambada_acc is not None
        and arithmetic43_lambada_acc is not None
    ):
        public_verdict = "Arithmetic 两任务退化均跨 seed 复现"
        public_detail = (
            f"WikiText PPL：Base 12.430，seed42 {arithmetic42_wiki:.3f}，"
            f"seed43 {arithmetic43_wiki:.3f}；LAMBADA Acc：Base 63.21%，"
            f"seed42 {arithmetic42_lambada_acc:.2%}，seed43 "
            f"{arithmetic43_lambada_acc:.2%}。等待 Semantic 与 Shuffled。"
        )
    elif arithmetic42_wiki is not None and arithmetic42_lambada_acc is not None:
        public_verdict = "Arithmetic seed42 明显退化"
        public_detail = (
            f"WikiText PPL 12.430→{arithmetic42_wiki:.3f}；"
            f"LAMBADA Acc 63.21%→{arithmetic42_lambada_acc:.2%}。"
            "仅是阶段性结果，等待 Semantic 与 Shuffled。"
        )
    else:
        public_verdict = "尚无公开 benchmark 正向结论"
        public_detail = "地址几何成立，但实际收益等待 Gate A / Gate B。"
    xnli = snapshot.get("external", {}).get("xnli", {})
    arithmetic_xnli = [
        xnli.get("corrected_matched_seed42"),
        xnli.get("corrected_matched_seed43"),
    ]
    semantic_xnli = [xnli.get("rq_seed42"), xnli.get("rq_seed43")]
    xnli_pairs = [
        (float(a), float(s))
        for a, s in zip(arithmetic_xnli, semantic_xnli, strict=True)
        if isinstance(a, int | float) and isinstance(s, int | float)
    ]
    if len(xnli_pairs) == 2:
        arithmetic_xnli_mean = statistics.mean(a for a, _ in xnli_pairs)
        semantic_xnli_mean = statistics.mean(s for _, s in xnli_pairs)
        xnli_delta = 100 * (semantic_xnli_mean - arithmetic_xnli_mean)
        xnli_summary = (
            f"Arithmetic-fixed {arithmetic_xnli_mean:.3%} · Semantic-RQ {semantic_xnli_mean:.3%} · "
            f"Δ {xnli_delta:+.3f} pp · 两 seed 方向变号，结论：持平"
        )
    else:
        xnli_summary = "尚无完整公平对照"
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
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic Hash · Benchmark Lab</title>
<style>:root{{--paper:#f3f0e8;--ink:#17201d;--muted:#68716c;--card:#fffdf8;--line:#d8d3c7;--green:#176b4d;--blue:#235ca5;--amber:#9a6015;--red:#a43636}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 Inter,ui-sans-serif,system-ui,"PingFang SC",sans-serif}}main{{max-width:1320px;margin:auto;padding:38px 24px 80px}}header{{display:flex;justify-content:space-between;gap:24px;align-items:end;border-bottom:2px solid var(--ink);padding-bottom:18px}}h1{{font-family:Georgia,serif;font-size:38px;line-height:1;margin:0}}h2{{font-size:20px;margin:34px 0 11px}}.kicker{{text-transform:uppercase;letter-spacing:.14em;color:var(--green);font-weight:800;font-size:12px}}.meta,.muted{{color:var(--muted)}}.verdicts,.gpus{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}}.experiments{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}.experiment{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}}.experiment h3{{margin:0 0 10px;font-size:16px}}.experiment dl{{display:grid;grid-template-columns:72px 1fr;gap:6px 10px;margin:0}}.experiment dt{{font-weight:800;color:var(--muted)}}.experiment dd{{margin:0}}.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;align-items:stretch}}.flow article{{background:var(--card);border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:10px;padding:14px}}.flow b{{display:block;margin-bottom:6px}}.flow p{{margin:0;color:var(--muted)}}.method-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px}}.method-grid article{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}.method-grid h3{{margin:0 0 6px;font-size:15px}}.verdict{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:17px;min-height:122px}}.verdict.primary{{border-top:4px solid var(--green)}}.verdict b{{display:block;font-size:13px;margin-bottom:9px}}.verdict strong{{font-family:Georgia,serif;font-size:23px;line-height:1.12}}.gpu{{background:#1d2924;color:#f8f5ec;border-radius:10px;padding:14px 16px;display:flex;justify-content:space-between;align-items:center}}.gpu span{{color:#b9c7c0}}.gpu strong{{font-size:23px}}.box{{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:760px}}th,td{{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}}th:first-child,td:first-child{{text-align:left}}thead th{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;background:#f7f4ed}}.pill{{display:inline-block;padding:3px 8px;border-radius:999px;font-size:11px;font-weight:800}}.done{{color:var(--green);background:#dcefe6}}.run{{color:var(--blue);background:#e2ebf8}}.wait{{color:var(--amber);background:#f6ead7}}.fail{{color:var(--red);background:#f6dddd}}.empty{{text-align:center!important;color:var(--muted);padding:26px}}details{{margin-top:24px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px}}summary{{cursor:pointer;font-weight:800}}code{{color:var(--blue)}}@media(max-width:900px){{.verdicts,.gpus,.experiments,.flow,.method-grid{{grid-template-columns:repeat(2,1fr)}}header{{display:block}}}}@media(max-width:620px){{.experiments,.flow,.method-grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><header><div><div class="kicker">Benchmark-first · live report</div><h1>Semantic Hash for Engram</h1><div class="meta">公开 benchmark 先判生死，机制实验后置</div></div><div class="meta">更新于 {html.escape(snapshot['updated_at'])}<br>自动刷新：15 秒</div></header>
<section class="verdicts"><div class="verdict primary"><b>Gate A · 语言建模</b><strong>{html.escape(public_verdict)}</strong><p class="muted">{html.escape(public_detail)}</p></div><div class="verdict"><b>Gate B · QQP → PAWS</b><strong>{html.escape(gate_b_verdict)}</strong><p class="muted">{html.escape(gate_b_detail)}</p></div><div class="verdict"><b>地址几何</b><strong>结构信号明确</strong><p class="muted">2g ρ=.738 vs .006；3g ρ=.699 vs −.011。不能替代 benchmark。</p></div><div class="verdict"><b>XNLI 附录诊断</b><strong>基本持平</strong><p class="muted">{html.escape(xnli_summary)}</p></div></section>
<h2>阶段 1 到底怎么训练？</h2><p class="muted"><b>准确定位：冻结底座的 Engram memory 预训练。</b> 它形式上类似 continued pretraining，但不是全参数 CPT：Qwen3-1.7B 始终冻结，只用 causal LM 目标学习新增的 memory value。先训练是必要的，因为地址仅决定 n-gram 去哪个 slot，而 slot 初始是随机向量。该做法对新增 memory / adapter 是合理常规操作；但下面的具体数据与预算是本文的受控实验协议，不是社区统一的 Engram benchmark，也不能单独证明方法有效。</p><div class="flow"><article><b>1 · 训练文本</b><p>FineWeb-Edu：48,832 条序列，每条最多 256 tokens。</p></article><article><b>2 · 冻结底座</b><p>Qwen3-1.7B 全部冻结，只更新约 27M Engram 参数。</p></article><article><b>3 · 查 memory</b><p>当前位置的 2-gram / 3-gram 经各自地址函数找到 memory slot。</p></article><article><b>4 · 注入并预测</b><p>读取 memory 向量注入 Qwen，用标准 causal LM loss 预测下一 token。</p></article><article><b>5 · 跑满并保存</b><p>batch 4、累积 8、12,208 steps、8 epochs；保存完整 checkpoint。</p></article></div><div class="method-grid"><article><h3>Arithmetic-fixed</h3><p>token ID → 固定算术 hash → slot。语义相近短语不会刻意靠近，是原始随机离散寻址基线。</p></article><article><h3>Semantic-RQ（我们的方法）</h3><p>离线 n-gram embedding → residual quantization code → slot。地址表和 codebook 固定，训练期只学习 memory value。</p></article><article><h3>RQ-Shuffled（关键因果对照）</h3><p>保留 Semantic-RQ 的容量与访问频率，再打乱 n-gram↔地址关系，用来单独消除语义几何。</p></article></div><p class="muted"><b>为什么是 9 个？</b> 三种寻址 × seeds 42/43/44。seed 改变 memory 初始化和数据顺序；最终报告均值和方差。<b>阶段 1 只负责把随机 memory 训练成可用 checkpoint 并控制训练预算；能否迁移到 WikiText、C4、LAMBADA 与 PAWS，才是论文结论。</b></p>
<h2>现在每个实验在做什么</h2><div class="experiments"><article class="experiment"><h3>① 公平 checkpoint 训练 <span class="pill run">{completed_checkpoint_count}/9 已完成</span></h3><dl><dt>目的</dt><dd>获得三种地址方法可直接比较的完整模型。</dd><dt>协议</dt><dd>冻结 Qwen3-1.7B，只训练 Engram；FineWeb-Edu；12,208 steps；3 seeds。</dd><dt>对照</dt><dd>Arithmetic-fixed、RQ-Shuffled、Semantic-RQ；参数与表容量严格一致。</dd><dt>成功标准</dt><dd>这里只产出 checkpoint，不用不同 step 的中间 loss 下结论。</dd></dl></article><article class="experiment"><h3>② Base 公开 LM 基线 {base_status_badge}</h3><dl><dt>目的</dt><dd>测没有 Engram 时 Qwen3-1.7B 的公开 benchmark 基准线。</dd><dt>结果/状态</dt><dd>{base_status_text}</dd><dt>协议</dt><dd>WikiText-103、C4 validation 与 LAMBADA 均使用公开标准任务。</dd><dt>对照</dt><dd>后续所有 Arithmetic、Shuffled、Semantic checkpoint 与同一 Base 比较。</dd></dl></article><article class="experiment"><h3>③ Gate A：标准语言建模 <span class="pill run">{completed_checkpoint_count}/9 checkpoint 可评测</span></h3><dl><dt>目的</dt><dd>直接判断 Semantic Hash 是否带来公开 LM 收益。</dd><dt>当前状态</dt><dd>每个 checkpoint 的 WikiText、LAMBADA、C4 以 batch=1 独立运行并逐项入表。WikiText/LAMBADA 可与训练共卡；C4 的超长样本在共卡时出现 logits 峰值 OOM，需待独占 GPU 后重跑，失败结果不入表。</dd><dt>任务</dt><dd>WikiText-103、C4 validation、LAMBADA、FineWeb held-out。</dd><dt>成功标准</dt><dd>Semantic-RQ 同时优于 Arithmetic 和 Shuffled，至少两个 benchmark 方向一致。</dd><dt>失败处理</dt><dd>若持平或退化，不用地址几何包装成方法提升。</dd></dl></article><article class="experiment"><h3>④ Gate B：QQP → PAWS <span class="pill run">全量 QQP 运行中</span></h3><dl><dt>目的</dt><dd>验证公开 paraphrase OOD 泛化，而不是自建“语义泛化”样本。</dd><dt>当前协议</dt><dd>冻结阶段 1 checkpoint，只训练统一分类头；全量 GLUE QQP train → QQP validation 与 PAWS-Wiki test zero-update。</dd><dt>待补 endpoint</dt><dd>PAWS-QQP dev_and_test 因官方 index 链接失效待恢复；PAWS-Wiki 明确作为辅助 OOD，不冒充 PAWS-QQP。</dd><dt>成功标准</dt><dd>PAWS 提升且 QQP 不退化；否则不能 claim 语义泛化。</dd></dl></article></div>
<h2>四卡资源</h2><div class="gpus">{''.join(f'<div class="gpu"><div><b>GPU {g["index"]}</b><br><span>{g["used_mib"]}/{g["total_mib"]} MiB</span></div><strong>{g["util"]}%</strong></div>' for g in snapshot['gpus'])}</div>
<h2>正在运行</h2><div class="box"><table><thead><tr><th>设备</th><th>Runner</th><th>方法</th><th>Seed</th><th>阶段</th><th>Step</th><th>Loss</th><th>PID</th></tr></thead><tbody>{active_rows}</tbody></table></div>
<h2>阶段 1 · 训练三组公平对照模型（不是最终 Benchmark）</h2><p class="muted">这一步只制造后续公开 benchmark 所需的实验模型。三种方法使用同一底座、数据、参数量、表容量、训练 token 和 seed；只有跑满 12,208 steps 的 checkpoint 才能进入 Gate A / Gate B。表里的中间 loss 仅用于确认训练正常，禁止据此声称哪种方法更好。</p><div class="box"><table><thead><tr><th>对照</th><th>唯一关键差别</th><th>它回答的问题</th></tr></thead><tbody><tr><th>Semantic-RQ vs Arithmetic-fixed</th><td>语义量化地址 vs 原始离散 n-gram hash</td><td>新寻址整体是否优于原始 Engram 寻址？</td></tr><tr><th>Semantic-RQ vs RQ-Shuffled</th><td>容量和访问频率匹配，但 Shuffled 破坏语义邻域</td><td>若有收益，是否真的来自语义地址几何？这是最关键的因果对照。</td></tr><tr><th>RQ-Shuffled vs Arithmetic-fixed</th><td>量化桶统计变化，但没有正确语义对应</td><td>单纯改变碰撞/桶分布是否已经足以产生收益？</td></tr></tbody></table></div><div class="box"><table><thead><tr><th>待训练方法</th><th>Seed 42</th><th>Seed 43</th><th>Seed 44</th></tr></thead><tbody>{''.join(replication_rows)}</tbody></table></div>
<h2>Gate A · 标准语言建模 Benchmark</h2><p class="muted">第一主表。Semantic-RQ 必须同时优于 Arithmetic-fixed 与 RQ-Shuffled，且至少两个公开 benchmark 方向一致。</p><div class="box"><table><thead><tr><th>方法</th><th>Seed</th><th>WikiText-103 PPL ↓</th><th>C4 validation PPL ↓</th><th>LAMBADA Acc ↑</th><th>LAMBADA PPL ↓</th></tr></thead><tbody>{standard_body}</tbody></table></div>
<h2>Gate B · 冻结表示的 QQP → PAWS</h2><p class="muted">阶段 1 Base/Engram checkpoint 全部冻结，只在完整可用 GLUE QQP train（363,846 条）上训练同构线性分类头；QQP validation 40,430 条，PAWS-Wiki test 8,000 条。PAWS-Wiki 是当前可直接复现的辅助 OOD；PAWS-QQP 官方 index 链接失效，恢复并校验 11,988/677 行之前该列保持为空，绝不拿 Wiki 数据冒充。</p><div class="box"><table><thead><tr><th>方法</th><th>Seed</th><th>状态</th><th>QQP Acc ↑</th><th>QQP F1 ↑</th><th>PAWS-Wiki Acc ↑</th><th>PAWS-Wiki F1 ↑</th><th>PAWS-QQP Acc ↑</th><th>PAWS-QQP F1 ↑</th></tr></thead><tbody>{qqp_paws_body}</tbody></table></div>
<details><summary>附录证据：地址几何与公平性</summary><h2>地址结构诊断</h2><div class="box"><table><thead><tr><th>Order</th><th>ρ(semantic,RQ)</th><th>ρ(semantic,shuffle)</th><th>低词面高语义 overlap</th><th>shuffle overlap</th><th>coverage</th><th>pairs</th></tr></thead><tbody>{''.join(diagnostic_rows)}</tbody></table></div><h2>容量公平性</h2><div class="box"><table><thead><tr><th>方法</th><th>Rows/order</th><th>可训练参数</th><th>资格</th></tr></thead><tbody><tr><th>Semantic-RQ M8,K256</th><td>2048</td><td>26,984,448</td><td><span class="pill done">有效</span></td></tr><tr><th>Arithmetic-fixed 8×256</th><td>2048</td><td>26,984,448</td><td><span class="pill done">有效</span></td></tr><tr><th>旧 arithmetic/matched-v2</th><td>不匹配</td><td>不匹配</td><td><span class="pill fail">永久排除</span></td></tr></tbody></table></div></details>
<p class="muted" style="margin-top:28px">判死规则：若标准 LM 与 QQP→PAWS 两条主线均不优于两个公平基线，则停止扩规模；不使用自建 slice 或地址相关性包装成方法收益。</p></main></body></html>'''


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
