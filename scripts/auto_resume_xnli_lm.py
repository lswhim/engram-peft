#!/usr/bin/env python3
"""Resume XNLI / LM workers as soon as their RQ warm-cache pre-encode finishes.

Run this once on the cluster, ideally from a small non-GPU shell:

    nohup python scripts/auto_resume_xnli_lm.py --machine 3 >> run_logs/auto_resume.log 2>&1 &

It watches the pre-encode logs, then starts each benchmark worker on a GPU that
is genuinely free.  The scheduler is idempotent: launch decisions are persisted
under ``run_logs/.auto_resume`` so restarting it does not double-start a job.

Only the ``semantic_keyed`` warm cache is being pre-encoded right now.  The RQ
code for a given n-gram is the same regardless of flatten vs keyed readout, so
once the keyed cache finishes, this scheduler copies it into the flatten cache
directory instead of re-running the expensive pre-encode.  This is the intended
"pre-encode once, reuse for both readouts" path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/anguszhang-cfs-nj/seokliu_workspace/engram")
PY = (
    "/anguszhang-cfs-nj/seokliu_workspace/"
    "miniconda3/envs/engram/bin/python"
)
SEM_TABLE = ROOT / "rq_tables/fineweb_qwen3emb06b_M8K256_300k_strict"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def state_dir() -> Path:
    path = ROOT / "run_logs" / ".auto_resume"
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_for(name: str) -> Path:
    return state_dir() / f"{name}.json"


def running_pids() -> set[int]:
    """Return all PIDs visible to this pod that own an active GPU compute context."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    pids: set[int] = set()
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids


def gpu_is_free(index: int) -> bool:
    """Free means no visible compute app AND ample free memory.

    The shared pod's nvidia-smi can list compute processes owned by other
    namespaces that are invisible under /proc.  Requiring both conditions
    avoids placing a worker on a card that is actually busy elsewhere.
    Qwen3-1.7B + Engram needs roughly 15-20GB, so only treat a card as
    schedulable when at least 20GB is genuinely free.
    """
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            f"--id={index}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    for line in apps.splitlines():
        if line.strip():
            return False
    try:
        free = int(
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    f"--id={index}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
    except (ValueError, TypeError):
        return False
    return free >= 20_000


def process_uses_gpu(pid: int, index: int) -> bool:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            f"--id={index}",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return str(pid) in set(line.strip() for line in out.splitlines())


def process_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def log_tail(path: Path, limit: int = 20_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


@dataclass
class TrainJob:
    name: str
    worker: str
    mode: str
    preencode_name: str | None = None
    gpu: int | None = None

    def command(self, gpu: int) -> list[str]:
        return ["bash", self.worker, str(gpu), self.mode]


def preencode_complete(name: str, log: Path, proc_pid: int | None) -> bool:
    """Complete only when the recorded pre-encode PID exited and log has marker."""
    st = read_json(state_for(name))
    if st.get("status") == "done":
        return True
    if proc_pid is not None and process_alive(proc_pid):
        return False
    return "[preencode] done" in log_tail(log)


def copy_flatten_caches(seed: int = 42) -> None:
    """Copy a completed keyed cache to the flatten namespace.

    RQ codes are readout-independent, so this is semantically identical to a
    separate flatten pre-encode and avoids re-running the expensive encoder.
    """
    pairs = [
        (
            SEM_TABLE / f"lm100m_cache_semantic_keyed_seed{seed}",
            SEM_TABLE / f"lm100m_cache_semantic_flatten_seed{seed}",
        ),
        (
            SEM_TABLE / f"xnli_cache_semantic_keyed_seed{seed}",
            SEM_TABLE / f"xnli_cache_semantic_flatten_seed{seed}",
        ),
    ]
    for src_dir, dst_dir in pairs:
        src = src_dir / "semantic_codes.sqlite3"
        dst = dst_dir / "semantic_codes.sqlite3"
        if not src.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        temporary = dst.with_suffix(".sqlite3.tmp")
        shutil.copy2(src, temporary)
        os.replace(temporary, dst)


def flatten_cache_ready(seed: int = 42) -> bool:
    lm = SEM_TABLE / f"lm100m_cache_semantic_flatten_seed{seed}" / "semantic_codes.sqlite3"
    xnli = SEM_TABLE / f"xnli_cache_semantic_flatten_seed{seed}" / "semantic_codes.sqlite3"
    return lm.exists() and xnli.exists()


def find_free_gpu(gpus: list[int]) -> int | None:
    # Prefer the card with the most free memory among schedulable cards.
    candidates: list[tuple[int, int]] = []
    for idx in gpus:
        if gpu_is_free(idx):
            try:
                free = int(
                    subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.free",
                            f"--id={idx}",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    ).stdout.strip()
                )
            except (ValueError, TypeError):
                continue
            candidates.append((free, idx))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def mark(job: TrainJob, status: str, gpu: int | None = None) -> None:
    write_json(
        state_for(job.name),
        {
            "name": job.name,
            "status": status,
            "gpu": gpu,
            "updated_at": now(),
        },
    )


def launch(job: TrainJob, gpu: int) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = ROOT / "run_logs" / f"auto_resume_{job.name}.log"
    with log.open("ab") as handle:
        handle.write(f"\n===== {now()} {job.name} gpu={gpu} =====\n".encode())
        handle.flush()
        subprocess.Popen(
            job.command(gpu),
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    mark(job, "launched", gpu)
    print(f"[auto-resume] launched {job.name} on gpu {gpu}", flush=True)


def launched_worker_alive(job: TrainJob) -> bool:
    """Return True when a worker for this job is still running."""
    if job.worker.endswith("run_lm_100m_worker.sh"):
        needle = f"run_lm_100m_worker.sh {job.mode}"
    else:
        # The XNLI worker may have been launched manually (directly through
        # run_xtreme_xnli.py), so detect by the router instead of the wrapper.
        router = "semantic_keyed" if job.mode == "semantic_keyed" else "flatten"
        needle = f"run_xtreme_xnli.py .*rq_router {router}"
    out = subprocess.run(
        ["pgrep", "-f", needle],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--machine",
        type=int,
        default=3,
        help="Which Taiji machine host this scheduler controls (3 or 1).",
    )
    parser.add_argument("--interval", type=float, default=90.0)
    args = parser.parse_args()

    gpus = list(range(8)) if args.machine == 3 else list(range(4))
    preencode_logs = {
        "preencode_lm_keyed": (
            Path("/tmp/preencode_lm_full.log"),
            15250,
        ),
        "preencode_xnli_keyed": (
            Path("/tmp/preencode_xnli_full.log"),
            14935,
        ),
    }
    jobs = [
        # XNLI keyed is intentionally running on 1号机 (PID 18414).  This
        # scheduler only manages 3号机 and cannot see that process, so it must
        # not be in the scheduling list or it would launch a duplicate.
        # XNLI flatten is intentionally launched manually on 3号机 GPU7
        # (PID 75524) and must not be rescheduled by this process either.
        TrainJob("lm_semantic_keyed", "scripts/run_lm_100m_worker.sh",
                 "semantic_keyed", "preencode_lm_keyed"),
        TrainJob("lm_semantic_flatten", "scripts/run_lm_100m_worker.sh",
                 "semantic_flatten", "preencode_lm_keyed"),
        TrainJob("lm_arithmetic", "scripts/run_lm_100m_worker.sh", "arithmetic"),
    ]
    by_name = {job.name: job for job in jobs}

    while True:
        try:
            ready_deps: set[str] = set()
            for dep_name, (log, proc_pid) in preencode_logs.items():
                st = read_json(state_for(dep_name))
                if st.get("status") == "done":
                    ready_deps.add(dep_name)
                elif preencode_complete(dep_name, log, proc_pid):
                    write_json(
                        state_for(dep_name),
                        {"name": dep_name, "status": "done", "updated_at": now()},
                    )
                    ready_deps.add(dep_name)
                    print(f"[auto-resume] preencode done: {dep_name}", flush=True)

            # Only reuse a fully-finished keyed cache for the flatten namespace.
            if "preencode_lm_keyed" in ready_deps or "preencode_xnli_keyed" in ready_deps:
                copy_flatten_caches()

            reserved: set[int] = set()
            launched_this_pass = False
            for job in jobs:
                st = read_json(state_for(job.name))
                if st.get("status") == "launched":
                    # Clear stale launched markers when the worker exited
                    # (OOM / readonly-cache crashes) so it can be relaunched.
                    if not launched_worker_alive(job):
                        write_json(
                            state_for(job.name),
                            {"name": job.name, "status": "queued", "updated_at": now()},
                        )
                    print(f"[auto-resume] {job.name} worker died; requeued", flush=True)
                    continue
                if launched_this_pass:
                    # Only start one worker per pass: the free-GPU probe can be
                    # stale for a freshly launched process, and piling several
                    # jobs onto one card caused the earlier OOM pile-up.
                    continue
                if job.preencode_name and job.preencode_name not in ready_deps:
                    continue
                if job.mode == "semantic_flatten" and not flatten_cache_ready():
                    continue

                gpu = find_free_gpu([idx for idx in gpus if idx not in reserved])
                if gpu is None:
                    print(
                        f"[auto-resume] no free gpu for {job.name}; waiting",
                        flush=True,
                    )
                    continue
                launch(job, gpu)
                reserved.add(gpu)
                launched_this_pass = True

        except Exception:
            # A single transient failure must not kill the long-lived scheduler.
            import traceback

            traceback.print_exc()

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
