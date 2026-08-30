#!/usr/bin/env python
"""Run standard lm-eval tasks for a base or completed Engram experiment.

The default task set uses the public lm-eval ``wikitext`` and ``c4`` corpus
perplexity tasks plus LAMBADA last-word prediction. An Engram checkpoint may be supplied
directly or resolved from a completed benchmark ``run_suffix``.  Result-suffix
resolution rejects partial fixed-step runs so early-stop diagnostics cannot
silently enter the paper table.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


MMLU_SUBJECTS = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer path. Defaults to --model; useful for scratch checkpoints.",
    )
    parser.add_argument(
        "--tasks",
        default="wikitext,c4,lambada_openai",
        help="Comma-separated public lm-evaluation-harness task names.",
    )
    parser.add_argument("--engram-weights", type=Path)
    parser.add_argument("--result-suffix")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Evaluation context length in tokens; matches the 1B scratch-PT setup.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        help="Optional harness limit for debugging only; omit in paper runs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def completed_checkpoint(suffix: str) -> Path:
    candidates: list[Path] = []
    for path in Path("outputs/benchmarks").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        params = payload.get("params")
        metrics = payload.get("metrics")
        if not isinstance(params, dict) or params.get("run_suffix") != suffix:
            continue
        if not isinstance(metrics, dict):
            continue
        if "_fixedsteps_" in suffix and not (
            metrics.get("fixed_steps_complete") is True
            and metrics.get("completed_steps") == metrics.get("planned_steps") == 12_208
        ):
            raise RuntimeError(f"refusing incomplete fixed-step result: {path}")
        save_dir = metrics.get("save_dir")
        if isinstance(save_dir, str) and Path(save_dir).is_dir():
            candidates.append(Path(save_dir))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one completed checkpoint for suffix={suffix!r}, found {len(candidates)}"
        )
    return candidates[0]


def numeric_results(results: dict[str, Any]) -> dict[str, dict[str, int | float]]:
    """Keep only scalar task metrics, excluding stderr aliases and metadata."""
    selected: dict[str, dict[str, int | float]] = {}
    for task, values in results.items():
        if not isinstance(values, dict):
            continue
        selected[str(task)] = {
            str(name): value
            for name, value in values.items()
            if isinstance(value, int | float) and "_stderr" not in str(name)
        }
    return selected


def patch_datasets_feature_alias() -> None:
    """Read cached dataset metadata written by newer datasets releases."""
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    try:
        from datasets.features.features import _FEATURE_TYPES, Sequence
    except ImportError:
        return
    _FEATURE_TYPES.setdefault("List", Sequence)


def normalize_tasks(raw_tasks: str) -> list[str]:
    tasks = [value.strip() for value in raw_tasks.split(",") if value.strip()]
    expanded = [f"mmlu_{subject}" for subject in MMLU_SUBJECTS]
    return [task for value in tasks for task in (expanded if value == "mmlu" else [value])]


def main() -> None:
    args = parse_args()

    import torch
    patch_datasets_feature_alias()
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.engram_weights is not None and args.result_suffix:
        raise ValueError("use only one of --engram-weights and --result-suffix")
    if args.result_suffix:
        args.engram_weights = completed_checkpoint(args.result_suffix)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    model: Any = base
    if args.engram_weights is not None:
        from engram_peft import EngramModel

        model = EngramModel.from_pretrained(
            base, str(args.engram_weights), tokenizer=tokenizer
        ).cuda().eval()
        print(f"[standard-lm] loaded {args.engram_weights}", flush=True)

    tasks = normalize_tasks(args.tasks)
    harness = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    raw = simple_evaluate(
        model=harness,
        tasks=tasks,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        log_samples=False,
    )
    task_results = numeric_results(raw.get("results", {}))
    payload = {
        "status": "complete",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": args.model,
        "method": args.method,
        "seed": args.seed,
        "engram_weights": str(args.engram_weights) if args.engram_weights else None,
        "result_suffix": args.result_suffix,
        "tasks": tasks,
        "limit": args.limit,
        "paper_eligible": args.limit is None,
        "results": task_results,
        "versions": raw.get("versions", {}),
        "n-shot": raw.get("n-shot", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
