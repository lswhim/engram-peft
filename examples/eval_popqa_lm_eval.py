# pyright: reportMissingTypeStubs=none, reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from lm_eval import evaluator
from lm_eval import utils as lm_eval_utils
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from peft import PeftModel
from transformers import AutoTokenizer

from engram_peft import EngramModel
from engram_peft.utils.compat import wash_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from examples.engram_knowledge_memory import load_4bit_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PopQA with lm-evaluation-harness.")
    parser.add_argument("--model", required=True, help="Base HF model id/path.")
    parser.add_argument("--engram_path", default=None, help="Optional Engram adapter path.")
    parser.add_argument("--lora_path", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--batch_size", default="auto", help="lm-eval batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for debugging.")
    parser.add_argument("--output_path", default=None, help="Optional JSON output path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbosity", default="INFO")
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: Any = load_4bit_backbone(args.model)
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
    if args.engram_path:
        model = EngramModel.from_pretrained(
            model,
            args.engram_path,
            tokenizer=wash_tokenizer(tokenizer),
        )
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.verbosity.upper()))

    model, tokenizer = load_model(args)
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=args.batch_size,
        device=args.device,
        trust_remote_code=True,
        enable_thinking=False,
    )

    task_dir = Path(__file__).resolve().parents[1] / "lm_eval_tasks"
    task_manager = TaskManager(include_path=str(task_dir), include_defaults=False)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["popqa"],
        batch_size=args.batch_size,
        limit=args.limit,
        task_manager=task_manager,
        log_samples=False,
        confirm_run_unsafe_code=True,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no results")

    print(lm_eval_utils.make_table(results))
    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
