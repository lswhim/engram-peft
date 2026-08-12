# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
"""Train on full English PAWS-X and evaluate all seven languages zero-shot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from examples.benchmarks.xtreme_pawsx import (
    CausalLabelCollator,
    evaluate_all_languages,
    load_pawsx_train_and_tests,
    tokenize_pawsx_train,
)
from examples.run_xtreme_xnli import _learning_rate, _write_result, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["base", "full_ft", "lora", "arithmetic", "arithmetic_matched", "rq"],
        required=True,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--rq_table_dir")
    parser.add_argument("--output_dir", default="outputs/xtreme_pawsx")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_layers", type=int, nargs="+", default=[11, 21])
    parser.add_argument("--engram_embedding_dim", type=int, default=1280)
    parser.add_argument("--arith_buckets", type=int, default=512000)
    parser.add_argument("--logging_steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Launch one PAWS-X method per visible CUDA GPU")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    args.tokenizer = tokenizer
    args.pad_token_id = tokenizer.pad_token_id

    run_dir = Path(args.output_dir) / f"{args.method}_seed{args.seed}"
    result_path = run_dir / "metrics.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_result(
        result_path,
        {"status": "loading_data", "args": vars(args) | {"tokenizer": None}},
    )

    train_raw, tests = load_pawsx_train_and_tests()
    train_dataset = tokenize_pawsx_train(
        train_raw, tokenizer, args.max_length, args.num_workers
    )
    model, trainer_class = build_model(args)
    device = torch.device("cuda")
    model.to(device)
    started_at = time.time()
    train_metrics: dict[str, Any] = {}
    if args.method != "base":
        training_args = TrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=_learning_rate(args.method),
            warmup_ratio=0.06,
            lr_scheduler_type="cosine",
            logging_steps=args.logging_steps,
            save_strategy="epoch",
            save_total_limit=1,
            eval_strategy="no",
            report_to="none",
            bf16=True,
            tf32=True,
            seed=args.seed,
            data_seed=args.seed,
            dataloader_num_workers=args.num_workers,
            remove_unused_columns=False,
            gradient_checkpointing=args.method in {"full_ft", "lora"},
            max_grad_norm=(
                0.0
                if args.method in {"arithmetic", "arithmetic_matched", "rq"}
                else 1.0
            ),
        )
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=CausalLabelCollator(tokenizer),
        )
        result = trainer.train()
        train_metrics = {
            key: float(value)
            for key, value in result.metrics.items()
            if isinstance(value, int | float)
        }
        model.save_pretrained(run_dir / "final_model")
        tokenizer.save_pretrained(run_dir / "final_model")

    payload: dict[str, Any] = {
        "status": "evaluating",
        "protocol": "PAWS-X English full train -> 7-language test, zero target updates",
        "method": args.method,
        "seed": args.seed,
        "train_examples": len(train_raw),
        "train_metrics": train_metrics,
        "languages": {},
    }
    _write_result(result_path, payload)

    def record_language(language: str, accuracy: float) -> None:
        payload["languages"][language] = accuracy
        _write_result(result_path, payload)
        print(f"[PAWS-X] {language}: {accuracy * 100:.2f}", flush=True)

    payload["languages"] = evaluate_all_languages(
        model,
        tokenizer,
        tests,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        device=device,
        on_language=record_language,
    )
    payload["status"] = "complete"
    payload["wall_time_seconds"] = time.time() - started_at
    payload["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    _write_result(result_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
