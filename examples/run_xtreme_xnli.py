# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
"""Run full English-source XNLI training and 15-language zero-shot evaluation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from engram_peft import EngramConfig, EngramTrainer, get_engram_model
from engram_peft.utils.compat import wash_tokenizer
from examples.benchmarks.xtreme_xnli import (
    CausalLabelCollator,
    evaluate_all_languages,
    load_xnli_train_and_tests,
    tokenize_xnli_train,
)

Method = Literal["base", "full_ft", "lora", "arithmetic", "arithmetic_matched", "rq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["base", "full_ft", "lora", "arithmetic", "arithmetic_matched", "rq"],
        required=True,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--rq_table_dir")
    parser.add_argument("--rq_cache_dir")
    parser.add_argument(
        "--rq_router",
        choices=["flatten", "collision", "learned", "semantic_keyed"],
        default="flatten",
        help="RQ memory readout. Ignored by non-RQ methods.",
    )
    parser.add_argument("--output_dir", default="outputs/xtreme_xnli")
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


def _load_rq_meta(table_dir: str | None) -> dict[str, Any]:
    if not table_dir:
        raise ValueError(
            "--rq_table_dir is required for method=rq or arithmetic_matched"
        )
    with open(Path(table_dir) / "meta.json", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _engram_config(args: argparse.Namespace, model: Any) -> EngramConfig:
    method = str(args.method)
    rq_meta = (
        _load_rq_meta(args.rq_table_dir)
        if method in {"rq", "arithmetic_matched"}
        else None
    )
    num_heads = int(rq_meta["num_levels"]) if rq_meta else 8
    if method == "arithmetic_matched":
        # ``engram_vocab_size_per_ngram`` is the total capacity for an
        # n-gram order.  Arithmetic hashing divides it across all heads,
        # whereas RQ's codebook_size is the capacity of *each* head.
        # Matching RQ(M levels, K codes) therefore requires M * K total
        # arithmetic rows per n-gram order, not merely K.
        bucket_size = int(rq_meta["codebook_size"]) * num_heads
    else:
        bucket_size = int(args.arith_buckets)
    router = str(args.rq_router)
    routing_config: dict[str, Any]
    if method != "rq" or router == "flatten":
        routing_config = {"memory_fusion": "flatten"}
    elif router == "collision":
        routing_config = {
            "memory_fusion": "head_factorized",
            "head_router_selection": "specificity",
            "head_router_top_k": 4,
            "head_router_preserve_mass": True,
        }
    elif router == "learned":
        routing_config = {
            "memory_fusion": "head_factorized",
            "head_router_selection": "learned",
            "head_router_top_k": 0,
        }
    elif router == "semantic_keyed":
        routing_config = {
            "memory_fusion": "head_factorized",
            "head_router_selection": "semantic_keyed",
            "head_router_top_k": 0,
            "semantic_router_dim": 64,
        }
    else:
        raise ValueError(f"Unknown RQ router: {router}")
    return EngramConfig(
        ngram_sizes=[2, 3],
        n_head_per_ngram=num_heads,
        engram_vocab_size_per_ngram=[bucket_size, bucket_size],
        target_layers=list(args.target_layers),
        hidden_size=int(model.config.hidden_size),
        embedding_dim=int(args.engram_embedding_dim),
        enable_tokenizer_compression=True,
        tokenizer_name_or_path=args.model,
        pad_id=int(args.pad_token_id),
        learning_rate_multiplier=15.0,
        hash_backend=(
            "rq"
            if method == "rq"
            else "arithmetic_fixed"
            if method == "arithmetic_matched"
            else "arithmetic"
        ),
        rq_table_dir=args.rq_table_dir if method == "rq" else None,
        rq_cache_dir=args.rq_cache_dir if method == "rq" else None,
        seed=int(args.seed),
        use_sparse_embeddings=False,
        **routing_config,
    )


def build_model(args: argparse.Namespace) -> tuple[Any, type[Trainer]]:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    method: Method = args.method
    if method == "full_ft":
        model.requires_grad_(True)
        model.gradient_checkpointing_enable()
        return model, Trainer
    if method == "lora":
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, config)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.print_trainable_parameters()
        return model, Trainer
    if method in {"arithmetic", "arithmetic_matched", "rq"}:
        config = _engram_config(args, model)
        model = get_engram_model(model, config, wash_tokenizer(args.tokenizer))
        model.print_trainable_parameters()
        return model, EngramTrainer
    if method == "base":
        model.requires_grad_(False)
        return model, Trainer
    raise ValueError(f"Unknown method: {method}")


def _learning_rate(method: Method) -> float:
    if method == "full_ft":
        return 2e-5
    if method == "lora":
        return 2e-4
    return 3e-4


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This full benchmark requires one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Launch one independent method per GPU via CUDA_VISIBLE_DEVICES")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    args.tokenizer = tokenizer
    args.pad_token_id = tokenizer.pad_token_id

    run_name = (
        f"rq_{args.rq_router}_seed{args.seed}"
        if args.method == "rq"
        else f"{args.method}_seed{args.seed}"
    )
    run_dir = Path(args.output_dir) / run_name
    result_path = run_dir / "metrics.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_result(
        result_path,
        {"status": "loading_data", "args": vars(args) | {"tokenizer": None}},
    )

    train_raw, tests = load_xnli_train_and_tests()
    train_dataset = tokenize_xnli_train(
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
            # Transformers 4.57 clips through Accelerate before consulting
            # EngramTrainer._clip_grad_norm; torch's global implementation cannot
            # reduce SparseCUDA gradients. SparseAdam remains enabled, while the
            # unsupported outer clipping pass is disabled for Engram methods.
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
        "protocol": "MNLI English full train -> XNLI 15-language test, zero target updates",
        "method": args.method,
        "rq_router": args.rq_router if args.method == "rq" else None,
        "seed": args.seed,
        "train_examples": len(train_raw),
        "train_metrics": train_metrics,
        "languages": {},
    }
    _write_result(result_path, payload)

    def record_language(language: str, accuracy: float) -> None:
        payload["languages"][language] = accuracy
        _write_result(result_path, payload)
        print(f"[XNLI] {language}: {accuracy * 100:.2f}", flush=True)

    language_metrics = evaluate_all_languages(
        model,
        tokenizer,
        tests,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        device=device,
        on_language=record_language,
    )
    payload["languages"] = language_metrics
    payload["status"] = "complete"
    payload["wall_time_seconds"] = time.time() - started_at
    payload["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    _write_result(result_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
