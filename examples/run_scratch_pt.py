#!/usr/bin/env python3
"""Train a matched scratch decoder and/or full-finetune Engram model.

The runner intentionally does not load pretrained weights.  A Qwen3 base
checkpoint is used only for its tokenizer and model-family configuration.
Launch distributed runs with ``torchrun --nproc_per_node=8``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Qwen3Config,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.modeling_utils import unwrap_model

from engram_peft import EngramConfig, EngramModel, get_engram_model
from engram_peft.trainer import EngramTrainer
from engram_peft.utils.compat import wash_tokenizer


class PackedTokenDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        path: Path,
        sequence_length: int,
        max_examples: int | None = None,
    ) -> None:
        self.tokens = np.memmap(path, mode="r", dtype=np.uint32)
        self.sequence_length = sequence_length
        available_examples = (len(self.tokens) - 1) // sequence_length
        self.example_count = (
            min(available_examples, max_examples)
            if max_examples is not None
            else available_examples
        )
        if self.example_count <= 0:
            raise ValueError(f"not enough tokens in {path}")

    def __len__(self) -> int:
        return self.example_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.sequence_length
        raw = np.asarray(self.tokens[start : start + self.sequence_length + 1])
        values = torch.from_numpy(raw.astype(np.int64, copy=True))
        return {"input_ids": values[:-1], "labels": values[1:]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("base", "arithmetic", "semantic_flatten", "semantic_keyed"),
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rq-table-dir", type=Path, default=None)
    parser.add_argument("--rq-cache-dir", type=Path, default=None)
    parser.add_argument("--train-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--optimizer",
        choices=("adamw_torch", "adamw_torch_fused"),
        default="adamw_torch",
        help="Explicitly select the HF Trainer AdamW backend.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="eager",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable BF16 autocast for numerical-stability diagnostics.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable activation checkpointing for stability diagnostics.",
    )
    parser.add_argument(
        "--debug-gradients",
        action="store_true",
        help="Print non-finite gradients before the first optimizer step.",
    )
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Enable PyTorch autograd anomaly detection for diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-steps", type=int, default=95)
    parser.add_argument("--checkpoint-steps", type=int, default=239)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_scratch_config(tokenizer: Any) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        initializer_range=0.01,
        hidden_size=1536,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=12,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=32768,
        tie_word_embeddings=False,
        use_cache=False,
    )
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    return config


def make_model(args: argparse.Namespace, tokenizer: Any) -> torch.nn.Module:
    model_config = make_scratch_config(tokenizer)
    model = torch.nn.Module()  # replaced immediately; keeps type checkers quiet
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_config(
        model_config,
        attn_implementation=getattr(args, "attn_implementation", "eager"),
    )
    # Keep the random scratch decoder stable across its 24 residual blocks.
    residual_scale = 1.0 / math.sqrt(2 * model_config.num_hidden_layers)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.o_proj.weight.mul_(residual_scale)
            layer.mlp.down_proj.weight.mul_(residual_scale)
    # Keep master weights in FP32; Trainer's BF16 autocast handles the matmuls.
    # Casting a randomly initialized billion-parameter model to BF16 before the
    # optimizer step can make the short warmup numerically unstable.
    if args.mode == "base":
        return model

    semantic_mode = args.mode in {"semantic_flatten", "semantic_keyed"}
    if semantic_mode:
        if args.rq_table_dir is None or args.rq_cache_dir is None:
            raise ValueError(
                f"{args.mode} requires --rq-table-dir and --rq-cache-dir"
            )
        hash_backend = "rq"
    else:
        hash_backend = "arithmetic_fixed"

    rank = int(os.environ.get("RANK", "0"))
    cache_dir = None
    if semantic_mode:
        # Each DDP rank owns its SQLite writer; sharing one live SQLite cache is
        # a reliable source of lock failures during first-seen n-gram encoding.
        cache_dir = args.rq_cache_dir / f"rank{rank}"

    engram_config = EngramConfig(
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        engram_vocab_size_per_ngram=[16, 16],
        hidden_size=model_config.hidden_size,
        embedding_dim=1280,
        target_layers=[9, 18],
        enable_tokenizer_compression=True,
        tokenizer_name_or_path=args.tokenizer,
        pad_id=tokenizer.pad_token_id,
        hash_backend=hash_backend,
        rq_table_dir=str(args.rq_table_dir) if args.rq_table_dir else None,
        rq_cache_dir=str(cache_dir) if cache_dir else None,
        memory_fusion=(
            "head_factorized" if args.mode == "semantic_keyed" else "flatten"
        ),
        head_router_selection="semantic_keyed",
        use_sparse_embeddings=False,
        engram_dtype="float32" if args.fp32 else "bfloat16",
        backbone_freeze_steps=0,
        entropy_loss_weight=0.0,
        clip_grad_per_group=True,
    )
    wrapped = get_engram_model(
        model,
        engram_config,
        wash_tokenizer(tokenizer),
        train_mode="full_finetune",
    )
    return wrapped


class ScratchCheckpointCallback(TrainerCallback):
    def __init__(self, every_steps: int, output_dir: Path) -> None:
        self.every_steps = every_steps
        self.output_dir = output_dir
        self.saved: set[int] = set()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args
        step = int(state.global_step)
        if step == 0 or step % self.every_steps != 0 or step in self.saved:
            return control
        self.saved.add(step)
        if not state.is_world_process_zero:
            return control
        model = unwrap_model(kwargs["model"])
        checkpoint = self.output_dir / f"checkpoint-{step:05d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        if isinstance(model, EngramModel):
            model.base_model.save_pretrained(checkpoint / "base_model", safe_serialization=True)
            model.save_pretrained(checkpoint / "engram", safe_serialization=True)
        else:
            model.save_pretrained(checkpoint, safe_serialization=True)
        print(f"[scratch checkpoint] step={step} path={checkpoint}", flush=True)
        return control


class GradientDebugCallback(TrainerCallback):
    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args
        if not kwargs.get("model") or state.global_step > 0:
            return control
        model = unwrap_model(kwargs["model"])
        bad = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                bad.append(
                    {
                        "name": name,
                        "finite": int(torch.isfinite(parameter.grad).sum()),
                        "total": parameter.grad.numel(),
                        "max_abs": float(torch.nan_to_num(parameter.grad.detach()).abs().max()),
                    }
                )
                if len(bad) >= 8:
                    break
        print(f"[gradient debug] nonfinite_gradients={len(bad)} examples={bad}", flush=True)
        return control


def save_final(model: torch.nn.Module, output_dir: Path) -> None:
    model = unwrap_model(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, EngramModel):
        model.base_model.save_pretrained(output_dir / "base_model", safe_serialization=True)
        model.save_pretrained(output_dir / "engram", safe_serialization=True)
    else:
        model.save_pretrained(output_dir, safe_serialization=True)


def main() -> None:
    args = parse_args()
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tokens_per_step = (
        world_size
        * args.per_device_batch_size
        * args.gradient_accumulation_steps
        * args.sequence_length
    )
    max_steps = args.max_steps or math.ceil(args.train_tokens / tokens_per_step)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = PackedTokenDataset(args.data_dir / "train.bin", args.sequence_length)
    eval_dataset = PackedTokenDataset(args.data_dir / "eval.bin", args.sequence_length)
    eval_batch_size = world_size * args.per_device_batch_size
    eval_dataset.example_count -= eval_dataset.example_count % eval_batch_size
    if eval_dataset.example_count <= 0:
        raise ValueError("eval stream is too short for one complete distributed batch")
    model = make_model(args, tokenizer)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"mode={args.mode} world_size={world_size} params={parameter_count:,} "
        f"tokens_per_step={tokens_per_step:,} max_steps={max_steps}",
        flush=True,
    )
    if args.dry_run:
        return

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "hf_tmp"),
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        optim=args.optimizer,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.1,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        report_to="none",
        bf16=not args.fp32,
        tf32=not args.fp32,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    callbacks: list[TrainerCallback] = [
        ScratchCheckpointCallback(args.checkpoint_steps, args.output_dir)
    ]
    if args.debug_gradients:
        callbacks.append(GradientDebugCallback())
    if args.mode == "base":
        trainer: Trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            callbacks=callbacks,
        )
    else:
        assert isinstance(model, EngramModel)
        trainer = EngramTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            callbacks=callbacks,
            optimizer_kwargs={
                "backbone_learning_rate": args.learning_rate,
                "engram_dense_learning_rate": args.learning_rate,
                "engram_sparse_learning_rate": args.learning_rate,
                "backbone_optimizer": torch.optim.AdamW,
                "engram_dense_optimizer": "adamw",
                "engram_sparse_optimizer": "adamw",
            },
        )

    result = trainer.train()
    metrics = trainer.evaluate()
    metrics["train_steps"] = int(trainer.state.global_step)
    metrics["tokens_per_step"] = tokens_per_step
    metrics["trained_token_slots"] = int(trainer.state.global_step * tokens_per_step)
    metrics["parameter_count"] = parameter_count
    if trainer.is_world_process_zero():
        save_final(model, args.output_dir / "final")
        (args.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=True, default=float) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=True, default=float), flush=True)
    del result


if __name__ == "__main__":
    main()
