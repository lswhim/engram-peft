#!/usr/bin/env python3
"""Continue-pretrain Qwen3-1.7B-Base on the packed FineWeb stream.

This reuses the validated distributed runner for token accounting, checkpointing,
and evaluation, but replaces its random scratch initialization with the actual
pretrained Qwen3 checkpoint.  ``--tokenizer`` is intentionally the model path:
Qwen3's tokenizer and model are co-located in the checkpoint directory.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM

import examples.run_scratch_pt as runner
from engram_peft import EngramConfig, EngramModel, get_engram_model
from engram_peft.utils.compat import wash_tokenizer


def make_pretrained_model(args, tokenizer):
    dtype = (
        torch.float32
        if args.fp32
        else (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.tokenizer,
        dtype=dtype,
        attn_implementation=getattr(args, "attn_implementation", "eager"),
    )
    if args.mode == "base":
        return model

    semantic_mode = args.mode in {"semantic_flatten", "semantic_keyed"}
    if semantic_mode:
        if args.rq_table_dir is None or args.rq_cache_dir is None:
            raise ValueError(f"{args.mode} requires --rq-table-dir and --rq-cache-dir")
        hash_backend = "rq"
    else:
        hash_backend = "arithmetic_fixed"

    rank = int(os.environ.get("RANK", "0"))
    cache_dir = args.rq_cache_dir / f"rank{rank}" if semantic_mode else None
    model_config = model.config
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
    if not isinstance(wrapped, EngramModel):
        raise TypeError("expected get_engram_model to return EngramModel")
    return wrapped


runner.make_model = make_pretrained_model


if __name__ == "__main__":
    runner.main()
