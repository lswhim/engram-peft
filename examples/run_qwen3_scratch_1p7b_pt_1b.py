#!/usr/bin/env python3
"""Train a strict scratch model with the Qwen3-1.7B architecture.

The checkpoint path is used only for tokenizer and architecture config.  No
pretrained parameter is loaded; all model weights are initialized by
``AutoModelForCausalLM.from_config``.
"""

from __future__ import annotations

import os
import math

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import examples.run_scratch_pt as runner
from engram_peft import EngramConfig, EngramModel, get_engram_model
from engram_peft.utils.compat import wash_tokenizer


def make_qwen3_scratch_model(args, tokenizer):
    config = AutoConfig.from_pretrained(args.tokenizer)
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_config(
        config,
        attn_implementation=getattr(args, "attn_implementation", "eager"),
    )
    # Scale residual branches for a deep randomly initialized decoder.  This is
    # the same variance-preserving initialization used by the validated scratch
    # runner; without it the first BF16 loss overflows before warmup can help.
    residual_scale = 1.0 / math.sqrt(2 * config.num_hidden_layers)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.o_proj.weight.mul_(residual_scale)
            layer.mlp.down_proj.weight.mul_(residual_scale)
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
    engram_config = EngramConfig(
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        engram_vocab_size_per_ngram=[16, 16],
        hidden_size=config.hidden_size,
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


runner.make_model = make_qwen3_scratch_model


if __name__ == "__main__":
    runner.main()
