# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none, reportUnknownLambdaType=none, reportMissingTypeStubs=none, reportAssignmentType=none, reportArgumentType=none
"""
PopQA Knowledge Memorization Benchmark with Engram-PEFT.

Evaluates Engram's internal RAG capability on long-tail factual knowledge
(PopQA), comparing Exact Match (EM) accuracy across configurations:
  - Base model (no adapter)
  - + Engram adapter
  - + LoRA adapter
  - + Engram + LoRA (combined)

Protocol:
  1. PopQA 80/20 random split
  2. Train adapter(s) on held-in 80%
  3. Evaluate EM (generation, greedy) on held-out 20%
  4. Compare all available configurations in a table

Usage:
  # Train Engram only
  python examples/engram_knowledge_memory.py

  # Train LoRA only
  python examples/engram_knowledge_memory.py --lora

  # Joint train Engram + LoRA together
  python examples/engram_knowledge_memory.py --joint

  # Train all three: Engram-only, LoRA-only, and Joint
  python examples/engram_knowledge_memory.py --lora --joint

  # Evaluate only (load saved adapters)
  python examples/engram_knowledge_memory.py --mode eval \\
    --engram_path outputs/popqa_benchmark/engram \\
    --lora_path outputs/popqa_benchmark/lora

  # Distributed training (DDP, 8 GPUs)
  torchrun --nproc_per_node=8 examples/engram_knowledge_memory.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from typing import Any, cast

import torch
from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramModel,
    EngramTrainer,
    get_engram_model,
)
from engram_peft.utils.compat import wash_tokenizer

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_OUTPUT_DIR = "outputs/popqa_benchmark"
DEFAULT_SEED = 42
TRAIN_RATIO = 0.8


def normalize_backend_name(name: str) -> str:
    name = name.strip().lower()
    if name in {"arith", "arithmetic"}:
        return "arithmetic"
    if name in {"rq", "mixed", "mixed_v2"}:
        return name
    raise ValueError(f"Unsupported hash backend: {name}")


def backend_tag_for_output(backend: str, rq_table_dir: str | None) -> str:
    backend = normalize_backend_name(backend)
    if backend == "arithmetic":
        return "arith"
    if backend == "rq":
        suffix = os.path.basename(os.path.normpath(rq_table_dir)) if rq_table_dir else "rq"
        return f"rq_{suffix}"
    if backend == "mixed":
        suffix = os.path.basename(os.path.normpath(rq_table_dir)) if rq_table_dir else "mixed"
        return f"mixed_{suffix}"
    return f"{backend}_{os.path.basename(os.path.normpath(rq_table_dir)) if rq_table_dir else backend}"


def engram_output_dir(
    base_output_dir: str, backend: str, rq_table_dir: str | None
) -> str:
    tag = backend_tag_for_output(backend, rq_table_dir)
    if backend == "arithmetic":
        return os.path.join(base_output_dir, "engram")
    return os.path.join(base_output_dir, f"engram_{tag}")


def build_engram_config(
    args: argparse.Namespace,
    backend: str,
    sparse_embeddings: bool,
) -> EngramConfig:
    normalized_backend = normalize_backend_name(backend)
    if normalized_backend in {"rq", "mixed", "mixed_v2"} and not args.rq_table_dir:
        raise ValueError(
            f"backend={normalized_backend} requires --rq_table_dir"
        )

    return EngramConfig(
        embedding_dim=args.embedding_dim,
        target_layers=args.target_layers,
        use_sparse_embeddings=sparse_embeddings,
        entropy_loss_weight=args.entropy_loss_weight if not args.use_deepspeed else 0.0,
        hash_backend=normalized_backend,
        rq_table_dir=args.rq_table_dir if normalized_backend != "arithmetic" else None,
        n_rq_levels_used=args.n_rq_levels_used,
        n_arith_heads_per_ngram=args.n_arith_heads_per_ngram,
    )


# ──────────────────────────────────────────────────────────────────────
# Answer Normalization (Exact Match)
# ──────────────────────────────────────────────────────────────────────


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ──────────────────────────────────────────────────────────────────────
# Data Loading (PopQA)
# ──────────────────────────────────────────────────────────────────────


def format_qa_text(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def parse_possible_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v)]
        if parsed is None:
            return []
        return [str(parsed)]
    return [str(value)]


def load_popqa(
    max_samples: int | None = None,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[Dataset, Dataset]:
    """Load PopQA, return (train_dataset, test_dataset).

    PopQA only has a ``test`` split on HF (14,267 rows), so we do an
    80/20 random split ourselves.  Each row is kept raw (with
    ``question`` and ``possible_answers`` columns) for later EM eval.
    A ``text`` column is added for training.
    """
    raw: Dataset = load_dataset("akariasai/PopQA", split="test")
    if max_samples is not None and max_samples < len(raw):
        raw = raw.select(range(max_samples))

    def add_text(example: dict[str, Any]) -> dict[str, Any]:
        possible_answers = parse_possible_answers(example.get("possible_answers"))
        answer = possible_answers[0] if possible_answers else ""
        example["possible_answers"] = possible_answers
        example["text"] = format_qa_text(example["question"], answer)
        return example

    raw = raw.map(add_text)

    split_result: DatasetDict = raw.train_test_split(
        test_size=1 - train_ratio, seed=DEFAULT_SEED
    )
    return split_result["train"], split_result["test"]


# ──────────────────────────────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────────────────────────────


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
    num_proc: int = 4,
) -> Dataset:
    """Tokenize ``text`` column and produce causal-LM labels."""

    def tokenize_fn(examples: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )


# ──────────────────────────────────────────────────────────────────────
# DeepSpeed Config
# ──────────────────────────────────────────────────────────────────────


def write_default_ds_config(output_dir: str) -> str:
    config = {
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "gradient_clipping": 1.0,
        "bf16": {"enabled": "auto"},
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
    }
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "ds_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


# ──────────────────────────────────────────────────────────────────────
# 4-bit Backbone Loading
# ──────────────────────────────────────────────────────────────────────


def load_4bit_backbone(
    model_id: str,
    use_deepspeed: bool = False,
) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    if use_deepspeed:
        device_map: str | None = None
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    return cast(
        "AutoModelForCausalLM",
        AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# Model Construction — Engram
# ──────────────────────────────────────────────────────────────────────


def build_engram_model(
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    use_deepspeed: bool,
    engram_config: EngramConfig,
) -> EngramModel:
    base_model = load_4bit_backbone(model_id, use_deepspeed)
    if not use_deepspeed:
        base_model = prepare_model_for_kbit_training(base_model)
    model = get_engram_model(
        base_model,
        engram_config,
        tokenizer=wash_tokenizer(tokenizer),
        train_mode="engram_only",
    )
    model.print_trainable_parameters()
    return model


def build_lora_model(
    model_id: str,
    r: int = 16,
    lora_alpha: int = 32,
) -> PeftModel:
    base_model = load_4bit_backbone(model_id)
    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return cast("PeftModel", model)


def build_joint_model(
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    use_deepspeed: bool,
    engram_config: EngramConfig,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> EngramModel:
    base_model = load_4bit_backbone(model_id, use_deepspeed)
    if not use_deepspeed:
        base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_config)
    model = get_engram_model(
        base_model,
        engram_config,
        tokenizer=wash_tokenizer(tokenizer),
        train_mode="preserve_trainable",
    )
    model.print_trainable_parameters()
    return model


# ──────────────────────────────────────────────────────────────────────
# Exact Match Evaluation
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_em(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    test_dataset: Dataset,
    max_samples: int = 200,
    max_new_tokens: int = 32,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Compute Exact Match accuracy on a held-out PopQA test set.

    Greedy-decode batched prompts of the form ``"Question: {q}\\nAnswer:"``,
    extract the first generated line, normalize, compare with all possible answers.

    Returns:
        ``{"correct": int, "total": int, "accuracy": float}``
    """
    device = model.device
    total = min(max_samples, len(test_dataset))
    correct = 0

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    model.eval()
    batch_size = max(1, batch_size)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = [test_dataset[i] for i in range(start, end)]
        prompts = [f"Question: {row['question']}\nAnswer:" for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        for offset, row in enumerate(batch):
            response = cast(
                "str",
                tokenizer.decode(
                    outputs[offset][input_len:],
                    skip_special_tokens=True,
                ),
            )
            pred = response.split("\n")[0].strip()
            pred_norm = normalize_answer(pred)

            possible: list[str] = row["possible_answers"]
            if any(normalize_answer(a) == pred_norm for a in possible):
                correct += 1

        done = end
        if done % 50 == 0 or done == total:
            logging.info(
                "  EM progress: %d/%d  (acc so far: %.1f%%)",
                done,
                total,
                correct / done * 100,
            )

    tokenizer.padding_side = old_padding_side

    accuracy = correct / total * 100
    return {"correct": correct, "total": total, "accuracy": accuracy}


# ──────────────────────────────────────────────────────────────────────
# Comparison Table
# ──────────────────────────────────────────────────────────────────────


def print_comparison(results: dict[str, dict[str, Any]]) -> None:
    if not results:
        return

    base_acc = results.get("Base", {}).get("accuracy", 0.0)

    logging.info("")
    logging.info("=" * 60)
    logging.info("  PopQA Benchmark Results")
    logging.info("=" * 60)
    logging.info("  %-25s %10s %10s", "Config", "Accuracy", "Δ vs Base")
    logging.info("  " + "-" * 47)
    for name, res in results.items():
        acc = res.get("accuracy", 0.0)
        delta = acc - base_acc
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        if name == "Base":
            delta_str = "—"
        logging.info("  %-25s %8.1f%% %10s", name, acc, delta_str)
    logging.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# Arg Parsing
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PopQA Knowledge Memorization Benchmark with Engram-PEFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Train Engram only
              python examples/engram_knowledge_memory.py

              # Train LoRA only
              python examples/engram_knowledge_memory.py --lora

              # Joint train Engram + LoRA
              python examples/engram_knowledge_memory.py --joint

              # Train all three (Engram + LoRA + Joint)
              python examples/engram_knowledge_memory.py --lora --joint

              # Evaluate only (load saved adapters)
              python examples/engram_knowledge_memory.py --mode eval \\
                  --engram_path outputs/popqa_benchmark/engram \\
                  --lora_path outputs/popqa_benchmark/lora

              # Compare arithmetic and RQ backends in one run (need --rq_table_dir)
              python examples/engram_knowledge_memory.py \\
                  --backends arith rq \\
                  --rq_table_dir /path/to/rq_tables/biomed_qwen3_06b \\
                  --lora

              # Distributed training
              torchrun --nproc_per_node=8 examples/engram_knowledge_memory.py
        """),
    )

    # Mode
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    # Model
    p.add_argument("--model", default=DEFAULT_MODEL)
    # Data
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap PopQA samples (None = all 14,267)",
    )
    p.add_argument(
        "--eval_max_samples",
        type=int,
        default=200,
        help="Cap evaluation samples (0 = all)",
    )
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size for greedy EM generation.",
    )
    # Distributed
    p.add_argument("--use_deepspeed", action="store_true")
    # Training
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # Adapters
    p.add_argument(
        "--engram", action="store_true", default=True, help="Train Engram-only"
    )
    p.add_argument("--no-engram", action="store_false", dest="engram")
    p.add_argument("--lora", action="store_true", default=False, help="Train LoRA-only")
    p.add_argument(
        "--joint", action="store_true", default=False, help="Joint train Engram + LoRA"
    )
    p.add_argument(
        "--engram_path",
        type=str,
        default=None,
        help="Path to saved Engram adapter (eval mode or post-train load)",
    )
    p.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to saved LoRA adapter (eval mode or post-train load)",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    # Engram
    p.add_argument("--embedding_dim", type=int, default=1280)
    p.add_argument("--target_layers", type=int, nargs="+", default=[1, 14])
    p.add_argument("--entropy_loss_weight", type=float, default=0.01)
    p.add_argument(
        "--backends",
        nargs="+",
        default=["arithmetic"],
        choices=["arith", "arithmetic", "rq", "mixed", "mixed_v2"],
        help="Run Engram with one or more hash backends: arith/rq/mixed/mixed_v2.",
    )
    p.add_argument("--rq_table_dir", type=str, default=None)
    p.add_argument("--n_rq_levels_used", type=int, default=4)
    p.add_argument("--n_arith_heads_per_ngram", type=int, default=4)
    p.add_argument(
        "--disable_sparse_embeddings",
        action="store_true",
        help="Disable sparse embeddings to avoid sparse-CUDA grad norm issues in single-GPU runs.",
    )
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Trainer max grad norm (pass 0 to disable clipping).",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # ── Distributed detection ─────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", local_rank))
    is_main = global_rank <= 0
    use_deepspeed = args.use_deepspeed

    # ── Inject 'rank' into all log records so the format string can use %(rank)s ──
    _log_rank = global_rank

    def _log_record_factory(*args, **kwargs):
        record = logging.LogRecord(*args, **kwargs)
        record.rank = _log_rank
        return record

    if local_rank >= 0:
        logging.setLogRecordFactory(_log_record_factory)

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="[%(levelname)s|rank=%(rank)s] %(message)s"
        if local_rank >= 0
        else "[%(levelname)s] %(message)s",
    )

    if is_main:
        logging.info("=" * 60)
        logging.info("PopQA Knowledge Memorization Benchmark")
        logging.info("  Mode:       %s", args.mode)
        logging.info("  Model:      %s", args.model)
        logging.info("  GPUs:       %d", world_size)
        logging.info(
            "  Backend:    %s", "DeepSpeed ZeRO-2" if use_deepspeed else "DDP + sparse"
        )
        logging.info("=" * 60)

    set_seed(args.seed)

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Data ──────────────────────────────────────────────────────
    train_raw, test_raw = load_popqa(max_samples=args.max_samples)

    if is_main:
        logging.info("PopQA: %d train / %d test", len(train_raw), len(test_raw))

    # ── Compute paths ─────────────────────────────────────────────
    requested_backends = [normalize_backend_name(b) for b in args.backends]
    requested_backends = list(dict.fromkeys(requested_backends))
    if args.mode == "eval" and len(requested_backends) != 1:
        raise ValueError("Eval mode supports a single --backends entry for a single --engram_path.")

    if args.mode == "train":
        engram_paths = {
            backend: engram_output_dir(
                args.output_dir, backend, args.rq_table_dir
            )
            for backend in requested_backends
        }
    else:
        engram_paths = {requested_backends[0]: args.engram_path}

    lora_save_path = (
        os.path.join(args.output_dir, "lora")
        if args.mode == "train" and (args.lora or args.joint)
        else args.lora_path
    )

    if is_main:
        logging.info("  Engram backend(s): %s", ", ".join(requested_backends))
        logging.info("  Backend save paths: %s", engram_paths)

    eval_max = args.eval_max_samples if args.eval_max_samples > 0 else len(test_raw)
    results: dict[str, dict[str, Any]] = {}

    def run_eval_step(step: tuple[str, ...]) -> None:
        if not is_main:
            return

        name = step[0]
        if name in results:
            logging.info("Skipping %s; already evaluated.", name)
            return

        logging.info("Evaluating %s with lm-eval on %d held-out PopQA samples...", name, eval_max)
        cmd = [
            sys.executable,
            "examples/eval_popqa_lm_eval.py",
            "--model",
            args.model,
            "--batch_size",
            str(args.eval_batch_size),
        ]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "eval"
        out_path = os.path.join(args.output_dir, "lm_eval", f"{safe_name}.json")
        cmd.extend(["--output_path", out_path])
        if args.eval_max_samples > 0:
            cmd.extend(["--limit", str(args.eval_max_samples)])
        if "lora" in step:
            if not lora_save_path or not os.path.isdir(lora_save_path):
                logging.info("Skipping %s; LoRA adapter is not available.", name)
                return
            cmd.extend(["--lora_path", lora_save_path])
        if "engram" in step:
            engram_path = step[step.index("engram") + 1]
            if not engram_path or not os.path.isdir(engram_path):
                logging.info("Skipping %s; Engram adapter is not available.", name)
                return
            cmd.extend(["--engram_path", engram_path])

        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        subprocess.run(cmd, check=True, env=env)
        with open(out_path, encoding="utf-8") as f:
            lm_eval_results = json.load(f)
        popqa_result = lm_eval_results["results"]["popqa"]
        em = next(
            float(value)
            for key, value in popqa_result.items()
            if key == "em" or key.startswith("em,")
        )
        results[name] = {"accuracy": em * 100.0}
        logging.info("  %s lm-eval EM: %.2f%%", name, results[name]["accuracy"])

    def current_eval_steps() -> list[tuple[str, ...]]:
        eval_steps: list[tuple[str, ...]] = [("Base",)]
        engram_available = {
            backend: path
            for backend, path in engram_paths.items()
            if path and os.path.isdir(path)
        }
        lora_available = bool(lora_save_path and os.path.isdir(lora_save_path))

        for backend, path in engram_available.items():
            tag = backend_tag_for_output(backend, args.rq_table_dir)
            eval_steps.append((f"+Engram({tag})", "engram", path))
        if lora_available:
            eval_steps.append(("+LoRA", "lora"))
            for backend, path in engram_available.items():
                tag = backend_tag_for_output(backend, args.rq_table_dir)
                eval_steps.append((f"Joint({tag}+LoRA)", "engram", path, "lora"))
        return eval_steps

    # ═══════════════════════════════════════════════════════════════
    #  TRAINING
    # ═══════════════════════════════════════════════════════════════
    if args.mode == "train":
        # ── Engram config ────────────────────────────────────────
        sparse_embeddings = (
            not args.disable_sparse_embeddings
            and not (use_deepspeed or int(os.environ.get("WORLD_SIZE", "1")) > 1)
        )
        if use_deepspeed and args.entropy_loss_weight > 0:
            logging.warning(
                "DeepSpeed disables MixedOptimizer; entropy loss may not be applied."
            )

        deepspeed_config: str | None = None
        if use_deepspeed:
            deepspeed_config = write_default_ds_config(args.output_dir)

        # ── Engram-only ──────────────────────────────────────────
        if args.engram:
            logging.info(">>> Training Engram-only...")
            for backend in requested_backends:
                engram_save_path = engram_paths[backend]
                backend_config = build_engram_config(args, backend, sparse_embeddings)
                logging.info(
                    "  -> Engram backend=%s, tag=%s",
                    backend,
                    backend_tag_for_output(backend, args.rq_table_dir),
                )
                model = build_engram_model(
                    args.model, tokenizer, use_deepspeed, backend_config
                )
                train_tokenized = tokenize_dataset(
                    train_raw, tokenizer, max_length=args.max_length
                )
                trainer = EngramTrainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=os.path.join(
                            args.output_dir,
                            f"tmp_engram_{backend_tag_for_output(backend, args.rq_table_dir)}",
                        ),
                        per_device_train_batch_size=args.batch_size,
                        gradient_accumulation_steps=args.grad_accum,
                        learning_rate=args.learning_rate,
                        num_train_epochs=args.num_epochs,
                        max_steps=args.max_steps if args.max_steps > 0 else -1,
                        max_grad_norm=args.max_grad_norm,
                        warmup_ratio=args.warmup_ratio,
                        logging_steps=args.logging_steps,
                        save_steps=0,
                        save_total_limit=0,
                        eval_strategy="no",
                        bf16=True,
                        deepspeed=deepspeed_config,
                        ddp_find_unused_parameters=False,
                        dataloader_num_workers=2,
                        seed=args.seed,
                        report_to="none",
                        remove_unused_columns=False,
                    ),
                    train_dataset=train_tokenized,
                    data_collator=EngramDataCollator(
                        tokenizer=tokenizer, config=model.config, mlm=False
                    ),
                )
                trainer.train()
                if is_main:
                    os.makedirs(engram_save_path, exist_ok=True)
                    try:
                        saved = trainer.accelerator.unwrap_model(trainer.model)
                        saved.save_pretrained(engram_save_path)
                    except Exception:
                        model.save_pretrained(engram_save_path)
                    logging.info("Engram adapter saved to %s", engram_save_path)
                del model, trainer, train_tokenized
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if is_main:
                    tag = backend_tag_for_output(backend, args.rq_table_dir)
                    logging.info(">>> Immediate eval after Engram(%s) training", tag)
                    run_eval_step((f"+Engram({tag})", "engram", engram_save_path))

        # ── LoRA-only ────────────────────────────────────────────
        if args.lora:
            logging.info(">>> Training LoRA-only...")
            lora_model = build_lora_model(
                args.model, r=args.lora_r, lora_alpha=args.lora_alpha
            )
            train_tokenized = tokenize_dataset(
                train_raw, tokenizer, max_length=args.max_length
            )
            lora_trainer = Trainer(
                model=lora_model,
                args=TrainingArguments(
                    output_dir=args.output_dir,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    learning_rate=args.learning_rate,
                    num_train_epochs=args.num_epochs,
                    max_steps=args.max_steps if args.max_steps > 0 else -1,
                    max_grad_norm=args.max_grad_norm,
                    warmup_ratio=args.warmup_ratio,
                    logging_steps=args.logging_steps,
                    save_steps=0,
                    save_total_limit=0,
                    eval_strategy="no",
                    bf16=True,
                    deepspeed=deepspeed_config,
                    ddp_find_unused_parameters=False,
                    dataloader_num_workers=2,
                    seed=args.seed,
                    report_to="none",
                    remove_unused_columns=False,
                ),
                train_dataset=train_tokenized,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=False
                ),
            )
            lora_trainer.train()
            if is_main:
                os.makedirs(lora_save_path, exist_ok=True)
                lora_model.save_pretrained(lora_save_path)
                logging.info("LoRA adapter saved to %s", lora_save_path)
            del lora_model, lora_trainer, train_tokenized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if is_main:
                logging.info(">>> Immediate eval after LoRA training")
                run_eval_step(("+LoRA", "lora"))
                for backend, engram_path in engram_paths.items():
                    if not engram_path or not os.path.isdir(engram_path):
                        continue
                    tag = backend_tag_for_output(backend, args.rq_table_dir)
                    run_eval_step((f"Joint({tag}+LoRA)", "engram", engram_path, "lora"))

        # ── Joint (Engram + LoRA) ────────────────────────────────
        if args.joint:
            logging.info(">>> Joint training Engram + LoRA...")
            for backend in requested_backends:
                engram_save_path = engram_paths[backend]
                backend_config = build_engram_config(args, backend, sparse_embeddings)
                model = build_joint_model(
                    args.model,
                    tokenizer,
                    use_deepspeed,
                    backend_config,
                    lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                )
                train_tokenized = tokenize_dataset(
                    train_raw, tokenizer, max_length=args.max_length
                )
                trainer = EngramTrainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=os.path.join(
                            args.output_dir,
                            f"tmp_joint_{backend_tag_for_output(backend, args.rq_table_dir)}",
                        ),
                        per_device_train_batch_size=args.batch_size,
                        gradient_accumulation_steps=args.grad_accum,
                        learning_rate=args.learning_rate,
                        num_train_epochs=args.num_epochs,
                        max_steps=args.max_steps if args.max_steps > 0 else -1,
                        max_grad_norm=args.max_grad_norm,
                        warmup_ratio=args.warmup_ratio,
                        logging_steps=args.logging_steps,
                        save_steps=0,
                        save_total_limit=0,
                        eval_strategy="no",
                        bf16=True,
                        deepspeed=deepspeed_config,
                        ddp_find_unused_parameters=False,
                        dataloader_num_workers=2,
                        seed=args.seed,
                        report_to="none",
                        remove_unused_columns=False,
                    ),
                    train_dataset=train_tokenized,
                    data_collator=EngramDataCollator(
                        tokenizer=tokenizer, config=model.config, mlm=False
                    ),
                )
                trainer.train()
                if is_main:
                    os.makedirs(engram_save_path, exist_ok=True)
                    os.makedirs(lora_save_path, exist_ok=True)
                    saved = trainer.accelerator.unwrap_model(trainer.model)
                    if not isinstance(saved, EngramModel):
                        saved = model
                    saved.save_pretrained(engram_save_path)
                    saved.base_model.save_pretrained(lora_save_path)
                    logging.info(
                        "Engram adapter saved to %s",
                        engram_save_path,
                    )
                    logging.info("LoRA adapter saved to %s", lora_save_path)
                del model, trainer, train_tokenized
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if is_main:
                    tag = backend_tag_for_output(backend, args.rq_table_dir)
                    logging.info(">>> Immediate eval after Joint(%s+LoRA) training", tag)
                    run_eval_step((f"Joint({tag}+LoRA)", "engram", engram_save_path, "lora"))

    # ═══════════════════════════════════════════════════════════════
    #  EVALUATION
    # ═══════════════════════════════════════════════════════════════
    if not is_main:
        return  # evaluation runs on main process only

    logging.info("\n" + "=" * 60)
    logging.info("  Evaluation on %d held-out PopQA samples", eval_max)
    logging.info("=" * 60)

    for step in current_eval_steps():
        run_eval_step(step)

    # --- Print table ---
    print_comparison(results)

    if args.mode == "train" and is_main:
        logging.info("\nAdapters saved to: %s", args.output_dir)
        logging.info("  Engram backends: %s", ", ".join(engram_paths.values()))
        if args.lora or args.joint:
            logging.info("  LoRA adapter: %s", lora_save_path)
        logging.info(
            "Re-run with: --mode eval --engram_path %s [--lora_path %s]",
            ", ".join(engram_paths.values()),
            lora_save_path,
        )


if __name__ == "__main__":
    main()
