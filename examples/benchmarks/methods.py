# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
import logging
import os
import shutil
from typing import Any

import torch
from torch.utils.data import SequentialSampler
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim.adamw import AdamW
from transformers import (
    DataCollatorForLanguageModeling,
    default_data_collator,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramTrainer,
    get_engram_model,
)
from engram_peft.types import ModelProtocol, PeftUnloadable
from engram_peft.utils.compat import wash_tokenizer

# Configure logging to see Engram injection logs
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("engram_peft").setLevel(logging.INFO)


class ChronologicalEngramTrainer(EngramTrainer):
    """HF Trainer variant that consumes the manifest in its original order."""

    def _get_train_sampler(self, train_dataset: Any | None = None) -> SequentialSampler:
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return SequentialSampler(dataset)


class MilestoneSaveCallback(TrainerCallback):
    def __init__(self, milestones: dict[int, str]):
        self.milestones = milestones
        self.saved: set[int] = set()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args
        step=int(state.global_step)
        if step in self.milestones and step not in self.saved:
            model=kwargs["model"]
            model.save_pretrained(self.milestones[step])
            self.saved.add(step)
            print(f"[lifelong milestone] step={step} saved={self.milestones[step]}",flush=True)
        return control


def extract_trainer_metrics(trainer: Trainer, train_result: Any) -> dict[str, Any]:
    """Helper to extract common metrics from a Trainer object."""
    eval_results = trainer.evaluate()
    eval_loss_raw = eval_results.get("eval_loss", 0.0)
    eval_loss = float(eval_loss_raw)

    avg_time_per_step = train_result.metrics.get("train_runtime", 0) / max(
        1, train_result.global_step
    )

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    log_history = []
    for log in trainer.state.log_history:
        if "step" in log:
            entry = {"step": log["step"]}
            if "loss" in log:
                entry["loss"] = log["loss"]
            if "eval_loss" in log:
                entry["eval_loss"] = log["eval_loss"]
            if len(entry) > 1:
                log_history.append(entry)

    return {
        "log_history": log_history,
        "completed_steps": int(trainer.state.global_step),
        "peak_memory_gb": peak_memory,
        "avg_time_per_step": avg_time_per_step,
        "eval_loss": eval_loss,
    }


def train_lora(
    base_model: ModelProtocol,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: Any,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    print("\n>>> Method: LoRA")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    if not isinstance(base_model, PreTrainedModel):
        # Fallback to nominal check for get_peft_model
        raise TypeError("base_model must be a PreTrainedModel for PEFT")
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    warmup_steps = int(args.max_steps * 0.03)
    num_decay_steps = int(args.max_steps * 0.77)
    scheduler_kwargs = {
        "num_decay_steps": num_decay_steps,
        "min_lr_ratio": 1e-6 / 3e-4,
    }

    model_short = str(getattr(args, "model_name", "model")).split("/")[-1]
    seed = getattr(args, "seed", 42)
    run_tag = f"{model_short}_lora_r{peft_config.r}_seed{seed}"
    run_tag += getattr(args, "run_suffix", "") or ""

    training_args = TrainingArguments(
        output_dir=f"outputs/benchmarks/tmp/{run_tag}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=3e-4,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs=scheduler_kwargs,
        warmup_steps=warmup_steps,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=(
            500 if getattr(args, "disable_early_stopping", False) else 100
        ),
        report_to="wandb" if args.wandb else "none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        save_strategy="steps",
        save_steps=(
            1000 if getattr(args, "disable_early_stopping", False) else 100
        ),
        save_total_limit=2,
        load_best_model_at_end=not getattr(args, "disable_early_stopping", False),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    # Apply overrides
    for k, v in overrides.items():
        if hasattr(peft_config, k):
            setattr(peft_config, k, v)
        if hasattr(training_args, k):
            setattr(training_args, k, v)

    callbacks = (
        []
        if getattr(args, "disable_early_stopping", False)
        else [EarlyStoppingCallback(early_stopping_patience=5)]
    )
    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    train_result = trainer.train()
    metrics = extract_trainer_metrics(trainer, train_result)

    save_dir = f"outputs/benchmarks/ckpt_{run_tag}"
    model.save_pretrained(save_dir)
    print(f"[lora] saved to {save_dir}")
    metrics["save_dir"] = save_dir
    # Clean up

    if isinstance(model, PeftUnloadable):
        model = model.unload()
    return metrics


def train_engram(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: Any,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    print("\n>>> Method: Engram Only")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    config = EngramConfig(
        n_head_per_ngram=16,
        target_layers=[11, 21],
        engram_vocab_size_per_ngram=[512000, 512000],
        hidden_size=base_model.config.hidden_size,
        embedding_dim=1280,
        enable_tokenizer_compression=True,
        tokenizer_name_or_path=args.model_name_or_path
        if hasattr(args, "model_name_or_path")
        else None,
        pad_id=tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0,
        learning_rate_multiplier=15.0,
    )
    # Apply overrides to engram config
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    # base_model is already a PreTrainedModel
    model = get_engram_model(base_model, config, wash_tokenizer(tokenizer))
    model.print_trainable_parameters()

    warmup_steps = int(args.max_steps * 0.03)
    num_decay_steps = int(args.max_steps * 0.77)
    scheduler_kwargs = {
        "num_decay_steps": num_decay_steps,
        "min_lr_ratio": 1e-6 / 3e-4,
    }

    backend = getattr(config, "hash_backend", "arithmetic")
    model_short = str(getattr(args, "model_name", "model")).split("/")[-1]
    seed = getattr(args, "seed", 42)
    run_tag = f"{model_short}_{backend}_h{config.n_head_per_ngram}_seed{seed}"
    if backend == "rq":
        _rqd = getattr(config, "rq_table_dir", "") or ""
        if _rqd:
            run_tag += "_" + os.path.basename(_rqd.rstrip("/"))
    run_tag += getattr(args, "run_suffix", "") or ""

    training_args = TrainingArguments(
        output_dir=f"outputs/benchmarks/tmp/{run_tag}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=3e-4,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs=scheduler_kwargs,
        warmup_steps=warmup_steps,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        report_to="wandb" if args.wandb else "none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=not getattr(args, "disable_early_stopping", False),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    # Apply overrides to training_args
    for k, v in overrides.items():
        if hasattr(training_args, k):
            setattr(training_args, k, v)

    if getattr(config, "hash_backend", "arithmetic") == "mixed_v2":
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
    else:
        collator = EngramDataCollator(tokenizer=wash_tokenizer(tokenizer), config=config)
    callbacks: list[Any] = (
        []
        if getattr(args, "disable_early_stopping", False)
        else [EarlyStoppingCallback(early_stopping_patience=5)]
    )
    trainer_class = (
        ChronologicalEngramTrainer
        if getattr(args, "chronological", False)
        else EngramTrainer
    )
    if getattr(args, "milestone_examples", []):
        effective_batch = int(args.batch_size) * int(args.grad_accum)
        invalid = [
            point for point in args.milestone_examples
            if point % effective_batch
        ]
        if invalid:
            raise ValueError(
                f"milestones {invalid} are not divisible by effective batch "
                f"{effective_batch}"
            )
        milestone_dirs = {
            point // effective_batch: (
                f"outputs/benchmarks/milestones/{run_tag}/writes_{point}"
            )
            for point in args.milestone_examples
        }
        callbacks.append(MilestoneSaveCallback(milestone_dirs))
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    train_result = trainer.train()
    metrics = extract_trainer_metrics(trainer, train_result)
    metrics["planned_steps"] = int(args.max_steps)
    metrics["processed_token_slots"] = int(
        trainer.state.global_step
        * args.batch_size
        * args.grad_accum
        * args.max_length
    )
    example_presentations = int(
        trainer.state.global_step * args.batch_size * args.grad_accum
    )
    metrics["example_presentations"] = example_presentations
    if len(train_dataset) > 0 and example_presentations % len(train_dataset) == 0:
        repetitions = example_presentations // len(train_dataset)
        non_padding_per_epoch = sum(
            int(sum(example["attention_mask"])) for example in train_dataset
        )
        causal_targets_per_epoch = sum(
            max(0, int(sum(example["attention_mask"])) - 1)
            for example in train_dataset
        )
        metrics["full_dataset_repetitions"] = repetitions
        metrics["non_padding_token_presentations"] = int(
            non_padding_per_epoch * repetitions
        )
        metrics["causal_target_token_presentations"] = int(
            causal_targets_per_epoch * repetitions
        )
    metrics["fixed_steps_complete"] = bool(
        getattr(args, "disable_early_stopping", False)
        and trainer.state.global_step == args.max_steps
    )

    # Standard runs restore the best checkpoint. Fixed-step paper runs disable
    # that path and deliberately save the exactly-max_steps final state.
    # Per-config save dir so a matrix (base x backend x seed) does not collide.
    save_dir = f"outputs/benchmarks/ckpt_{run_tag}"
    model.save_pretrained(save_dir)
    print(f"[engram] saved checkpoint to {save_dir}")
    metrics["save_dir"] = save_dir
    model.unload_engram()
    return metrics


def train_full_finetune(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: Any,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    print("\n>>> Method: Full Finetune")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    base_model.requires_grad_(True)
    print(
        f"trainable params: {sum(p.numel() for p in base_model.parameters() if p.requires_grad):,}"
    )

    warmup_steps = int(args.max_steps * 0.03)
    num_decay_steps = int(args.max_steps * 0.77)
    scheduler_kwargs = {
        "num_decay_steps": num_decay_steps,
        "min_lr_ratio": 1e-6 / 5e-5,
    }

    # Also print it manually if it were a PEFT model, but for base model we do it like this
    training_args = TrainingArguments(
        output_dir="outputs/benchmarks/tmp/full_ft",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=5e-5,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs=scheduler_kwargs,
        warmup_steps=warmup_steps,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        seed=args.seed,
        data_seed=args.seed,
        report_to="wandb" if args.wandb else "none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    # Apply overrides
    for k, v in overrides.items():
        if hasattr(training_args, k):
            setattr(training_args, k, v)

    trainer = EngramTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    train_result = trainer.train()
    metrics = extract_trainer_metrics(trainer, train_result)

    save_path = "outputs/benchmarks/full_ft_only_weights"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    base_model.save_pretrained(save_path)
    return metrics


def train_lora_engram(
    base_model: ModelProtocol,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: Any,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    print("\n>>> Method: LoRA + Engram")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    # Apply overrides to peft_config
    for k, v in overrides.items():
        if hasattr(peft_config, k):
            setattr(peft_config, k, v)

    if not isinstance(base_model, PreTrainedModel):
        raise TypeError("base_model must be a PreTrainedModel for LoRA")
    lora_model = get_peft_model(base_model, peft_config)

    # Load Engram wrapper
    config = EngramConfig(
        n_head_per_ngram=16,
        target_layers=[11, 21],
        engram_vocab_size_per_ngram=[512000, 512000],
        hidden_size=base_model.config.hidden_size,
        embedding_dim=1280,
        enable_tokenizer_compression=True,
        tokenizer_name_or_path=args.model_name_or_path
        if hasattr(args, "model_name_or_path")
        else None,
        pad_id=tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0,
        learning_rate_multiplier=15.0,
    )
    # Apply overrides to engram config
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    # lora_model is already a torch.nn.Module
    model = get_engram_model(
        lora_model,
        config,
        wash_tokenizer(tokenizer),
        train_mode="preserve_trainable",
    )
    model.print_trainable_parameters()

    warmup_steps = int(args.max_steps * 0.03)
    num_decay_steps = int(args.max_steps * 0.77)
    scheduler_kwargs = {
        "num_decay_steps": num_decay_steps,
        "min_lr_ratio": 1e-6 / 3e-4,
    }

    training_args = TrainingArguments(
        output_dir="outputs/benchmarks/tmp/lora_engram",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=3e-4,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs=scheduler_kwargs,
        warmup_steps=warmup_steps,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        seed=args.seed,
        data_seed=args.seed,
        report_to="wandb" if args.wandb else "none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    # Apply overrides to training_args
    for k, v in overrides.items():
        if hasattr(training_args, k):
            setattr(training_args, k, v)

    collator = EngramDataCollator(tokenizer=wash_tokenizer(tokenizer), config=config)
    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    train_result = trainer.train()
    metrics = extract_trainer_metrics(trainer, train_result)

    model.save_pretrained("outputs/benchmarks/lora_engram_weights")
    model.unload_engram()
    if isinstance(lora_model, PeftUnloadable):
        lora_model.unload()
    return metrics


def train_full_finetune_engram(
    base_model: ModelProtocol,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: Any,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    print("\n>>> Method: Full Finetune + Engram")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    config = EngramConfig(
        n_head_per_ngram=16,
        target_layers=[11, 21],
        engram_vocab_size_per_ngram=[512000, 512000],
        hidden_size=base_model.config.hidden_size,
        embedding_dim=1280,
        enable_tokenizer_compression=True,
        tokenizer_name_or_path=args.model_name_or_path
        if hasattr(args, "model_name_or_path")
        else None,
        pad_id=tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0,
        entropy_loss_weight=0.0,
        learning_rate_multiplier=5.0,
        backbone_freeze_steps=0,
        enable_telemetry=True,
        clip_grad_per_group=True,
    )
    # Apply overrides to config
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    if not isinstance(base_model, PreTrainedModel | torch.nn.Module):
        raise TypeError("base_model must be a Module for Engram injection")
    # base_model is already a PreTrainedModel
    model = get_engram_model(
        base_model,
        config,
        wash_tokenizer(tokenizer),
        train_mode="full_finetune",
    )
    model.print_trainable_parameters()

    warmup_steps = int(args.max_steps * 0.03)
    num_decay_steps = int(args.max_steps * 0.77)
    scheduler_kwargs = {
        "num_decay_steps": num_decay_steps,
        "min_lr_ratio": 1e-6 / 5e-5,
    }

    training_args = TrainingArguments(
        output_dir="outputs/benchmarks/tmp/full_ft_engram",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=3e-4,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs=scheduler_kwargs,
        warmup_steps=warmup_steps,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        seed=args.seed,
        data_seed=args.seed,
        report_to="wandb" if args.wandb else "none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    # Apply overrides to training_args
    for k, v in overrides.items():
        if hasattr(training_args, k):
            setattr(training_args, k, v)

    collator = EngramDataCollator(tokenizer=wash_tokenizer(tokenizer), config=config)
    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        optimizer_kwargs={
            "backbone_learning_rate": 5e-5,
            "engram_dense_learning_rate": 5e-4,
            "engram_sparse_learning_rate": 1e-3,
            "backbone_optimizer": AdamW,
            "engram_dense_optimizer": "adamw",
            "engram_sparse_optimizer": "sparse_adam",
        },
    )

    train_result = trainer.train()
    metrics = extract_trainer_metrics(trainer, train_result)

    model.save_pretrained("outputs/benchmarks/full_ft_engram_weights")
    # Base model saving is now handled by EngramModel.save_pretrained()
    # via save_pretrained_unified if detecting a need to save backbone.
    # However, for full finetune, we might want to be explicit about saving the backbone
    # to a subfolder if we don't want it to merge with Engram artifacts.
    save_fn = getattr(model.base_model, "save_pretrained", None)
    if save_fn is not None:
        save_fn("outputs/benchmarks/full_ft_engram_weights/base_model")
    else:
        print(
            "Warning: model.base_model does not have save_pretrained; backbone saving skipped."
        )
    model.unload_engram()
    return metrics
