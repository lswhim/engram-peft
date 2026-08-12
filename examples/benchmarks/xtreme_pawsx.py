# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
"""Utilities for English-source, seven-language zero-shot PAWS-X."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset

from examples.benchmarks.xtreme_xnli import CausalLabelCollator

PAWSX_LANGUAGES: tuple[str, ...] = ("en", "de", "es", "fr", "ja", "ko", "zh")
PAWSX_LABELS: tuple[str, ...] = ("not paraphrase", "paraphrase")


def format_pawsx_prompt(sentence1: str, sentence2: str) -> str:
    return (
        "Decide whether the two sentences are paraphrases. Answer with "
        "paraphrase or not paraphrase.\n"
        f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nLabel:"
    )


def _encode_text(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    return [int(token_id) for token_id in encoded["input_ids"]]


def encode_train_example(
    example: Mapping[str, Any], tokenizer: Any, max_length: int
) -> dict[str, list[int]]:
    label_id = int(example["label"])
    if not 0 <= label_id < len(PAWSX_LABELS):
        raise ValueError(f"Invalid PAWS-X label: {label_id}")
    target_ids = _encode_text(
        tokenizer, f" {PAWSX_LABELS[label_id]}", add_special_tokens=False
    )
    if len(target_ids) >= max_length:
        raise ValueError("max_length is too small to hold a PAWS-X label")
    prompt = format_pawsx_prompt(str(example["sentence1"]), str(example["sentence2"]))
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length - len(target_ids),
    )["input_ids"]
    input_ids = [int(token_id) for token_id in prompt_ids] + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def load_pawsx_train_and_tests() -> tuple[Dataset, dict[str, Dataset]]:
    train = load_dataset("google-research-datasets/paws-x", "en", split="train")
    tests = {
        language: load_dataset(
            "google-research-datasets/paws-x", language, split="test"
        )
        for language in PAWSX_LANGUAGES
    }
    return train, tests


def tokenize_pawsx_train(
    dataset: Dataset, tokenizer: Any, max_length: int, num_proc: int
) -> Dataset:
    return dataset.map(
        lambda example: encode_train_example(example, tokenizer, max_length),
        remove_columns=list(dataset.column_names),
        num_proc=num_proc,
        desc="Tokenizing full English PAWS-X train split",
    )


def _batched(
    rows: Sequence[Mapping[str, Any]], size: int
) -> Iterator[Sequence[Mapping[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _candidate_batch(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[int]]:
    label_token_ids = [
        _encode_text(tokenizer, f" {label}", add_special_tokens=False)
        for label in PAWSX_LABELS
    ]
    max_target_length = max(len(ids) for ids in label_token_ids)
    sequences: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    gold: list[int] = []
    for row in rows:
        prompt = format_pawsx_prompt(str(row["sentence1"]), str(row["sentence2"]))
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length - max_target_length,
        )["input_ids"]
        prompt_ids = [int(token_id) for token_id in prompt_ids]
        for target_ids in label_token_ids:
            sequences.append(prompt_ids + target_ids)
            spans.append((len(prompt_ids), len(prompt_ids) + len(target_ids)))
        gold.append(int(row["label"]))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id must be set")
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[index, : len(sequence)] = 1
    return input_ids, attention_mask, spans, gold


@torch.inference_mode()
def evaluate_pawsx_language(
    model: Any,
    tokenizer: Any,
    dataset: Dataset | Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> float:
    was_training = bool(model.training)
    model.eval()
    correct = 0
    total = 0
    rows = dataset if isinstance(dataset, Sequence) else list(dataset)
    for batch_rows in _batched(rows, batch_size):
        input_ids, attention_mask, spans, gold = _candidate_batch(
            batch_rows, tokenizer, max_length
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        log_probs = F.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)
        candidate_scores: list[float] = []
        for row_index, (start, end) in enumerate(spans):
            positions = torch.arange(start - 1, end - 1, device=device)
            targets = input_ids[row_index, start:end]
            candidate_scores.append(
                float(log_probs[row_index, positions, targets].mean().item())
            )
        predictions = (
            torch.tensor(candidate_scores)
            .view(-1, len(PAWSX_LABELS))
            .argmax(dim=1)
            .tolist()
        )
        correct += sum(
            int(prediction == target)
            for prediction, target in zip(predictions, gold, strict=True)
        )
        total += len(gold)
    if was_training:
        model.train()
    return correct / total if total else 0.0


def evaluate_all_languages(
    model: Any,
    tokenizer: Any,
    tests: Mapping[str, Dataset],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    on_language: Callable[[str, float], None] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for language in PAWSX_LANGUAGES:
        accuracy = evaluate_pawsx_language(
            model,
            tokenizer,
            tests[language],
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        metrics[language] = accuracy
        if on_language is not None:
            on_language(language, accuracy)
    metrics["macro"] = sum(metrics.values()) / len(PAWSX_LANGUAGES)
    return metrics


__all__ = [
    "CausalLabelCollator",
    "PAWSX_LABELS",
    "PAWSX_LANGUAGES",
    "encode_train_example",
    "evaluate_all_languages",
    "evaluate_pawsx_language",
    "format_pawsx_prompt",
    "load_pawsx_train_and_tests",
    "tokenize_pawsx_train",
]
