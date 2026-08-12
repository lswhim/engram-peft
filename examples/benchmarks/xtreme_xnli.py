# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
"""Data and evaluation utilities for the standard XNLI zero-shot protocol.

The source model is trained on the complete English MNLI training split and is
evaluated, without target-language updates, on every XNLI test language.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset

XNLI_LANGUAGES: tuple[str, ...] = (
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
)
XNLI_LABELS: tuple[str, ...] = ("entailment", "neutral", "contradiction")


def format_xnli_prompt(premise: str, hypothesis: str) -> str:
    """Use one fixed English task prompt for source training and all targets."""
    return (
        "Decide whether the premise entails, is neutral toward, or contradicts "
        "the hypothesis. Answer with entailment, neutral, or contradiction.\n"
        f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"
    )


def _encode_text(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    return [int(token_id) for token_id in encoded["input_ids"]]


def encode_train_example(
    example: Mapping[str, Any], tokenizer: Any, max_length: int
) -> dict[str, list[int]]:
    """Encode one English training example and mask every non-answer token."""
    label_id = int(example["label"])
    if not 0 <= label_id < len(XNLI_LABELS):
        raise ValueError(f"Invalid XNLI label: {label_id}")

    target_ids = _encode_text(
        tokenizer, f" {XNLI_LABELS[label_id]}", add_special_tokens=False
    )
    if len(target_ids) >= max_length:
        raise ValueError("max_length is too small to hold an XNLI label")

    prompt = format_xnli_prompt(str(example["premise"]), str(example["hypothesis"]))
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


@dataclass
class CausalLabelCollator:
    """Right-pad causal-label examples while preserving prompt loss masks."""

    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(
        self, features: Sequence[Mapping[str, Sequence[int]]]
    ) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        lengths = [len(feature["input_ids"]) for feature in features]
        target_length = max(lengths)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            target_length = ((target_length + multiple - 1) // multiple) * multiple
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be set")

        input_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for feature, length in zip(features, lengths, strict=True):
            padding = target_length - length
            input_rows.append(list(feature["input_ids"]) + [int(pad_id)] * padding)
            mask_rows.append(list(feature["attention_mask"]) + [0] * padding)
            label_rows.append(list(feature["labels"]) + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }


def load_xnli_train_and_tests() -> tuple[Dataset, dict[str, Dataset]]:
    """Load English source train and all 15 untouched target test splits."""
    train = load_dataset("facebook/xnli", "en", split="train")
    tests = {
        language: load_dataset("facebook/xnli", language, split="test")
        for language in XNLI_LANGUAGES
    }
    return train, tests


def tokenize_xnli_train(
    dataset: Dataset, tokenizer: Any, max_length: int, num_proc: int
) -> Dataset:
    """Tokenize all English source examples for causal-label fine-tuning."""
    columns = list(dataset.column_names)
    return dataset.map(
        lambda example: encode_train_example(example, tokenizer, max_length),
        remove_columns=columns,
        num_proc=num_proc,
        desc="Tokenizing full English XNLI train split",
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
        for label in XNLI_LABELS
    ]
    max_target_length = max(len(ids) for ids in label_token_ids)
    sequences: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    gold: list[int] = []
    for row in rows:
        prompt = format_xnli_prompt(str(row["premise"]), str(row["hypothesis"]))
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
def evaluate_xnli_language(
    model: Any,
    tokenizer: Any,
    dataset: Dataset | Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> float:
    """Return exact 3-way accuracy using normalized label log likelihood."""
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
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        log_probs = F.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)
        candidate_scores: list[float] = []
        for row_index, (start, end) in enumerate(spans):
            positions = torch.arange(start - 1, end - 1, device=device)
            targets = input_ids[row_index, start:end]
            score = log_probs[row_index, positions, targets].mean()
            candidate_scores.append(float(score.item()))
        scores = torch.tensor(candidate_scores).view(-1, len(XNLI_LABELS))
        predictions = scores.argmax(dim=1).tolist()
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
    """Evaluate each language independently and add the unweighted macro mean."""
    metrics: dict[str, float] = {}
    for language in XNLI_LANGUAGES:
        accuracy = evaluate_xnli_language(
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
    metrics["macro"] = sum(metrics.values()) / len(XNLI_LANGUAGES)
    return metrics
