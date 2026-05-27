# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
import copy
from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def _load_tinystories(subset_size: int, eval_size: int) -> tuple[Any, Any]:
    print(f"Loading TinyStories dataset (subset={subset_size})...")
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
    val_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=False)
    train_ds = train_ds.select(range(subset_size))
    val_ds = val_ds.select(range(min(len(val_ds), eval_size)))
    return train_ds, val_ds


def _load_biomed(subset_size: int, eval_size: int, seed: int = 42) -> tuple[Any, Any]:
    """Biomed-Enriched (TinyEngram protocol): en + biomedical; train edu>4.0, eval edu<4.0."""
    print(f"Loading Biomed-Enriched (streaming, train subset={subset_size}, seed={seed})...")
    raw = load_dataset("almanach/Biomed-Enriched", split="commercial", streaming=True)
    # Shuffle the stream so different seeds draw different train subsets.
    # Without this we always took the fixed first-N rows -> seed was a no-op
    # (every seed trained on identical data, see SKILL.md §11.1).
    raw = raw.shuffle(seed=seed, buffer_size=max(subset_size + eval_size, 20000))
    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    for ex in raw:
        if ex.get("language") != "en" or ex.get("domain") != "biomedical":
            continue
        text = ex.get("text")
        if not text:
            continue
        edu = ex.get("educational_score") or 0
        if edu > 4.0 and len(train_rows) < subset_size:
            train_rows.append({"text": text})
        elif edu < 4.0 and len(eval_rows) < eval_size:
            eval_rows.append({"text": text})
        if len(train_rows) >= subset_size and len(eval_rows) >= eval_size:
            break
    print(f"Biomed: {len(train_rows)} train / {len(eval_rows)} eval examples.")
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    subset_size: int,
    eval_size: int,
    max_length: int,
    num_proc: int = 4,
    dataset: str = "tinystories",
    seed: int = 42,
) -> tuple[Any, Any]:
    """Standardized dataset preparation. dataset in {tinystories, biomed}."""
    if dataset == "biomed":
        train_ds, val_ds = _load_biomed(subset_size, eval_size, seed=seed)
    else:
        train_ds, val_ds = _load_tinystories(subset_size, eval_size)

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized_dict = dict(tokenized)
        tokenized_dict["labels"] = copy.deepcopy(tokenized_dict["input_ids"])
        return tokenized_dict

    print(f"Tokenizing with {num_proc} processes...")
    train_dataset = train_ds.map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=num_proc
    )
    eval_dataset = val_ds.map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=num_proc
    )
    return train_dataset, eval_dataset
