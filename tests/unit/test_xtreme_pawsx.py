from __future__ import annotations

import pytest

from examples.benchmarks.xtreme_pawsx import (
    PAWSX_LANGUAGES,
    encode_train_example,
    format_pawsx_prompt,
)


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, **kwargs):
        ids = [ord(char) % 31 + 1 for char in text]
        if kwargs.get("truncation"):
            ids = ids[: kwargs["max_length"]]
        return {"input_ids": ids}


def test_protocol_has_all_seven_languages() -> None:
    assert PAWSX_LANGUAGES == ("en", "de", "es", "fr", "ja", "ko", "zh")


def test_prompt_is_fixed_english_task_instruction() -> None:
    prompt = format_pawsx_prompt("eins", "zwei")
    assert "Sentence 1: eins" in prompt
    assert "Sentence 2: zwei" in prompt
    assert prompt.endswith("Label:")


def test_training_masks_prompt_and_keeps_answer() -> None:
    encoded = encode_train_example(
        {"sentence1": "a", "sentence2": "b", "label": 1}, TinyTokenizer(), 128
    )
    answer_length = len(TinyTokenizer()(" paraphrase")["input_ids"])
    assert encoded["labels"][:-answer_length] == [-100] * (
        len(encoded["labels"]) - answer_length
    )
    assert encoded["labels"][-answer_length:] == encoded["input_ids"][-answer_length:]


def test_invalid_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid PAWS-X label"):
        encode_train_example(
            {"sentence1": "a", "sentence2": "b", "label": 2}, TinyTokenizer(), 128
        )
