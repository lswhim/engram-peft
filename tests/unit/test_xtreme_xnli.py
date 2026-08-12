from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from examples.benchmarks.xtreme_xnli import (
    CausalLabelCollator,
    XNLI_LANGUAGES,
    encode_train_example,
    evaluate_xnli_language,
)
from examples.run_xtreme_xnli import _engram_config


class TinyTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        ids = [ord(char) % 31 + 1 for char in text]
        if add_special_tokens:
            ids = [31] + ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


class FirstLabelModel(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 64)
        for row in range(batch):
            for position in range(length - 1):
                logits[row, position, int(input_ids[row, position + 1])] = 10.0
        return SimpleNamespace(logits=logits)


def test_xnli_protocol_has_all_fifteen_languages() -> None:
    assert len(XNLI_LANGUAGES) == 15
    assert set(XNLI_LANGUAGES) == {
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
    }


def test_encode_and_collate_preserve_answer_only_loss() -> None:
    tokenizer = TinyTokenizer()
    first = encode_train_example(
        {"premise": "p", "hypothesis": "h", "label": 0}, tokenizer, 128
    )
    second = encode_train_example(
        {"premise": "long premise", "hypothesis": "h", "label": 2}, tokenizer, 128
    )
    assert first["labels"].count(-100) > 0
    assert [label for label in first["labels"] if label != -100] == tokenizer(
        " entailment", add_special_tokens=False
    )["input_ids"]

    batch = CausalLabelCollator(tokenizer)([first, second])
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] % 8 == 0
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)


def test_language_evaluation_returns_exact_accuracy() -> None:
    tokenizer = TinyTokenizer()
    model = FirstLabelModel()
    rows = [
        {"premise": "p", "hypothesis": "h", "label": 0},
        {"premise": "q", "hypothesis": "r", "label": 0},
        {"premise": "s", "hypothesis": "t", "label": 0},
    ]
    accuracy = evaluate_xnli_language(
        model,
        tokenizer,
        rows,
        batch_size=2,
        max_length=128,
        device=torch.device("cpu"),
    )
    assert accuracy == 1.0


def test_arithmetic_matched_uses_rq_capacity_per_head(tmp_path: Path) -> None:
    table_dir = tmp_path / "rq"
    table_dir.mkdir()
    (table_dir / "meta.json").write_text(
        json.dumps({"num_levels": 8, "codebook_size": 256}), encoding="utf-8"
    )
    args = SimpleNamespace(
        method="arithmetic_matched",
        rq_table_dir=str(table_dir),
        arith_buckets=512000,
        target_layers=[11, 21],
        engram_embedding_dim=1280,
        model="Qwen/Qwen3-1.7B-Base",
        pad_token_id=0,
        seed=42,
    )
    config = _engram_config(args, SimpleNamespace(config=SimpleNamespace(hidden_size=2048)))

    assert config.n_head_per_ngram == 8
    assert config.engram_vocab_size_per_ngram == [2048, 2048]
    assert config.hash_backend == "arithmetic_fixed"
