from __future__ import annotations

import json
import re
from typing import Any

DEFAULT_SEED = 42
TRAIN_RATIO = 0.8


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


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


def process_docs(dataset):
    def _process_doc(doc: dict[str, Any]) -> dict[str, Any]:
        answers = parse_possible_answers(doc.get("possible_answers"))
        return {
            "id": doc.get("id"),
            "question": doc["question"],
            "answers": answers,
        }

    processed = dataset.map(_process_doc)
    return processed.train_test_split(
        test_size=1 - TRAIN_RATIO,
        seed=DEFAULT_SEED,
    )["test"]


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = results[0].split("\n", 1)[0].strip()
    pred_norm = normalize_answer(prediction)
    gold_norms = [normalize_answer(answer) for answer in doc["answers"]]
    return {"em": float(pred_norm in gold_norms)}
