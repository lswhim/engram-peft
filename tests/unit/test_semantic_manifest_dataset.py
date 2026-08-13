import json
from unittest.mock import patch

from examples.benchmarks.data import prepare_dataset


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True, **kwargs):
        del add_special_tokens, kwargs
        if isinstance(text, list):
            return {"input_ids": [[1] for _ in text], "attention_mask": [[1] for _ in text]}
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}


def test_semantic_manifest_trains_only_on_canonical_prompt_target(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"prompt":"canonical","target":"new","queries":[{"prompt":"held out","answers":["new"]}]}) + "\n")
    train, _ = prepare_dataset(TinyTokenizer(), 0, 1, 8, num_proc=1, dataset="semantic_manifest", manifest_path=str(path))
    assert len(train) == 1
    labels = train[0]["labels"]
    assert labels[:2] == [-100, -100]
    assert labels[2:4] == [1, 2]
    assert all(value == -100 for value in labels[4:])
