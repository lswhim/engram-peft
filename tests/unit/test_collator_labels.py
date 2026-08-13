from types import SimpleNamespace

import torch

from engram_peft.collator import EngramDataCollator


class TinyTokenizer:
    pad_token_id = 0

    def pad(self, features, **kwargs):
        del kwargs
        return {key: torch.tensor([feature[key] for feature in features]) for key in features[0]}

    def get_special_tokens_mask(self, values, already_has_special_tokens=True):
        del already_has_special_tokens
        return [0] * len(values)


def test_engram_collator_preserves_dataset_label_mask() -> None:
    collator = EngramDataCollator.__new__(EngramDataCollator)
    collator.tokenizer = TinyTokenizer()
    collator.mlm = False
    collator.return_tensors = "pt"
    collator.pad_to_multiple_of = None
    collator.tf_experimental_compile = False
    collator.seed = None
    collator.generator = None
    collator.config = SimpleNamespace(hash_backend="rq")
    features = [{"input_ids": [7, 8, 9, 0], "attention_mask": [1, 1, 1, 0], "labels": [-100, -100, 9, -100]}]
    batch = collator(features)
    assert batch["labels"].tolist() == [[-100, -100, 9, -100]]
