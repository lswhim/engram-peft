import json
from unittest.mock import patch

import torch

from examples.benchmarks.data import prepare_dataset
from examples.benchmarks.methods import ChronologicalEngramTrainer, MilestoneSaveCallback


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


def test_semantic_manifest_supports_wikibigedit_qa_format(tmp_path):
    class CharacterTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text, add_special_tokens=True, **kwargs):
            del add_special_tokens, kwargs
            assert isinstance(text, str)
            values = [ord(character) for character in text]
            return {"input_ids": values, "attention_mask": [1] * len(values)}

    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"prompt":"canonical","target":"new","queries":[]}) + "\n")
    train, _ = prepare_dataset(
        CharacterTokenizer(),
        0,
        1,
        16,
        num_proc=1,
        dataset="semantic_manifest",
        manifest_path=str(path),
        prompt_format="qa",
    )
    assert len(train) == 1
    labels = train[0]["labels"]
    assert labels[: len("Q: canonical A:")] == [-100] * len("Q: canonical A:")
    assert labels[len("Q: canonical A:") : len("Q: canonical A: new")] == [
        ord(character) for character in " new"
    ]


def test_chronological_trainer_uses_sequential_sampler():
    trainer=object.__new__(ChronologicalEngramTrainer)
    trainer.train_dataset=[1,2,3]
    assert list(trainer._get_train_sampler([4,5]))==[0,1]


def test_milestone_callback_saves_once(tmp_path):
    class Model:
        def __init__(self): self.paths=[]
        def save_pretrained(self,path): self.paths.append(path)
    model=Model(); callback=MilestoneSaveCallback({2:str(tmp_path/"two")})
    state=type("State",(),{"global_step":2})(); control=object()
    callback.on_step_end(None,state,control,model=model)
    callback.on_step_end(None,state,control,model=model)
    assert model.paths==[str(tmp_path/"two")]
