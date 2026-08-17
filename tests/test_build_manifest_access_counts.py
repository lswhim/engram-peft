from scripts.build_manifest_access_counts import canonical_token_ids


class CharacterTokenizer:
    def __call__(self, text: str, add_special_tokens: bool):
        return {"input_ids": ([999] if add_special_tokens else []) + [ord(c) for c in text]}


def test_canonical_token_ids_matches_qa_training_boundary() -> None:
    ids = canonical_token_ids(CharacterTokenizer(), "Question", "Answer", 100, "qa")

    expected = [999] + [ord(c) for c in "Q: Question A: Answer"]
    assert ids == expected
