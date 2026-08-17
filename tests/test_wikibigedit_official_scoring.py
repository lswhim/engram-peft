from examples.evaluate_semantic_memory import formatted_pair, prediction_preservation


class CharacterTokenizer:
    def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[int]]:
        prefix = [999] if add_special_tokens else []
        return {"input_ids": prefix + [ord(character) for character in text]}


def test_qa_format_uses_official_question_answer_prefix() -> None:
    tokenizer = CharacterTokenizer()

    prompt_ids, answer_ids = formatted_pair(
        tokenizer, "Who leads Exampleland?", "Ada", "qa"
    )

    assert prompt_ids == [999] + [ord(character) for character in "Q: Who leads Exampleland? A:"]
    assert answer_ids == [ord(character) for character in " Ada"]


def test_locality_is_pre_post_prediction_preservation() -> None:
    assert prediction_preservation([1, 2, 3, 4], [1, 9, 3, 4]) == 0.75
