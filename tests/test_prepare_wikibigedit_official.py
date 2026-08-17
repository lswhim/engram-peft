from examples.prepare_wikibigedit_official import convert_row


def test_convert_row_preserves_all_official_axes() -> None:
    row = {
        "update": "Who leads Exampleland?",
        "ans": "Ada",
        "rephrase": "Who is Exampleland's leader?",
        "personas": "Exampleland leader who?",
        "personas_formal": "Identify the leader of Exampleland.",
        "mhop": "Where was Exampleland's leader born?",
        "mhop_ans": "London",
        "loc": "Who leads Otherland?",
        "loc_ans": "Bob",
        "subject": "Exampleland",
    }

    case = convert_row(row, "t0", 7)

    assert case is not None
    assert case["case_id"] == "t0:7"
    assert case["prompt"] == row["update"]
    assert case["target"] == row["ans"]
    assert case["metadata"]["cohort_origin"] == "t0"
    assert [item["axis"] for item in case["queries"]] == [
        "efficacy",
        "generalization",
        "personas",
        "personas",
        "multi_hop",
        "locality",
    ]
    assert case["queries"][-1]["role"] == "should_not_propagate"


def test_convert_row_skips_untrainable_case_and_empty_optional_queries() -> None:
    assert convert_row({"update": "Question", "ans": None}, "t0", 0) is None

    case = convert_row(
        {"update": "Question", "ans": "Answer", "rephrase": "", "personas": None},
        "t0",
        1,
    )

    assert case is not None
    assert [item["axis"] for item in case["queries"]] == ["efficacy"]
