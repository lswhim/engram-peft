from scripts.compare_bucket_target_conflict import paired_interactions


def test_paired_interaction_is_difference_of_differences() -> None:
    semantic = {"a": {"top_specific": 1.0, "random_k": 2.0, "bottom_specific": 3.0}}
    control = {"a": {"top_specific": 1.5, "random_k": 2.0, "bottom_specific": 2.5}}

    values = paired_interactions(semantic, control, "random_k")

    assert values == {"a": -0.5}
