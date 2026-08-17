from scripts.analyze_bucket_target_conflict import choose_heads, posterior_probability


def test_choose_heads_uses_low_load_as_specific() -> None:
    groups = choose_heads([9, 1, 5, 2], top_k=2, seed=3)
    assert groups["top_specific"] == [1, 3]
    assert groups["bottom_specific"] == [2, 0]
    assert len(groups["random_k"]) == 2


def test_posterior_probability_interpolates_global_prior() -> None:
    probability = posterior_probability(
        target_count=3, bucket_total=10, global_probability=0.1, prior_strength=10
    )
    assert probability == 0.2
