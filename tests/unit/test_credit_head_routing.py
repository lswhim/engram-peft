from __future__ import annotations

import torch

from engram_peft.config import EngramConfig
from engram_peft.layer import ContextAwareGating, HeadFactorizedGating


def make_config(**overrides: object) -> EngramConfig:
    values = {
        "hidden_size": 12,
        "embedding_dim": 24,
        "hc_mult": 2,
        "memory_fusion": "head_factorized",
        "head_router_top_k": 0,
    }
    values.update(overrides)
    return EngramConfig(**values)


def test_factorized_value_blocks_match_flatten_when_all_gates_equal() -> None:
    torch.manual_seed(7)
    config = make_config()
    flat = ContextAwareGating(config, 24, 12, hc_mult=2, zero_init=False)
    routed = HeadFactorizedGating(
        config,
        num_heads=4,
        embedding_dim_per_head=6,
        hidden_size=12,
        hc_mult=2,
        zero_init=False,
    )
    routed.w_v.weight.data.copy_(flat.w_v.weight.data)
    # Zero keys make every route gate exactly 0.5 in both implementations.  This
    # isolates the block decomposition identity without conflating router behavior.
    for flat_key, routed_key in zip(flat.w_k, routed.w_k, strict=True):
        flat_key.weight.data.zero_()
        routed_key.weight.data.zero_()

    embeddings = torch.randn(3, 5, 4, 6)
    hidden = torch.randn(3, 5, 2, 12)
    expected = flat(embeddings.flatten(start_dim=-2), hidden)
    actual = routed(embeddings, hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_forced_null_and_equal_budget_masks() -> None:
    torch.manual_seed(11)
    config = make_config()
    routed = HeadFactorizedGating(
        config,
        num_heads=4,
        embedding_dim_per_head=6,
        hidden_size=12,
        hc_mult=2,
        zero_init=False,
    )
    embeddings = torch.randn(2, 3, 4, 6)
    hidden = torch.randn(2, 3, 2, 12)

    routed.set_forced_head_mask(torch.zeros(2, 4))
    null_output = routed(embeddings, hidden)
    torch.testing.assert_close(null_output, torch.zeros_like(null_output))

    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
    routed.set_forced_head_mask(mask)
    routed(embeddings, hidden)
    assert routed.last_gate is not None
    assert torch.equal(
        routed.last_gate.ne(0).sum(dim=-1),
        torch.full((2, 3, 2), 2, dtype=torch.long),
    )


def test_top_k_routes_exactly_requested_number_of_heads() -> None:
    config = make_config(head_router_top_k=2)
    routed = HeadFactorizedGating(
        config,
        num_heads=4,
        embedding_dim_per_head=6,
        hidden_size=12,
        hc_mult=2,
        zero_init=False,
    )
    routed(torch.randn(2, 3, 4, 6), torch.randn(2, 3, 2, 12))
    assert routed.last_gate is not None
    assert torch.equal(
        routed.last_gate.ne(0).sum(dim=-1),
        torch.full((2, 3, 2), 2, dtype=torch.long),
    )


def test_null_route_closes_memory_below_threshold() -> None:
    config = make_config(
        head_router_top_k=2,
        head_router_use_null=True,
        head_router_null_threshold=0.1,
    )
    routed = HeadFactorizedGating(
        config,
        num_heads=4,
        embedding_dim_per_head=6,
        hidden_size=12,
        hc_mult=2,
        zero_init=False,
    )
    for key in routed.w_k:
        key.weight.data.zero_()
    output = routed(torch.randn(2, 3, 4, 6), torch.randn(2, 3, 2, 12))
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_credit_config_round_trip() -> None:
    config = make_config(
        head_router_top_k=4,
        head_router_use_null=True,
        head_router_null_threshold=0.2,
        credit_loss_weight=0.2,
        credit_pair_fraction=0.25,
        credit_route_k=3,
        credit_temperature=0.7,
    )
    restored = EngramConfig.from_dict(config.to_dict())
    assert restored.memory_fusion == "head_factorized"
    assert restored.head_router_top_k == 4
    assert restored.head_router_use_null is True
    assert restored.head_router_null_threshold == 0.2
    assert restored.credit_loss_weight == 0.2
    assert restored.credit_pair_fraction == 0.25
    assert restored.credit_route_k == 3
    assert restored.credit_temperature == 0.7
