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


def test_learned_router_is_dense_and_preserves_gate_mass() -> None:
    config = make_config(head_router_selection="learned")
    routed = HeadFactorizedGating(
        config,
        num_heads=4,
        embedding_dim_per_head=6,
        hidden_size=12,
        hc_mult=2,
        zero_init=False,
    )
    routed(torch.randn(2, 3, 4, 6), torch.randn(2, 3, 2, 12))
    assert routed.last_route_logits is not None
    assert routed.last_gate is not None
    assert torch.all(routed.last_gate > 0)
    torch.testing.assert_close(
        routed.last_gate.sum(dim=-1),
        routed.last_route_logits.sigmoid().sum(dim=-1),
    )


def test_learned_router_receives_task_gradient() -> None:
    config = make_config(head_router_selection="learned")
    routed = HeadFactorizedGating(
        config,
        num_heads=4,
        embedding_dim_per_head=6,
        hidden_size=12,
        hc_mult=2,
        zero_init=False,
    )
    output = routed(
        torch.randn(2, 3, 4, 6), torch.randn(2, 3, 2, 12)
    )
    output.square().mean().backward()
    assert all(key.weight.grad is not None for key in routed.w_k)
    assert all(torch.count_nonzero(key.weight.grad) > 0 for key in routed.w_k)


def test_specificity_selects_heads_without_changing_gate_amplitudes() -> None:
    config = make_config(
        head_router_top_k=2,
        head_router_selection="specificity",
        head_router_preserve_mass=True,
    )
    routed = HeadFactorizedGating(
        config,
        num_heads=4,
        embedding_dim_per_head=6,
        hidden_size=12,
        hc_mult=2,
        zero_init=False,
    )
    embeddings = torch.randn(1, 2, 4, 6)
    hidden = torch.randn(1, 2, 2, 12)
    specificity = torch.tensor([[[9.0, 8.0, -2.0, -3.0], [-2.0, -3.0, 9.0, 8.0]]])
    routed(embeddings, hidden, head_selection_scores=specificity)
    assert routed.last_gate is not None
    assert torch.equal(
        routed.last_gate.ne(0)[0, 0, 0], torch.tensor([True, True, False, False])
    )
    assert torch.equal(
        routed.last_gate.ne(0)[0, 1, 0], torch.tensor([False, False, True, True])
    )
    assert routed.last_route_logits is not None
    torch.testing.assert_close(
        routed.last_gate.sum(dim=-1), routed.last_route_logits.sigmoid().sum(dim=-1)
    )


def test_router_config_round_trip() -> None:
    config = make_config(
        head_router_top_k=4,
        head_router_preserve_mass=True,
        head_router_selection="specificity",
    )
    restored = EngramConfig.from_dict(config.to_dict())
    assert restored.memory_fusion == "head_factorized"
    assert restored.head_router_top_k == 4
    assert restored.head_router_preserve_mass is True
    assert restored.head_router_selection == "specificity"
