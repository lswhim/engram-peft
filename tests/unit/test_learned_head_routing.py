from __future__ import annotations

import torch

from engram_peft.config import EngramConfig
from engram_peft.layer import SemanticKeyedGating


def make_config(**overrides: object) -> EngramConfig:
    values = {
        "hidden_size": 12,
        "embedding_dim": 24,
        "hc_mult": 2,
        "memory_fusion": "head_factorized",
    }
    values.update(overrides)
    return EngramConfig(**values)


def test_semantic_keyed_descriptors_follow_rq_level_order() -> None:
    config = make_config(
        head_router_selection="semantic_keyed",
    )
    codebooks = torch.zeros(4, 3, 2)
    for head in range(4):
        for code in range(3):
            codebooks[head, code] = torch.tensor([10 * head + code, 1.0])
    routed = SemanticKeyedGating(
        config,
        codebooks=codebooks,
        num_levels=2,
        embedding_dim_per_head=6,
        hidden_size=12,
        hc_mult=2,
        zero_init=False,
    )
    indices = torch.tensor([[[0, 1, 2, 0]]])
    centroid, prefix, tail = routed.semantic_descriptors(indices)
    torch.testing.assert_close(centroid[0, 0, :, 0], torch.tensor([0.0, 11.0, 22.0, 30.0]))
    torch.testing.assert_close(prefix[0, 0, :, 0], torch.tensor([0.0, 11.0, 22.0, 52.0]))
    torch.testing.assert_close(tail[0, 0, :, 0], torch.tensor([11.0, 0.0, 30.0, 0.0]))


def test_semantic_keyed_routes_do_not_use_memory_values_as_keys() -> None:
    torch.manual_seed(19)
    config = make_config(
        head_router_selection="semantic_keyed",
    )
    routed = SemanticKeyedGating(
        config,
        codebooks=torch.randn(4, 5, 7),
        num_levels=2,
        embedding_dim_per_head=6,
        hidden_size=12,
        hc_mult=2,
        zero_init=False,
    )
    hidden = torch.randn(2, 3, 2, 12)
    indices = torch.randint(0, 5, (2, 3, 4))
    first = routed(torch.randn(2, 3, 4, 6), hidden, indices)
    assert routed.last_route_logits is not None
    first_logits = routed.last_route_logits.detach().clone()
    second = routed(torch.randn(2, 3, 4, 6), hidden, indices)
    assert routed.last_route_logits is not None
    torch.testing.assert_close(routed.last_route_logits, first_logits)
    assert not torch.allclose(first, second)


def test_semantic_keyed_router_and_values_receive_task_gradient() -> None:
    config = make_config(
        head_router_selection="semantic_keyed",
    )
    routed = SemanticKeyedGating(
        config,
        codebooks=torch.randn(4, 5, 7),
        num_levels=2,
        embedding_dim_per_head=6,
        hidden_size=12,
        hc_mult=2,
        zero_init=False,
    )
    output = routed(
        torch.randn(2, 3, 4, 6),
        torch.randn(2, 3, 2, 12),
        torch.randint(0, 5, (2, 3, 4)),
    )
    output.square().mean().backward()
    assert routed.semantic_proj.weight.grad is not None
    assert torch.count_nonzero(routed.semantic_proj.weight.grad) > 0
    assert all(key.weight.grad is not None for key in routed.w_k)
    assert all(torch.count_nonzero(key.weight.grad) > 0 for key in routed.w_k)
    assert routed.w_v.weight.grad is not None
    assert torch.count_nonzero(routed.w_v.weight.grad) > 0
