"""Per-head mixed hash: first K RQ semantic levels + then H arithmetic heads, per n-gram size.

Motivation: pure RQ aggressively clusters semantically-similar n-grams to the same slot
(slot conflict on template-heavy data like MQuAKE: 36 templates / 6015 facts, RQ -2pp).
Pure arith disperses every n-gram randomly (no semantic sharing).
Mixed = best of both: some heads share semantically (RQ), some always disperse (arith).

Layout per n-gram size (e.g. ngram_size=2): [rq_l0, rq_l1, ..., rq_l{R-1}, arith_h0, ..., arith_h{A-1}]
Total heads per layer = (R + A) * len(ngram_sizes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engram_peft.hashing import NgramHashMapping
from engram_peft.rq_hashing import RQNgramMapping


@dataclass
class MixedHashMapping:
    """RQ semantic hash + arith random hash combined per-head."""

    table_dir: str
    compressed_vocab_size: int
    engram_vocab_size_per_ngram: list[int]
    ngram_sizes: list[int]
    layer_ids: list[int]
    pad_id: int
    n_arith_heads_per_ngram: int  # how many arith heads per n-gram size
    n_rq_levels_used: int  # how many of RQ's M levels to use per n-gram size
    seed: int = 0

    rq: RQNgramMapping = field(init=False)
    arith: NgramHashMapping = field(init=False)
    total_heads: int = field(init=False)
    max_ngram_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.rq = RQNgramMapping(table_dir=self.table_dir, pad_id=self.pad_id)
        assert self.n_rq_levels_used <= self.rq.num_levels, (
            f"n_rq_levels_used={self.n_rq_levels_used} > RQ table M={self.rq.num_levels}"
        )
        self.arith = NgramHashMapping(
            engram_vocab_size_per_ngram=self.engram_vocab_size_per_ngram,
            ngram_sizes=self.ngram_sizes,
            n_head_per_ngram=self.n_arith_heads_per_ngram,
            layer_ids=self.layer_ids,
            compressed_vocab_size=self.compressed_vocab_size,
            pad_id=self.pad_id,
            seed=self.seed,
        )
        self.total_heads = (
            self.n_rq_levels_used + self.n_arith_heads_per_ngram
        ) * len(self.ngram_sizes)
        self.max_ngram_size = max(self.ngram_sizes)

    def flat_primes(self, layer_id: int) -> list[int]:
        """Per-head embedding table sizes for `layer_id`.

        Layout per n-gram size: [K, K, ..., K]*rq_used  then  [p0, p1, ...]*arith_heads
        Then concat across n-gram sizes.
        """
        K = self.rq.codebook_size
        arith_prime_list = self.arith.prime_tables[layer_id]  # [n_sizes][H]
        out: list[int] = []
        for i in range(len(self.ngram_sizes)):
            out += [K] * self.n_rq_levels_used
            out += list(arith_prime_list[i])
        return out

    def hash(self, input_ids) -> dict[int, np.ndarray]:
        """Compressed ids [B, L] -> per-layer codes [B, L, total_heads]."""
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.asarray(input_ids)

        rq_view = self.rq.hash(input_ids)
        rq_arr = rq_view._codes  # [B, L, M * n_sizes]
        arith_per_layer = self.arith.hash(input_ids)  # dict[layer_id, [B, L, H * n_sizes]]

        M = self.rq.num_levels
        H = self.n_arith_heads_per_ngram
        R = self.n_rq_levels_used
        n_sizes = len(self.ngram_sizes)

        out: dict[int, np.ndarray] = {}
        for layer_id in self.layer_ids:
            arith_arr = arith_per_layer[layer_id]
            chunks: list[np.ndarray] = []
            for i in range(n_sizes):
                # take first R of RQ's M for this n-gram size
                rq_chunk = rq_arr[..., i * M : i * M + R]
                arith_chunk = arith_arr[..., i * H : (i + 1) * H]
                chunks.append(rq_chunk)
                chunks.append(arith_chunk)
            out[layer_id] = np.concatenate(chunks, axis=-1)
        return out
