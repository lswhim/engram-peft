# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from sympy import nextprime


@dataclass
class NgramHashMapping:
    """
    Implements Multi-Head Hashing as described in the Engram paper section 2.2.
    Aligned with the official Engram demo implementation.
    """

    compressed_vocab_size: int
    engram_vocab_size_per_ngram: list[int] = field(
        default_factory=lambda: [2262400 // 2, 2262400 // 2]
    )
    ngram_sizes: list[int] = field(default_factory=lambda: [2, 3])
    max_ngram_size: int = 3
    n_head_per_ngram: int = 8
    layer_ids: list[int] = field(default_factory=lambda: [1, 14])
    pad_id: int = 2
    seed: int = 0

    # Internal state fields (excluded from __init__)
    all_multipliers: dict[int, np.ndarray] = field(init=False)
    prime_tables: dict[int, list[list[int]]] = field(init=False)

    def __post_init__(self) -> None:
        if self.compressed_vocab_size <= 0:
            raise ValueError("compressed_vocab_size must be a positive integer.")

        self.max_ngram_size = max(self.ngram_sizes)
        self.all_multipliers = {}

        for layer_id in self.layer_ids:
            # Layer-specific seed
            PRIME_1 = 10007
            base_seed = int(self.seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)

            # Heuristic bound based on derived compressed_vocab_size
            max_long = np.iinfo(np.int64).max
            M_max = int(max_long // self.compressed_vocab_size)
            half_bound = max(1, M_max // 2)

            # Generate max_ngram_size multipliers for this layer
            r = g.integers(
                low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64
            )
            # Must be odd numbers
            self.all_multipliers[layer_id] = r * 2 + 1

        self.prime_tables = self.calculate_vocab_size_across_layers()

    def find_next_prime(self, start: int, seen_primes: set[int]) -> int:
        """Finds the next unused global prime number strictly greater than start."""
        p_val = cast("int", nextprime(start))
        assert isinstance(p_val, int)
        p = p_val
        while p in seen_primes:
            p_val = cast("int", nextprime(p))
            assert isinstance(p_val, int)
            p = p_val
        return p

    def calculate_vocab_size_across_layers(self) -> dict[int, list[list[int]]]:
        """
        Calculates unique prime table sizes for all layers and heads.
        Matches the official demo's globally unique prime generation.
        Returns:
            Dict mapping layer_id -> List of Lists of primes [ngram_idx][head_idx]
        """
        seen_primes: set[int] = set()
        primes_across_layers: dict[int, list[list[int]]] = {}

        for layer_id in sorted(self.layer_ids):
            layer_primes: list[list[int]] = []
            for i in range(len(self.ngram_sizes)):
                head_primes: list[int] = []
                # Distribute the total bucket capacity among the heads
                base_vocab_size = (
                    self.engram_vocab_size_per_ngram[i] // self.n_head_per_ngram
                )
                current_start = base_vocab_size - 1

                for _ in range(self.n_head_per_ngram):
                    p = self.find_next_prime(current_start, seen_primes)
                    seen_primes.add(p)
                    head_primes.append(p)
                    current_start = p

                layer_primes.append(head_primes)
            primes_across_layers[layer_id] = layer_primes

        return primes_across_layers

    def _get_ngram_indices(self, input_ids: np.ndarray) -> dict[int, np.ndarray]:
        """
        Internal implementation of multi-head hashing using NumPy vectorization.
        Extracts ngrams once and broadcasts multipliers and primes across layers and heads.
        """
        batch_size, seq_len = input_ids.shape
        max_n = self.max_ngram_size

        # 1. Pad with pad_id for left context
        padding = np.full((batch_size, max_n - 1), self.pad_id, dtype=np.int64)
        padded_tokens = np.concatenate([padding, input_ids], axis=1).astype(np.int64)

        # 2. Extract sliding windows for all unique n-gram sizes
        # ngrams_cache: Dict[n_size, np.ndarray] of shape [batch_size, seq_len, n_size]
        ngrams_cache: dict[int, np.ndarray] = {}
        for n in set(self.ngram_sizes):
            view_shape = (batch_size, seq_len, n)
            strides = padded_tokens.strides + (padded_tokens.strides[-1],)
            ngrams_cache[n] = np.lib.stride_tricks.as_strided(
                padded_tokens, shape=view_shape, strides=strides
            )

        # 3. Pre-process multipliers and primes for efficient broadcasting
        # We process layer by layer but vectorize across heads.
        # Layer vectorization is possible but often memory-intensive if batch_size is large.
        # Head vectorization is the most critical for Engram (8x speedup).
        layer_results: dict[int, np.ndarray] = {}

        for layer_id in self.layer_ids:
            all_head_hashes: list[np.ndarray] = []
            multipliers = self.all_multipliers[layer_id]
            layer_primes = self.prime_tables[layer_id]

            for i, n in enumerate(self.ngram_sizes):
                # shape: [B, L, n]
                ngrams: np.ndarray = ngrams_cache[n]
                m = multipliers[:n]

                # weighted: [B, L, n]
                weighted: np.ndarray = ngrams * m

                # mix: [B, L]
                # Using reduce for bitwise_xor for better performance
                mix = np.bitwise_xor.reduce(weighted, axis=-1)

                # Vectorize across all heads for this ngram size
                # primes: [n_head_per_ngram]
                primes = np.array(layer_primes[i], dtype=np.int64)

                # Broadcasting mix [B, L] against primes [H] -> [B, L, H]
                # h = (mix % p + p) % p
                h = np.mod(np.mod(mix[..., np.newaxis], primes) + primes, primes)
                all_head_hashes.append(h)

            # Stack all n-gram heads along the last dimension
            # Result: [B, L, total_heads]
            layer_results[layer_id] = np.concatenate(all_head_hashes, axis=-1)

        return layer_results

    def hash(self, input_ids: torch.Tensor | np.ndarray) -> dict[int, np.ndarray]:
        """
        Compute hashes for all tracked layers using vectorized operations.

        Args:
            input_ids: Original compressed ids as torch Tensor or numpy Array.

        Returns:
            Dict mapping layer_id -> np.ndarray of hash indices [B, L, total_heads].
        """
        if isinstance(input_ids, torch.Tensor):
            arr = input_ids.cpu().numpy()
        else:
            arr = np.array(input_ids)

        return self._get_ngram_indices(arr)


@dataclass
class FixedNgramHashMapping:
    """Independent arithmetic heads with exactly equal table sizes.

    The original Engram arithmetic mapping intentionally uses a different prime
    modulus for every head.  It therefore cannot be parameter-matched exactly to
    an RQ table with K rows in every head.  This controlled backend fixes every
    head at ``total_capacity / n_head_per_ngram`` rows and obtains independence
    through separate multiplier vectors instead of unequal moduli.
    """

    compressed_vocab_size: int
    engram_vocab_size_per_ngram: list[int]
    ngram_sizes: list[int]
    layer_ids: list[int]
    pad_id: int
    n_head_per_ngram: int = 8
    seed: int = 0

    max_ngram_size: int = field(init=False)
    table_sizes: list[int] = field(init=False)
    prime_tables: dict[int, list[list[int]]] = field(init=False)
    all_head_multipliers: dict[int, list[np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        if self.compressed_vocab_size <= 0:
            raise ValueError("compressed_vocab_size must be a positive integer")
        if len(self.engram_vocab_size_per_ngram) != len(self.ngram_sizes):
            raise ValueError("one total capacity is required per n-gram size")
        self.max_ngram_size = max(self.ngram_sizes)
        self.table_sizes = []
        for total in self.engram_vocab_size_per_ngram:
            if total <= 0 or total % self.n_head_per_ngram != 0:
                raise ValueError(
                    f"capacity {total} must be positive and divisible by "
                    f"n_head_per_ngram={self.n_head_per_ngram}"
                )
            self.table_sizes.append(total // self.n_head_per_ngram)

        # Keep the established attribute name because model/layer construction
        # consumes it as generic per-head embedding table sizes.
        self.prime_tables = {
            layer_id: [
                [self.table_sizes[index]] * self.n_head_per_ngram
                for index in range(len(self.ngram_sizes))
            ]
            for layer_id in self.layer_ids
        }
        self.all_head_multipliers = {}
        max_long = np.iinfo(np.int64).max
        half_bound = max(1, int(max_long // self.compressed_vocab_size) // 2)
        for layer_id in self.layer_ids:
            per_order: list[np.ndarray] = []
            for order_index, ngram_size in enumerate(self.ngram_sizes):
                rng = np.random.default_rng(
                    int(self.seed + 10_007 * layer_id + 1_000_003 * order_index)
                )
                values = rng.integers(
                    0,
                    half_bound,
                    size=(self.n_head_per_ngram, ngram_size),
                    dtype=np.int64,
                )
                per_order.append(values * 2 + 1)
            self.all_head_multipliers[layer_id] = per_order

    def hash(
        self,
        input_ids: torch.Tensor | np.ndarray,
        original_ids: np.ndarray | None = None,
    ) -> dict[int, np.ndarray]:
        del original_ids
        if isinstance(input_ids, torch.Tensor):
            array = input_ids.detach().cpu().numpy()
        else:
            array = np.asarray(input_ids)
        if array.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, seq], got {array.shape}")

        batch_size, sequence_length = array.shape
        padding = np.full(
            (batch_size, self.max_ngram_size - 1), self.pad_id, dtype=np.int64
        )
        padded = np.concatenate([padding, array.astype(np.int64)], axis=1)
        cached: dict[int, np.ndarray] = {}
        for ngram_size in set(self.ngram_sizes):
            cached[ngram_size] = np.lib.stride_tricks.as_strided(
                padded,
                shape=(batch_size, sequence_length, ngram_size),
                strides=padded.strides + (padded.strides[-1],),
            )

        outputs: dict[int, np.ndarray] = {}
        for layer_id in self.layer_ids:
            chunks: list[np.ndarray] = []
            for order_index, ngram_size in enumerate(self.ngram_sizes):
                ngrams = cached[ngram_size][..., np.newaxis, :]
                multipliers = self.all_head_multipliers[layer_id][order_index]
                mixed = np.bitwise_xor.reduce(ngrams * multipliers, axis=-1)
                chunks.append(np.mod(mixed, self.table_sizes[order_index]))
            outputs[layer_id] = np.concatenate(chunks, axis=-1)
        return outputs
