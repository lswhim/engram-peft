# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none
"""
Semantic-hash address mapping for Engram (RQ variant).

Drop-in replacement for NgramHashMapping: same `.hash(compressed_ids) -> {layer_id:
ndarray[B, L, total_heads]}` interface, so MultiHeadEmbedding / ShortConv / gating are
untouched.

Instead of an arithmetic XOR-multiply hash on token ids, the bucket addresses come from a
frozen, offline-built table that maps each compressed n-gram to M residual-quantization
codes (semantically structured sharing). See scripts/build_rq_table.py.

  * total_heads = num_levels (M) x len(ngram_sizes)   -- matches the arithmetic layout
    [ngram2_level0..M-1, ngram3_level0..M-1, ...]; each "head" indexes a size-K table.
  * Codes are layer-independent (the semantic address does not depend on layer); each
    EngramLayer still owns its own embedding table, looked up with these codes.
  * OOV n-grams (not in the offline table) fall back to a deterministic arithmetic hash
    into [0, K), preserving train/inference consistency and O(1) lookup.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

# fixed odd multipliers for the OOV fallback hash (per level), deterministic
_OOV_MULTS = np.array(
    [0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F,
     0x165667B1, 0xD3A2646D, 0xFD7046C5, 0xB55A4F09],
    dtype=np.int64,
)


@dataclass
class RQNgramMapping:
    """Frozen RQ semantic-hash mapping. Loaded from an offline-built table dir."""

    table_dir: str
    pad_id: int = 2

    # filled in __post_init__
    base: int = field(init=False)
    ngram_sizes: list[int] = field(init=False)
    num_levels: int = field(init=False)
    codebook_size: int = field(init=False)
    sorted_keys: dict[int, np.ndarray] = field(init=False)
    codes: dict[int, np.ndarray] = field(init=False)
    total_heads: int = field(init=False)

    def __post_init__(self) -> None:
        with open(os.path.join(self.table_dir, "meta.json")) as f:
            meta = json.load(f)
        self.base = int(meta["base"])
        self.ngram_sizes = list(meta["ngram_sizes"])
        self.num_levels = int(meta["num_levels"])
        self.codebook_size = int(meta["codebook_size"])
        self.sorted_keys = {}
        self.codes = {}
        for n in self.ngram_sizes:
            self.sorted_keys[n] = np.load(os.path.join(self.table_dir, f"keys_{n}.npy"))
            self.codes[n] = np.load(
                os.path.join(self.table_dir, f"codes_{n}.npy")
            ).astype(np.int64)
        self.total_heads = self.num_levels * len(self.ngram_sizes)
        self.max_ngram_size = max(self.ngram_sizes)

    # --- mirrors NgramHashMapping field used by MultiHeadEmbedding sizing ---
    @property
    def primes(self) -> list[int]:
        """Per-head table sizes (= K for every RQ level head)."""
        return [self.codebook_size] * self.total_heads

    def _poly_keys(self, ngrams: np.ndarray) -> np.ndarray:
        """[*, n] compressed-id windows -> [*] int64 poly keys (base-radix)."""
        k = np.zeros(ngrams.shape[:-1], dtype=np.int64)
        for j in range(ngrams.shape[-1]):
            k = k * self.base + ngrams[..., j].astype(np.int64)
        return k

    def _oov_codes(self, keys: np.ndarray) -> np.ndarray:
        """Deterministic fallback codes [*, M] in [0, K) for unseen keys."""
        K = self.codebook_size
        mults = _OOV_MULTS[: self.num_levels]
        mixed = (keys[..., None] * mults) ^ (keys[..., None] >> np.int64(7))
        return np.mod(np.mod(mixed, K) + K, K)

    def _codes_for_ngram_size(self, cids: np.ndarray, n: int) -> np.ndarray:
        """Return [B, L, M] codes for n-gram size n, OOV -> fallback."""
        b, length = cids.shape
        pad = np.full((b, n - 1), self.pad_id, dtype=np.int64)
        padded = np.concatenate([pad, cids.astype(np.int64)], axis=1)
        # suffix windows aligned to each output position: [B, L, n]
        windows = np.stack([padded[:, i : i + length] for i in range(n)], axis=-1)
        keys = self._poly_keys(windows)  # [B, L]

        sk = self.sorted_keys[n]
        out = self._oov_codes(keys)  # [B, L, M] fallback default
        if sk.size > 0:
            pos = np.searchsorted(sk, keys)
            pos_clipped = np.clip(pos, 0, sk.size - 1)
            hit = sk[pos_clipped] == keys
            if hit.any():
                out[hit] = self.codes[n][pos_clipped[hit]]
        return out

    def hash(self, input_ids: np.ndarray) -> dict[int, np.ndarray]:
        """Compressed ids [B, L] -> {layer_id-agnostic} [B, L, total_heads] codes.

        Returns a dict-like keyed by any requested layer_id via _LayerView so callers can
        do `.hash(c)[layer_id]` exactly like NgramHashMapping.
        """
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.asarray(input_ids)
        per_size = [self._codes_for_ngram_size(input_ids, n) for n in self.ngram_sizes]
        codes = np.concatenate(per_size, axis=-1)  # [B, L, total_heads]
        return _LayerView(codes)


class _LayerView(dict):
    """Returns the same layer-independent codes for any layer_id key."""

    def __init__(self, codes: np.ndarray) -> None:
        super().__init__()
        self._codes = codes

    def __getitem__(self, _layer_id: int) -> np.ndarray:  # type: ignore[override]
        return self._codes
