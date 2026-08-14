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
  * The semantic backend is strict: every runtime n-gram must have been encoded with the
    frozen semantic encoder and RQ codebook into the table. Missing keys raise an error;
    they never silently change the method into arithmetic hashing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

@dataclass
class RQNgramMapping:
    """Frozen RQ semantic-hash mapping. Loaded from an offline-built table dir."""

    table_dir: str
    pad_id: int = 2
    cache_dir: str | None = None
    embed_device: str = "cuda"
    embed_batch_size: int = 256

    # filled in __post_init__
    base: int = field(init=False)
    ngram_sizes: list[int] = field(init=False)
    num_levels: int = field(init=False)
    codebook_size: int = field(init=False)
    sorted_keys: dict[int, np.ndarray] = field(init=False)
    codes: dict[int, np.ndarray] = field(init=False)
    total_heads: int = field(init=False)
    _cache: sqlite3.Connection | None = field(init=False, default=None, repr=False)
    _runtime_codes: dict[int, dict[int, np.ndarray]] = field(
        init=False, default_factory=dict, repr=False
    )
    _tokenizer: Any = field(init=False, default=None, repr=False)
    _embedder: Any = field(init=False, default=None, repr=False)
    _rq_indices: dict[int, Any] = field(init=False, default_factory=dict, repr=False)
    _projectors: dict[int, Any] = field(init=False, default_factory=dict, repr=False)
    _meta: dict[str, Any] = field(init=False, repr=False)
    _trace_enabled: bool = field(init=False, default=False, repr=False)
    _traced_rows: set[tuple[int, int, int]] = field(
        init=False, default_factory=set, repr=False
    )
    _traced_row_counts: Counter[tuple[int, int, int]] = field(
        init=False, default_factory=Counter, repr=False
    )

    def __post_init__(self) -> None:
        with open(os.path.join(self.table_dir, "meta.json")) as f:
            meta = json.load(f)
        self._meta = meta
        self.base = int(meta["base"])
        self.ngram_sizes = list(meta["ngram_sizes"])
        self.num_levels = int(meta["num_levels"])
        self.codebook_size = int(meta["codebook_size"])
        self.runtime_shuffle_seed = meta.get("runtime_shuffle_seed")
        self.sorted_keys = {}
        self.codes = {}
        for n in self.ngram_sizes:
            self.sorted_keys[n] = np.load(os.path.join(self.table_dir, f"keys_{n}.npy"))
            self.codes[n] = np.load(
                os.path.join(self.table_dir, f"codes_{n}.npy")
            ).astype(np.int64)
            self.codes[n] = self._shuffle_codes(n, self.codes[n])
        self.total_heads = self.num_levels * len(self.ngram_sizes)
        self.max_ngram_size = max(self.ngram_sizes)
        if self.cache_dir is not None:
            cache_path = Path(self.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self._cache = sqlite3.connect(cache_path / "semantic_codes.sqlite3")
            self._cache.execute(
                "CREATE TABLE IF NOT EXISTS codes "
                "(n INTEGER NOT NULL, key INTEGER NOT NULL, code BLOB NOT NULL, "
                "PRIMARY KEY (n, key))"
            )
            self._cache.commit()
            self._runtime_codes = {n: {} for n in self.ngram_sizes}
            for n, key, code in self._cache.execute("SELECT n, key, code FROM codes"):
                if int(n) in self._runtime_codes:
                    self._runtime_codes[int(n)][int(key)] = np.frombuffer(
                        code, dtype=np.uint16
                    ).astype(np.int64)

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

    def _shuffle_codes(self, n: int, codes: np.ndarray) -> np.ndarray:
        """Destroy partial-level RQ geometry with a frozen joint-code mapping."""
        if self.runtime_shuffle_seed is None:
            return codes
        output = np.empty_like(codes, dtype=np.int64)
        seed = int(self.runtime_shuffle_seed)
        for row_index, row in enumerate(codes):
            payload = np.asarray(row, dtype=np.uint32).tobytes()
            digest = hashlib.blake2b(
                payload,
                digest_size=8 * self.num_levels,
                person=f"rq{seed}:{n}".encode()[:16],
            ).digest()
            output[row_index] = (
                np.frombuffer(digest, dtype=np.uint64) % self.codebook_size
            ).astype(np.int64)
        return output

    def _load_encoder(self) -> None:
        if self._embedder is not None:
            return
        for n in self.ngram_sizes:
            path = os.path.join(self.table_dir, f"rq_{n}.faiss")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{path} is missing; rebuild the RQ table with its frozen quantizer"
                )
        import faiss
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._meta["embedder"], trust_remote_code=True
        )
        self._embedder = AutoModel.from_pretrained(
            self._meta["embedder"], trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.embed_device).eval()
        for parameter in self._embedder.parameters():
            parameter.requires_grad_(False)
        self._rq_indices = {
            n: faiss.read_index(os.path.join(self.table_dir, f"rq_{n}.faiss"))
            for n in self.ngram_sizes
        }
        if self._meta.get("projection") == "autoencoder":
            for n in self.ngram_sizes:
                payload = torch.load(
                    os.path.join(self.table_dir, f"projector_{n}.pt"),
                    map_location=self.embed_device,
                    weights_only=True,
                )
                projector = torch.nn.Sequential(
                    torch.nn.Linear(payload["input_dim"], payload["hidden_dim"]),
                    torch.nn.GELU(),
                    torch.nn.Linear(payload["hidden_dim"], payload["projection_dim"]),
                ).to(self.embed_device)
                projector.load_state_dict(payload["state_dict"])
                projector.eval()
                for parameter in projector.parameters():
                    parameter.requires_grad_(False)
                self._projectors[n] = projector

    def _encode_missing(
        self, n: int, keys: np.ndarray, original_windows: np.ndarray
    ) -> np.ndarray:
        if self._cache is None:
            raise KeyError(
                "semantic RQ encountered unseen n-grams but rq_cache_dir is unset"
            )
        self._load_encoder()
        import torch

        texts = [
            self._tokenizer.decode(row.tolist(), skip_special_tokens=False)
            for row in original_windows
        ]
        encoded: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.embed_batch_size):
                batch = self._tokenizer(
                    texts[start : start + self.embed_batch_size],
                    padding=True,
                    truncation=True,
                    max_length=32,
                    return_tensors="pt",
                ).to(self.embed_device)
                hidden = self._embedder(**batch).last_hidden_state
                lengths = batch["attention_mask"].sum(dim=1) - 1
                vectors = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
                encoded.append(vectors.float().cpu().numpy())
        vectors_np = np.ascontiguousarray(np.concatenate(encoded).astype(np.float32))
        if n in self._projectors:
            with torch.inference_mode():
                projected = self._projectors[n](
                    torch.from_numpy(vectors_np).to(self.embed_device)
                )
                projected = torch.nn.functional.normalize(projected, dim=-1)
            vectors_np = np.ascontiguousarray(projected.float().cpu().numpy())
        packed = self._rq_indices[n].sa_encode(vectors_np)
        import faiss

        nbits = int(np.log2(self.codebook_size))
        codes = np.asarray(
            faiss.unpack_bitstrings(packed, self.num_levels, nbits), dtype=np.int64
        )
        codes = self._shuffle_codes(n, codes)
        self._cache.executemany(
            "INSERT OR IGNORE INTO codes(n, key, code) VALUES (?, ?, ?)",
            [
                (n, int(key), sqlite3.Binary(code.astype(np.uint16).tobytes()))
                for key, code in zip(keys, codes, strict=True)
            ],
        )
        self._cache.commit()
        self._runtime_codes[n].update(
            {int(key): code.copy() for key, code in zip(keys, codes, strict=True)}
        )
        return codes

    def _codes_for_ngram_size(
        self, cids: np.ndarray, original_ids: np.ndarray | None, n: int
    ) -> np.ndarray:
        """Return [B, L, M] codes, refusing keys not semantically encoded offline."""
        b, length = cids.shape
        pad = np.full((b, n - 1), self.pad_id, dtype=np.int64)
        padded = np.concatenate([pad, cids.astype(np.int64)], axis=1)
        # suffix windows aligned to each output position: [B, L, n]
        windows = np.stack([padded[:, i : i + length] for i in range(n)], axis=-1)
        keys = self._poly_keys(windows)  # [B, L]

        sk = self.sorted_keys[n]
        if sk.size == 0:
            raise KeyError(f"semantic RQ table has no {n}-gram entries")
        pos = np.searchsorted(sk, keys)
        pos_clipped = np.clip(pos, 0, sk.size - 1)
        hit = sk[pos_clipped] == keys
        out = np.empty((*keys.shape, self.num_levels), dtype=np.int64)
        out[hit] = self.codes[n][pos_clipped[hit]]
        if not hit.all():
            if original_ids is None:
                raise KeyError("unseen semantic n-grams require original_ids")
            original_pad = np.full(
                (b, n - 1), int(self._meta.get("pad_token_id", 0)), dtype=np.int64
            )
            original_padded = np.concatenate([original_pad, original_ids.astype(np.int64)], axis=1)
            original_windows = np.stack(
                [original_padded[:, i : i + length] for i in range(n)], axis=-1
            )
            missing_keys, inverse = np.unique(keys[~hit], return_inverse=True)
            cached = {
                int(key): self._runtime_codes[n][int(key)]
                for key in missing_keys
                if int(key) in self._runtime_codes.get(n, {})
            }
            needs = [i for i, key in enumerate(missing_keys) if int(key) not in cached]
            if needs:
                flat_keys = keys.reshape(-1)
                flat_original = original_windows.reshape(-1, n)
                first_positions = {int(key): int(np.flatnonzero(flat_keys == key)[0]) for key in missing_keys[needs]}
                new_windows = np.stack([flat_original[first_positions[int(missing_keys[i])]] for i in needs])
                new_codes = self._encode_missing(n, missing_keys[needs], new_windows)
                self._runtime_codes.setdefault(n, {}).update(
                    {
                        int(missing_keys[i]): code.copy()
                        for i, code in zip(needs, new_codes, strict=True)
                    }
                )
                cached.update({int(missing_keys[i]): code for i, code in zip(needs, new_codes, strict=True)})
            unique_codes = np.stack([cached[int(key)] for key in missing_keys])
            out[~hit] = unique_codes[inverse]
        return out

    def hash(
        self, input_ids: np.ndarray, original_ids: np.ndarray | None = None
    ) -> dict[int, np.ndarray]:
        """Compressed ids [B, L] -> {layer_id-agnostic} [B, L, total_heads] codes.

        Returns a dict-like keyed by any requested layer_id via _LayerView so callers can
        do `.hash(c)[layer_id]` exactly like NgramHashMapping.
        """
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.asarray(input_ids)
        per_size = [
            self._codes_for_ngram_size(input_ids, original_ids, n)
            for n in self.ngram_sizes
        ]
        if self._trace_enabled:
            for n, ngram_codes in zip(self.ngram_sizes, per_size, strict=True):
                for level in range(self.num_levels):
                    self._traced_rows.update(
                        (n, level, int(code))
                        for code in np.unique(ngram_codes[..., level])
                    )
                    self._traced_row_counts.update(
                        (n, level, int(code))
                        for code in ngram_codes[..., level].reshape(-1)
                    )
        codes = np.concatenate(per_size, axis=-1)  # [B, L, total_heads]
        return _LayerView(codes)

    def start_row_trace(self, *, clear: bool = True) -> None:
        """Record unique physical memory rows addressed by subsequent forwards."""
        if clear:
            self._traced_rows.clear()
            self._traced_row_counts.clear()
        self._trace_enabled = True

    def stop_row_trace(self) -> set[tuple[int, int, int]]:
        self._trace_enabled = False
        return set(self._traced_rows)

    def traced_row_counts(self) -> dict[tuple[int, int, int], int]:
        return dict(self._traced_row_counts)


class _LayerView(dict):
    """Returns the same layer-independent codes for any layer_id key."""

    def __init__(self, codes: np.ndarray) -> None:
        super().__init__()
        self._codes = codes

    def __getitem__(self, _layer_id: int) -> np.ndarray:  # type: ignore[override]
        return self._codes
