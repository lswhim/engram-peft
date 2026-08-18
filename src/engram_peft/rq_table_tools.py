"""Utilities for constructing controlled variants of frozen RQ address tables."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def residual_energy_gains(
    vectors: NDArray[np.floating[Any]],
    codes: NDArray[np.integer[Any]],
    codebooks: NDArray[np.floating[Any]],
) -> NDArray[np.float32]:
    """Measure how much squared residual energy each RQ level removes per row."""
    if vectors.ndim != 2 or codes.ndim != 2 or codebooks.ndim != 3:
        raise ValueError("vectors/codes/codebooks must have ranks 2/2/3")
    rows, dimension = vectors.shape
    levels = codes.shape[1]
    if codes.shape[0] != rows or codebooks.shape[0] != levels:
        raise ValueError("RQ row/level dimensions do not match")
    if codebooks.shape[2] != dimension:
        raise ValueError("RQ codebook dimension does not match vectors")
    if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= codebooks.shape[1]):
        raise ValueError("codes contain an out-of-range centroid id")

    residual = np.asarray(vectors, dtype=np.float32).copy()
    initial_energy = np.einsum("nd,nd->n", residual, residual).clip(min=1e-12)
    gains = np.empty((rows, levels), dtype=np.float32)
    for level in range(levels):
        before = np.einsum("nd,nd->n", residual, residual)
        residual -= codebooks[level, codes[:, level].astype(np.int64)]
        after = np.einsum("nd,nd->n", residual, residual)
        gains[:, level] = (before - after) / initial_energy
    return gains


def bucket_signal_to_interference(
    codes: NDArray[np.integer[Any]],
    residual_gains: NDArray[np.floating[Any]],
    codebook_size: int,
) -> NDArray[np.float32]:
    """Return per-level/bucket log semantic-signal-to-collision scores.

    Signal is the mean positive residual-energy reduction of rows assigned to a
    bucket. Interference is its distinct-row load. Their log ratio gives one
    scale-comparable reliability statistic without downstream labels.
    """
    if codes.shape != residual_gains.shape or codes.ndim != 2:
        raise ValueError("codes and residual_gains must have the same rank-2 shape")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    levels = codes.shape[1]
    scores = np.empty((levels, codebook_size), dtype=np.float32)
    positive = np.maximum(np.asarray(residual_gains, dtype=np.float32), 0.0)
    for level in range(levels):
        ids = codes[:, level].astype(np.int64)
        counts = np.bincount(ids, minlength=codebook_size).astype(np.float32)
        signal_sum = np.bincount(
            ids, weights=positive[:, level], minlength=codebook_size
        ).astype(np.float32)
        mean_signal = signal_sum / np.maximum(counts, 1.0)
        # Keep the absolute cross-level scale.  Residual gains are normalized by
        # each source vector's initial energy, so a coarse level explaining 60%
        # and a late level explaining 1% are directly comparable.  Per-level
        # z-scoring would erase exactly that RQ hierarchy and could promote a
        # relatively good but absolutely weak late-level code over a genuinely
        # informative early-level code.
        scores[level] = np.log(mean_signal.clip(min=1e-8)) - np.log1p(counts)
    return scores


def bucket_residual_signal(
    codes: NDArray[np.integer[Any]],
    residual_gains: NDArray[np.floating[Any]],
    codebook_size: int,
) -> NDArray[np.float32]:
    """Return absolute per-level/bucket log residual signal without load penalty."""
    if codes.shape != residual_gains.shape or codes.ndim != 2:
        raise ValueError("codes and residual_gains must have the same rank-2 shape")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    levels = codes.shape[1]
    scores = np.empty((levels, codebook_size), dtype=np.float32)
    positive = np.maximum(np.asarray(residual_gains, dtype=np.float32), 0.0)
    for level in range(levels):
        ids = codes[:, level].astype(np.int64)
        counts = np.bincount(ids, minlength=codebook_size).astype(np.float32)
        signal_sum = np.bincount(
            ids, weights=positive[:, level], minlength=codebook_size
        ).astype(np.float32)
        mean_signal = signal_sum / np.maximum(counts, 1.0)
        scores[level] = np.log(mean_signal.clip(min=1e-8))
    return scores


def shuffled_row_indices(row_count: int, seed: int) -> NDArray[np.int64]:
    """Return a deterministic row permutation for an RQ code matrix."""
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    return np.random.default_rng(seed).permutation(row_count).astype(np.int64)


def shuffled_codes(
    codes: NDArray[np.integer[Any]], seed: int
) -> tuple[NDArray[np.integer[Any]], NDArray[np.int64]]:
    """Shuffle complete RQ code vectors while preserving every level histogram.

    A complete row is reassigned to another n-gram.  This preserves both the
    per-level bucket loads and the joint distribution of multi-level codes, so
    the only removed property is which n-gram owns which semantic address.
    """
    if codes.ndim != 2:
        raise ValueError(f"codes must be rank 2, got shape={codes.shape}")
    permutation = shuffled_row_indices(codes.shape[0], seed)
    return codes[permutation].copy(), permutation


def frequency_matched_row_indices(
    access_counts: NDArray[np.integer[Any]], seed: int
) -> NDArray[np.int64]:
    """Permute only among rows with exactly equal runtime access counts.

    Because every source and destination row inside a group has the same
    weight, this preserves each code bucket's access-weighted load exactly,
    while destroying semantic ownership whenever a group has multiple rows.
    """
    if access_counts.ndim != 1:
        raise ValueError("access_counts must be rank 1")
    if access_counts.size and int(access_counts.min()) < 0:
        raise ValueError("access_counts must be non-negative")
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(access_counts), dtype=np.int64)
    for count in np.unique(access_counts):
        group = np.flatnonzero(access_counts == count)
        if len(group) > 1:
            permutation[group] = rng.permutation(group)
    return permutation


def frequency_matched_codes(
    codes: NDArray[np.integer[Any]],
    access_counts: NDArray[np.integer[Any]],
    seed: int,
) -> tuple[NDArray[np.integer[Any]], NDArray[np.int64]]:
    if codes.ndim != 2:
        raise ValueError(f"codes must be rank 2, got shape={codes.shape}")
    if len(codes) != len(access_counts):
        raise ValueError("codes and access_counts must have the same row count")
    permutation = frequency_matched_row_indices(access_counts, seed)
    return codes[permutation].copy(), permutation


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def copy_rq_runtime_artifacts(
    source_dir: Path, output_dir: Path, ngram_sizes: list[int]
) -> dict[int, list[str]]:
    """Copy frozen quantizers/projectors required for dynamic OOV encoding."""
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[int, list[str]] = {}
    for ngram_size in ngram_sizes:
        copied[ngram_size] = []
        for artifact_name in (
            f"rq_{ngram_size}.faiss",
            f"projector_{ngram_size}.pt",
        ):
            source_artifact = source_dir / artifact_name
            if source_artifact.is_file():
                shutil.copy2(source_artifact, output_dir / artifact_name)
                copied[ngram_size].append(artifact_name)
    return copied


def shuffle_rq_table(
    source_dir: Path,
    output_dir: Path,
    seed: int,
    access_counts_path: Path | None = None,
) -> dict[str, Any]:
    """Create an auditable RQ-Shuffled table without changing bucket loads."""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("source_dir and output_dir must differ")
    if not (source_dir / "meta.json").is_file():
        raise FileNotFoundError(source_dir / "meta.json")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = _read_json(source_dir / "meta.json")
    ngram_sizes = [int(value) for value in meta["ngram_sizes"]]
    num_levels = int(meta["num_levels"])
    codebook_size = int(meta["codebook_size"])
    runtime_artifacts = copy_rq_runtime_artifacts(
        source_dir, output_dir, ngram_sizes
    )
    per_size: dict[str, Any] = {}
    access_payload = (
        np.load(access_counts_path, allow_pickle=False)
        if access_counts_path is not None
        else None
    )

    for offset, ngram_size in enumerate(ngram_sizes):
        keys = np.load(source_dir / f"keys_{ngram_size}.npy", allow_pickle=False)
        codes = np.load(source_dir / f"codes_{ngram_size}.npy", allow_pickle=False)
        if keys.ndim != 1:
            raise ValueError(f"keys_{ngram_size}.npy must be rank 1")
        if codes.shape != (keys.shape[0], num_levels):
            raise ValueError(
                f"codes_{ngram_size}.npy has shape {codes.shape}; "
                f"expected {(keys.shape[0], num_levels)}"
            )
        if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= codebook_size):
            raise ValueError(f"codes_{ngram_size}.npy contains an out-of-range code")

        level_seed = seed + offset * 1_000_003
        access_counts = None
        if access_payload is not None:
            field = f"train_access_count_{ngram_size}"
            if field not in access_payload:
                raise KeyError(f"{field} missing from {access_counts_path}")
            access_counts = np.asarray(access_payload[field], dtype=np.int64)
            shuffled, permutation = frequency_matched_codes(
                codes, access_counts, level_seed
            )
        else:
            shuffled, permutation = shuffled_codes(codes, level_seed)
        np.save(output_dir / f"keys_{ngram_size}.npy", keys)
        np.save(output_dir / f"codes_{ngram_size}.npy", shuffled)
        np.save(output_dir / f"permutation_{ngram_size}.npy", permutation)
        gains_path = source_dir / f"residual_gains_{ngram_size}.npy"
        if gains_path.is_file():
            gains = np.load(gains_path, allow_pickle=False)
            if gains.shape != codes.shape:
                raise ValueError(
                    f"{gains_path.name} has shape {gains.shape}; expected {codes.shape}"
                )
            # Gains describe each destination n-gram's semantic RQ signal and stay
            # aligned with keys. Re-aggregating them against shuffled codes destroys
            # semantic ownership while retaining the exact same score definition.
            np.save(output_dir / gains_path.name, gains)

        # A shuffled table must remain a complete runtime RQ address function,
        # not merely an offline key/code dictionary.  Dynamic OOVs use the same
        # frozen quantizer (and optional projector) as the semantic source table.
        histograms_preserved = all(
            np.array_equal(
                np.bincount(codes[:, level].astype(np.int64), minlength=codebook_size),
                np.bincount(
                    shuffled[:, level].astype(np.int64), minlength=codebook_size
                ),
            )
            for level in range(num_levels)
        )
        moved_fraction = (
            float(np.mean(permutation != np.arange(permutation.size)))
            if permutation.size
            else 0.0
        )
        weighted_preserved = None
        accessed_moved_fraction = None
        singleton_rows = None
        if access_counts is not None:
            weighted_preserved = all(
                np.array_equal(
                    np.bincount(
                        codes[:, level].astype(np.int64),
                        weights=access_counts,
                        minlength=codebook_size,
                    ),
                    np.bincount(
                        shuffled[:, level].astype(np.int64),
                        weights=access_counts,
                        minlength=codebook_size,
                    ),
                )
                for level in range(num_levels)
            )
            accessed = access_counts > 0
            accessed_moved_fraction = (
                float(np.mean(permutation[accessed] != np.flatnonzero(accessed)))
                if accessed.any()
                else 0.0
            )
            _, group_sizes = np.unique(access_counts, return_counts=True)
            singleton_rows = int(group_sizes[group_sizes == 1].sum())
        per_size[str(ngram_size)] = {
            "rows": int(keys.shape[0]),
            "seed": level_seed,
            "moved_fraction": moved_fraction,
            "level_histograms_preserved": histograms_preserved,
            "access_weighted_histograms_preserved": weighted_preserved,
            "accessed_rows_moved_fraction": accessed_moved_fraction,
            "singleton_frequency_rows": singleton_rows,
            "runtime_artifacts": runtime_artifacts[ngram_size],
        }

    shuffled_meta = dict(meta)
    shuffled_meta.update(
        {
            "address_variant": (
                "rq_shuffled_frequency_matched"
                if access_counts_path is not None
                else "rq_shuffled"
            ),
            "shuffle_seed": seed,
            "shuffle_strategy": (
                "whole_code_vector_permutation_within_exact_access_frequency"
                if access_counts_path is not None
                else "whole_code_vector_row_permutation"
            ),
            "source_table": str(source_dir),
            # Offline rows use the exact frequency/load-matched permutation above.
            # Truly unseen runtime codes still need their semantic ownership removed.
            "runtime_oov_shuffle_seed": seed,
            "runtime_oov_shuffle_protocol": "blake2b_joint_code_vector_v1",
        }
    )
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(shuffled_meta, handle, indent=2)

    manifest = {
        "status": "complete",
        "source_table": str(source_dir),
        "output_table": str(output_dir),
        "shuffle_seed": seed,
        "access_counts": str(access_counts_path) if access_counts_path else None,
        "ngram_sizes": per_size,
    }
    with (output_dir / "shuffle_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
