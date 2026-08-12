"""Utilities for constructing controlled variants of frozen RQ address tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


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
