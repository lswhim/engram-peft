from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engram_peft.rq_table_tools import (
    frequency_matched_codes,
    shuffle_rq_table,
    shuffled_codes,
)


def test_shuffled_codes_preserves_level_histograms_and_joint_rows() -> None:
    codes = np.asarray(
        [[0, 1], [0, 2], [1, 1], [2, 0], [2, 2]], dtype=np.uint16
    )
    shuffled, permutation = shuffled_codes(codes, seed=7)

    assert not np.array_equal(permutation, np.arange(len(codes)))
    assert sorted(map(tuple, shuffled.tolist())) == sorted(map(tuple, codes.tolist()))
    for level in range(codes.shape[1]):
        assert np.array_equal(
            np.bincount(shuffled[:, level]), np.bincount(codes[:, level])
        )


def test_shuffle_rq_table_is_deterministic_and_auditable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "meta.json").write_text(
        json.dumps(
            {
                "base": 101,
                "ngram_sizes": [2, 3],
                "num_levels": 2,
                "codebook_size": 4,
            }
        ),
        encoding="utf-8",
    )
    for ngram_size in (2, 3):
        np.save(source / f"keys_{ngram_size}.npy", np.arange(8, dtype=np.int64))
        np.save(
            source / f"codes_{ngram_size}.npy",
            np.asarray([[i % 4, (i // 2) % 4] for i in range(8)], dtype=np.uint16),
        )

    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = shuffle_rq_table(source, first, seed=11)
    shuffle_rq_table(source, second, seed=11)

    assert manifest["status"] == "complete"
    assert manifest["ngram_sizes"]["2"]["level_histograms_preserved"] is True
    assert json.loads((first / "meta.json").read_text())["address_variant"] == "rq_shuffled"
    for ngram_size in (2, 3):
        assert np.array_equal(
            np.load(first / f"keys_{ngram_size}.npy"),
            np.load(source / f"keys_{ngram_size}.npy"),
        )
        assert np.array_equal(
            np.load(first / f"codes_{ngram_size}.npy"),
            np.load(second / f"codes_{ngram_size}.npy"),
        )


def test_shuffle_rq_table_refuses_nonempty_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "meta.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError):
        shuffle_rq_table(source, output, seed=1)


def test_frequency_matched_shuffle_preserves_weighted_bucket_loads() -> None:
    codes = np.asarray(
        [[0, 0], [1, 1], [0, 1], [1, 0], [2, 2], [3, 3]], dtype=np.uint16
    )
    counts = np.asarray([0, 0, 1, 1, 1, 7], dtype=np.int64)
    shuffled, permutation = frequency_matched_codes(codes, counts, seed=19)

    assert np.array_equal(counts[permutation], counts)
    assert permutation[-1] == len(counts) - 1  # unique-frequency rows cannot move
    for level in range(codes.shape[1]):
        original = np.bincount(codes[:, level], weights=counts, minlength=4)
        control = np.bincount(shuffled[:, level], weights=counts, minlength=4)
        assert np.array_equal(original, control)
