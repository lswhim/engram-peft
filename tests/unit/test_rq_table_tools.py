from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engram_peft.rq_table_tools import (
    bucket_residual_signal,
    bucket_signal_to_interference,
    frequency_matched_codes,
    residual_energy_gains,
    shuffle_rq_table,
    shuffled_codes,
)


def test_residual_gains_and_bucket_snr_reward_explained_signal() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    codes = np.asarray([[0, 0], [0, 0], [1, 1]], dtype=np.uint16)
    codebooks = np.asarray(
        [
            [[0.7, 0.0], [0.0, 0.9]],
            [[0.2, 0.0], [0.0, 0.1]],
        ],
        dtype=np.float32,
    )
    gains = residual_energy_gains(vectors, codes, codebooks)
    assert gains.shape == codes.shape
    assert np.all(gains[:, 0] > 0)
    scores = bucket_signal_to_interference(codes, gains, codebook_size=2)
    assert scores.shape == (2, 2)
    assert np.isfinite(scores).all()


def test_bucket_snr_preserves_absolute_rq_level_strength() -> None:
    codes = np.asarray([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.uint16)
    gains = np.asarray(
        [[0.8, 0.02], [0.8, 0.02], [0.6, 0.01], [0.6, 0.01]],
        dtype=np.float32,
    )
    scores = bucket_signal_to_interference(codes, gains, codebook_size=2)

    # Loads are identical, so the level that removes much more normalized
    # residual energy must remain globally preferable.  A per-level z-score
    # would incorrectly make the two levels indistinguishable here.
    assert np.all(scores[0] > scores[1])


def test_residual_signal_ablation_does_not_penalize_bucket_load() -> None:
    codes = np.asarray([[0], [0], [1]], dtype=np.uint16)
    gains = np.asarray([[0.4], [0.4], [0.4]], dtype=np.float32)
    signal = bucket_residual_signal(codes, gains, codebook_size=2)
    snr = bucket_signal_to_interference(codes, gains, codebook_size=2)
    np.testing.assert_allclose(signal[0, 0], signal[0, 1], rtol=1e-6)
    assert snr[0, 0] < snr[0, 1]


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
        np.save(
            source / f"residual_gains_{ngram_size}.npy",
            np.full((8, 2), 0.25, dtype=np.float16),
        )
        (source / f"rq_{ngram_size}.faiss").write_bytes(f"rq-{ngram_size}".encode())
    (source / "projector_2.pt").write_bytes(b"projector")

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
        assert (first / f"rq_{ngram_size}.faiss").read_bytes() == f"rq-{ngram_size}".encode()
        np.testing.assert_array_equal(
            np.load(first / f"residual_gains_{ngram_size}.npy"),
            np.load(source / f"residual_gains_{ngram_size}.npy"),
        )
    assert (first / "projector_2.pt").read_bytes() == b"projector"
    assert manifest["ngram_sizes"]["2"]["runtime_artifacts"] == [
        "rq_2.faiss",
        "projector_2.pt",
    ]


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
