import json
from pathlib import Path

import numpy as np
import pytest

from engram_peft.rq_hashing import RQNgramMapping


def _table(path: Path) -> None:
    path.mkdir()
    (path / "meta.json").write_text(
        json.dumps(
            {
                "base": 100,
                "ngram_sizes": [2],
                "num_levels": 2,
                "codebook_size": 8,
            }
        )
    )
    # With pad_id=0, input [1, 2] produces keys 1 and 102.
    np.save(path / "keys_2.npy", np.asarray([1, 102], dtype=np.int64))
    np.save(path / "codes_2.npy", np.asarray([[3, 4], [5, 6]], dtype=np.uint16))


def test_semantic_rq_uses_preencoded_codes(tmp_path: Path) -> None:
    _table(tmp_path / "table")
    mapping = RQNgramMapping(str(tmp_path / "table"), pad_id=0)
    actual = mapping.hash(np.asarray([[1, 2]], dtype=np.int64))[7]
    np.testing.assert_array_equal(actual, np.asarray([[[3, 4], [5, 6]]]))


def test_signal_to_interference_uses_residual_gain_artifact(tmp_path: Path) -> None:
    table = tmp_path / "table"
    _table(table)
    np.save(
        table / "residual_gains_2.npy",
        np.asarray([[0.8, 0.2], [0.1, 0.7]], dtype=np.float16),
    )
    mapping = RQNgramMapping(str(table), pad_id=0)
    scores = mapping.signal_to_interference_table()
    assert scores.shape == (2, 8)
    assert np.isfinite(scores).all()


def test_signal_to_interference_requires_gain_artifact(tmp_path: Path) -> None:
    _table(tmp_path / "table")
    mapping = RQNgramMapping(str(tmp_path / "table"), pad_id=0)
    with pytest.raises(FileNotFoundError, match="residual gain artifacts"):
        mapping.signal_to_interference_table()


def test_semantic_rq_never_silently_falls_back_without_lazy_cache(
    tmp_path: Path,
) -> None:
    _table(tmp_path / "table")
    mapping = RQNgramMapping(str(tmp_path / "table"), pad_id=0)
    with pytest.raises(KeyError, match="rq_cache_dir is unset"):
        mapping.hash(
            np.asarray([[1, 9]], dtype=np.int64),
            original_ids=np.asarray([[11, 19]], dtype=np.int64),
        )


def test_lazy_semantic_codes_are_used_on_first_access_and_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _table(tmp_path / "table")
    calls: list[tuple[int, list[int]]] = []

    def fake_encode(self, n, keys, original_windows):
        calls.append((n, keys.tolist()))
        codes = np.tile(np.asarray([[7, 1]], dtype=np.int64), (len(keys), 1))
        self._cache.executemany(
            "INSERT OR REPLACE INTO codes(n, key, code) VALUES (?, ?, ?)",
            [(n, int(key), code.astype(np.uint16).tobytes()) for key, code in zip(keys, codes, strict=True)],
        )
        self._cache.commit()
        return codes

    monkeypatch.setattr(RQNgramMapping, "_encode_missing", fake_encode)
    cache = tmp_path / "cache"
    first = RQNgramMapping(str(tmp_path / "table"), pad_id=0, cache_dir=str(cache))
    inputs = np.asarray([[1, 9]], dtype=np.int64)
    originals = np.asarray([[11, 19]], dtype=np.int64)
    actual = first.hash(inputs, original_ids=originals)[0]
    np.testing.assert_array_equal(actual[0, 1], [7, 1])
    assert calls == [(2, [109])]

    # A hot lookup must use the process-local cache, not issue another SQLite query or
    # invoke the semantic encoder.
    statements: list[str] = []
    first._cache.set_trace_callback(statements.append)
    again_hot = first.hash(inputs, original_ids=originals)[0]
    np.testing.assert_array_equal(again_hot, actual)
    assert calls == [(2, [109])]
    assert not any(statement.startswith("SELECT") for statement in statements)

    calls.clear()
    reopened = RQNgramMapping(str(tmp_path / "table"), pad_id=0, cache_dir=str(cache))
    again = reopened.hash(inputs, original_ids=originals)[0]
    np.testing.assert_array_equal(again, actual)
    assert calls == []


def test_row_trace_records_unique_addressed_rows(tmp_path: Path) -> None:
    _table(tmp_path / "table")
    mapping = RQNgramMapping(str(tmp_path / "table"), pad_id=0)
    mapping.start_row_trace()
    mapping.hash(np.asarray([[1, 2]], dtype=np.int64))
    assert mapping.stop_row_trace() == {
        (2, 0, 3),
        (2, 0, 5),
        (2, 1, 4),
        (2, 1, 6),
    }
    assert sum(mapping.traced_row_counts().values()) == 4


def test_runtime_shuffle_applies_identically_to_offline_and_lazy_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = tmp_path / "table"
    _table(table)
    meta = json.loads((table / "meta.json").read_text())
    meta["runtime_shuffle_seed"] = 42
    (table / "meta.json").write_text(json.dumps(meta))
    mapping = RQNgramMapping(str(table), pad_id=0, cache_dir=str(tmp_path / "cache"))
    semantic = np.asarray([[3, 4]], dtype=np.int64)
    expected = mapping._shuffle_codes(2, semantic)[0]
    assert not np.array_equal(expected, semantic[0])
    np.testing.assert_array_equal(mapping.codes[2][0], expected)

    def fake_encode(self, n, keys, original_windows):
        del original_windows
        codes = self._shuffle_codes(n, np.tile(semantic, (len(keys), 1)))
        self._cache.executemany(
            "INSERT OR REPLACE INTO codes(n, key, code) VALUES (?, ?, ?)",
            [(n, int(key), code.astype(np.uint16).tobytes()) for key, code in zip(keys, codes, strict=True)],
        )
        self._cache.commit()
        return codes

    monkeypatch.setattr(RQNgramMapping, "_encode_missing", fake_encode)
    actual = mapping.hash(
        np.asarray([[1, 9]], dtype=np.int64),
        original_ids=np.asarray([[11, 19]], dtype=np.int64),
    )[0]
    np.testing.assert_array_equal(actual[0, 1], expected)


def test_oov_only_shuffle_keeps_offline_frequency_matched_codes(
    tmp_path: Path,
) -> None:
    table = tmp_path / "table"
    _table(table)
    original = np.load(table / "codes_2.npy").astype(np.int64)
    meta = json.loads((table / "meta.json").read_text())
    meta["runtime_oov_shuffle_seed"] = 42
    (table / "meta.json").write_text(json.dumps(meta))

    mapping = RQNgramMapping(str(table), pad_id=0)

    np.testing.assert_array_equal(mapping.codes[2], original)
    oov = np.asarray([[3, 4]], dtype=np.int64)
    shuffled = mapping._shuffle_codes(2, oov, oov=True)
    assert not np.array_equal(shuffled, oov)
