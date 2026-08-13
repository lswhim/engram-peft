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

    calls.clear()
    reopened = RQNgramMapping(str(tmp_path / "table"), pad_id=0, cache_dir=str(cache))
    again = reopened.hash(inputs, original_ids=originals)[0]
    np.testing.assert_array_equal(again, actual)
    assert calls == []
