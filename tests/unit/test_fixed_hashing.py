import numpy as np

from engram_peft.hashing import FixedNgramHashMapping


def test_fixed_hashing_is_capacity_matched_and_head_independent() -> None:
    mapping = FixedNgramHashMapping(
        compressed_vocab_size=100_000,
        engram_vocab_size_per_ngram=[8192, 8192],
        ngram_sizes=[2, 3],
        n_head_per_ngram=8,
        layer_ids=[11, 21],
        pad_id=0,
        seed=42,
    )
    assert mapping.prime_tables[11] == [[1024] * 8, [1024] * 8]
    inputs = np.asarray([[11, 29, 47, 83, 101]], dtype=np.int64)
    outputs = mapping.hash(inputs)
    assert outputs[11].shape == (1, 5, 16)
    assert outputs[21].shape == (1, 5, 16)
    assert int(outputs[11].min()) >= 0
    assert int(outputs[11].max()) < 1024
    # Independent multiplier vectors should not collapse every head to one code.
    assert np.unique(outputs[11][0, -1, :8]).size > 1
    assert not np.array_equal(outputs[11], outputs[21])
