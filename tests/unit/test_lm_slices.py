import importlib.util
from pathlib import Path

import numpy as np

from examples.evaluate_lm_slices import metric


_SPEC = importlib.util.spec_from_file_location(
    "build_lm_slice_manifest",
    Path(__file__).resolve().parents[2] / "scripts" / "build_lm_slice_manifest.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
poly_keys = _MODULE.poly_keys
table_positions = _MODULE.table_positions
valid_context_mask = _MODULE.valid_context_mask
initial_categories = _MODULE.initial_categories

_ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_lm_slice_results",
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_lm_slice_results.py",
)
assert _ANALYSIS_SPEC is not None and _ANALYSIS_SPEC.loader is not None
_ANALYSIS = importlib.util.module_from_spec(_ANALYSIS_SPEC)
_ANALYSIS_SPEC.loader.exec_module(_ANALYSIS)


def test_poly_keys_are_aligned_to_context_end() -> None:
    compressed = np.asarray([[1, 2, 3, 4]], dtype=np.int64)
    assert poly_keys(compressed, order=2, base=10).tolist() == [[0, 12, 23, 34]]
    assert poly_keys(compressed, order=3, base=10).tolist() == [[0, 0, 123, 234]]


def test_valid_context_excludes_prefix_and_last_prediction() -> None:
    attention = np.asarray([[1, 1, 1, 1, 0]], dtype=np.uint8)
    assert valid_context_mask(attention, order=2).tolist() == [
        [False, True, True, False, False]
    ]
    assert valid_context_mask(attention, order=3).tolist() == [
        [False, False, True, False, False]
    ]


def test_table_positions_marks_only_exact_hits() -> None:
    positions, hit = table_positions(
        np.asarray([1, 2, 4, 8]), np.asarray([2, 4, 6])
    )
    assert positions.tolist() == [0, 0, 1, 2]
    assert hit.tolist() == [False, True, True, False]


def test_exact_seen_has_precedence_over_address_oov() -> None:
    categories = initial_categories(
        valid=np.asarray([True, True, True, False]),
        covered=np.asarray([False, True, True, False]),
        exact=np.asarray([True, False, True, False]),
    )
    assert categories.tolist() == [1, 3, 1, 0]


def test_metric_uses_only_masked_finite_losses() -> None:
    losses = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = metric(losses, np.asarray([[True, False], [True, False]]))
    assert result["tokens"] == 2
    assert result["nll"] == 2.0


def test_paired_document_statistics_preserve_sign() -> None:
    semantic = np.asarray([[1.0, 2.0], [2.0, 4.0]])
    control = np.asarray([[2.0, 2.0], [3.0, 3.0]])
    values, counts = _ANALYSIS.document_statistics(
        semantic, control, np.asarray([[True, False], [True, True]])
    )
    assert values.tolist() == [-1.0, 0.0]
    assert counts.tolist() == [1, 2]
