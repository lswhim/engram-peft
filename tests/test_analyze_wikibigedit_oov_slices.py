from scripts.analyze_wikibigedit_oov_slices import oov_bin, summarize


def test_oov_bin_has_fixed_boundaries() -> None:
    assert oov_bin(0.0) == "00_exact"
    assert oov_bin(0.1) == "01_0-10pct"
    assert oov_bin(0.10001) == "02_10-25pct"
    assert oov_bin(0.5) == "03_25-50pct"
    assert oov_bin(0.9) == "04_50-100pct"


def test_summarize_separates_offline_and_dynamic_queries() -> None:
    rows = [
        {"eligible": True, "axis": "efficacy", "address_slice": "offline_all", "dynamic_oov_bin": "00_exact", "accuracy": 1.0},
        {"eligible": True, "axis": "efficacy", "address_slice": "dynamic_any", "dynamic_oov_bin": "02_10-25pct", "accuracy": 0.25},
        {"eligible": False, "axis": "efficacy", "address_slice": "dynamic_any", "dynamic_oov_bin": "04_50-100pct", "accuracy": 0.0},
    ]

    metrics = summarize(rows)

    assert metrics["slice/offline_all"] == {"mean": 1.0, "n": 1}
    assert metrics["slice/dynamic_any"] == {"mean": 0.25, "n": 1}
    assert metrics["axis/efficacy/slice/dynamic_any"]["n"] == 1
    assert metrics["axis/efficacy/oov_bin/02_10-25pct"]["mean"] == 0.25
