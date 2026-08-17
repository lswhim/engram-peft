from scripts.analyze_wikibigedit_oov_slices import summarize


def test_summarize_separates_offline_and_dynamic_queries() -> None:
    rows = [
        {"eligible": True, "axis": "efficacy", "address_slice": "offline_all", "accuracy": 1.0},
        {"eligible": True, "axis": "efficacy", "address_slice": "dynamic_any", "accuracy": 0.25},
        {"eligible": False, "axis": "efficacy", "address_slice": "dynamic_any", "accuracy": 0.0},
    ]

    metrics = summarize(rows)

    assert metrics["slice/offline_all"] == {"mean": 1.0, "n": 1}
    assert metrics["slice/dynamic_any"] == {"mean": 0.25, "n": 1}
    assert metrics["axis/efficacy/slice/dynamic_any"]["n"] == 1
