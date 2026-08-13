from examples.wikibigedit_stream_protocol import build_protocol, cohort_indices


def test_cohorts_are_deterministic_bounded_and_spread():
    first=cohort_indices(1000,100,42)
    assert first==cohort_indices(1000,100,42)
    assert len(first)==100
    assert min(first)>=0 and max(first)<1000
    assert max(first)-min(first)>900


def test_retention_matrix_only_uses_past_cohorts():
    protocol=build_protocol([{}]*10_000,(1000,5000,10000),50,42)
    assert set(protocol["retention"]["1000"])=={"1000"}
    assert set(protocol["retention"]["10000"])=={"1000","5000","10000"}
