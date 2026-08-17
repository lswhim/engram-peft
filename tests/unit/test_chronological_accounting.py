from examples.benchmarks.methods import example_presentations_for_run


def test_partial_final_accumulation_reports_exact_dataset_size() -> None:
    assert example_presentations_for_run(
        global_step=1347,
        dataset_size=26922,
        batch_size=4,
        grad_accum=5,
        chronological=True,
    ) == 26922


def test_repeated_or_nonchronological_run_uses_step_budget() -> None:
    assert example_presentations_for_run(
        global_step=1348,
        dataset_size=26922,
        batch_size=4,
        grad_accum=5,
        chronological=True,
    ) == 26960
    assert example_presentations_for_run(
        global_step=1347,
        dataset_size=26922,
        batch_size=4,
        grad_accum=5,
        chronological=False,
    ) == 26940
