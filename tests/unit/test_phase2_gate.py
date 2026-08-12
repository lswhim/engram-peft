from __future__ import annotations

from copy import deepcopy

from examples.run_semantic_hash_phase2 import gate_decision


CONTROLS = ("arithmetic_matched", "rq_shuffled")
PRIMARY = "3gram_semantic_neighbor_shared_code"


def passing_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "aggregate": {
            control: {
                PRIMARY: {"ci95": [-0.20, -0.05]},
                "overall": {"ci95": [-0.02, 0.01]},
                "2gram_covered_no_neighbor_high_lexical": {
                    "tokens": 4,
                    "delta_nll": 0.0,
                    "ci95": [-0.2, 0.2],
                },
                "3gram_covered_no_neighbor_high_lexical": {
                    "tokens": 8,
                    "delta_nll": 0.0,
                    "ci95": [-0.2, 0.2],
                },
            }
            for control in CONTROLS
        },
        "interactions": {
            control: {
                "3gram_shared_minus_no_shared": {"ci95": [-0.18, -0.01]}
            }
            for control in CONTROLS
        },
        "per_seed": {
            str(seed): {
                control: {
                    PRIMARY: {"delta_nll": -0.1 if seed in (42, 43) else 0.01}
                }
                for control in CONTROLS
            }
            for seed in (42, 43, 44)
        },
    }


def test_gate_passes_only_when_all_evaluable_conditions_hold() -> None:
    decision = gate_decision(passing_payload())
    assert decision["status"] == "pass"
    assert decision["primary_pass"] is True
    assert decision["interaction_pass"] is True
    assert decision["overall_safe"] is True
    assert decision["seed_consistent"] is True
    assert decision["false_sharing_status"].startswith("underpowered")


def test_gate_rejects_missing_shared_code_interaction() -> None:
    payload = deepcopy(passing_payload())
    payload["interactions"]["rq_shuffled"][
        "3gram_shared_minus_no_shared"
    ]["ci95"] = [-0.1, 0.03]
    decision = gate_decision(payload)
    assert decision["status"] == "no_go"
    assert decision["interaction_pass"] is False


def test_gate_rejects_single_seed_primary_effect() -> None:
    payload = deepcopy(passing_payload())
    for seed in (43, 44):
        payload["per_seed"][str(seed)]["arithmetic_matched"][PRIMARY][
            "delta_nll"
        ] = 0.01
    decision = gate_decision(payload)
    assert decision["status"] == "no_go"
    assert decision["seed_consistent"] is False
