from __future__ import annotations

from teamgym.data.legal.decision_permutation import (
    apply_permutation_to_public_common,
    permute_common_datapoint_decisions,
)
from teamgym.data.legal.permute_generated_decisions import balanced_target_answers
from teamgym.core.schema import build_common_datapoint


def _datapoint() -> dict:
    return build_common_datapoint(
        instance_id="case_001",
        dataset="cuad",
        source={"source_dataset": "unit"},
        scenario_family="unit",
        task_description={"objective": "Pick one."},
        decision_set=["decision_a", "decision_b", "decision_c"],
        decision_option_definitions={
            "decision_a": "True option.",
            "decision_b": "Distractor one.",
            "decision_c": "Distractor two.",
        },
        initial_common_information={
            "decision_option_definitions": {
                "decision_a": "True option.",
                "decision_b": "Distractor one.",
                "decision_c": "Distractor two.",
            },
            "decision_option_metadata": {
                "decision_a": {"feature": "true"},
                "decision_b": {"feature": "b"},
                "decision_c": {"feature": "c"},
            },
        },
        participants=[
            {"participant_id": "participant_a", "private_context_unit_ids": ["u1"], "owned_rui_ids": ["r1"]},
            {"participant_id": "participant_b", "private_context_unit_ids": ["u2"], "owned_rui_ids": ["r2"]},
        ],
        private_context_units=[
            {"context_unit_id": "u1", "text": "A", "assigned_to": ["participant_a"]},
            {"context_unit_id": "u2", "text": "B", "assigned_to": ["participant_b"]},
        ],
        reference_information_units=[
            {"rui_id": "r1", "text": "A"},
            {"rui_id": "r2", "text": "B"},
        ],
        oracle={
            "answer": "decision_a",
            "smallest_sufficient_rui_sets": [{"rui_ids": ["r1", "r2"], "answer": "decision_a"}],
        },
        generation_metadata={"method": "unit"},
        dataset_specific={
            "decision_option_metadata": {
                "decision_a": {"feature": "true"},
                "decision_b": {"feature": "b"},
                "decision_c": {"feature": "c"},
            }
        },
    )


def test_permute_common_datapoint_decisions_remaps_answer_and_option_payloads() -> None:
    datapoint = _datapoint()

    mixed, metadata = permute_common_datapoint_decisions(
        datapoint,
        seed_material="unit-seed",
        target_answer="decision_c",
    )

    assert mixed["format_validation"]["valid"] is True
    assert mixed["oracle"]["answer"] == "decision_c"
    assert mixed["oracle"]["smallest_sufficient_rui_sets"][0]["answer"] == "decision_c"
    assert mixed["decision_option_definitions"]["decision_c"] == "True option."
    assert mixed["dataset_specific"]["decision_option_metadata"]["decision_c"] == {"feature": "true"}
    assert mixed["initial_common_information"]["decision_option_definitions"]["decision_c"] == "True option."
    assert mixed["initial_common_information"]["decision_option_metadata"]["decision_c"] == {"feature": "true"}
    assert metadata["old_to_new"]["decision_a"] == "decision_c"
    assert metadata["new_to_old"]["decision_c"] == "decision_a"
    assert mixed["generation_metadata"]["decision_option_permutation"]["new_answer"] == "decision_c"


def test_apply_permutation_to_public_common_remaps_public_options() -> None:
    mixed, metadata = permute_common_datapoint_decisions(
        _datapoint(),
        seed_material="unit-seed",
        target_answer="decision_b",
    )

    public = apply_permutation_to_public_common(
        {
            "decision_set": ["decision_a", "decision_b", "decision_c"],
            "decision_option_definitions": {
                "decision_a": "True option.",
                "decision_b": "Distractor one.",
                "decision_c": "Distractor two.",
            },
            "decision_option_metadata": {
                "decision_a": {"feature": "true"},
                "decision_b": {"feature": "b"},
                "decision_c": {"feature": "c"},
            },
        },
        metadata,
        source_common_datapoint=mixed,
    )

    assert public["decision_option_definitions"]["decision_b"] == "True option."
    assert public["decision_option_metadata"]["decision_b"] == {"feature": "true"}
    assert public["decision_option_permutation"]["old_to_new"]["decision_a"] == "decision_b"


def test_balanced_target_answers_spreads_available_labels() -> None:
    rows = [{"common_datapoint": _datapoint()} for _ in range(9)]

    targets = balanced_target_answers(rows, seed="unit")

    counts = {label: list(targets.values()).count(label) for label in ["decision_a", "decision_b", "decision_c"]}
    assert counts == {"decision_a": 3, "decision_b": 3, "decision_c": 3}
