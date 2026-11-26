from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from teamgym.data.legal.participant_role_augmentation import (
    riu_text_sha256,
)
from teamgym.data.long_context_reasoning.finance_participant_roles import (
    PROMPT_VERSION,
    apply_public_roles,
    audit_information_features,
    finance_dataset,
    generate_information_features,
    hidden_reference_decision,
    normalize_finance_information_features,
    valid_cached_feature_rows,
)


def _datapoint() -> dict:
    return {
        "schema_version": "teamgym_common_datapoint_v2",
        "instance_id": "finance_example",
        "dataset": "finlongdocqa",
        "scenario_family": "finance_test",
        "source": {"source_id": "report"},
        "task_description": {"objective": "Calculate the requested ratio."},
        "decision_set": ["free_response"],
        "decision_option_definitions": {"free_response": "Write the answer."},
        "initial_common_information": {"common_reference_units": []},
        "participant_directory": [
            {"participant_id": "participant_a", "role_name": "reviewer 1"},
            {"participant_id": "participant_b", "role_name": "reviewer 2"},
        ],
        "participants": [
            {
                "participant_id": "participant_a",
                "role_name": "reviewer 1",
                "private_context_unit_ids": ["ctx_a"],
                "owned_rui_ids": ["rui_a"],
            },
            {
                "participant_id": "participant_b",
                "role_name": "reviewer 2",
                "private_context_unit_ids": ["ctx_b"],
                "owned_rui_ids": ["rui_b"],
            },
        ],
        "private_context_units": [
            {
                "context_unit_id": "ctx_a",
                "text": "Cash was $100.",
                "assigned_to": ["participant_a"],
                "hidden_contains_rui_ids": ["rui_a"],
            },
            {
                "context_unit_id": "ctx_b",
                "text": "Debt was $25.",
                "assigned_to": ["participant_b"],
                "hidden_contains_rui_ids": ["rui_b"],
            },
        ],
        "reference_information_units": [
            {
                "rui_id": "rui_a",
                "text": "Cash was $100.",
                "summary": "Cash amount.",
                "information_feature": "Cash was $100.",
            },
            {
                "rui_id": "rui_b",
                "text": "Debt was $25.",
                "summary": "Debt amount.",
                "information_feature": "Debt was $25.",
            },
        ],
        "oracle": {
            "answer": "free_response",
            "answer_text": "Debt was 25% of cash.",
            "smallest_sufficient_rui_sets": [
                {
                    "rui_ids": ["rui_a", "rui_b"],
                    "answer": "free_response",
                }
            ],
        },
        "generation_metadata": {},
        "dataset_specific": {},
        "format_validation": {"valid": True},
    }


class _FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.last_request_meta = {"model": "fake"}

    def generate_json(self, **_: object) -> dict:
        return self.responses.pop(0)


def test_dataset_and_reference_answer_support_both_finance_sources() -> None:
    row = _datapoint()
    assert finance_dataset(row) == "finlongdocqa"
    assert hidden_reference_decision(row) == "Debt was 25% of cash."
    row["dataset"] = "findver"
    row["oracle"] = {"canonical_response": "The filing answer."}
    assert finance_dataset(row) == "findver"
    assert hidden_reference_decision(row) == "The filing answer."


@pytest.mark.parametrize(
    "feature",
    ["cash increased", "controls were ineffective", "revenue $100"],
)
def test_finance_feature_validation_rejects_results(feature: str) -> None:
    with pytest.raises(ValueError):
        normalize_finance_information_features(
            {
                "information_features": [
                    {"rui_id": "rui_a", "information_feature": feature}
                ]
            },
            expected_rui_ids=["rui_a"],
        )


def test_generation_repairs_invalid_feature(tmp_path: Path) -> None:
    client = _FakeClient(
        [
            {
                "information_features": [
                    {"rui_id": "rui_a", "information_feature": "cash was $100"}
                ]
            },
            {
                "information_features": [
                    {"rui_id": "rui_a", "information_feature": "cash resources"}
                ]
            },
        ]
    )
    features, requests = generate_information_features(
        client=client,
        dataset="finlongdocqa",
        instance_id="example",
        system_prompt="system",
        user_prompt="user",
        expected_rui_ids=["rui_a"],
        trace_root=tmp_path,
        attempts=2,
    )
    assert list(features) == ["rui_a"]
    assert features["rui_a"] == "cash resources"
    assert len(requests) == 2


def test_audit_reaudits_safe_repair(tmp_path: Path) -> None:
    row = _datapoint()
    client = _FakeClient(
        [
            {
                "audits": [
                    {
                        "rui_id": "rui_a",
                        "safe": False,
                        "reason": "Too specific.",
                        "suggested_information_feature": "liquidity resources",
                    },
                    {
                        "rui_id": "rui_b",
                        "safe": True,
                        "reason": "Broad subject only.",
                        "suggested_information_feature": "",
                    },
                ]
            },
            {
                "audits": [
                    {
                        "rui_id": "rui_a",
                        "safe": True,
                        "reason": "Broad subject only.",
                        "suggested_information_feature": "",
                    }
                ]
            },
        ]
    )
    features, audits, requests = audit_information_features(
        client=client,
        dataset="finlongdocqa",
        instance_id="example",
        system_prompt="system",
        user_template=(
            "{{public_problem_description}} {{reference_decision}} "
            "{{rius_and_features}}"
        ),
        datapoint=row,
        units=row["reference_information_units"],
        features={"rui_a": "cash balances", "rui_b": "debt balances"},
        trace_root=tmp_path,
        attempts=2,
    )
    assert features["rui_a"] == "liquidity resources"
    assert features["rui_b"] == "debt balances"
    assert all(audit["safe"] for audit in audits.values())
    assert len(requests) == 2


def test_apply_roles_preserves_private_features() -> None:
    original = _datapoint()
    snapshot = deepcopy(original)
    augmented, preview = apply_public_roles(
        original,
        features_by_rui_id={
            "rui_a": "liquidity resources",
            "rui_b": "debt obligations",
        },
    )
    assert original == snapshot
    assert [
        participant["role_name"] for participant in augmented["participants"]
    ] == ["liquidity resources reviewer", "debt obligations reviewer"]
    assert augmented["participant_directory"] == [
        {
            "participant_id": "participant_a",
            "role_name": "liquidity resources reviewer",
        },
        {
            "participant_id": "participant_b",
            "role_name": "debt obligations reviewer",
        },
    ]
    assert [
        unit["information_feature"]
        for unit in augmented["reference_information_units"]
    ] == ["Cash was $100.", "Debt was $25."]
    assert [
        unit["public_information_feature"]
        for unit in augmented["reference_information_units"]
    ] == ["liquidity resources", "debt obligations"]
    assert len(preview) == 2


def test_apply_roles_numbers_duplicate_gpt_names() -> None:
    augmented, preview = apply_public_roles(
        _datapoint(),
        features_by_rui_id={
            "rui_a": "joint venture ownership",
            "rui_b": "joint venture ownership",
        },
    )

    assert [
        participant["role_name"] for participant in augmented["participants"]
    ] == [
        "joint venture ownership reviewer 1",
        "joint venture ownership reviewer 2",
    ]
    assert len({item["proposed_role_name"] for item in preview}) == 2


def test_valid_cache_uses_existing_rui_id_schema() -> None:
    units = _datapoint()["reference_information_units"]
    rows = [
        {
            "prompt_version": PROMPT_VERSION,
            "rui_id": unit["rui_id"],
            "riu_text_sha256": riu_text_sha256(unit),
            "information_feature": feature,
            "leakage_audit": {"safe": True},
        }
        for unit, feature in zip(
            units,
            ["liquidity resources", "debt obligations"],
            strict=True,
        )
    ]

    assert valid_cached_feature_rows(rows, units=units) == rows
    malformed_rows = [
        {
            **{key: value for key, value in row.items() if key != "rui_id"},
            "riu_id": row["rui_id"],
        }
        for row in rows
    ]
    assert valid_cached_feature_rows(malformed_rows, units=units) == []
