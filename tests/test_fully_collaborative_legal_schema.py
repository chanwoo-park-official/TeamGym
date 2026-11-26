from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from teamgym.core import schema as shared_schema
from teamgym.core.schema import (
    COMMON_DATAPOINT_SCHEMA_VERSION,
    build_common_datapoint,
)


def test_shared_schema_imports_without_construction_or_training_dependencies() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import teamgym.core.schema; "
            "blocked = ('teamgym.data', "
            "'teamgym.interaction', 'rllm', 'verl', 'torch'); "
            "assert not [name for name in sys.modules "
            "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def test_build_common_datapoint_has_shared_context_unit_format() -> None:
    datapoint = build_common_datapoint(
        instance_id="common_001",
        dataset="cuad",
        source={"source_dataset": "CUAD", "source_id": "source_1"},
        scenario_family="insurance_review",
        task_description={"objective": "Choose the matching decision."},
        decision_set=["decision_a", "decision_b"],
        decision_option_definitions={"decision_a": "First option.", "decision_b": "Second option."},
        initial_common_information={"source_context": {"dataset": "CUAD"}},
        participants=[
            {
                "participant_id": "participant_a",
                "role_name": "limit reviewer",
                "private_context_unit_ids": ["du_0001"],
                "owned_rui_ids": ["span_01"],
            },
            {
                "participant_id": "participant_b",
                "role_name": "notice reviewer",
                "private_context_unit_ids": ["du_0002"],
                "owned_rui_ids": ["span_02"],
            },
        ],
        private_context_units=[
            {"context_unit_id": "du_0001", "text": "Limit text.", "assigned_to": ["participant_a"]},
            {"context_unit_id": "du_0002", "text": "Notice text.", "assigned_to": ["participant_b"]},
        ],
        reference_information_units=[
            {"rui_id": "span_01", "text": "Limit text."},
            {"rui_id": "span_02", "text": "Notice text."},
        ],
        oracle={
            "answer": "decision_b",
            "smallest_sufficient_rui_sets": [{"rui_ids": ["span_01", "span_02"]}],
        },
        generation_metadata={"method": "unit_test", "no_synthetic_fallback": True},
    )

    assert datapoint["schema_version"] == COMMON_DATAPOINT_SCHEMA_VERSION
    assert datapoint["format_validation"]["valid"] is True
    assert list(datapoint.keys()) == [
        "schema_version",
        "instance_id",
        "dataset",
        "scenario_family",
        "source",
        "task_description",
        "decision_set",
        "decision_option_definitions",
        "initial_common_information",
        "participant_directory",
        "participants",
        "private_context_units",
        "reference_information_units",
        "oracle",
        "generation_metadata",
        "dataset_specific",
        "format_validation",
    ]
    assert datapoint["participant_directory"] == [
        {"participant_id": "participant_a", "role_name": "limit reviewer"},
        {"participant_id": "participant_b", "role_name": "notice reviewer"},
    ]
    assert shared_schema.validate_common_datapoint(datapoint)["valid"]
