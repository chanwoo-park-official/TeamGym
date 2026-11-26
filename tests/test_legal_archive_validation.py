from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from teamgym.data.legal.archive_validation import validate_task, verify_checksums
from teamgym.core.schema import build_common_datapoint


def task() -> dict:
    return build_common_datapoint(
        instance_id="archive_fixture", dataset="fixture", source={"source_id": "source_1"}, scenario_family="fixture",
        task_description={"question": "Combine the two conditions."}, decision_set=["yes", "no"],
        decision_option_definitions={"yes": "Both conditions hold.", "no": "At least one condition fails."},
        participants=[
            {"participant_id": "a", "role_name": "first", "owned_rui_ids": ["r1"], "private_context_unit_ids": ["c1"]},
            {"participant_id": "b", "role_name": "second", "owned_rui_ids": ["r2"], "private_context_unit_ids": ["c2"]},
        ],
        private_context_units=[
            {"context_unit_id": "c1", "text": "First condition.", "assigned_to": ["a"], "hidden_contains_rui_ids": ["r1"]},
            {"context_unit_id": "c2", "text": "Second condition.", "assigned_to": ["b"], "hidden_contains_rui_ids": ["r2"]},
        ],
        reference_information_units=[
            {"rui_id": "r1", "text": "First condition.", "source_ref": {"source_id": "source_1"}},
            {"rui_id": "r2", "text": "Second condition.", "source_ref": {"source_id": "source_1"}},
        ],
        oracle={"answer": "yes", "required_rui_ids": ["r1", "r2"], "smallest_sufficient_rui_sets": [{"rui_ids": ["r1", "r2"]}]},
        generation_metadata={},
    )


def test_ownership_can_be_encoded_on_participants_and_source_contexts() -> None:
    assert validate_task(task()) == []


@pytest.mark.parametrize("corruption, message", [
    ("duplicate_participant", "participant_ids_nonempty_and_unique"),
    ("missing_source", "missing source reference"),
    ("assignment_inverse", "private_context_assignments_match"),
    ("missing_evidence", "missing from its owner's source contexts"),
])
def test_integrity_validation_rejects_broken_references(corruption: str, message: str) -> None:
    row = task()
    if corruption == "duplicate_participant":
        row["participants"].append(copy.deepcopy(row["participants"][0]))
    elif corruption == "missing_source":
        row["reference_information_units"][0]["source_ref"] = {}
    elif corruption == "assignment_inverse":
        row["participants"][0]["private_context_unit_ids"] = []
    else:
        row["private_context_units"][0]["hidden_contains_rui_ids"] = []
    with pytest.raises(ValueError, match=message):
        validate_task(row)


def test_shared_chunk_visibility_is_reported_without_rewriting_the_frozen_task() -> None:
    row = task()
    row["private_context_units"][0]["assigned_to"].append("b")
    row["participants"][1]["private_context_unit_ids"].append("c1")
    before = copy.deepcopy(row)
    findings = validate_task(row)
    assert findings == [{"rui_id": "r1", "declared_owners": ["a"], "visible_to": ["a", "b"], "extra_viewers": ["b"]}]
    assert row == before


def test_checksums_detect_changed_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "instances.jsonl"
    source.write_text('original source bytes\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "checksums.sha256").write_text(f"{digest}  instances.jsonl\n")
    assert verify_checksums(tmp_path, {}) == 1
    source.write_text('changed source bytes\n')
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksums(tmp_path, {})
