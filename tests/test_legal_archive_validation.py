from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from teamgym.data.legal.archive_validation import validate_task, verify_checksums
from teamgym.core.dataset import load_common_datapoints
from teamgym.core.schema import (
    RELEASE_DATAPOINT_SCHEMA_VERSION,
    build_common_datapoint,
    to_release_datapoint,
    validate_common_datapoint,
)
from teamgym.core.release import FULL_CONTEXT, TEAM_PIU_ONLY, write_release_datapoints
from teamgym.interaction.rui_context import rui_only_datapoint


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
    assert validate_task(to_release_datapoint(task())) == []


@pytest.mark.parametrize("condition", [FULL_CONTEXT, TEAM_PIU_ONLY])
def test_accepted_task_writer_round_trips_hugging_face_schema(tmp_path, condition):
    original = task()
    original["generation_metadata"]["allocation_mode"] = "one_each"
    if condition == TEAM_PIU_ONLY:
        original = rui_only_datapoint(original)
        original["private_context_units"][0]["text"] = "The separately qualified evidence packet."
    original["_input_path"] = "/source"
    before = copy.deepcopy(original)

    path = tmp_path / "common_datapoints.jsonl"
    write_release_datapoints(path, [{"common_datapoint": original}])
    assert original == before
    public = json.loads(path.read_text(encoding="utf-8"))
    assert public["schema_version"] == RELEASE_DATAPOINT_SCHEMA_VERSION
    assert public["evidence_condition"] == condition
    assert public["distribution_key"] == "one_each"
    assert public["pivotal_information_units"][0]["piu_id"] == "r1"
    assert public["participants"][0]["owned_piu_ids"] == ["r1"]
    assert "_input_path" not in public
    assert validate_common_datapoint(public)["valid"]
    normalized = load_common_datapoints(path)[0]
    assert normalized["oracle"] == original["oracle"]
    assert normalized["generation_metadata"] == original["generation_metadata"]
    assert normalized["private_context_units"] == original["private_context_units"]
    if condition == TEAM_PIU_ONLY:
        assert rui_only_datapoint(normalized) == normalized


def test_accepting_a_hosted_row_preserves_its_null_distribution_name(tmp_path):
    original = task()
    original["distribution_key"] = "one_each"
    original["distribution_name"] = None
    path = tmp_path / "common_datapoints.jsonl"
    write_release_datapoints(path, [original])
    public = json.loads(path.read_text())
    assert public["distribution_key"] == "one_each"
    assert public["distribution_name"] is None


def test_interrupted_task_writer_preserves_previous_output(tmp_path):
    path = tmp_path / "common_datapoints.jsonl"
    write_release_datapoints(path, [task()])
    saved = path.read_bytes()
    with pytest.raises(ValueError, match="expected a TeamGym task"):
        write_release_datapoints(path, [task(), {"instance_id": "invalid"}])
    assert path.read_bytes() == saved


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
