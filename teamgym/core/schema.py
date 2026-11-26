"""Shared common datapoint schema for legal fully collaborative instances."""

from __future__ import annotations

from typing import Any

COMMON_DATAPOINT_SCHEMA_VERSION = "teamgym_common_datapoint_v2"
LEGACY_COMMON_DATAPOINT_SCHEMA_VERSION = "collabgym_fully_collaborative_common_datapoint_v2"
SUPPORTED_COMMON_DATAPOINT_SCHEMA_VERSIONS = (
    COMMON_DATAPOINT_SCHEMA_VERSION,
    LEGACY_COMMON_DATAPOINT_SCHEMA_VERSION,
)


def build_common_datapoint(
    *,
    instance_id: str,
    dataset: str,
    source: dict[str, Any],
    scenario_family: str,
    task_description: dict[str, Any],
    decision_set: list[str],
    decision_option_definitions: dict[str, str] | None = None,
    initial_common_information: dict[str, Any] | None = None,
    participants: list[dict[str, Any]],
    private_context_units: list[dict[str, Any]],
    reference_information_units: list[dict[str, Any]],
    oracle: dict[str, Any],
    generation_metadata: dict[str, Any],
    dataset_specific: dict[str, Any] | None = None,
) -> dict[str, Any]:
    datapoint = {
        "schema_version": COMMON_DATAPOINT_SCHEMA_VERSION,
        "instance_id": instance_id,
        "dataset": dataset,
        "scenario_family": scenario_family,
        "source": source,
        "task_description": task_description,
        "decision_set": decision_set,
        "decision_option_definitions": decision_option_definitions or {},
        "initial_common_information": initial_common_information or {},
        "participant_directory": _participant_directory(participants),
        "participants": participants,
        "private_context_units": private_context_units,
        "reference_information_units": reference_information_units,
        "oracle": oracle,
        "generation_metadata": generation_metadata,
        "dataset_specific": dataset_specific or {},
        "format_validation": {},
    }
    datapoint["format_validation"] = validate_common_datapoint(datapoint)
    return datapoint


def validate_common_datapoint(datapoint: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    participants = datapoint.get("participants") or []
    participant_directory = datapoint.get("participant_directory") or []
    private_context_units = datapoint.get("private_context_units") or []
    reference_information_units = datapoint.get("reference_information_units") or []
    oracle = datapoint.get("oracle") or {}
    initial_common_information = datapoint.get("initial_common_information") or {}
    decision_set = {str(item) for item in datapoint.get("decision_set", [])}
    decision_option_definitions = datapoint.get("decision_option_definitions") or {}

    participant_ids = {
        str(participant.get("participant_id"))
        for participant in participants
        if participant.get("participant_id")
    }
    directory_ids = {
        str(participant.get("participant_id"))
        for participant in participant_directory
        if participant.get("participant_id")
    }
    context_unit_ids = {
        str(unit.get("context_unit_id"))
        for unit in private_context_units
        if unit.get("context_unit_id")
    }
    rui_ids = {
        str(unit.get("rui_id"))
        for unit in reference_information_units
        if unit.get("rui_id")
    }
    participant_context_refs = [
        {str(item) for item in participant.get("private_context_unit_ids", [])}
        for participant in participants
    ]
    context_assignees = [
        {str(item) for item in unit.get("assigned_to", [])}
        for unit in private_context_units
    ]
    smallest_sets = [
        {str(item) for item in row.get("rui_ids", [])}
        for row in oracle.get("smallest_sufficient_rui_sets", [])
        if isinstance(row, dict)
    ]
    common_rui_ids = {
        str(row.get("rui_id"))
        for row in initial_common_information.get("common_reference_units", [])
        if isinstance(row, dict) and row.get("rui_id")
    }

    _check(checks, "schema_version", datapoint.get("schema_version") in SUPPORTED_COMMON_DATAPOINT_SCHEMA_VERSIONS)
    _check(checks, "dataset_present", bool(datapoint.get("dataset")))
    _check(checks, "instance_id_present", bool(datapoint.get("instance_id")))
    _check(checks, "task_description_present", bool(datapoint.get("task_description")))
    _check(checks, "finite_decision_set", bool(decision_set))
    _check(
        checks,
        "decision_option_definitions_match_decision_set",
        set(str(item) for item in decision_option_definitions.keys()) == decision_set,
    )
    _check(
        checks,
        "decision_option_definitions_are_text",
        all(isinstance(value, str) and bool(value.strip()) for value in decision_option_definitions.values()),
    )
    _check(checks, "participants_present", len(participants) >= 2)
    for field, rows, id_field in (
        ("participant", participants, "participant_id"),
        ("participant_directory", participant_directory, "participant_id"),
        ("private_context_unit", private_context_units, "context_unit_id"),
        ("reference_information_unit", reference_information_units, "rui_id"),
    ):
        ids = [row.get(id_field) for row in rows]
        # Valid rows retain the stored validation report used by frozen artifacts.
        if not all(isinstance(item, str) and item.strip() for item in ids) or len(set(ids)) != len(ids):
            _check(checks, f"{field}_ids_nonempty_and_unique", False)
    _check(checks, "participant_directory_matches_participants", directory_ids == participant_ids)
    _check(checks, "private_context_units_present", bool(private_context_units))
    _check(checks, "private_context_units_assigned", all(unit.get("assigned_to") for unit in private_context_units))
    _check(checks, "reference_information_units_present", bool(reference_information_units))
    _check(checks, "oracle_answer_present", "answer" in oracle)
    _check(checks, "oracle_answer_in_decision_set", str(oracle.get("answer")) in decision_set)
    _check(checks, "participant_context_refs_exist", all(refs <= context_unit_ids for refs in participant_context_refs))
    _check(checks, "context_assignees_exist", all(assignees <= participant_ids for assignees in context_assignees))
    declared_assignments = {
        (str(participant.get("participant_id")), context_id)
        for participant, refs in zip(participants, participant_context_refs)
        for context_id in refs
    }
    context_assignments = {
        (participant_id, str(unit.get("context_unit_id")))
        for unit, assignees in zip(private_context_units, context_assignees)
        for participant_id in assignees
    }
    if declared_assignments != context_assignments:
        _check(checks, "private_context_assignments_match", False)
    _check(checks, "smallest_sufficient_sets_exist", bool(smallest_sets))
    _check(checks, "smallest_sufficient_refs_exist", all(rui_set <= rui_ids for rui_set in smallest_sets))
    _check(
        checks,
        "no_single_participant_has_smallest_sufficient_set",
        not any(
            rui_set and rui_set <= (set(participant.get("owned_rui_ids", [])) | common_rui_ids)
            for participant in participants
            for rui_set in smallest_sets
        ),
    )
    return {
        "valid": all(row["passed"] for row in checks),
        "checks": checks,
        "metrics": {
            "participant_count": len(participants),
            "private_context_unit_count": len(private_context_units),
            "reference_information_unit_count": len(reference_information_units),
            "common_reference_unit_count": len(common_rui_ids),
            "smallest_sufficient_set_count": len(smallest_sets),
        },
    }


def _participant_directory(participants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "participant_id": participant.get("participant_id"),
            "role_name": participant.get("role_name") or participant.get("public_role"),
        }
        for participant in participants
    ]


def _check(checks: list[dict[str, Any]], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})
