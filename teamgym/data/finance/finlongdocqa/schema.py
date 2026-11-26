"""Common datapoint construction used by the FinLongDocQA pipeline."""

from __future__ import annotations

from typing import Any

from teamgym.core.schema import COMMON_DATAPOINT_SCHEMA_VERSION as SCHEMA_VERSION


def build_common_datapoint(**fields: Any) -> dict[str, Any]:
    participants = fields["participants"]
    datapoint = {
        "schema_version": SCHEMA_VERSION,
        "instance_id": fields["instance_id"],
        "dataset": fields["dataset"],
        "scenario_family": fields["scenario_family"],
        "source": fields["source"],
        "task_description": fields["task_description"],
        "decision_set": fields["decision_set"],
        "decision_option_definitions": fields.get("decision_option_definitions") or {},
        "initial_common_information": fields.get("initial_common_information") or {},
        "participant_directory": [
            {
                "participant_id": row.get("participant_id"),
                "role_name": row.get("role_name") or row.get("public_role"),
            }
            for row in participants
        ],
        "participants": participants,
        "private_context_units": fields["private_context_units"],
        "reference_information_units": fields["reference_information_units"],
        "oracle": fields["oracle"],
        "generation_metadata": fields["generation_metadata"],
        "dataset_specific": fields.get("dataset_specific") or {},
        "format_validation": {},
    }
    datapoint["format_validation"] = validate_common_datapoint(datapoint)
    return datapoint


def validate_common_datapoint(datapoint: dict[str, Any]) -> dict[str, Any]:
    participants = [
        row for row in datapoint.get("participants") or [] if isinstance(row, dict)
    ]
    contexts = [
        row
        for row in datapoint.get("private_context_units") or []
        if isinstance(row, dict)
    ]
    rius = [
        row
        for row in datapoint.get("reference_information_units") or []
        if isinstance(row, dict)
    ]
    participant_ids = {
        str(row.get("participant_id")) for row in participants if row.get("participant_id")
    }
    directory_ids = {
        str(row.get("participant_id"))
        for row in datapoint.get("participant_directory") or []
        if isinstance(row, dict) and row.get("participant_id")
    }
    context_ids = {
        str(row.get("context_unit_id")) for row in contexts if row.get("context_unit_id")
    }
    rui_ids = {str(row.get("rui_id")) for row in rius if row.get("rui_id")}
    decision_set = {str(value) for value in datapoint.get("decision_set") or []}
    definitions = {
        str(key): value
        for key, value in (
            datapoint.get("decision_option_definitions") or {}
        ).items()
    }
    oracle = datapoint.get("oracle") or {}
    sufficient_sets = [
        {str(value) for value in row.get("rui_ids") or []}
        for row in oracle.get("smallest_sufficient_rui_sets") or []
        if isinstance(row, dict)
    ]
    common_rius = {
        str(row.get("rui_id"))
        for row in (
            datapoint.get("initial_common_information") or {}
        ).get("common_reference_units", [])
        if isinstance(row, dict) and row.get("rui_id")
    }
    checks = [
        {"name": "schema_version", "passed": datapoint.get("schema_version") == SCHEMA_VERSION},
        {"name": "instance_id_present", "passed": bool(datapoint.get("instance_id"))},
        {"name": "dataset_present", "passed": bool(datapoint.get("dataset"))},
        {"name": "finite_decision_set", "passed": bool(decision_set)},
        {
            "name": "decision_definitions_match",
            "passed": set(definitions) == decision_set
            and all(isinstance(value, str) and value.strip() for value in definitions.values()),
        },
        {"name": "participants_present", "passed": len(participant_ids) >= 2},
        {
            "name": "participant_directory_matches",
            "passed": directory_ids == participant_ids,
        },
        {"name": "contexts_present", "passed": bool(contexts)},
        {
            "name": "contexts_assigned",
            "passed": all(
                row.get("assigned_to")
                and {
                    str(value) for value in row.get("assigned_to") or []
                } <= participant_ids
                for row in contexts
            ),
        },
        {
            "name": "participant_context_refs_exist",
            "passed": all(
                {
                    str(value)
                    for value in participant.get("private_context_unit_ids") or []
                } <= context_ids
                for participant in participants
            ),
        },
        {"name": "rius_present", "passed": bool(rui_ids)},
        {
            "name": "oracle_answer_in_decision_set",
            "passed": str(oracle.get("answer") or "") in decision_set,
        },
        {"name": "sufficient_sets_present", "passed": bool(sufficient_sets)},
        {
            "name": "sufficient_set_refs_exist",
            "passed": all(values <= rui_ids for values in sufficient_sets),
        },
        {
            "name": "no_single_participant_sufficient",
            "passed": not any(
                sufficient
                and sufficient
                <= (
                    {
                        str(value)
                        for value in participant.get("owned_rui_ids") or []
                    }
                    | common_rius
                )
                for sufficient in sufficient_sets
                for participant in participants
            ),
        },
    ]
    return {
        "valid": all(row["passed"] for row in checks),
        "checks": checks,
        "metrics": {
            "participant_count": len(participant_ids),
            "private_context_unit_count": len(contexts),
            "reference_information_unit_count": len(rius),
        },
    }
