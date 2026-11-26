"""Schema helpers for long-context distributed reasoning datapoints."""

from __future__ import annotations

from typing import Any

from teamgym.core.schema import build_common_datapoint

LONG_CONTEXT_REASONING_SCHEMA_VERSION = "collabgym_long_context_reasoning_seed_v1"

DEFAULT_DECISION_SET = [
    "yes",
    "no",
    "conditional",
    "not_enough_information",
]

DEFAULT_DECISION_OPTION_DEFINITIONS = {
    "yes": "The answer to the task question is yes.",
    "no": "The answer to the task question is no.",
    "conditional": "The answer depends on a condition or exception that must be stated.",
    "not_enough_information": "The supplied information is insufficient to answer.",
}

ANSWER_ALIASES = {
    "y": "yes",
    "true": "yes",
    "allowed": "yes",
    "permitted": "yes",
    "n": "no",
    "false": "no",
    "not_allowed": "no",
    "prohibited": "no",
    "depends": "conditional",
    "it_depends": "conditional",
    "insufficient": "not_enough_information",
    "unknown": "not_enough_information",
    "cannot_determine": "not_enough_information",
}


def decision_set_for_seed(seed: dict[str, Any]) -> list[str]:
    """Return the finite answer set for one seed."""

    custom = seed.get("decision_set")
    if isinstance(custom, list) and custom:
        decision_set = [str(item) for item in custom if str(item)]
        if len(decision_set) != len(set(decision_set)):
            raise ValueError(f"Duplicate decision ids in decision_set: {decision_set}")
        return decision_set
    options = seed.get("decision_options")
    if isinstance(options, list) and options:
        decision_set = [str(row.get("decision_id") or row.get("id") or "") for row in options if isinstance(row, dict)]
        decision_set = [item for item in decision_set if item]
        if decision_set:
            if len(decision_set) != len(set(decision_set)):
                raise ValueError(f"Duplicate decision ids in decision_options: {decision_set}")
            return decision_set
    return list(DEFAULT_DECISION_SET)


def decision_option_definitions_for_seed(seed: dict[str, Any], decision_set: list[str]) -> dict[str, str]:
    """Return text definitions for all options in a seed decision set."""

    custom = seed.get("decision_option_definitions")
    if isinstance(custom, dict) and custom:
        definitions = {str(key): str(value) for key, value in custom.items()}
    else:
        definitions = {}
        options = seed.get("decision_options")
        if isinstance(options, list):
            for row in options:
                if not isinstance(row, dict):
                    continue
                decision_id = str(row.get("decision_id") or row.get("id") or "")
                if not decision_id:
                    continue
                label = str(row.get("label") or "").strip()
                description = str(row.get("description") or row.get("text") or "").strip()
                definitions[decision_id] = f"{label}: {description}" if label and description else label or description
    if not definitions and decision_set == DEFAULT_DECISION_SET:
        definitions = dict(DEFAULT_DECISION_OPTION_DEFINITIONS)
    missing = [item for item in decision_set if not str(definitions.get(item) or "").strip()]
    extra = [item for item in definitions if item not in set(decision_set)]
    if missing or extra:
        raise ValueError(f"Decision option definitions must match decision_set; missing={missing}, extra={extra}")
    return definitions


def normalize_answer(answer: str, *, decision_set: list[str] | None = None) -> str:
    """Normalize a seed answer into its finite decision set."""

    normalized = str(answer).strip().lower().replace(" ", "_").replace("-", "_")
    allowed_set = decision_set or DEFAULT_DECISION_SET
    if allowed_set == DEFAULT_DECISION_SET:
        normalized = ANSWER_ALIASES.get(normalized, normalized)
    else:
        raw = str(answer).strip()
        if raw in allowed_set:
            normalized = raw
    if normalized not in allowed_set:
        allowed = ", ".join(allowed_set)
        raise ValueError(f"Unsupported answer {answer!r}; expected one of: {allowed}")
    return normalized


def build_reasoning_oracle(
    *,
    seed: dict[str, Any],
    answer: str,
    required_rui_ids: list[str],
) -> dict[str, Any]:
    """Build the oracle payload expected by the common datapoint schema."""

    canonical_response = seed.get("canonical_response") or seed.get("short_answer") or seed.get("short_reason") or ""
    required_decision_steps = seed.get("required_decision_steps") or seed.get("reasoning_steps", [])
    answer_type = seed.get("answer_type", "finite_multiple_choice")
    return {
        "answer": answer,
        "short_answer": seed.get("short_answer") or seed.get("short_reason") or canonical_response,
        "canonical_response": canonical_response,
        "required_rui_ids": required_rui_ids,
        "reasoning_steps": seed.get("reasoning_steps", []),
        "required_decision_steps": required_decision_steps,
        "judge_rubric": seed.get("judge_rubric", {}),
        "disqualifying_errors": seed.get("disqualifying_errors", []),
        "semantic_judge": {
            "enabled": answer_type == "short_answer_judged",
            "primary_answer_field": seed.get("primary_answer_field", "written_answer"),
            "fallback_answer_fields": seed.get("fallback_answer_fields", ["rationale", "answer"]),
        },
        "smallest_sufficient_rui_sets": [
            {
                "rui_ids": required_rui_ids,
                "answer": answer,
                "why_sufficient": seed.get(
                    "why_sufficient",
                    "These RUIs jointly establish the answer and no single participant should have all of them.",
                ),
            }
        ],
        "metadata": {
            "source_schema": LONG_CONTEXT_REASONING_SCHEMA_VERSION,
            "source_family": seed.get("source_family", "unknown"),
            "answer_type": answer_type,
        },
    }


def build_long_context_common_datapoint(
    *,
    seed: dict[str, Any],
    participants: list[dict[str, Any]],
    private_context_units: list[dict[str, Any]],
    reference_information_units: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Wrap a long-context seed task in the shared common datapoint schema."""

    required_rui_ids = [str(rui_id) for rui_id in seed["required_rui_ids"]]
    decision_set = decision_set_for_seed(seed)
    decision_option_definitions = decision_option_definitions_for_seed(seed, decision_set)
    answer = normalize_answer(seed["answer"], decision_set=decision_set)
    source_family = seed.get("source_family", "long_context")
    instance_id = seed.get("instance_id") or metadata.get("instance_id")
    if not instance_id:
        raise ValueError("Seed is missing instance_id")

    task_description = {
        "question": seed["question"],
        "answer_type": seed.get("answer_type", "finite_multiple_choice"),
        "response_format_note": seed.get(
            "response_format_note",
            "Select exactly one decision id from the decision set and provide the reasoning that supports it.",
        ),
    }

    common_rui_ids = set(metadata.get("common_rui_ids", []))
    common_reference_units = [
        {
            "rui_id": unit.get("rui_id"),
            "summary": unit.get("summary") or unit.get("information_feature") or unit.get("text"),
            "text": unit.get("text", ""),
            "source_ref": unit.get("source_ref", seed.get("source_id", instance_id)),
        }
        for unit in reference_information_units
        if unit.get("rui_id") in common_rui_ids
    ]

    initial_common_information = {
        "task": seed["question"],
        "decision_options": decision_option_definitions,
        "public_case_facts": seed.get("public_case_facts", []),
        "common_reference_units": common_reference_units,
        "collaboration_note": (
            "Participants have different private document excerpts. The correct "
            "answer may require combining multiple excerpts."
        ),
    }

    oracle = build_reasoning_oracle(
        seed=seed,
        answer=answer,
        required_rui_ids=required_rui_ids,
    )

    return build_common_datapoint(
        instance_id=instance_id,
        dataset="long_context_reasoning",
        source={
            "source_id": seed.get("source_id", instance_id),
            "source_family": source_family,
            "source_title": seed.get("source_title", ""),
            "source_path": seed.get("source_path", ""),
        },
        scenario_family=f"{source_family}_distributed_reasoning",
        task_description=task_description,
        decision_set=decision_set,
        decision_option_definitions=decision_option_definitions,
        initial_common_information=initial_common_information,
        participants=participants,
        private_context_units=private_context_units,
        reference_information_units=reference_information_units,
        oracle=oracle,
        generation_metadata={
            **metadata,
            "long_context_reasoning_schema_version": LONG_CONTEXT_REASONING_SCHEMA_VERSION,
            "source_family": source_family,
            "answer": answer,
            "required_rui_count": len(required_rui_ids),
        },
        dataset_specific={
            "source_family": source_family,
            "answer_type": seed.get("answer_type", "finite_multiple_choice"),
            "reasoning_steps": seed.get("reasoning_steps", []),
            "required_decision_steps": seed.get("required_decision_steps", seed.get("reasoning_steps", [])),
            "canonical_response": seed.get("canonical_response") or seed.get("short_answer") or "",
            "judge_rubric": seed.get("judge_rubric", {}),
            "disqualifying_errors": seed.get("disqualifying_errors", []),
        },
    )
