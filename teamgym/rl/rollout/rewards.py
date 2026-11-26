"""Scalar rewards derived only from the ordinary TeamGym evaluation."""

from __future__ import annotations

from typing import Any

from teamgym.scoring.written_answers import holistic_score


REWARD_CONFIGS: dict[str, dict[str, float]] = {
    "task_success": {"outcome_success": 1.0},
    "written_holistic_score": {"holistic_score": 1.0},
    "task_success_plus_informed_success": {
        "outcome_success": 0.5,
        "informed_success": 0.5,
    },
    "task_success_plus_complete_probe_success": {
        "outcome_success": 0.5,
        "complete_probe_success": 0.5,
    },
}


def scalarize_evaluation(
    evaluation: dict[str, Any],
    config_name: str = "task_success",
    *,
    written_judgment: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    if config_name not in REWARD_CONFIGS:
        raise ValueError(f"Unknown TeamGym reward config: {config_name}")
    primary = (
        evaluation.get("primary_eval")
        if isinstance(evaluation.get("primary_eval"), dict)
        else {}
    )
    participation = (
        evaluation.get("participation_eval")
        if isinstance(evaluation.get("participation_eval"), dict)
        else {}
    )
    outcome_success = primary.get("task_success") is True
    components = {
        "outcome_success": 1.0 if outcome_success else 0.0,
        "informed_success": 1.0 if primary.get("informed_success") is True else 0.0,
        "complete_probe_success": (
            1.0
            if outcome_success
            and participation.get("complete_participant_probe") is True
            else 0.0
        ),
        "holistic_score": holistic_score(written_judgment),
    }
    weights = REWARD_CONFIGS[config_name]
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ValueError("Reward weights must have a positive sum")
    reward = sum(weights[key] * components[key] for key in weights) / denominator
    return float(reward), {key: components[key] for key in weights}


def format_scores(
    attempt_records: list[dict[str, Any]],
    *,
    trainable_only: bool,
) -> dict[str, float]:
    """Return exact-schema success rates over logical actor invocations.

    Retries share a prompt id. An invocation succeeds after retry when any
    attempt parses as exactly one JSON object and passes the requested action
    schema. The first-pass score requires retry zero to pass. A successful
    parse/schema check is the action returned to the actor runtime, so this
    matches the prior delivered-format definition without rewarding repairs.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(attempt_records):
        if trainable_only and not bool(record.get("train_mask")):
            continue
        prompt_id = str(record.get("prompt_id") or f"missing_prompt_{index}")
        grouped.setdefault(prompt_id, []).append(record)
    if not grouped:
        return {
            "invocation_count": 0.0,
            "format_score": 0.0,
            "first_pass_format_score": 0.0,
        }
    successful = 0
    first_pass_successful = 0
    for records in grouped.values():
        records.sort(key=lambda row: int(row.get("retry_number") or 0))
        successful += int(
            any(record.get("parse_status") == "ok" for record in records)
        )
        first_pass_successful += int(records[0].get("parse_status") == "ok")
    count = len(grouped)
    return {
        "invocation_count": float(count),
        "format_score": float(successful) / float(count),
        "first_pass_format_score": (
            float(first_pass_successful) / float(count)
        ),
    }
