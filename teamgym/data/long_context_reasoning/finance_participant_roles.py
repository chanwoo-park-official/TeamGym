"""Model-assisted public participant roles for finance datapoints.

The finance datasets store evidence-bearing ``information_feature`` values that
often contain the exact private numbers needed to answer the task.  This module
creates a separate, public-safe feature for role naming and deliberately leaves
the evidence-bearing feature unchanged.
"""

from __future__ import annotations

from copy import deepcopy
import json
import re
from pathlib import Path
from typing import Any

from teamgym.core.io import render_template
from teamgym.core.participant_roles import (
    duplicate_role_names,
    normalize_information_features,
    normalize_leakage_audit,
    reference_information_units,
    riu_text_sha256,
    role_name_preview,
    validate_feature,
)
from teamgym.core.schema import validate_common_datapoint


FINANCE_DATASETS = ("findver", "finlongdocqa")
PROMPT_VERSION = "finance_riu_information_feature_v1"

_FINANCE_CONCLUSION_PATTERN = re.compile(
    r"\b(?:"
    r"declin(?:e|ed|ing)|decreas(?:e|ed|ing)|effective|exceed(?:s|ed|ing)?|"
    r"grew|higher|improv(?:e|ed|ing)|increas(?:e|ed|ing)|ineffective|lower|"
    r"outperform(?:s|ed|ing)?|reduc(?:e|ed|ing)|total(?:ed|ing)|"
    r"underperform(?:s|ed|ing)?|was|were"
    r")\b",
    flags=re.IGNORECASE,
)


def finance_dataset(row: dict[str, Any]) -> str:
    """Return the supported finance dataset name for a common datapoint."""

    dataset = str(row.get("dataset") or "").strip().casefold()
    if dataset not in FINANCE_DATASETS:
        raise ValueError(
            f"Unsupported finance dataset {row.get('dataset')!r} for "
            f"{row.get('instance_id')!r}"
        )
    return dataset


def problem_instance_id(row: dict[str, Any]) -> str:
    identifier = str(row.get("instance_id") or "").strip()
    if not identifier:
        raise ValueError("Finance datapoint is missing instance_id.")
    return identifier


def public_problem_description(row: dict[str, Any]) -> str:
    task = (
        row.get("task_description")
        if isinstance(row.get("task_description"), dict)
        else {}
    )
    value = task.get("question") or task.get("objective") or task.get("title")
    return str(value or "").strip()


def hidden_reference_decision(row: dict[str, Any]) -> str:
    """Return hidden answer material used only by the leakage audit."""

    oracle = row.get("oracle") if isinstance(row.get("oracle"), dict) else {}
    for key in (
        "canonical_response",
        "short_answer",
        "answer_text",
        "rationale",
    ):
        value = oracle.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    sufficient_sets = oracle.get("smallest_sufficient_rui_sets")
    if isinstance(sufficient_sets, list):
        answers = [
            item.get("answer_text")
            for item in sufficient_sets
            if isinstance(item, dict)
            and isinstance(item.get("answer_text"), str)
            and item.get("answer_text").strip()
        ]
        if answers:
            return "\n".join(str(answer).strip() for answer in answers)
    value = oracle.get("answer")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "").strip()


def finance_information_feature_input(unit: dict[str, Any]) -> dict[str, Any]:
    """Build the private RIU payload shown to the feature generator/auditor."""

    return {
        "rui_id": str(unit.get("rui_id") or "").strip(),
        "riu_text": str(unit.get("text") or unit.get("raw_text") or "").strip(),
        "riu_summary": str(unit.get("summary") or "").strip(),
        "current_private_information_feature": str(
            unit.get("information_feature") or ""
        ).strip(),
    }


def validate_finance_feature(feature: str, *, rui_id: str) -> None:
    """Apply deterministic public-role leakage checks."""

    validate_feature(feature, rui_id=rui_id)
    if _FINANCE_CONCLUSION_PATTERN.search(feature):
        raise ValueError(
            f"{rui_id}: information feature discloses a finance result or conclusion."
        )


def normalize_finance_information_features(
    response: dict[str, Any],
    *,
    expected_rui_ids: list[str],
) -> dict[str, str]:
    features = normalize_information_features(
        response,
        expected_rui_ids=expected_rui_ids,
    )
    for rui_id, feature in features.items():
        validate_finance_feature(feature, rui_id=rui_id)
    return features


def generate_information_features(
    *,
    client: Any,
    dataset: str,
    instance_id: str,
    system_prompt: str,
    user_prompt: str,
    expected_rui_ids: list[str],
    trace_root: Path,
    attempts: int,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Generate one validated public feature per RIU, with repair attempts."""

    previous_response: dict[str, Any] | None = None
    validation_error: str | None = None
    request_metas: list[dict[str, Any]] = []
    for attempt in range(1, max(1, attempts) + 1):
        prompt = user_prompt
        if previous_response is not None:
            prompt += (
                "\n\nThe previous response failed validation. Return a complete "
                "corrected response for every RIU."
            )
            prompt += "\nDo not preserve a feature that caused the validation error."
            prompt += "\nPrevious response:\n" + json.dumps(
                previous_response,
                indent=2,
                ensure_ascii=False,
            )
            prompt += f"\nValidation error:\n{validation_error}"
        response = client.generate_json(
            prompt_id=(
                f"{dataset}.{safe_id(instance_id)}."
                f"finance_information_features.attempt_{attempt}"
            ),
            system_prompt=system_prompt,
            user_prompt=prompt,
            trace_dir=trace_root / f"attempt_{attempt}",
        )
        request_metas.append(dict(getattr(client, "last_request_meta", {})))
        try:
            return (
                normalize_finance_information_features(
                    response,
                    expected_rui_ids=expected_rui_ids,
                ),
                request_metas,
            )
        except ValueError as exc:
            previous_response = response
            validation_error = str(exc)
            if attempt >= max(1, attempts):
                raise
    raise RuntimeError("Finance information-feature generation produced no result.")


def audit_information_features(
    *,
    client: Any,
    dataset: str,
    instance_id: str,
    system_prompt: str,
    user_template: str,
    datapoint: dict[str, Any],
    units: list[dict[str, Any]],
    features: dict[str, str],
    trace_root: Path,
    attempts: int,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Audit proposed features and re-audit model-suggested repairs."""

    expected_rui_ids = [str(unit["rui_id"]) for unit in units]
    units_by_id = {str(unit["rui_id"]): unit for unit in units}
    current_features = dict(features)
    request_metas: list[dict[str, Any]] = []
    accepted_audits: dict[str, dict[str, Any]] = {}
    pending_rui_ids = list(expected_rui_ids)
    final_unsafe_audits: dict[str, dict[str, Any]] = {}

    for attempt in range(1, max(1, attempts) + 1):
        payload = {
            "public_problem_description": public_problem_description(datapoint),
            "reference_decision": hidden_reference_decision(datapoint),
            "rius_and_features": [
                {
                    **finance_information_feature_input(units_by_id[rui_id]),
                    "proposed_information_feature": current_features[rui_id],
                }
                for rui_id in pending_rui_ids
            ],
        }
        response = client.generate_json(
            prompt_id=(
                f"{dataset}.{safe_id(instance_id)}."
                f"finance_information_feature_audit.attempt_{attempt}"
            ),
            system_prompt=system_prompt,
            user_prompt=render_template(user_template, payload),
            trace_dir=trace_root / f"attempt_{attempt}",
        )
        request_metas.append(dict(getattr(client, "last_request_meta", {})))
        attempt_audits = normalize_leakage_audit(
            response,
            expected_rui_ids=pending_rui_ids,
        )
        unsafe = [
            rui_id
            for rui_id, audit in attempt_audits.items()
            if not audit["safe"]
        ]
        accepted_audits.update(
            {
                rui_id: audit
                for rui_id, audit in attempt_audits.items()
                if audit["safe"]
            }
        )
        final_unsafe_audits = {
            rui_id: attempt_audits[rui_id] for rui_id in unsafe
        }
        if not unsafe:
            return current_features, accepted_audits, request_metas
        if attempt >= max(1, attempts):
            return (
                current_features,
                {**accepted_audits, **final_unsafe_audits},
                request_metas,
            )
        for rui_id in unsafe:
            suggestion = attempt_audits[rui_id]["suggested_information_feature"]
            validate_finance_feature(suggestion, rui_id=rui_id)
            current_features[rui_id] = suggestion
        pending_rui_ids = unsafe

    return current_features, {**accepted_audits, **final_unsafe_audits}, request_metas


def apply_public_roles(
    row: dict[str, Any],
    *,
    features_by_rui_id: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a role-augmented copy while preserving private RIU features."""

    augmented = deepcopy(row)
    dataset = finance_dataset(augmented)
    instance_id = problem_instance_id(augmented)
    units = reference_information_units(augmented)
    expected_ids = {str(unit["rui_id"]) for unit in units}
    if set(features_by_rui_id) != expected_ids:
        raise ValueError(
            f"{instance_id}: public feature coverage mismatch: "
            f"missing={sorted(expected_ids - set(features_by_rui_id))}, "
            f"extra={sorted(set(features_by_rui_id) - expected_ids)}"
        )

    for unit in units:
        rui_id = str(unit["rui_id"])
        feature = str(features_by_rui_id[rui_id])
        validate_finance_feature(feature, rui_id=rui_id)
        unit["public_information_feature"] = feature

    previews = role_name_preview(
        augmented,
        features_by_rui_id=features_by_rui_id,
    )
    collisions = duplicate_role_names(previews)
    if collisions:
        collision_ordinals: dict[str, int] = {}
        for preview in previews:
            role_name = str(preview.get("proposed_role_name") or "")
            if role_name not in collisions:
                continue
            collision_ordinals[role_name] = collision_ordinals.get(role_name, 0) + 1
            preview["proposed_role_name"] = (
                f"{role_name} {collision_ordinals[role_name]}"
            )
    remaining_collisions = duplicate_role_names(previews)
    if remaining_collisions:
        raise ValueError(
            f"{instance_id}: duplicate public role names: {remaining_collisions}"
        )
    preview_by_id = {
        str(preview["participant_id"]): preview for preview in previews
    }
    for participant in augmented.get("participants", []):
        participant_id = str(participant.get("participant_id") or "")
        participant["role_name"] = preview_by_id[participant_id][
            "proposed_role_name"
        ]
    augmented["participant_directory"] = [
        {
            "participant_id": participant.get("participant_id"),
            "role_name": participant.get("role_name"),
        }
        for participant in augmented.get("participants", [])
    ]
    generation = augmented.setdefault("generation_metadata", {})
    generation["participant_role_augmentation"] = {
        "schema": "finance_participant_role_augmentation_v1",
        "prompt_version": PROMPT_VERSION,
        "dataset": dataset,
        "problem_instance_id": instance_id,
        "role_source": "privately_assigned_reference_information_units",
        "private_information_feature_preserved": True,
    }

    validation = validate_common_datapoint(augmented)
    if not validation["valid"]:
        failed = [
            check["name"]
            for check in validation["checks"]
            if not check["passed"]
        ]
        raise ValueError(
            f"{instance_id}: common datapoint validation failed: {failed}"
        )
    return augmented, previews


def valid_cached_feature_rows(
    rows: list[dict[str, Any]],
    *,
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a complete safe cache group, or an empty list if stale."""

    if not rows or any(row.get("prompt_version") != PROMPT_VERSION for row in rows):
        return []
    by_id = {str(row.get("rui_id") or ""): row for row in rows}
    expected_ids = {str(unit["rui_id"]) for unit in units}
    if set(by_id) != expected_ids:
        return []
    for unit in units:
        rui_id = str(unit["rui_id"])
        cached = by_id[rui_id]
        if cached.get("riu_text_sha256") != riu_text_sha256(unit):
            return []
        audit = cached.get("leakage_audit")
        if not isinstance(audit, dict) or audit.get("safe") is not True:
            return []
        try:
            validate_finance_feature(
                str(cached.get("information_feature") or ""),
                rui_id=rui_id,
            )
        except ValueError:
            return []
    return [by_id[rui_id] for rui_id in sorted(expected_ids)]


def safe_id(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    return cleaned[:160] or "problem_instance"
