"""Utilities for model-assisted participant-role augmentation.

CUAD and MAUD already construct participant roles from model-produced RIU
information features.  This module supports the remaining legal source corpora
without modifying the original problem-instance artifacts.
"""

from __future__ import annotations

import re
from typing import Any

from teamgym.core.participant_roles import (
    Counter as Counter,
    allocation_mode as allocation_mode,
    duplicate_role_names as duplicate_role_names,
    hashlib as hashlib,
    normalize_feature as normalize_feature,
    normalize_information_features as normalize_information_features,
    normalize_leakage_audit as normalize_leakage_audit,
    private_rui_assignment as private_rui_assignment,
    reference_information_units as reference_information_units,
    riu_text_sha256 as riu_text_sha256,
    role_name_from_information_features as role_name_from_information_features,
    role_name_preview as role_name_preview,
    validate_feature as validate_feature,
)


TARGET_SOURCE_CORPORA = ("casebook", "mslr", "multilexsum", "sara")
EXCLUDED_SOURCE_CORPORA = ("cuad", "maud")
PROMPT_VERSION = "legal_riu_information_feature_v4"


class ExcludedSourceCorpusError(ValueError):
    """Raised when augmentation is requested for CUAD or MAUD."""


def source_corpus(row: dict[str, Any]) -> str:
    """Return the normalized legal source corpus for a common datapoint."""

    cross = row.get("_cross") if isinstance(row.get("_cross"), dict) else {}
    specific = (
        row.get("dataset_specific")
        if isinstance(row.get("dataset_specific"), dict)
        else {}
    )
    source = row.get("source") if isinstance(row.get("source"), dict) else {}

    candidates = (
        cross.get("dataset"),
        specific.get("benchmark_dataset"),
        source.get("source_family"),
        specific.get("source_family"),
        row.get("dataset"),
    )
    normalized = " ".join(str(value or "").strip().lower() for value in candidates)

    if re.search(r"\bcuad\b", normalized):
        raise ExcludedSourceCorpusError("CUAD already has participant roles and is excluded.")
    if re.search(r"\bmaud\b", normalized):
        raise ExcludedSourceCorpusError("MAUD already has participant roles and is excluded.")
    if "ecfr_casebook" in normalized or re.search(r"\bcasebook\b", normalized):
        return "casebook"
    if "mslr" in normalized:
        return "mslr"
    if "multilexsum" in normalized or "multi-lexsum" in normalized:
        return "multilexsum"
    if "sara" in normalized:
        return "sara"
    raise ValueError(f"Cannot identify a supported source corpus for {row.get('instance_id')}")


def canonical_problem_instance_id(row: dict[str, Any]) -> str:
    """Return the identifier shared by RIU-distribution variants."""

    cross = row.get("_cross") if isinstance(row.get("_cross"), dict) else {}
    specific = (
        row.get("dataset_specific")
        if isinstance(row.get("dataset_specific"), dict)
        else {}
    )
    value = (
        cross.get("canonical_task_id")
        or specific.get("cross_canonical_task_id")
        or specific.get("canonical_task_id")
        or specific.get("pilot_case_id")
        or specific.get("source_instance_id")
        or row.get("instance_id")
    )
    identifier = str(value or "").strip()
    if not identifier:
        raise ValueError("Problem instance is missing a stable identifier.")
    return identifier


def public_problem_description(row: dict[str, Any]) -> str:
    task = row.get("task_description") if isinstance(row.get("task_description"), dict) else {}
    value = task.get("question") or task.get("objective") or task.get("title")
    return str(value or "").strip()


def reference_decision(row: dict[str, Any]) -> str:
    """Return hidden decision content for leakage auditing only."""

    oracle = row.get("oracle") if isinstance(row.get("oracle"), dict) else {}
    value = (
        oracle.get("canonical_response")
        or oracle.get("short_answer")
        or oracle.get("rationale")
        or oracle.get("answer")
    )
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value or "").strip()


def information_feature_input(unit: dict[str, Any]) -> dict[str, Any]:
    metadata = unit.get("metadata") if isinstance(unit.get("metadata"), dict) else {}
    return {
        "rui_id": str(unit["rui_id"]),
        "riu_text": str(unit.get("text") or unit.get("raw_text") or "").strip(),
        "rui_type": str(unit.get("rui_type") or "").strip(),
        "current_information_feature": str(unit.get("information_feature") or "").strip(),
        "reasoning_role": str(metadata.get("reasoning_role") or "").strip(),
        "supports_goals": metadata.get("supports_goals") or [],
    }
