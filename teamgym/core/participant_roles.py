"""Shared evidence-feature validation and participant-role rendering.

These helpers preserve the existing augmentation and naming rules. Domain-specific
corpus selection, prompts, and additional checks remain with the domain code.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Any


def reference_information_units(row: dict[str, Any]) -> list[dict[str, Any]]:
    units = [
        unit
        for unit in row.get("reference_information_units", [])
        if isinstance(unit, dict) and str(unit.get("rui_id") or "").strip()
    ]
    if not units:
        raise ValueError(f"No RIUs found for {row.get('instance_id')}")
    identifiers = [str(unit["rui_id"]) for unit in units]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate RIU identifiers in {row.get('instance_id')}")
    return units


def riu_text_sha256(unit: dict[str, Any]) -> str:
    text = str(unit.get("text") or unit.get("raw_text") or "").strip()
    if not text:
        raise ValueError(f"RIU {unit.get('rui_id')} has no text.")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_information_features(
    response: dict[str, Any],
    *,
    expected_rui_ids: list[str],
) -> dict[str, str]:
    """Validate and normalize one model-produced feature per expected RIU."""

    rows = response.get("information_features")
    if not isinstance(rows, list):
        raise ValueError("Model response must contain an information_features list.")

    features: dict[str, str] = {}
    for entry in rows:
        if not isinstance(entry, dict):
            raise ValueError("Every information_features entry must be an object.")
        rui_id = str(entry.get("rui_id") or "").strip()
        feature = normalize_feature(entry.get("information_feature"))
        if not rui_id:
            raise ValueError("An information_features entry is missing rui_id.")
        if rui_id in features:
            raise ValueError(f"Duplicate information feature for {rui_id}.")
        validate_feature(feature, rui_id=rui_id)
        features[rui_id] = feature

    expected = set(expected_rui_ids)
    actual = set(features)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"RIU feature coverage mismatch: missing={missing}, extra={extra}")
    return features


def normalize_feature(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").split()).strip()


def validate_feature(feature: str, *, rui_id: str) -> None:
    if not feature:
        raise ValueError(f"{rui_id}: information feature is empty.")
    if len(feature) > 64:
        raise ValueError(f"{rui_id}: information feature exceeds 64 characters.")
    words = feature.split()
    if not 2 <= len(words) <= 6:
        raise ValueError(f"{rui_id}: information feature must contain 2 to 6 words.")
    if re.search(r"\d|[$%]", feature):
        raise ValueError(f"{rui_id}: information feature contains a number or value.")
    if re.search(r"\breviewer\b", feature, flags=re.IGNORECASE):
        raise ValueError(f"{rui_id}: information feature must not contain reviewer.")
    if re.search(r"\bdecision[_ -]?[a-z]\b", feature, flags=re.IGNORECASE):
        raise ValueError(f"{rui_id}: information feature contains an answer label.")
    directional = re.compile(
        r"\b(?:allowed|approved|correct|denied|exceeds?|failed|failing|"
        r"ineligible|incorrect|not|prohibited|rejected|satisfied|successful|"
        r"unsatisfied|unsuccessful|violated|violating|within)\b",
        flags=re.IGNORECASE,
    )
    if directional.search(feature):
        raise ValueError(f"{rui_id}: information feature discloses a direction or conclusion.")


def normalize_leakage_audit(
    response: dict[str, Any],
    *,
    expected_rui_ids: list[str],
) -> dict[str, dict[str, Any]]:
    rows = response.get("audits")
    if not isinstance(rows, list):
        raise ValueError("Audit response must contain an audits list.")
    audits: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every audit entry must be an object.")
        rui_id = str(row.get("rui_id") or "").strip()
        if not rui_id or rui_id in audits:
            raise ValueError(f"Invalid or duplicate audit RIU identifier: {rui_id!r}")
        safe = row.get("safe")
        if not isinstance(safe, bool):
            raise ValueError(f"{rui_id}: audit safe field must be boolean.")
        audits[rui_id] = {
            "safe": safe,
            "reason": " ".join(str(row.get("reason") or "").split()),
            "suggested_information_feature": normalize_feature(
                row.get("suggested_information_feature")
            ),
        }
    expected = set(expected_rui_ids)
    actual = set(audits)
    if actual != expected:
        raise ValueError(
            f"RIU audit coverage mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return audits


def allocation_mode(row: dict[str, Any]) -> str:
    cross = row.get("_cross") if isinstance(row.get("_cross"), dict) else {}
    specific = (
        row.get("dataset_specific")
        if isinstance(row.get("dataset_specific"), dict)
        else {}
    )
    generation = (
        row.get("generation_metadata")
        if isinstance(row.get("generation_metadata"), dict)
        else {}
    )
    return str(
        cross.get("partition_mode")
        or specific.get("partition_mode")
        or generation.get("allocation_mode")
        or ""
    ).strip()


def private_rui_assignment(row: dict[str, Any]) -> dict[str, list[str]]:
    """Return privately assigned RIUs, excluding common RIUs."""

    participants = [
        participant
        for participant in row.get("participants", [])
        if isinstance(participant, dict) and participant.get("participant_id")
    ]
    if not participants:
        raise ValueError(f"No participants found for {row.get('instance_id')}")

    generation = (
        row.get("generation_metadata")
        if isinstance(row.get("generation_metadata"), dict)
        else {}
    )
    allocation = (
        generation.get("private_information_allocation")
        if isinstance(generation.get("private_information_allocation"), dict)
        else {}
    )
    mapped = allocation.get("mapped_private_assignment")
    if isinstance(mapped, dict):
        result = {
            str(participant["participant_id"]): [str(rui_id) for rui_id in mapped.get(participant["participant_id"], [])]
            for participant in participants
        }
        if any(result.values()) or allocation_mode(row) == "common_anchor":
            return result

    common_ids = {
        str(rui_id)
        for rui_id in generation.get("common_rui_ids", [])
        if str(rui_id)
    }
    if allocation_mode(row) == "common_anchor" and not common_ids:
        owned_sets = [
            {str(rui_id) for rui_id in participant.get("owned_rui_ids", [])}
            for participant in participants
        ]
        if owned_sets:
            common_ids = set.intersection(*owned_sets)

    return {
        str(participant["participant_id"]): [
            str(rui_id)
            for rui_id in participant.get("owned_rui_ids", [])
            if str(rui_id) not in common_ids
        ]
        for participant in participants
    }


def role_name_from_information_features(features: list[str], participant_index: int) -> str:
    """Mirror the deterministic CUAD/MAUD feature-to-role transformation."""

    cleaned = [
        feature.replace("_", " ").replace("-", " ").strip().lower()
        for feature in features
        if feature.strip()
    ]
    if not cleaned:
        return f"context reviewer {participant_index}"
    if len(cleaned) == 1:
        return f"{cleaned[0]} reviewer"

    stopwords = {
        "about",
        "above",
        "after",
        "again",
        "against",
        "and",
        "agreement",
        "between",
        "clause",
        "contract",
        "could",
        "each",
        "for",
        "from",
        "have",
        "into",
        "must",
        "only",
        "or",
        "other",
        "party",
        "profile",
        "review",
        "reviewer",
        "shall",
        "should",
        "span",
        "such",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "to",
        "with",
        "within",
        "would",
    }
    first_terms: list[str] = []
    for feature in cleaned[:2]:
        tokens = [
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]*", feature.lower())
            if token not in stopwords and token != "reviewer" and len(token) > 2
        ]
        if tokens:
            first_terms.append(" ".join(tokens[:2]))
    return (
        f"{' and '.join(first_terms)} reviewer"
        if first_terms
        else f"reviewer {participant_index}"
    )


def role_name_preview(
    row: dict[str, Any],
    *,
    features_by_rui_id: dict[str, str],
) -> list[dict[str, Any]]:
    assignment = private_rui_assignment(row)
    preview: list[dict[str, Any]] = []
    for index, participant in enumerate(row.get("participants", []), start=1):
        if not isinstance(participant, dict):
            continue
        participant_id = str(participant.get("participant_id") or "")
        private_ids = assignment.get(participant_id, [])
        missing = [rui_id for rui_id in private_ids if rui_id not in features_by_rui_id]
        if missing:
            raise ValueError(f"Missing information features for assigned RIUs: {missing}")
        features = [features_by_rui_id[rui_id] for rui_id in private_ids]
        preview.append(
            {
                "participant_id": participant_id,
                "private_rui_ids": private_ids,
                "information_features": features,
                "original_role_name": participant.get("role_name"),
                "proposed_role_name": role_name_from_information_features(features, index),
            }
        )
    return preview


def duplicate_role_names(preview: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("proposed_role_name") or "") for row in preview)
    return {name: count for name, count in counts.items() if name and count > 1}
