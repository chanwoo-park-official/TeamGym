"""Map contract participants and private context to TeamGym records."""

from __future__ import annotations

from typing import Any


def common_participants(participants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep each generated role name alongside its common participant ID."""
    participant_map = participant_id_map(participants)
    return [
        {
            "participant_id": participant_map[str(participant.get("participant_id"))],
            "role_name": participant.get("public_role"),
            "private_context_unit_ids": list(participant.get("private_doc_unit_ids") or []),
            "private_word_count": int(participant.get("private_word_count") or 0),
            "owned_rui_ids": list(participant.get("owned_rui_ids") or []),
            "generation_features": {
                "role_terms": participant.get("role_terms") or [],
                "owned_rui_feature_terms": participant.get("owned_rui_feature_terms") or [],
                "owned_rui_section_keys": participant.get("owned_rui_section_keys") or [],
            },
        }
        for participant in participants
    ]


def common_context_units(
    document_units: list[dict[str, Any]],
    participant_map: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "context_unit_id": unit.get("doc_unit_id"),
            "text": unit.get("text"),
            "assigned_to": [participant_map.get(str(item), str(item)) for item in unit.get("assigned_to") or []],
            "word_count": unit.get("word_count"),
            "source_ref": {
                "start": unit.get("start"),
                "end": unit.get("end"),
                "section_key": unit.get("section_key"),
            },
            "hidden_contains_rui_ids": unit.get("hidden_contains_rui_ids") or [],
            "assignment": {
                "reason": unit.get("assignment_reason"),
                "scores": unit.get("assignment_scores") or {},
            },
        }
        for unit in document_units
    ]


def participant_id_map(participants: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(participant.get("participant_id")): letter_id("participant", idx)
        for idx, participant in enumerate(participants, start=1)
    }


def letter_id(prefix: str, idx: int) -> str:
    """Format a one-based record ID with suffixes a through z, then aa and up."""
    if idx < 1:
        raise ValueError("idx must be >= 1")
    letters = ""
    current = idx
    while current:
        current, remainder = divmod(current - 1, 26)
        letters = chr(ord("a") + remainder) + letters
    return f"{prefix}_{letters}"
