"""Full-document partition validation shared by CUAD and MAUD."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .assignment import participant_doc_unit_counts as _participant_doc_unit_counts
from .chunks import EvidenceSpan


def validate_full_document_partition(
    *,
    context: str,
    rui_ranges: Sequence[EvidenceSpan],
    smallest_sets: list[list[str]],
    participants: list[dict[str, Any]],
    document_units: list[dict[str, Any]],
    common_rui_ids: set[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    sorted_units = sorted(document_units, key=lambda unit: int(unit["start"]))
    _check(checks, "document_units_present", bool(sorted_units))
    _check(checks, "full_document_coverage", _covers_full_document(context, sorted_units))
    _check(checks, "non_overlapping_ordered_units", _non_overlapping(sorted_units))
    _check(checks, "no_rui_split_across_chunks", _no_rui_split(rui_ranges, sorted_units))
    assigned_rui_by_participant = _participant_rui_sets(participants, document_units)
    participant_doc_unit_counts = _participant_doc_unit_counts(participants)
    all_assigned_rui = set().union(*assigned_rui_by_participant.values()) if assigned_rui_by_participant else set()
    required = {rui.unit_id for rui in rui_ranges}
    _check(checks, "all_rui_covered_by_private_or_common_chunks", required <= (all_assigned_rui | common_rui_ids))
    _check(
        checks,
        "no_single_participant_has_smallest_sufficient_set",
        not any(
            set(smallest) <= (participant_set | common_rui_ids)
            for participant_set in assigned_rui_by_participant.values()
            for smallest in smallest_sets
        ),
    )
    _check(checks, "all_doc_units_assigned", all(unit.get("assigned_to") for unit in document_units))
    _check(
        checks,
        "every_participant_has_private_context",
        bool(participant_doc_unit_counts) and all(count > 0 for count in participant_doc_unit_counts.values()),
    )
    return {
        "valid": all(row["passed"] for row in checks),
        "checks": checks,
        "metrics": {
            "document_unit_count": len(document_units),
            "participant_count": len(participants),
            "rui_count": len(rui_ranges),
            "common_rui_ids": sorted(common_rui_ids),
            "participant_rui_sets": {pid: sorted(values) for pid, values in assigned_rui_by_participant.items()},
            "participant_doc_unit_counts": participant_doc_unit_counts,
            "participant_doc_unit_count_summary": number_summary(list(participant_doc_unit_counts.values())),
            "participant_doc_unit_count_gap": number_gap(list(participant_doc_unit_counts.values())),
            "participant_doc_word_counts": {
                participant["participant_id"]: participant.get("private_word_count", 0)
                for participant in participants
            },
            "chunk_word_count_summary": number_summary([int(unit["word_count"]) for unit in document_units]),
        },
    }


def _participant_rui_sets(
    participants: list[dict[str, Any]],
    document_units: list[dict[str, Any]],
) -> dict[str, set[str]]:
    out = {participant["participant_id"]: set() for participant in participants}
    for unit in document_units:
        rui_ids = {str(item) for item in unit.get("hidden_contains_rui_ids", [])}
        for participant_id in unit.get("assigned_to", []):
            out.setdefault(str(participant_id), set()).update(rui_ids)
    return out


def _covers_full_document(context: str, sorted_units: list[dict[str, Any]]) -> bool:
    if not sorted_units:
        return False
    if int(sorted_units[0]["start"]) != 0 or int(sorted_units[-1]["end"]) != len(context):
        return False
    return all(int(left["end"]) == int(right["start"]) for left, right in zip(sorted_units, sorted_units[1:]))


def _non_overlapping(sorted_units: list[dict[str, Any]]) -> bool:
    return all(int(left["end"]) <= int(right["start"]) for left, right in zip(sorted_units, sorted_units[1:]))


def _no_rui_split(rui_ranges: Sequence[EvidenceSpan], sorted_units: list[dict[str, Any]]) -> bool:
    for rui in rui_ranges:
        containing = [
            unit
            for unit in sorted_units
            if int(unit["start"]) <= rui.raw_start and rui.raw_end <= int(unit["end"])
        ]
        if len(containing) != 1:
            return False
    return True


def _check(checks: list[dict[str, Any]], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})


def number_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 1),
        "median": ordered[len(ordered) // 2],
    }


def number_gap(values: list[int]) -> int | None:
    if not values:
        return None
    return max(values) - min(values)
