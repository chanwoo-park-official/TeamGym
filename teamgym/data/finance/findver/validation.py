"""Deterministic validation for FinDVer RIU selection."""

from __future__ import annotations

from typing import Any


def normalize_selection(
    selection: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    by_candidate_id = {
        str(row["candidate_block_id"]): row
        for row in candidate.get("candidate_contexts") or []
        if row.get("candidate_block_id")
    }
    by_context = {
        int(row["context_id"]): row
        for row in candidate.get("candidate_contexts") or []
    }
    selected = selection.get("selected_contexts")
    if isinstance(selected, list):
        for index, row in enumerate(selected, start=1):
            if not isinstance(row, dict):
                continue
            candidate_block_id = str(row.get("candidate_block_id") or "")
            context_id = _as_int(row.get("context_id"))
            source = by_candidate_id.get(candidate_block_id) or by_context.get(
                context_id
            )
            if source is None:
                continue
            candidate_block_id = str(
                source.get("candidate_block_id")
                or candidate_block_id
                or f"ctx_{source['context_id']}"
            )
            row["candidate_block_id"] = candidate_block_id
            row["unit_id"] = str(
                row.get("unit_id") or f"riu_{candidate_block_id}"
            )
            row["context_id"] = int(source["context_id"])
            row["source_context_ids"] = [
                int(value)
                for value in source.get("source_context_ids")
                or [source["context_id"]]
            ]
            row["text"] = source["text"]
            row["context_type"] = source["type"]
            row["source_context_types"] = list(
                source.get("source_context_types") or [source["type"]]
            )
            row["source_qa_ids"] = list(source.get("source_qa_ids") or [])
            row["source_subsets"] = list(source.get("source_subsets") or [])
            row["selection_order"] = index
        selection["source_qa_ids_used"] = sorted(
            {
                str(qa_id)
                for row in selected
                if isinstance(row, dict)
                for qa_id in row.get("source_qa_ids") or []
                if str(qa_id)
            }
        )
    return selection


def validate_selection(
    selection: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_units: int,
    max_units: int,
) -> list[str]:
    errors: list[str] = []
    if selection.get("valid") is not True:
        reason = str(selection.get("failed_reason") or "").strip()
        errors.append(f"selector rejected candidate: {reason}".strip())
    source_by_candidate_id = {
        str(row.get("candidate_block_id") or f"ctx_{row['context_id']}"): row
        for row in candidate.get("candidate_contexts") or []
    }
    selected = selection.get("selected_contexts")
    if not isinstance(selected, list):
        return errors + ["selected_contexts must be a list"]
    if not min_units <= len(selected) <= max_units:
        errors.append(
            f"selected_contexts must contain {min_units}..{max_units} candidate blocks"
        )
    unit_ids: list[str] = []
    candidate_block_ids: list[str] = []
    selected_native_ids: list[int] = []
    features: list[str] = []
    for row in selected:
        if not isinstance(row, dict):
            errors.append("each selected_contexts row must be an object")
            continue
        unit_id = str(row.get("unit_id") or "")
        candidate_block_id = str(row.get("candidate_block_id") or "")
        feature = str(row.get("information_feature") or "").strip()
        unit_ids.append(unit_id)
        candidate_block_ids.append(candidate_block_id)
        features.append(feature.lower())
        if not unit_id:
            errors.append("each selected context requires unit_id")
        source = source_by_candidate_id.get(candidate_block_id)
        if source is None:
            errors.append(f"{unit_id}: unknown candidate block {candidate_block_id}")
        else:
            expected_ids = [
                int(value)
                for value in source.get("source_context_ids")
                or [source["context_id"]]
            ]
            actual_ids = [
                int(value) for value in row.get("source_context_ids") or []
            ]
            if actual_ids != expected_ids:
                errors.append(f"{unit_id}: source_context_ids do not match candidate")
            if str(row.get("text") or "") != str(source.get("text") or ""):
                errors.append(f"{unit_id}: selected text is not the exact candidate text")
            selected_native_ids.extend(actual_ids)
        if not feature:
            errors.append(f"{unit_id}: information_feature is required")
        if candidate.get("source_qas") and not row.get("source_qa_ids"):
            errors.append(f"{unit_id}: selected context has no valid source QA")
        for key in ("entity_scope", "period_scope", "why_critical"):
            if not str(row.get(key) or "").strip():
                errors.append(f"{unit_id}: {key} is required")
    if len(set(unit_ids)) != len(unit_ids):
        errors.append("selected unit IDs must be unique")
    if len(set(candidate_block_ids)) != len(candidate_block_ids):
        errors.append("selected candidate block IDs must be unique")
    if len(set(selected_native_ids)) != len(selected_native_ids):
        errors.append("selected candidate blocks overlap in native contexts")
    if len(set(features)) != len(features):
        errors.append("selected information features must be distinct")
    return _dedupe(errors)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
