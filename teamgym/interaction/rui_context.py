"""Datapoint transforms for RUI-only interaction baselines."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from teamgym.core.io import estimate_tokens
from teamgym.core.schema import validate_common_datapoint


_RAW_SOURCE_DATASETS = {'findver', 'finlongdocqa', 'evidencebench', 'mlcr', 'mlcr_aa',
                        'swe_atlas', 'deep_swe', 'deepswe'}


def _raw_source_only(datapoint: dict[str, Any]) -> bool:
    """Non-Legal constructors certify source evidence, not hidden annotations."""
    return str(datapoint.get('dataset') or '').lower().replace('-', '_') in _RAW_SOURCE_DATASETS


def model_common_information(datapoint: dict[str, Any]) -> Any:
    """Keep public facts while excluding constructor annotations on public PIUs."""
    common = deepcopy(datapoint.get('initial_common_information'))
    if _raw_source_only(datapoint) and isinstance(common, dict):
        units = common.get('common_reference_units')
        if units is not None:
            if not isinstance(units, list):
                raise ValueError('common reference units must be a list')
            common['common_reference_units'] = [
                {'text': _rui_context_text(unit, raw_only=True)} for unit in units]
    return common


def rui_only_datapoint(datapoint: dict[str, Any]) -> dict[str, Any]:
    """Return a copy whose participant private context contains only owned RIUs."""

    out = deepcopy(datapoint)
    rui_by_id = {
        str(row.get("rui_id")): row
        for row in out.get("reference_information_units", [])
        if isinstance(row, dict) and row.get("rui_id")
    }
    private_context_units: list[dict[str, Any]] = []
    for participant in out.get("participants", []):
        if not isinstance(participant, dict):
            continue
        participant_id = str(participant.get("participant_id") or "")
        owned_rui_ids = [
            str(rui_id)
            for rui_id in participant.get("owned_rui_ids", [])
            if str(rui_id) in rui_by_id
        ]
        context_unit_ids: list[str] = []
        for rui_id in owned_rui_ids:
            rui = rui_by_id[rui_id]
            context_unit_id = f"rui_only_{participant_id}_{rui_id}"
            text = _rui_context_text(rui, raw_only=_raw_source_only(datapoint))
            private_context_units.append(
                {
                    "context_unit_id": context_unit_id,
                    "text": text,
                    "assigned_to": [participant_id],
                    "word_count": len(text.split()),
                    "source_ref": {
                        "context_variant": "rui_only",
                        "rui_id": rui_id,
                        "original_source_ref": rui.get("source_ref") if isinstance(rui.get("source_ref"), dict) else {},
                    },
                    "hidden_contains_rui_ids": [rui_id],
                    "assignment": {
                        "reason": "rui_only_owned_reference_unit",
                        "scores": {},
                    },
                }
            )
            context_unit_ids.append(context_unit_id)
        participant["private_context_unit_ids"] = context_unit_ids
        participant["private_word_count"] = sum(
            len(unit["text"].split())
            for unit in private_context_units
            if participant_id in set(unit.get("assigned_to") or [])
        )

    generation_metadata = out.get("generation_metadata") if isinstance(out.get("generation_metadata"), dict) else {}
    out["generation_metadata"] = {
        **generation_metadata,
        "interaction_context_variant": "rui_only_private_context",
        "rui_only_context": {
            "private_context_unit_policy": "one_private_context_unit_per_participant_owned_rui",
            "full_document_context_removed": True,
        },
    }
    dataset_specific = out.get("dataset_specific") if isinstance(out.get("dataset_specific"), dict) else {}
    out["dataset_specific"] = {
        **dataset_specific,
        "interaction_context_variant": "rui_only",
    }
    out["private_context_units"] = private_context_units
    out["format_validation"] = validate_common_datapoint(out)
    return out


def all_rui_context_text(datapoint: dict[str, Any]) -> str:
    """Format every reference information unit as one single-agent context block."""

    blocks = [_rui_context_text(row, raw_only=_raw_source_only(datapoint))
              for row in datapoint.get("reference_information_units", []) if isinstance(row, dict)]
    return "\n\n".join(block for block in blocks if block.strip())


def rui_context_units(datapoint: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "rui_id": row.get("rui_id"),
            "text": _rui_context_text(row, raw_only=_raw_source_only(datapoint)),
            "estimated_tokens": estimate_tokens(_rui_context_text(row, raw_only=_raw_source_only(datapoint))),
        }
        for row in datapoint.get("reference_information_units", [])
        if isinstance(row, dict)
    ]


def _rui_context_text(rui: dict[str, Any], *, raw_only=False) -> str:
    if raw_only:
        text = rui.get('text') or rui.get('raw_text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('PIU source text is required; hidden summaries cannot replace source evidence')
        return text
    rui_id = str(rui.get("rui_id") or "")
    lines = [f"RIU {rui_id}".strip()]
    if rui.get("information_feature"):
        lines.append(f"Feature: {rui.get('information_feature')}")
    if rui.get("summary"):
        lines.append(f"Summary: {rui.get('summary')}")
    text = str(rui.get("text") or rui.get("raw_text") or "").strip()
    if text:
        lines.append(f"Text: {text}")
    return "\n".join(lines)
