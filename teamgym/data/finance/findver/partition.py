"""Assign every native FinDVer report context to RIU-aware private packets."""

from __future__ import annotations

import random
from copy import deepcopy
import re
from typing import Any

from ....core.information_allocation import (
    PAPER_ALLOCATION_MODES,
    assign_reference_units,
    owners_by_reference_unit,
)
from ....core.schema import build_common_datapoint


def build_full_context_partition(
    *,
    candidate: dict[str, Any],
    selection: dict[str, Any],
    plan: dict[str, Any],
    audits: dict[str, Any],
    seed: int,
    allocation_mode: str = "one_each",
    min_participants: int = 2,
) -> dict[str, Any]:
    selected = selection.get("selected_contexts") or []
    rui_ids = [str(row["unit_id"]) for row in selected]
    smallest_sets = _smallest_sufficient_sets(
        plan.get("oracle") or {},
        rui_ids,
    )
    allocation = assign_reference_units(
        rui_ids,
        mode=allocation_mode,
        rng=random.Random(f"{candidate['source_id']}:{seed}:riu"),
        min_participants=min_participants,
        smallest_sufficient_sets=smallest_sets,
        allow_common_anchor=True,
        allowed_modes=PAPER_ALLOCATION_MODES,
    )
    if not allocation.validation.get("valid"):
        raise ValueError(f"Invalid RIU allocation: {allocation.validation}")

    selected_by_unit = {str(row["unit_id"]): row for row in selected}
    owners_by_rui = owners_by_reference_unit(allocation.private_assignment)
    participants: list[dict[str, Any]] = []
    internal_to_public: dict[str, str] = {}
    for index, (internal_id, owned) in enumerate(
        sorted(allocation.private_assignment.items()),
        start=1,
    ):
        public_id = _participant_id(index)
        internal_to_public[internal_id] = public_id
        selected_context_ids = [
            context_id
            for rui_id in owned
            for context_id in _source_context_ids(selected_by_unit[rui_id])
        ]
        participants.append(
            {
                "participant_id": public_id,
                "role_name": f"financial filing context reviewer {index}",
                "private_context_unit_ids": [],
                "owned_rui_ids": list(owned),
                "selected_riu_context_ids": selected_context_ids,
                "approx_input_tokens": 0,
            }
        )

    participant_by_id = {row["participant_id"]: row for row in participants}
    all_participant_ids = sorted(participant_by_id)
    public_owners_by_rui = {
        rui_id: [
            internal_to_public[internal_id]
            for internal_id in internal_ids
            if internal_id in internal_to_public
        ]
        for rui_id, internal_ids in owners_by_rui.items()
    }
    common_rui_ids = set(allocation.common_rui_ids)
    owner_by_context: dict[int, list[str]] = {}
    for rui_id in rui_ids:
        for context_id in _source_context_ids(selected_by_unit[rui_id]):
            owner_by_context[context_id] = (
                list(all_participant_ids)
                if rui_id in common_rui_ids
                else list(public_owners_by_rui.get(rui_id, []))
            )
    selected_riu_positions = [
        (context_id, participant_id)
        for rui_id in rui_ids
        for context_id in _source_context_ids(selected_by_unit[rui_id])
        for participant_id in owner_by_context[context_id]
    ]
    affinity_terms = _unique_affinity_terms(
        {
            internal_to_public[internal_id]: " ".join(
                f"{selected_by_unit[rui_id].get('information_feature', '')} "
                f"{selected_by_unit[rui_id].get('text', '')}"
                for rui_id in rui_ids_for_owner
            )
            for internal_id, rui_ids_for_owner in allocation.private_assignment.items()
        }
    )
    contexts: list[dict[str, Any]] = []
    for source in candidate.get("full_report_contexts") or []:
        context_id = int(source["context_id"])
        text = str(source.get("text") or "")
        approx_tokens = max(1, len(text) // 4)
        owners = owner_by_context.get(context_id)
        allocation_reason = "selected_riu"
        if owners is None:
            owner = _affinity_owner(text, affinity_terms)
            allocation_reason = "unique_riu_lexical_affinity"
            if owner is not None:
                owners = [owner]
        if owners is None:
            owner = _nearest_selected_riu_owner(
                context_id,
                selected_riu_positions,
                participant_by_id,
                approx_tokens,
            )
            owners = [owner]
            allocation_reason = "token_balance_with_selected_riu_tiebreak"
        unit_id = f"ctx_{candidate['source_id']}_{context_id:04d}"
        hidden = [
            str(row["unit_id"])
            for row in selected
            if context_id in _source_context_ids(row)
        ]
        contexts.append(
            {
                "context_unit_id": unit_id,
                "text": text,
                "assigned_to": owners,
                "approx_tokens": approx_tokens,
                "source_ref": {
                    "context_id": context_id,
                    "context_type": source.get("type"),
                    "report": candidate.get("report_name"),
                },
                "hidden_contains_rui_ids": hidden,
                "allocation_reason": allocation_reason,
            }
        )
        for owner in owners:
            participant_by_id[owner]["private_context_unit_ids"].append(unit_id)
            participant_by_id[owner]["approx_input_tokens"] += approx_tokens

    public = plan["public_task"]
    oracle = plan["oracle"]
    decision_set = ["free_response"]
    decision_option_definitions = {
        "free_response": str(
            public.get("response_format_note")
            or "Submit a concise source-grounded financial answer."
        )
    }
    rui_units = [
        {
            "rui_id": str(row["unit_id"]),
            "text": row["text"],
            "information_feature": row.get("information_feature"),
            "entity_scope": row.get("entity_scope"),
            "period_scope": row.get("period_scope"),
            "source_qa_ids": list(row.get("source_qa_ids") or []),
            "source_ref": {
                "context_id": row.get("context_id"),
                "context_ids": _source_context_ids(row),
                "context_type": row.get("context_type"),
                "context_types": list(row.get("source_context_types") or []),
                "report": candidate.get("report_name"),
            },
        }
        for row in selected
    ]
    common = build_common_datapoint(
        instance_id=f"findver_ie_{candidate['source_id']}",
        dataset="findver",
        source={
            "source_dataset": "FinDVer",
            "source_id": candidate.get("source_id"),
            "source_subset": candidate.get("subset"),
            "source_split": candidate.get("split"),
            "report": candidate.get("report_name"),
            "report_type": candidate.get("report_type"),
            "report_url": candidate.get("report_url"),
            "source_preservation": candidate.get("context_policy")
            or "all_native_report_contexts_partitioned_once",
        },
        scenario_family="findver_free_response_reasoning",
        task_description={
            "title": public.get("title"),
            "question": public.get("question") or public.get("objective"),
            "objective": public.get("question") or public.get("objective"),
            "answer_type": "short_answer_judged",
            "response_format_note": public.get("response_format_note"),
            "answer_format": {
                "type": "free_response",
                "instruction": public.get("response_format_note"),
            },
        },
        decision_set=decision_set,
        decision_option_definitions=decision_option_definitions,
        initial_common_information={
            "instruction": public.get("instruction"),
            "report_type": candidate.get("report_type"),

            "common_reference_units": [
                {
                    "rui_id": rui_id,
                    "text": selected_by_unit[rui_id]["text"],
                    "information_feature": selected_by_unit[rui_id].get(
                        "information_feature"
                    ),
                }
                for rui_id in sorted(common_rui_ids)
            ],
        },
        participants=[
            {
                key: value
                for key, value in participant.items()
                if key != "selected_riu_context_ids"
            }
            for participant in participants
        ],
        private_context_units=contexts,
        reference_information_units=rui_units,
        oracle={
            "answer": "free_response",
            "short_answer": oracle.get("short_answer"),
            "canonical_response": oracle.get("canonical_response"),
            "answer_spec": oracle.get("answer_spec"),
            "reasoning_steps": oracle.get("reasoning_steps"),
            "required_decision_steps": oracle.get("required_decision_steps")
            or oracle.get("reasoning_steps"),
            "judge_rubric": {**(oracle.get("judge_rubric") or {}), "require_all_steps_for_pass": True,
                             "answer_spec": deepcopy(oracle.get("answer_spec") or {})},
            "pass_policy": "all_required_decision_steps",
            "disqualifying_errors": oracle.get("disqualifying_errors"),
            "semantic_judge": {
                "enabled": True,
                "primary_answer_field": "written_answer",
                "fallback_answer_fields": ["answer", "rationale"],
            },
            "minimum_sufficient_rui_set_size": min(
                len(row) for row in smallest_sets
            ),
            "smallest_sufficient_rui_sets": [
                {
                    "rui_ids": list(sufficient_set),
                    "answer": "free_response",
                }
                for sufficient_set in smallest_sets
            ],
        },
        generation_metadata={
            "method": "findver_free_response_context_funnel_v1",
            "seed": seed,
            "private_information_allocation": allocation.to_metadata(),
            "audit_stage_names": sorted(audits),
            "audit_passed": {name: True for name in audits},
            "audit_detail_location": "generation_byproducts.jsonl",
        },
        dataset_specific={
            "selected_context_ids": [
                context_id
                for row in selected
                for context_id in _source_context_ids(row)
            ],
            "selected_candidate_block_ids": [
                row.get("candidate_block_id") for row in selected
            ],
            "source_qa_ids_used": list(
                selection.get("source_qa_ids_used") or []
            ),
        },
    )
    permutation = {
        "schema_version": "collabgym_fc_decision_option_permutation_v1",
        "skipped": True,
        "reason": "free_response",
    }
    return {
        "schema": "findver_ie_full_context_partition_v1",
        "instance_id": common["instance_id"],
        "dataset": "findver",
        "source_id": candidate.get("source_id"),
        "common_datapoint": common,
        "decision_permutation": permutation,
        "packet_manifest": packet_manifest(
            participants=participants,
            context_units=contexts,
            include_text=False,
        ),
        "validation": common.get("format_validation"),
    }


def _source_context_ids(row: dict[str, Any]) -> list[int]:
    return [
        int(value)
        for value in row.get("source_context_ids") or [row["context_id"]]
    ]


def _smallest_sufficient_sets(
    oracle: dict[str, Any],
    selected_rui_ids: list[str],
) -> list[list[str]]:
    selected = set(selected_rui_ids)
    raw_sets = oracle.get("smallest_sufficient_rui_sets")
    if raw_sets is None:
        return [list(selected_rui_ids)]
    if not isinstance(raw_sets, list) or not raw_sets:
        raise ValueError("oracle.smallest_sufficient_rui_sets must be non-empty")

    normalized: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for index, row in enumerate(raw_sets, start=1):
        values = row
        if isinstance(row, dict):
            values = row.get("rui_ids") or row.get("unit_ids")
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"smallest sufficient set {index} must contain rui_ids"
            )
        sufficient_set = list(
            dict.fromkeys(str(value) for value in values if str(value))
        )
        unknown = set(sufficient_set) - selected
        if unknown:
            raise ValueError(
                f"smallest sufficient set {index} references unknown RIUs: "
                f"{sorted(unknown)}"
            )
        key = tuple(sorted(sufficient_set))
        if key not in seen:
            normalized.append(sufficient_set)
            seen.add(key)
    if not normalized:
        raise ValueError("oracle.smallest_sufficient_rui_sets has no usable sets")
    return normalized


def packet_manifest(
    *,
    participants: list[dict[str, Any]],
    context_units: list[dict[str, Any]],
    include_text: bool,
) -> list[dict[str, Any]]:
    by_owner: dict[str, list[dict[str, Any]]] = {
        str(row["participant_id"]): []
        for row in participants
    }
    for unit in context_units:
        for owner in unit.get("assigned_to") or []:
            by_owner.setdefault(str(owner), []).append(unit)
    out: list[dict[str, Any]] = []
    for participant in participants:
        participant_id = str(participant["participant_id"])
        units = sorted(
            by_owner.get(participant_id, []),
            key=lambda row: int((row.get("source_ref") or {}).get("context_id") or 0),
        )
        packet = {
            "participant_id": participant_id,
            "owned_rui_ids": list(participant.get("owned_rui_ids") or []),
            "context_unit_ids": [row["context_unit_id"] for row in units],
            "source_context_ids": [
                (row.get("source_ref") or {}).get("context_id")
                for row in units
            ],
            "approx_tokens": sum(int(row.get("approx_tokens") or 0) for row in units),
        }
        if include_text:
            packet["text"] = "\n\n".join(
                f"[Native context {row['source_ref']['context_id']}]\n{row['text']}"
                for row in units
            )
        out.append(packet)
    return out


def remove_surrounding_contexts(
    scenario: dict[str, Any],
    *,
    source_context_ids: set[int],
    selected_riu_context_ids: set[int],
) -> dict[str, Any]:
    """Remove leaking surrounding blocks without changing packet ownership."""

    protected = source_context_ids & selected_riu_context_ids
    if protected:
        raise ValueError(
            "selected RIU blocks cannot be removed: "
            f"{sorted(protected)}"
        )

    common = scenario["common_datapoint"]
    original_units = list(common.get("private_context_units") or [])
    original_source_ids = {
        int((row.get("source_ref") or {}).get("context_id"))
        for row in original_units
    }
    requested = {int(value) for value in source_context_ids}
    removed_source_ids = requested & original_source_ids
    retained_units = [
        row
        for row in original_units
        if int((row.get("source_ref") or {}).get("context_id"))
        not in removed_source_ids
    ]
    retained_by_id = {
        str(row["context_unit_id"]): row for row in retained_units
    }
    participants = common.get("participants") or []
    for participant in participants:
        retained_ids = [
            str(unit_id)
            for unit_id in participant.get("private_context_unit_ids") or []
            if str(unit_id) in retained_by_id
        ]
        participant["private_context_unit_ids"] = retained_ids
        participant["approx_input_tokens"] = sum(
            int(retained_by_id[unit_id].get("approx_tokens") or 0)
            for unit_id in retained_ids
        )

    common["private_context_units"] = retained_units
    scenario["packet_manifest"] = packet_manifest(
        participants=participants,
        context_units=retained_units,
        include_text=False,
    )
    return {
        "requested_context_ids": sorted(requested),
        "removed_context_ids": sorted(removed_source_ids),
        "unknown_context_ids": sorted(requested - original_source_ids),
        "original_context_count": len(original_units),
        "retained_context_count": len(retained_units),
        "policy": (
            "remove_clear_unassigned_riu_visibility_from_surrounding_blocks_"
            "without_reallocation"
        ),
    }


def validate_packet_context_integrity(
    scenario: dict[str, Any],
    *,
    selected_riu_context_ids: set[int],
    removed_context_ids: set[int],
) -> list[str]:
    """Check the fixed packet assignment after one-pass context removal."""

    errors: list[str] = []
    common = scenario.get("common_datapoint") or {}
    participants = common.get("participants") or []
    participant_ids = {
        str(row.get("participant_id"))
        for row in participants
        if isinstance(row, dict) and row.get("participant_id")
    }
    units = common.get("private_context_units") or []
    units_by_id = {
        str(row.get("context_unit_id")): row
        for row in units
        if isinstance(row, dict) and row.get("context_unit_id")
    }
    source_ids = {
        int((row.get("source_ref") or {}).get("context_id"))
        for row in units_by_id.values()
    }
    missing_selected = selected_riu_context_ids - source_ids
    if missing_selected:
        errors.append(
            f"selected RIU contexts are missing: {sorted(missing_selected)}"
        )
    retained_removed = removed_context_ids & source_ids
    if retained_removed:
        errors.append(
            f"removed contexts are still present: {sorted(retained_removed)}"
        )

    expected_units_by_participant = {
        participant_id: set() for participant_id in participant_ids
    }
    for unit_id, unit in units_by_id.items():
        owners = [str(value) for value in unit.get("assigned_to") or []]
        if not owners:
            errors.append(f"{unit_id}: context has no participant")
            continue
        unknown_owners = set(owners) - participant_ids
        if unknown_owners:
            errors.append(
                f"{unit_id}: unknown participants {sorted(unknown_owners)}"
            )
        context_id = int((unit.get("source_ref") or {}).get("context_id"))
        if context_id not in selected_riu_context_ids and len(set(owners)) != 1:
            errors.append(
                f"{unit_id}: surrounding block must belong to exactly one packet"
            )
        for owner in owners:
            if owner in expected_units_by_participant:
                expected_units_by_participant[owner].add(unit_id)

    for participant in participants:
        participant_id = str(participant["participant_id"])
        actual_ids = {
            str(value)
            for value in participant.get("private_context_unit_ids") or []
        }
        expected_ids = expected_units_by_participant.get(participant_id, set())
        if actual_ids != expected_ids:
            errors.append(
                f"{participant_id}: packet context references do not match assignment"
            )
        expected_tokens = sum(
            int(units_by_id[unit_id].get("approx_tokens") or 0)
            for unit_id in expected_ids
        )
        if int(participant.get("approx_input_tokens") or 0) != expected_tokens:
            errors.append(f"{participant_id}: packet token count is stale")

    allocation = (
        (common.get("generation_metadata") or {})
        .get("private_information_allocation")
        or {}
    )
    if not (allocation.get("validation") or {}).get("valid"):
        errors.append("RIU allocation validation is not valid")
    return list(dict.fromkeys(errors))


def _nearest_selected_riu_owner(
    context_id: int,
    selected_riu_positions: list[tuple[int, str]],
    participant_by_id: dict[str, dict[str, Any]],
    approx_tokens: int,
) -> str:
    selected_positions_by_owner: dict[str, list[int]] = {}
    for selected_context_id, participant_id in selected_riu_positions:
        selected_positions_by_owner.setdefault(participant_id, []).append(
            selected_context_id
        )
    return min(
        participant_by_id,
        key=lambda participant_id: (
            participant_by_id[participant_id]["approx_input_tokens"] + approx_tokens,
            min(
                (
                    abs(context_id - selected_context_id)
                    for selected_context_id in selected_positions_by_owner.get(
                        participant_id, []
                    )
                ),
                default=float("inf"),
            ),
            participant_id,
        ),
    )


_TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{4,}|\d+(?:\.\d+)?", re.IGNORECASE)


def _unique_affinity_terms(text_by_owner: dict[str, str]) -> dict[str, set[str]]:
    tokens_by_owner = {
        owner: set(_TOKEN_RE.findall(text.lower()))
        for owner, text in text_by_owner.items()
    }
    owners_by_token: dict[str, set[str]] = {}
    for owner, tokens in tokens_by_owner.items():
        for token in tokens:
            owners_by_token.setdefault(token, set()).add(owner)
    return {
        owner: {
            token
            for token in tokens
            if len(owners_by_token.get(token, set())) == 1
        }
        for owner, tokens in tokens_by_owner.items()
    }


def _affinity_owner(
    text: str,
    affinity_terms: dict[str, set[str]],
) -> str | None:
    tokens = set(_TOKEN_RE.findall(text.lower()))
    scored = sorted(
        (
            (
                sum(4 if any(char.isdigit() for char in token) else (2 if len(token) >= 9 else 1)
                    for token in tokens & terms),
                owner,
            )
            for owner, terms in affinity_terms.items()
        ),
        reverse=True,
    )
    if not scored or scored[0][0] < 4:
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _participant_id(index: int) -> str:
    if index < 1:
        raise ValueError("participant index must be positive")
    letters = ""
    current = index
    while current:
        current, remainder = divmod(current - 1, 26)
        letters = chr(ord("a") + remainder) + letters
    return f"participant_{letters}"
