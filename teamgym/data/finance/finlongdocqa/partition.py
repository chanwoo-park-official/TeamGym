"""Build RIU-aware full-report partitions for FinLongDocQA."""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Any

from teamgym.data.finance.finlongdocqa.schema import (
    build_common_datapoint,
    validate_common_datapoint,
)
from teamgym.core.information_allocation import (
    PAPER_ALLOCATION_MODES,
    assign_reference_units,
    normalize_allocation_mode,
    owners_by_reference_unit,
)


def build_one_each_datapoint(
    *,
    report_id: str,
    company: str,
    year: str,
    report_source: dict[str, Any],
    report_index: dict[str, Any],
    task: dict[str, Any],
    qa_mappings: list[dict[str, Any]],
    solver_audit: dict[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper for the original deterministic partition."""
    return build_partitioned_datapoint(
        report_id=report_id,
        company=company,
        year=year,
        report_source=report_source,
        report_index=report_index,
        task=task,
        qa_mappings=qa_mappings,
        solver_audit=solver_audit,
        allocation_mode="one_each",
        allocation_seed=0,
        preserve_rui_order=True,
    )


def build_partitioned_datapoint(
    *,
    report_id: str,
    company: str,
    year: str,
    report_source: dict[str, Any],
    report_index: dict[str, Any],
    task: dict[str, Any],
    qa_mappings: list[dict[str, Any]],
    solver_audit: dict[str, Any],
    allocation_mode: str = "one_each",
    allocation_seed: int = 0,
    min_participants: int = 2,
    preserve_rui_order: bool = False,
) -> dict[str, Any]:
    units = [
        row for row in task.get("evidence_units") or [] if isinstance(row, dict)
    ]
    if len(units) < 2:
        raise ValueError("partition requires at least two final RIUs")
    rui_ids = [str(row.get("unit_id") or "") for row in units]
    normalized_mode = normalize_allocation_mode(allocation_mode)
    rng: random.Random = random.Random(f"{report_id}:{allocation_seed}:riu")
    if preserve_rui_order and normalized_mode == "one_each":
        rng = _NoShuffleRandom(f"{report_id}:{allocation_seed}:riu")
    allocation = assign_reference_units(
        rui_ids,
        mode=normalized_mode,
        rng=rng,
        min_participants=min_participants,
        smallest_sufficient_sets=[rui_ids],
        allow_common_anchor=True,
        allowed_modes=PAPER_ALLOCATION_MODES,
    )
    if not allocation.validation.get("valid"):
        raise ValueError(f"Invalid RIU allocation: {allocation.validation}")

    canonical_block_by_rui = {
        str(row.get("unit_id") or ""): str(row.get("source_block_id") or "")
        for row in units
    }
    internal_to_public = {
        internal_id: _letter_id("participant", index)
        for index, internal_id in enumerate(
            sorted(allocation.private_assignment), start=1
        )
    }
    participants = []
    for index, internal_id in enumerate(
        sorted(allocation.private_assignment), start=1
    ):
        participants.append(
            {
                "participant_id": internal_to_public[internal_id],
                "role_name": f"annual-report evidence reviewer {index}",
                "owned_rui_ids": list(
                    allocation.private_assignment[internal_id]
                ),
                "private_context_unit_ids": [],
                "private_word_count": 0,
            }
        )
    participant_by_id = {
        str(row["participant_id"]): row for row in participants
    }
    private_owners = owners_by_reference_unit(allocation.private_assignment)
    common_rui_ids = set(allocation.common_rui_ids)
    owners_by_rui = {
        rui_id: (
            sorted(participant_by_id)
            if rui_id in common_rui_ids
            else [
                internal_to_public[internal_id]
                for internal_id in private_owners.get(rui_id, [])
            ]
        )
        for rui_id in rui_ids
    }
    owners_by_block: dict[str, list[str]] = {}
    rui_ids_by_block: dict[str, list[str]] = {}
    for rui_id in rui_ids:
        block_id = canonical_block_by_rui[rui_id]
        owners_by_block.setdefault(block_id, [])
        owners_by_block[block_id].extend(owners_by_rui[rui_id])
        owners_by_block[block_id] = sorted(set(owners_by_block[block_id]))
        rui_ids_by_block.setdefault(block_id, []).append(rui_id)
    unit_by_rui = {
        str(row.get("unit_id") or ""): row for row in units
    }
    all_anchor_pages = [
        int((unit_by_rui[rui_id].get("source_ref") or {}).get("page_number") or 0)
        for rui_id in rui_ids
    ]
    fallback_anchor = sum(all_anchor_pages) // max(1, len(all_anchor_pages))
    anchors = {}
    for participant_id, participant in participant_by_id.items():
        pages = [
            int(
                (unit_by_rui[rui_id].get("source_ref") or {}).get(
                    "page_number"
                )
                or 0
            )
            for rui_id in participant.get("owned_rui_ids") or []
        ]
        anchors[participant_id] = (
            sum(pages) // len(pages) if pages else fallback_anchor
        )
    blocks = [
        row
        for row in report_index.get("blocks") or []
        if isinstance(row, dict)
    ]
    words_by_block = {
        str(block.get("block_id") or ""): len(
            str(next((unit["text"] for unit in units if unit["source_block_id"] == block["block_id"]), block.get("text") or "")).split()
        )
        for block in blocks
    }
    # Account for every fixed canonical RIU before allocating any other block.
    # Otherwise early non-canonical blocks are balanced against incomplete
    # participant loads and a late canonical block can create a large skew.
    assigned_words = {
        participant_id: sum(
            words_by_block.get(block_id, 0)
            for block_id, owners in owners_by_block.items()
            if participant_id in owners
        )
        for participant_id in participant_by_id
    }
    contexts: list[dict[str, Any]] = []
    for block in blocks:
        block_id = str(block.get("block_id") or "")
        if block_id in owners_by_block:
            owners = owners_by_block[block_id]
            reason = "canonical_rui_owner"
        else:
            page_number = int(block.get("page_number") or 0)
            owner = min(
                participant_by_id,
                key=lambda participant_id: (
                    assigned_words[participant_id],
                    abs(page_number - anchors[participant_id]),
                    participant_id,
                ),
            )
            owners = [owner]
            reason = "least_loaded_packet_then_nearest_rui_page"
        context_id = f"ctx_{report_id}_{block_id}"
        matching_unit = next((unit for unit in units if str(unit.get("source_block_id") or "") == block_id), None)
        text = str(matching_unit.get("text") if matching_unit else block.get("text") or "")
        word_count = len(text.split())
        contexts.append(
            {
                "context_unit_id": context_id,
                "text": text,
                "assigned_to": owners,
                "word_count": word_count,
                "source_ref": {
                    "block_id": block_id,
                    "page_number": block.get("page_number"),
                    "block_type": block.get("block_type"),
                    "heading": block.get("heading"),
                    "raw_start": block.get("raw_start"),
                    "raw_end": block.get("raw_end"),
                    **({k: v for k, v in (matching_unit.get("source_ref") or {}).items() if k in {"source_row_numbers", "source_row_spans", "original_block_sha256"}} if matching_unit else {}),
                },
                "hidden_contains_rui_ids": list(
                    rui_ids_by_block.get(block_id, [])
                ),
                "assignment": {"reason": reason},
            }
        )
        for owner in owners:
            participant_by_id[owner]["private_context_unit_ids"].append(
                context_id
            )
            participant_by_id[owner]["private_word_count"] += word_count
        if block_id not in owners_by_block:
            assigned_words[owners[0]] += word_count

    public = task.get("public_task") or {}
    response_requirements = [
        str(value)
        for value in public.get("response_requirements") or []
        if str(value).strip()
    ]
    answer_token = "free_response"
    reference_answer = str(
        (task.get("oracle") or {}).get("reference_answer") or ""
    )
    datapoint = build_common_datapoint(
        instance_id=f"finlongdocqa_{report_id}",
        dataset="finlongdocqa",
        source={
            "source_dataset": "FinLongDocQA",
            "report_id": report_id,
            "company": company,
            "year": year,
            "report_source": report_source,
            "parser_version": report_index.get("parser_version"),
        },
        scenario_family="multi_qa_annual_report_reasoning",
        task_description={
            "answer_type": "short_answer_judged",
            "title": public.get("title"),
            "objective": public.get("question"),
            "report_context": public.get("report_context"),
            "answer_format": {
                "type": "free_response",
                "instruction": " ".join(response_requirements),
            },
        },
        decision_set=[answer_token],
        decision_option_definitions={
            answer_token: "Submit a concise evidence-grounded written response."
        },
        initial_common_information={
            "instruction": (
                "Write the requested response using private annual-report "
                "evidence distributed across the team."
            ),
            "report_id": report_id,
            "common_reference_units": [
                {
                    "rui_id": rui_id,
                    "text": str(unit_by_rui[rui_id].get("text") or ""),
                    "information_feature": str(
                        unit_by_rui[rui_id].get("information_feature") or ""
                    ),
                }
                for rui_id in sorted(common_rui_ids)
            ],
            **({'source_unit_conventions': [r['text'] for r in report_index.get('source_unit_conventions', [])]}
               if report_index.get('source_unit_conventions') else {}),
        },
        participants=participants,
        private_context_units=contexts,
        reference_information_units=[
            {
                "rui_id": str(unit.get("unit_id") or ""),
                "text": str(unit.get("text") or ""),
                "information_feature": str(
                    unit.get("information_feature") or ""
                ),
                "summary": str(unit.get("hidden_summary") or ""),
                "supports_claim_ids": [
                    str(value)
                    for value in unit.get("supports_claim_ids") or []
                ],
                "source_ref": unit.get("source_ref") or {},
            }
            for unit in units
        ],
        oracle={
            "answer": answer_token,
            "answer_text": reference_answer,
            "minimum_sufficient_rui_set_size": len(rui_ids),
            "smallest_sufficient_rui_sets": [
                {
                    "rui_ids": rui_ids,
                    "answer": answer_token,
                    "answer_text": reference_answer,
                }
            ],
        },
        generation_metadata={
            "method": "finlongdocqa_free_response_allocation_v4",
            "answer_format": "free_response",
            "interaction_context_variant": (
                "shared_full_context"
                if allocation.mode in {"overlap", "common_anchor"}
                else "exclusive_full_context"
            ),
            "partition_strategy": allocation.mode,
            "private_information_allocation": allocation.to_metadata(),
            "noncanonical_allocation": (
                "least_loaded_packet_then_nearest_rui_page"
            ),
            "solver_audit": solver_audit,
            "packet_leakage_audits": [],
            "removed_leaking_context_unit_ids": [],
        },
        dataset_specific={
            "qa_evidence_mappings": qa_mappings,
            "source_qa_ids": [
                str(row.get("qa_id") or "") for row in qa_mappings
            ],
            "source_qa_ids_used": task.get("source_qa_ids_used") or [],
            "free_response_rubric": task.get("rubric") or {},
        },
    )
    if not (datapoint.get("format_validation") or {}).get("valid"):
        raise ValueError(
            f"invalid common datapoint: {datapoint.get('format_validation')}"
        )
    return datapoint


def repartition_common_datapoint(
    datapoint: dict[str, Any],
    *,
    allocation_mode: str,
    allocation_seed: int = 0,
    min_participants: int = 2,
) -> dict[str, Any]:
    """Reallocate a frozen audited task without changing its text or RIUs."""
    out = deepcopy(datapoint)
    rius = [
        row
        for row in out.get("reference_information_units") or []
        if isinstance(row, dict) and row.get("rui_id")
    ]
    rui_ids = [str(row["rui_id"]) for row in rius]
    if len(rui_ids) < 2:
        raise ValueError("partition requires at least two final RIUs")
    normalized_mode = normalize_allocation_mode(allocation_mode)
    allocation = assign_reference_units(
        rui_ids,
        mode=normalized_mode,
        rng=random.Random(
            f"{out.get('instance_id')}:{allocation_seed}:frozen-riu"
        ),
        min_participants=min_participants,
        smallest_sufficient_sets=[rui_ids],
        allow_common_anchor=True,
        allowed_modes=PAPER_ALLOCATION_MODES,
    )
    if not allocation.validation.get("valid"):
        raise ValueError(f"Invalid RIU allocation: {allocation.validation}")

    internal_to_public = {
        internal_id: _letter_id("participant", index)
        for index, internal_id in enumerate(
            sorted(allocation.private_assignment), start=1
        )
    }
    participants = [
        {
            "participant_id": internal_to_public[internal_id],
            "role_name": f"annual-report evidence reviewer {index}",
            "owned_rui_ids": list(
                allocation.private_assignment[internal_id]
            ),
            "private_context_unit_ids": [],
            "private_word_count": 0,
        }
        for index, internal_id in enumerate(
            sorted(allocation.private_assignment), start=1
        )
    ]
    participant_by_id = {
        str(row["participant_id"]): row for row in participants
    }
    all_participant_ids = sorted(participant_by_id)
    common_rui_ids = set(allocation.common_rui_ids)
    private_owners = owners_by_reference_unit(allocation.private_assignment)
    owners_by_rui = {
        rui_id: (
            all_participant_ids
            if rui_id in common_rui_ids
            else [
                internal_to_public[internal_id]
                for internal_id in private_owners.get(rui_id, [])
            ]
        )
        for rui_id in rui_ids
    }
    rui_by_id = {str(row["rui_id"]): row for row in rius}
    anchor_pages = {
        rui_id: int(
            (rui_by_id[rui_id].get("source_ref") or {}).get("page_number")
            or 0
        )
        for rui_id in rui_ids
    }
    fallback_anchor = sum(anchor_pages.values()) // max(1, len(anchor_pages))
    anchors: dict[str, int] = {}
    for participant_id, participant in participant_by_id.items():
        pages = [
            anchor_pages[rui_id]
            for rui_id in participant.get("owned_rui_ids") or []
            if rui_id in anchor_pages
        ]
        anchors[participant_id] = (
            sum(pages) // len(pages) if pages else fallback_anchor
        )

    contexts = [
        row
        for row in out.get("private_context_units") or []
        if isinstance(row, dict)
    ]
    canonical_owners: dict[str, list[str]] = {}
    for context in contexts:
        context_id = str(context.get("context_unit_id") or "")
        hidden = {
            str(value)
            for value in context.get("hidden_contains_rui_ids") or []
            if str(value) in owners_by_rui
        }
        if hidden:
            canonical_owners[context_id] = sorted(
                {
                    participant_id
                    for rui_id in hidden
                    for participant_id in owners_by_rui[rui_id]
                }
            )

    assigned_words = {
        participant_id: sum(
            int(context.get("word_count") or 0)
            for context in contexts
            if participant_id
            in canonical_owners.get(
                str(context.get("context_unit_id") or ""), []
            )
        )
        for participant_id in all_participant_ids
    }
    for context in contexts:
        context_id = str(context.get("context_unit_id") or "")
        word_count = int(context.get("word_count") or 0)
        if context_id in canonical_owners:
            owners = canonical_owners[context_id]
            reason = "canonical_rui_owner_or_common_recipient"
        else:
            page_number = int(
                (context.get("source_ref") or {}).get("page_number") or 0
            )
            owner = min(
                all_participant_ids,
                key=lambda participant_id: (
                    assigned_words[participant_id],
                    abs(page_number - anchors[participant_id]),
                    participant_id,
                ),
            )
            owners = [owner]
            reason = "least_loaded_packet_then_nearest_rui_page"
            assigned_words[owner] += word_count
        context["assigned_to"] = owners
        context["assignment"] = {"reason": reason}
        for owner in owners:
            participant_by_id[owner]["private_context_unit_ids"].append(
                context_id
            )
            participant_by_id[owner]["private_word_count"] += word_count

    out["participants"] = participants
    out["participant_directory"] = [
        {
            "participant_id": row["participant_id"],
            "role_name": row["role_name"],
        }
        for row in participants
    ]
    common_information = out.setdefault("initial_common_information", {})
    common_information["common_reference_units"] = [
        {
            "rui_id": rui_id,
            "text": str(rui_by_id[rui_id].get("text") or ""),
            "information_feature": str(
                rui_by_id[rui_id].get("information_feature") or ""
            ),
        }
        for rui_id in sorted(common_rui_ids)
    ]
    metadata = out.setdefault("generation_metadata", {})
    base_strategy = str(metadata.get("partition_strategy") or "")
    base_removed = list(metadata.get("removed_leaking_context_unit_ids") or [])
    metadata.update(
        {
            "method": "finlongdocqa_frozen_task_repartition_v1",
            "partition_strategy": allocation.mode,
            "private_information_allocation": allocation.to_metadata(),
            "interaction_context_variant": (
                "shared_full_context"
                if allocation.mode in {"overlap", "common_anchor"}
                else "exclusive_full_context"
            ),
            "frozen_task_repartition": {
                "base_partition_strategy": base_strategy,
                "base_removed_leaking_context_unit_ids": base_removed,
                "question_riu_oracle_unchanged": True,
            },
            "packet_leakage_audits": [],
            "removed_leaking_context_unit_ids": [],
        }
    )
    out["format_validation"] = validate_common_datapoint(out)
    if not out["format_validation"].get("valid"):
        raise ValueError(
            f"invalid repartitioned datapoint: {out['format_validation']}"
        )
    return out


class _NoShuffleRandom(random.Random):
    """Preserve legacy one-each ownership while using the shared allocator."""

    def shuffle(self, _x: list[Any]) -> None:
        return None


def _letter_id(prefix: str, index: int) -> str:
    letters = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("a") + remainder) + letters
    return f"{prefix}_{letters}"
