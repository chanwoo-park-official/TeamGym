"""Full-document MAUD partitioning around grounded PIU blocks."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from teamgym.core.information_allocation import DEFAULT_ALLOCATION_MODE_WEIGHTS, assign_reference_units
from teamgym.core.io import write_json, write_jsonl
from teamgym.core.schema import COMMON_DATAPOINT_SCHEMA_VERSION, build_common_datapoint

from ..contracts.generation_records import read_generation_trials
from ..contracts.assignment import (
    ASSIGNMENT_SCORE_WEIGHTS,
    assign_chunks as _assign_chunks,
    participant_doc_unit_counts as _participant_doc_unit_counts,
    participants_from_assignment as _participants_from_assignment,
)
from ..contracts.chunks import (
    chunk_context as _chunk_context,
    section_key_at as _section_key_at,
    split_conflicting_piu_chunks as _split_conflicting_rui_chunks,
    word_count as _word_count,
)
from ..contracts.partition_validation import (
    number_gap as _number_gap,
    number_summary as _number_summary,
    validate_full_document_partition as _validate_partition,
)
from ..contracts.records import (
    common_context_units as _common_context_units,
    common_participants as _common_participants,
    participant_id_map as _participant_id_map,
)
from ..decision_permutation import apply_permutation_to_public_common, permute_common_datapoint_decisions
from .evidence import (
    _aggressive_normalize_with_offsets,
    _agreement_chunks,
    _normalize_with_offsets,
    match_evidence_block,
)
from ..sources import _hf_download

PARTITION_SCHEMA = "maud_full_document_partition_v1"


@dataclass(frozen=True)
class MAUDRUIRange:
    unit_id: str
    source_rui_id: str
    issue_group_id: str
    clean_text: str
    raw_start: int
    raw_end: int
    raw_text: str
    legal_feature: str
    summary: str
    section_key: str
    match_status: str
    match_score: float | None


def partition_accepted_trials(
    *,
    accepted_path: str | Path,
    cache_dir: str | Path,
    output_root: str | Path,
    max_examples: int | None,
    seed: int,
    strategy: str,
    role_style: str,
    target_words: int,
    min_words: int,
    soft_max_words: int,
    hard_max_words: int,
    min_grounded_rius: int,
    min_participants: int = 3,
    context_mode: str = "full",
    context_budget_tokens: int | None = None,
    context_neighbor_chunks: int = 1,
) -> dict[str, Any]:
    trials = read_generation_trials(accepted_path)
    if max_examples is not None:
        trials = trials[:max_examples]
    rng = random.Random(seed)
    partitions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, trial in enumerate(trials, start=1):
        try:
            contract_text = _load_contract_text(trial, cache_dir)
            partition = build_full_document_partition(
                trial,
                contract_text,
                rng=rng,
                seed=seed,
                strategy=strategy,
                role_style=role_style,
                target_words=target_words,
                min_words=min_words,
                soft_max_words=soft_max_words,
                hard_max_words=hard_max_words,
                min_participants=min_participants,
                min_grounded_rius=min_grounded_rius,
                context_mode=context_mode,
                context_budget_tokens=context_budget_tokens,
                context_neighbor_chunks=context_neighbor_chunks,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "trial_index": idx,
                    "instance_id": trial.get("instance_id"),
                    "source_id": trial.get("source_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        validation = partition.get("validation") if isinstance(partition.get("validation"), dict) else {}
        if validation.get("valid") is not True:
            failures.append(
                {
                    "trial_index": idx,
                    "instance_id": trial.get("instance_id"),
                    "source_id": trial.get("source_id"),
                    "error": "partition validation failed",
                    "failed_checks": [
                        check.get("name")
                        for check in validation.get("checks") or []
                        if isinstance(check, dict) and not check.get("passed")
                    ],
                }
            )
            continue
        partitions.append(partition)

    output_root_path = Path(output_root).expanduser().resolve()
    write_jsonl(output_root_path / "maud_full_document_partitions.jsonl", partitions)
    write_json(output_root_path / "maud_full_document_partitions.pretty.json", partitions)
    report = {
        "schema": "maud_full_document_partition_report_v1",
        "accepted_path": str(Path(accepted_path).expanduser()),
        "output_root": str(output_root_path),
        "seed": seed,
        "strategy": strategy,
        "role_style": role_style,
        "min_participants": min_participants,
        "context_mode": context_mode,
        "context_budget_tokens": context_budget_tokens,
        "requested_examples": len(trials),
        "partitioned_count": len(partitions),
        "failure_count": len(failures),
        "failures": failures,
        "summary": _report_summary(partitions),
    }
    write_json(output_root_path / "partition_report.json", report)
    return report


def build_full_document_partition(
    trial: dict[str, Any],
    contract_text: str,
    *,
    rng: random.Random,
    seed: int,
    strategy: str,
    role_style: str,
    target_words: int,
    min_words: int,
    soft_max_words: int,
    hard_max_words: int,
    min_grounded_rius: int,
    min_participants: int = 3,
    context_mode: str = "full",
    context_budget_tokens: int | None = None,
    context_neighbor_chunks: int = 1,
) -> dict[str, Any]:
    rui_ranges, dropped = _matched_rui_ranges_for_trial(trial, contract_text)
    rui_ranges, overlap_drops = _drop_overlapping_rui_ranges(rui_ranges)
    dropped.extend(overlap_drops)
    if len(rui_ranges) < min_grounded_rius:
        raise ValueError(
            f"only {len(rui_ranges)} grounded RIUs remain after dropping unmatched units; "
            f"need at least {min_grounded_rius}"
        )
    smallest_sets = _smallest_sufficient_sets_after_dropping(trial, rui_ranges)
    allocation = assign_reference_units(
        [rui.unit_id for rui in rui_ranges],
        mode=strategy,
        rng=rng,
        min_participants=min_participants,
        smallest_sufficient_sets=smallest_sets,
    )
    rui_assignment = allocation.private_assignment
    common_rui_ids = set(allocation.common_rui_ids)
    participants = _participants_from_assignment(rui_assignment, rui_ranges, role_style=role_style)

    chunks = _chunk_context(
        contract_text,
        rui_ranges,
        target_words=target_words,
        min_words=min_words,
        soft_max_words=soft_max_words,
        hard_max_words=hard_max_words,
    )
    chunks = _split_conflicting_rui_chunks(contract_text, chunks, rui_ranges, rui_assignment, min_words=min_words)
    context_budgeting = _build_context_budgeting_metadata(
        original_chunks=chunks,
        selected_chunks=chunks,
        mode="full",
        budget_tokens=context_budget_tokens,
        neighbor_chunks=context_neighbor_chunks,
    )
    if context_mode == "budgeted":
        chunks, context_budgeting = _select_budgeted_chunks(
            chunks,
            rui_ranges,
            budget_tokens=context_budget_tokens,
            neighbor_chunks=context_neighbor_chunks,
            rng=_stable_rng(seed, str(trial.get("instance_id") or ""), "context_budget"),
        )
    elif context_mode != "full":
        raise ValueError("context_mode must be 'full' or 'budgeted'.")
    # Known MAUD edge case confirmed by the frozen-archive check. A shared
    # chunk can expose a PIU to participants beyond its declared owners.
    doc_units = _assign_chunks(
        chunks,
        participants,
        rui_assignment,
        rui_ranges,
        common_rui_ids=common_rui_ids,
        target_words=target_words,
        rng=rng,
    )
    validation = (
        _validate_partition(
            context=contract_text,
            rui_ranges=rui_ranges,
            smallest_sets=smallest_sets,
            participants=participants,
            document_units=doc_units,
            common_rui_ids=common_rui_ids,
        )
        if context_mode == "full"
        else _validate_budgeted_partition(
            context=contract_text,
            rui_ranges=rui_ranges,
            smallest_sets=smallest_sets,
            participants=participants,
            document_units=doc_units,
            common_rui_ids=common_rui_ids,
            context_budgeting=context_budgeting,
        )
    )
    common_datapoint = build_common_datapoint(
        instance_id=str(trial.get("instance_id") or ""),
        dataset="maud",
        source=_common_source(trial, contract_text, context_budgeting),
        scenario_family=f"maud_{_slug(str(trial.get('category') or 'merger_agreement_review'))}",
        task_description=_common_task_description(trial),
        decision_set=_common_decision_set(trial),
        decision_option_definitions=_common_decision_option_definitions(trial),
        initial_common_information=_common_initial_information(trial, rui_ranges, common_rui_ids),
        participants=_common_participants(participants),
        private_context_units=_common_context_units(doc_units, _participant_id_map(participants)),
        reference_information_units=_common_rui_units(rui_ranges),
        oracle=_common_oracle(trial, smallest_sets),
        generation_metadata=_common_generation_metadata(
            trial=trial,
            seed=seed,
            strategy=allocation.mode,
            role_style=role_style,
            target_words=target_words,
            min_words=min_words,
            soft_max_words=soft_max_words,
            hard_max_words=hard_max_words,
            validation=validation,
            allocation_metadata=allocation.to_metadata(),
            dropped_rui_units=dropped,
            context_budgeting=context_budgeting,
            min_participants=min_participants,
        ),
        dataset_specific={
            "maud_category": trial.get("category"),
            "candidate_index": trial.get("candidate_index"),
            "partition_strategy": allocation.mode,
            "private_information_allocation": allocation.to_metadata(),
            "dropped_rui_units": dropped,
            "context_budgeting": context_budgeting,
        },
    )
    common_datapoint, decision_permutation = permute_common_datapoint_decisions(
        common_datapoint,
        seed_material=f"maud:{trial.get('instance_id')}:{seed}:decision_options",
    )
    public_common = apply_permutation_to_public_common(
        {
            "contract_name": trial.get("contract_name"),
            "dataset": "MAUD",
            "document_kind": "merger_agreement",
            "task_objective": ((trial.get("plan") or {}).get("public_task") or {}).get("objective"),
            "decision_set": _common_decision_set(trial),
            "note": "Participants receive private agreement chunks; hidden metadata records which chunks contain RIUs.",
        },
        decision_permutation,
        source_common_datapoint=common_datapoint,
    )
    return {
        "schema": PARTITION_SCHEMA,
        "common_schema_version": COMMON_DATAPOINT_SCHEMA_VERSION,
        "source_trial_instance_id": trial.get("instance_id"),
        "instance_id": trial.get("instance_id"),
        "dataset": "maud",
        "source_id": trial.get("source_id"),
        "candidate_index": trial.get("candidate_index"),
        "category": trial.get("category"),
        "common_datapoint": common_datapoint,
        "seed": seed,
        "partition_strategy": allocation.mode,
        "role_style": role_style,
        "min_participants": min_participants,
        "context_mode": context_mode,
        "public_common": public_common,
        "source_document": {
            "contract_name": trial.get("contract_name"),
            "context_char_count": len(contract_text),
            "context_word_count": _word_count(contract_text),
            "selected_context_word_count": context_budgeting.get("selected_context_word_count"),
            "selected_context_token_count": context_budgeting.get("selected_context_token_count"),
        },
        "partition_scoring_policy": {
            "weights": ASSIGNMENT_SCORE_WEIGHTS,
            "role_overlap": "fraction of participant role terms that appear in the non-RUI chunk",
            "feature_overlap": "fraction of owned-RUI feature terms that appear in the non-RUI chunk",
            "proximity": "1 / (1 + word distance from the chunk midpoint to the nearest owned RIU midpoint divided by target chunk words)",
            "section_match": "1 when the chunk has the same detected section key as an owned RIU, otherwise 0",
            "load_penalty": "participant private word count divided by twice the target chunk words",
            "neutral_chunk_assignment": (
                "RIU chunks are assigned to their RIU owners first. Non-RIU chunks are classified as "
                "RIU-related or background; RIU-related chunks avoid participants who own the related RIUs "
                "when possible, then use balance-first seeded softmax sampling. Background chunks use "
                "stronger semantic fit with balance penalties."
            ),
            "allocation_mode_weights": DEFAULT_ALLOCATION_MODE_WEIGHTS,
            "private_information_allocation": allocation.to_metadata(),
            "context_budgeting": context_budgeting,
        },
        "hidden_rui_units": [_hidden_rui_unit(rui) for rui in rui_ranges],
        "hidden_dropped_rui_units": dropped,
        "hidden_dropped_unmatched_rius": [row for row in dropped if row.get("reason") == "no_full_agreement_offsets"],
        "hidden_smallest_sufficient_sets": smallest_sets,
        "hidden_common_rui_ids": sorted(common_rui_ids),
        "participants": participants,
        "document_units": doc_units,
        "validation": validation,
    }


def _matched_rui_ranges_for_trial(
    trial: dict[str, Any],
    contract_text: str,
) -> tuple[list[MAUDRUIRange], list[dict[str, Any]]]:
    plan_units = (trial.get("plan") or {}).get("evidence_units") or []
    if not isinstance(plan_units, list) or not plan_units:
        raise ValueError("trial has no plan.evidence_units")
    normalized_contract = _normalize_with_offsets(contract_text)
    aggressive_contract = _aggressive_normalize_with_offsets(contract_text)
    chunks = _agreement_chunks(contract_text)
    out: list[MAUDRUIRange] = []
    dropped: list[dict[str, Any]] = []
    for unit in plan_units:
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("unit_id") or "")
        text = str(unit.get("text") or "")
        source_ref = unit.get("source_ref") if isinstance(unit.get("source_ref"), dict) else {}
        grounding = match_evidence_block(text, contract_text, normalized_contract, chunks, aggressive_contract)
        start = grounding.get("agreement_start")
        end = grounding.get("agreement_end")
        if not unit_id or not isinstance(start, int) or not isinstance(end, int):
            dropped.append(
                {
                    "unit_id": unit_id,
                    "source_rui_id": source_ref.get("source_rui_id"),
                    "issue_group_id": source_ref.get("issue_group_id"),
                    "match_status": grounding.get("match_status"),
                    "match_score": grounding.get("match_score"),
                    "reason": "no_full_agreement_offsets",
                    "text": text,
                }
            )
            continue
        out.append(
            MAUDRUIRange(
                unit_id=unit_id,
                source_rui_id=str(source_ref.get("source_rui_id") or ""),
                issue_group_id=str(source_ref.get("issue_group_id") or ""),
                clean_text=text,
                raw_start=start,
                raw_end=end,
                raw_text=contract_text[start:end],
                legal_feature=str(unit.get("information_feature") or source_ref.get("information_feature") or ""),
                summary=str(unit.get("summary") or ""),
                section_key=_section_key_at(contract_text, start),
                match_status=str(grounding.get("match_status") or ""),
                match_score=grounding.get("match_score"),
            )
        )
    return sorted(out, key=lambda rui: (rui.raw_start, rui.raw_end, rui.unit_id)), dropped


def _drop_overlapping_rui_ranges(
    rui_ranges: list[MAUDRUIRange],
) -> tuple[list[MAUDRUIRange], list[dict[str, Any]]]:
    drop_ids: set[str] = set()
    dropped: list[dict[str, Any]] = []
    ordered = sorted(rui_ranges, key=lambda rui: (rui.raw_start, rui.raw_end, rui.unit_id))
    for idx, left in enumerate(ordered):
        for right in ordered[idx + 1 :]:
            if right.raw_start >= left.raw_end:
                break
            if max(left.raw_start, right.raw_start) >= min(left.raw_end, right.raw_end):
                continue
            to_drop, kept = _overlap_drop_choice(left, right)
            drop_ids.add(to_drop.unit_id)
            dropped.append(
                {
                    "unit_id": to_drop.unit_id,
                    "source_rui_id": to_drop.source_rui_id,
                    "issue_group_id": to_drop.issue_group_id,
                    "match_status": to_drop.match_status,
                    "match_score": to_drop.match_score,
                    "raw_start": to_drop.raw_start,
                    "raw_end": to_drop.raw_end,
                    "reason": "overlaps_grounded_rui",
                    "overlaps_with_unit_id": kept.unit_id,
                    "text": to_drop.clean_text,
                }
            )
    if not drop_ids:
        return rui_ranges, []
    return [rui for rui in rui_ranges if rui.unit_id not in drop_ids], dropped


def _select_budgeted_chunks(
    chunks: list[Any],
    rui_ranges: list[MAUDRUIRange],
    *,
    budget_tokens: int | None,
    neighbor_chunks: int,
    rng: random.Random,
) -> tuple[list[Any], dict[str, Any]]:
    if not chunks:
        return chunks, _build_context_budgeting_metadata(
            original_chunks=[],
            selected_chunks=[],
            mode="budgeted",
            budget_tokens=budget_tokens,
            neighbor_chunks=neighbor_chunks,
        )
    if not budget_tokens or budget_tokens <= 0:
        return chunks, _build_context_budgeting_metadata(
            original_chunks=chunks,
            selected_chunks=chunks,
            mode="budgeted_unlimited",
            budget_tokens=budget_tokens,
            neighbor_chunks=neighbor_chunks,
        )

    rui_ids = {rui.unit_id for rui in rui_ranges}
    mandatory = {
        idx
        for idx, chunk in enumerate(chunks)
        if rui_ids & set(chunk.contains_rui_ids)
    }
    selected = set(mandatory)
    reasons: dict[int, str] = {idx: "mandatory_rui_chunk" for idx in mandatory}
    for idx in sorted(mandatory):
        for offset in range(1, max(0, neighbor_chunks) + 1):
            for neighbor in (idx - offset, idx + offset):
                if 0 <= neighbor < len(chunks) and neighbor not in selected:
                    selected.add(neighbor)
                    reasons[neighbor] = f"neighbor_of_rui_chunk_{offset}"

    mandatory_tokens = _chunk_token_sum([chunks[idx] for idx in mandatory])
    selected_tokens = _chunk_token_sum([chunks[idx] for idx in selected])
    if selected_tokens > budget_tokens and mandatory_tokens <= budget_tokens:
        optional = sorted(selected - mandatory, key=lambda idx: abs(_nearest_index_distance(idx, mandatory)), reverse=True)
        for idx in optional:
            if selected_tokens <= budget_tokens:
                break
            selected.remove(idx)
            selected_tokens -= _token_count(chunks[idx].text)
            reasons.pop(idx, None)

    selected_sections = {chunks[idx].section_key for idx in selected if chunks[idx].section_key}
    rui_sections = {rui.section_key for rui in rui_ranges if rui.section_key}
    candidates = [idx for idx in range(len(chunks)) if idx not in selected]
    scored = []
    for idx in candidates:
        chunk = chunks[idx]
        token_count = _token_count(chunk.text)
        if selected_tokens + token_count > budget_tokens:
            continue
        distance = _nearest_index_distance(idx, mandatory)
        proximity = 1.0 / (1.0 + distance)
        same_section = 1.0 if chunk.section_key and chunk.section_key in rui_sections else 0.0
        diversity = 1.0 if chunk.section_key and chunk.section_key not in selected_sections else 0.0
        preamble = 1.0 if idx == 0 else 0.0
        score = 3.0 * proximity + 2.0 * same_section + 0.6 * diversity + 0.4 * preamble + rng.random() * 0.01
        scored.append((score, idx, token_count))
    for _, idx, token_count in sorted(scored, reverse=True):
        if selected_tokens + token_count > budget_tokens:
            continue
        selected.add(idx)
        selected_tokens += token_count
        chunk = chunks[idx]
        if idx == 0:
            reason = "document_preamble_distractor"
        elif chunk.section_key and chunk.section_key in rui_sections:
            reason = "same_section_distractor"
        elif chunk.section_key and chunk.section_key not in selected_sections:
            reason = "section_diverse_distractor"
        else:
            reason = "nearby_distractor"
        reasons[idx] = reason
        if chunk.section_key:
            selected_sections.add(chunk.section_key)

    selected_chunks = [chunks[idx] for idx in sorted(selected)]
    metadata = _build_context_budgeting_metadata(
        original_chunks=chunks,
        selected_chunks=selected_chunks,
        mode="budgeted",
        budget_tokens=budget_tokens,
        neighbor_chunks=neighbor_chunks,
    )
    metadata["mandatory_rui_chunk_count"] = len(mandatory)
    metadata["mandatory_rui_chunk_token_count"] = mandatory_tokens
    metadata["budget_exceeded_by_mandatory_chunks"] = mandatory_tokens > budget_tokens
    metadata["selected_chunk_reasons"] = {
        f"chunk_{idx + 1:04d}": reasons[idx]
        for idx in sorted(selected)
        if idx in reasons
    }
    return selected_chunks, metadata


def _build_context_budgeting_metadata(
    *,
    original_chunks: list[Any],
    selected_chunks: list[Any],
    mode: str,
    budget_tokens: int | None,
    neighbor_chunks: int,
) -> dict[str, Any]:
    original_tokens = _chunk_token_sum(original_chunks)
    selected_tokens = _chunk_token_sum(selected_chunks)
    original_words = sum(int(chunk.word_count or 0) for chunk in original_chunks)
    selected_words = sum(int(chunk.word_count or 0) for chunk in selected_chunks)
    return {
        "mode": mode,
        "budget_tokens": budget_tokens,
        "neighbor_chunks": neighbor_chunks,
        "original_chunk_count": len(original_chunks),
        "selected_chunk_count": len(selected_chunks),
        "dropped_chunk_count": max(0, len(original_chunks) - len(selected_chunks)),
        "original_context_token_count": original_tokens,
        "selected_context_token_count": selected_tokens,
        "original_context_word_count": original_words,
        "selected_context_word_count": selected_words,
        "compression_ratio_tokens": round(selected_tokens / original_tokens, 4) if original_tokens else None,
    }


def _chunk_token_sum(chunks: Iterable[Any]) -> int:
    return sum(_token_count(str(chunk.text or "")) for chunk in chunks)


def _nearest_index_distance(idx: int, anchors: set[int]) -> int:
    if not anchors:
        return 10**6
    return min(abs(idx - anchor) for anchor in anchors)


def _token_count(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(str(text or "")))
    except Exception:  # noqa: BLE001
        return max(1, round(_word_count(str(text or "")) * 1.33))


def _stable_rng(seed: int, instance_id: str, purpose: str) -> random.Random:
    payload = f"{seed}:{instance_id}:{purpose}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return random.Random(int(digest[:16], 16))


def _validate_budgeted_partition(
    *,
    context: str,
    rui_ranges: list[MAUDRUIRange],
    smallest_sets: list[list[str]],
    participants: list[dict[str, Any]],
    document_units: list[dict[str, Any]],
    common_rui_ids: set[str],
    context_budgeting: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    sorted_units = sorted(document_units, key=lambda unit: int(unit["start"]))
    assigned_rui_by_participant = _participant_rui_sets_for_doc_units(participants, sorted_units)
    participant_doc_unit_counts = _participant_doc_unit_counts(participants)
    all_assigned_rui = set().union(*assigned_rui_by_participant.values()) if assigned_rui_by_participant else set()
    required = {rui.unit_id for rui in rui_ranges}
    budget = context_budgeting.get("budget_tokens")
    selected_tokens = int(context_budgeting.get("selected_context_token_count") or 0)
    mandatory_tokens = int(context_budgeting.get("mandatory_rui_chunk_token_count") or 0)
    _check_local(checks, "document_units_present", bool(sorted_units))
    _check_local(checks, "selected_units_in_source_bounds", all(0 <= int(unit["start"]) <= int(unit["end"]) <= len(context) for unit in sorted_units))
    _check_local(checks, "selected_units_non_overlapping_ordered", _non_overlapping_units(sorted_units))
    _check_local(checks, "all_rui_covered_by_private_or_common_chunks", required <= (all_assigned_rui | common_rui_ids))
    _check_local(
        checks,
        "no_single_participant_has_smallest_sufficient_set",
        not any(
            set(smallest) <= (participant_set | common_rui_ids)
            for participant_set in assigned_rui_by_participant.values()
            for smallest in smallest_sets
        ),
    )
    _check_local(checks, "all_doc_units_assigned", all(unit.get("assigned_to") for unit in sorted_units))
    _check_local(
        checks,
        "every_participant_has_private_context",
        bool(participant_doc_unit_counts) and all(count > 0 for count in participant_doc_unit_counts.values()),
    )
    _check_local(
        checks,
        "context_budget_respected_or_mandatory_overflow",
        not budget or selected_tokens <= int(budget) or mandatory_tokens > int(budget),
    )
    _check_local(checks, "mandatory_rui_chunks_selected", required <= _covered_rui_ids(sorted_units))
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
            "participant_doc_unit_count_summary": _number_summary(list(participant_doc_unit_counts.values())),
            "participant_doc_unit_count_gap": _number_gap(list(participant_doc_unit_counts.values())),
            "participant_doc_word_counts": {
                participant["participant_id"]: participant.get("private_word_count", 0)
                for participant in participants
            },
            "chunk_word_count_summary": _number_summary([int(unit["word_count"]) for unit in document_units]),
            "context_budgeting": context_budgeting,
        },
    }


def _participant_rui_sets_for_doc_units(
    participants: list[dict[str, Any]],
    document_units: list[dict[str, Any]],
) -> dict[str, set[str]]:
    out = {participant["participant_id"]: set() for participant in participants}
    for unit in document_units:
        rui_ids = {str(item) for item in unit.get("hidden_contains_rui_ids", [])}
        for participant_id in unit.get("assigned_to", []):
            out.setdefault(str(participant_id), set()).update(rui_ids)
    return out


def _covered_rui_ids(document_units: list[dict[str, Any]]) -> set[str]:
    return {
        str(rui_id)
        for unit in document_units
        for rui_id in unit.get("hidden_contains_rui_ids", [])
    }


def _non_overlapping_units(sorted_units: list[dict[str, Any]]) -> bool:
    return all(int(left["end"]) <= int(right["start"]) for left, right in zip(sorted_units, sorted_units[1:]))


def _check_local(checks: list[dict[str, Any]], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})


def _overlap_drop_choice(left: MAUDRUIRange, right: MAUDRUIRange) -> tuple[MAUDRUIRange, MAUDRUIRange]:
    left_len = left.raw_end - left.raw_start
    right_len = right.raw_end - right.raw_start
    if left_len != right_len:
        return (left, right) if left_len > right_len else (right, left)
    left_score = float(left.match_score or 0.0)
    right_score = float(right.match_score or 0.0)
    if left_score != right_score:
        return (left, right) if left_score < right_score else (right, left)
    return (right, left)


def _smallest_sufficient_sets_after_dropping(
    trial: dict[str, Any],
    rui_ranges: list[MAUDRUIRange],
) -> list[list[str]]:
    known = {rui.unit_id for rui in rui_ranges}
    out: list[list[str]] = []
    audit_sets = (trial.get("audit") or {}).get("smallest_sufficient_rui_sets") or []
    if isinstance(audit_sets, list):
        for row in audit_sets:
            if not isinstance(row, dict):
                continue
            # Known MAUD edge case reproduced in the generation smoke test.
            # Trimming can leave only part of an audited sufficient set.
            # Checking the remaining IDs can conservatively reject the partition.
            ids = [str(item) for item in row.get("unit_ids", []) if str(item) in known]
            if ids:
                out.append(ids)
    return out or [[rui.unit_id for rui in rui_ranges]]


def _load_contract_text(trial: dict[str, Any], cache_dir: str | Path) -> str:
    contract_name = str(trial.get("contract_name") or "")
    if not contract_name:
        raise ValueError("trial is missing contract_name")
    path = _hf_download("theatticusproject/maud", f"MAUD_v1/contracts/{contract_name}.txt", cache_dir)
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _common_source(trial: dict[str, Any], context: str, context_budgeting: dict[str, Any]) -> dict[str, Any]:
    context_mode = str(context_budgeting.get("mode") or "full")
    return {
        "source_dataset": "MAUD",
        "source_id": trial.get("source_id"),
        "document_kind": "merger_agreement",
        "contract_name": trial.get("contract_name"),
        "category": trial.get("category"),
        "context_char_count": len(context),
        "context_word_count": _word_count(context),
        "selected_context_word_count": context_budgeting.get("selected_context_word_count"),
        "selected_context_token_count": context_budgeting.get("selected_context_token_count"),
        "source_preservation": (
            "budgeted_agreement_context_partitioned_into_private_chunks"
            if context_mode.startswith("budgeted")
            else "full_agreement_context_partitioned_into_private_chunks"
        ),
    }


def _common_task_description(trial: dict[str, Any]) -> dict[str, Any]:
    public_task = (trial.get("plan") or {}).get("public_task") or {}
    return {
        "title": f"MAUD {trial.get('category') or 'agreement'} review",
        "objective": public_task.get("objective"),
        "contract_context": public_task.get("contract_context"),
        "answer_format": {
            "type": "single_choice",
            "choices": _common_decision_set(trial),
        },
    }


def _common_decision_set(trial: dict[str, Any]) -> list[str]:
    public_task = (trial.get("plan") or {}).get("public_task") or {}
    return [str(item) for item in public_task.get("decision_set") or []]


def _common_decision_option_definitions(trial: dict[str, Any]) -> dict[str, Any]:
    public_task = (trial.get("plan") or {}).get("public_task") or {}
    definitions = public_task.get("decision_option_definitions")
    return definitions if isinstance(definitions, dict) else {}


def _common_initial_information(
    trial: dict[str, Any],
    rui_ranges: list[MAUDRUIRange],
    common_rui_ids: set[str],
) -> dict[str, Any]:
    common = {
        "source_context": {
            "contract_name": trial.get("contract_name"),
            "document_kind": "merger_agreement",
            "dataset": "MAUD",
            "category": trial.get("category"),
        }
    }
    if common_rui_ids:
        rui_by_id = {rui.unit_id: rui for rui in rui_ranges}
        common["common_reference_units"] = [
            {
                "rui_id": rui_id,
                "text": rui_by_id[rui_id].clean_text,
                "summary": rui_by_id[rui_id].summary,
            }
            for rui_id in sorted(common_rui_ids)
            if rui_id in rui_by_id
        ]
    return common


def _common_rui_units(rui_ranges: list[MAUDRUIRange]) -> list[dict[str, Any]]:
    return [
        {
            "rui_id": rui.unit_id,
            "text": rui.clean_text,
            "raw_text": rui.raw_text,
            "information_feature": rui.legal_feature,
            "summary": rui.summary,
            "source_ref": {
                "source": "maud_row_evidence_block",
                "source_rui_id": rui.source_rui_id,
                "issue_group_id": rui.issue_group_id,
                "raw_start": rui.raw_start,
                "raw_end": rui.raw_end,
                "section_key": rui.section_key,
                "match_status": rui.match_status,
                "match_score": rui.match_score,
            },
        }
        for rui in rui_ranges
    ]


def _hidden_rui_unit(rui: MAUDRUIRange) -> dict[str, Any]:
    return {
        "unit_id": rui.unit_id,
        "source_rui_id": rui.source_rui_id,
        "issue_group_id": rui.issue_group_id,
        "raw_start": rui.raw_start,
        "raw_end": rui.raw_end,
        "raw_text": rui.raw_text,
        "clean_text": rui.clean_text,
        "legal_feature": rui.legal_feature,
        "summary": rui.summary,
        "section_key": rui.section_key,
        "match_status": rui.match_status,
        "match_score": rui.match_score,
    }


def _common_oracle(trial: dict[str, Any], smallest_sets: list[list[str]]) -> dict[str, Any]:
    plan = trial.get("plan") or {}
    oracle = plan.get("oracle") if isinstance(plan.get("oracle"), dict) else {}
    audit = trial.get("audit") if isinstance(trial.get("audit"), dict) else {}
    return {
        "answer": oracle.get("answer"),
        "rationale": oracle.get("rationale"),
        "features": oracle.get("features") or {},
        "minimum_sufficient_rui_set_size": min((len(row) for row in smallest_sets), default=None),
        "pre_drop_minimum_sufficient_rui_set_size": audit.get("minimum_sufficient_rui_set_size"),
        "smallest_sufficient_rui_sets": [
            {
                "rui_ids": list(row),
                "answer": oracle.get("answer"),
            }
            for row in smallest_sets
        ],
        "necessity_audit_ref": trial.get("generation_byproduct_ref"),
    }


def _common_generation_metadata(
    *,
    trial: dict[str, Any],
    seed: int,
    strategy: str,
    role_style: str,
    target_words: int,
    min_words: int,
    soft_max_words: int,
    hard_max_words: int,
    validation: dict[str, Any],
    allocation_metadata: dict[str, Any],
    dropped_rui_units: list[dict[str, Any]],
    context_budgeting: dict[str, Any],
    min_participants: int,
) -> dict[str, Any]:
    inherited = trial.get("generation_metadata") if isinstance(trial.get("generation_metadata"), dict) else {}
    return {
        "method": "maud_measured_rui_generation_with_full_document_partitioning",
        "no_synthetic_fallback": True,
        "source_candidate": {
            "source_id": trial.get("source_id"),
            "candidate_index": trial.get("candidate_index"),
            "category": trial.get("category"),
        },
        "task_generation": {
            "reference_information_unit_policy": "cleaned_maud_row_evidence_blocks_with_full_agreement_offsets_only",
            "min_accept_size": trial.get("min_accept_size"),
            "planning_attempts": trial.get("planning_attempts"),
            "prompt_stage_count": inherited.get("prompt_stage_count") or 3,
            "generation_byproduct_ref": trial.get("generation_byproduct_ref"),
            "post_audit_trimming": trial.get("post_audit_trimming"),
        },
        "partitioning": {
            "seed": seed,
            "strategy": strategy,
            "role_style": role_style,
            "target_words": target_words,
            "min_words": min_words,
            "soft_max_words": soft_max_words,
            "hard_max_words": hard_max_words,
            "min_participants": min_participants,
            "context_budgeting": context_budgeting,
            "score_weights": ASSIGNMENT_SCORE_WEIGHTS,
            "allocation_mode_weights": DEFAULT_ALLOCATION_MODE_WEIGHTS,
            "private_information_allocation": allocation_metadata,
            "dropped_rui_units": dropped_rui_units,
            "score_components": [
                "role_overlap",
                "feature_overlap",
                "proximity",
                "section_match",
                "load_penalty",
                "related_rui_safety_filter",
                "word_balance_penalty",
                "chunk_balance_penalty",
                "seeded_softmax_sampling",
            ],
        },
        "validation_summary": {
            "valid": validation.get("valid"),
            "check_count": len(validation.get("checks") or []),
        },
    }


def _report_summary(partitions: list[dict[str, Any]]) -> dict[str, Any]:
    strategy_counts = Counter(partition.get("partition_strategy") for partition in partitions)
    context_mode_counts = Counter(partition.get("context_mode") for partition in partitions)
    valid_count = sum(1 for partition in partitions if (partition.get("validation") or {}).get("valid") is True)
    unit_counts = [len(partition.get("document_units") or []) for partition in partitions]
    participant_counts = [len(partition.get("participants") or []) for partition in partitions]
    rui_counts = [len(partition.get("hidden_rui_units") or []) for partition in partitions]
    dropped_counts = [len(partition.get("hidden_dropped_rui_units") or []) for partition in partitions]
    dropped_unmatched_counts = [len(partition.get("hidden_dropped_unmatched_rius") or []) for partition in partitions]
    chunk_words = [
        int(unit.get("word_count") or 0)
        for partition in partitions
        for unit in partition.get("document_units") or []
    ]
    participant_chunk_gaps = [
        int(((partition.get("validation") or {}).get("metrics") or {}).get("participant_doc_unit_count_gap") or 0)
        for partition in partitions
    ]
    match_statuses = Counter(
        rui.get("match_status")
        for partition in partitions
        for rui in partition.get("hidden_rui_units") or []
    )
    context_budgeting_rows = [
        ((partition.get("common_datapoint") or {}).get("dataset_specific") or {}).get("context_budgeting") or {}
        for partition in partitions
    ]
    return {
        "valid_count": valid_count,
        "strategy_counts": dict(strategy_counts),
        "context_mode_counts": dict(context_mode_counts),
        "document_unit_count_summary": _number_summary(unit_counts),
        "participant_count_summary": _number_summary(participant_counts),
        "rui_count_summary": _number_summary(rui_counts),
        "dropped_rui_count_summary": _number_summary(dropped_counts),
        "dropped_unmatched_rui_count_summary": _number_summary(dropped_unmatched_counts),
        "chunk_word_count_summary": _number_summary(chunk_words),
        "participant_doc_unit_count_gap_summary": _number_summary(participant_chunk_gaps),
        "rui_match_status_counts": dict(match_statuses),
        "selected_context_token_count_summary": _number_summary(
            [int(row.get("selected_context_token_count") or 0) for row in context_budgeting_rows]
        ),
        "selected_context_word_count_summary": _number_summary(
            [int(row.get("selected_context_word_count") or 0) for row in context_budgeting_rows]
        ),
        "compression_ratio_token_summary": _number_summary(
            [
                int(round(float(row.get("compression_ratio_tokens") or 0.0) * 10000))
                for row in context_budgeting_rows
                if row.get("compression_ratio_tokens") is not None
            ]
        ),
    }


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _parse_args(argv: list[str] | None = None, *, prog: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--accepted-path", required=True)
    parser.add_argument("--cache-dir", default="data/legal_raw")
    parser.add_argument("--output-root", default="generated/maud_full_document_partitions")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--strategy",
        choices=(
            "auto",
            "one_each",
            "one_span_each",
            "one-span-each",
            "paired",
            "overlap",
            "redundant_player",
            "redundant-player",
            "common_anchor",
            "common-anchor",
            "bottleneck",
        ),
        default="auto",
    )
    parser.add_argument("--role-style", choices=("feature", "generic"), default="feature")
    parser.add_argument("--target-words", type=int, default=500)
    parser.add_argument("--min-words", type=int, default=120)
    parser.add_argument("--soft-max-words", type=int, default=900)
    parser.add_argument("--hard-max-words", type=int, default=1200)
    parser.add_argument("--min-participants", type=int, default=3)
    parser.add_argument("--min-grounded-rius", type=int, default=2)
    parser.add_argument("--context-mode", choices=("full", "budgeted"), default="full")
    parser.add_argument(
        "--context-budget-tokens",
        type=int,
        default=16000,
        help="Approximate total selected context-token budget for --context-mode budgeted. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--context-neighbor-chunks",
        type=int,
        default=1,
        help="Number of chunks before/after each PIU chunk to force-include in budgeted mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    args = _parse_args(argv, prog=prog)
    report = partition_accepted_trials(
        accepted_path=args.accepted_path,
        cache_dir=args.cache_dir,
        output_root=args.output_root,
        max_examples=args.max_examples,
        seed=args.seed,
        strategy=args.strategy,
        role_style=args.role_style,
        target_words=args.target_words,
        min_words=args.min_words,
        soft_max_words=args.soft_max_words,
        hard_max_words=args.hard_max_words,
        min_participants=args.min_participants,
        min_grounded_rius=args.min_grounded_rius,
        context_mode=args.context_mode,
        context_budget_tokens=args.context_budget_tokens,
        context_neighbor_chunks=args.context_neighbor_chunks,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
