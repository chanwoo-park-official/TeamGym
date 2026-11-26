"""Full-document CUAD partitioning around hidden PIU spans."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import random
import re
from pathlib import Path
from typing import Any

from teamgym.core.information_allocation import (
    DEFAULT_ALLOCATION_MODE_WEIGHTS,
    assign_reference_units,
)
from teamgym.core.io import write_json, write_jsonl
from teamgym.core.schema import COMMON_DATAPOINT_SCHEMA_VERSION, build_common_datapoint

from ..contracts.generation_records import read_generation_trials
from ..contracts.assignment import (
    ASSIGNMENT_SCORE_WEIGHTS,
    STOPWORDS as STOPWORDS,
    _balanced_participant_for_neutral_chunk as _balanced_participant_for_neutral_chunk,
    _best_participant_for_chunk as _best_participant_for_chunk,
    _char_distance_between_ranges as _char_distance_between_ranges,
    _dedupe as _dedupe,
    _document_unit_record as _document_unit_record,
    _fraction_overlap as _fraction_overlap,
    _keyword_set as _keyword_set,
    _owned_related_rui_ids_by_participant as _owned_related_rui_ids_by_participant,
    _proximity_to_owned_rui as _proximity_to_owned_rui,
    _record_participant_assignment as _record_participant_assignment,
    _related_rui_ids_for_chunk as _related_rui_ids_for_chunk,
    _role_from_features as _role_from_features,
    _safe_candidate_ids_for_related_ruis as _safe_candidate_ids_for_related_ruis,
    _sample_participant_id as _sample_participant_id,
    _section_match as _section_match,
    _tokens,
    assign_chunks as _assign_chunks,
    participant_doc_unit_counts as _participant_doc_unit_counts,
    participants_from_assignment as _participants_from_assignment,
)
from ..contracts.chunks import (
    SECTION_HEADING_PATTERN as SECTION_HEADING_PATTERN,
    Chunk as Chunk,
    _add_boundary as _add_boundary,
    _best_internal_split as _best_internal_split,
    _candidate_boundaries as _candidate_boundaries,
    _choose_boundary as _choose_boundary,
    _make_chunk as _make_chunk,
    _merge_tiny_chunks as _merge_tiny_chunks,
    _safe_boundary_at_or_after as _safe_boundary_at_or_after,
    _splits_rui as _splits_rui,
    _word_end_at_count as _word_end_at_count,
    _word_index_at_or_after as _word_index_at_or_after,
    chunk_context as _chunk_context,
    section_key_at as _section_key_at,
    split_conflicting_piu_chunks as _split_conflicting_rui_chunks,
    word_count as _word_count,
)
from ..contracts.partition_validation import (
    _check as _check,
    _covers_full_document as _covers_full_document,
    _no_rui_split as _no_rui_split,
    _non_overlapping as _non_overlapping,
    _participant_rui_sets as _participant_rui_sets,
    number_gap as _number_gap,
    number_summary as _number_summary,
    validate_full_document_partition as _validate_partition,
)
from ..contracts.records import (
    common_context_units as _common_context_units,
    common_participants as _common_participants,
    letter_id as _letter_id,
    participant_id_map as _participant_id_map,
)
from ..decision_permutation import apply_permutation_to_public_common, permute_common_datapoint_decisions
from ..sources import cuad_candidates

PARTITION_SCHEMA = "cuad_full_document_partition_v1"


@dataclass(frozen=True)
class RUIRange:
    unit_id: str
    source_span_index: int
    clean_text: str
    raw_start: int
    raw_end: int
    raw_text: str
    legal_feature: str
    summary: str
    section_key: str


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
    min_participants: int = 3,
) -> dict[str, Any]:
    trials = read_generation_trials(accepted_path)
    if max_examples is not None:
        trials = trials[:max_examples]
    candidate_by_source = {candidate["source_id"]: candidate for candidate in cuad_candidates(cache_dir)}
    rng = random.Random(seed)
    partitions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, trial in enumerate(trials, start=1):
        candidate = candidate_by_source.get(str(trial.get("source_id") or ""))
        if candidate is None:
            failures.append(
                {
                    "trial_index": idx,
                    "instance_id": trial.get("instance_id"),
                    "error": "source_id not found in CUAD candidates",
                }
            )
            continue
        try:
            partition = build_full_document_partition(
                trial,
                candidate,
                rng=rng,
                seed=seed,
                strategy=strategy,
                role_style=role_style,
                target_words=target_words,
                min_words=min_words,
                soft_max_words=soft_max_words,
                hard_max_words=hard_max_words,
                min_participants=min_participants,
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
    write_jsonl(output_root_path / "cuad_full_document_partitions.jsonl", partitions)
    write_json(output_root_path / "cuad_full_document_partitions.pretty.json", partitions)
    report = {
        "schema": "cuad_full_document_partition_report_v1",
        "accepted_path": str(Path(accepted_path).expanduser()),
        "output_root": str(output_root_path),
        "seed": seed,
        "strategy": strategy,
        "role_style": role_style,
        "min_participants": min_participants,
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
    candidate: dict[str, Any],
    *,
    rng: random.Random,
    seed: int,
    strategy: str,
    role_style: str,
    target_words: int,
    min_words: int,
    soft_max_words: int,
    hard_max_words: int,
    min_participants: int = 3,
) -> dict[str, Any]:
    context = str(candidate["contract_text"])
    candidate_rui_ranges = _rui_ranges_for_trial(trial, candidate)
    rui_ranges, smallest_sets = _retain_one_smallest_sufficient_set(
        trial,
        candidate_rui_ranges,
    )
    allocation = assign_reference_units(
        [rui.unit_id for rui in rui_ranges],
        mode=strategy,
        rng=rng,
        min_participants=min_participants,
        smallest_sufficient_sets=smallest_sets,
    )
    selected_strategy = allocation.mode
    rui_assignment = allocation.private_assignment
    common_rui_ids = set(allocation.common_rui_ids)
    participants = _participants_from_assignment(
        rui_assignment,
        rui_ranges,
        role_style=role_style,
    )

    chunks = _chunk_context(
        context,
        rui_ranges,
        target_words=target_words,
        min_words=min_words,
        soft_max_words=soft_max_words,
        hard_max_words=hard_max_words,
    )
    chunks = _split_conflicting_rui_chunks(context, chunks, rui_ranges, rui_assignment, min_words=min_words)
    doc_units = _assign_chunks(
        chunks,
        participants,
        rui_assignment,
        rui_ranges,
        common_rui_ids=common_rui_ids,
        target_words=target_words,
        rng=rng,
    )
    validation = _validate_partition(
        context=context,
        rui_ranges=rui_ranges,
        smallest_sets=smallest_sets,
        participants=participants,
        document_units=doc_units,
        common_rui_ids=common_rui_ids,
    )
    common_datapoint = build_common_datapoint(
        instance_id=str(trial.get("instance_id") or ""),
        dataset="cuad",
        source=_common_source(candidate, context),
        scenario_family=str(trial.get("label_family") or "cuad_contract_review"),
        task_description=_common_task_description(trial),
        decision_set=_common_decision_set(trial),
        decision_option_definitions=_common_decision_option_definitions(trial),
        initial_common_information=_common_initial_common_information(candidate, rui_ranges, common_rui_ids),
        participants=_common_participants(participants),
        private_context_units=_common_context_units(doc_units, _participant_id_map(participants)),
        reference_information_units=_common_rui_units(rui_ranges),
        oracle=_common_oracle(trial, smallest_sets),
        generation_metadata=_common_generation_metadata(
            trial=trial,
            seed=seed,
            strategy=selected_strategy,
            role_style=role_style,
            target_words=target_words,
            min_words=min_words,
            soft_max_words=soft_max_words,
            hard_max_words=hard_max_words,
            validation=validation,
            allocation_metadata=allocation.to_metadata(),
            min_participants=min_participants,
        ),
        dataset_specific={
            "cuad_label_family": trial.get("label_family"),
            "candidate_index": trial.get("candidate_index"),
            "partition_strategy": selected_strategy,
            "private_information_allocation": allocation.to_metadata(),
        },
    )
    common_datapoint, decision_permutation = permute_common_datapoint_decisions(
        common_datapoint,
        seed_material=f"cuad:{trial.get('instance_id')}:{seed}:decision_options",
    )
    public_common = apply_permutation_to_public_common(
        {
            "contract_title": candidate.get("contract_title"),
            "dataset": "CUAD",
            "document_kind": "commercial_contract",
            "task_objective": (trial.get("plan") or {}).get("public_task", {}).get("objective"),
            "decision_set": _common_decision_set(trial),
            "note": "Participants receive private document chunks; hidden metadata records which chunks contain RUIs.",
        },
        decision_permutation,
        source_common_datapoint=common_datapoint,
    )
    return {
        "schema": PARTITION_SCHEMA,
        "common_schema_version": COMMON_DATAPOINT_SCHEMA_VERSION,
        "source_trial_instance_id": trial.get("instance_id"),
        "instance_id": trial.get("instance_id"),
        "dataset": "cuad",
        "source_id": trial.get("source_id"),
        "candidate_index": trial.get("candidate_index"),
        "label_family": trial.get("label_family"),
        "common_datapoint": common_datapoint,
        "seed": seed,
        "partition_strategy": selected_strategy,
        "role_style": role_style,
        "min_participants": min_participants,
        "public_common": public_common,
        "source_document": {
            "contract_index": candidate.get("contract_index"),
            "contract_title": candidate.get("contract_title"),
            "context_char_count": len(context),
            "context_word_count": _word_count(context),
        },
        "partition_scoring_policy": {
            "weights": ASSIGNMENT_SCORE_WEIGHTS,
            "role_overlap": "fraction of participant role terms that appear in the non-RUI chunk",
            "feature_overlap": "fraction of owned-RUI feature terms that appear in the non-RUI chunk",
            "proximity": "1 / (1 + word distance from the chunk midpoint to the nearest owned RUI midpoint divided by target chunk words)",
            "section_match": "1 when the chunk has the same detected section key as an owned RUI, otherwise 0",
            "load_penalty": "participant private word count divided by twice the target chunk words",
            "neutral_chunk_assignment": (
                "RUI chunks are assigned to their RUI owners first. Non-RUI chunks are classified as "
                "RIU-related or background; RIU-related chunks avoid participants who own the related RIUs "
                "when possible, then use balance-first seeded softmax sampling. Background chunks use "
                "stronger semantic fit with balance penalties."
            ),
            "allocation_mode_weights": DEFAULT_ALLOCATION_MODE_WEIGHTS,
            "private_information_allocation": allocation.to_metadata(),
        },
        "hidden_rui_units": [
            {
                "unit_id": rui.unit_id,
                "source_span_index": rui.source_span_index,
                "raw_start": rui.raw_start,
                "raw_end": rui.raw_end,
                "raw_text": rui.raw_text,
                "clean_text": rui.clean_text,
                "legal_feature": rui.legal_feature,
                "summary": rui.summary,
                "section_key": rui.section_key,
            }
            for rui in rui_ranges
        ],
        "hidden_smallest_sufficient_sets": smallest_sets,
        "hidden_common_rui_ids": sorted(common_rui_ids),
        "participants": participants,
        "document_units": doc_units,
        "validation": validation,
    }


def _rui_ranges_for_trial(trial: dict[str, Any], candidate: dict[str, Any]) -> list[RUIRange]:
    selected = (trial.get("selection") or {}).get("selected_span_basis") or []
    if not isinstance(selected, list) or not selected:
        raise ValueError("trial has no selected_span_basis")
    candidate_spans = {
        int(span["span_index"]): span
        for span in candidate.get("positive_spans", [])
        if str(span.get("span_index", "")).isdigit()
    }
    plan_units = {
        str(unit.get("unit_id")): unit
        for unit in (trial.get("plan") or {}).get("evidence_units", [])
        if isinstance(unit, dict)
    }
    out: list[RUIRange] = []
    for row in selected:
        if not isinstance(row, dict):
            continue
        unit_id = str(row.get("unit_id") or "")
        source_span_index = int(row.get("source_span_index") or 0)
        candidate_span = candidate_spans.get(source_span_index)
        if not unit_id or candidate_span is None:
            raise ValueError(f"could not resolve selected span {unit_id}")
        clean_text = str(row.get("text") or candidate_span.get("text") or "")
        answer_start = candidate_span.get("answer_start")
        if not isinstance(answer_start, int):
            raise ValueError(f"{unit_id} has no integer answer_start")
        raw_start, raw_end = _recover_raw_span_range(candidate["contract_text"], answer_start, clean_text)
        plan_unit = plan_units.get(unit_id, {})
        out.append(
            RUIRange(
                unit_id=unit_id,
                source_span_index=source_span_index,
                clean_text=clean_text,
                raw_start=raw_start,
                raw_end=raw_end,
                raw_text=candidate["contract_text"][raw_start:raw_end],
                legal_feature=str(row.get("legal_feature") or plan_unit.get("information_feature") or ""),
                summary=str(plan_unit.get("summary") or row.get("possible_decision_boundary") or ""),
                section_key=_section_key_at(candidate["contract_text"], raw_start),
            )
        )
    return sorted(out, key=lambda rui: (rui.raw_start, rui.raw_end, rui.unit_id))


def _recover_raw_span_range(context: str, answer_start: int, clean_text: str) -> tuple[int, int]:
    goal_tokens = _tokens(clean_text, keep_stopwords=True)
    if not goal_tokens:
        raise ValueError("empty span text")
    exact = context[answer_start : answer_start + len(clean_text)]
    if exact == clean_text:
        return answer_start, answer_start + len(clean_text)
    window_end = min(len(context), answer_start + max(len(clean_text) * 5, len(clean_text) + 1000))
    window = context[answer_start:window_end]
    matches = list(re.finditer(r"\S+", window))
    current: list[str] = []
    for match in matches:
        current.append(match.group(0))
        if len(current) == len(goal_tokens):
            if current == goal_tokens:
                return answer_start + matches[0].start(), answer_start + match.end()
            break
    normalized_window = " ".join(window.split())
    normalized_goal = " ".join(str(clean_text).split())
    if normalized_goal in normalized_window:
        start = answer_start
        end = min(window_end, answer_start + len(clean_text) * 3)
        while end < window_end and " ".join(context[start:end].split()) != normalized_goal:
            end += 1
        if " ".join(context[start:end].split()) == normalized_goal:
            return start, end
    raise ValueError(f"could not recover raw span range at answer_start={answer_start}")


def _smallest_sufficient_sets(trial: dict[str, Any], rui_ranges: list[RUIRange]) -> list[list[str]]:
    known = {rui.unit_id for rui in rui_ranges}
    audit_sets = (trial.get("audit") or {}).get("smallest_sufficient_span_sets") or []
    out: list[list[str]] = []
    if isinstance(audit_sets, list):
        for row in audit_sets:
            if not isinstance(row, dict):
                continue
            ids = [str(item) for item in row.get("unit_ids", []) if str(item)]
            if ids and set(ids) <= known:
                out.append(ids)
    return out


def _retain_one_smallest_sufficient_set(
    trial: dict[str, Any],
    rui_ranges: list[RUIRange],
) -> tuple[list[RUIRange], list[list[str]]]:
    """Restrict partitioning to one audited inclusion-minimal sufficient set."""

    smallest_sets = _smallest_sufficient_sets(trial, rui_ranges)
    if not smallest_sets:
        raise ValueError("CUAD partitioning requires an audited smallest sufficient span set.")
    retained_ids = min(smallest_sets, key=lambda ids: (len(ids), tuple(ids)))
    retained_id_set = set(retained_ids)
    retained_ranges = [rui for rui in rui_ranges if rui.unit_id in retained_id_set]
    if {rui.unit_id for rui in retained_ranges} != retained_id_set:
        raise ValueError("Could not retain the complete audited CUAD smallest sufficient span set.")
    return retained_ranges, [retained_ids]


def _common_source(candidate: dict[str, Any], context: str) -> dict[str, Any]:
    return {
        "source_dataset": "CUAD",
        "source_id": candidate.get("source_id"),
        "document_kind": "commercial_contract",
        "contract_index": candidate.get("contract_index"),
        "contract_title": candidate.get("contract_title"),
        "context_char_count": len(context),
        "context_word_count": _word_count(context),
        "source_preservation": "full_contract_context_partitioned_into_private_chunks",
    }


def _common_task_description(trial: dict[str, Any]) -> dict[str, Any]:
    plan_public = (trial.get("plan") or {}).get("public_task") or {}
    return {
        "title": plan_public.get("title") or f"CUAD {trial.get('label_family') or 'contract'} review",
        "objective": plan_public.get("objective"),
        "answer_format": {
            "type": "single_choice",
            "choices": _common_decision_set(trial),
        },
    }


def _common_decision_set(trial: dict[str, Any]) -> list[str]:
    return list(_decision_value_map(trial).values())


def _common_decision_option_definitions(trial: dict[str, Any]) -> dict[str, Any]:
    plan_public = (trial.get("plan") or {}).get("public_task") or {}
    definitions = plan_public.get("decision_option_definitions")
    definitions = definitions if isinstance(definitions, dict) else {}
    decision_map = _decision_value_map(trial)
    return {
        mapped: definitions.get(original) or ""
        for original, mapped in decision_map.items()
    }


def _common_initial_common_information(
    candidate: dict[str, Any],
    rui_ranges: list[RUIRange],
    common_rui_ids: set[str],
) -> dict[str, Any]:
    common = {
        "source_context": {
            "contract_title": candidate.get("contract_title"),
            "document_kind": "commercial_contract",
            "dataset": "CUAD",
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


def _common_rui_units(rui_ranges: list[RUIRange]) -> list[dict[str, Any]]:
    return [
        {
            "rui_id": rui.unit_id,
            "text": rui.clean_text,
            "raw_text": rui.raw_text,
            "information_feature": rui.legal_feature,
            "summary": rui.summary,
            "source_ref": {
                "source_span_index": rui.source_span_index,
                "raw_start": rui.raw_start,
                "raw_end": rui.raw_end,
                "section_key": rui.section_key,
            },
        }
        for rui in rui_ranges
    ]


def _common_oracle(trial: dict[str, Any], smallest_sets: list[list[str]]) -> dict[str, Any]:
    plan = trial.get("plan") or {}
    oracle = plan.get("oracle") if isinstance(plan.get("oracle"), dict) else {}
    audit = trial.get("audit") if isinstance(trial.get("audit"), dict) else {}
    decision_map = _decision_value_map(trial)
    answer = decision_map.get(str(oracle.get("answer") or ""), oracle.get("answer"))
    return {
        "answer": answer,
        "rationale": oracle.get("rationale"),
        "decision_rule": oracle.get("decision_rule") or oracle.get("rule"),
        "minimum_sufficient_rui_set_size": audit.get("minimum_sufficient_span_set_size")
        or min((len(row) for row in smallest_sets), default=None),
        "smallest_sufficient_rui_sets": [
            {
                "rui_ids": list(row),
                "answer": answer,
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
    min_participants: int,
) -> dict[str, Any]:
    inherited = trial.get("generation_metadata") if isinstance(trial.get("generation_metadata"), dict) else {}
    return {
        "method": "cuad_measured_direct_span_generation_with_full_document_partitioning",
        "no_synthetic_fallback": True,
        "source_candidate": {
            "source_id": trial.get("source_id"),
            "candidate_index": trial.get("candidate_index"),
            "label_family": trial.get("label_family"),
        },
        "task_generation": {
            "reference_information_unit_policy": "direct_cuad_positive_spans_only",
            "min_accept_size": trial.get("min_accept_size"),
            "max_span_count": trial.get("max_span_count"),
            "planning_attempts": trial.get("planning_attempts"),
            "prompt_stage_count": inherited.get("prompt_stage_count") or 3,
            "generation_byproduct_ref": trial.get("generation_byproduct_ref"),
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
            "score_weights": ASSIGNMENT_SCORE_WEIGHTS,
            "allocation_mode_weights": DEFAULT_ALLOCATION_MODE_WEIGHTS,
            "private_information_allocation": allocation_metadata,
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


def _decision_value_map(trial: dict[str, Any]) -> dict[str, str]:
    plan_public = (trial.get("plan") or {}).get("public_task") or {}
    decision_set = [str(item) for item in plan_public.get("decision_set") or []]
    return {
        value: _letter_id("decision", idx)
        for idx, value in enumerate(decision_set, start=1)
    }


def _report_summary(partitions: list[dict[str, Any]]) -> dict[str, Any]:
    strategy_counts = Counter(partition.get("partition_strategy") for partition in partitions)
    valid_count = sum(1 for partition in partitions if (partition.get("validation") or {}).get("valid") is True)
    unit_counts = [len(partition.get("document_units") or []) for partition in partitions]
    participant_counts = [len(partition.get("participants") or []) for partition in partitions]
    chunk_words = [
        int(unit.get("word_count") or 0)
        for partition in partitions
        for unit in partition.get("document_units") or []
    ]
    participant_chunk_gaps = [
        int(((partition.get("validation") or {}).get("metrics") or {}).get("participant_doc_unit_count_gap") or 0)
        for partition in partitions
    ]
    return {
        "valid_count": valid_count,
        "strategy_counts": dict(strategy_counts),
        "document_unit_count_summary": _number_summary(unit_counts),
        "participant_count_summary": _number_summary(participant_counts),
        "chunk_word_count_summary": _number_summary(chunk_words),
        "participant_doc_unit_count_gap_summary": _number_summary(participant_chunk_gaps),
    }


def _parse_args(argv: list[str] | None = None, *, prog: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--accepted-path", required=True)
    parser.add_argument("--cache-dir", default="data/legal_raw")
    parser.add_argument("--output-root", default="generated/cuad_full_document_partitions")
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
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
