"""Generate and audit short free-response FinDVer collaboration tasks."""

from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import islice
import json
from pathlib import Path
from typing import Any, Callable

from ....llm.construction_client import OpenAIJsonClient, OpenAIJsonClientConfig
from ....core.information_allocation import PAPER_ALLOCATION_MODES
from ....core.io import prompt_text, render_template
from .integrity import client_identity, start_run, sha256, verify_inputs, write_json, write_jsonl
from .free_response_validation import (
    leaked_context_ids,
    normalize_evidence_consolidation,
    normalize_free_response_judgment,
    normalize_free_response_plan,
    normalize_qa_evidence_mapping,
    validate_free_grounding_audit,
    public_answer_disclosure_errors,
    validate_objective_disclosure_audit,
    validate_free_response_judgment,
    validate_free_response_plan,
    validate_free_solver_response,
    validate_packet_riu_audit,
    validate_evidence_consolidation,
    validate_qa_evidence_mapping,
)
from .leakage import packet_bm25_hints
from .partition import (
    build_full_context_partition,
    remove_surrounding_contexts,
    validate_packet_context_integrity,
)
from .source import findver_report_candidates
from .validation import normalize_selection, validate_selection


SCHEMA = "findver_free_response_funnel_v1"
SUBSETS = ("all", "ie", "numeric", "knowledge")
DEFAULT_MODEL = "openai.gpt-5.5"
DEFAULT_BASE_URL = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
DEFAULT_ENV_FILE = ".env"
SYSTEM_PROMPT = "finance/findver/free_response_system.txt"
QUESTION_GENERATE_STAGE = "free_question_generate_high_riu_user"


def build_dataset(
    *,
    qa_path: str | Path,
    reports_root: str | Path,
    output_root: str | Path,
    generator_client: Any,
    audit_client: Any,
    blind_clients: list[tuple[str, Any]],
    subset: str,
    allocation_mode: str = "one_each",
    min_participants: int = 2,
    min_units: int = 2,
    max_units: int | None = None,
    candidate_limit: int = 10,
    candidate_start_index: int = 1,
    plan_attempts: int = 3,
    audit_attempts: int = 3,
    full_boundary_min_passes: int | None = None,
    seed: int = 20260730,
    question_generate_stage: str = QUESTION_GENERATE_STAGE,
    source_split: str | None = None,
    participant_context_token_budget: int = 20000,
) -> dict[str, Any]:
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {', '.join(SUBSETS)}")
    if allocation_mode not in (*PAPER_ALLOCATION_MODES, "auto"):
        raise ValueError("invalid allocation mode")
    if not blind_clients:
        raise ValueError("at least one blind solver is required")
    if min_units < 2 or min_participants < 2 or participant_context_token_budget < 1:
        raise ValueError("Finance collaboration requires at least two units/participants and a positive context budget")
    if full_boundary_min_passes is not None and not 1 <= full_boundary_min_passes <= len(blind_clients):
        raise ValueError("full_boundary_min_passes must be within the solver panel size")

    output = Path(output_root).expanduser().resolve()
    manifest = start_run(output, qa_path=qa_path, reports_root=reports_root,
        config={"subset": subset, "source_split": source_split, "allocation_mode": allocation_mode,
                "min_participants": min_participants, "min_units": min_units, "max_units": max_units,
                "candidate_limit": candidate_limit, "candidate_start_index": candidate_start_index,
                "plan_attempts": plan_attempts, "audit_attempts": audit_attempts, "seed": seed,
                "full_boundary_min_passes": full_boundary_min_passes,
                "participant_context_token_budget": participant_context_token_budget,
                "question_generate_stage": question_generate_stage},
        clients={"generator": client_identity(generator_client), "auditor": client_identity(audit_client),
                 "blind_solvers": {name: client_identity(client) for name, client in blind_clients}})
    trace_root = output / "_llm_traces"
    scenarios: list[dict[str, Any]] = []
    byproducts: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    candidates = findver_report_candidates(
        qa_path=qa_path,
        reports_root=reports_root,
        subset=subset,
        source_split=source_split,
    )

    start_index = max(0, candidate_start_index - 1)
    selected_candidates = islice(
        candidates, start_index, start_index + max(0, candidate_limit)
    )
    for source_position, candidate in enumerate(selected_candidates, start=start_index + 1):
        try:
            prepared, mapping_record = _prepare_report_candidate(
                candidate=candidate,
                audit_client=audit_client,
                trace_root=trace_root,
                min_units=min_units,
            )
            if prepared is None:
                result = {
                    "candidate_index": source_position,
                    "source_id": candidate.get("source_id"),
                    "subset": subset,
                    "report": candidate.get("report_name"),
                    "status": "candidate_pool_rejected",
                    "errors": mapping_record["errors"],
                }
                scenario = None
                byproduct = mapping_record
            else:
                result, scenario, byproduct = _try_candidate(
                    candidate=prepared,
                    candidate_index=source_position,
                    output_root=output,
                    trace_root=trace_root,
                    generator_client=generator_client,
                    audit_client=audit_client,
                    blind_clients=blind_clients,
                    allocation_mode=allocation_mode,
                    min_participants=min_participants,
                    min_units=min_units,
                    max_units=max_units
                    or len(prepared.get("candidate_contexts") or []),
                    plan_attempts=plan_attempts,
                    audit_attempts=audit_attempts,
                    full_boundary_min_passes=(
                        len(blind_clients)
                        if full_boundary_min_passes is None
                        else full_boundary_min_passes
                    ),
                    seed=seed,
                    question_generate_stage=question_generate_stage,
                    participant_context_token_budget=participant_context_token_budget,
                )
                byproduct["qa_evidence_mapping"] = mapping_record
                byproduct["source_provenance"] = candidate.get("source_provenance")
                if scenario is not None:
                    scenario["common_datapoint"]["source"]["construction_provenance"] = candidate.get("source_provenance")
        except Exception as exc:  # noqa: BLE001
            result = {
                "candidate_index": source_position,
                "source_id": candidate.get("source_id"),
                "subset": subset,
                "status": "exception",
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
            scenario = None
            byproduct = {
                "schema": "findver_free_response_generation_byproduct_v1",
                "candidate_index": source_position,
                "source_id": candidate.get("source_id"),
                "exception": result["errors"][0],
            }
        verify_inputs(manifest)
        if candidate.get("source_provenance") and sha256(candidate["report_source_path"]) != candidate["source_provenance"]["report_sha256"]:
            raise RuntimeError("FinDVer raw report changed during execution")
        results.append(result)
        byproducts.append(byproduct)
        if scenario is not None:
            scenarios.append(scenario)
        _write_progress(output, results, scenarios, byproducts)

    report = {
        "schema": SCHEMA,
        "source_qa_path": str(Path(qa_path).expanduser().resolve()),
        "reports_root": str(Path(reports_root).expanduser().resolve()),
        "output_root": str(output),
        "subset": subset,
        "candidate_unit": "report",
        "allocation_mode": allocation_mode,
        "candidate_count": len(results),
        "accepted_count": len(scenarios),
        "funnel_counts": _status_counts(results),
        "model_roles": {
            "generator": generator_client.cfg.model,
            "grounding_and_judge": audit_client.cfg.model,
            "blind_solvers": [
                {"auditor_id": auditor_id, "model": client.cfg.model}
                for auditor_id, client in blind_clients
            ],
        },
        "response_mode": "short_answer_judged",
        "packet_context_policy": (
            "selected RIU blocks follow the allocation mode; surrounding native "
            "blocks with clear unassigned-RIU visibility are removed once without "
            "reallocation, followed by repeated collective and participant-answerability audits"
        ),
        "prompt_files": _prompt_manifest(question_generate_stage),
        "outputs": {
            "scenarios": str(output / "scenarios.jsonl"),
            "generation_byproducts": str(output / "generation_byproducts.jsonl"),
            "candidate_results": str(output / "candidate_results.jsonl"),
            "traces": str(trace_root),
        },
    }
    write_json(output / "funnel_report.json", report)
    manifest.update(status="accepted" if scenarios else "rejected", accepted_count=len(scenarios),
                    outputs={name: sha256(output / name) for name in (
                        "scenarios.jsonl", "candidate_results.jsonl", "generation_byproducts.jsonl",
                        "funnel_report.json") if (output / name).exists()})
    write_json(output / "manifest.json", manifest)
    return report


def _prepare_report_candidate(
    *,
    candidate: dict[str, Any],
    audit_client: Any,
    trace_root: Path,
    min_units: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for qa in candidate.get("source_qas") or []:
        qa_id = str(qa.get("qa_id") or "")
        mapping = _call(
            client=audit_client,
            stage="free_qa_evidence_map_user",
            variables={
                "SOURCE_QA_JSON": {
                    key: value
                    for key, value in qa.items()
                    if key != "relevant_contexts"
                },
                "ANNOTATED_CONTEXTS_JSON": qa.get("relevant_contexts") or [],
                "AUDIT_FEEDBACK": "None. This is the first and only semantic mapping pass.",
            },
            trace_root=trace_root,
            instance_id=f"findver_map_{candidate['source_id']}_{qa_id}",
        )
        mapping = normalize_qa_evidence_mapping(mapping)
        errors = validate_qa_evidence_mapping(mapping, qa=qa)
        row = {
            "qa_id": qa_id,
            "mapping": mapping,
            "errors": errors,
            "model": audit_client.cfg.model,
            "request_meta": dict(
                getattr(audit_client, "last_request_meta", {}) or {}
            ),
        }
        if errors:
            rejected.append(row)
        else:
            accepted.append(row)

    context_to_qas: dict[int, set[str]] = {}
    context_to_subsets: dict[int, set[str]] = {}
    context_contributions: dict[int, list[dict[str, str]]] = {}
    for row in accepted:
        qa_id = str(row["qa_id"])
        source_qa = next(
            (
                qa
                for qa in candidate.get("source_qas") or []
                if str(qa.get("qa_id") or "") == qa_id
            ),
            {},
        )
        qa_subset = str(source_qa.get("subset") or "")
        for evidence in row["mapping"].get("evidence_blocks") or []:
            context_id = int(evidence["context_id"])
            context_to_qas.setdefault(context_id, set()).add(qa_id)
            if qa_subset:
                context_to_subsets.setdefault(context_id, set()).add(qa_subset)
            context_contributions.setdefault(context_id, []).append(
                {
                    "qa_id": qa_id,
                    "role": str(evidence.get("role") or ""),
                    "roles": list(evidence.get("roles") or []),
                    "contribution": str(evidence.get("contribution") or ""),
                }
            )
    context_ids = list(context_to_qas)
    mapping_record = {
        "schema": "findver_qa_evidence_mapping_v1",
        "source_id": candidate.get("source_id"),
        "report": candidate.get("report_name"),
        "accepted_mappings": accepted,
        "rejected_mappings": rejected,
        "candidate_context_ids": context_ids,
        "errors": [],
    }
    if len(context_ids) < min_units:
        mapping_record["errors"] = [
            f"fewer than {min_units} distinct mapped native contexts remain"
        ]
        return None, mapping_record

    contexts_by_id = {
        int(row["context_id"]): row
        for row in candidate.get("full_report_contexts") or []
    }
    valid_qa_ids = {str(row["qa_id"]) for row in accepted}
    valid_qas = [
        qa
        for qa in candidate.get("source_qas") or []
        if str(qa.get("qa_id") or "") in valid_qa_ids
    ]
    mapped_candidates = [
        {
            **contexts_by_id[context_id],
            "source_context_ids": [context_id],
            "source_qa_ids": sorted(context_to_qas[context_id]),
            "source_subsets": sorted(context_to_subsets.get(context_id, set())),
            "contributions": context_contributions.get(context_id, []),
        }
        for context_id in context_ids
    ]
    consolidation = _call(
        client=audit_client,
        stage="free_evidence_consolidate_user",
        variables={
            "REPORT_ID": candidate.get("report_name"),
            "MAPPED_CANDIDATE_BLOCKS_JSON": mapped_candidates,
        },
        trace_root=trace_root,
        instance_id=f"findver_consolidate_{candidate['source_id']}",
    )
    consolidation = normalize_evidence_consolidation(consolidation)
    consolidation_errors = validate_evidence_consolidation(
        consolidation,
        candidate_context_ids=context_ids,
    )
    mapping_record["evidence_consolidation"] = {
        "response": consolidation,
        "errors": consolidation_errors,
        "model": audit_client.cfg.model,
        "request_meta": dict(
            getattr(audit_client, "last_request_meta", {}) or {}
        ),
    }
    if consolidation_errors:
        mapping_record["errors"] = consolidation_errors
        return None, mapping_record

    normalized_candidates = _consolidate_candidate_contexts(
        mapped_candidates,
        consolidation,
    )
    if len(normalized_candidates) < min_units:
        mapping_record["errors"] = [
            f"fewer than {min_units} normalized candidate blocks remain"
        ]
        return None, mapping_record

    prepared = {
        **candidate,
        "candidate_context_ids": context_ids,
        "candidate_contexts": normalized_candidates,
        "candidate_relations": consolidation.get("information_relations") or [],
        "evidence_consolidation": consolidation,
        "source_qas": valid_qas,
        "source_knowledge": _dedupe(
            [
                str(value)
                for qa in valid_qas
                for value in qa.get("public_knowledge") or []
                if str(value)
            ]
        ),
        "qa_evidence_mappings": [
            row["mapping"] for row in accepted
        ],
        "rejected_qa_evidence_mappings": [
            row for row in rejected
        ],
    }
    return prepared, mapping_record


def _consolidate_candidate_contexts(
    mapped_candidates: list[dict[str, Any]],
    consolidation: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = {int(row["context_id"]): row for row in mapped_candidates}
    excluded = {
        int(row["context_id"])
        for row in consolidation.get("excluded_contexts") or []
        if isinstance(row, dict)
    }
    inherited_by_id: dict[int, list[dict[str, Any]]] = {}
    for exclusion in consolidation.get("excluded_contexts") or []:
        if not isinstance(exclusion, dict):
            continue
        excluded_row = by_id[int(exclusion["context_id"])]
        for covering_id in exclusion.get("covered_by_context_ids") or []:
            inherited_by_id.setdefault(int(covering_id), []).append(excluded_row)
    groups = [
        [int(value) for value in row.get("source_context_ids") or []]
        for row in consolidation.get("merge_groups") or []
        if isinstance(row, dict)
    ]
    grouped = {context_id for group in groups for context_id in group}
    units = groups + [
        [context_id]
        for context_id in sorted(by_id)
        if context_id not in grouped and context_id not in excluded
    ]
    normalized: list[dict[str, Any]] = []
    for source_ids in sorted(units, key=lambda values: values[0]):
        rows = [by_id[context_id] for context_id in source_ids]
        metadata_rows = rows + [
            inherited
            for context_id in source_ids
            for inherited in inherited_by_id.get(context_id, [])
        ]
        candidate_block_id = (
            f"ctx_{source_ids[0]:06d}"
            if len(source_ids) == 1
            else "merged_" + "_".join(f"{value:06d}" for value in source_ids)
        )
        normalized.append(
            {
                "candidate_block_id": candidate_block_id,
                "context_id": source_ids[0],
                "source_context_ids": source_ids,
                "type": rows[0].get("type") if len(rows) == 1 else "merged",
                "source_context_types": [str(row.get("type") or "") for row in rows],
                "text": "\n\n".join(str(row.get("text") or "") for row in rows),
                "source_qa_ids": sorted(
                    {
                        str(qa_id)
                        for row in metadata_rows
                        for qa_id in row.get("source_qa_ids") or []
                    }
                ),
                "source_subsets": sorted(
                    {
                        str(value)
                        for row in metadata_rows
                        for value in row.get("source_subsets") or []
                    }
                ),
                "contributions": [
                    contribution
                    for row in metadata_rows
                    for contribution in row.get("contributions") or []
                ],
            }
        )
    return normalized


def _try_candidate(
    *,
    candidate: dict[str, Any],
    candidate_index: int,
    output_root: Path,
    trace_root: Path,
    generator_client: Any,
    audit_client: Any,
    blind_clients: list[tuple[str, Any]],
    allocation_mode: str,
    min_participants: int,
    min_units: int,
    max_units: int,
    plan_attempts: int,
    audit_attempts: int,
    full_boundary_min_passes: int,
    seed: int,
    question_generate_stage: str,
    participant_context_token_budget: int = 20000,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    instance_id = f"findver_free_{candidate['source_id']}"
    source_seed = _source_seed(candidate)
    stages: list[dict[str, Any]] = []
    feedback = "None. This is the first generation attempt."
    last_errors: list[str] = []
    last_selection: dict[str, Any] | None = None
    last_plan: dict[str, Any] | None = None
    last_audits: dict[str, Any] = {}
    for plan_attempt in range(1, max(1, plan_attempts) + 1):
        generated = _call(
            client=generator_client,
            stage=question_generate_stage,
            variables={
                "SOURCE_SEED_JSON": source_seed,
                "CANDIDATE_CONTEXTS_JSON": candidate["candidate_contexts"],
                "CANDIDATE_RELATIONS_JSON": candidate.get("candidate_relations")
                or [],
                "MIN_UNITS": min_units,
                "MAX_UNITS": max_units,
                "REJECTION_FEEDBACK": feedback,
            },
            trace_root=trace_root,
            instance_id=instance_id,
        )
        selection = generated.get("selection")
        selection = selection if isinstance(selection, dict) else {}
        selection = normalize_selection(selection, candidate)
        selection_errors = validate_selection(
            selection,
            candidate,
            min_units=min_units,
            max_units=max_units,
        )
        stages.append(
            _stage(
                "task_generation_selection",
                generator_client,
                generated,
                selection_errors,
                attempt=plan_attempt,
            )
        )
        last_selection = selection
        if selection_errors:
            last_errors = selection_errors
            feedback = _feedback(
                "context_selection",
                selection_errors,
                action="reselect_rius",
                previous_selection=selection,
            )
            continue

        plan = {
            "public_task": generated.get("public_task"),
            "oracle": generated.get("oracle"),
        }
        plan = normalize_free_response_plan(plan, selection)
        plan_errors = validate_free_response_plan(plan, selection, candidate) + public_answer_disclosure_errors(plan)
        stages.append(
            _stage(
                "free_question_generation",
                generator_client,
                plan,
                plan_errors,
                attempt=plan_attempt,
            )
        )
        last_plan = plan
        if plan_errors:
            last_errors = plan_errors
            feedback = _feedback(
                "question_generation",
                plan_errors,
                action="redesign_task_and_reselect_if_needed",
                previous_selection=selection,
                previous_task=plan,
            )
            continue

        objective_audit, objective_errors = _run_structured_audit(
            client=audit_client, stage="free_objective_disclosure_audit_user",
            variables={"PUBLIC_TASK_JSON": plan["public_task"], "ORACLE_JSON": plan["oracle"],
                       "SELECTED_CONTEXTS_JSON": selection},
            validator=lambda row: validate_objective_disclosure_audit(row, plan),
            trace_root=trace_root, instance_id=instance_id, audit_attempts=audit_attempts,
            stages=stages, plan_attempt=plan_attempt)
        if objective_errors:
            last_errors = objective_errors
            last_audits = {"objective_disclosure": objective_audit}
            feedback = _feedback("objective_disclosure", objective_errors,
                action="redesign_public_task_without_answer_disclosure",
                previous_selection=selection, previous_task=plan, audits=last_audits)
            continue

        audit_ready = False
        max_prunes = max(0, len(selection.get("selected_contexts") or []) - min_units)
        for repair_round in range(max_prunes + 1):
            grounding, grounding_errors = _run_structured_audit(
                client=audit_client,
                stage="free_grounding_audit_user",
                variables={
                    "PLAN_JSON": plan,
                    "SELECTED_CONTEXTS_JSON": selection,
                },
                validator=lambda row: validate_free_grounding_audit(row, plan),
                trace_root=trace_root,
                instance_id=instance_id,
                audit_attempts=audit_attempts,
                stages=stages,
                plan_attempt=plan_attempt,
            )
            blocking_grounding_errors = [
                error
                for error in grounding_errors
                if error
                != "grounding audit did not confirm all_selected_rius_necessary"
            ]
            if blocking_grounding_errors:
                last_errors = grounding_errors
                last_audits = {"grounding": grounding}
                feedback = _feedback(
                    "grounding",
                    grounding_errors,
                    action="redesign_task_and_reselect_if_needed",
                    previous_selection=selection,
                    previous_task=plan,
                    audits=last_audits,
                )
                break

            public_panel, public_errors = _run_answerability_panel(
                cases=[
                    {
                        "case_id": "public_only",
                        "is_full_set": False,
                        "available_riu_ids": [],
                        "evidence": [],
                    }
                ],
                public_task=plan["public_task"],
                public_information=_public_information(candidate, selection),
                oracle=plan["oracle"],
                blind_clients=blind_clients,
                judge_client=audit_client,
                trace_root=trace_root,
                instance_id=instance_id,
                plan_attempt=plan_attempt,
                audit_attempts=audit_attempts,
                stages=stages,
                require_full_passes=0,
                reject_any_non_full_pass=True,
                panel_name="public_only",
            )
            if public_errors:
                last_errors = public_errors
                last_audits = {
                    "grounding": grounding,
                    "public_only": public_panel,
                }
                feedback = _feedback(
                    "public_only",
                    public_errors,
                    action="redesign_task",
                    previous_selection=selection,
                    previous_task=plan,
                    audits=last_audits,
                )
                break

            boundary_panel, boundary_errors = _run_answerability_panel(
                cases=_riu_boundary_cases(selection),
                public_task=plan["public_task"],
                public_information=_public_information(candidate, selection),
                oracle=plan["oracle"],
                blind_clients=blind_clients,
                judge_client=audit_client,
                trace_root=trace_root,
                instance_id=instance_id,
                plan_attempt=plan_attempt,
                audit_attempts=audit_attempts,
                stages=stages,
                require_full_passes=full_boundary_min_passes,
                reject_any_non_full_pass=True,
                panel_name="riu_boundary",
            )
            if not boundary_errors and not grounding_errors:
                audit_ready = True
                break

            repaired = _prune_nonessential_riu(
                selection=selection,
                plan=plan,
                boundary_panel=boundary_panel,
                boundary_errors=boundary_errors,
                min_units=min_units,
            )
            if repaired is not None and repair_round < max_prunes:
                selection, plan, repair_record = repaired
                repair_errors = validate_selection(
                    selection,
                    candidate,
                    min_units=min_units,
                    max_units=max_units,
                ) + validate_free_response_plan(plan, selection, candidate)
                if repair_errors:
                    last_errors = _dedupe(repair_errors)
                    feedback = _feedback(
                        "riu_pruning_repair",
                        last_errors,
                        action="redesign_task_and_reselect_if_needed",
                        previous_selection=selection,
                        previous_task=plan,
                        audits={
                            "grounding": grounding,
                            "public_only": public_panel,
                            "riu_boundary": boundary_panel,
                        },
                    )
                    break
                last_selection = selection
                last_plan = plan
                stages.append(
                    {
                        "stage": "riu_pruning_repair",
                        "attempt": plan_attempt,
                        "repair_round": repair_round + 1,
                        "valid": True,
                        "errors": [],
                        "response": repair_record,
                        "local_only": True,
                    }
                )
                continue

            last_errors = boundary_errors or grounding_errors
            last_audits = {
                "grounding": grounding,
                "public_only": public_panel,
                "riu_boundary": boundary_panel,
            }
            feedback = _feedback(
                "riu_boundary",
                last_errors,
                action="reselect_rius_or_redesign_task",
                previous_selection=selection,
                previous_task=plan,
                audits=last_audits,
            )
            break
        if not audit_ready:
            continue

        packet_candidate = deepcopy(candidate)
        packet_candidate["context_policy"] = (
            "sanitized_native_context_packets_with_no_unassigned_riu_visibility"
        )
        packet_candidate["source_knowledge"] = _public_information(
            candidate,
            selection,
        )["source_knowledge"]
        selected_riu_context_ids = {
            int(context_id)
            for row in selection.get("selected_contexts") or []
            for context_id in row.get("source_context_ids") or [row["context_id"]]
        }
        scenario = build_full_context_partition(
            candidate=packet_candidate,
            selection=selection,
            plan=plan,
            audits={
                "grounding": grounding,
                "public_only": public_panel,
                "riu_boundary": boundary_panel,
            },
            seed=seed,
            allocation_mode=allocation_mode,
            min_participants=min_participants,
        )
        packet_audits, packet_errors, leaking_ids = _run_packet_audits(
            scenario=scenario,
            selection=selection,
            plan=plan,
            audit_client=audit_client,
            trace_root=trace_root,
            instance_id=instance_id,
            plan_attempt=plan_attempt,
            audit_attempts=audit_attempts,
            stages=stages,
        )
        if packet_errors:
            last_errors = packet_errors
            last_audits = {
                "grounding": grounding,
                "public_only": public_panel,
                "riu_boundary": boundary_panel,
                "packet_leakage": packet_audits,
            }
            feedback = _feedback(
                "packet_audit",
                packet_errors,
                action="reselect_rius",
                previous_selection=selection,
                previous_task=plan,
                audits=last_audits,
            )
            continue

        selected_riu_leaks = leaking_ids & selected_riu_context_ids
        if selected_riu_leaks:
            errors = [
                "selected RIU block exposes an unassigned RIU and cannot be "
                f"removed: {sorted(selected_riu_leaks)}"
            ]
            last_errors = errors
            last_audits = {
                "grounding": grounding,
                "public_only": public_panel,
                "riu_boundary": boundary_panel,
                "packet_leakage": packet_audits,
            }
            feedback = _feedback(
                "selected_riu_leakage",
                errors,
                action="reselect_rius",
                previous_selection=selection,
                previous_task=plan,
                audits=last_audits,
            )
            continue

        sanitization = remove_surrounding_contexts(
            scenario,
            source_context_ids=leaking_ids,
            selected_riu_context_ids=selected_riu_context_ids,
        )
        integrity_errors = validate_packet_context_integrity(
            scenario,
            selected_riu_context_ids=selected_riu_context_ids,
            removed_context_ids=set(sanitization["removed_context_ids"]),
        )
        stages.append(
            {
                "stage": "packet_integrity_gate",
                "attempt": plan_attempt,
                "valid": not integrity_errors,
                "errors": integrity_errors,
                "response": sanitization,
                "local_only": True,
            }
        )
        if integrity_errors:
            last_errors = integrity_errors
            last_audits = {
                "grounding": grounding,
                "public_only": public_panel,
                "riu_boundary": boundary_panel,
                "packet_leakage": packet_audits,
                "sanitization": sanitization,
            }
            feedback = _feedback(
                "packet_integrity",
                integrity_errors,
                action="reselect_rius",
                previous_selection=selection,
                previous_task=plan,
                audits=last_audits,
            )
            continue

        post_packet_audits, post_errors, post_leaks = _run_packet_audits(
            scenario=scenario, selection=selection, plan=plan, audit_client=audit_client,
            trace_root=trace_root, instance_id=instance_id, plan_attempt=plan_attempt,
            audit_attempts=audit_attempts, stages=stages)
        if post_leaks or any(row.get("visible") is True for report in post_packet_audits
                             for row in report["audit"].get("unassigned_riu_results") or []):
            post_errors.append("post-removal packet still reconstructs an unassigned PIU")
        participant_panel, participant_errors = _run_answerability_panel(
            cases=_participant_answerability_cases(scenario, token_budget=participant_context_token_budget),
            public_task=plan["public_task"],
            public_information=_visible_common_information(scenario["common_datapoint"]),
            oracle=plan["oracle"], blind_clients=blind_clients, judge_client=audit_client,
            trace_root=trace_root, instance_id=instance_id, plan_attempt=plan_attempt,
            audit_attempts=audit_attempts, stages=stages, require_full_passes=0,
            reject_any_non_full_pass=True, panel_name="participant_full_view")
        final_errors = post_errors + participant_errors
        if final_errors:
            last_errors = final_errors
            last_audits = {"grounding": grounding, "public_only": public_panel,
                           "riu_boundary": boundary_panel, "packet_leakage": packet_audits,
                           "sanitization": sanitization, "post_removal_packet": post_packet_audits,
                           "participant_full_view": participant_panel}
            feedback = _feedback("participant_full_view", final_errors,
                action="reselect_rius_or_redesign_task", previous_selection=selection,
                previous_task=plan, audits=last_audits)
            continue
        scenario["common_datapoint"]["generation_metadata"]["objective_disclosure_audit"] = objective_audit
        scenario["common_datapoint"]["generation_metadata"]["participant_full_view_audit"] = participant_panel
        scenario["common_datapoint"]["generation_metadata"]["post_removal_packet_audit"] = post_packet_audits

        scenario["common_datapoint"]["generation_metadata"]["context_sanitization"] = sanitization
        scenario["common_datapoint"]["dataset_specific"]["interaction_context_variant"] = (
            "sanitized_context"
        )
        scenario["common_datapoint"]["format_validation"] = scenario["validation"]
        audits = {
            "objective_disclosure": objective_audit,
            "grounding": grounding,
            "public_only": public_panel,
            "riu_boundary": boundary_panel,
            "packet_leakage": packet_audits,
            "sanitization": sanitization,
            "post_removal_packet": post_packet_audits,
            "participant_full_view": participant_panel,
        }
        byproduct = _byproduct(
            candidate_index,
            candidate,
            selection,
            plan,
            audits,
            stages,
            question_generate_stage=question_generate_stage,
            accepted_attempt=plan_attempt,
        )
        return (
            _result(candidate_index, candidate, "accepted", []),
            scenario,
            byproduct,
        )

    return _terminal(
        candidate_index,
        candidate,
        "audit_failed",
        last_errors,
        last_selection,
        last_plan,
        last_audits,
        stages,
        question_generate_stage=question_generate_stage,
    )



def _visible_common_information(common: dict[str, Any]) -> dict[str, Any]:
    from ....interaction.llm_actors import _public_task_payload
    return _public_task_payload(common)["initial_common_information"]


def _participant_answerability_cases(scenario: dict[str, Any], *, token_budget: int) -> list[dict[str, Any]]:
    from ....interaction.llm_actors import _private_context_payload
    from ....interaction.types import BudgetSpec
    common = scenario["common_datapoint"]
    shared = {str(row.get("rui_id")) for row in
              (common.get("initial_common_information") or {}).get("common_reference_units") or []}
    cases = []
    for participant in common.get("participants") or []:
        visible = _private_context_payload(common, participant,
            BudgetSpec(participant_context_token_budget=token_budget))
        cases.append({"case_id": f"participant_{participant['participant_id']}",
                      "is_full_set": False,
                      "available_riu_ids": sorted(set(participant.get("owned_rui_ids") or []) | shared),
                      "evidence": [{"text": visible}]})
    return cases


def _run_answerability_panel(
    *,
    cases: list[dict[str, Any]],
    public_task: dict[str, Any],
    public_information: dict[str, Any],
    oracle: dict[str, Any],
    blind_clients: list[tuple[str, Any]],
    judge_client: Any,
    trace_root: Path,
    instance_id: str,
    plan_attempt: int,
    audit_attempts: int,
    stages: list[dict[str, Any]],
    require_full_passes: int,
    reject_any_non_full_pass: bool,
    panel_name: str,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    case_results: list[dict[str, Any]] = []
    for case in cases:
        solver_results: list[dict[str, Any]] = []
        for auditor_id, client in blind_clients:
            response, response_errors = _run_structured_audit(
                client=client,
                stage="free_solver_user",
                variables={
                    "PUBLIC_TASK_JSON": public_task,
                    "PUBLIC_INFORMATION_JSON": public_information,
                    "RIU_EVIDENCE_JSON": [{"text": row.get("text", "")} for row in case.get("evidence") or []],
                },
                validator=validate_free_solver_response,
                trace_root=trace_root,
                instance_id=instance_id,
                audit_attempts=audit_attempts,
                stages=stages,
                plan_attempt=plan_attempt,
                system_prompt="finance/findver/free_solver_system.txt",
                stage_name=f"{panel_name}_solver",
            )
            judgment: dict[str, Any] = {}
            judgment_errors: list[str] = []
            if not response_errors:
                judgment, judgment_errors = _run_structured_audit(
                    client=judge_client,
                    stage="free_response_judge_user",
                    variables={
                        "PUBLIC_TASK_JSON": public_task,
                        "ORACLE_JSON": oracle,
                        "PUBLIC_INFORMATION_JSON": public_information,
                        "AVAILABLE_EVIDENCE_JSON": case.get("evidence") or [],
                        "BOUNDARY_CASE_JSON": {
                            key: value
                            for key, value in case.items()
                            if key != "evidence"
                        },
                        "SOLVER_RESPONSE_JSON": response,
                    },
                    validator=lambda row, case=case: _validate_normalized_judgment(
                        row,
                        oracle=oracle,
                        available_riu_ids=case.get("available_riu_ids") or [],
                    ),
                    trace_root=trace_root,
                    instance_id=instance_id,
                    audit_attempts=audit_attempts,
                    stages=stages,
                    plan_attempt=plan_attempt,
                    stage_name=f"{panel_name}_judge",
                )
            structural = response_errors + judgment_errors
            if structural:
                errors.extend(
                    f"{case['case_id']}/{auditor_id}: {error}"
                    for error in structural
                )
            solver_results.append(
                {
                    "auditor_id": auditor_id,
                    "model": client.cfg.model,
                    "response": response,
                    "judgment": judgment,
                    "passes": judgment.get("passes") is True and not structural,
                    "errors": structural,
                }
            )
        pass_count = sum(row["passes"] for row in solver_results)
        is_full = bool(case.get("is_full_set"))
        if is_full and pass_count < require_full_passes:
            errors.append(
                f"{case['case_id']}: full RIU set passed only "
                f"{pass_count}/{len(blind_clients)} solvers"
            )
        if not is_full and reject_any_non_full_pass and pass_count:
            errors.append(
                f"{case['case_id']}: answer recovered by "
                f"{pass_count}/{len(blind_clients)} solvers"
            )
        case_results.append(
            {
                "case_id": case["case_id"],
                "is_full_set": is_full,
                "omitted_riu_id": case.get("omitted_riu_id"),
                "available_riu_ids": case.get("available_riu_ids") or [],
                "pass_count": pass_count,
                "solver_results": solver_results,
            }
        )
    return {
        "schema": f"findver_free_{panel_name}_panel_v1",
        "passed": not errors,
        "case_results": case_results,
    }, _dedupe(errors)


def _validate_normalized_judgment(
    judgment: dict[str, Any],
    *,
    oracle: dict[str, Any],
    available_riu_ids: list[str],
) -> list[str]:
    supplied = {key: deepcopy(judgment[key]) for key in ("passes", "required_step_coverage") if key in judgment}
    statuses = {str(row.get("claim_id")): row.get("status") for row in judgment.get("claim_results") or []
                if isinstance(row, dict) and "status" in row}
    normalize_free_response_judgment(judgment, oracle=oracle)
    contradictions = [f"judge supplied {key} contradicts atomic judgments"
                      for key, value in supplied.items() if value != judgment.get(key)]
    contradictions += ["judge supplied claim status contradicts atomic judgments" for row in
                       judgment.get("claim_results") or [] if isinstance(row, dict)
                       and str(row.get("claim_id")) in statuses
                       and statuses[str(row.get("claim_id"))] != row.get("status")]
    return contradictions + validate_free_response_judgment(
        judgment,
        oracle=oracle,
        available_riu_ids=available_riu_ids,
    )


def _run_packet_audits(
    *,
    scenario: dict[str, Any],
    selection: dict[str, Any],
    plan: dict[str, Any],
    audit_client: Any,
    trace_root: Path,
    instance_id: str,
    plan_attempt: int,
    audit_attempts: int,
    stages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], set[int]]:
    common = scenario["common_datapoint"]
    contexts_by_id = {
        str(row["context_unit_id"]): row
        for row in common.get("private_context_units") or []
    }
    common_ids = {
        str(row.get("rui_id"))
        for row in (common.get("initial_common_information") or {}).get(
            "common_reference_units"
        )
        or []
        if isinstance(row, dict) and row.get("rui_id")
    }
    rui_specs = _riu_specs(selection, plan)
    all_ids = {str(row["riu_id"]) for row in rui_specs}
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    leaking_ids: set[int] = set()
    for participant in common.get("participants") or []:
        participant_id = str(participant["participant_id"])
        assigned_or_shared = {
            str(value) for value in participant.get("owned_rui_ids") or []
        } | common_ids
        packet_contexts = [
            {
                "context_unit_id": contexts_by_id[context_id].get(
                    "context_unit_id"
                ),
                "text": contexts_by_id[context_id].get("text"),
                "source_ref": contexts_by_id[context_id].get("source_ref"),
                "approx_tokens": contexts_by_id[context_id].get(
                    "approx_tokens"
                ),
            }
            for context_id in participant.get("private_context_unit_ids") or []
            if context_id in contexts_by_id
        ]
        packet = {
            "participant_id": participant_id,
            "contexts": packet_contexts,
            "approx_tokens": sum(
                int(row.get("approx_tokens") or 0) for row in packet_contexts
            ),
        }
        hints = packet_bm25_hints(
            packet,
            rui_specs,
            unauthorized_rui_ids=all_ids - assigned_or_shared,
        )
        packet_source_context_ids = {
            int((row.get("source_ref") or {}).get("context_id"))
            for row in packet_contexts
        }

        def validate_packet_response(row: dict[str, Any]) -> list[str]:
            return validate_packet_riu_audit(
                row,
                all_rui_ids=all_ids,
                assigned_or_shared_rui_ids=assigned_or_shared,
                packet_context_ids=packet_source_context_ids,
                expected_participant_id=participant_id,
            )

        audit, audit_errors = _run_structured_audit(
            client=audit_client,
            stage="free_packet_leakage_audit_user",
            variables={
                "PUBLIC_TASK_JSON": plan["public_task"],
                "PUBLIC_INFORMATION_JSON": _visible_common_information(common),
                "RIU_SPECS_JSON": rui_specs,
                "PACKET_JSON": packet,
                "ASSIGNED_OR_SHARED_RIU_IDS_JSON": sorted(assigned_or_shared),
                "UNASSIGNED_RIU_IDS_JSON": sorted(all_ids - assigned_or_shared),
                "BM25_HINTS_JSON": hints,
            },
            validator=validate_packet_response,
            trace_root=trace_root,
            instance_id=instance_id,
            audit_attempts=audit_attempts,
            stages=stages,
            plan_attempt=plan_attempt,
            stage_name="packet_unassigned_riu_visibility",
        )
        leaked = leaked_context_ids(audit)
        leaking_ids.update(leaked)
        errors.extend(
            f"{participant_id}: {error}"
            for error in audit_errors
        )
        reports.append(
            {
                "participant_id": participant_id,
                "assigned_or_shared_rui_ids": sorted(assigned_or_shared),
                "unassigned_rui_ids": sorted(all_ids - assigned_or_shared),
                "bm25_hints": hints,
                "audit": audit,
                "errors": audit_errors,
                "leaking_context_ids": sorted(leaked),
            }
        )
    return reports, _dedupe(errors), leaking_ids


def _run_structured_audit(
    *,
    client: Any,
    stage: str,
    variables: dict[str, Any],
    validator: Callable[[dict[str, Any]], list[str]],
    trace_root: Path,
    instance_id: str,
    audit_attempts: int,
    stages: list[dict[str, Any]],
    plan_attempt: int,
    system_prompt: str = SYSTEM_PROMPT,
    stage_name: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    response: dict[str, Any] = {}
    errors = ["audit did not run"]
    feedback = "None. This is the first response attempt."
    for audit_attempt in range(1, max(1, audit_attempts) + 1):
        response = _call(
            client=client,
            stage=stage,
            variables={**variables, "AUDIT_FEEDBACK": feedback, "FORMAT_FEEDBACK": feedback},
            trace_root=trace_root,
            instance_id=instance_id,
            system_prompt=system_prompt,
        )
        errors = validator(response)
        stages.append(
            _stage(
                stage_name or stage,
                client,
                response,
                errors,
                attempt=plan_attempt,
                audit_retry=audit_attempt,
            )
        )
        if not errors:
            break
        feedback = json.dumps(
            {
                "previous_output_invalid": True,
                "errors": errors,
                "instruction": "Correct the JSON and re-audit the unchanged input.",
            },
            ensure_ascii=False,
            indent=2,
        )
    return response, errors


def _call(
    *,
    client: Any,
    stage: str,
    variables: dict[str, Any],
    trace_root: Path,
    instance_id: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    return client.generate_json(
        prompt_id=f"findver.free.{stage}",
        system_prompt=prompt_text(system_prompt),
        user_prompt=render_template(
            prompt_text(f"finance/findver/{stage}.txt"),
            variables,
        ),
        trace_dir=trace_root / instance_id,
    )


def _riu_boundary_cases(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "riu_id": str(row["unit_id"]),
            "text": str(row.get("text") or ""),
        }
        for row in selection.get("selected_contexts") or []
    ]
    cases: list[dict[str, Any]] = []
    for omitted in rows:
        available = [row for row in rows if row["riu_id"] != omitted["riu_id"]]
        cases.append(
            {
                "case_id": f"leave_out_{omitted['riu_id']}",
                "is_full_set": False,
                "omitted_riu_id": omitted["riu_id"],
                "available_riu_ids": [row["riu_id"] for row in available],
                "evidence": available,
            }
        )
    cases.append(
        {
            "case_id": "full",
            "is_full_set": True,
            "available_riu_ids": [row["riu_id"] for row in rows],
            "evidence": rows,
        }
    )
    return cases


def _prune_nonessential_riu(
    *,
    selection: dict[str, Any],
    plan: dict[str, Any],
    boundary_panel: dict[str, Any],
    boundary_errors: list[str],
    min_units: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    if not boundary_errors or any(
        not (error.startswith("leave_out_") and ": answer recovered by " in error)
        for error in boundary_errors
    ):
        return None
    selected = [
        row
        for row in selection.get("selected_contexts") or []
        if isinstance(row, dict)
    ]
    if len(selected) <= min_units:
        return None
    recovered = next(
        (
            row
            for row in boundary_panel.get("case_results") or []
            if not row.get("is_full_set") and int(row.get("pass_count") or 0) > 0
        ),
        None,
    )
    if not isinstance(recovered, dict):
        return None
    omitted = str(recovered.get("omitted_riu_id") or "")
    remaining_ids = {
        str(row.get("unit_id") or "")
        for row in selected
        if str(row.get("unit_id") or "") != omitted
    }
    if not omitted or len(remaining_ids) < min_units:
        return None

    support_by_claim: dict[str, list[str]] = {}
    for result in recovered.get("solver_results") or []:
        if not isinstance(result, dict) or result.get("passes") is not True:
            continue
        for claim in (result.get("judgment") or {}).get("claim_results") or []:
            if not isinstance(claim, dict) or claim.get("status") != "satisfied":
                continue
            support = [
                str(value)
                for value in claim.get("supporting_riu_ids") or []
                if str(value) in remaining_ids
            ]
            if support:
                support_by_claim.setdefault(
                    str(claim.get("claim_id") or ""), support
                )

    repaired_selection = deepcopy(selection)
    repaired_selection["selected_contexts"] = [
        row
        for row in repaired_selection.get("selected_contexts") or []
        if str(row.get("unit_id") or "") != omitted
    ]
    repaired_selection["source_qa_ids_used"] = sorted(
        {
            str(qa_id)
            for row in repaired_selection["selected_contexts"]
            for qa_id in row.get("source_qa_ids") or []
        }
    )
    repaired_plan = deepcopy(plan)
    changed_claims: list[str] = []
    for step in (repaired_plan.get("oracle") or {}).get("reasoning_steps") or []:
        if not isinstance(step, dict) or omitted not in {
            str(value) for value in step.get("uses") or []
        }:
            continue
        claim_id = str(step.get("claim_id") or "")
        replacement = support_by_claim.get(claim_id)
        if not replacement:
            return None
        step["uses"] = list(dict.fromkeys(replacement))
        changed_claims.append(claim_id)
    repaired_plan["oracle"]["required_decision_steps"] = deepcopy(
        repaired_plan["oracle"].get("reasoning_steps") or []
    )
    repaired_plan = normalize_free_response_plan(
        repaired_plan,
        repaired_selection,
    )
    return (
        repaired_selection,
        repaired_plan,
        {
            "removed_riu_id": omitted,
            "relinked_claim_ids": changed_claims,
            "basis_case_id": recovered.get("case_id"),
            "remaining_riu_ids": sorted(remaining_ids),
            "policy": "prune_recovered_loo_then_rerun_full_none_and_all_loo",
        },
    )


def _riu_specs(selection: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    steps = (plan.get("oracle") or {}).get("reasoning_steps") or []
    return [
        {
            "riu_id": str(row["unit_id"]),
            "canonical_fact": row.get("information_feature"),
            "text": row.get("text"),
            "source_context_ids": list(row.get("source_context_ids") or []),
            "entity_scope": row.get("entity_scope"),
            "period_scope": row.get("period_scope"),
            "reasoning_steps": [
                {
                    "claim_id": step.get("claim_id"),
                    "step": step.get("step"),
                    "claim": step.get("claim"),
                    "expected_answer_element": step.get(
                        "expected_answer_element"
                    ),
                    "grading_rule": step.get("grading_rule"),
                    "judge_focus": step.get("judge_focus"),
                }
                for step in steps
                if str(row["unit_id"]) in {
                    str(value) for value in step.get("uses") or []
                }
            ],
        }
        for row in selection.get("selected_contexts") or []
    ]


def _source_seed(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": candidate.get("source_id"),
        "subset": candidate.get("subset"),
        "source_qas_do_not_copy": [
            {
                key: qa.get(key)
                for key in (
                    "qa_id",
                    "subset",
                    "statement",
                    "entailment_label",
                    "execution_result",
                    "construction_knowledge",
                )
            }
            for qa in candidate.get("source_qas") or []
            if isinstance(qa, dict)
        ],
        "report": candidate.get("report_name"),
    }


def _public_information(
    candidate: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    # Native knowledge annotations are constructor evidence, never common input.
    return {"source_knowledge": [], "report_type": candidate.get("report_type")}


def _stage(
    name: str,
    client: Any,
    response: dict[str, Any],
    errors: list[str],
    *,
    attempt: int | None = None,
    audit_retry: int | None = None,
) -> dict[str, Any]:
    return {
        "stage": name,
        "attempt": attempt,
        "audit_retry": audit_retry,
        "model": client.cfg.model,
        "valid": not errors,
        "errors": errors,
        "response": response,
        "request_meta": dict(getattr(client, "last_request_meta", {}) or {}),
    }


def _feedback(
    stage: str,
    errors: list[str],
    *,
    action: str,
    previous_selection: dict[str, Any] | None = None,
    previous_task: dict[str, Any] | None = None,
    audits: dict[str, Any] | None = None,
) -> str:
    return json.dumps(
        {
            "rejected_stage": stage,
            "required_action": action,
            "errors": errors,
            "previous_selection": _compact_selection(previous_selection),
            "previous_task": previous_task,
            "audit_summary": _compact_audit_summary(audits or {}),
            "instruction": "Generate a new task that fixes every listed error.",
        },
        ensure_ascii=False,
        indent=2,
    )


def _compact_selection(selection: dict[str, Any] | None) -> dict[str, Any] | None:
    if not selection:
        return None
    return {
        "joint_question_theme": selection.get("joint_question_theme"),
        "target_operation": selection.get("target_operation"),
        "selected_contexts": [
            {
                key: row.get(key)
                for key in (
                    "unit_id",
                    "candidate_block_id",
                    "context_id",
                    "source_context_ids",
                    "information_feature",
                    "entity_scope",
                    "period_scope",
                    "why_critical",
                )
            }
            for row in selection.get("selected_contexts") or []
            if isinstance(row, dict)
        ],
    }


def _compact_audit_summary(audits: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    grounding = audits.get("grounding")
    if isinstance(grounding, dict):
        summary["grounding"] = {
            key: grounding.get(key)
            for key in (
                "valid",
                "question_answerable_from_full_riu_set",
                "short_answer_correct",
                "all_selected_rius_necessary",
                "answer_spec_correct",
                "unsupported_claims",
                "failed_reason",
                "notes",
            )
        }
    for panel_name in ("public_only", "riu_boundary"):
        panel = audits.get(panel_name)
        if not isinstance(panel, dict):
            continue
        summary[panel_name] = [
            {
                "case_id": row.get("case_id"),
                "is_full_set": row.get("is_full_set"),
                "available_riu_ids": row.get("available_riu_ids"),
                "pass_count": row.get("pass_count"),
                "judgments": [
                    {
                        "auditor_id": result.get("auditor_id"),
                        "passes": result.get("passes"),
                        "reason": (result.get("judgment") or {}).get("reason"),
                        "missing_claim_ids": [
                            claim.get("claim_id")
                            for claim in (
                                (result.get("judgment") or {}).get(
                                    "claim_results"
                                )
                                or []
                            )
                            if claim.get("status") != "satisfied"
                        ],
                    }
                    for result in row.get("solver_results") or []
                    if isinstance(result, dict)
                ],
            }
            for row in panel.get("case_results") or []
            if isinstance(row, dict)
        ]
    packet_rows = audits.get("packet_leakage")
    if isinstance(packet_rows, list):
        summary["packet_leakage"] = [
            {
                "participant_id": row.get("participant_id"),
                "unassigned_rui_ids": row.get("unassigned_rui_ids"),
                "leaking_context_ids": row.get("leaking_context_ids"),
                "errors": row.get("errors"),
            }
            for row in packet_rows
            if isinstance(row, dict)
        ]
    return summary


def _result(
    candidate_index: int,
    candidate: dict[str, Any],
    status: str,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "candidate_index": candidate_index,
        "source_id": candidate.get("source_id"),
        "subset": candidate.get("subset"),
        "report": candidate.get("report_name"),
        "status": status,
        "errors": errors,
    }


def _byproduct(
    candidate_index: int,
    candidate: dict[str, Any],
    selection: dict[str, Any] | None,
    plan: dict[str, Any] | None,
    audits: dict[str, Any],
    stages: list[dict[str, Any]],
    *,
    question_generate_stage: str = QUESTION_GENERATE_STAGE,
    accepted_attempt: int | None = None,
) -> dict[str, Any]:
    return {
        "schema": "findver_free_response_generation_byproduct_v1",
        "candidate_index": candidate_index,
        "source_id": candidate.get("source_id"),
        "subset": candidate.get("subset"),
        "source": {
            "report": candidate.get("report_name"),
            "candidate_context_ids": candidate.get("candidate_context_ids"),
            "source_qa_ids": [
                qa.get("qa_id")
                for qa in candidate.get("source_qas") or []
                if isinstance(qa, dict)
            ],
            "qa_evidence_mappings": candidate.get("qa_evidence_mappings"),
            "evidence_consolidation": candidate.get("evidence_consolidation"),
            "candidate_relations": candidate.get("candidate_relations"),
            "normalized_candidate_blocks": candidate.get("candidate_contexts"),
        },
        "selection": selection,
        "plan": plan,
        "audits": audits,
        "accepted_attempt": accepted_attempt,
        "stage_records": stages,
        "prompt_files": _prompt_manifest(question_generate_stage),
    }


def _terminal(
    candidate_index: int,
    candidate: dict[str, Any],
    status: str,
    errors: list[str],
    selection: dict[str, Any] | None,
    plan: dict[str, Any] | None,
    audits: dict[str, Any],
    stages: list[dict[str, Any]],
    question_generate_stage: str = QUESTION_GENERATE_STAGE,
) -> tuple[dict[str, Any], None, dict[str, Any]]:
    return (
        _result(candidate_index, candidate, status, errors),
        None,
        _byproduct(
            candidate_index,
            candidate,
            selection,
            plan,
            audits,
            stages,
            question_generate_stage=question_generate_stage,
        ),
    )


def _write_progress(
    output: Path,
    results: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    byproducts: list[dict[str, Any]],
) -> None:
    write_jsonl(output / "candidate_results.jsonl", results)
    write_jsonl(output / "scenarios.jsonl", scenarios)
    write_jsonl(output / "generation_byproducts.jsonl", byproducts)
    progress = {
        "schema": "findver_free_response_progress_v1",
        "candidate_count": len(results),
        "accepted_count": len(scenarios),
        "funnel_counts": _status_counts(results),
    }
    write_json(output / "progress_report.json", progress)
    print(json.dumps({"progress": progress}, ensure_ascii=False), flush=True)


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _prompt_manifest(
    question_generate_stage: str = QUESTION_GENERATE_STAGE,
) -> list[str]:
    return [
        SYSTEM_PROMPT,
        "finance/findver/free_qa_evidence_map_user.txt",
        "finance/findver/free_evidence_consolidate_user.txt",
        f"finance/findver/{question_generate_stage}.txt",
        "finance/findver/free_grounding_audit_user.txt",
        "finance/findver/free_objective_disclosure_audit_user.txt",
        "finance/findver/free_solver_system.txt",
        "finance/findver/free_solver_user.txt",
        "finance/findver/free_response_judge_user.txt",
        "finance/findver/free_packet_leakage_audit_user.txt",
    ]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-path", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--subset", choices=SUBSETS, default="all")
    parser.add_argument("--source-split", default="testmini")
    parser.add_argument("--participant-context-token-budget", type=int, default=20000)
    parser.add_argument("--allocation-mode", choices=("auto", *PAPER_ALLOCATION_MODES), default="one_each")
    parser.add_argument("--min-participants", type=int, default=2)
    parser.add_argument("--min-units", type=int, default=2)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument("--candidate-start-index", type=int, default=1)
    parser.add_argument(
        "--generation-attempts",
        "--plan-attempts",
        dest="plan_attempts",
        type=int,
        default=3,
        help="Maximum combined PIU-selection/task-generation attempts per report.",
    )
    parser.add_argument("--audit-attempts", type=int, default=3)
    parser.add_argument("--blind-solver-count", type=int, default=2)
    parser.add_argument(
        "--full-boundary-min-passes",
        type=int,
        default=0,
        help="Required full-evidence solver passes; 0 requires every blind solver.",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--question-generate-stage",
        default=QUESTION_GENERATE_STAGE,
        help=(
            "Prompt filename stem under finance/findver/ for PIU selection and "
            "task generation."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--endpoint", default="/responses", choices=["/responses"])
    parser.add_argument("--api-key-env", default="AWS_BEARER_TOKEN_BEDROCK")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--credentials-file")
    parser.add_argument("--max-output-tokens", type=int, default=8_000)
    parser.add_argument(
        "--blind-solver-max-output-tokens",
        type=int,
        default=5_000,
        help="Maximum output tokens for each blind solver call.",
    )
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--request-retry-attempts", type=int, default=3)
    return parser.parse_args()


def _client(
    args: argparse.Namespace,
    *,
    request_log_path: Path,
    max_output_tokens: int | None = None,
) -> OpenAIJsonClient:
    return OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider="bedrock_openai",
            model=args.model,
            base_url=args.base_url,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            env_file=args.env_file,
            credentials_file=args.credentials_file,
            temperature=None,
            max_output_tokens=max_output_tokens or args.max_output_tokens,
            timeout_sec=args.timeout_sec,
            request_retry_attempts=args.request_retry_attempts,
            request_log_path=str(request_log_path),
        )
    )


def main() -> None:
    args = _parse_args()
    output = Path(args.output_root).expanduser().resolve()
    generator = _client(
        args, request_log_path=output / "_llm_requests.generator.jsonl"
    )
    auditor = _client(
        args, request_log_path=output / "_llm_requests.auditor.jsonl"
    )
    blind_clients = [
        (
            f"blind_solver_{index}",
            _client(
                args,
                request_log_path=output
                / f"_llm_requests.blind_solver_{index}.jsonl",
                max_output_tokens=min(
                    args.max_output_tokens,
                    args.blind_solver_max_output_tokens,
                ),
            ),
        )
        for index in range(1, max(1, args.blind_solver_count) + 1)
    ]
    report = build_dataset(
        qa_path=args.qa_path,
        reports_root=args.reports_root,
        output_root=output,
        generator_client=generator,
        audit_client=auditor,
        blind_clients=blind_clients,
        subset=args.subset,
        allocation_mode=args.allocation_mode,
        min_participants=args.min_participants,
        min_units=args.min_units,
        max_units=args.max_units,
        candidate_limit=args.candidate_limit,
        candidate_start_index=args.candidate_start_index,
        plan_attempts=args.plan_attempts,
        audit_attempts=args.audit_attempts,
        full_boundary_min_passes=(
            None
            if args.full_boundary_min_passes <= 0
            else args.full_boundary_min_passes
        ),
        seed=args.seed,
        question_generate_stage=args.question_generate_stage,
        source_split=args.source_split,
        participant_context_token_budget=args.participant_context_token_budget,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("accepted_count"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
