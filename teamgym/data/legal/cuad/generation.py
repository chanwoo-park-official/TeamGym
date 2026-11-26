"""CUAD measured direct-span sufficiency funnel."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable

from teamgym.core.io import compact_text, prompt_text, write_json, write_jsonl

from ....llm import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)
from ..contracts.decisions import (
    validate_decision_relevance,
    _has_all_or_both_marker as _has_all_or_both_marker,
    _letter_id as _letter_id,
    _option_component_score as _option_component_score,
    _option_text as _option_text,
    _option_value_component_score as _option_value_component_score,
    normalize_decision_set as _normalize_decision_set,
    normalize_smallest_sufficient_sets as _smallest_sets,
    validate_decision_set_balance as _validate_decision_set_balance,
    validate_option_value_names as _validate_option_value_names,
)
from ..contracts.generation_records import split_generation_byproducts
from ..converter_utils import call_legal_planner, compact_json
from ..prepartition_export import write_pre_partition_task_tuples
from ..question_history import (
    load_source_question_history,
    question_attempt_entry,
    question_history_prompt,
    summarize_question_attempt,
)
from ..sources import cuad_candidates

MEASURED_PLAN_ATTEMPTS = 3


def run_measured_span_funnel(
    *,
    output_root: str | Path,
    cache_dir: str | Path,
    llm_client: OpenAIJsonClient,
    min_accept_size: int,
    max_span_count: int,
    max_candidates: int,
    target_accepted: int,
    candidate_start_index: int = 1,
    min_positive_spans: int = 0,
    parallelism: int = 1,
    trace_root: str | Path | None = None,
    save_generation_byproducts: bool = True,
    question_history_path: str | Path | None = None,
) -> dict[str, Any]:
    if min_accept_size < 1:
        raise ValueError("min_accept_size must be >= 1.")
    if max_span_count < min_accept_size:
        raise ValueError("max_span_count must be >= min_accept_size.")
    if candidate_start_index < 1:
        raise ValueError("candidate_start_index must be >= 1.")
    if min_positive_spans < 0:
        raise ValueError("min_positive_spans must be >= 0.")
    if parallelism < 1:
        raise ValueError("parallelism must be >= 1.")
    output_root_path = Path(output_root).expanduser().resolve()
    trace_root_path = Path(trace_root).expanduser().resolve() if trace_root else output_root_path / "_llm_traces"
    source_question_history = load_source_question_history(question_history_path)
    if parallelism == 1:
        rows, accepted = _run_candidates_sequential(
            cache_dir=cache_dir,
            llm_client=llm_client,
            min_accept_size=min_accept_size,
            max_span_count=max_span_count,
            max_candidates=max_candidates,
            target_accepted=target_accepted,
            candidate_start_index=candidate_start_index,
            min_positive_spans=min_positive_spans,
            trace_root=trace_root_path,
            source_question_history=source_question_history,
        )
        target_stop_mode = "exact"
    else:
        rows, accepted = _run_candidates_parallel(
            cache_dir=cache_dir,
            llm_client_config=llm_client.cfg,
            min_accept_size=min_accept_size,
            max_span_count=max_span_count,
            max_candidates=max_candidates,
            target_accepted=target_accepted,
            candidate_start_index=candidate_start_index,
            min_positive_spans=min_positive_spans,
            trace_root=trace_root_path,
            parallelism=parallelism,
            source_question_history=source_question_history,
        )
        target_stop_mode = "soft_in_flight_candidates_may_exceed_target"

    accepted_to_write, generation_byproducts = split_generation_byproducts(
        accepted,
        min_accept_size=min_accept_size,
        save_generation_byproducts=save_generation_byproducts,
    )
    byproduct_path = output_root_path / f"generation_byproducts_minaccept{min_accept_size}.jsonl"
    question_attempt_history_path = output_root_path / f"question_attempt_history_minaccept{min_accept_size}.jsonl"
    pre_partition_task_tuple_paths = write_pre_partition_task_tuples(
        output_root=output_root_path,
        dataset="cuad",
        min_accept_size=min_accept_size,
        accepted_trials=accepted_to_write,
    )

    report = {
        "schema": "teamgym_cuad_measured_span_funnel_v1",
        "output_root": str(output_root_path),
        "trace_root": str(trace_root_path),
        "min_accept_size": min_accept_size,
        "max_span_count": max_span_count,
        "max_candidates": max_candidates,
        "target_accepted": target_accepted,
        "candidate_start_index": candidate_start_index,
        "min_positive_spans": min_positive_spans,
        "parallelism": parallelism,
        "question_history_path": str(Path(question_history_path).expanduser()) if question_history_path else None,
        "target_stop_mode": target_stop_mode,
        "save_generation_byproducts": save_generation_byproducts,
        "generation_byproduct_path": str(byproduct_path) if save_generation_byproducts else None,
        "question_attempt_history_path": str(question_attempt_history_path),
        "pre_partition_task_tuple_paths": pre_partition_task_tuple_paths,
        "accepted_count": len(accepted),
        "candidate_count_seen": len(rows),
        "funnel": _funnel_counts(rows),
        "rows": rows,
    }
    write_json(output_root_path / f"funnel_measured_span_minaccept{min_accept_size}.json", report)
    write_jsonl(output_root_path / f"accepted_measured_span_minaccept{min_accept_size}.jsonl", accepted_to_write)
    write_json(output_root_path / f"accepted_measured_span_minaccept{min_accept_size}.pretty.json", accepted_to_write)
    if save_generation_byproducts:
        write_jsonl(byproduct_path, generation_byproducts)
    write_jsonl(question_attempt_history_path, [summarize_question_attempt(row) for row in rows])
    return report


def _run_candidates_sequential(
    *,
    cache_dir: str | Path,
    llm_client: OpenAIJsonClient,
    min_accept_size: int,
    max_span_count: int,
    max_candidates: int,
    target_accepted: int,
    candidate_start_index: int,
    min_positive_spans: int,
    trace_root: Path,
    source_question_history: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in _iter_candidate_items(
        cache_dir=cache_dir,
        candidate_start_index=candidate_start_index,
        min_positive_spans=min_positive_spans,
        max_candidates=max_candidates,
    ):
        if len(accepted) >= target_accepted:
            break
        row, accepted_plan = _try_candidate(
            candidate=candidate,
            candidate_index=candidate_index,
            min_accept_size=min_accept_size,
            max_span_count=max_span_count,
            llm_client=llm_client,
            trace_root=trace_root,
            source_question_history=source_question_history.get(str(candidate.get("source_id") or ""), []),
        )
        rows.append(row)
        if accepted_plan is not None:
            accepted.append(accepted_plan)
    return rows, accepted


def _run_candidates_parallel(
    *,
    cache_dir: str | Path,
    llm_client_config: OpenAIJsonClientConfig,
    min_accept_size: int,
    max_span_count: int,
    max_candidates: int,
    target_accepted: int,
    candidate_start_index: int,
    min_positive_spans: int,
    trace_root: Path,
    parallelism: int,
    source_question_history: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    candidate_iter = iter(
        _iter_candidate_items(
            cache_dir=cache_dir,
            candidate_start_index=candidate_start_index,
            min_positive_spans=min_positive_spans,
            max_candidates=max_candidates,
        )
    )
    pending: dict[Future[tuple[dict[str, Any], dict[str, Any] | None]], int] = {}
    stop_scheduling = False

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            candidate_index, candidate = next(candidate_iter)
        except StopIteration:
            return False
        client = OpenAIJsonClient(llm_client_config)
        future = executor.submit(
            _try_candidate,
            candidate=candidate,
            candidate_index=candidate_index,
            min_accept_size=min_accept_size,
            max_span_count=max_span_count,
            llm_client=client,
            trace_root=trace_root,
            source_question_history=source_question_history.get(str(candidate.get("source_id") or ""), []),
        )
        pending[future] = candidate_index
        return True

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        for _ in range(parallelism):
            if not submit_next(executor):
                break

        while pending:
            for future in as_completed(list(pending)):
                pending.pop(future)
                row, accepted_plan = future.result()
                rows.append(row)
                if accepted_plan is not None:
                    accepted.append(accepted_plan)
                    if len(accepted) >= target_accepted:
                        stop_scheduling = True
                if not stop_scheduling:
                    submit_next(executor)
                break

    rows.sort(key=lambda row: int(row.get("candidate_index") or 0))
    accepted.sort(key=lambda item: int(item.get("candidate_index") or 0))
    return rows, accepted


def _iter_candidate_items(
    *,
    cache_dir: str | Path,
    candidate_start_index: int,
    min_positive_spans: int,
    max_candidates: int,
) -> Iterable[tuple[int, dict[str, Any]]]:
    yielded = 0
    for candidate_index, candidate in enumerate(cuad_candidates(cache_dir), start=1):
        if candidate_index < candidate_start_index:
            continue
        if min_positive_spans and len(candidate.get("positive_spans") or []) < min_positive_spans:
            continue
        if yielded >= max_candidates:
            break
        yielded += 1
        yield candidate_index, candidate


def _try_candidate(
    *,
    candidate: dict[str, Any],
    candidate_index: int,
    min_accept_size: int,
    max_span_count: int,
    llm_client: OpenAIJsonClient,
    trace_root: Path,
    source_question_history: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    instance_id = f"cuad_measured_span_{candidate_index:04d}_{candidate['contract_index']:04d}_{_slug(candidate['label_family'])}"
    row = _candidate_row(candidate_index, candidate, "started", [])
    row["instance_id"] = instance_id
    prior_question_history = list(source_question_history or [])
    row["source_question_history_count"] = len(prior_question_history)

    selection = call_legal_planner(
        dataset="cuad",
        prompt_name="measured_span_select_user",
        variables={
            "CONTRACT_TITLE": candidate["contract_title"],
            "LABEL_FAMILY": candidate["label_family"],
            "LABEL_DETAILS": candidate.get("label_details", ""),
            "POSITIVE_SPANS_JSON": candidate["positive_spans"],
            "CONTRACT_TEXT_EXCERPT": compact_text(candidate["contract_text"], max_chars=12000),
            "MIN_ACCEPT_SIZE": str(min_accept_size),
            "MAX_SPAN_COUNT": str(max_span_count),
            "SELECTION_EXAMPLES": _prompt_example_text("legal/cuad/measured_span_select_examples.txt"),
        },
        llm_client=llm_client,
        trace_root=trace_root,
        instance_id=instance_id,
    )
    row["selection_request_meta"] = dict(llm_client.last_request_meta)
    selection_errors = _validate_selection(
        selection,
        candidate,
        min_accept_size=min_accept_size,
        max_span_count=max_span_count,
    )
    row["selection_summary"] = _selection_summary(selection)
    if selection_errors:
        row["status"] = "selection_local_failed"
        row["errors"] = selection_errors
        return row, None
    row["selection_local_valid"] = True

    feedback = "None."
    rejection_history: list[dict[str, Any]] = []
    row["plan_attempts"] = []
    for plan_attempt in range(1, MEASURED_PLAN_ATTEMPTS + 1):
        plan = call_legal_planner(
            dataset="cuad",
            prompt_name="measured_span_question_user",
            variables={
                "CONTRACT_TITLE": candidate["contract_title"],
                "LABEL_FAMILY": candidate["label_family"],
                "LABEL_DETAILS": candidate.get("label_details", ""),
                "POSITIVE_SPANS_JSON": candidate["positive_spans"],
                "SELECTED_SPAN_BASIS_JSON": selection,
                "QUESTION_EXAMPLES": _prompt_example_text("legal/cuad/measured_span_question_examples.txt"),
                "SUGGESTED_OPTION_COUNT": str(_suggested_option_count(candidate_index)),
                "SOURCE_QUESTION_HISTORY": question_history_prompt(prior_question_history),
                "REJECTION_FEEDBACK": feedback,
            },
            llm_client=llm_client,
            trace_root=trace_root,
            instance_id=instance_id,
        )
        plan_meta = dict(llm_client.last_request_meta)
        plan = _normalize_decision_set(plan)
        plan = _normalize_plan_against_selection(plan, selection)
        plan_errors = _validate_plan(plan, selection, candidate)
        row["plan_request_meta"] = plan_meta
        row["plan_summary"] = _plan_summary(plan)
        row["plan_attempts"].append(
            {
                "attempt": plan_attempt,
                "stage": "planner_local_validation",
                "request_meta": plan_meta,
                "errors": plan_errors,
                "summary": _plan_summary(plan),
            }
        )
        if plan_errors:
            rejection_history.append(
                {
                    "attempt": plan_attempt,
                    "stage": "planner_local_validation",
                    "errors": plan_errors,
                    "previous_attempt_summary": _plan_feedback_summary(plan),
                }
            )
            prior_question_history.append(
                question_attempt_entry(
                    source_id=str(candidate.get("source_id") or ""),
                    instance_id=instance_id,
                    candidate_index=candidate_index,
                    status="rejected_planner_local_validation",
                    plan=plan,
                    errors=plan_errors,
                )
            )
            feedback = _measured_feedback_from_errors(plan_errors, plan=plan)
            continue
        row["plan_local_valid"] = True

        audit = call_legal_planner(
            dataset="cuad",
            prompt_name="measured_span_sufficiency_audit_user",
            variables={
                "LABEL_FAMILY": candidate["label_family"],
                "LABEL_DETAILS": candidate.get("label_details", ""),
                "POSITIVE_SPANS_JSON": candidate["positive_spans"],
                "PLAN_JSON": compact_json(plan, max_chars=40000),
                "MIN_ACCEPT_SIZE": str(min_accept_size),
                "AUDIT_EXAMPLES": _prompt_example_text("legal/cuad/measured_span_sufficiency_audit_examples.txt"),
            },
            llm_client=llm_client,
            trace_root=trace_root,
            instance_id=instance_id,
        )
        audit_meta = dict(llm_client.last_request_meta)
        row["audit_request_meta"] = audit_meta
        audit_errors = _validate_audit(audit, plan, min_accept_size=min_accept_size)
        row["audit_summary"] = _audit_summary(audit)
        row["plan_attempts"][-1]["audit_request_meta"] = audit_meta
        row["plan_attempts"][-1]["audit_errors"] = audit_errors
        row["plan_attempts"][-1]["audit_summary"] = _audit_summary(audit)
        if audit_errors:
            rejection_history.append(
                {
                    "attempt": plan_attempt,
                    "stage": "sufficiency_audit",
                    "errors": audit_errors,
                    "previous_attempt_summary": _plan_feedback_summary(plan),
                    "audit": audit,
                }
            )
            prior_question_history.append(
                question_attempt_entry(
                    source_id=str(candidate.get("source_id") or ""),
                    instance_id=instance_id,
                    candidate_index=candidate_index,
                    status="rejected_sufficiency_audit",
                    plan=plan,
                    errors=audit_errors,
                )
            )
            feedback = _measured_feedback_from_errors(audit_errors, plan=plan, audit=audit)
            continue

        try:
            final_selection, final_plan, post_audit_trimming = _trim_to_smallest_sufficient_span_set(
                selection,
                plan,
                audit,
            )
        except ValueError as exc:
            trim_errors = [str(exc)]
            row["plan_attempts"][-1]["post_audit_trim_errors"] = trim_errors
            rejection_history.append(
                {
                    "attempt": plan_attempt,
                    "stage": "post_audit_minimal_set_trim",
                    "errors": trim_errors,
                    "previous_attempt_summary": _plan_feedback_summary(plan),
                    "audit": audit,
                }
            )
            feedback = _measured_feedback_from_errors(trim_errors, plan=plan, audit=audit)
            continue
        row["selection_summary"] = _selection_summary(final_selection)
        row["plan_summary"] = _plan_summary(final_plan)
        row["post_audit_trimming"] = post_audit_trimming
        row["status"] = "accepted"
        row["errors"] = []
        row["rejection_history"] = rejection_history
        generation_metadata = {
            "method": "cuad_measured_direct_span_generation",
            "no_synthetic_fallback": True,
            "prompt_stage_count": 3,
            "acceptance_mode": "trim_to_smallest_sufficient_span_set",
            "thresholds": {
                "min_accept_size": min_accept_size,
                "max_span_count": max_span_count,
            },
            "planning_attempts": plan_attempt,
            "validation_path": [
                "selection_local_validation",
                "planner_local_validation",
                "llm_sufficiency_audit",
                "post_audit_minimal_set_trim",
            ],
        }
        generation_byproduct = {
            "schema": "cuad_measured_span_generation_byproduct_v1",
            "instance_id": instance_id,
            "candidate_index": candidate_index,
            "source_id": candidate["source_id"],
            "source_candidate": {
                "contract_index": candidate.get("contract_index"),
                "contract_title": candidate.get("contract_title"),
                "label_family": candidate.get("label_family"),
                "positive_span_count": len(candidate.get("positive_spans") or []),
            },
            "prompt_template_ids": [
                "legal/cuad/measured_span_select_user",
                "legal/cuad/measured_span_question_user",
                "legal/cuad/measured_span_sufficiency_audit_user",
            ],
            "request_metadata": {
                "selection": row.get("selection_request_meta") or {},
                "accepted_plan": row.get("plan_request_meta") or {},
                "accepted_audit": row.get("audit_request_meta") or {},
            },
            "planning_attempts": plan_attempt,
            "source_question_history_used_count": row.get("source_question_history_count") or 0,
            "plan_attempt_records": row.get("plan_attempts") or [],
            "rejection_history": rejection_history,
            "selection_summary": row.get("selection_summary") or {},
            "plan_summary": row.get("plan_summary") or {},
            "audit_summary": row.get("audit_summary") or {},
            "post_audit_trimming": post_audit_trimming,
        }
        return row, {
            "trial_schema": "cuad_measured_span_trial_plan_v1",
            "instance_id": instance_id,
            "candidate_index": candidate_index,
            "source_id": candidate["source_id"],
            "label_family": candidate["label_family"],
            "min_accept_size": min_accept_size,
            "max_span_count": max_span_count,
            "selection": final_selection,
            "plan": final_plan,
            "audit": audit,
            "post_audit_trimming": post_audit_trimming,
            "planning_attempts": plan_attempt,
            "generation_metadata": generation_metadata,
            "_generation_byproduct": generation_byproduct,
        }

    last_rejection = rejection_history[-1] if rejection_history else {}
    row["status"] = "audit_failed" if last_rejection.get("stage") == "sufficiency_audit" else "plan_local_failed"
    row["errors"] = list(last_rejection.get("errors") or ["planning failed without a recorded validation error."])
    row["rejection_history"] = rejection_history
    return row, None


def _validate_selection(
    selection: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_accept_size: int,
    max_span_count: int,
) -> list[str]:
    errors: list[str] = []
    if "atomic_information_units" in selection:
        errors.append("selection must not include atomic_information_units.")
    spans_by_index = _spans_by_index(candidate)
    assessment = selection.get("candidate_assessment") if isinstance(selection.get("candidate_assessment"), dict) else {}
    if assessment.get("feasible") is not True:
        errors.append(f"selection marked infeasible: {assessment.get('rationale') or assessment.get('reason')}")
    classifications = selection.get("source_span_classification")
    if not isinstance(classifications, list):
        errors.append("source_span_classification must be a list.")
    else:
        classified = {_as_int(row.get("span_index")) for row in classifications if isinstance(row, dict)}
        missing = set(spans_by_index) - {idx for idx in classified if idx is not None}
        if missing:
            errors.append(f"source_span_classification missing spans {sorted(missing)}.")
    selected = selection.get("selected_span_basis")
    if not isinstance(selected, list):
        errors.append("selected_span_basis must be a list.")
        selected = []
    if not (min_accept_size <= len(selected) <= max_span_count):
        errors.append(
            f"selected_span_basis must contain {min_accept_size} to {max_span_count} direct spans."
        )
    unit_ids: list[str] = []
    indices: list[int] = []
    for item in selected:
        if not isinstance(item, dict):
            errors.append("each selected_span_basis row must be an object.")
            continue
        unit_id = str(item.get("unit_id") or "")
        unit_ids.append(unit_id)
        span_index = _as_int(item.get("source_span_index"))
        indices.append(span_index or -1)
        if not unit_id:
            errors.append("each selected span needs unit_id.")
        if span_index not in spans_by_index:
            errors.append(f"{unit_id}: source_span_index must reference a supplied positive span.")
        elif _norm(item.get("text")) != _norm(spans_by_index[span_index].get("text")):
            errors.append(f"{unit_id}: text must exactly equal supplied span {span_index}.")
        if not str(item.get("legal_feature") or "").strip():
            errors.append(f"{unit_id}: legal_feature is required.")
        if not str(item.get("possible_decision_boundary") or "").strip():
            errors.append(f"{unit_id}: possible_decision_boundary is required.")
    if len(set(unit_ids)) != len(unit_ids):
        errors.append("selected span unit IDs must be unique.")
    if len(set(indices)) != len(indices):
        errors.append("selected source spans must be unique.")
    return errors


def _validate_plan(plan: dict[str, Any], selection: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    selected = selection.get("selected_span_basis") if isinstance(selection.get("selected_span_basis"), list) else []
    selected_ids = {str(row.get("unit_id")) for row in selected if isinstance(row, dict)}
    selected_text_by_id = {
        str(row.get("unit_id")): str(row.get("text") or "")
        for row in selected
        if isinstance(row, dict)
    }
    selected_span_by_id = {
        str(row.get("unit_id")): _as_int(row.get("source_span_index"))
        for row in selected
        if isinstance(row, dict)
    }
    public_task = plan.get("public_task") if isinstance(plan.get("public_task"), dict) else {}
    decision_set = public_task.get("decision_set")
    if not isinstance(decision_set, list) or not (2 <= len(decision_set) <= 6):
        errors.append("public_task.decision_set must contain 2 to 6 choices.")
        decision_values: set[str] = set()
    else:
        decision_values = {str(item) for item in decision_set}
        if len(decision_values) != len(decision_set):
            errors.append("public_task.decision_set choices must be unique.")
    if not str(public_task.get("objective") or "").strip():
        errors.append("public_task.objective is required.")
    oracle = plan.get("oracle") if isinstance(plan.get("oracle"), dict) else {}
    if str(oracle.get("answer") or "") not in decision_values:
        errors.append("oracle.answer must exactly match one generated decision_set value.")
    errors.extend(_validate_option_value_names(decision_values))
    errors.extend(_validate_decision_set_balance(public_task, str(oracle.get("answer") or "")))

    units = plan.get("evidence_units")
    if not isinstance(units, list):
        errors.append("evidence_units must be a list.")
        units = []
    unit_ids = {str(unit.get("unit_id")) for unit in units if isinstance(unit, dict)}
    if unit_ids != selected_ids:
        errors.append("evidence_units must correspond exactly to selected_span_basis IDs.")
    for unit in units:
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("unit_id") or "")
        if _norm(unit.get("text")) != _norm(selected_text_by_id.get(unit_id, "")):
            errors.append(f"{unit_id}: evidence text must equal selected CUAD span text.")
        source_ref = unit.get("source_ref") if isinstance(unit.get("source_ref"), dict) else {}
        if _as_int(source_ref.get("span_index")) != selected_span_by_id.get(unit_id):
            errors.append(f"{unit_id}: source_ref.span_index must match selected source span.")
        if str(unit.get("criticality", "")).lower() != "critical":
            errors.append(f"{unit_id}: criticality must be critical.")
        if str(unit.get("information_unit_type") or "") != "reference_information_unit":
            errors.append(f"{unit_id}: information_unit_type must be reference_information_unit.")
        errors.extend(validate_decision_relevance(unit_id, unit.get("decision_relevance"), decision_values))

    participants = plan.get("participants")
    if not isinstance(participants, list) or len(participants) < 2:
        errors.append("participants must contain at least two packets.")
        participants = []
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        refs = {str(item) for item in participant.get("evidence_unit_ids", [])}
        if not refs:
            errors.append(f"{participant.get('participant_id')}: participant packet cannot be empty.")
        if refs and not refs <= selected_ids:
            errors.append(f"{participant.get('participant_id')}: participant references unknown spans.")
        if selected_ids and selected_ids <= refs:
            errors.append(f"{participant.get('participant_id')}: participant has all selected spans.")

    spans_by_index = _spans_by_index(candidate)
    assessment = plan.get("candidate_assessment") if isinstance(plan.get("candidate_assessment"), dict) else {}
    excluded = assessment.get("excluded_spans") if isinstance(assessment.get("excluded_spans"), list) else []
    selected_indices = {idx for idx in selected_span_by_id.values() if idx is not None}
    excluded_indices = {_as_int(row.get("span_index")) for row in excluded if isinstance(row, dict)}
    missing = set(spans_by_index) - selected_indices - {idx for idx in excluded_indices if idx is not None}
    if missing:
        errors.append(f"candidate_assessment must account for every supplied span; missing {sorted(missing)}.")
    return errors


def _validate_audit(audit: dict[str, Any], plan: dict[str, Any], *, min_accept_size: int) -> list[str]:
    errors: list[str] = []
    units = plan.get("evidence_units") if isinstance(plan.get("evidence_units"), list) else []
    unit_ids = {str(unit.get("unit_id")) for unit in units if isinstance(unit, dict)}
    oracle = plan.get("oracle") if isinstance(plan.get("oracle"), dict) else {}
    oracle_answer = str(oracle.get("answer") or "")
    if audit.get("valid") is not True:
        errors.append("audit did not mark the plan valid.")
    if audit.get("accept_under_threshold") is not True:
        errors.append("audit did not accept under the measured threshold.")
    if audit.get("copied_from_positive_spans") is not True:
        errors.append("audit did not confirm positive-span provenance.")
    if audit.get("full_span_set_sufficient") is not True:
        errors.append("audit did not confirm full selected span set sufficiency.")
    if audit.get("single_unit_shortcut") is True:
        errors.append("audit found a single-unit shortcut.")
    if audit.get("single_participant_shortcut") is True:
        errors.append("audit found a single-participant shortcut.")
    min_size = _as_int(audit.get("minimum_sufficient_span_set_size"))
    if min_size is None:
        errors.append("audit must report minimum_sufficient_span_set_size.")
    elif min_size < min_accept_size:
        errors.append(f"minimum sufficient span set size {min_size} is below threshold {min_accept_size}.")
    smallest = _smallest_sets(audit.get("smallest_sufficient_span_sets", []))
    if not smallest:
        errors.append("audit must report smallest_sufficient_span_sets.")
    for row in smallest:
        row_ids = set(row["unit_ids"])
        if not row_ids <= unit_ids:
            errors.append(f"smallest_sufficient_span_sets references unknown spans: {sorted(row_ids - unit_ids)}.")
        if oracle_answer and str(row.get("answer") or "") != oracle_answer:
            errors.append("smallest_sufficient_span_sets must be for the exact oracle answer.")
        if len(row_ids) < min_accept_size:
            errors.append("smallest_sufficient_span_sets contains a set below threshold.")
    errors.extend(_local_participant_smallest_set_errors(plan, smallest))
    return errors


def _local_participant_smallest_set_errors(plan: dict[str, Any], smallest: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    participants = plan.get("participants") if isinstance(plan.get("participants"), list) else []
    smallest_sets = [set(row["unit_ids"]) for row in smallest]
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        refs = {str(item) for item in participant.get("evidence_unit_ids", [])}
        for span_set in smallest_sets:
            if span_set and span_set <= refs:
                errors.append(f"{participant.get('participant_id')}: participant has a smallest sufficient span set.")
                break
    return errors


def _candidate_row(candidate_index: int, candidate: dict[str, Any], status: str, errors: list[str]) -> dict[str, Any]:
    return {
        "candidate_index": candidate_index,
        "source_id": candidate.get("source_id"),
        "contract_index": candidate.get("contract_index"),
        "contract_title": candidate.get("contract_title"),
        "label_family": candidate.get("label_family"),
        "positive_span_count": len(candidate.get("positive_spans") or []),
        "status": status,
        "errors": errors,
    }


def _selection_summary(selection: dict[str, Any]) -> dict[str, Any]:
    assessment = selection.get("candidate_assessment") if isinstance(selection.get("candidate_assessment"), dict) else {}
    selected = selection.get("selected_span_basis") if isinstance(selection.get("selected_span_basis"), list) else []
    return {
        "feasible": assessment.get("feasible"),
        "basis_size": len(selected),
        "rationale": assessment.get("rationale"),
        "span_indices": [row.get("source_span_index") for row in selected if isinstance(row, dict)],
        "legal_features": [row.get("legal_feature") for row in selected if isinstance(row, dict)],
    }


def _plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    public_task = plan.get("public_task") if isinstance(plan.get("public_task"), dict) else {}
    oracle = plan.get("oracle") if isinstance(plan.get("oracle"), dict) else {}
    units = plan.get("evidence_units") if isinstance(plan.get("evidence_units"), list) else []
    return {
        "objective": public_task.get("objective"),
        "decision_set": public_task.get("decision_set"),
        "answer": oracle.get("answer"),
        "unit_ids": [unit.get("unit_id") for unit in units if isinstance(unit, dict)],
        "normalization_notes": plan.get("normalization_notes"),
    }


def _audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "accept_under_threshold": audit.get("accept_under_threshold"),
        "minimum_sufficient_span_set_size": audit.get("minimum_sufficient_span_set_size"),
        "smallest_sufficient_span_sets": audit.get("smallest_sufficient_span_sets"),
        "single_unit_shortcut": audit.get("single_unit_shortcut"),
        "single_participant_shortcut": audit.get("single_participant_shortcut"),
        "non_required_unit_ids": audit.get("non_required_unit_ids"),
        "failed_reason": audit.get("failed_reason"),
        "notes": audit.get("notes"),
    }


def _measured_feedback_from_errors(
    errors: list[str],
    *,
    plan: dict[str, Any] | None = None,
    audit: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "instruction": (
            "Revise the previous design while keeping the selected direct CUAD spans unchanged. "
            "Fix every validation issue. If the decision_set was too obvious, make wrong options "
            "comparably detailed and legally plausible, not merely shorter single-feature choices."
        ),
        "validation_errors": errors,
    }
    if plan:
        payload["previous_attempt"] = _plan_feedback_summary(plan)
    if audit:
        payload["audit_feedback"] = _audit_summary(audit)
    return "Previous attempt was rejected. Use this JSON feedback:\n" + json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )


def _plan_feedback_summary(plan: dict[str, Any]) -> dict[str, Any]:
    public_task = plan.get("public_task") if isinstance(plan.get("public_task"), dict) else {}
    oracle = plan.get("oracle") if isinstance(plan.get("oracle"), dict) else {}
    participants = plan.get("participants") if isinstance(plan.get("participants"), list) else []
    units = plan.get("evidence_units") if isinstance(plan.get("evidence_units"), list) else []
    return {
        "objective": public_task.get("objective"),
        "decision_set": public_task.get("decision_set"),
        "decision_option_definitions": public_task.get("decision_option_definitions"),
        "oracle_answer": oracle.get("answer"),
        "oracle_rationale": oracle.get("rationale"),
        "evidence_units": [
            {
                "unit_id": unit.get("unit_id"),
                "span_index": (unit.get("source_ref") or {}).get("span_index")
                if isinstance(unit.get("source_ref"), dict)
                else None,
                "information_feature": unit.get("information_feature"),
                "summary": unit.get("summary"),
                "decision_relevance": unit.get("decision_relevance"),
            }
            for unit in units
            if isinstance(unit, dict)
        ],
        "participant_packets": [
            {
                "participant_id": participant.get("participant_id"),
                "public_role": participant.get("public_role"),
                "evidence_unit_ids": participant.get("evidence_unit_ids"),
            }
            for participant in participants
            if isinstance(participant, dict)
        ],
    }


def _funnel_counts(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    statuses = Counter(str(row.get("status")) for row in row_list)
    selected_basis_sizes = Counter(
        _counter_key((row.get("selection_summary") or {}).get("basis_size"))
        for row in row_list
        if isinstance(row.get("selection_summary"), dict)
    )
    accepted_basis_sizes = Counter(
        _counter_key((row.get("selection_summary") or {}).get("basis_size"))
        for row in row_list
        if row.get("status") == "accepted" and isinstance(row.get("selection_summary"), dict)
    )
    option_counts = Counter(
        _counter_key(len((row.get("plan_summary") or {}).get("decision_set") or []))
        for row in row_list
        if isinstance(row.get("plan_summary"), dict)
    )
    accepted_option_counts = Counter(
        _counter_key(len((row.get("plan_summary") or {}).get("decision_set") or []))
        for row in row_list
        if row.get("status") == "accepted" and isinstance(row.get("plan_summary"), dict)
    )
    audit_min_sizes = Counter(
        _counter_key((row.get("audit_summary") or {}).get("minimum_sufficient_span_set_size"))
        for row in row_list
        if isinstance(row.get("audit_summary"), dict)
    )
    return {
        "status_counts": dict(statuses),
        "candidate_count": sum(statuses.values()),
        "selection_local_passed": sum(1 for row in row_list if row.get("selection_local_valid") is True),
        "plan_local_passed": sum(1 for row in row_list if row.get("plan_local_valid") is True),
        "accepted": statuses.get("accepted", 0),
        "selected_basis_size_counts": dict(selected_basis_sizes),
        "accepted_basis_size_counts": dict(accepted_basis_sizes),
        "option_count_counts": dict(option_counts),
        "accepted_option_count_counts": dict(accepted_option_counts),
        "audit_minimum_sufficient_size_counts": dict(audit_min_sizes),
    }


def _counter_key(value: Any) -> str:
    parsed = _as_int(value)
    return str(parsed) if parsed is not None else "unknown"


def _trim_to_smallest_sufficient_span_set(
    selection: dict[str, Any],
    plan: dict[str, Any],
    audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Retain exactly one audited inclusion-minimal sufficient span set."""

    smallest = _smallest_sets(audit.get("smallest_sufficient_span_sets", []))
    if not smallest:
        raise ValueError("Cannot trim CUAD evidence without an audited smallest sufficient span set.")

    retained_set = min(
        smallest,
        key=lambda row: (len(row["unit_ids"]), tuple(row["unit_ids"])),
    )
    kept_ids = set(retained_set["unit_ids"])
    if len(kept_ids) != len(retained_set["unit_ids"]):
        raise ValueError("The selected CUAD smallest sufficient span set contains duplicate unit IDs.")
    original_units = plan.get("evidence_units") if isinstance(plan.get("evidence_units"), list) else []
    original_ids = [str(unit.get("unit_id")) for unit in original_units if isinstance(unit, dict)]
    removed_ids = [unit_id for unit_id in original_ids if unit_id not in kept_ids]

    final_selection = deepcopy(selection)
    selected = final_selection.get("selected_span_basis")
    if isinstance(selected, list):
        removed_span_indices = {
            _as_int(row.get("source_span_index"))
            for row in selected
            if isinstance(row, dict) and str(row.get("unit_id")) in removed_ids
        }
        final_selection["selected_span_basis"] = [
            row
            for row in selected
            if isinstance(row, dict) and str(row.get("unit_id")) in kept_ids
        ]
        assessment = final_selection.get("candidate_assessment")
        if isinstance(assessment, dict):
            assessment["basis_size"] = len(kept_ids)
        classifications = final_selection.get("source_span_classification")
        if isinstance(classifications, list):
            for row in classifications:
                if isinstance(row, dict) and _as_int(row.get("span_index")) in removed_span_indices:
                    row["role"] = "not_needed"
                    row["reason"] = "Removed after the sufficiency audit; not part of the retained minimal set."

    final_plan = deepcopy(plan)
    units = final_plan.get("evidence_units")
    if isinstance(units, list):
        final_plan["evidence_units"] = [
            unit
            for unit in units
            if isinstance(unit, dict) and str(unit.get("unit_id")) in kept_ids
        ]

    participants = final_plan.get("participants")
    if isinstance(participants, list):
        trimmed_participants: list[dict[str, Any]] = []
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            refs = [
                str(item)
                for item in participant.get("evidence_unit_ids", [])
                if str(item) in kept_ids
            ]
            if not refs:
                continue
            participant_row = dict(participant)
            participant_row["evidence_unit_ids"] = refs
            trimmed_participants.append(participant_row)
        if len(trimmed_participants) < 2:
            raise ValueError("Post-audit CUAD trimming must retain evidence across at least two participants.")
        final_plan["participants"] = trimmed_participants

    final_ids = {
        str(unit.get("unit_id"))
        for unit in final_plan.get("evidence_units", [])
        if isinstance(unit, dict)
    }
    if final_ids != kept_ids:
        raise ValueError(
            "Post-audit CUAD trimming did not retain exactly the selected smallest sufficient span set."
        )

    notes = final_plan.get("normalization_notes")
    if not isinstance(notes, list):
        notes = []
    if removed_ids:
        notes.append(f"post-audit trimmed non-required spans: {', '.join(removed_ids)}.")
    final_plan["normalization_notes"] = notes

    return final_selection, final_plan, {
        "trimmed": bool(removed_ids),
        "mode": "smallest_sufficient_span_set",
        "retained_smallest_sufficient_span_set": retained_set,
        "kept_unit_ids": [unit_id for unit_id in original_ids if unit_id in kept_ids],
        "removed_unit_ids": removed_ids,
        "original_included_span_count": len(original_ids),
        "final_included_span_count": len(kept_ids),
        "audit_minimum_sufficient_span_set_size": audit.get(
            "minimum_sufficient_span_set_size"
        ),
    }


def _normalize_plan_against_selection(plan: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    selected_rows = selection.get("selected_span_basis") if isinstance(selection.get("selected_span_basis"), list) else []
    selected_by_id = {
        str(row.get("unit_id")): row
        for row in selected_rows
        if isinstance(row, dict) and str(row.get("unit_id") or "")
    }
    if not selected_by_id:
        return plan

    notes: list[str] = []
    units = plan.get("evidence_units") if isinstance(plan.get("evidence_units"), list) else []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("unit_id") or "")
        selected = selected_by_id.get(unit_id)
        if not selected:
            continue
        selected_text = str(selected.get("text") or "")
        if selected_text and _norm(unit.get("text")) != _norm(selected_text):
            unit["text"] = selected_text
            notes.append(f"{unit_id}: canonicalized evidence text to selected CUAD span.")

        source_ref = unit.get("source_ref") if isinstance(unit.get("source_ref"), dict) else {}
        if source_ref is not unit.get("source_ref"):
            unit["source_ref"] = source_ref
        selected_span_index = _as_int(selected.get("source_span_index"))
        if selected_span_index is not None and _as_int(source_ref.get("span_index")) != selected_span_index:
            source_ref["span_index"] = selected_span_index
            notes.append(f"{unit_id}: canonicalized source_ref.span_index to selected CUAD span.")
        legal_feature = str(selected.get("legal_feature") or "").strip()
        if legal_feature:
            source_ref.setdefault("information_feature", legal_feature)
            unit.setdefault("information_feature", legal_feature)

    participants = plan.get("participants") if isinstance(plan.get("participants"), list) else []
    selected_ids = set(selected_by_id)
    if participants and selected_ids:
        kept: list[dict[str, Any]] = []
        removed_ids: list[str] = []
        for participant in participants:
            if not isinstance(participant, dict):
                kept.append(participant)
                continue
            refs = {str(item) for item in participant.get("evidence_unit_ids", [])}
            if selected_ids <= refs:
                removed_ids.append(str(participant.get("participant_id") or "unknown"))
            else:
                kept.append(participant)
        kept_refs: set[str] = set()
        for participant in kept:
            if isinstance(participant, dict):
                kept_refs.update(str(item) for item in participant.get("evidence_unit_ids", []))
        if removed_ids and len([p for p in kept if isinstance(p, dict)]) >= 2 and selected_ids <= kept_refs:
            plan["participants"] = kept
            notes.append(f"removed all-evidence participant packets: {', '.join(removed_ids)}.")

    if notes:
        existing_notes = plan.get("normalization_notes")
        if not isinstance(existing_notes, list):
            existing_notes = []
        existing_notes.extend(notes)
        plan["normalization_notes"] = existing_notes
    return plan


def _spans_by_index(candidate: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(span["span_index"]): span
        for span in candidate.get("positive_spans", [])
        if str(span.get("span_index", "")).isdigit()
    }


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use 'teamgym-generate legal/cuad partition --help' to partition accepted trials.",
    )
    parser.add_argument("--output-root", default="generated/cuad_measured_span_funnel")
    parser.add_argument("--cache-dir", default="data/legal_raw")
    parser.add_argument("--trace-root", default=None)
    parser.add_argument("--min-accept-size", type=int, default=2)
    parser.add_argument("--max-span-count", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--target-accepted", type=int, default=2)
    parser.add_argument("--candidate-start-index", type=int, default=1)
    parser.add_argument("--min-positive-spans", type=int, default=0)
    parser.add_argument(
        "--question-history-path",
        default=None,
        help="Optional JSON/JSONL file of prior same-source question attempts to avoid repeating.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help=(
            "Number of candidates to process concurrently. A value of 1 preserves exact sequential "
            "target stopping; larger values use a soft target because in-flight candidates finish."
        ),
    )
    parser.add_argument(
        "--no-save-generation-byproducts",
        action="store_true",
        help="Do not write the separate generation-byproduct JSONL with request metadata and rejection history.",
    )
    parser.add_argument("--provider", choices=("openai", "openai_compatible", "bedrock_openai"), default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--endpoint", choices=("/responses", "/chat/completions"), default=DEFAULT_LLM_ENDPOINT)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--credentials-file", default=_default_credentials_file())
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=5000)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--request-retry-attempts", type=int, default=4)
    parser.add_argument("--request-log-path", default=None)
    parser.add_argument("--no-log-prompts", action="store_true")
    parser.add_argument("--no-log-responses", action="store_true")
    return parser.parse_args()


def _default_credentials_file() -> str | None:
    path = Path("api_key/api_key.py")
    return str(path) if path.is_file() else None


def _prompt_example_text(relative_path: str) -> str:
    return prompt_text(relative_path)


def _suggested_option_count(candidate_index: int) -> int:
    return 3 + ((candidate_index - 1) % 4)


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    request_log_path = args.request_log_path or str(output_root / "_llm_requests.jsonl")
    client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            credentials_file=args.credentials_file,
            endpoint=args.endpoint,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            timeout_sec=args.timeout_sec,
            request_retry_attempts=args.request_retry_attempts,
            request_log_path=request_log_path,
            log_prompts=not args.no_log_prompts,
            log_responses=not args.no_log_responses,
        )
    )
    report = run_measured_span_funnel(
        output_root=output_root,
        cache_dir=args.cache_dir,
        llm_client=client,
        min_accept_size=args.min_accept_size,
        max_span_count=args.max_span_count,
        max_candidates=args.max_candidates,
        target_accepted=args.target_accepted,
        candidate_start_index=args.candidate_start_index,
        min_positive_spans=args.min_positive_spans,
        parallelism=args.parallelism,
        trace_root=args.trace_root,
        save_generation_byproducts=not args.no_save_generation_byproducts,
        question_history_path=args.question_history_path,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
