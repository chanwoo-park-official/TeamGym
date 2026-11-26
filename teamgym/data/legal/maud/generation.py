"""MAUD measured-PIU finite-choice sufficiency funnel.

This pipeline mirrors the CUAD measured direct-span funnel, but uses cleaned
MAUD row-evidence blocks as candidate pivotal information units (PIUs).
Native MAUD questions and answers are construction metadata only: the accepted
task is a newly generated finite-choice question whose exact oracle answer must
require multiple selected PIUs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

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
    normalize_decision_set as _normalize_decision_set,
    normalize_smallest_sufficient_sets as _smallest_sets,
    validate_decision_set_balance as _validate_decision_set_balance,
    validate_option_value_names as _validate_option_value_names,
)
from ..contracts.generation_records import split_generation_byproducts
from ..converter_utils import call_legal_planner, compact_json
from .evidence import (
    _aggressive_normalize_with_offsets,
    _agreement_chunks,
    _normalize_with_offsets,
    match_evidence_block,
    split_maud_row_text,
)
from ..prepartition_export import write_pre_partition_task_tuples
from ..question_history import (
    load_source_question_history,
    question_attempt_entry,
    question_history_prompt,
    summarize_question_attempt,
)
from ..sources import _hf_download, _maud_split_name

MAUD_MEASURED_PLAN_ATTEMPTS = 3
MAUD_PRIORITY_CATEGORIES = [
    "Deal Protection and Related Provisions",
    "Conditions to Closing",
    "Material Adverse Effect",
    "Operating and Efforts Covenant",
    "Remedies",
    "General Information",
]


def run_maud_measured_rui_funnel(
    *,
    output_root: str | Path,
    cache_dir: str | Path,
    llm_client: OpenAIJsonClient,
    min_accept_size: int,
    target_min_rui_count: int,
    max_rui_count: int | None,
    max_candidates: int,
    target_accepted: int,
    split: str = "train",
    candidate_start_index: int = 1,
    min_candidate_ruis: int = 0,
    max_issue_groups_per_candidate: int = 10,
    max_candidate_ruis: int = 30,
    parallelism: int = 1,
    trace_root: str | Path | None = None,
    save_generation_byproducts: bool = True,
    acceptance_mode: str = "trim",
    question_history_path: str | Path | None = None,
) -> dict[str, Any]:
    if min_accept_size < 1:
        raise ValueError("min_accept_size must be >= 1.")
    if target_min_rui_count < min_accept_size:
        raise ValueError("target_min_rui_count must be >= min_accept_size.")
    if max_rui_count is not None and max_rui_count < min_accept_size:
        raise ValueError("max_rui_count must be >= min_accept_size.")
    if candidate_start_index < 1:
        raise ValueError("candidate_start_index must be >= 1.")
    if parallelism < 1:
        raise ValueError("parallelism must be >= 1.")
    if acceptance_mode not in {"trim", "strict", "measured"}:
        raise ValueError("acceptance_mode must be one of: trim, strict, measured.")
    effective_min_candidate_ruis = max(min_candidate_ruis, min_accept_size)

    output_root_path = Path(output_root).expanduser().resolve()
    trace_root_path = Path(trace_root).expanduser().resolve() if trace_root else output_root_path / "_llm_traces"
    source_question_history = load_source_question_history(question_history_path)
    if parallelism == 1:
        rows, accepted = _run_candidates_sequential(
            cache_dir=cache_dir,
            split=split,
            llm_client=llm_client,
            min_accept_size=min_accept_size,
            target_min_rui_count=target_min_rui_count,
            max_rui_count=max_rui_count,
            max_candidates=max_candidates,
            target_accepted=target_accepted,
            candidate_start_index=candidate_start_index,
            min_candidate_ruis=effective_min_candidate_ruis,
            max_issue_groups_per_candidate=max_issue_groups_per_candidate,
            max_candidate_ruis=max_candidate_ruis,
            trace_root=trace_root_path,
            acceptance_mode=acceptance_mode,
            source_question_history=source_question_history,
        )
        target_stop_mode = "exact"
    else:
        rows, accepted = _run_candidates_parallel(
            cache_dir=cache_dir,
            split=split,
            llm_client_config=llm_client.cfg,
            min_accept_size=min_accept_size,
            target_min_rui_count=target_min_rui_count,
            max_rui_count=max_rui_count,
            max_candidates=max_candidates,
            target_accepted=target_accepted,
            candidate_start_index=candidate_start_index,
            min_candidate_ruis=effective_min_candidate_ruis,
            max_issue_groups_per_candidate=max_issue_groups_per_candidate,
            max_candidate_ruis=max_candidate_ruis,
            trace_root=trace_root_path,
            parallelism=parallelism,
            acceptance_mode=acceptance_mode,
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
        dataset="maud",
        min_accept_size=min_accept_size,
        accepted_trials=accepted_to_write,
    )
    report = {
        "schema": "teamgym_maud_measured_rui_funnel_v1",
        "output_root": str(output_root_path),
        "trace_root": str(trace_root_path),
        "split": split,
        "min_accept_size": min_accept_size,
        "target_min_rui_count": target_min_rui_count,
        "max_rui_count": max_rui_count,
        "acceptance_mode": acceptance_mode,
        "max_candidates": max_candidates,
        "target_accepted": target_accepted,
        "candidate_start_index": candidate_start_index,
        "min_candidate_ruis": min_candidate_ruis,
        "effective_min_candidate_ruis": effective_min_candidate_ruis,
        "max_issue_groups_per_candidate": max_issue_groups_per_candidate,
        "max_candidate_ruis": max_candidate_ruis,
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
    write_json(output_root_path / f"funnel_maud_measured_rui_minaccept{min_accept_size}.json", report)
    write_jsonl(output_root_path / f"accepted_maud_measured_rui_minaccept{min_accept_size}.jsonl", accepted_to_write)
    write_json(output_root_path / f"accepted_maud_measured_rui_minaccept{min_accept_size}.pretty.json", accepted_to_write)
    if save_generation_byproducts:
        write_jsonl(byproduct_path, generation_byproducts)
    write_jsonl(question_attempt_history_path, [summarize_question_attempt(row) for row in rows])
    return report


def maud_measured_rui_candidates(
    cache_dir: str | Path,
    *,
    split: str = "train",
    min_candidate_ruis: int = 2,
    max_issue_groups_per_candidate: int = 10,
    max_candidate_ruis: int = 30,
) -> Iterable[dict[str, Any]]:
    """Yield contract/category bundles of issue groups and cleaned MAUD RIUs."""

    csv_path = _hf_download("theatticusproject/maud", f"MAUD_v1/MAUD_{_maud_split_name(split)}.csv", cache_dir)
    df = pd.read_csv(csv_path).fillna("")
    df = df[~df["contract_name"].astype(str).str.startswith("<")].copy()
    grouped = df.groupby("contract_name", sort=True)
    for contract_name, contract_rows in grouped:
        try:
            contract_text_path = _hf_download(
                "theatticusproject/maud",
                f"MAUD_v1/contracts/{contract_name}.txt",
                cache_dir,
            )
            contract_text = Path(contract_text_path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            yield {
                "dataset": "maud_measured_rui",
                "source_id": contract_name,
                "status": "missing_contract_text",
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue

        issue_groups = _contract_issue_groups(contract_rows)
        for category in MAUD_PRIORITY_CATEGORIES:
            category_groups = [row for row in issue_groups if row["category"] == category]
            if len(category_groups) < 2:
                continue
            selected_groups = category_groups[:max_issue_groups_per_candidate]
            candidate = _build_candidate(
                split=split,
                contract_name=str(contract_name),
                contract_text=contract_text,
                category=category,
                issue_groups=selected_groups,
                max_candidate_ruis=max_candidate_ruis,
            )
            if len(candidate["candidate_rui_units"]) < max(1, min_candidate_ruis):
                continue
            yield candidate


def _run_candidates_sequential(
    *,
    cache_dir: str | Path,
    split: str,
    llm_client: OpenAIJsonClient,
    min_accept_size: int,
    target_min_rui_count: int,
    max_rui_count: int | None,
    max_candidates: int,
    target_accepted: int,
    candidate_start_index: int,
    min_candidate_ruis: int,
    max_issue_groups_per_candidate: int,
    max_candidate_ruis: int,
    trace_root: Path,
    acceptance_mode: str,
    source_question_history: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in _iter_candidate_items(
        cache_dir=cache_dir,
        split=split,
        candidate_start_index=candidate_start_index,
        min_candidate_ruis=min_candidate_ruis,
        max_candidates=max_candidates,
        max_issue_groups_per_candidate=max_issue_groups_per_candidate,
        max_candidate_ruis=max_candidate_ruis,
    ):
        if len(accepted) >= target_accepted:
            break
        row, accepted_plan = _try_candidate(
            candidate=candidate,
            candidate_index=candidate_index,
            min_accept_size=min_accept_size,
            target_min_rui_count=target_min_rui_count,
            max_rui_count=max_rui_count,
            llm_client=llm_client,
            trace_root=trace_root,
            acceptance_mode=acceptance_mode,
            source_question_history=source_question_history.get(str(candidate.get("source_id") or ""), []),
        )
        rows.append(row)
        if accepted_plan is not None:
            accepted.append(accepted_plan)
    return rows, accepted


def _run_candidates_parallel(
    *,
    cache_dir: str | Path,
    split: str,
    llm_client_config: OpenAIJsonClientConfig,
    min_accept_size: int,
    target_min_rui_count: int,
    max_rui_count: int | None,
    max_candidates: int,
    target_accepted: int,
    candidate_start_index: int,
    min_candidate_ruis: int,
    max_issue_groups_per_candidate: int,
    max_candidate_ruis: int,
    trace_root: Path,
    parallelism: int,
    acceptance_mode: str,
    source_question_history: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    candidate_iter = iter(
        _iter_candidate_items(
            cache_dir=cache_dir,
            split=split,
            candidate_start_index=candidate_start_index,
            min_candidate_ruis=min_candidate_ruis,
            max_candidates=max_candidates,
            max_issue_groups_per_candidate=max_issue_groups_per_candidate,
            max_candidate_ruis=max_candidate_ruis,
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
            target_min_rui_count=target_min_rui_count,
            max_rui_count=max_rui_count,
            llm_client=client,
            trace_root=trace_root,
            acceptance_mode=acceptance_mode,
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
    split: str,
    candidate_start_index: int,
    min_candidate_ruis: int,
    max_candidates: int,
    max_issue_groups_per_candidate: int,
    max_candidate_ruis: int,
) -> Iterable[tuple[int, dict[str, Any]]]:
    yielded = 0
    for candidate_index, candidate in enumerate(
        maud_measured_rui_candidates(
            cache_dir,
            split=split,
            min_candidate_ruis=min_candidate_ruis,
            max_issue_groups_per_candidate=max_issue_groups_per_candidate,
            max_candidate_ruis=max_candidate_ruis,
        ),
        start=1,
    ):
        if candidate_index < candidate_start_index:
            continue
        if candidate.get("status") and candidate.get("status") != "ok":
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
    target_min_rui_count: int,
    max_rui_count: int | None,
    llm_client: OpenAIJsonClient,
    trace_root: Path,
    acceptance_mode: str,
    source_question_history: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    instance_id = f"maud_measured_rui_{candidate_index:04d}_{_slug(candidate['contract_name'])}_{_slug(candidate['category'])}"
    row = _candidate_row(candidate_index, candidate, "started", [])
    row["instance_id"] = instance_id
    prior_question_history = list(source_question_history or [])
    row["source_question_history_count"] = len(prior_question_history)

    selection = call_legal_planner(
        dataset="maud_measured_rui",
        prompt_name="measured_rui_select_user",
        variables={
            "CONTRACT_NAME": candidate["contract_name"],
            "CATEGORY": candidate["category"],
            "ISSUE_GROUPS_JSON": compact_json(candidate["issue_groups"], max_chars=25000),
            "CANDIDATE_RUIS_JSON": compact_json(candidate["candidate_rui_units"], max_chars=45000),
            "CONTRACT_TEXT_EXCERPT": compact_text(candidate["contract_text"], max_chars=12000),
            "MIN_ACCEPT_SIZE": str(min_accept_size),
            "TARGET_MIN_RUI_COUNT": str(target_min_rui_count),
            "MAX_RUI_COUNT_TEXT": _max_rui_count_text(max_rui_count),
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
        max_rui_count=max_rui_count,
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
    for plan_attempt in range(1, MAUD_MEASURED_PLAN_ATTEMPTS + 1):
        plan = call_legal_planner(
            dataset="maud_measured_rui",
            prompt_name="measured_rui_question_user",
            variables={
                "CONTRACT_NAME": candidate["contract_name"],
                "CATEGORY": candidate["category"],
                "ISSUE_GROUPS_JSON": compact_json(candidate["issue_groups"], max_chars=20000),
                "CANDIDATE_RUIS_JSON": compact_json(candidate["candidate_rui_units"], max_chars=30000),
                "SELECTED_RUI_BASIS_JSON": compact_json(selection, max_chars=35000),
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
        plan_errors = _validate_plan(plan, selection)
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
            feedback = _feedback_from_errors(plan_errors, plan=plan)
            continue

        row["plan_local_valid"] = True
        audit = call_legal_planner(
            dataset="maud_measured_rui",
            prompt_name="measured_rui_sufficiency_audit_user",
            variables={
                "CATEGORY": candidate["category"],
                "ISSUE_GROUPS_JSON": compact_json(candidate["issue_groups"], max_chars=20000),
                "CANDIDATE_RUIS_JSON": compact_json(candidate["candidate_rui_units"], max_chars=30000),
                "PLAN_JSON": compact_json(plan, max_chars=40000),
                "MIN_ACCEPT_SIZE": str(min_accept_size),
            },
            llm_client=llm_client,
            trace_root=trace_root,
            instance_id=instance_id,
        )
        audit_meta = dict(llm_client.last_request_meta)
        audit_errors = _validate_audit(
            audit,
            plan,
            min_accept_size=min_accept_size,
            strict_all_included_required=acceptance_mode == "strict",
        )
        row["audit_request_meta"] = audit_meta
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
            feedback = _feedback_from_errors(audit_errors, plan=plan, audit=audit)
            continue

        row["status"] = "accepted"
        row["errors"] = []
        row["rejection_history"] = rejection_history
        final_selection = selection
        final_plan = plan
        post_audit_trimming: dict[str, Any] | None = None
        if acceptance_mode == "trim":
            final_selection, final_plan, post_audit_trimming = _trim_to_smallest_sufficient_set(selection, plan, audit)
            row["post_audit_trimming"] = post_audit_trimming
            row["selection_summary"] = _selection_summary(final_selection)
            row["plan_summary"] = _plan_summary(final_plan)
        generation_byproduct = {
            "schema": "maud_measured_rui_generation_byproduct_v1",
            "instance_id": instance_id,
            "candidate_index": candidate_index,
            "source_id": candidate["source_id"],
            "source_candidate": _source_candidate_summary(candidate),
            "prompt_template_ids": [
                "legal/maud_measured_rui/measured_rui_select_user",
                "legal/maud_measured_rui/measured_rui_question_user",
                "legal/maud_measured_rui/measured_rui_sufficiency_audit_user",
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
            "trial_schema": "maud_measured_rui_trial_plan_v1",
            "instance_id": instance_id,
            "candidate_index": candidate_index,
            "source_id": candidate["source_id"],
            "contract_name": candidate["contract_name"],
            "category": candidate["category"],
            "min_accept_size": min_accept_size,
            "target_min_rui_count": target_min_rui_count,
            "max_rui_count": max_rui_count,
            "source_candidate": _source_candidate_summary(candidate),
            "selection": final_selection,
            "plan": final_plan,
            "audit": audit,
            "post_audit_trimming": post_audit_trimming,
            "planning_attempts": plan_attempt,
            "generation_metadata": {
                "method": "maud_measured_rui_finite_choice_generation",
                "no_synthetic_fallback": True,
                "prompt_stage_count": 3,
                "acceptance_mode": acceptance_mode,
                "thresholds": {
                    "min_accept_size": min_accept_size,
                    "target_min_rui_count": target_min_rui_count,
                    "max_rui_count": max_rui_count,
                },
                "validation_path": [
                    "selection_local_validation",
                    "planner_local_validation",
                    "llm_sufficiency_audit",
                ],
            },
            "_generation_byproduct": generation_byproduct,
        }

    last_rejection = rejection_history[-1] if rejection_history else {}
    row["status"] = "audit_failed" if last_rejection.get("stage") == "sufficiency_audit" else "plan_local_failed"
    row["errors"] = list(last_rejection.get("errors") or ["planning failed without a recorded validation error."])
    row["rejection_history"] = rejection_history
    return row, None


def _contract_issue_groups(contract_rows: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped = contract_rows.groupby(["question", "subquestion"], sort=True, dropna=False)
    for group_index, ((question, subquestion), group) in enumerate(grouped, start=1):
        rows = []
        for _, row in group.sort_values(["id", "text_type"]).iterrows():
            answer = str(row.get("answer") or "").strip()
            text = str(row.get("text") or "").strip()
            if not answer or not text:
                continue
            rows.append({key: _json_scalar(row.get(key)) for key in contract_rows.columns})
        if not rows:
            continue
        answer_keys = {_norm(row.get("answer")) for row in rows if str(row.get("answer") or "").strip()}
        if len(answer_keys) != 1:
            continue
        category = str(rows[0].get("category") or "").strip()
        if category not in MAUD_PRIORITY_CATEGORIES:
            continue
        out.append(
            {
                "issue_group_id": f"issue_group_{group_index:04d}",
                "question": str(question),
                "subquestion": str(subquestion),
                "category": category,
                "native_answer": rows[0].get("answer"),
                "row_count": len(rows),
                "row_ids": [row.get("id") for row in rows],
                "text_types": sorted({str(row.get("text_type") or "") for row in rows if str(row.get("text_type") or "")}),
                "rows": rows,
            }
        )
    return out


def _build_candidate(
    *,
    split: str,
    contract_name: str,
    contract_text: str,
    category: str,
    issue_groups: list[dict[str, Any]],
    max_candidate_ruis: int,
) -> dict[str, Any]:
    normalized_contract = _normalize_with_offsets(contract_text)
    aggressive_contract = _aggressive_normalize_with_offsets(contract_text)
    chunks = _agreement_chunks(contract_text)
    candidate_rui_units: list[dict[str, Any]] = []
    dropped_fragments: list[dict[str, Any]] = []
    match_counts: Counter[str] = Counter()
    seen_texts: set[str] = set()

    for group in issue_groups:
        group_rui_ids: list[str] = []
        for row_index, row in enumerate(group["rows"], start=1):
            blocks, dropped = split_maud_row_text(str(row.get("text") or ""))
            for text in dropped:
                dropped_fragments.append(
                    {
                        "issue_group_id": group["issue_group_id"],
                        "row_id": row.get("id"),
                        "text": text,
                    }
                )
            for block_index, block in enumerate(blocks, start=1):
                text_key = _norm(block)
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
                grounding = match_evidence_block(block, contract_text, normalized_contract, chunks, aggressive_contract)
                match_status = str(grounding.get("match_status") or "")
                match_counts[match_status] += 1
                if not _has_contract_offsets(grounding):
                    dropped_fragments.append(
                        {
                            "issue_group_id": group["issue_group_id"],
                            "row_id": row.get("id"),
                            "text": block,
                            "reason": "unmatched_to_full_agreement",
                            "match_status": match_status,
                            "match_score": grounding.get("match_score"),
                        }
                    )
                    continue
                source_rui_id = f"source_rui_{len(candidate_rui_units) + 1:03d}"
                group_rui_ids.append(source_rui_id)
                candidate_rui_units.append(
                    {
                        "source_rui_id": source_rui_id,
                        "text": block,
                        "issue_group_id": group["issue_group_id"],
                        "native_question": group["question"],
                        "native_subquestion": group["subquestion"],
                        "native_answer": group["native_answer"],
                        "category": group["category"],
                        "text_type": row.get("text_type"),
                        "row_id": row.get("id"),
                        "row_index_within_issue_group": row_index,
                        "block_index_within_row": block_index,
                        "source_ref": {
                            "source": "maud_row_evidence_block",
                            "match_status": match_status,
                            "match_score": grounding.get("match_score"),
                            "agreement_start": grounding.get("agreement_start"),
                            "agreement_end": grounding.get("agreement_end"),
                        },
                    }
                )
                if len(candidate_rui_units) >= max_candidate_ruis:
                    break
            if len(candidate_rui_units) >= max_candidate_ruis:
                break
        group["candidate_rui_ids"] = group_rui_ids
        group.pop("rows", None)
        if len(candidate_rui_units) >= max_candidate_ruis:
            break

    candidate_rui_units = _filter_overlapping_candidate_ruis(candidate_rui_units, issue_groups, dropped_fragments)

    return {
        "dataset": "maud_measured_rui",
        "source_id": f"{contract_name}__{_slug(category)}",
        "split": split,
        "contract_name": contract_name,
        "contract_text": contract_text,
        "category": category,
        "issue_groups": issue_groups,
        "candidate_rui_units": candidate_rui_units,
        "candidate_rui_count": len(candidate_rui_units),
        "evidence_grounding_report": {
            "match_counts": dict(match_counts),
            "dropped_fragment_count": len(dropped_fragments),
            "dropped_fragments": dropped_fragments[:20],
        },
    }


def _validate_selection(
    selection: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_accept_size: int,
    max_rui_count: int | None,
) -> list[str]:
    errors: list[str] = []
    source_by_id = _source_rui_by_id(candidate)
    assessment = selection.get("candidate_assessment") if isinstance(selection.get("candidate_assessment"), dict) else {}
    if assessment.get("feasible") is not True:
        errors.append(f"selection marked infeasible: {assessment.get('rationale') or assessment.get('reason')}")
    classifications = selection.get("source_rui_classification")
    if not isinstance(classifications, list):
        errors.append("source_rui_classification must be a list.")
    else:
        classified = {str(row.get("source_rui_id")) for row in classifications if isinstance(row, dict)}
        missing = set(source_by_id) - classified
        if missing:
            errors.append(f"source_rui_classification missing source RIUs {sorted(missing)}.")
    selected = selection.get("selected_rui_basis")
    if not isinstance(selected, list):
        errors.append("selected_rui_basis must be a list.")
        selected = []
    if len(selected) < min_accept_size:
        errors.append(f"selected_rui_basis must contain at least {min_accept_size} RIUs.")
    if max_rui_count is not None and len(selected) > max_rui_count:
        errors.append(f"selected_rui_basis must contain at most {max_rui_count} RIUs.")
    unit_ids: list[str] = []
    source_ids: list[str] = []
    for item in selected:
        if not isinstance(item, dict):
            errors.append("each selected_rui_basis row must be an object.")
            continue
        unit_id = str(item.get("unit_id") or "")
        source_rui_id = str(item.get("source_rui_id") or "")
        unit_ids.append(unit_id)
        source_ids.append(source_rui_id)
        source = source_by_id.get(source_rui_id)
        if not unit_id:
            errors.append("each selected RIU needs unit_id.")
        if source is None:
            errors.append(f"{unit_id}: source_rui_id must reference a supplied candidate RIU.")
        elif _norm(item.get("text")) != _norm(source.get("text")):
            errors.append(f"{unit_id}: text must exactly equal supplied source RIU {source_rui_id}.")
        if not str(item.get("legal_feature") or "").strip():
            errors.append(f"{unit_id}: legal_feature is required.")
        if not str(item.get("possible_decision_boundary") or "").strip():
            errors.append(f"{unit_id}: possible_decision_boundary is required.")
    if len(set(unit_ids)) != len(unit_ids):
        errors.append("selected RIU unit IDs must be unique.")
    if len(set(source_ids)) != len(source_ids):
        errors.append("selected source RIUs must be unique.")
    return errors


def _validate_plan(plan: dict[str, Any], selection: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    selected = selection.get("selected_rui_basis") if isinstance(selection.get("selected_rui_basis"), list) else []
    selected_ids = {str(row.get("unit_id")) for row in selected if isinstance(row, dict)}
    selected_by_id = {
        str(row.get("unit_id")): row
        for row in selected
        if isinstance(row, dict) and str(row.get("unit_id") or "")
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
        errors.append("evidence_units must correspond exactly to selected_rui_basis IDs.")
    for unit in units:
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("unit_id") or "")
        selected_row = selected_by_id.get(unit_id, {})
        if _norm(unit.get("text")) != _norm(selected_row.get("text")):
            errors.append(f"{unit_id}: evidence text must equal selected MAUD RIU text.")
        source_ref = unit.get("source_ref") if isinstance(unit.get("source_ref"), dict) else {}
        if str(source_ref.get("source_rui_id") or "") != str(selected_row.get("source_rui_id") or ""):
            errors.append(f"{unit_id}: source_ref.source_rui_id must match selected source RIU.")
        if str(source_ref.get("issue_group_id") or "") != str(selected_row.get("issue_group_id") or ""):
            errors.append(f"{unit_id}: source_ref.issue_group_id must match selected issue group.")
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
            errors.append(f"{participant.get('participant_id')}: participant references unknown RIUs.")
        if selected_ids and selected_ids <= refs:
            errors.append(f"{participant.get('participant_id')}: participant has all selected RIUs.")
    return errors


def _validate_audit(
    audit: dict[str, Any],
    plan: dict[str, Any],
    *,
    min_accept_size: int,
    strict_all_included_required: bool = True,
) -> list[str]:
    errors: list[str] = []
    units = plan.get("evidence_units") if isinstance(plan.get("evidence_units"), list) else []
    unit_ids = {str(unit.get("unit_id")) for unit in units if isinstance(unit, dict)}
    oracle = plan.get("oracle") if isinstance(plan.get("oracle"), dict) else {}
    oracle_answer = str(oracle.get("answer") or "")
    if audit.get("valid") is not True:
        errors.append("audit did not mark the plan valid.")
    if audit.get("accept_under_threshold") is not True:
        errors.append("audit did not accept under the measured threshold.")
    if audit.get("copied_from_maud_rui_blocks") is not True:
        errors.append("audit did not confirm MAUD RIU block provenance.")
    if audit.get("full_rui_set_sufficient") is not True:
        errors.append("audit did not confirm full selected RIU set sufficiency.")
    if audit.get("single_unit_shortcut") is True:
        errors.append("audit found a single-unit shortcut.")
    if audit.get("single_participant_shortcut") is True:
        errors.append("audit found a single-participant shortcut.")
    min_size = _as_int(audit.get("minimum_sufficient_rui_set_size"))
    if min_size is None:
        errors.append("audit must report minimum_sufficient_rui_set_size.")
    elif min_size < min_accept_size:
        errors.append(f"minimum sufficient RIU set size {min_size} is below threshold {min_accept_size}.")
    elif strict_all_included_required and min_size != len(unit_ids):
        errors.append(
            "strict critical RIU mode requires minimum_sufficient_rui_set_size "
            "to equal the number of included evidence_units."
        )
    non_required = audit.get("non_required_unit_ids")
    if strict_all_included_required:
        if not isinstance(non_required, list):
            errors.append("strict critical RIU mode requires non_required_unit_ids to be an empty list.")
        elif non_required:
            errors.append(f"strict critical RIU mode found non-required RIUs: {non_required}.")
    smallest = _smallest_sets(audit.get("smallest_sufficient_rui_sets", []))
    if not smallest:
        errors.append("audit must report smallest_sufficient_rui_sets.")
    for row in smallest:
        row_ids = set(row["unit_ids"])
        if not row_ids <= unit_ids:
            errors.append(f"smallest_sufficient_rui_sets references unknown RIUs: {sorted(row_ids - unit_ids)}.")
        if oracle_answer and str(row.get("answer") or "") != oracle_answer:
            errors.append("smallest_sufficient_rui_sets must be for the exact oracle answer.")
        if len(row_ids) < min_accept_size:
            errors.append("smallest_sufficient_rui_sets contains a set below threshold.")
        if strict_all_included_required and row_ids != unit_ids:
            errors.append("strict critical RIU mode requires each smallest sufficient set to include every RIU.")
    if strict_all_included_required:
        leave_one_out = audit.get("leave_one_out_results")
        if not isinstance(leave_one_out, list):
            errors.append("strict critical RIU mode requires leave_one_out_results.")
        else:
            seen_omissions: set[str] = set()
            for row in leave_one_out:
                if not isinstance(row, dict):
                    continue
                omitted_id = str(row.get("omitted_unit_id") or "")
                if omitted_id:
                    seen_omissions.add(omitted_id)
                if omitted_id in unit_ids and row.get("can_determine_oracle_without_omitted") is not False:
                    errors.append(f"{omitted_id}: strict mode requires omission to make oracle indeterminate.")
            missing_omissions = unit_ids - seen_omissions
            if missing_omissions:
                errors.append(f"strict critical RIU mode missing leave-one-out results for {sorted(missing_omissions)}.")
    errors.extend(_local_participant_smallest_set_errors(plan, smallest))
    return errors


def _trim_to_smallest_sufficient_set(
    selection: dict[str, Any],
    plan: dict[str, Any],
    audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Keep only the RIUs that the audit found sufficient for the exact answer."""

    smallest = _smallest_sets(audit.get("smallest_sufficient_rui_sets", []))
    if not smallest:
        return selection, plan, {"trimmed": False, "reason": "no smallest sufficient set reported"}

    kept_ids = set(smallest[0]["unit_ids"])
    original_units = plan.get("evidence_units") if isinstance(plan.get("evidence_units"), list) else []
    original_ids = [str(unit.get("unit_id")) for unit in original_units if isinstance(unit, dict)]
    removed_ids = [unit_id for unit_id in original_ids if unit_id not in kept_ids]
    if not removed_ids:
        return selection, plan, {"trimmed": False, "kept_unit_ids": sorted(kept_ids), "removed_unit_ids": []}

    final_selection = deepcopy(selection)
    selected = final_selection.get("selected_rui_basis")
    if isinstance(selected, list):
        final_selection["selected_rui_basis"] = [
            row for row in selected if isinstance(row, dict) and str(row.get("unit_id")) in kept_ids
        ]

    final_plan = deepcopy(plan)
    units = final_plan.get("evidence_units")
    if isinstance(units, list):
        final_plan["evidence_units"] = [
            unit for unit in units if isinstance(unit, dict) and str(unit.get("unit_id")) in kept_ids
        ]

    participants = final_plan.get("participants")
    if isinstance(participants, list):
        trimmed_participants: list[dict[str, Any]] = []
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            refs = [str(item) for item in participant.get("evidence_unit_ids", []) if str(item) in kept_ids]
            if not refs:
                continue
            participant_row = dict(participant)
            participant_row["evidence_unit_ids"] = refs
            trimmed_participants.append(participant_row)
        if len(trimmed_participants) >= 2:
            final_plan["participants"] = trimmed_participants

    notes = final_plan.get("normalization_notes")
    if not isinstance(notes, list):
        notes = []
    notes.append(f"post-audit trimmed non-required RIUs: {', '.join(removed_ids)}.")
    final_plan["normalization_notes"] = notes

    return final_selection, final_plan, {
        "trimmed": True,
        "mode": "smallest_sufficient_rui_set",
        "kept_unit_ids": [unit_id for unit_id in original_ids if unit_id in kept_ids],
        "removed_unit_ids": removed_ids,
        "original_included_rui_count": len(original_ids),
        "final_included_rui_count": len(kept_ids),
        "audit_minimum_sufficient_rui_set_size": audit.get("minimum_sufficient_rui_set_size"),
    }


def _normalize_plan_against_selection(plan: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    selected_rows = selection.get("selected_rui_basis") if isinstance(selection.get("selected_rui_basis"), list) else []
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
            notes.append(f"{unit_id}: canonicalized evidence text to selected MAUD RIU.")

        source_ref = unit.get("source_ref") if isinstance(unit.get("source_ref"), dict) else {}
        if source_ref is not unit.get("source_ref"):
            unit["source_ref"] = source_ref
        for key in ("source_rui_id", "issue_group_id"):
            value = str(selected.get(key) or "").strip()
            if value and str(source_ref.get(key) or "") != value:
                source_ref[key] = value
                notes.append(f"{unit_id}: canonicalized source_ref.{key}.")
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


def _local_participant_smallest_set_errors(plan: dict[str, Any], smallest: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    participants = plan.get("participants") if isinstance(plan.get("participants"), list) else []
    smallest_sets = [set(row["unit_ids"]) for row in smallest]
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        refs = {str(item) for item in participant.get("evidence_unit_ids", [])}
        for rui_set in smallest_sets:
            if rui_set and rui_set <= refs:
                errors.append(f"{participant.get('participant_id')}: participant has a smallest sufficient RIU set.")
                break
    return errors


def _source_rui_by_id(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(unit.get("source_rui_id")): unit
        for unit in candidate.get("candidate_rui_units", [])
        if unit.get("source_rui_id")
    }


def _has_contract_offsets(grounding: dict[str, Any]) -> bool:
    return isinstance(grounding.get("agreement_start"), int) and isinstance(grounding.get("agreement_end"), int)


def _filter_overlapping_candidate_ruis(
    candidate_rui_units: list[dict[str, Any]],
    issue_groups: list[dict[str, Any]],
    dropped_fragments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove grounded RIU candidates whose contract offsets overlap.

    Overlapping RIUs cannot be cleanly assigned to different participants as
    private full-document chunks. We keep the shorter, more localized unit.
    """

    drop_ids: set[str] = set()
    units = [
        unit
        for unit in candidate_rui_units
        if isinstance((unit.get("source_ref") or {}).get("agreement_start"), int)
        and isinstance((unit.get("source_ref") or {}).get("agreement_end"), int)
    ]
    for left_idx, left in enumerate(units):
        for right in units[left_idx + 1 :]:
            left_ref = left.get("source_ref") or {}
            right_ref = right.get("source_ref") or {}
            left_start, left_end = int(left_ref["agreement_start"]), int(left_ref["agreement_end"])
            right_start, right_end = int(right_ref["agreement_start"]), int(right_ref["agreement_end"])
            if max(left_start, right_start) >= min(left_end, right_end):
                continue
            dropped, kept = _overlap_drop_choice(left, right)
            dropped_id = str(dropped.get("source_rui_id") or "")
            kept_id = str(kept.get("source_rui_id") or "")
            if dropped_id:
                drop_ids.add(dropped_id)
                dropped_fragments.append(
                    {
                        "issue_group_id": dropped.get("issue_group_id"),
                        "row_id": dropped.get("row_id"),
                        "text": dropped.get("text"),
                        "reason": "overlaps_grounded_rui",
                        "source_rui_id": dropped_id,
                        "overlaps_with_source_rui_id": kept_id,
                    }
                )
    if not drop_ids:
        return candidate_rui_units
    kept_units = [unit for unit in candidate_rui_units if str(unit.get("source_rui_id") or "") not in drop_ids]
    kept_ids = {str(unit.get("source_rui_id") or "") for unit in kept_units}
    for group in issue_groups:
        ids = group.get("candidate_rui_ids")
        if isinstance(ids, list):
            group["candidate_rui_ids"] = [source_rui_id for source_rui_id in ids if str(source_rui_id) in kept_ids]
    return kept_units


def _overlap_drop_choice(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    left_ref = left.get("source_ref") or {}
    right_ref = right.get("source_ref") or {}
    left_len = int(left_ref["agreement_end"]) - int(left_ref["agreement_start"])
    right_len = int(right_ref["agreement_end"]) - int(right_ref["agreement_start"])
    if left_len != right_len:
        return (left, right) if left_len > right_len else (right, left)
    left_score = float(left_ref.get("match_score") or 0.0)
    right_score = float(right_ref.get("match_score") or 0.0)
    if left_score != right_score:
        return (left, right) if left_score < right_score else (right, left)
    return (right, left)


def _source_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_name": candidate.get("contract_name"),
        "category": candidate.get("category"),
        "issue_group_count": len(candidate.get("issue_groups") or []),
        "candidate_rui_count": len(candidate.get("candidate_rui_units") or []),
        "issue_groups": [
            {
                "issue_group_id": row.get("issue_group_id"),
                "question": row.get("question"),
                "subquestion": row.get("subquestion"),
                "native_answer": row.get("native_answer"),
                "candidate_rui_ids": row.get("candidate_rui_ids"),
            }
            for row in candidate.get("issue_groups", [])
        ],
        "evidence_grounding_report": candidate.get("evidence_grounding_report"),
    }


def _candidate_row(candidate_index: int, candidate: dict[str, Any], status: str, errors: list[str]) -> dict[str, Any]:
    return {
        "candidate_index": candidate_index,
        "source_id": candidate.get("source_id"),
        "contract_name": candidate.get("contract_name"),
        "category": candidate.get("category"),
        "issue_group_count": len(candidate.get("issue_groups") or []),
        "candidate_rui_count": len(candidate.get("candidate_rui_units") or []),
        "status": status,
        "errors": errors,
    }


def _selection_summary(selection: dict[str, Any]) -> dict[str, Any]:
    assessment = selection.get("candidate_assessment") if isinstance(selection.get("candidate_assessment"), dict) else {}
    selected = selection.get("selected_rui_basis") if isinstance(selection.get("selected_rui_basis"), list) else []
    return {
        "feasible": assessment.get("feasible"),
        "basis_size": len(selected),
        "rationale": assessment.get("rationale"),
        "source_rui_ids": [row.get("source_rui_id") for row in selected if isinstance(row, dict)],
        "issue_group_ids": [row.get("issue_group_id") for row in selected if isinstance(row, dict)],
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
        "minimum_sufficient_rui_set_size": audit.get("minimum_sufficient_rui_set_size"),
        "smallest_sufficient_rui_sets": audit.get("smallest_sufficient_rui_sets"),
        "single_unit_shortcut": audit.get("single_unit_shortcut"),
        "single_participant_shortcut": audit.get("single_participant_shortcut"),
        "non_required_unit_ids": audit.get("non_required_unit_ids"),
        "failed_reason": audit.get("failed_reason"),
        "notes": audit.get("notes"),
    }


def _feedback_from_errors(
    errors: list[str],
    *,
    plan: dict[str, Any] | None = None,
    audit: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "instruction": (
            "Revise the previous MAUD measured-RIU design while keeping selected RIU text unchanged. "
            "Fix every validation issue. Make the public finite-choice options balanced and make each "
            "included RIU control an answer boundary."
        ),
        "validation_errors": errors,
    }
    if plan:
        payload["previous_attempt"] = _plan_feedback_summary(plan)
    if audit:
        payload["audit_feedback"] = _audit_summary(audit)
    return "Previous attempt was rejected. Use this JSON feedback:\n" + json.dumps(payload, indent=2, ensure_ascii=False)


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
                "source_rui_id": (unit.get("source_ref") or {}).get("source_rui_id")
                if isinstance(unit.get("source_ref"), dict)
                else None,
                "issue_group_id": (unit.get("source_ref") or {}).get("issue_group_id")
                if isinstance(unit.get("source_ref"), dict)
                else None,
                "information_feature": unit.get("information_feature"),
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
        _counter_key((row.get("audit_summary") or {}).get("minimum_sufficient_rui_set_size"))
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


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _max_rui_count_text(max_rui_count: int | None) -> str:
    if max_rui_count is None:
        return (
            "No hard maximum. Select as many RIUs as can each remain independently "
            "necessary for the generated finite-choice oracle answer."
        )
    return str(max_rui_count)


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _suggested_option_count(candidate_index: int) -> int:
    return 3 + ((candidate_index - 1) % 4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use 'teamgym-generate legal/maud partition --help' to partition accepted trials.",
    )
    parser.add_argument("--output-root", default="generated/maud_measured_rui_funnel")
    parser.add_argument("--cache-dir", default="data/legal_raw")
    parser.add_argument("--trace-root", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-accept-size", type=int, default=2)
    parser.add_argument(
        "--target-min-rui-count",
        type=int,
        default=3,
        help="Prompt-level target for selected PIUs. Acceptance can still allow --min-accept-size.",
    )
    parser.add_argument(
        "--max-rui-count",
        type=int,
        default=0,
        help="Maximum selected PIUs before audit. Use 0 for no hard selected-PIU maximum.",
    )
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--target-accepted", type=int, default=2)
    parser.add_argument("--candidate-start-index", type=int, default=1)
    parser.add_argument("--min-candidate-ruis", type=int, default=2)
    parser.add_argument("--max-issue-groups-per-candidate", type=int, default=10)
    parser.add_argument("--max-candidate-ruis", type=int, default=30)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument(
        "--question-history-path",
        default=None,
        help="Optional JSON/JSONL file of prior same-source question attempts to avoid repeating.",
    )
    parser.add_argument(
        "--acceptance-mode",
        choices=("trim", "strict", "measured"),
        default="trim",
        help=(
            "trim keeps only the audited smallest sufficient PIU set in accepted outputs; "
            "strict rejects any non-required included PIU; measured accepts any minimum sufficient set "
            "above threshold without trimming."
        ),
    )
    parser.add_argument("--no-save-generation-byproducts", action="store_true")
    parser.add_argument("--provider", choices=("openai", "openai_compatible", "bedrock_openai"), default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--endpoint", choices=("/responses", "/chat/completions"), default=DEFAULT_LLM_ENDPOINT)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--credentials-file", default=_default_credentials_file())
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--request-retry-attempts", type=int, default=4)
    parser.add_argument("--request-log-path", default=None)
    parser.add_argument("--no-log-prompts", action="store_true")
    parser.add_argument("--no-log-responses", action="store_true")
    return parser.parse_args()


def _default_credentials_file() -> str | None:
    path = Path("api_key/api_key.py")
    return str(path) if path.is_file() else None


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
    report = run_maud_measured_rui_funnel(
        output_root=output_root,
        cache_dir=args.cache_dir,
        llm_client=client,
        min_accept_size=args.min_accept_size,
        target_min_rui_count=args.target_min_rui_count,
        max_rui_count=args.max_rui_count or None,
        max_candidates=args.max_candidates,
        target_accepted=args.target_accepted,
        split=args.split,
        candidate_start_index=args.candidate_start_index,
        min_candidate_ruis=args.min_candidate_ruis,
        max_issue_groups_per_candidate=args.max_issue_groups_per_candidate,
        max_candidate_ruis=args.max_candidate_ruis,
        parallelism=args.parallelism,
        trace_root=args.trace_root,
        save_generation_byproducts=not args.no_save_generation_byproducts,
        acceptance_mode=args.acceptance_mode,
        question_history_path=args.question_history_path,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
