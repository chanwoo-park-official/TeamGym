"""FinLongDocQA free-response generation, audit, and partition pipeline."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from teamgym.data.finance.finlongdocqa.partition import (
    build_partitioned_datapoint,
)
from teamgym.data.finance.finlongdocqa.packet_budget import (
    PAPER_CONTEXT_BUDGET, PacketBudgetError, materialize_context_cap,
    certify_materialized_views, complete_private_text,
)
from teamgym.core.information_allocation import (
    PAPER_ALLOCATION_MODES,
)
from teamgym.data.finance.finlongdocqa.runtime import (
    REPO_ROOT,
    append_jsonl,
    build_client,
    call_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from teamgym.data.finance.finlongdocqa.schema import (
    validate_common_datapoint,
)
from teamgym.data.finance.finlongdocqa.source import (
    ReportStore,
    blocks_for_pages,
    group_qas_by_report,
    index_report,
    load_qa_rows,
    qa_for_generation,
    validate_report_index,
)
from teamgym.data.finance.finlongdocqa.provenance import (
    source_identity, run_identity, establish_run, read_commits,
    commit_report, rebuild_exports,
)
from teamgym.data.finance.finlongdocqa.validation import (
    hydrate_task,
    consolidated_qa_mappings,
    mapping_block_ids,
    solver_evidence_view,
    summarize_free_response_audit,
    synchronize_task_claim_links,
    validate_evidence_consolidation,
    validate_generated_task,
    validate_judge_result,
    validate_packet_leakage_audit,
    validate_packet_leakage_candidates,
    validate_qa_evidence_mapping,
)


def build_block_index(args: argparse.Namespace) -> dict[str, Any]:
    from tqdm.auto import tqdm

    output = Path(args.output_root).expanduser().resolve()
    rows = load_qa_rows(args.qa_path)
    groups = group_qas_by_report(rows)
    requested_report_ids = _requested_report_ids(args)
    if requested_report_ids:
        by_report_id = dict(groups)
        groups = [
            (report_id, by_report_id[report_id])
            for report_id in requested_report_ids
            if report_id in by_report_id
        ]
    elif args.sample_reports:
        random.Random(args.sample_seed).shuffle(groups)
        groups = groups[: args.sample_reports]
    elif args.max_reports:
        groups = groups[: args.max_reports]
    store = ReportStore(args.reports_zip)
    indexes = []
    failures = []
    try:
        for report_id, qas in tqdm(
            groups,
            desc="Indexing reports",
            unit="report",
            dynamic_ncols=True,
        ):
            company = str(qas[0].get("company") or "")
            year = str(qas[0].get("year") or "")
            try:
                report_text, report_source = store.load(company, year)
                report_index = index_report(report_id, report_text)
                errors = validate_report_index(report_index, report_text)
                if errors:
                    raise ValueError("; ".join(errors))
                report_index["report_source"] = report_source
                indexes.append(report_index)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "report_id": report_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        store.close()
    index_path = output / "report_block_indexes.jsonl"
    write_jsonl(index_path, indexes)
    report = {
        "schema": "finlongdocqa_block_index_report_v1",
        "qa_path": str(Path(args.qa_path).expanduser().resolve()),
        "reports_zip": str(Path(args.reports_zip).expanduser().resolve()),
        "report_count": len(indexes),
        "failure_count": len(failures),
        "block_count": sum(int(row.get("block_count") or 0) for row in indexes),
        "review_block_count": sum(
            1
            for row in indexes
            for block in row.get("blocks") or []
            if block.get("needs_segmentation_review") is True
        ),
        "index_path": str(index_path),
        "failures": failures,
    }
    write_json(output / "block_index_report.json", report)
    return report


def generate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    from tqdm.auto import tqdm

    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    qa_rows = load_qa_rows(args.qa_path)
    qa_groups = dict(group_qas_by_report(qa_rows))
    indexes = read_jsonl(args.block_index_jsonl)
    index_by_report = {
        str(row.get("report_id") or ""): row for row in indexes
    }
    report_ids = [
        report_id for report_id in qa_groups if report_id in index_by_report
    ]
    requested_report_ids = _requested_report_ids(args)
    if requested_report_ids:
        available = set(report_ids)
        report_ids = [
            report_id
            for report_id in requested_report_ids
            if report_id in available
        ]
    elif args.sample_reports:
        random.Random(args.sample_seed).shuffle(report_ids)
        report_ids = report_ids[: args.sample_reports]
    elif args.max_reports:
        report_ids = report_ids[: args.max_reports]

    if requested_report_ids and set(requested_report_ids) != set(report_ids):
        raise ValueError("requested reports absent from source QA/index")
    if len(index_by_report) != len(indexes):
        raise ValueError("duplicate report IDs in block index")
    if not report_ids:
        raise ValueError("no reports selected from QA and index")
    sources = source_identity(args, qa_groups, index_by_report, report_ids)
    identity = run_identity(args, sources, report_ids)
    establish_run(output, identity)
    commits, invalid_commits = read_commits(output, identity["fingerprint"], report_ids)
    attempts, accepted, common_rows = rebuild_exports(output, commits)
    accepted_ids = {str(row["report_id"]) for row in accepted}
    client = None
    attempts_path = output / "attempts.jsonl"
    accepted_path = output / "accepted.jsonl"
    common_path = output / "common_datapoints.jsonl"

    progress = tqdm(
        report_ids,
        desc="Generating and auditing",
        unit="report",
        dynamic_ncols=True,
    )
    for position, report_id in enumerate(progress, start=1):
        if report_id in accepted_ids:
            continue
        if client is None:
            client = build_client(args, output / "_llm_requests.jsonl")
        commit_paths = list((output / "_commits" / report_id).glob("*.json"))
        attempt_number = max([int(path.stem) for path in commit_paths if path.stem.isdigit()] or [0]) + 1
        qas = qa_groups[report_id]
        report_index = index_by_report[report_id]
        trace_root = output / "_llm_traces" / report_id / f"attempt_{attempt_number:05d}"
        try:
            result, common = process_report(
                report_id=report_id,
                qas=qas,
                report_index=report_index,
                client=client,
                trace_root=trace_root,
                generation_attempts=args.generation_attempts,
                audit_workers=args.audit_workers,
                packet_repair_rounds=args.packet_repair_rounds,
                packet_audit_max_chars=args.packet_audit_max_chars,
                participant_context_token_budget=getattr(args, "participant_context_token_budget", PAPER_CONTEXT_BUDGET),
                allocation_mode=args.allocation_mode,
                allocation_seed=args.sample_seed,
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "schema": "finlongdocqa_report_attempt_v1",
                "report_id": report_id,
                "status": "failed",
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
            common = None
        result["run_fingerprint"] = identity["fingerprint"]
        if common is not None:
            common.setdefault("generation_metadata", {})["run_fingerprint"] = identity["fingerprint"]
        committed = commit_report(output, identity["fingerprint"], report_id, attempt_number, result, common)
        commits.append(committed)
        attempts, accepted, common_rows = rebuild_exports(output, commits)
        progress.set_postfix(
            attempted=len(attempts),
            accepted=len(accepted),
            status=str(result.get("status") or "unknown"),
        )
        tqdm.write(
            json.dumps(
                {
                    "event": "report_done",
                    "position": position,
                    "report_id": report_id,
                    "status": result.get("status"),
                    "accepted": len(accepted),
                    "attempted": len(attempts),
                },
                ensure_ascii=False,
            )
        )

    report = {
        "schema": "finlongdocqa_generation_report_v1",
        "answer_format": "free_response",
        "run_fingerprint": identity["fingerprint"],
        "invalid_commits": invalid_commits,
        "allocation_mode": args.allocation_mode,
        "report_count": len(report_ids),
        "attempt_count": len(attempts),
        "accepted_count": len(accepted),
        "status_counts": _counts(
            str(row.get("status") or "unknown") for row in attempts
        ),
        "attempts_path": str(attempts_path),
        "accepted_path": str(accepted_path),
        "common_path": str(common_path),
    }
    write_json(output / "generation_report.json", report)
    return report


def process_report(
    *,
    report_id: str,
    qas: list[dict[str, Any]],
    report_index: dict[str, Any],
    client: Any,
    trace_root: Path,
    generation_attempts: int,
    audit_workers: int,
    packet_repair_rounds: int,
    packet_audit_max_chars: int,
    allocation_mode: str,
    allocation_seed: int,
    participant_context_token_budget: int = PAPER_CONTEXT_BUDGET,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    blocks_by_id = {
        str(block.get("block_id") or ""): block
        for block in report_index.get("blocks") or []
    }
    unit_conventions = [r['text'] for r in report_index.get('source_unit_conventions', [])]
    mappings = []
    rejected_mappings = []
    for qa in qas:
        qa_public = qa_for_generation(qa)
        qa_public['report_unit_conventions'] = unit_conventions
        available = blocks_for_pages(
            report_index,
            qa_public["page_numbers"],
        )
        mapping = call_json(
            client,
            system_prompt_name="generation_system.txt",
            user_prompt_name="01_qa_evidence_map_user.txt",
            variables={
                "SOURCE_QA_JSON": qa_public,
                "ANNOTATED_PAGE_BLOCKS_JSON": available,
            },
            trace_dir=trace_root / "qa_mapping" / qa_public["qa_id"],
        )
        errors = validate_qa_evidence_mapping(
            mapping,
            qa=qa_public,
            available_blocks=available,
        )
        if errors:
            rejected_mappings.append(
                {
                    "qa": qa_public,
                    "mapping": mapping,
                    "errors": errors,
                }
            )
        else:
            mappings.append(mapping)

    candidate_block_ids = list(
        dict.fromkeys(
            str(block_id)
            for mapping in mappings
            for block_id in mapping_block_ids(mapping)
        )
    )
    if len(candidate_block_ids) < 2:
        return (
            {
                "schema": "finlongdocqa_report_attempt_v1",
                "report_id": report_id,
                "status": "candidate_pool_rejected",
                "errors": [
                    "fewer than two distinct candidate blocks remain after "
                    + "dropping invalid QA mappings"
                ],
                "qa_mappings": mappings,
                "rejected_qa_mappings": rejected_mappings,
                "evidence_candidate_block_ids": candidate_block_ids,
            },
            None,
        )

    mapped_candidate_blocks = [
        blocks_by_id[block_id] for block_id in candidate_block_ids
    ]
    valid_qa_ids = {
        str(mapping.get("qa_id") or "") for mapping in mappings
    }
    valid_qas = [
        public_qa
        for qa in qas
        if (public_qa := qa_for_generation(qa))["qa_id"] in valid_qa_ids
    ]

    consolidation, consolidation_rows, consolidation_errors = (
        run_evidence_consolidation(
            report_id=report_id,
            qas=valid_qas,
            mappings=mappings,
            candidate_blocks=mapped_candidate_blocks,
            client=client,
            trace_root=trace_root / "evidence_consolidation",
            max_attempts=generation_attempts,
        )
    )
    if consolidation_errors:
        return (
            {
                "schema": "finlongdocqa_report_attempt_v1",
                "report_id": report_id,
                "status": "evidence_consolidation_rejected",
                "errors": consolidation_errors,
                "qa_mappings": mappings,
                "rejected_qa_mappings": rejected_mappings,
                "evidence_candidate_block_ids": candidate_block_ids,
                "evidence_consolidation_attempts": consolidation_rows,
            },
            None,
        )
    retained_ids = {
        str(value)
        for value in consolidation.get("retained_block_ids") or []
    }
    candidate_blocks = [
        block
        for block in mapped_candidate_blocks
        if str(block.get("block_id") or "") in retained_ids
    ]

    mappings = consolidated_qa_mappings(mappings, consolidation)

    feedback: dict[str, Any] | str = "None."
    generation_rows = []
    for attempt in range(1, max(1, generation_attempts) + 1):
        task = call_json(
            client,
            system_prompt_name="generation_system.txt",
            user_prompt_name="02_task_generate_user.txt",
            variables={
                "REPORT_ID": report_id,
                "SOURCE_QAS_JSON": [{**qa, 'report_unit_conventions': unit_conventions} for qa in valid_qas],
                "QA_EVIDENCE_MAPPINGS_JSON": mappings,
                "EVIDENCE_CONSOLIDATION_JSON": consolidation,
                "EVIDENCE_CANDIDATE_BLOCKS_JSON": candidate_blocks,
                "REJECTION_FEEDBACK": feedback,
            },
            trace_dir=trace_root / f"task_attempt_{attempt:02d}",
        )
        task = hydrate_task(task, candidate_blocks)
        if unit_conventions:
            task['public_task']['source_unit_conventions'] = unit_conventions
        else:
            task['public_task'].pop('source_unit_conventions', None)
        task_errors = validate_generated_task(
            task,
            candidate_blocks=candidate_blocks,
            qa_mappings=mappings,
        )
        row = {
            "attempt": attempt,
            "task": task,
            "task_errors": task_errors,
        }
        generation_rows.append(row)
        if task_errors:
            feedback = {"stage": "task_validation", "errors": task_errors}
            continue
        solver_audit, solver_errors = run_blind_solver_audit(
            client=client,
            task=task,
            trace_root=trace_root / f"task_attempt_{attempt:02d}_solver",
            max_workers=audit_workers,
        )
        row["solver_audit"] = solver_audit
        row["solver_errors"] = solver_errors
        minimization_rows = []
        while solver_errors:
            repaired_task, repair = minimize_redundant_riu(
                task=task,
                solver_audit=solver_audit,
                solver_errors=solver_errors,
                candidate_blocks=candidate_blocks,
                qa_mappings=mappings,
                consolidation=consolidation,
            )
            if repaired_task is None:
                break
            repaired_audit, repaired_errors = run_blind_solver_audit(
                client=client,
                task=repaired_task,
                trace_root=(
                    trace_root
                    / f"task_attempt_{attempt:02d}_solver"
                    / f"minimize_{len(minimization_rows) + 1:02d}"
                ),
                max_workers=audit_workers,
            )
            minimization_rows.append(
                {
                    **repair,
                    "task": repaired_task,
                    "solver_audit": repaired_audit,
                    "solver_errors": repaired_errors,
                }
            )
            task = repaired_task
            solver_audit = repaired_audit
            solver_errors = repaired_errors
        row["task"] = task
        row["solver_audit"] = solver_audit
        row["solver_errors"] = solver_errors
        row["riu_minimization"] = minimization_rows
        if solver_errors:
            feedback = {
                "stage": "free_response_solver_and_judge",
                "errors": solver_errors,
            }
            continue

        final_block_ids = {
            str(unit.get("source_block_id") or "")
            for unit in task.get("evidence_units") or []
            if isinstance(unit, dict)
        }

        company = str(qas[0].get("company") or "")
        year = str(qas[0].get("year") or "")
        common = build_partitioned_datapoint(
            report_id=report_id,
            company=company,
            year=year,
            report_source=report_index.get("report_source") or {},
            report_index=report_index,
            task=task,
            qa_mappings=mappings,
            solver_audit=solver_audit,
            allocation_mode=allocation_mode,
            allocation_seed=allocation_seed,
            preserve_rui_order=allocation_mode == "one_each",
        )
        try:
            materialize_context_cap(common, token_budget=participant_context_token_budget)
        except PacketBudgetError as exc:
            row["packet_budget_errors"] = [str(exc)]
            feedback = {"stage": "canonical_preserving_context_budget", "errors": row["packet_budget_errors"]}
            continue
        (
            common,
            packet_audits,
            removed_context_ids,
            packet_errors,
        ) = run_packet_leakage_repair(
            client=client,
            datapoint=common,
            public_task=task.get("public_task") or {},
            trace_root=(
                trace_root
                / f"task_attempt_{attempt:02d}_packet_leakage"
            ),
            max_workers=audit_workers,
            max_rounds=packet_repair_rounds,
            max_prompt_chars=packet_audit_max_chars,
            task=task, report_index=report_index,
        )
        row["packet_leakage_audits"] = packet_audits
        row["packet_leakage_errors"] = packet_errors
        row["removed_leaking_context_unit_ids"] = sorted(
            removed_context_ids
        )
        if packet_errors:
            feedback = {
                "stage": "packet_leakage_audit",
                "errors": packet_errors,
            }
            continue
        if common.get("generation_metadata", {}).get("canonical_table_restrictions"):
            # Row changes alter both the canonical PIU and actual private view;
            # previous qualification cannot certify the restricted bundle.
            revised_errors = validate_generated_task(task, candidate_blocks=candidate_blocks, qa_mappings=mappings)
            revised_audit, revised_solver_errors = run_blind_solver_audit(client=client, task=task,
                trace_root=trace_root / f"task_attempt_{attempt:02d}_restricted_solver", max_workers=audit_workers)
            row["restricted_solver_audit"] = revised_audit
            row["restricted_solver_errors"] = revised_errors + revised_solver_errors
            if revised_errors or revised_solver_errors:
                row["solver_errors"] = revised_errors + revised_solver_errors
                feedback = {"stage": "restricted_table_grounding", "errors": row["solver_errors"]}
                continue
            solver_audit = revised_audit
            common["generation_metadata"]["solver_audit"] = revised_audit
        try:
            certify_materialized_views(common, token_budget=participant_context_token_budget)
        except PacketBudgetError as exc:
            row["packet_budget_errors"] = [str(exc)]
            feedback = {"stage": "final_exact_consumer_budget", "errors": row["packet_budget_errors"]}
            continue
        salience_rows, salience_errors = run_packet_salience_audit(client=client, datapoint=common,
            trace_root=trace_root / f"task_attempt_{attempt:02d}_salience", max_workers=audit_workers,
            context_token_budget=participant_context_token_budget)
        row["packet_salience_audits"], row["packet_salience_errors"] = salience_rows, salience_errors
        common["generation_metadata"]["packet_salience_audits"] = salience_rows
        if salience_errors:
            feedback = {"stage": "actual_initial_input_salience", "errors": salience_errors}
            continue
        local_rows, local_errors = run_initial_view_audit(client=client, datapoint=common, task=task,
            trace_root=trace_root / f"task_attempt_{attempt:02d}_initial_view", max_workers=audit_workers,
            context_token_budget=participant_context_token_budget)
        row["initial_view_audits"] = local_rows
        row["initial_view_errors"] = local_errors
        common["generation_metadata"]["initial_view_audits"] = local_rows
        if local_errors:
            feedback = {"stage": "initial_public_plus_local", "errors": local_errors}
            continue
        result = {
            "schema": "finlongdocqa_report_attempt_v1",
            "report_id": report_id,
            "company": company,
            "year": year,
            "status": "accepted",
            "errors": [],
            "qa_mappings": mappings,
            "rejected_qa_mappings": rejected_mappings,
            "evidence_candidate_block_ids": candidate_block_ids,
            "consolidated_candidate_block_ids": [
                str(block.get("block_id") or "") for block in candidate_blocks
            ],
            "evidence_consolidation": consolidation,
            "evidence_consolidation_attempts": consolidation_rows,
            "final_riu_block_ids": sorted(final_block_ids),
            "task": task,
            "solver_audit": solver_audit,
            "packet_leakage_audits": packet_audits,
            "initial_view_audits": local_rows,
            "packet_salience_audits": salience_rows,
            "source_packet_materialization": common["generation_metadata"]["source_packet_materialization"],
            "removed_leaking_context_unit_ids": sorted(
                removed_context_ids
            ),
            "generation_attempts": generation_rows,
        }
        return result, common

    last = generation_rows[-1] if generation_rows else {}
    return (
        {
            "schema": "finlongdocqa_report_attempt_v1",
            "report_id": report_id,
            "status": (
                "packet_budget_rejected"
                if last.get("packet_budget_errors")
                else "packet_salience_rejected"
                if last.get("packet_salience_errors")
                else "initial_view_rejected"
                if last.get("initial_view_errors")
                else "solver_rejected"
                if last.get("solver_errors")
                else "packet_leakage_rejected"
                if last.get("packet_leakage_errors")
                else "task_generation_rejected"
            ),
            "errors": (
                last.get("packet_budget_errors")
                or last.get("packet_salience_errors")
                or last.get("initial_view_errors")
                or last.get("solver_errors")
                or last.get("packet_leakage_errors")
                or last.get("task_errors")
                or ["no valid task"]
            ),
            "qa_mappings": mappings,
            "rejected_qa_mappings": rejected_mappings,
            "evidence_candidate_block_ids": candidate_block_ids,
            "consolidated_candidate_block_ids": [
                str(block.get("block_id") or "") for block in candidate_blocks
            ],
            "evidence_consolidation": consolidation,
            "evidence_consolidation_attempts": consolidation_rows,
            "generation_attempts": generation_rows,
        },
        None,
    )


def run_evidence_consolidation(
    *,
    report_id: str,
    qas: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    candidate_blocks: list[dict[str, Any]],
    client: Any,
    trace_root: Path,
    max_attempts: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Identify fully subsumed evidence while preserving unique contributions."""

    feedback: dict[str, Any] | str = "None."
    rows: list[dict[str, Any]] = []
    last_result: dict[str, Any] = {}
    last_errors: list[str] = ["consolidation was not attempted"]
    for attempt in range(1, max(1, max_attempts) + 1):
        result = call_json(
            client,
            system_prompt_name="generation_system.txt",
            user_prompt_name="02_evidence_consolidate_user.txt",
            variables={
                "REPORT_ID": report_id,
                "SOURCE_QAS_JSON": qas,
                "QA_EVIDENCE_MAPPINGS_JSON": mappings,
                "EVIDENCE_CANDIDATE_BLOCKS_JSON": candidate_blocks,
                "CONSOLIDATION_FEEDBACK": feedback,
            },
            trace_dir=trace_root / f"attempt_{attempt:02d}",
        )
        errors = validate_evidence_consolidation(
            result,
            report_id=report_id,
            candidate_blocks=candidate_blocks,
        )
        rows.append(
            {
                "attempt": attempt,
                "result": result,
                "errors": errors,
            }
        )
        last_result = result
        last_errors = errors
        if not errors:
            return result, rows, []
        feedback = {
            "stage": "evidence_consolidation_validation",
            "errors": errors,
        }
    return last_result, rows, last_errors


def minimize_redundant_riu(
    *,
    task: dict[str, Any],
    solver_audit: dict[str, Any],
    solver_errors: list[str],
    candidate_blocks: list[dict[str, Any]],
    qa_mappings: list[dict[str, Any]],
    consolidation: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Drop one LOO-proven redundant RIU and relink claims to cited evidence."""

    if not _full_and_none_are_safe(solver_audit):
        return None, {}
    if not solver_errors or any(
        not (
            "leave-one-out response remained complete" in error
            or "all linked claims remained correct and grounded" in error
        )
        for error in solver_errors
    ):
        return None, {}
    redundant = [
        row
        for row in solver_audit.get("leave_one_out") or []
        if isinstance(row, dict) and not row.get("broken_claim_ids")
    ]
    if not redundant:
        return None, {}

    preferred = {
        str(group.get("preferred_block_id") or "")
        for group in consolidation.get("overlap_groups") or []
        if isinstance(group, dict)
    }
    unit_by_id = {
        str(unit.get("unit_id") or ""): unit
        for unit in task.get("evidence_units") or []
        if isinstance(unit, dict)
    }
    redundant.sort(
        key=lambda row: (
            str(
                (unit_by_id.get(str(row.get("omitted_unit_id") or "")) or {}).get(
                    "source_block_id"
                )
                or ""
            )
            in preferred,
            str(row.get("omitted_unit_id") or ""),
        )
    )
    for loo_row in redundant:
        omitted = str(loo_row.get("omitted_unit_id") or "")
        remaining = [
            unit
            for unit in task.get("evidence_units") or []
            if isinstance(unit, dict)
            and str(unit.get("unit_id") or "") != omitted
        ]
        if len(remaining) < 2:
            continue
        number_to_unit = {
            index: str(unit.get("unit_id") or "")
            for index, unit in enumerate(remaining, start=1)
        }
        repaired = json.loads(json.dumps(task))
        repaired["evidence_units"] = [
            unit
            for unit in repaired.get("evidence_units") or []
            if str(unit.get("unit_id") or "") != omitted
        ]
        judged_claims = {
            str(row.get("claim_id") or ""): row
            for row in (loo_row.get("judge_result") or {}).get(
                "claim_results", []
            )
            if isinstance(row, dict)
        }
        for claim in (repaired.get("rubric") or {}).get(
            "required_claims", []
        ):
            if not isinstance(claim, dict):
                continue
            supporting = [
                str(value)
                for value in claim.get("supporting_unit_ids") or []
                if str(value) != omitted
            ]
            judged = judged_claims.get(str(claim.get("claim_id") or "")) or {}
            cited = [
                number_to_unit.get(int(value))
                for value in judged.get("supporting_evidence_numbers") or []
                if str(value).isdigit()
            ]
            claim["supporting_unit_ids"] = list(
                dict.fromkeys(
                    supporting + [value for value in cited if value]
                )
            )
        synchronize_task_claim_links(repaired)
        validation_errors = validate_generated_task(
            repaired,
            candidate_blocks=candidate_blocks,
            qa_mappings=qa_mappings,
        )
        if validation_errors:
            continue
        return (
            repaired,
            {
                "action": "remove_loo_redundant_riu",
                "removed_unit_id": omitted,
                "reason": (
                    "all claims linked to the omitted RIU remained correct and "
                    "grounded; claim links were rebuilt from judge citations"
                ),
            },
        )
    return None, {}


def _full_and_none_are_safe(audit: dict[str, Any]) -> bool:
    full_judge = (audit.get("full_evidence") or {}).get("judge_result") or {}
    none_judge = (audit.get("no_evidence") or {}).get("judge_result") or {}
    none_claims = [
        row
        for row in none_judge.get("claim_results") or []
        if isinstance(row, dict)
    ]
    return (
        full_judge.get("overall_complete_and_correct") is True
        and none_judge.get("overall_complete_and_correct") is not True
        and not any(
            row.get("status") == "satisfied"
            and row.get("content_correct") is True
            and row.get("grounded_in_available_material") is True
            for row in none_claims
        )
    )


def run_blind_solver_audit(
    *,
    client: Any,
    task: dict[str, Any],
    trace_root: Path,
    max_workers: int = 8,
) -> tuple[dict[str, Any], list[str]]:
    public_task = task.get("public_task") or {}
    rubric = task.get("rubric") or {}
    reference_answer = str(
        (task.get("oracle") or {}).get("reference_answer") or ""
    )
    units = [
        row for row in task.get("evidence_units") or [] if isinstance(row, dict)
    ]

    def solve_and_judge(
        available: list[dict[str, Any]],
        trace_name: str,
    ) -> dict[str, Any]:
        evidence_view = solver_evidence_view(available)
        solver_response = call_json(
            client,
            system_prompt_name="blind_solver_system.txt",
            user_prompt_name="03_blind_solver_user.txt",
            variables={
                "PUBLIC_TASK_JSON": public_task,
                "AVAILABLE_EVIDENCE_JSON": evidence_view,
            },
            trace_dir=trace_root / trace_name / "solver",
        )
        judge_result = call_json(
            client,
            system_prompt_name="free_response_judge_system.txt",
            user_prompt_name="04_free_response_judge_user.txt",
            variables={
                "PUBLIC_TASK_JSON": public_task,
                "RUBRIC_JSON": rubric,
                "REFERENCE_ANSWER": reference_answer,
                "AVAILABLE_EVIDENCE_JSON": evidence_view,
                "SOLVER_RESPONSE_JSON": solver_response,
            },
            trace_dir=trace_root / trace_name / "judge",
        )
        return {
            "solver_response": solver_response,
            "judge_result": judge_result,
        }

    conditions = [
        {
            "kind": "full",
            "omitted_unit_id": "",
            "available": units,
            "trace_name": "full",
        },
        {
            "kind": "none",
            "omitted_unit_id": "",
            "available": [],
            "trace_name": "none",
        },
        *[
            {
                "kind": "leave_one_out",
                "omitted_unit_id": str(unit.get("unit_id") or ""),
                "available": [
                    row
                    for row in units
                    if str(row.get("unit_id") or "")
                    != str(unit.get("unit_id") or "")
                ],
                "trace_name": "without_"
                + str(unit.get("unit_id") or ""),
            }
            for unit in units
        ],
    ]

    def run_condition(condition: dict[str, Any]) -> dict[str, Any]:
        return {
            **condition,
            **solve_and_judge(
                condition["available"],
                str(condition["trace_name"]),
            ),
        }

    with ThreadPoolExecutor(
        max_workers=max(1, min(max_workers, len(conditions)))
    ) as executor:
        condition_results = list(executor.map(run_condition, conditions))
    full = condition_results[0]
    none = condition_results[1]
    leave_one_out = [
        {
            "omitted_unit_id": str(row["omitted_unit_id"]),
            "solver_response": row["solver_response"],
            "judge_result": row["judge_result"],
        }
        for row in condition_results[2:]
    ]
    return summarize_free_response_audit(
        task=task,
        full_result=full,
        no_evidence_result=none,
        leave_one_out_results=leave_one_out,
    )




def run_packet_salience_audit(*, client, datapoint, trace_root, max_workers=8,
                             context_token_budget=PAPER_CONTEXT_BUDGET):
    """Audit construction cues in the actual full-context actor request."""
    from teamgym.interaction.llm_actors import LLMParticipantActor, PromptTraceRecorder
    from teamgym.interaction.types import BudgetSpec
    from .provenance import digest
    def audit(participant):
        class CaptureClient:
            last_request_meta = {"capture_only": True}
            def generate_json(self, **kwargs):
                return {"utterance": "Initial input capture."}
        recorder = PromptTraceRecorder()
        LLMParticipantActor(CaptureClient(), recorder).respond(datapoint=datapoint,
            participant_id=participant["participant_id"], state={"turn": 0, "messages": []}, request=None,
            budgets=BudgetSpec(participant_context_token_budget=context_token_budget))
        actual = {key: recorder.prompt_records[0][key] for key in ("system_prompt", "user_prompt")}
        payload = json.loads(actual["user_prompt"])
        if payload["private_source_material"] != complete_private_text(datapoint, participant):
            return {"participant_id": participant["participant_id"]}, ["salience input was truncated by actual consumer"]
        result = call_json(client, system_prompt_name="packet_leakage_audit_system.txt",
            user_prompt_name="07_actual_input_salience_user.txt", variables={"ACTUAL_MODEL_INPUT_JSON": actual},
            trace_dir=trace_root / participant["participant_id"])
        categories = ("leaks_constructor_metadata", "reveals_answer_in_identity", "marks_private_piu_as_important")
        errors = []
        if not isinstance(result, dict) or set(result) != {*categories, "rationale"}:
            errors.append("invalid actual-input salience result schema")
        else:
            if any(type(result.get(key)) is not bool for key in categories) or not isinstance(result["rationale"], str) or not result["rationale"].strip():
                errors.append("invalid actual-input salience assessment types")
            elif any(result[key] for key in categories):
                errors.append("actual model input contains artificial construction/answer salience")
        return {"participant_id": participant["participant_id"], "context_token_budget": context_token_budget,
                "actual_model_input_sha256": digest(actual), "audit": result, "errors": errors}, errors
    participants = datapoint["participants"]
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(participants)))) as executor:
        rows = list(executor.map(audit, participants))
    return [row for row, _ in rows], [f"{row['participant_id']}: {error}" for row, errors in rows for error in errors]


def run_initial_view_audit(*, client, datapoint, task, trace_root, max_workers=8, context_token_budget=PAPER_CONTEXT_BUDGET):
    """Blind complete-answer checks on the actual consumer's public+local view.

    Use the same fixed context budget as materialization and the actual actor.
    Any source truncation or budget mismatch is an invalid audit.
    """
    from teamgym.interaction.llm_actors import _public_task_payload, _private_context_payload
    from teamgym.interaction.rui_context import rui_only_datapoint
    from teamgym.interaction.types import BudgetSpec
    from teamgym.tokenization import count_tokens
    from .provenance import digest
    variants = {"multi_agent_full_context": datapoint, "multi_agent_piu_only": rui_only_datapoint(datapoint)}
    jobs = [(mode, view, participant) for mode, view in variants.items() for participant in view["participants"]]
    claim_ids = {row["claim_id"] for row in task["rubric"]["required_claims"]}
    def audit(job):
        mode, view, participant = job
        public = _public_task_payload(view)
        full_text = complete_private_text(view, participant)
        budget = context_token_budget
        recorded_budget = datapoint.get("generation_metadata", {}).get("source_packet_materialization", {}).get("context_token_budget", budget)
        if recorded_budget != budget:
            return {"mode": mode, "participant_id": participant["participant_id"]}, ["consumer budget differs from materialization budget"]
        private = _private_context_payload(view, participant, BudgetSpec(participant_context_token_budget=budget))
        if private != full_text:
            return {"mode": mode, "participant_id": participant["participant_id"]}, ["complete private consumer input was truncated"]
        evidence = [{"evidence_number": 1, "text": private}] if private else []
        trace = trace_root / mode / participant["participant_id"]
        solver = call_json(client, system_prompt_name="blind_solver_system.txt", user_prompt_name="03_blind_solver_user.txt",
            variables={"PUBLIC_TASK_JSON": public, "AVAILABLE_EVIDENCE_JSON": evidence}, trace_dir=trace / "solver")
        judge = call_json(client, system_prompt_name="free_response_judge_system.txt", user_prompt_name="04_free_response_judge_user.txt",
            variables={"PUBLIC_TASK_JSON": public, "RUBRIC_JSON": task["rubric"],
                "REFERENCE_ANSWER": task["oracle"]["reference_answer"], "AVAILABLE_EVIDENCE_JSON": evidence,
                "SOLVER_RESPONSE_JSON": solver}, trace_dir=trace / "judge")
        errors = validate_judge_result(judge, claim_ids=claim_ids, available_evidence_count=len(evidence))
        if judge.get("overall_complete_and_correct") is True:
            errors.append("participant common+local view answered the complete task")
        return {"mode": mode, "participant_id": participant["participant_id"],
                "consumer_public_input": public, "consumer_private_input": private,
                "consumer_input_sha256": digest({"public": public, "private": private}),
                "context_token_budget": budget, "solver_response": solver, "judge_result": judge,
                "limitations": "One blind model execution, not a proof of impossibility."}, errors
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(jobs)))) as executor:
        rows = list(executor.map(audit, jobs))
    return [row for row, _ in rows], [f"{row['mode']}/{row['participant_id']}: {error}"
        for row, errors in rows for error in errors]


def run_packet_leakage_repair(
    *,
    client: Any,
    datapoint: dict[str, Any],
    public_task: dict[str, Any],
    trace_root: Path,
    max_workers: int = 8,
    max_rounds: int = 3,
    max_prompt_chars: int = 500_000,
    task: dict[str, Any] | None = None,
    report_index: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str], list[str]]:
    """Remove non-canonical contexts that collectively leak foreign RIUs."""

    all_audits: list[dict[str, Any]] = []
    audit_cache: dict[str, tuple[dict, str]] = {}
    removed_ids: set[str] = set()
    errors: list[str] = []

    final_audit_round = max(1, max_rounds) + 1
    for round_number in range(1, final_audit_round + 1):
        participants = [
            row
            for row in datapoint.get("participants") or []
            if isinstance(row, dict)
        ]
        contexts = [
            row
            for row in datapoint.get("private_context_units") or []
            if isinstance(row, dict)
        ]
        rius = [
            row
            for row in datapoint.get("reference_information_units") or []
            if isinstance(row, dict)
        ]

        def audit_participant(
            participant: dict[str, Any],
            current_rius: list[dict[str, Any]] = rius,
            current_contexts: list[dict[str, Any]] = contexts,
            current_round: int = round_number,
        ) -> tuple[dict[str, Any], list[str]]:
            participant_id = str(participant.get("participant_id") or "")
            owned = {
                str(value) for value in participant.get("owned_rui_ids") or []
            }
            shared = {
                str(row.get("rui_id") or "")
                for row in (
                    datapoint.get("initial_common_information") or {}
                ).get("common_reference_units", [])
                if isinstance(row, dict) and row.get("rui_id")
            }
            authorized = owned | shared
            foreign = [
                row
                for row in current_rius
                if str(row.get("rui_id") or "") not in authorized
            ]
            packet = [
                row
                for row in current_contexts
                if participant_id
                in {str(value) for value in row.get("assigned_to") or []}
            ]
            canonical_context_ids = {
                str(row.get("context_unit_id") or "")
                for row in packet
                if row.get("hidden_contains_rui_ids")
            }
            participant_public = {
                "participant_id": participant_id,
                "owned_rui_ids": sorted(owned),
                "shared_rui_ids": sorted(shared),
                "assigned_or_shared_rui_ids": sorted(authorized),
            }
            participant_trace = (
                trace_root
                / f"round_{current_round:02d}"
                / participant_id
            )
            input_hash = None
            origin_trace = str(participant_trace)
            if _serialized_chars(packet) <= max_prompt_chars:
                from .runtime import _prompt_text, _render
                variables = {"PUBLIC_TASK_JSON": public_task, "PARTICIPANT_JSON": participant_public,
                    "FOREIGN_RIUS_JSON": foreign, "PACKET_CONTEXTS_JSON": packet}
                rendered = _prompt_text("packet_leakage_audit_system.txt") + "\n" + _render(
                    _prompt_text("05_packet_leakage_audit_user.txt"), variables)
                input_hash = hashlib.sha256(rendered.encode()).hexdigest()
                if input_hash in audit_cache:
                    audit, origin_trace = deepcopy(audit_cache[input_hash])
                    audit_mode = "direct_identical_input_reuse"
                else:
                    audit = call_json(client, system_prompt_name="packet_leakage_audit_system.txt",
                        user_prompt_name="05_packet_leakage_audit_user.txt", variables=variables,
                        trace_dir=participant_trace)
                    audit_cache[input_hash] = (deepcopy(audit), str(participant_trace))
                    audit_mode = "direct"
                candidate_errors: list[str] = []
            else:
                audit, candidate_errors = _reject_oversize_packet(
                    participant_id=participant_id,
                    packet=packet,
                    max_prompt_chars=max_prompt_chars,
                )
                audit_mode = "complete_view_over_budget_rejected"
            audit_errors = validate_packet_leakage_audit(
                audit,
                participant_id=participant_id,
                packet_context_unit_ids={
                    str(row.get("context_unit_id") or "") for row in packet
                },
                canonical_context_unit_ids=canonical_context_ids,
                foreign_rui_ids={
                    str(row.get("rui_id") or "") for row in foreign
                },
            )
            return (
                {
                    "round": current_round,
                    "participant_id": participant_id,
                    "audit_mode": audit_mode,
                    "audit_input_sha256": input_hash,
                    "audit_origin_trace": origin_trace,
                    "audit": audit,
                },
                [
                    f"{participant_id}: {error}"
                    for error in candidate_errors + audit_errors
                ],
            )

        with ThreadPoolExecutor(
            max_workers=max(1, min(max_workers, len(participants)))
        ) as executor:
            round_rows = list(executor.map(audit_participant, participants))
        all_audits.extend(row[0] for row in round_rows)
        round_errors = [error for row in round_rows for error in row[1]]
        if round_errors:
            errors.extend(round_errors)
            break

        removable_ids = {
            str(context_id)
            for row, _ in round_rows
            for primary in (row.get("audit") or {}).get("foreign_rui_results", [])
            if isinstance(primary, dict)
            for result in _reconstruction_routes(primary)
            if result.get("reconstructable_from_packet") is True
            for context_id in result.get(
                "removable_noncanonical_context_ids", []
            )
        }
        canonical_overlaps = {
            (
                str(row.get("participant_id") or ""),
                str(result.get("rui_id") or ""),
            )
            for row, _ in round_rows
            for primary in (row.get("audit") or {}).get("foreign_rui_results", [])
            if isinstance(primary, dict)
            for result in _reconstruction_routes(primary)
            if result.get("reconstructable_from_packet") is True
            and result.get("canonical_overlap") is True
        }
        if canonical_overlaps and task is not None and report_index is not None and round_number < final_audit_round:
            from .table_repair import repair_canonical_tables
            changed, repair_errors = repair_canonical_tables(client=client, datapoint=datapoint,
                task=task, report_index=report_index, canonical_overlaps=canonical_overlaps,
                trace_root=trace_root / f"round_{round_number:02d}" / "canonical_table_repair")
            if changed:
                continue
            errors.extend(repair_errors)
        if canonical_overlaps:
            errors.append(
                "canonical participant context reconstructs a foreign RIU: "
                + ", ".join(
                    f"{participant_id}/{rui_id}"
                    for participant_id, rui_id in sorted(canonical_overlaps)
                )
            )
            break
        reconstructable = any(
            result.get("reconstructable_from_packet") is True
            for row, _ in round_rows
            for result in (row.get("audit") or {}).get(
                "foreign_rui_results", []
            )
            if isinstance(result, dict)
        )
        if not reconstructable:
            break
        if not removable_ids:
            errors.append(
                "foreign RIU remained reconstructable without a removable "
                "non-canonical context"
            )
            break
        if round_number == final_audit_round:
            errors.append("packet leakage remained after maximum repair rounds")
            break
        _remove_contexts(datapoint, removable_ids)
        removed_ids.update(removable_ids)

    metadata = datapoint.setdefault("generation_metadata", {})
    metadata["packet_leakage_audits"] = all_audits
    metadata["removed_leaking_context_unit_ids"] = sorted(removed_ids)
    datapoint["format_validation"] = validate_common_datapoint(datapoint)
    if not datapoint["format_validation"].get("valid"):
        errors.append(
            f"repaired datapoint is structurally invalid: "
            f"{datapoint['format_validation']}"
        )
    return datapoint, all_audits, removed_ids, list(dict.fromkeys(errors))



def _reconstruction_routes(result):
    return [result, *[{"rui_id": result["rui_id"], "reconstructable_from_packet": True, **route}
                     for route in result.get("additional_reconstruction_routes") or []]]


def _reject_oversize_packet(
    *,
    participant_id: str,
    packet: list[dict[str, Any]],
    max_prompt_chars: int,
) -> tuple[dict[str, Any], list[str]]:
    """Refuse incomplete long-packet audits instead of accepting a shortlist.

    Increase the explicit complete-view budget only when the model can consume
    the complete packet. Candidate scanner recall cannot justify acceptance.
    """
    return {"participant_id": participant_id,
            "foreign_rui_results": []}, [
        f"complete packet exceeds whole-packet audit budget ({_serialized_chars(packet)} > {max_prompt_chars}); no shortlist accepted"
    ]


def _serialized_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False))


def _remove_contexts(
    datapoint: dict[str, Any],
    context_ids: set[str],
) -> None:
    contexts = [
        row
        for row in datapoint.get("private_context_units") or []
        if isinstance(row, dict)
    ]
    removed_by_participant: dict[str, set[str]] = {}
    removed_words: dict[str, int] = {}
    for row in contexts:
        context_id = str(row.get("context_unit_id") or "")
        if context_id not in context_ids:
            continue
        for participant_id in row.get("assigned_to") or []:
            key = str(participant_id)
            removed_by_participant.setdefault(key, set()).add(context_id)
            removed_words[key] = removed_words.get(key, 0) + int(
                row.get("word_count") or 0
            )
    datapoint["private_context_units"] = [
        row
        for row in contexts
        if str(row.get("context_unit_id") or "") not in context_ids
    ]
    for participant in datapoint.get("participants") or []:
        if not isinstance(participant, dict):
            continue
        participant_id = str(participant.get("participant_id") or "")
        removed = removed_by_participant.get(participant_id, set())
        participant["private_context_unit_ids"] = [
            str(value)
            for value in participant.get("private_context_unit_ids") or []
            if str(value) not in removed
        ]
        participant["private_word_count"] = max(
            0,
            int(participant.get("private_word_count") or 0)
            - removed_words.get(participant_id, 0),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["index", "generate", "all"],
        default="all",
    )
    parser.add_argument(
        "--qa-path",
        required=True,
    )
    parser.add_argument(
        "--reports-zip",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "artifacts/generated/finance/finlongdocqa"),
    )
    parser.add_argument("--block-index-jsonl", default=None)
    parser.add_argument("--max-reports", type=int, default=0)
    parser.add_argument("--sample-reports", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260730)
    parser.add_argument(
        "--allocation-mode",
        choices=("auto", *PAPER_ALLOCATION_MODES),
        default="one_each",
        help="PIU-to-participant allocation structure.",
    )
    parser.add_argument(
        "--report-ids",
        default="",
        help=(
            "Comma-separated report IDs to process in the supplied order. "
            "When set, this takes precedence over sampling and max-reports."
        ),
    )
    parser.add_argument("--generation-attempts", type=int, default=3)
    parser.add_argument(
        "--packet-repair-rounds",
        type=int,
        default=3,
        help="Maximum packet-leak removal rounds before rejecting a task.",
    )
    parser.add_argument(
        "--audit-workers",
        type=int,
        default=8,
        help=(
            "Concurrent solver/judge conditions and participant packet audits "
            "conditions per report."
        ),
    )
    parser.add_argument("--participant-context-token-budget", type=int, default=PAPER_CONTEXT_BUDGET,
                        help="Initial local source cap, default 40000; configurable from 1 to 40000.")
    parser.add_argument(
        "--packet-audit-max-chars",
        type=int,
        default=500_000,
        help=(
            "Maximum serialized complete packet size. Larger packets are "
            "rejected; shortlist-only audits cannot certify a complete view."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["openai_compatible", "bedrock_anthropic"],
        default="openai_compatible",
    )
    parser.add_argument(
        "--base-url",
        default="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
    )
    parser.add_argument(
        "--endpoint",
        choices=["/responses", "/chat/completions"],
        default="/responses",
    )
    parser.add_argument("--api-key-env", default="AWS_BEARER_TOKEN_BEDROCK")
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--model", default="openai.gpt-5.5")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--request-retry-attempts", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.participant_context_token_budget <= PAPER_CONTEXT_BUDGET:
        parser.error("--participant-context-token-budget must be between 1 and 40000")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output_root).expanduser().resolve()
    reports = {}
    if args.stage in {"index", "all"}:
        reports["index"] = build_block_index(args)
    if args.stage in {"generate", "all"}:
        if args.block_index_jsonl:
            index_path = Path(args.block_index_jsonl).expanduser().resolve()
        else:
            index_path = output / "report_block_indexes.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"block index does not exist: {index_path}; run --stage index first"
            )
        args.block_index_jsonl = str(index_path)
        reports["generation"] = generate_dataset(args)
    print(json.dumps(reports, indent=2, ensure_ascii=False), flush=True)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _requested_report_ids(args: argparse.Namespace) -> list[str]:
    return list(
        dict.fromkeys(
            value.strip()
            for value in str(getattr(args, "report_ids", "") or "").split(",")
            if value.strip()
        )
    )


if __name__ == "__main__":
    main()
