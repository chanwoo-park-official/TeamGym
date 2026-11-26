"""Iteratively remove participant context chunks that leak the oracle answer."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import random
import re
import threading
from typing import Any

from teamgym.core.io import write_json, write_jsonl
from teamgym.core.schema import validate_common_datapoint
from teamgym.core.dataset import iter_common_datapoints
from teamgym.llm.openai_client import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)


SYSTEM_PROMPT = """You are a strict leakage auditor for a distributed legal QA benchmark.

You receive the public question, all public decision options, and exactly one participant's private context.
Some private context units may contain reference information units (RIUs). RIUs are required evidence for the
collaborative task. They are allowed to be useful, but a single participant should not be able to determine the
final answer from their own private context alone.
Do not use outside knowledge. Do not infer from decision label order. Treat labels such as decision_a and decision_b as arbitrary.
If the private context is insufficient, say it is not answerable.
If it is answerable, choose the option and cite the exact private context unit ids and short quotes that make it answerable.
Do not recommend deleting RIU-bearing units. If RIU-bearing evidence is decisive, mark it as structural leakage and
recommend excluding/regenerating the datapoint. Only recommend deleting non-RIU context units when those non-RIU
units make the answer answerable.
Return JSON only and follow response_format exactly.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Generated legal dataset root or JSONL file.")
    parser.add_argument("--output-root", required=True, help="Directory for pruned datapoints and funnel reports.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum datapoints to process.")
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle datapoints before taking --limit.")
    parser.add_argument("--dataset", action="append", help="Optional dataset filter; repeat for multiple datasets.")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument(
        "--exclude-unconverged",
        action="store_true",
        help="Exclude datapoints that still leak after --max-rounds deletion rounds plus final verification.",
    )
    parser.add_argument(
        "--exclude-empty-participants",
        action="store_true",
        help="Exclude datapoints where deletion leaves any participant with no private context.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Datapoint-level parallel workers.")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--max-unit-chars", type=int, default=2500)
    parser.add_argument("--max-participant-context-chars", type=int, default=20000)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--endpoint", default=DEFAULT_LLM_ENDPOINT, choices=["/responses", "/chat/completions"])
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--credentials-file")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--trace-prompts", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run audits but do not delete cited chunks.")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trace_dir = output_root / "audit_prompts" if args.trace_prompts else None

    datapoints = list(iter_common_datapoints(args.input_root))
    if args.dataset:
        allowed = {item.lower() for item in args.dataset}
        datapoints = [row for row in datapoints if str(row.get("dataset") or "").lower() in allowed]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(datapoints)
    if args.limit is not None:
        datapoints = datapoints[: max(0, args.limit)]
    if not datapoints:
        raise ValueError("No common datapoints selected.")

    client_config = OpenAIJsonClientConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        credentials_file=args.credentials_file,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_sec=args.timeout_sec,
        request_log_path=str(output_root / "llm_requests.jsonl"),
        log_prompts=bool(args.trace_prompts),
        log_responses=bool(args.trace_prompts),
    )
    thread_state = threading.local()

    def client_for_worker() -> OpenAIJsonClient:
        client = getattr(thread_state, "client", None)
        if client is None:
            client = OpenAIJsonClient(client_config)
            thread_state.client = client
        return client

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    datapoint_reports: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    completed = 0

    def run_one(index: int, datapoint: dict[str, Any]) -> tuple[int, dict[str, Any] | None, dict[str, Any]]:
        processed, report = process_datapoint(
            datapoint,
            client=client_for_worker(),
            trace_dir=trace_dir,
            max_rounds=args.max_rounds,
            confidence_threshold=args.confidence_threshold,
            max_unit_chars=args.max_unit_chars,
            max_participant_context_chars=args.max_participant_context_chars,
            dry_run=args.dry_run,
            exclude_unconverged=args.exclude_unconverged,
            exclude_empty_participants=args.exclude_empty_participants,
        )
        report["selection_index"] = index
        return index, processed, report

    workers = max(1, int(args.workers or 1))
    results: dict[int, tuple[dict[str, Any] | None, dict[str, Any]]] = {}
    if workers == 1:
        for index, datapoint in enumerate(datapoints, start=1):
            result_index, processed, report = run_one(index, datapoint)
            results[result_index] = (processed, report)
            completed += 1
            write_json(output_root / "progress.json", {"completed": completed, "total": len(datapoints), "workers": workers})
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(run_one, index, datapoint): index
                for index, datapoint in enumerate(datapoints, start=1)
            }
            for future in as_completed(future_to_index):
                result_index, processed, report = future.result()
                results[result_index] = (processed, report)
                completed += 1
                write_json(
                    output_root / "progress.json",
                    {"completed": completed, "total": len(datapoints), "workers": workers},
                )

    for index in sorted(results):
        processed, report = results[index]
        report_audits = report.pop("audit_rows")
        audit_rows.extend(report_audits)
        datapoint_reports.append(report)
        if processed is None:
            excluded_rows.append(report)
        else:
            rows.append(processed)

    write_jsonl(output_root / "leakage_deleted_common_datapoints.jsonl", rows)
    write_jsonl(output_root / "participant_leakage_audits.jsonl", audit_rows)
    write_jsonl(output_root / "excluded_datapoints.jsonl", excluded_rows)
    write_jsonl(output_root / "excluded_unconverged_datapoints.jsonl", excluded_rows)
    write_jsonl(output_root / "datapoint_funnel_reports.jsonl", datapoint_reports)
    summary = summarize(rows, audit_rows, args, selected_count=len(datapoints), excluded_rows=excluded_rows)
    write_json(output_root / "leakage_deletion_report.json", summary)
    write_markdown_report(output_root / "leakage_deletion_report.md", summary)
    write_html_report(output_root / "leakage_deletion_report.html", summary)
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))


def process_datapoint(
    datapoint: dict[str, Any],
    *,
    client: OpenAIJsonClient,
    trace_dir: Path | None,
    max_rounds: int,
    confidence_threshold: float,
    max_unit_chars: int,
    max_participant_context_chars: int,
    dry_run: bool,
    exclude_unconverged: bool,
    exclude_empty_participants: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    out = deepcopy(datapoint)
    out.setdefault("generation_metadata", {})
    metadata = out["generation_metadata"]
    if not isinstance(metadata, dict):
        metadata = {}
        out["generation_metadata"] = metadata
    funnel_meta = {
        "schema": "participant_leakage_deletion_funnel_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_rounds": max_rounds,
        "confidence_threshold": confidence_threshold,
        "dry_run": dry_run,
        "exclude_unconverged": exclude_unconverged,
        "exclude_empty_participants": exclude_empty_participants,
        "deletions": [],
        "structural_riu_leaks": [],
        "converged": False,
        "excluded": False,
        "exclusion_reasons": [],
    }
    metadata["participant_leakage_deletion_funnel"] = funnel_meta

    report: dict[str, Any] = {
        "instance_id": out.get("instance_id"),
        "dataset": out.get("dataset"),
        "audit_rows": [],
    }

    converged = False
    pending_after_cap: list[dict[str, Any]] = []
    structural_riu_leaks: list[dict[str, Any]] = []
    for round_index in range(1, max(0, max_rounds) + 1):
        round_deletions, round_structural_leaks = audit_datapoint_round(
            out,
            client=client,
            trace_dir=trace_dir,
            round_index=round_index,
            confidence_threshold=confidence_threshold,
            max_unit_chars=max_unit_chars,
            max_participant_context_chars=max_participant_context_chars,
            audit_rows=report["audit_rows"],
        )
        if round_structural_leaks:
            structural_riu_leaks.extend(round_structural_leaks)
            break
        if not round_deletions:
            converged = True
            break
        pending_after_cap = round_deletions
        if not dry_run:
            for deletion in round_deletions:
                remove_participant_context_units(out, deletion["participant_id"], deletion["context_unit_ids"])
            out["format_validation"] = validate_common_datapoint(out)
        funnel_meta["deletions"].extend(round_deletions)
        if dry_run:
            break

    if not structural_riu_leaks and not converged and not dry_run and pending_after_cap:
        verification_deletions, verification_structural_leaks = audit_datapoint_round(
            out,
            client=client,
            trace_dir=trace_dir,
            round_index=max(0, max_rounds) + 1,
            confidence_threshold=confidence_threshold,
            max_unit_chars=max_unit_chars,
            max_participant_context_chars=max_participant_context_chars,
            audit_rows=report["audit_rows"],
            verification_only=True,
        )
        if verification_structural_leaks:
            structural_riu_leaks.extend(verification_structural_leaks)
        elif not verification_deletions:
            converged = True
        else:
            pending_after_cap = verification_deletions

    funnel_meta["converged"] = converged
    report["converged"] = converged
    report["pending_leaks_after_cap"] = pending_after_cap if not converged and not structural_riu_leaks else []
    empty_participants = empty_participant_ids(out)
    exclusion_reasons: list[str] = []
    if structural_riu_leaks and not dry_run:
        exclusion_reasons.append("structural_riu_leak")
    if exclude_unconverged and not converged and not dry_run and not structural_riu_leaks:
        exclusion_reasons.append("unconverged")
    if exclude_empty_participants and empty_participants and not dry_run:
        exclusion_reasons.append("empty_participant")

    report["structural_riu_leaks"] = structural_riu_leaks
    report["structural_riu_leak_count"] = len(structural_riu_leaks)
    report["empty_participants_after_deletion"] = empty_participants
    report["excluded"] = bool(exclusion_reasons)
    report["exclusion_reasons"] = exclusion_reasons
    funnel_meta["structural_riu_leaks"] = structural_riu_leaks
    funnel_meta["excluded"] = report["excluded"]
    funnel_meta["exclusion_reasons"] = exclusion_reasons
    funnel_meta["empty_participants_after_deletion"] = empty_participants

    report["deletion_count"] = sum(len(row["context_unit_ids"]) for row in funnel_meta["deletions"])
    report["deletion_events"] = funnel_meta["deletions"]
    if report["excluded"]:
        return None, report
    return out, report


def audit_datapoint_round(
    datapoint: dict[str, Any],
    *,
    client: OpenAIJsonClient,
    trace_dir: Path | None,
    round_index: int,
    confidence_threshold: float,
    max_unit_chars: int,
    max_participant_context_chars: int,
    audit_rows: list[dict[str, Any]],
    verification_only: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    round_deletions: list[dict[str, Any]] = []
    round_structural_leaks: list[dict[str, Any]] = []
    for participant in participant_records(datapoint):
        participant_id = str(participant.get("participant_id") or "")
        if not participant_id:
            continue
        context_units = participant_context_units(datapoint, participant_id)
        if not context_units:
            continue
        audit = run_participant_audit(
            datapoint,
            participant_id=participant_id,
            context_units=context_units,
            client=client,
            trace_dir=trace_dir,
            round_index=round_index,
            max_unit_chars=max_unit_chars,
            max_participant_context_chars=max_participant_context_chars,
        )
        scored = score_audit(
            datapoint,
            participant_id=participant_id,
            audit=audit,
            context_units=context_units,
            confidence_threshold=confidence_threshold,
        )
        scored["verification_only"] = bool(verification_only)
        audit_rows.append(scored)
        if scored.get("structural_riu_leak"):
            round_structural_leaks.append(
                {
                    "round": round_index,
                    "participant_id": participant_id,
                    "context_unit_ids": scored.get("decisive_riu_context_unit_ids") or [],
                    "decisive_rui_ids": scored.get("decisive_rui_ids") or [],
                    "selected_answer": scored["selected_answer"],
                    "confidence": scored["confidence"],
                    "reason": scored.get("reason"),
                    "verification_only": bool(verification_only),
                    "recommended_action": "exclude_datapoint",
                }
            )
            continue
        if not scored["delete_context_unit_ids"]:
            continue
        round_deletions.append(
            {
                "round": round_index,
                "participant_id": participant_id,
                "context_unit_ids": scored["delete_context_unit_ids"],
                "selected_answer": scored["selected_answer"],
                "confidence": scored["confidence"],
                "reason": scored.get("reason"),
                "verification_only": bool(verification_only),
                "recommended_action": "delete_non_riu_context",
            }
        )
    return round_deletions, round_structural_leaks


def run_participant_audit(
    datapoint: dict[str, Any],
    *,
    participant_id: str,
    context_units: list[dict[str, Any]],
    client: OpenAIJsonClient,
    trace_dir: Path | None,
    round_index: int,
    max_unit_chars: int,
    max_participant_context_chars: int,
) -> dict[str, Any]:
    payload = audit_payload(
        datapoint,
        participant_id=participant_id,
        context_units=context_units,
        max_unit_chars=max_unit_chars,
        max_participant_context_chars=max_participant_context_chars,
    )
    prompt_id = safe_slug(f"{datapoint.get('dataset')}_{datapoint.get('instance_id')}_{participant_id}_round{round_index}")
    response = client.generate_json(
        prompt_id=prompt_id,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, indent=2, ensure_ascii=False),
        trace_dir=trace_dir,
    )
    return {
        "prompt_id": prompt_id,
        "round": round_index,
        "participant_id": participant_id,
        "response": response,
        "request_meta": dict(client.last_request_meta),
    }


def audit_payload(
    datapoint: dict[str, Any],
    *,
    participant_id: str,
    context_units: list[dict[str, Any]],
    max_unit_chars: int,
    max_participant_context_chars: int,
) -> dict[str, Any]:
    rui_by_id = reference_information_by_id(datapoint)
    units: list[dict[str, Any]] = []
    remaining = max(0, max_participant_context_chars)
    for unit in context_units:
        text = str(unit.get("text") or "")
        clipped = clip(text, min(max_unit_chars, remaining)) if remaining else ""
        contains_rui_ids = context_unit_rui_ids(unit)
        units.append(
            {
                "context_unit_id": str(unit.get("context_unit_id") or ""),
                "text": clipped,
                "contains_rui_ids": contains_rui_ids,
                "contains_rui_units": [
                    {
                        "rui_id": rui_id,
                        "information_feature": str(rui_by_id.get(rui_id, {}).get("information_feature") or ""),
                        "text": clip(str(rui_by_id.get(rui_id, {}).get("text") or ""), 800),
                    }
                    for rui_id in contains_rui_ids
                ],
                "is_riu_bearing": bool(contains_rui_ids),
            }
        )
        remaining -= len(clipped)
        if remaining <= 0:
            break
    return {
        "audit_task": "single_participant_answerability_leakage_check",
        "instructions": [
            "Use only the provided public task and this participant's private context.",
            "Do not use outside knowledge.",
            "Do not infer from decision label order.",
            "If the exact answer is not supported by this private context, set answerable_from_this_context=false.",
            "If answerable, cite exact context_unit_ids and short quotes from the private context.",
            "RIUs are required evidence units for the collaborative task. Context units with contains_rui_ids are RIU-bearing.",
            "Do not recommend deleting RIU-bearing context units.",
            "If RIU-bearing context is decisive for the participant answering alone, set riu_units_make_answerable=true and recommended_action='exclude_datapoint'.",
            "Only set recommended_action='delete_non_riu_context' for decisive context units with no contains_rui_ids.",
        ],
        "public_task": {
            "dataset": datapoint.get("dataset"),
            "instance_id": datapoint.get("instance_id"),
            "task_description": datapoint.get("task_description"),
            "decision_set": datapoint.get("decision_set"),
            "decision_option_definitions": datapoint.get("decision_option_definitions"),
            "initial_common_information": datapoint.get("initial_common_information"),
        },
        "participant": {"participant_id": participant_id},
        "private_context_units": units,
        "response_format": {
            "answerable_from_this_context": "boolean",
            "selected_answer": "decision id string from decision_set, or null",
            "confidence": "number from 0.0 to 1.0",
            "decisive_context_unit_ids": ["context_unit_id strings"],
            "decisive_rui_ids": ["rui_id strings from RIU-bearing decisive units, if any"],
            "non_riu_context_unit_ids_to_delete": ["decisive context_unit_id strings with no contains_rui_ids"],
            "riu_units_make_answerable": "boolean",
            "recommended_action": "keep | delete_non_riu_context | exclude_datapoint",
            "decisive_quotes": ["short exact quotes from private context"],
            "reason": "brief explanation",
        },
    }


def score_audit(
    datapoint: dict[str, Any],
    *,
    participant_id: str,
    audit: dict[str, Any],
    context_units: list[dict[str, Any]],
    confidence_threshold: float,
) -> dict[str, Any]:
    response = audit.get("response") if isinstance(audit.get("response"), dict) else {}
    answerable = bool(response.get("answerable_from_this_context"))
    selected = str(response.get("selected_answer") or "")
    confidence = as_float(response.get("confidence"))
    oracle = datapoint.get("oracle") if isinstance(datapoint.get("oracle"), dict) else {}
    oracle_answer = str(oracle.get("answer") or "")
    available_ids = {str(unit.get("context_unit_id") or "") for unit in context_units}
    unit_by_id = {str(unit.get("context_unit_id") or ""): unit for unit in context_units}
    cited_ids = [
        str(item)
        for item in response.get("decisive_context_unit_ids", [])
        if str(item) in available_ids
    ]
    rui_context_ids = [unit_id for unit_id in cited_ids if context_unit_rui_ids(unit_by_id.get(unit_id, {}))]
    non_riu_context_ids = [unit_id for unit_id in cited_ids if unit_id not in set(rui_context_ids)]
    decisive_rui_ids = sorted(
        {
            rui_id
            for unit_id in rui_context_ids
            for rui_id in context_unit_rui_ids(unit_by_id.get(unit_id, {}))
        }
    )
    quotes = [str(item) for item in response.get("decisive_quotes", []) if str(item).strip()]
    quote_hits = quote_hit_count(quotes, context_units)
    correct = bool(selected and selected == oracle_answer)
    leak = bool(answerable and correct and confidence >= confidence_threshold and cited_ids)
    structural_riu_leak = bool(leak and rui_context_ids)
    delete_context_unit_ids = non_riu_context_ids if leak and not structural_riu_leak else []
    if structural_riu_leak:
        leak_type = "structural_riu_leak"
        recommended_action = "exclude_datapoint"
    elif delete_context_unit_ids:
        leak_type = "non_riu_context_leak"
        recommended_action = "delete_non_riu_context"
    else:
        leak_type = "none"
        recommended_action = "keep"
    return {
        "schema": "participant_leakage_audit_result_v1",
        "dataset": datapoint.get("dataset"),
        "instance_id": datapoint.get("instance_id"),
        "participant_id": participant_id,
        "round": audit.get("round"),
        "prompt_id": audit.get("prompt_id"),
        "answerable_from_this_context": answerable,
        "selected_answer": selected or None,
        "oracle_answer": oracle_answer,
        "correct": correct,
        "confidence": confidence,
        "confidence_threshold": confidence_threshold,
        "decisive_context_unit_ids": cited_ids,
        "decisive_riu_context_unit_ids": rui_context_ids,
        "decisive_non_riu_context_unit_ids": non_riu_context_ids,
        "decisive_rui_ids": decisive_rui_ids,
        "decisive_quote_count": len(quotes),
        "decisive_quote_hit_count": quote_hits,
        "leak_found": leak,
        "structural_riu_leak": structural_riu_leak,
        "leak_type": leak_type,
        "recommended_action": recommended_action,
        "delete_context_unit_ids": delete_context_unit_ids,
        "reason": response.get("reason"),
        "request_meta": audit.get("request_meta") or {},
        "raw_response": response,
    }


def remove_participant_context_units(datapoint: dict[str, Any], participant_id: str, context_unit_ids: list[str]) -> None:
    remove_ids = {str(item) for item in context_unit_ids}
    for participant in participant_records(datapoint):
        if str(participant.get("participant_id") or "") != participant_id:
            continue
        refs = [str(item) for item in participant.get("private_context_unit_ids", [])]
        participant["private_context_unit_ids"] = [item for item in refs if item not in remove_ids]
    kept_units: list[dict[str, Any]] = []
    for unit in datapoint.get("private_context_units", []) or []:
        if str(unit.get("context_unit_id") or "") not in remove_ids:
            kept_units.append(unit)
            continue
        assigned = [str(item) for item in unit.get("assigned_to", []) if str(item) != participant_id]
        if assigned:
            edited = dict(unit)
            edited["assigned_to"] = assigned
            kept_units.append(edited)
    datapoint["private_context_units"] = kept_units


def participant_records(datapoint: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in datapoint.get("participants", []) or [] if isinstance(row, dict)]


def participant_context_units(datapoint: dict[str, Any], participant_id: str) -> list[dict[str, Any]]:
    units_by_id = {
        str(unit.get("context_unit_id")): unit
        for unit in datapoint.get("private_context_units", []) or []
        if isinstance(unit, dict) and unit.get("context_unit_id")
    }
    participant = next(
        (row for row in participant_records(datapoint) if str(row.get("participant_id") or "") == participant_id),
        None,
    )
    ids: list[str]
    if participant and participant.get("private_context_unit_ids"):
        ids = [str(item) for item in participant.get("private_context_unit_ids", [])]
    else:
        ids = [
            str(unit.get("context_unit_id"))
            for unit in datapoint.get("private_context_units", []) or []
            if participant_id in {str(item) for item in unit.get("assigned_to", [])}
        ]
    return [units_by_id[item] for item in ids if item in units_by_id]


def reference_information_by_id(datapoint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("rui_id")): row
        for row in datapoint.get("reference_information_units", []) or []
        if isinstance(row, dict) and row.get("rui_id")
    }


def context_unit_rui_ids(unit: dict[str, Any]) -> list[str]:
    rui_ids = unit.get("hidden_contains_rui_ids") if isinstance(unit, dict) else []
    if not isinstance(rui_ids, list):
        return []
    return [str(item) for item in rui_ids if str(item)]


def empty_participant_ids(datapoint: dict[str, Any]) -> list[str]:
    empty: list[str] = []
    for participant in participant_records(datapoint):
        participant_id = str(participant.get("participant_id") or "")
        if participant_id and not participant_context_units(datapoint, participant_id):
            empty.append(participant_id)
    return empty


def summarize(
    rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    selected_count: int,
    excluded_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    leak_rows = [row for row in audit_rows if row.get("leak_found")]
    structural_leak_rows = [row for row in audit_rows if row.get("structural_riu_leak")]
    deletion_count = sum(len(row.get("delete_context_unit_ids") or []) for row in leak_rows)
    structural_riu_context_refs = sum(len(row.get("decisive_riu_context_unit_ids") or []) for row in structural_leak_rows)
    by_dataset: dict[str, dict[str, Any]] = {}
    for datapoint in rows:
        dataset = str(datapoint.get("dataset") or "unknown")
        entry = by_dataset.setdefault(dataset, {"datapoints": 0, "context_units_after": 0, "participants": 0})
        entry["datapoints"] += 1
        entry["context_units_after"] += len(datapoint.get("private_context_units") or [])
        entry["participants"] += len(datapoint.get("participants") or [])
    for audit in audit_rows:
        dataset = str(audit.get("dataset") or "unknown")
        entry = by_dataset.setdefault(dataset, {"datapoints": 0, "context_units_after": 0, "participants": 0})
        entry["audit_calls"] = entry.get("audit_calls", 0) + 1
        entry["leaks_found"] = entry.get("leaks_found", 0) + int(bool(audit.get("leak_found")))
        entry["deleted_context_refs"] = entry.get("deleted_context_refs", 0) + len(audit.get("delete_context_unit_ids") or [])
        entry["structural_riu_leaks"] = entry.get("structural_riu_leaks", 0) + int(
            bool(audit.get("structural_riu_leak"))
        )
        entry["structural_riu_context_refs"] = entry.get("structural_riu_context_refs", 0) + len(
            audit.get("decisive_riu_context_unit_ids") or []
        )
    excluded_by_dataset: dict[str, int] = {}
    excluded_reason_counts: dict[str, int] = {}
    for row in excluded_rows:
        dataset = str(row.get("dataset") or "unknown")
        excluded_by_dataset[dataset] = excluded_by_dataset.get(dataset, 0) + 1
        entry = by_dataset.setdefault(dataset, {"datapoints": 0, "context_units_after": 0, "participants": 0})
        entry["excluded"] = entry.get("excluded", 0) + 1
        reasons = [str(item) for item in row.get("exclusion_reasons") or ["excluded"]]
        for reason in reasons:
            excluded_reason_counts[reason] = excluded_reason_counts.get(reason, 0) + 1
            entry[f"excluded_{reason}"] = entry.get(f"excluded_{reason}", 0) + 1

    return {
        "schema": "participant_leakage_deletion_funnel_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "input_root": args.input_root,
            "audit_public_task_source": (
                "common_datapoint.task_description plus common_datapoint.decision_option_definitions; "
                "this preserves any post-generation decision-label permutation. Pre-partition task tuples are "
                "useful provenance, but may contain pre-permutation labels."
            ),
            "limit": args.limit,
            "seed": args.seed,
            "shuffle": args.shuffle,
            "dataset": args.dataset,
            "max_rounds": args.max_rounds,
            "exclude_unconverged": args.exclude_unconverged,
            "exclude_empty_participants": args.exclude_empty_participants,
            "workers": args.workers,
            "confidence_threshold": args.confidence_threshold,
            "model": args.model,
            "dry_run": args.dry_run,
        },
        "aggregate": {
            "datapoints_selected": selected_count,
            "datapoints_written": len(rows),
            "datapoints_excluded": len(excluded_rows),
            "datapoints_excluded_unconverged": excluded_reason_counts.get("unconverged", 0),
            "datapoints_excluded_empty_participant": excluded_reason_counts.get("empty_participant", 0),
            "datapoints_excluded_structural_riu_leak": excluded_reason_counts.get("structural_riu_leak", 0),
            "audit_calls": len(audit_rows),
            "leak_audits_found": len(leak_rows),
            "structural_riu_leak_audits": len(structural_leak_rows),
            "deleted_context_refs": deletion_count,
            "structural_riu_context_refs": structural_riu_context_refs,
            "answerable_rate": mean_bool(audit_rows, "answerable_from_this_context"),
            "correct_rate": mean_bool(audit_rows, "correct"),
            "leak_rate": mean_bool(audit_rows, "leak_found"),
        },
        "by_dataset": by_dataset,
        "excluded_by_dataset": excluded_by_dataset,
        "excluded_reason_counts": excluded_reason_counts,
        "excluded_examples": excluded_rows[:20],
        "leak_examples": leak_rows[:20],
        "structural_riu_leak_examples": structural_leak_rows[:20],
    }


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Participant Leakage Deletion Funnel",
        "",
        f"- Datapoints selected: {agg['datapoints_selected']}",
        f"- Datapoints written: {agg['datapoints_written']}",
        f"- Datapoints excluded: {agg['datapoints_excluded']}",
        f"- Datapoints excluded unconverged: {agg['datapoints_excluded_unconverged']}",
        f"- Datapoints excluded empty participant: {agg['datapoints_excluded_empty_participant']}",
        f"- Datapoints excluded structural RIU leak: {agg['datapoints_excluded_structural_riu_leak']}",
        f"- Audit calls: {agg['audit_calls']}",
        f"- Leak audits found: {agg['leak_audits_found']}",
        f"- Structural RIU leak audits: {agg['structural_riu_leak_audits']}",
        f"- Deleted participant context refs: {agg['deleted_context_refs']}",
        f"- Structural RIU context refs: {agg['structural_riu_context_refs']}",
        f"- Answerable rate: {agg['answerable_rate']:.3f}",
        f"- Correct rate: {agg['correct_rate']:.3f}",
        f"- Leak rate: {agg['leak_rate']:.3f}",
        "",
        "## By Dataset",
        "",
        "| dataset | datapoints written | audit calls | leaks | structural RIU leaks | deleted refs | excluded |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, row in sorted(summary["by_dataset"].items()):
        lines.append(
            f"| {dataset} | {row.get('datapoints', 0)} | {row.get('audit_calls', 0)} | "
            f"{row.get('leaks_found', 0)} | {row.get('structural_riu_leaks', 0)} | "
            f"{row.get('deleted_context_refs', 0)} | "
            f"{row.get('excluded', 0)} |"
        )
    lines.extend(["", "## Structural RIU Leak Examples", ""])
    for row in summary.get("structural_riu_leak_examples", [])[:10]:
        lines.append(
            f"- `{row.get('dataset')}` `{row.get('instance_id')}` `{row.get('participant_id')}` "
            f"round {row.get('round')}: {row.get('selected_answer')} confidence={row.get('confidence')}; "
            f"RIUs {row.get('decisive_rui_ids')}; context {row.get('decisive_riu_context_unit_ids')}"
        )
    lines.extend(["", "## Leak Examples", ""])
    for row in summary.get("leak_examples", [])[:10]:
        lines.append(
            f"- `{row.get('dataset')}` `{row.get('instance_id')}` `{row.get('participant_id')}` "
            f"round {row.get('round')}: {row.get('selected_answer')} confidence={row.get('confidence')}; "
            f"delete {row.get('delete_context_unit_ids')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_report(path: Path, summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    cards = [
        ("Selected", agg["datapoints_selected"]),
        ("Datapoints", agg["datapoints_written"]),
        ("Excluded", agg["datapoints_excluded"]),
        ("Unconverged", agg["datapoints_excluded_unconverged"]),
        ("Empty participant", agg["datapoints_excluded_empty_participant"]),
        ("Structural RIU excluded", agg["datapoints_excluded_structural_riu_leak"]),
        ("Audit calls", agg["audit_calls"]),
        ("Leaks found", agg["leak_audits_found"]),
        ("Structural RIU leaks", agg["structural_riu_leak_audits"]),
        ("Deleted refs", agg["deleted_context_refs"]),
        ("Leak rate", f"{agg['leak_rate']:.1%}"),
    ]
    card_html = "\n".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div><div class="metric">{html.escape(str(value))}</div></div>'
        for label, value in cards
    )
    dataset_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(dataset)}</td>"
        f"<td>{row.get('datapoints', 0)}</td>"
        f"<td>{row.get('audit_calls', 0)}</td>"
        f"<td>{row.get('leaks_found', 0)}</td>"
        f"<td>{row.get('structural_riu_leaks', 0)}</td>"
        f"<td>{row.get('deleted_context_refs', 0)}</td>"
        f"<td>{row.get('excluded', 0)}</td>"
        "</tr>"
        for dataset, row in sorted(summary["by_dataset"].items())
    )
    structural_examples = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('dataset')))}</td>"
        f"<td>{html.escape(str(row.get('instance_id')))}</td>"
        f"<td>{html.escape(str(row.get('participant_id')))}</td>"
        f"<td>{html.escape(str(row.get('round')))}</td>"
        f"<td>{html.escape(str(row.get('selected_answer')))}</td>"
        f"<td>{html.escape(str(row.get('confidence')))}</td>"
        f"<td><code>{html.escape(', '.join(row.get('decisive_rui_ids') or []))}</code></td>"
        f"<td><code>{html.escape(', '.join(row.get('decisive_riu_context_unit_ids') or []))}</code></td>"
        f"<td>{html.escape(str(row.get('reason') or ''))}</td>"
        "</tr>"
        for row in summary.get("structural_riu_leak_examples", [])[:20]
    )
    examples = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('dataset')))}</td>"
        f"<td>{html.escape(str(row.get('instance_id')))}</td>"
        f"<td>{html.escape(str(row.get('participant_id')))}</td>"
        f"<td>{html.escape(str(row.get('round')))}</td>"
        f"<td>{html.escape(str(row.get('selected_answer')))}</td>"
        f"<td>{html.escape(str(row.get('confidence')))}</td>"
        f"<td><code>{html.escape(', '.join(row.get('delete_context_unit_ids') or []))}</code></td>"
        f"<td>{html.escape(str(row.get('reason') or ''))}</td>"
        "</tr>"
        for row in summary.get("leak_examples", [])[:20]
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Participant Leakage Deletion Funnel</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2937; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }}
.card {{ border: 1px solid #d9dee7; border-radius: 8px; padding: 12px 14px; background: #fbfcfe; }}
.metric {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
.label {{ color: #5f6368; font-size: 13px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 13px; }}
th, td {{ border: 1px solid #d9dee7; padding: 7px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f3f6fa; }}
code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Participant Leakage Deletion Funnel</h1>
	<p>Audits one participant at a time, deletes cited private context units when that participant can correctly answer alone with confidence above threshold, then repeats.</p>
	<div class="summary">{card_html}</div>
	<h2>By Dataset</h2>
	<table><thead><tr><th>Dataset</th><th>Datapoints Written</th><th>Audit calls</th><th>Leaks</th><th>Structural RIU leaks</th><th>Deleted refs</th><th>Excluded</th></tr></thead><tbody>{dataset_rows}</tbody></table>
	<h2>Structural RIU Leak Examples</h2>
	<table><thead><tr><th>Dataset</th><th>Instance</th><th>Participant</th><th>Round</th><th>Answer</th><th>Confidence</th><th>RIUs</th><th>Context Units</th><th>Reason</th></tr></thead><tbody>{structural_examples}</tbody></table>
	<h2>Leak Examples</h2>
<table><thead><tr><th>Dataset</th><th>Instance</th><th>Participant</th><th>Round</th><th>Answer</th><th>Confidence</th><th>Deleted Context Units</th><th>Reason</th></tr></thead><tbody>{examples}</tbody></table>
</body>
</html>
""",
        encoding="utf-8",
    )


def clip(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 24)] + "\n[TRUNCATED]"


def quote_hit_count(quotes: list[str], context_units: list[dict[str, Any]]) -> int:
    context = normalize_quote(" ".join(str(unit.get("text") or "") for unit in context_units))
    count = 0
    for quote in quotes:
        norm = normalize_quote(quote)
        if norm and norm in context:
            count += 1
    return count


def normalize_quote(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def as_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key)) / len(rows)


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:180] or "audit"


if __name__ == "__main__":
    main()
