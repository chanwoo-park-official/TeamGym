"""Verify whether one participant can eliminate all non-oracle options."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import random
import threading
from typing import Any

from teamgym.core.release import write_release_datapoints
from teamgym.core.io import write_json, write_jsonl
from teamgym.data.legal.leakage_deletion_funnel import (
    as_float,
    clip,
    normalize_quote,
    participant_context_units,
    participant_records,
    safe_slug,
)
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


SYSTEM_PROMPT = """You are a strict option-exclusion auditor for a distributed legal QA benchmark.

You receive the public question, all public decision options, and exactly one participant's private context.
Your job is not to solve the final task. Your job is to decide which options this participant can safely rule
out using only their private context.

Treat decision labels such as decision_a and decision_b as arbitrary. Do not use outside knowledge, label order,
or likely benchmark patterns. Keep an option possible when the private context is incomplete, ambiguous, merely
silent, or requires missing party-role/context information. Exclude an option only when private context gives
affirmative evidence that contradicts it or supports a mutually exclusive fact.

Return JSON only and follow response_format exactly.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Generated legal dataset root or JSONL file.")
    parser.add_argument("--output-root", required=True, help="Directory for verifier outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum datapoints to process after filtering.")
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle datapoints before taking --limit.")
    parser.add_argument("--dataset", action="append", help="Optional dataset filter; repeat for multiple datasets.")
    parser.add_argument(
        "--require-leakage-safe",
        action="store_true",
        help="Keep only datapoints with a converged leakage-deletion funnel and excluded=false.",
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
    parser.add_argument("--request-retry-attempts", type=int, default=4)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--trace-prompts", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trace_dir = output_root / "audit_prompts" if args.trace_prompts else None

    datapoints = list(iter_common_datapoints(args.input_root))
    if args.dataset:
        allowed = {item.lower() for item in args.dataset}
        datapoints = [row for row in datapoints if str(row.get("dataset") or "").lower() in allowed]
    if args.require_leakage_safe:
        datapoints = [row for row in datapoints if is_leakage_safe(row)]
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
        request_retry_attempts=args.request_retry_attempts,
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

    results: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    completed = 0

    def run_one(index: int, datapoint: dict[str, Any]) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
        audit_rows, report = verify_datapoint(
            datapoint,
            client=client_for_worker(),
            trace_dir=trace_dir,
            confidence_threshold=args.confidence_threshold,
            max_unit_chars=args.max_unit_chars,
            max_participant_context_chars=args.max_participant_context_chars,
            continue_on_error=args.continue_on_error,
        )
        report["selection_index"] = index
        return index, audit_rows, report

    workers = max(1, int(args.workers or 1))
    if workers == 1:
        for index, datapoint in enumerate(datapoints, start=1):
            try:
                result_index, audit_rows, report = run_one(index, datapoint)
            except Exception as exc:  # noqa: BLE001
                if not args.continue_on_error:
                    raise
                result_index = index
                audit_rows = []
                report = failed_datapoint_report(datapoint, index=index, exc=exc)
            results[result_index] = (audit_rows, report)
            completed += 1
            write_json(output_root / "progress.json", {"completed": completed, "total": len(datapoints), "workers": workers})
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(run_one, index, datapoint): index
                for index, datapoint in enumerate(datapoints, start=1)
            }
            for future in as_completed(future_to_index):
                try:
                    result_index, audit_rows, report = future.result()
                except Exception as exc:  # noqa: BLE001
                    if not args.continue_on_error:
                        raise
                    result_index = future_to_index[future]
                    audit_rows = []
                    report = failed_datapoint_report(datapoints[result_index - 1], index=result_index, exc=exc)
                results[result_index] = (audit_rows, report)
                completed += 1
                write_json(
                    output_root / "progress.json",
                    {"completed": completed, "total": len(datapoints), "workers": workers},
                )

    all_audits: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for index in sorted(results):
        audit_rows, report = results[index]
        all_audits.extend(audit_rows)
        reports.append(report)

    kept_rows, dropped_rows, misleading_rows = filtered_datapoints(datapoints, reports)
    write_release_datapoints(output_root / "solo_oracle_kept_common_datapoints.jsonl", kept_rows)
    write_jsonl(output_root / "solo_oracle_dropped_datapoints.jsonl", dropped_rows)
    write_jsonl(output_root / "misleading_single_option_common_datapoints.jsonl", misleading_rows)
    write_jsonl(output_root / "participant_option_exclusion_audits.jsonl", all_audits)
    write_jsonl(output_root / "datapoint_option_exclusion_reports.jsonl", reports)
    summary = summarize(reports, all_audits, args, selected_count=len(datapoints))
    write_json(output_root / "option_exclusion_report.json", summary)
    write_markdown_report(output_root / "option_exclusion_report.md", summary)
    write_html_report(output_root / "option_exclusion_report.html", summary)
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))


def verify_datapoint(
    datapoint: dict[str, Any],
    *,
    client: OpenAIJsonClient,
    trace_dir: Path | None,
    confidence_threshold: float,
    max_unit_chars: int,
    max_participant_context_chars: int,
    continue_on_error: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    for participant in participant_records(datapoint):
        participant_id = str(participant.get("participant_id") or "")
        if not participant_id:
            continue
        context_units = participant_context_units(datapoint, participant_id)
        if not context_units:
            continue
        try:
            audit = run_participant_exclusion_audit(
                datapoint,
                participant_id=participant_id,
                context_units=context_units,
                client=client,
                trace_dir=trace_dir,
                max_unit_chars=max_unit_chars,
                max_participant_context_chars=max_participant_context_chars,
            )
            audit_rows.append(
                score_exclusion_audit(
                    datapoint,
                    participant_id=participant_id,
                    audit=audit,
                    context_units=context_units,
                    confidence_threshold=confidence_threshold,
                )
            )
        except Exception as exc:  # noqa: BLE001
            if not continue_on_error:
                raise
            audit_rows.append(
                failed_exclusion_audit_result(
                    datapoint,
                    participant_id=participant_id,
                    exc=exc,
                    confidence_threshold=confidence_threshold,
                )
            )
    oracle_answer = oracle_for(datapoint)
    solo_oracle_rows = [row for row in audit_rows if row.get("reduces_to_oracle")]
    single_option_rows = [row for row in audit_rows if row.get("remaining_option_count") == 1]
    wrong_single_option_rows = [row for row in single_option_rows if row.get("single_option_wrong")]
    oracle_single_option_rows = [row for row in single_option_rows if row.get("single_option_is_oracle")]
    failed_rows = [row for row in audit_rows if row.get("audit_failed")]
    keep_under_filter = not bool(solo_oracle_rows)
    if solo_oracle_rows:
        filter_decision = "drop_solo_oracle_reducible"
    elif wrong_single_option_rows:
        filter_decision = "keep_misleading_single_option"
    else:
        filter_decision = "keep_no_single_option_reduction"
    report = {
        "schema": "participant_option_exclusion_datapoint_report_v1",
        "dataset": datapoint.get("dataset"),
        "instance_id": datapoint.get("instance_id"),
        "oracle_answer": oracle_answer,
        "participant_count": len(participant_records(datapoint)),
        "audited_participant_count": len(audit_rows),
        "failed_participant_audit_count": len(failed_rows),
        "keep_under_solo_oracle_filter": keep_under_filter,
        "filter_decision": filter_decision,
        "has_single_participant_oracle_reduction": bool(solo_oracle_rows),
        "has_any_single_option_reduction": bool(single_option_rows),
        "has_wrong_single_option_reduction": bool(wrong_single_option_rows),
        "has_only_wrong_single_option_reductions": bool(single_option_rows) and not bool(oracle_single_option_rows),
        "solo_oracle_participants": [row.get("participant_id") for row in solo_oracle_rows],
        "single_option_participants": [row.get("participant_id") for row in single_option_rows],
        "wrong_single_option_participants": [row.get("participant_id") for row in wrong_single_option_rows],
        "oracle_single_option_participants": [row.get("participant_id") for row in oracle_single_option_rows],
        "min_remaining_option_count": min((int(row.get("remaining_option_count") or 0) for row in audit_rows), default=0),
        "max_excluded_option_count": max((int(row.get("excluded_option_count") or 0) for row in audit_rows), default=0),
    }
    return audit_rows, report


def failed_exclusion_audit_result(
    datapoint: dict[str, Any],
    *,
    participant_id: str,
    exc: Exception,
    confidence_threshold: float,
) -> dict[str, Any]:
    decision_set = [str(item) for item in datapoint.get("decision_set", []) if str(item)]
    return {
        "schema": "participant_option_exclusion_audit_result_v1",
        "dataset": datapoint.get("dataset"),
        "instance_id": datapoint.get("instance_id"),
        "participant_id": participant_id,
        "prompt_id": "",
        "oracle_answer": oracle_for(datapoint),
        "decision_set": decision_set,
        "excluded_options": [],
        "remaining_options_from_exclusions": decision_set,
        "reported_remaining_options": decision_set,
        "excluded_option_count": 0,
        "remaining_option_count": len(decision_set),
        "reduces_to_oracle": False,
        "single_option_remaining": len(decision_set) == 1,
        "single_remaining_option": decision_set[0] if len(decision_set) == 1 else "",
        "single_option_is_oracle": False,
        "single_option_wrong": False,
        "oracle_excluded": False,
        "confidence": 0.0,
        "confidence_threshold": confidence_threshold,
        "invalid_excluded_options": [],
        "invalid_remaining_options": [],
        "reported_remaining_consistent": True,
        "quote_count": 0,
        "quote_hit_count": 0,
        "reason": "Audit failed after retries; all options were kept possible for filtering.",
        "option_assessments": [],
        "request_meta": {},
        "raw_response": {},
        "audit_failed": True,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def failed_datapoint_report(datapoint: dict[str, Any], *, index: int, exc: Exception) -> dict[str, Any]:
    return {
        "schema": "participant_option_exclusion_datapoint_report_v1",
        "dataset": datapoint.get("dataset"),
        "instance_id": datapoint.get("instance_id"),
        "oracle_answer": oracle_for(datapoint),
        "participant_count": len(participant_records(datapoint)),
        "audited_participant_count": 0,
        "failed_participant_audit_count": len(participant_records(datapoint)),
        "keep_under_solo_oracle_filter": True,
        "filter_decision": "keep_audit_failed",
        "has_single_participant_oracle_reduction": False,
        "has_any_single_option_reduction": False,
        "has_wrong_single_option_reduction": False,
        "has_only_wrong_single_option_reductions": False,
        "solo_oracle_participants": [],
        "single_option_participants": [],
        "wrong_single_option_participants": [],
        "oracle_single_option_participants": [],
        "min_remaining_option_count": len(datapoint.get("decision_set", []) or []),
        "max_excluded_option_count": 0,
        "selection_index": index,
        "audit_failed": True,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def run_participant_exclusion_audit(
    datapoint: dict[str, Any],
    *,
    participant_id: str,
    context_units: list[dict[str, Any]],
    client: OpenAIJsonClient,
    trace_dir: Path | None,
    max_unit_chars: int,
    max_participant_context_chars: int,
) -> dict[str, Any]:
    payload = exclusion_payload(
        datapoint,
        participant_id=participant_id,
        context_units=context_units,
        max_unit_chars=max_unit_chars,
        max_participant_context_chars=max_participant_context_chars,
    )
    prompt_id = safe_slug(f"{datapoint.get('dataset')}_{datapoint.get('instance_id')}_{participant_id}_option_exclusion")
    response = client.generate_json(
        prompt_id=prompt_id,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, indent=2, ensure_ascii=False),
        trace_dir=trace_dir,
    )
    return {
        "prompt_id": prompt_id,
        "participant_id": participant_id,
        "response": response,
        "request_meta": dict(client.last_request_meta),
    }


def exclusion_payload(
    datapoint: dict[str, Any],
    *,
    participant_id: str,
    context_units: list[dict[str, Any]],
    max_unit_chars: int,
    max_participant_context_chars: int,
) -> dict[str, Any]:
    units: list[dict[str, Any]] = []
    remaining = max(0, max_participant_context_chars)
    for unit in context_units:
        text = str(unit.get("text") or "")
        clipped = clip(text, min(max_unit_chars, remaining)) if remaining else ""
        units.append(
            {
                "context_unit_id": str(unit.get("context_unit_id") or ""),
                "text": clipped,
            }
        )
        remaining -= len(clipped)
        if remaining <= 0:
            break
    return {
        "audit_task": "single_participant_option_exclusion_check",
        "instructions": [
            "Use only the public task and this participant's private context.",
            "Do not choose a final answer.",
            "For every decision option, decide whether this participant can safely rule it out.",
            "Exclude an option only when private context affirmatively contradicts it or supports a mutually exclusive fact.",
            "Keep an option possible if the private context is silent, incomplete, ambiguous, or requires missing context.",
            "Cite private context_unit_ids and short exact quotes for excluded options whenever possible.",
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
            "excluded_options": ["decision ids safely ruled out"],
            "remaining_options": ["decision ids still possible after exclusions"],
            "confidence": "number from 0.0 to 1.0 that the excluded_options are safely excluded",
            "option_assessments": [
                {
                    "option": "decision id",
                    "status": "excluded | possible",
                    "context_unit_ids": ["context_unit_ids supporting this assessment"],
                    "quotes": ["short exact quotes from private context"],
                    "reason": "brief reason",
                }
            ],
            "reason": "brief overall explanation",
        },
    }


def score_exclusion_audit(
    datapoint: dict[str, Any],
    *,
    participant_id: str,
    audit: dict[str, Any],
    context_units: list[dict[str, Any]],
    confidence_threshold: float,
) -> dict[str, Any]:
    response = audit.get("response") if isinstance(audit.get("response"), dict) else {}
    decision_set = [str(item) for item in datapoint.get("decision_set", []) if str(item)]
    decision_ids = set(decision_set)
    oracle_answer = oracle_for(datapoint)
    raw_excluded = response.get("excluded_options") if isinstance(response.get("excluded_options"), list) else []
    raw_remaining = response.get("remaining_options") if isinstance(response.get("remaining_options"), list) else []
    excluded = ordered_valid_options(raw_excluded, decision_set)
    remaining_from_exclusions = [item for item in decision_set if item not in set(excluded)]
    reported_remaining = ordered_valid_options(raw_remaining, decision_set)
    confidence = as_float(response.get("confidence"))
    assessments = response.get("option_assessments") if isinstance(response.get("option_assessments"), list) else []
    quotes = assessment_quotes(assessments)
    quote_hits = quote_hit_count(quotes, context_units)
    invalid_excluded = sorted({str(item) for item in raw_excluded if str(item) not in decision_ids})
    invalid_remaining = sorted({str(item) for item in raw_remaining if str(item) not in decision_ids})
    reduces_to_oracle = bool(
        oracle_answer
        and confidence >= confidence_threshold
        and remaining_from_exclusions == [oracle_answer]
    )
    remaining_count = len(remaining_from_exclusions)
    single_remaining_option = remaining_from_exclusions[0] if remaining_count == 1 else ""
    single_option_is_oracle = bool(oracle_answer and single_remaining_option == oracle_answer)
    single_option_wrong = bool(oracle_answer and single_remaining_option and single_remaining_option != oracle_answer)
    return {
        "schema": "participant_option_exclusion_audit_result_v1",
        "dataset": datapoint.get("dataset"),
        "instance_id": datapoint.get("instance_id"),
        "participant_id": participant_id,
        "prompt_id": audit.get("prompt_id"),
        "oracle_answer": oracle_answer,
        "decision_set": decision_set,
        "excluded_options": excluded,
        "remaining_options_from_exclusions": remaining_from_exclusions,
        "reported_remaining_options": reported_remaining,
        "excluded_option_count": len(excluded),
        "remaining_option_count": remaining_count,
        "reduces_to_oracle": reduces_to_oracle,
        "single_option_remaining": remaining_count == 1,
        "single_remaining_option": single_remaining_option,
        "single_option_is_oracle": single_option_is_oracle,
        "single_option_wrong": single_option_wrong,
        "oracle_excluded": bool(oracle_answer and oracle_answer in set(excluded)),
        "confidence": confidence,
        "confidence_threshold": confidence_threshold,
        "invalid_excluded_options": invalid_excluded,
        "invalid_remaining_options": invalid_remaining,
        "reported_remaining_consistent": set(reported_remaining) == set(remaining_from_exclusions),
        "quote_count": len(quotes),
        "quote_hit_count": quote_hits,
        "reason": response.get("reason"),
        "option_assessments": assessments,
        "request_meta": audit.get("request_meta") or {},
        "raw_response": response,
    }


def ordered_valid_options(values: list[Any], decision_set: list[str]) -> list[str]:
    requested = {str(item) for item in values}
    return [option for option in decision_set if option in requested]


def oracle_for(datapoint: dict[str, Any]) -> str:
    oracle = datapoint.get("oracle") if isinstance(datapoint.get("oracle"), dict) else {}
    return str(oracle.get("answer") or "")


def assessment_quotes(assessments: list[Any]) -> list[str]:
    quotes: list[str] = []
    for row in assessments:
        if not isinstance(row, dict):
            continue
        for quote in row.get("quotes") or []:
            text = str(quote).strip()
            if text:
                quotes.append(text)
    return quotes


def quote_hit_count(quotes: list[str], context_units: list[dict[str, Any]]) -> int:
    context = normalize_quote(" ".join(str(unit.get("text") or "") for unit in context_units))
    count = 0
    for quote in quotes:
        norm = normalize_quote(quote)
        if norm and norm in context:
            count += 1
    return count


def filtered_datapoints(
    datapoints: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    misleading: list[dict[str, Any]] = []
    for datapoint, report in zip(datapoints, reports):
        tagged = tag_datapoint_with_option_exclusion(datapoint, report)
        if report_keep_under_solo_oracle_filter(report):
            kept.append(tagged)
            if report_has_only_wrong_single_options(report):
                misleading.append(tagged)
        else:
            dropped.append(tagged)
    return kept, dropped, misleading


def tag_datapoint_with_option_exclusion(datapoint: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(datapoint)
    generation_metadata = out.get("generation_metadata")
    if not isinstance(generation_metadata, dict):
        generation_metadata = {}
        out["generation_metadata"] = generation_metadata
    generation_metadata["participant_option_exclusion_funnel"] = {
        "schema": "participant_option_exclusion_funnel_v1",
        "keep_under_solo_oracle_filter": report_keep_under_solo_oracle_filter(report),
        "filter_decision": report.get("filter_decision") or (
            "keep" if report_keep_under_solo_oracle_filter(report) else "drop_solo_oracle_reducible"
        ),
        "has_single_participant_oracle_reduction": bool(report.get("has_single_participant_oracle_reduction")),
        "has_any_single_option_reduction": bool(report.get("has_any_single_option_reduction")),
        "has_wrong_single_option_reduction": report_has_wrong_single_option(report),
        "has_only_wrong_single_option_reductions": report_has_only_wrong_single_options(report),
        "solo_oracle_participants": list(report.get("solo_oracle_participants") or []),
        "single_option_participants": list(report.get("single_option_participants") or []),
        "wrong_single_option_participants": list(report.get("wrong_single_option_participants") or []),
        "oracle_single_option_participants": list(report.get("oracle_single_option_participants") or []),
        "report_instance_id": report.get("instance_id"),
    }
    return out


def audit_single_option_wrong(row: dict[str, Any]) -> bool:
    if "single_option_wrong" in row:
        return bool(row.get("single_option_wrong"))
    remaining = row.get("remaining_options_from_exclusions") or []
    oracle_answer = str(row.get("oracle_answer") or "")
    return bool(len(remaining) == 1 and oracle_answer and str(remaining[0]) != oracle_answer)


def audit_single_option_is_oracle(row: dict[str, Any]) -> bool:
    if "single_option_is_oracle" in row:
        return bool(row.get("single_option_is_oracle"))
    remaining = row.get("remaining_options_from_exclusions") or []
    oracle_answer = str(row.get("oracle_answer") or "")
    return bool(len(remaining) == 1 and oracle_answer and str(remaining[0]) == oracle_answer)


def report_keep_under_solo_oracle_filter(row: dict[str, Any]) -> bool:
    if "keep_under_solo_oracle_filter" in row:
        return bool(row.get("keep_under_solo_oracle_filter"))
    return not bool(row.get("has_single_participant_oracle_reduction"))


def report_has_wrong_single_option(row: dict[str, Any]) -> bool:
    if "has_wrong_single_option_reduction" in row:
        return bool(row.get("has_wrong_single_option_reduction"))
    return bool(row.get("has_any_single_option_reduction")) and not bool(
        row.get("has_single_participant_oracle_reduction")
    )


def report_has_only_wrong_single_options(row: dict[str, Any]) -> bool:
    if "has_only_wrong_single_option_reductions" in row:
        return bool(row.get("has_only_wrong_single_option_reductions"))
    return bool(row.get("has_any_single_option_reduction")) and not bool(
        row.get("has_single_participant_oracle_reduction")
    )


def summarize(
    reports: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    selected_count: int,
) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for report in reports:
        dataset = str(report.get("dataset") or "unknown")
        entry = by_dataset.setdefault(dataset, {"datapoints": 0, "solo_oracle_datapoints": 0, "single_option_datapoints": 0})
        keep_under_filter = report_keep_under_solo_oracle_filter(report)
        entry["datapoints"] += 1
        entry["failed_datapoint_audits"] = entry.get("failed_datapoint_audits", 0) + int(bool(report.get("audit_failed")))
        entry["failed_participant_audits"] = entry.get("failed_participant_audits", 0) + int(
            report.get("failed_participant_audit_count") or 0
        )
        entry["kept_datapoints_under_solo_oracle_filter"] = entry.get(
            "kept_datapoints_under_solo_oracle_filter", 0
        ) + int(keep_under_filter)
        entry["dropped_datapoints_solo_oracle_reducible"] = entry.get(
            "dropped_datapoints_solo_oracle_reducible", 0
        ) + int(not keep_under_filter)
        entry["solo_oracle_datapoints"] += int(bool(report.get("has_single_participant_oracle_reduction")))
        entry["single_option_datapoints"] += int(bool(report.get("has_any_single_option_reduction")))
        entry["wrong_single_option_datapoints"] = entry.get("wrong_single_option_datapoints", 0) + int(
            report_has_wrong_single_option(report)
        )
        entry["only_wrong_single_option_datapoints"] = entry.get("only_wrong_single_option_datapoints", 0) + int(
            report_has_only_wrong_single_options(report)
        )
    for row in audit_rows:
        dataset = str(row.get("dataset") or "unknown")
        entry = by_dataset.setdefault(dataset, {"datapoints": 0, "solo_oracle_datapoints": 0, "single_option_datapoints": 0})
        entry["participant_audits"] = entry.get("participant_audits", 0) + 1
        entry["participant_reduces_to_oracle"] = entry.get("participant_reduces_to_oracle", 0) + int(bool(row.get("reduces_to_oracle")))
        entry["participant_single_option_remaining"] = entry.get("participant_single_option_remaining", 0) + int(
            bool(row.get("single_option_remaining"))
        )
        entry["participant_oracle_excluded"] = entry.get("participant_oracle_excluded", 0) + int(
            bool(row.get("oracle_excluded"))
        )
        entry["participant_single_option_wrong"] = entry.get("participant_single_option_wrong", 0) + int(
            audit_single_option_wrong(row)
        )
        entry["participant_single_option_is_oracle"] = entry.get("participant_single_option_is_oracle", 0) + int(
            audit_single_option_is_oracle(row)
        )
        entry["participant_audit_failures"] = entry.get("participant_audit_failures", 0) + int(bool(row.get("audit_failed")))
    solo_reports = [row for row in reports if row.get("has_single_participant_oracle_reduction")]
    single_option_reports = [row for row in reports if row.get("has_any_single_option_reduction")]
    kept_reports = [row for row in reports if report_keep_under_solo_oracle_filter(row)]
    wrong_single_option_reports = [row for row in reports if report_has_wrong_single_option(row)]
    only_wrong_single_option_reports = [row for row in reports if report_has_only_wrong_single_options(row)]
    participant_solo_rows = [row for row in audit_rows if row.get("reduces_to_oracle")]
    participant_single_rows = [row for row in audit_rows if row.get("single_option_remaining")]
    participant_wrong_single_rows = [row for row in audit_rows if audit_single_option_wrong(row)]
    participant_oracle_single_rows = [row for row in audit_rows if audit_single_option_is_oracle(row)]
    failed_datapoint_reports = [row for row in reports if row.get("audit_failed")]
    failed_participant_rows = [row for row in audit_rows if row.get("audit_failed")]
    aggregate = {
        "datapoints_selected": selected_count,
        "datapoints_audited": len(reports),
        "participant_audits": len(audit_rows),
        "datapoints_kept_under_solo_oracle_filter": len(kept_reports),
        "datapoints_dropped_solo_oracle_reducible": len(solo_reports),
        "datapoints_with_single_participant_oracle_reduction": len(solo_reports),
        "datapoints_with_any_single_option_reduction": len(single_option_reports),
        "datapoints_with_wrong_single_option_reduction": len(wrong_single_option_reports),
        "datapoints_with_only_wrong_single_option_reductions": len(only_wrong_single_option_reports),
        "participant_audits_reducing_to_oracle": len(participant_solo_rows),
        "participant_audits_with_single_option_remaining": len(participant_single_rows),
        "participant_audits_with_wrong_single_option": len(participant_wrong_single_rows),
        "participant_audits_with_oracle_single_option": len(participant_oracle_single_rows),
        "failed_datapoint_audits": len(failed_datapoint_reports),
        "failed_participant_audits": len(failed_participant_rows),
        "datapoint_keep_rate_under_solo_oracle_filter": safe_rate(len(kept_reports), len(reports)),
        "datapoint_drop_rate_solo_oracle_reducible": safe_rate(len(solo_reports), len(reports)),
        "datapoint_oracle_reduction_rate": safe_rate(len(solo_reports), len(reports)),
        "datapoint_single_option_rate": safe_rate(len(single_option_reports), len(reports)),
        "datapoint_wrong_single_option_rate": safe_rate(len(wrong_single_option_reports), len(reports)),
        "datapoint_only_wrong_single_option_rate": safe_rate(len(only_wrong_single_option_reports), len(reports)),
        "participant_oracle_reduction_rate": safe_rate(len(participant_solo_rows), len(audit_rows)),
        "participant_single_option_rate": safe_rate(len(participant_single_rows), len(audit_rows)),
        "participant_wrong_single_option_rate": safe_rate(len(participant_wrong_single_rows), len(audit_rows)),
        "participant_oracle_single_option_rate": safe_rate(len(participant_oracle_single_rows), len(audit_rows)),
    }
    return {
        "schema": "participant_option_exclusion_verifier_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "input_root": args.input_root,
            "limit": args.limit,
            "seed": args.seed,
            "shuffle": args.shuffle,
            "dataset": args.dataset,
            "require_leakage_safe": args.require_leakage_safe,
            "workers": args.workers,
            "confidence_threshold": args.confidence_threshold,
            "model": args.model,
            "max_participant_context_chars": args.max_participant_context_chars,
            "max_unit_chars": args.max_unit_chars,
            "continue_on_error": getattr(args, "continue_on_error", False),
        },
        "outputs": {
            "kept_dataset": "solo_oracle_kept_common_datapoints.jsonl",
            "dropped_dataset": "solo_oracle_dropped_datapoints.jsonl",
            "misleading_single_option_dataset": "misleading_single_option_common_datapoints.jsonl",
            "participant_audits": "participant_option_exclusion_audits.jsonl",
            "datapoint_reports": "datapoint_option_exclusion_reports.jsonl",
        },
        "aggregate": aggregate,
        "by_dataset": by_dataset,
        "solo_oracle_examples": participant_solo_rows[:20],
        "wrong_single_option_examples": participant_wrong_single_rows[:20],
        "single_option_examples": participant_single_rows[:20],
    }


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Participant Option-Exclusion Verifier",
        "",
        f"- Datapoints selected: {agg['datapoints_selected']}",
        f"- Datapoints audited: {agg['datapoints_audited']}",
        f"- Participant audits: {agg['participant_audits']}",
        f"- Keep under solo-oracle filter: {agg['datapoints_kept_under_solo_oracle_filter']}",
        f"- Keep rate under solo-oracle filter: {agg['datapoint_keep_rate_under_solo_oracle_filter']:.3f}",
        f"- Drop as solo-oracle reducible: {agg['datapoints_dropped_solo_oracle_reducible']}",
        f"- Drop rate as solo-oracle reducible: {agg['datapoint_drop_rate_solo_oracle_reducible']:.3f}",
        f"- Kept misleading single-option datapoints: {agg['datapoints_with_only_wrong_single_option_reductions']}",
        f"- Kept misleading single-option rate: {agg['datapoint_only_wrong_single_option_rate']:.3f}",
        f"- Datapoints with any single-option reduction: {agg['datapoints_with_any_single_option_reduction']}",
        f"- Datapoint single-option rate: {agg['datapoint_single_option_rate']:.3f}",
        f"- Participant audits reducing to oracle: {agg['participant_audits_reducing_to_oracle']}",
        f"- Participant audits with wrong single option: {agg['participant_audits_with_wrong_single_option']}",
        f"- Participant oracle-reduction rate: {agg['participant_oracle_reduction_rate']:.3f}",
        "",
        "## By Dataset",
        "",
        "| dataset | datapoints | keep | drop solo-oracle | misleading single-option | participant audits | participant oracle-reduced | participant wrong single-option |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, row in sorted(summary["by_dataset"].items()):
        lines.append(
            f"| {dataset} | {row.get('datapoints', 0)} | "
            f"{row.get('kept_datapoints_under_solo_oracle_filter', 0)} | "
            f"{row.get('dropped_datapoints_solo_oracle_reducible', 0)} | "
            f"{row.get('only_wrong_single_option_datapoints', 0)} | "
            f"{row.get('participant_audits', 0)} | "
            f"{row.get('participant_reduces_to_oracle', 0)} | "
            f"{row.get('participant_single_option_wrong', 0)} |"
        )
    lines.extend(["", "## Solo Oracle Examples", ""])
    for row in summary.get("solo_oracle_examples", [])[:10]:
        lines.append(
            f"- `{row.get('dataset')}` `{row.get('instance_id')}` `{row.get('participant_id')}`: "
            f"excluded {row.get('excluded_options')} -> remaining {row.get('remaining_options_from_exclusions')} "
            f"confidence={row.get('confidence')}"
        )
    lines.extend(["", "## Misleading Single-Option Examples", ""])
    for row in summary.get("wrong_single_option_examples", [])[:10]:
        lines.append(
            f"- `{row.get('dataset')}` `{row.get('instance_id')}` `{row.get('participant_id')}`: "
            f"oracle {row.get('oracle_answer')}, remaining {row.get('remaining_options_from_exclusions')} "
            f"confidence={row.get('confidence')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_report(path: Path, summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    cards = [
        ("Datapoints", agg["datapoints_audited"]),
        ("Participant audits", agg["participant_audits"]),
        ("Keep under filter", agg["datapoints_kept_under_solo_oracle_filter"]),
        ("Keep rate", f"{agg['datapoint_keep_rate_under_solo_oracle_filter']:.1%}"),
        ("Drop solo-oracle", agg["datapoints_dropped_solo_oracle_reducible"]),
        ("Drop rate", f"{agg['datapoint_drop_rate_solo_oracle_reducible']:.1%}"),
        ("Kept misleading", agg["datapoints_with_only_wrong_single_option_reductions"]),
        ("Single-option datapoints", agg["datapoints_with_any_single_option_reduction"]),
    ]
    card_html = "\n".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div><div class="metric">{html.escape(str(value))}</div></div>'
        for label, value in cards
    )
    dataset_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(dataset)}</td>"
        f"<td>{row.get('datapoints', 0)}</td>"
        f"<td>{row.get('kept_datapoints_under_solo_oracle_filter', 0)}</td>"
        f"<td>{row.get('dropped_datapoints_solo_oracle_reducible', 0)}</td>"
        f"<td>{row.get('only_wrong_single_option_datapoints', 0)}</td>"
        f"<td>{row.get('participant_audits', 0)}</td>"
        f"<td>{row.get('participant_reduces_to_oracle', 0)}</td>"
        f"<td>{row.get('participant_single_option_wrong', 0)}</td>"
        "</tr>"
        for dataset, row in sorted(summary["by_dataset"].items())
    )
    solo_examples = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('instance_id')))}</td>"
        f"<td>{html.escape(str(row.get('participant_id')))}</td>"
        f"<td><code>{html.escape(', '.join(row.get('excluded_options') or []))}</code></td>"
        f"<td><code>{html.escape(', '.join(row.get('remaining_options_from_exclusions') or []))}</code></td>"
        f"<td>{html.escape(str(row.get('confidence')))}</td>"
        f"<td>{html.escape(str(row.get('reason') or ''))}</td>"
        "</tr>"
        for row in summary.get("solo_oracle_examples", [])[:20]
    )
    wrong_examples = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('instance_id')))}</td>"
        f"<td>{html.escape(str(row.get('participant_id')))}</td>"
        f"<td><code>{html.escape(str(row.get('oracle_answer') or ''))}</code></td>"
        f"<td><code>{html.escape(', '.join(row.get('remaining_options_from_exclusions') or []))}</code></td>"
        f"<td>{html.escape(str(row.get('confidence')))}</td>"
        f"<td>{html.escape(str(row.get('reason') or ''))}</td>"
        "</tr>"
        for row in summary.get("wrong_single_option_examples", [])[:20]
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Participant Option-Exclusion Verifier</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2937; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 18px 0; }}
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
<h1>Participant Option-Exclusion Verifier</h1>
<p>Audits one participant at a time and checks whether that participant can rule out every non-oracle option.</p>
<div class="summary">{card_html}</div>
<h2>By Dataset</h2>
<table><thead><tr><th>Dataset</th><th>Datapoints</th><th>Keep</th><th>Drop solo-oracle</th><th>Misleading single-option</th><th>Participant audits</th><th>Participant oracle-reduced</th><th>Participant wrong single-option</th></tr></thead><tbody>{dataset_rows}</tbody></table>
<h2>Solo Oracle Examples</h2>
<table><thead><tr><th>Instance</th><th>Participant</th><th>Excluded</th><th>Remaining</th><th>Confidence</th><th>Reason</th></tr></thead><tbody>{solo_examples}</tbody></table>
<h2>Misleading Single-Option Examples</h2>
<table><thead><tr><th>Instance</th><th>Participant</th><th>Oracle</th><th>Remaining</th><th>Confidence</th><th>Reason</th></tr></thead><tbody>{wrong_examples}</tbody></table>
</body>
</html>
""",
        encoding="utf-8",
    )


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def leakage_funnel_metadata(datapoint: dict[str, Any]) -> dict[str, Any]:
    generation_meta = datapoint.get("generation_metadata") if isinstance(datapoint.get("generation_metadata"), dict) else {}
    direct_meta = datapoint.get("metadata") if isinstance(datapoint.get("metadata"), dict) else {}
    for meta in (generation_meta, direct_meta):
        funnel = meta.get("participant_leakage_deletion_funnel")
        if isinstance(funnel, dict):
            return funnel
    return {}


def is_leakage_safe(datapoint: dict[str, Any]) -> bool:
    funnel = leakage_funnel_metadata(datapoint)
    return bool(funnel) and bool(funnel.get("converged")) and not bool(funnel.get("excluded"))


if __name__ == "__main__":
    main()
