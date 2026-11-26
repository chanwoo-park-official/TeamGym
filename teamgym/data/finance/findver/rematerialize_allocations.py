"""Rematerialize fixed FinDVer tasks under alternate PIU allocations."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from teamgym.llm.construction_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)

from .free_response_funnel import (_run_packet_audits, _run_answerability_panel,
    _participant_answerability_cases, _visible_common_information)
from .integrity import sha256, code_files, runtime_identity, client_identity, write_json
from teamgym.core.information_allocation import PAPER_ALLOCATION_MODES
from .partition import (
    build_full_context_partition,
    remove_surrounding_contexts,
    validate_packet_context_integrity,
)
from .source import load_financial_report


ALTERNATIVE_MODES = tuple(mode for mode in PAPER_ALLOCATION_MODES if mode != "one_each")


def main() -> None:
    args = _parse_args()
    project = Path(__file__).resolve().parents[4]
    frozen_code = {str(p): sha256(p) for p in code_files(project)}
    input_rows = _read_jsonl(Path(args.input).expanduser())
    if args.limit is not None:
        input_rows = input_rows[: args.limit]
    output_root = Path(args.output_root).expanduser()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("Use a new empty allocation output directory; unverified resume is refused")
    producer_manifest = _read_jsonl_manifest(Path(args.input).expanduser())
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "run_manifest.json",
        {
            "schema": "findver_fixed_task_allocation_rematerialization_v1",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": str(Path(args.input).expanduser().resolve()),
            "input_sha256": sha256(args.input),
            "producer_manifest": producer_manifest,
            "reports_root": str(Path(args.reports_root).expanduser().resolve()),
            "scenario_count": len(input_rows),
            "allocation_modes": list(args.mode),
            "audit_model": args.model,
            "audit_scope": "complete_packet_recheck_and_actual_full_and_piu_only_answerability",
            "participant_context_token_budget": args.participant_context_token_budget,
            "packet_repair_rounds": args.packet_repair_rounds,
            "status": "running",
            "code_sha256": frozen_code,
            "runtime": runtime_identity(),
        },
    )
    client = _audit_client(args, output_root / "audit_requests.jsonl")
    manifest = json.loads((output_root / "run_manifest.json").read_text())
    manifest['client'] = client_identity(client)
    _write_json(output_root / "run_manifest.json", manifest)

    for mode in args.mode:
        mode_dir = output_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        results_path = mode_dir / "results.jsonl"
        completed = {
            str(row.get("instance_id"))
            for row in _read_jsonl(results_path)
            if row.get("instance_id")
        }
        for position, wrapper in enumerate(input_rows, start=1):
            instance_id = str(wrapper.get("instance_id") or "")
            if instance_id in completed:
                _print_event("skip", mode, position, len(input_rows), instance_id)
                continue
            result, scenario = _rematerialize_one(
                wrapper=wrapper,
                reports_root=Path(args.reports_root).expanduser(),
                allocation_mode=mode,
                audit_client=client,
                trace_root=output_root / "traces" / mode,
                audit_attempts=args.audit_attempts,
                seed=args.seed,
                packet_repair_rounds=args.packet_repair_rounds,
                participant_context_token_budget=args.participant_context_token_budget,
            )
            if scenario is not None:
                _append_jsonl(mode_dir / "scenarios.jsonl", scenario)
            _append_jsonl(results_path, result)
            _print_event(
                "done",
                mode,
                position,
                len(input_rows),
                instance_id,
                status=result["status"],
            )
        _write_summary(mode_dir)
    _write_overall_summary(output_root, args.mode)
    manifest_path = output_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(args.input) != manifest['input_sha256']:
        raise RuntimeError('Frozen FinDVer allocation input changed during audits')
    if ({str(p): sha256(p) for p in code_files(project)} != frozen_code
            or runtime_identity() != manifest['runtime']
            or sha256(producer_manifest['path']) != producer_manifest['sha256']):
        raise RuntimeError('FinDVer allocation dependencies or producer manifest changed during audits')
    manifest.update(status="completed", outputs={str(p.relative_to(output_root)): sha256(p)
        for mode in args.mode for p in (output_root / mode).glob('*.jsonl')})
    _write_json(manifest_path, manifest)


def _rematerialize_one(
    *,
    wrapper: dict[str, Any],
    reports_root: Path,
    allocation_mode: str,
    audit_client: Any,
    trace_root: Path,
    audit_attempts: int,
    seed: int,
    packet_repair_rounds: int = 3,
    participant_context_token_budget: int = 20000,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    instance_id = str(wrapper.get("instance_id") or "")
    try:
        candidate, selection, plan = _fixed_task_inputs(
            wrapper,
            reports_root=reports_root,
        )
        scenario = build_full_context_partition(
            candidate=candidate,
            selection=selection,
            plan=plan,
            audits={},
            seed=seed,
            allocation_mode=allocation_mode,
            min_participants=2,
        )
    except ValueError as exc:
        message = str(exc)
        status = (
            "ineligible"
            if "No eligible allocation mode" in message
            else "allocation_invalid"
        )
        return _result(instance_id, allocation_mode, status, message), None
    except Exception as exc:  # noqa: BLE001 - preserve batch progress
        return _result(
            instance_id,
            allocation_mode,
            "failed",
            f"{type(exc).__name__}: {exc}",
        ), None

    common = scenario["common_datapoint"]
    common['generation_metadata']['construction_audits_inherited_from_instance_id'] = instance_id
    selected_context_ids = {
        int(context_id)
        for row in selection.get("selected_contexts") or []
        for context_id in row.get("source_context_ids") or [row["context_id"]]
    }
    stages: list[dict[str, Any]] = []
    try:
        packet_audits, packet_errors, leaking_ids = _run_packet_audits(
            scenario=scenario,
            selection=selection,
            plan=plan,
            audit_client=audit_client,
            trace_root=trace_root,
            instance_id=instance_id,
            plan_attempt=1,
            audit_attempts=audit_attempts,
            stages=stages,
        )
    except Exception as exc:  # noqa: BLE001 - preserve batch progress
        return _result(
            instance_id,
            allocation_mode,
            "audit_failed",
            f"{type(exc).__name__}: {exc}",
        ), None
    if packet_errors:
        return _result(
            instance_id,
            allocation_mode,
            "audit_format_error",
            "; ".join(packet_errors),
            packet_audits=packet_audits,
        ), None

    selected_leaks = leaking_ids & selected_context_ids
    if selected_leaks:
        return _result(
            instance_id,
            allocation_mode,
            "selected_riu_leakage",
            f"selected RIU blocks expose foreign RIUs: {sorted(selected_leaks)}",
            packet_audits=packet_audits,
        ), None
    sanitization = remove_surrounding_contexts(
        scenario,
        source_context_ids=leaking_ids,
        selected_riu_context_ids=selected_context_ids,
    )
    integrity_errors = validate_packet_context_integrity(
        scenario,
        selected_riu_context_ids=selected_context_ids,
        removed_context_ids=set(sanitization["removed_context_ids"]),
    )
    if integrity_errors:
        return _result(
            instance_id,
            allocation_mode,
            "integrity_error",
            "; ".join(integrity_errors),
            packet_audits=packet_audits,
        ), None

    common["generation_metadata"]["context_sanitization"] = sanitization
    common["generation_metadata"]["rematerialized_from_instance_id"] = instance_id
    common["dataset_specific"]["interaction_context_variant"] = (
        "sanitized_context"
    )
    common["format_validation"] = scenario["validation"]
    audit_rounds = [{"round": 0, "audits": packet_audits, "removed_context_ids": sanitization["removed_context_ids"]}]
    for repair_round in range(packet_repair_rounds + 1):
        after, errors, leaks = _run_packet_audits(scenario=scenario, selection=selection, plan=plan,
            audit_client=audit_client, trace_root=trace_root / f"recheck_{repair_round}",
            instance_id=instance_id, plan_attempt=1, audit_attempts=audit_attempts, stages=stages)
        audit_rounds.append({"round": repair_round + 1, "audits": after, "removed_context_ids": sorted(leaks)})
        dirty = bool(leaks) or any(r.get("visible") is True for a in after
            for r in a.get("audit", {}).get("unassigned_riu_results", []))
        if errors or leaks & selected_context_ids or (dirty and (not leaks or repair_round == packet_repair_rounds)):
            return _result(instance_id, allocation_mode, "post_removal_audit_failed",
                "; ".join(errors or ["unallocated PIU remains reconstructable after packet repair"]),
                packet_audit_rounds=audit_rounds), None
        if not dirty:
            break
        extra = remove_surrounding_contexts(scenario, source_context_ids=leaks,
                                            selected_riu_context_ids=selected_context_ids)
        sanitization["removed_context_ids"] = sorted(set(sanitization["removed_context_ids"]) | set(extra["removed_context_ids"]))
    from teamgym.interaction.rui_context import rui_only_datapoint
    panels = {}
    for view_name, view in (("full_context", common), ("piu_only", rui_only_datapoint(common))):
        cases = _participant_answerability_cases({"common_datapoint": view}, token_budget=participant_context_token_budget)
        panel, errors = _run_answerability_panel(cases=cases, public_task=plan["public_task"],
            public_information=_visible_common_information(view), oracle=plan["oracle"],
            blind_clients=[("allocation_blind_solver", audit_client)], judge_client=audit_client,
            trace_root=trace_root / view_name, instance_id=instance_id, plan_attempt=1,
            audit_attempts=audit_attempts, stages=stages, require_full_passes=0,
            reject_any_non_full_pass=True, panel_name=f"allocation_{view_name}")
        panels[view_name] = {"panel": panel, "actual_cases": cases, "errors": errors}
        if errors:
            return _result(instance_id, allocation_mode, "participant_answer_failed", "; ".join(errors),
                packet_audit_rounds=audit_rounds, participant_audits=panels), None
    integrity_errors = validate_packet_context_integrity(scenario,
        selected_riu_context_ids=selected_context_ids,
        removed_context_ids=set(sanitization["removed_context_ids"]))
    if integrity_errors:
        return _result(instance_id, allocation_mode, "integrity_error", "; ".join(integrity_errors)), None
    common["generation_metadata"].update(allocation_packet_audits=audit_rounds,
        allocation_participant_audits=panels, allocation_context_token_budget=participant_context_token_budget)
    common['format_validation'] = scenario['validation']
    common['instance_id'] = f"{instance_id}__{allocation_mode}__seed{seed}"
    scenario['instance_id'] = common['instance_id']
    return (
        _result(
            instance_id,
            allocation_mode,
            "accepted",
            packet_audits=packet_audits,
            packet_audit_rounds=audit_rounds, participant_audits=panels,
            removed_context_ids=sanitization["removed_context_ids"],
        ),
        scenario,
    )


def _fixed_task_inputs(
    wrapper: dict[str, Any],
    *,
    reports_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    common = wrapper.get("common_datapoint")
    if not isinstance(common, dict):
        raise ValueError("input row has no common_datapoint")
    source = common.get("source") or {}
    report_name = str(source.get("report") or "")
    if not report_name:
        raise ValueError("fixed task has no source report")
    report = load_financial_report(reports_root, report_name)
    provenance = source.get('construction_provenance') or {}
    report_path = reports_root / report_name
    if provenance.get('report_sha256') and sha256(report_path) != provenance['report_sha256']:
        raise ValueError('FinDVer original report changed after construction')
    candidate = {
        "source_id": wrapper.get("source_id") or source.get("source_id"),
        "subset": source.get("source_subset") or "all",
        "split": source.get("source_split"),
        "report_name": report_name,
        "report_type": source.get("report_type") or report.get("report_type"),
        "report_url": source.get("report_url") or report.get("url"),
        "source_knowledge": list(
            (common.get("initial_common_information") or {}).get(
                "source_knowledge"
            )
            or []
        ),
        "context_policy": "sanitized_native_context_packets",
        "full_report_contexts": [
            {
                "context_id": int(row["id"]),
                "type": str(row.get("type") or "unknown"),
                "text": str(row.get("context") or ""),
            }
            for row in report["context"]
            if isinstance(row, dict)
        ],
    }
    selection = {
        "valid": True,
        "joint_question_theme": "fixed canonical FinDVer task",
        "target_operation": "fixed free response",
        "source_qa_ids_used": list(
            (common.get("dataset_specific") or {}).get("source_qa_ids_used")
            or []
        ),
        "selected_contexts": [
            {
                "unit_id": str(row["rui_id"]),
                "context_id": int((row.get("source_ref") or {}).get("context_id")),
                "source_context_ids": list(
                    (row.get("source_ref") or {}).get("context_ids") or []
                ),
                "context_type": (row.get("source_ref") or {}).get(
                    "context_type"
                ),
                "source_context_types": list(
                    (row.get("source_ref") or {}).get("context_types") or []
                ),
                "text": row.get("text"),
                "information_feature": row.get("information_feature"),
                "entity_scope": row.get("entity_scope"),
                "period_scope": row.get("period_scope"),
                "source_qa_ids": list(row.get("source_qa_ids") or []),
            }
            for row in common.get("reference_information_units") or []
        ],
    }
    task = common.get("task_description") or {}
    plan = {
        "public_task": {
            "title": task.get("title"),
            "question": task.get("question") or task.get("objective"),
            "response_format_note": task.get("response_format_note"),
            "instruction": (common.get("initial_common_information") or {}).get(
                "instruction"
            ),
        },
        "oracle": deepcopy(common.get("oracle") or {}),
    }
    return candidate, selection, plan


def _audit_client(args: argparse.Namespace, log_path: Path) -> OpenAIJsonClient:
    return OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider="bedrock_openai",
            model=args.model,
            base_url=args.base_url,
            endpoint="/responses",
            api_key_env=args.api_key_env,
            env_file=args.env_file,
            temperature=None,
            max_output_tokens=args.max_output_tokens,
            timeout_sec=args.timeout_sec,
            request_retry_attempts=args.request_retry_attempts,
            request_log_path=str(log_path),
        )
    )


def _result(
    instance_id: str,
    mode: str,
    status: str,
    error: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema": "findver_fixed_task_allocation_result_v1",
        "instance_id": instance_id,
        "allocation_mode": mode,
        "status": status,
        "error": error,
        **extra,
    }


def _write_summary(mode_dir: Path) -> None:
    counts: dict[str, int] = {}
    for row in _read_jsonl(mode_dir / "results.jsonl"):
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    _write_json(mode_dir / "summary.json", {"counts": counts})


def _write_overall_summary(output_root: Path, modes: list[str]) -> None:
    by_mode: dict[str, Any] = {}
    for mode in modes:
        path = output_root / mode / "summary.json"
        if path.is_file():
            by_mode[mode] = json.loads(path.read_text(encoding="utf-8"))
    _write_json(output_root / "summary.json", {"by_mode": by_mode})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + os.linesep)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _read_jsonl_manifest(path):
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("FinDVer redistribution requires a certified constructor output")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'accepted' or manifest.get('outputs', {}).get(path.name) != sha256(path):
        raise ValueError("FinDVer constructor output is unaccepted or changed after validation")
    return {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)}


def _print_event(
    event: str,
    mode: str,
    position: int,
    total: int,
    instance_id: str,
    **extra: Any,
) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "mode": mode,
                "position": position,
                "total": total,
                "instance_id": instance_id,
                **extra,
            }
        ),
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--mode",
        action="append",
        choices=PAPER_ALLOCATION_MODES,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N fixed tasks (useful for smoke tests).",
    )
    parser.add_argument("--audit-attempts", type=int, default=3)
    parser.add_argument("--packet-repair-rounds", type=int, default=3)
    parser.add_argument("--participant-context-token-budget", type=int, default=20000)
    parser.add_argument("--model", "--audit-model", dest="model", default="openai.gpt-5.5")
    parser.add_argument(
        "--base-url",
        default="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
    )
    parser.add_argument("--api-key-env", default="AWS_BEARER_TOKEN_BEDROCK")
    parser.add_argument(
        "--env-file",
        default=".env",
    )
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--request-retry-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.mode is None:
        args.mode = list(ALTERNATIVE_MODES)
    if min(args.audit_attempts, args.participant_context_token_budget) < 1 or args.packet_repair_rounds < 0:
        parser.error("Audit and token budgets must be positive; repair rounds nonnegative")
    return args


if __name__ == "__main__":
    main()
