"""Legal-style build and structural funnels for long-context reasoning."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any

from teamgym.core.release import write_release_datapoints
from teamgym.core.io import read_jsonl, write_json, write_jsonl
from teamgym.core.schema import validate_common_datapoint
from teamgym.data.long_context_reasoning.seed_builder import build_from_seed_specs
from teamgym.core.dataset import iter_common_datapoints


def run_pipeline(
    *,
    seed_spec_path: str | Path,
    output_root: str | Path,
    allocation_mode: str,
    target_context_tokens: int,
    min_participants: int,
    seed: int,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build seed specs and run structural leakage/solo-oracle funnels."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_specs = read_jsonl(seed_spec_path)
    if limit is not None:
        seed_specs = seed_specs[: max(0, int(limit))]

    candidates_root = output / "01_candidates"
    build_report = build_from_seed_specs(
        seed_specs=seed_specs,
        output_root=candidates_root,
        allocation_mode=allocation_mode,
        target_context_tokens=target_context_tokens,
        min_participants=min_participants,
        seed=seed,
    )

    leakage_root = output / "02_leakage_deleted"
    leakage_report = run_structural_leakage_funnel(
        input_root=candidates_root / "common_datapoints.jsonl",
        output_root=leakage_root,
    )

    option_root = output / "03_option_exclusion"
    option_report = run_structural_option_exclusion_funnel(
        input_root=leakage_root / "leakage_deleted_common_datapoints.jsonl",
        output_root=option_root,
    )

    report = {
        "schema": "long_context_reasoning_pipeline_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seed_spec_path": str(seed_spec_path),
            "output_root": str(output),
            "allocation_mode": allocation_mode,
            "target_context_tokens": target_context_tokens,
            "min_participants": min_participants,
            "seed": seed,
            "limit": limit,
        },
        "stages": {
            "candidates": {
                "root": str(candidates_root),
                "common_datapoints": "01_candidates/common_datapoints.jsonl",
                "report": build_report,
            },
            "leakage_deleted": {
                "root": str(leakage_root),
                "common_datapoints": "02_leakage_deleted/leakage_deleted_common_datapoints.jsonl",
                "report": leakage_report,
            },
            "option_exclusion": {
                "root": str(option_root),
                "kept_common_datapoints": "03_option_exclusion/solo_oracle_kept_common_datapoints.jsonl",
                "dropped_common_datapoints": "03_option_exclusion/solo_oracle_dropped_datapoints.jsonl",
                "report": option_report,
            },
        },
    }
    write_json(output / "pipeline_report.json", report)
    write_pipeline_markdown(output / "pipeline_report.md", report)
    write_pipeline_html(output / "pipeline_report.html", report)
    return report


def run_structural_leakage_funnel(
    *,
    input_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run deterministic leakage checks that do not call an LLM."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    datapoints = list(iter_common_datapoints(input_root))
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for datapoint in datapoints:
        tagged, report, audits = structural_leakage_report(datapoint)
        reports.append(report)
        audit_rows.extend(audits)
        if report["excluded"]:
            excluded.append(tagged)
        else:
            kept.append(tagged)

    write_jsonl(output / "leakage_deleted_common_datapoints.jsonl", kept)
    write_jsonl(output / "participant_leakage_audits.jsonl", audit_rows)
    write_jsonl(output / "excluded_datapoints.jsonl", excluded)
    write_jsonl(output / "excluded_unconverged_datapoints.jsonl", excluded)
    write_jsonl(output / "datapoint_funnel_reports.jsonl", reports)
    summary = summarize_structural_leakage(datapoints, kept, excluded, audit_rows, reports)
    write_json(output / "leakage_deletion_report.json", summary)
    write_simple_markdown(output / "leakage_deletion_report.md", "Structural Leakage Funnel", summary)
    write_simple_html(output / "leakage_deletion_report.html", "Structural Leakage Funnel", summary)
    return summary


def structural_leakage_report(datapoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    out = copy.deepcopy(datapoint)
    out["format_validation"] = validate_common_datapoint(out)
    audits: list[dict[str, Any]] = []
    report = {
        "schema": "long_context_reasoning_structural_leakage_datapoint_report_v1",
        "dataset": out.get("dataset"),
        "instance_id": out.get("instance_id"),
        "converged": True,
        "excluded": False,
        "exclusion_reasons": [],
        "structural_riu_leaks": [],
        "structural_riu_leak_count": 0,
        "missing_owner_rui_hits": [],
        "nonowner_rui_hits": [],
        "schema_valid": bool((out.get("format_validation") or {}).get("valid")),
        "deletion_count": 0,
        "deletion_events": [],
    }

    if not report["schema_valid"]:
        report["excluded"] = True
        report["exclusion_reasons"].append("invalid_common_schema")

    common_rui_ids = public_common_rui_ids(out)
    rui_by_id = reference_information_by_id(out)
    context_by_participant = participant_context_texts(out)
    owner_by_rui = owners_by_rui(out)
    required_ids = oracle_required_rui_ids(out)

    for rui_id in required_ids:
        unit = rui_by_id.get(rui_id)
        if not unit:
            report["excluded"] = True
            report["exclusion_reasons"].append("missing_required_rui")
            continue
        text = str(unit.get("text") or "")
        owners = set(owner_by_rui.get(rui_id, []))
        if rui_id in common_rui_ids:
            owners = set()
        owner_hits = sorted(pid for pid in owners if text and text in context_by_participant.get(pid, ""))
        nonowner_hits = sorted(
            pid
            for pid, context in context_by_participant.items()
            if pid not in owners and rui_id not in common_rui_ids and text and text in context
        )
        if owners and not owner_hits:
            report["missing_owner_rui_hits"].append({"rui_id": rui_id, "owners": sorted(owners)})
        if nonowner_hits:
            report["nonowner_rui_hits"].append({"rui_id": rui_id, "participants": nonowner_hits})
        audits.append(
            {
                "schema": "long_context_reasoning_structural_leakage_audit_v1",
                "dataset": out.get("dataset"),
                "instance_id": out.get("instance_id"),
                "rui_id": rui_id,
                "owners": sorted(owners),
                "common": rui_id in common_rui_ids,
                "owner_hits": owner_hits,
                "nonowner_hits": nonowner_hits,
                "leak_found": bool(nonowner_hits),
                "missing_owner_hit": bool(owners and not owner_hits),
            }
        )

    if report["missing_owner_rui_hits"]:
        report["excluded"] = True
        report["exclusion_reasons"].append("missing_owner_rui_text")
    if report["nonowner_rui_hits"]:
        report["excluded"] = True
        report["exclusion_reasons"].append("nonowner_rui_text_leak")

    report["structural_riu_leaks"] = [
        {
            "rui_id": row["rui_id"],
            "participants": row["participants"],
            "recommended_action": "exclude_datapoint",
        }
        for row in report["nonowner_rui_hits"]
    ]
    report["structural_riu_leak_count"] = len(report["structural_riu_leaks"])
    report["exclusion_reasons"] = sorted(set(report["exclusion_reasons"]))
    tag_leakage_metadata(out, report)
    return out, report, audits


def run_structural_option_exclusion_funnel(
    *,
    input_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run a deterministic solo-oracle proxy based on RUI ownership."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    datapoints = list(iter_common_datapoints(input_root))
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    misleading: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for datapoint in datapoints:
        tagged, report, audits = structural_option_report(datapoint)
        reports.append(report)
        audit_rows.extend(audits)
        if report["keep_under_solo_oracle_filter"]:
            kept.append(tagged)
            if report["has_only_wrong_single_option_reductions"]:
                misleading.append(tagged)
        else:
            dropped.append(tagged)

    write_release_datapoints(output / "solo_oracle_kept_common_datapoints.jsonl", kept)
    write_jsonl(output / "solo_oracle_dropped_datapoints.jsonl", dropped)
    write_jsonl(output / "misleading_single_option_common_datapoints.jsonl", misleading)
    write_jsonl(output / "participant_option_exclusion_audits.jsonl", audit_rows)
    write_jsonl(output / "datapoint_option_exclusion_reports.jsonl", reports)
    summary = summarize_structural_option(datapoints, kept, dropped, misleading, audit_rows, reports)
    write_json(output / "option_exclusion_report.json", summary)
    write_simple_markdown(output / "option_exclusion_report.md", "Structural Option-Exclusion Funnel", summary)
    write_simple_html(output / "option_exclusion_report.html", "Structural Option-Exclusion Funnel", summary)
    return summary


def structural_option_report(datapoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    out = copy.deepcopy(datapoint)
    decision_set = [str(item) for item in out.get("decision_set", []) if str(item)]
    oracle_answer = str(((out.get("oracle") or {}).get("answer")) or "")
    common_rui_ids = public_common_rui_ids(out)
    sufficient_sets = oracle_sufficient_sets(out)
    audits: list[dict[str, Any]] = []
    solo_oracle_participants: list[str] = []

    for participant in out.get("participants", []) or []:
        if not isinstance(participant, dict):
            continue
        participant_id = str(participant.get("participant_id") or "")
        if not participant_id:
            continue
        owned = set(str(item) for item in participant.get("owned_rui_ids", []) if str(item))
        available = owned | common_rui_ids
        has_complete_set = any(sufficient_set <= available for sufficient_set in sufficient_sets)
        remaining = [oracle_answer] if has_complete_set and oracle_answer in decision_set else list(decision_set)
        excluded = [option for option in decision_set if option not in set(remaining)]
        if has_complete_set:
            solo_oracle_participants.append(participant_id)
        audits.append(
            {
                "schema": "participant_option_exclusion_audit_result_v1",
                "auditor": "deterministic_rui_coverage_proxy",
                "dataset": out.get("dataset"),
                "instance_id": out.get("instance_id"),
                "participant_id": participant_id,
                "oracle_answer": oracle_answer,
                "decision_set": decision_set,
                "owned_rui_ids": sorted(owned),
                "common_rui_ids": sorted(common_rui_ids),
                "available_rui_ids": sorted(available),
                "excluded_options": excluded,
                "remaining_options_from_exclusions": remaining,
                "reported_remaining_options": remaining,
                "excluded_option_count": len(excluded),
                "remaining_option_count": len(remaining),
                "reduces_to_oracle": bool(has_complete_set),
                "single_option_remaining": len(remaining) == 1,
                "single_remaining_option": remaining[0] if len(remaining) == 1 else "",
                "single_option_is_oracle": bool(len(remaining) == 1 and remaining[0] == oracle_answer),
                "single_option_wrong": False,
                "oracle_excluded": False,
                "confidence": 1.0 if has_complete_set else 0.0,
                "confidence_threshold": 1.0,
                "invalid_excluded_options": [],
                "invalid_remaining_options": [],
                "reported_remaining_consistent": True,
                "quote_count": 0,
                "quote_hit_count": 0,
                "reason": (
                    "Participant covers a full oracle sufficient RUI set."
                    if has_complete_set
                    else "Participant does not cover a full oracle sufficient RUI set."
                ),
                "option_assessments": [],
                "request_meta": {},
                "raw_response": {},
            }
        )

    keep = not bool(solo_oracle_participants)
    report = {
        "schema": "participant_option_exclusion_datapoint_report_v1",
        "auditor": "deterministic_rui_coverage_proxy",
        "dataset": out.get("dataset"),
        "instance_id": out.get("instance_id"),
        "oracle_answer": oracle_answer,
        "participant_count": len(out.get("participants", []) or []),
        "audited_participant_count": len(audits),
        "keep_under_solo_oracle_filter": keep,
        "filter_decision": "keep_no_single_option_reduction" if keep else "drop_solo_oracle_reducible",
        "has_single_participant_oracle_reduction": bool(solo_oracle_participants),
        "has_any_single_option_reduction": bool(solo_oracle_participants),
        "has_wrong_single_option_reduction": False,
        "has_only_wrong_single_option_reductions": False,
        "solo_oracle_participants": solo_oracle_participants,
        "single_option_participants": solo_oracle_participants,
        "wrong_single_option_participants": [],
        "oracle_single_option_participants": solo_oracle_participants,
        "min_remaining_option_count": min((int(row["remaining_option_count"]) for row in audits), default=0),
        "max_excluded_option_count": max((int(row["excluded_option_count"]) for row in audits), default=0),
    }
    tag_option_metadata(out, report)
    return out, report, audits


def tag_leakage_metadata(datapoint: dict[str, Any], report: dict[str, Any]) -> None:
    metadata = datapoint.setdefault("generation_metadata", {})
    metadata["participant_leakage_deletion_funnel"] = {
        "schema": "participant_leakage_deletion_funnel_v1",
        "auditor": "deterministic_structural",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_rounds": 0,
        "confidence_threshold": 1.0,
        "dry_run": False,
        "exclude_unconverged": True,
        "exclude_empty_participants": False,
        "deletions": [],
        "structural_riu_leaks": report.get("structural_riu_leaks", []),
        "converged": bool(report.get("converged")),
        "excluded": bool(report.get("excluded")),
        "exclusion_reasons": list(report.get("exclusion_reasons") or []),
        "empty_participants_after_deletion": [],
    }


def tag_option_metadata(datapoint: dict[str, Any], report: dict[str, Any]) -> None:
    metadata = datapoint.setdefault("generation_metadata", {})
    metadata["participant_option_exclusion_funnel"] = {
        "schema": "participant_option_exclusion_funnel_v1",
        "auditor": "deterministic_rui_coverage_proxy",
        "keep_under_solo_oracle_filter": bool(report.get("keep_under_solo_oracle_filter")),
        "filter_decision": report.get("filter_decision"),
        "has_single_participant_oracle_reduction": bool(report.get("has_single_participant_oracle_reduction")),
        "has_any_single_option_reduction": bool(report.get("has_any_single_option_reduction")),
        "has_wrong_single_option_reduction": bool(report.get("has_wrong_single_option_reduction")),
        "has_only_wrong_single_option_reductions": bool(report.get("has_only_wrong_single_option_reductions")),
        "solo_oracle_participants": list(report.get("solo_oracle_participants") or []),
        "single_option_participants": list(report.get("single_option_participants") or []),
        "wrong_single_option_participants": list(report.get("wrong_single_option_participants") or []),
        "oracle_single_option_participants": list(report.get("oracle_single_option_participants") or []),
        "report_instance_id": report.get("instance_id"),
    }


def reference_information_by_id(datapoint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("rui_id")): row
        for row in datapoint.get("reference_information_units", []) or []
        if isinstance(row, dict) and row.get("rui_id")
    }


def oracle_required_rui_ids(datapoint: dict[str, Any]) -> list[str]:
    oracle = datapoint.get("oracle") if isinstance(datapoint.get("oracle"), dict) else {}
    ids = oracle.get("required_rui_ids") if isinstance(oracle.get("required_rui_ids"), list) else []
    if ids:
        return [str(item) for item in ids if str(item)]
    sets = oracle_sufficient_sets(datapoint)
    merged: list[str] = []
    seen: set[str] = set()
    for sufficient_set in sets:
        for rui_id in sorted(sufficient_set):
            if rui_id not in seen:
                merged.append(rui_id)
                seen.add(rui_id)
    return merged


def oracle_sufficient_sets(datapoint: dict[str, Any]) -> list[set[str]]:
    oracle = datapoint.get("oracle") if isinstance(datapoint.get("oracle"), dict) else {}
    rows = oracle.get("smallest_sufficient_rui_sets")
    if not isinstance(rows, list):
        return []
    sets: list[set[str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rui_ids = {str(item) for item in row.get("rui_ids", []) if str(item)}
        if rui_ids:
            sets.append(rui_ids)
    return sets


def public_common_rui_ids(datapoint: dict[str, Any]) -> set[str]:
    initial = datapoint.get("initial_common_information") if isinstance(datapoint.get("initial_common_information"), dict) else {}
    rows = initial.get("common_reference_units") if isinstance(initial.get("common_reference_units"), list) else []
    return {str(row.get("rui_id")) for row in rows if isinstance(row, dict) and row.get("rui_id")}


def owners_by_rui(datapoint: dict[str, Any]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for participant in datapoint.get("participants", []) or []:
        if not isinstance(participant, dict):
            continue
        participant_id = str(participant.get("participant_id") or "")
        for rui_id in participant.get("owned_rui_ids", []) or []:
            owners.setdefault(str(rui_id), []).append(participant_id)
    return {rui_id: sorted(set(participants)) for rui_id, participants in owners.items()}


def participant_context_texts(datapoint: dict[str, Any]) -> dict[str, str]:
    texts: dict[str, list[str]] = {}
    for unit in datapoint.get("private_context_units", []) or []:
        if not isinstance(unit, dict):
            continue
        text = str(unit.get("text") or "")
        for participant_id in unit.get("assigned_to", []) or []:
            texts.setdefault(str(participant_id), []).append(text)
    return {participant_id: "\n\n".join(parts) for participant_id, parts in texts.items()}


def summarize_structural_leakage(
    selected: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_counts = Counter(reason for report in reports for reason in report.get("exclusion_reasons", []))
    return {
        "schema": "long_context_reasoning_structural_leakage_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "datapoints_selected": len(selected),
            "datapoints_kept": len(kept),
            "datapoints_excluded": len(excluded),
            "datapoint_keep_rate": safe_rate(len(kept), len(selected)),
            "participant_or_rui_audits": len(audit_rows),
            "nonowner_rui_text_leaks": sum(len(report.get("nonowner_rui_hits", [])) for report in reports),
            "missing_owner_rui_text_hits": sum(len(report.get("missing_owner_rui_hits", [])) for report in reports),
            "invalid_schema_datapoints": sum(1 for report in reports if not report.get("schema_valid")),
        },
        "exclusion_reasons": dict(reason_counts),
        "outputs": {
            "kept_dataset": "leakage_deleted_common_datapoints.jsonl",
            "participant_audits": "participant_leakage_audits.jsonl",
            "datapoint_reports": "datapoint_funnel_reports.jsonl",
            "excluded_dataset": "excluded_datapoints.jsonl",
        },
        "examples": reports[:10],
    }


def summarize_structural_option(
    selected: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    misleading: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    solo_reports = [row for row in reports if row.get("has_single_participant_oracle_reduction")]
    participant_solo_rows = [row for row in audit_rows if row.get("reduces_to_oracle")]
    return {
        "schema": "participant_option_exclusion_verifier_report_v1",
        "auditor": "deterministic_rui_coverage_proxy",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "datapoints_selected": len(selected),
            "datapoints_audited": len(reports),
            "participant_audits": len(audit_rows),
            "datapoints_kept_under_solo_oracle_filter": len(kept),
            "datapoints_dropped_solo_oracle_reducible": len(dropped),
            "datapoints_with_single_participant_oracle_reduction": len(solo_reports),
            "datapoints_with_any_single_option_reduction": len(solo_reports),
            "datapoints_with_wrong_single_option_reduction": len(misleading),
            "datapoints_with_only_wrong_single_option_reductions": len(misleading),
            "participant_audits_reducing_to_oracle": len(participant_solo_rows),
            "participant_audits_with_single_option_remaining": len(participant_solo_rows),
            "participant_audits_with_wrong_single_option": 0,
            "participant_audits_with_oracle_single_option": len(participant_solo_rows),
            "datapoint_keep_rate_under_solo_oracle_filter": safe_rate(len(kept), len(reports)),
            "datapoint_drop_rate_solo_oracle_reducible": safe_rate(len(dropped), len(reports)),
            "participant_oracle_reduction_rate": safe_rate(len(participant_solo_rows), len(audit_rows)),
        },
        "outputs": {
            "kept_dataset": "solo_oracle_kept_common_datapoints.jsonl",
            "dropped_dataset": "solo_oracle_dropped_datapoints.jsonl",
            "misleading_single_option_dataset": "misleading_single_option_common_datapoints.jsonl",
            "participant_audits": "participant_option_exclusion_audits.jsonl",
            "datapoint_reports": "datapoint_option_exclusion_reports.jsonl",
        },
        "solo_oracle_examples": participant_solo_rows[:20],
        "examples": reports[:10],
    }


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def write_simple_markdown(path: str | Path, title: str, summary: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    aggregate = summary.get("aggregate", {})
    for key, value in aggregate.items():
        lines.append(f"- {key}: {value}")
    outputs = summary.get("outputs", {})
    if outputs:
        lines.extend(["", "## Outputs", ""])
        for key, value in outputs.items():
            lines.append(f"- {key}: `{value}`")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_simple_html(path: str | Path, title: str, summary: dict[str, Any]) -> None:
    aggregate = summary.get("aggregate", {})
    rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in aggregate.items()
    )
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:980px;margin:32px auto;line-height:1.45}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:6px 8px;text-align:left}}th{{background:#f6f6f6}}</style>
</head><body><h1>{html.escape(title)}</h1><table>{rows}</table></body></html>
"""
    Path(path).write_text(body, encoding="utf-8")


def write_pipeline_markdown(path: str | Path, report: dict[str, Any]) -> None:
    stages = report["stages"]
    lines = [
        "# Long-Context Reasoning Pipeline Report",
        "",
        f"- Output root: `{report['config']['output_root']}`",
        f"- Seed spec: `{report['config']['seed_spec_path']}`",
        f"- Allocation mode: `{report['config']['allocation_mode']}`",
        f"- Target context tokens: {report['config']['target_context_tokens']}",
        "",
        "## Stage Summary",
        "",
        f"- Candidates built: {stages['candidates']['report'].get('count', 0)}",
        f"- Leakage kept: {stages['leakage_deleted']['report']['aggregate'].get('datapoints_kept', 0)}",
        f"- Leakage excluded: {stages['leakage_deleted']['report']['aggregate'].get('datapoints_excluded', 0)}",
        f"- Option kept: {stages['option_exclusion']['report']['aggregate'].get('datapoints_kept_under_solo_oracle_filter', 0)}",
        f"- Option dropped solo-oracle: {stages['option_exclusion']['report']['aggregate'].get('datapoints_dropped_solo_oracle_reducible', 0)}",
        "",
        "## Evaluation Input",
        "",
        "`03_option_exclusion/solo_oracle_kept_common_datapoints.jsonl`",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pipeline_html(path: str | Path, report: dict[str, Any]) -> None:
    stages = report["stages"]
    rows = {
        "Candidates built": stages["candidates"]["report"].get("count", 0),
        "Leakage kept": stages["leakage_deleted"]["report"]["aggregate"].get("datapoints_kept", 0),
        "Leakage excluded": stages["leakage_deleted"]["report"]["aggregate"].get("datapoints_excluded", 0),
        "Option kept": stages["option_exclusion"]["report"]["aggregate"].get("datapoints_kept_under_solo_oracle_filter", 0),
        "Option dropped solo-oracle": stages["option_exclusion"]["report"]["aggregate"].get("datapoints_dropped_solo_oracle_reducible", 0),
    }
    table = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in rows.items()
    )
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Long-Context Reasoning Pipeline</title>
<style>body{{font-family:system-ui,sans-serif;max-width:980px;margin:32px auto;line-height:1.45}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:6px 8px;text-align:left}}th{{background:#f6f6f6}}</style>
</head><body><h1>Long-Context Reasoning Pipeline</h1><table>{table}</table><p>Evaluation input: <code>03_option_exclusion/solo_oracle_kept_common_datapoints.jsonl</code></p></body></html>
"""
    Path(path).write_text(body, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-spec", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allocation-mode", default="one_each")
    parser.add_argument("--target-context-tokens", type=int, default=12000)
    parser.add_argument("--min-participants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_pipeline(
        seed_spec_path=args.seed_spec,
        output_root=args.output_root,
        allocation_mode=args.allocation_mode,
        target_context_tokens=args.target_context_tokens,
        min_participants=args.min_participants,
        seed=args.seed,
        limit=args.limit,
    )
    compact = {
        "output_root": report["config"]["output_root"],
        "candidate_count": report["stages"]["candidates"]["report"].get("count", 0),
        "leakage_kept": report["stages"]["leakage_deleted"]["report"]["aggregate"].get("datapoints_kept", 0),
        "option_kept": report["stages"]["option_exclusion"]["report"]["aggregate"].get(
            "datapoints_kept_under_solo_oracle_filter", 0
        ),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

