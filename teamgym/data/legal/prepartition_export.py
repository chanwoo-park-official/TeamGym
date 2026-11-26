"""Pre-partition task tuple exports for accepted legal generation trials."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from teamgym.core.io import write_json, write_jsonl

PRE_PARTITION_TASK_TUPLE_SCHEMA = "legal_pre_partition_task_tuple_v1"


def write_pre_partition_task_tuples(
    *,
    output_root: str | Path,
    dataset: str,
    min_accept_size: int,
    accepted_trials: Iterable[dict[str, Any]],
) -> dict[str, str]:
    """Write compact RIU/question/options rows for accepted trials before partitioning."""

    output_root_path = Path(output_root).expanduser().resolve()
    rows = [pre_partition_task_tuple(trial, dataset=dataset) for trial in accepted_trials]
    jsonl_path = output_root_path / f"pre_partition_task_tuples_minaccept{min_accept_size}.jsonl"
    pretty_path = output_root_path / f"pre_partition_task_tuples_minaccept{min_accept_size}.pretty.json"
    write_jsonl(jsonl_path, rows)
    write_json(pretty_path, rows)
    return {
        "jsonl": str(jsonl_path),
        "pretty_json": str(pretty_path),
    }


def pre_partition_task_tuple(trial: dict[str, Any], *, dataset: str) -> dict[str, Any]:
    plan = trial.get("plan") if isinstance(trial.get("plan"), dict) else {}
    public_task = plan.get("public_task") if isinstance(plan.get("public_task"), dict) else {}
    oracle = plan.get("oracle") if isinstance(plan.get("oracle"), dict) else {}
    audit = trial.get("audit") if isinstance(trial.get("audit"), dict) else {}
    decision_set = _decision_set(public_task)
    definitions = public_task.get("decision_option_definitions")
    definitions = definitions if isinstance(definitions, dict) else {}
    rius = _rui_units(trial)
    retained_rui_ids = {
        str(rui.get("rui_id"))
        for rui in rius
        if isinstance(rui, dict) and rui.get("rui_id")
    }
    smallest_sufficient_rui_sets = _smallest_sufficient_sets(
        audit,
        retained_rui_ids=retained_rui_ids,
    )
    if dataset == "cuad" and not smallest_sufficient_rui_sets:
        raise ValueError("CUAD export requires an audited smallest sufficient span set.")
    if dataset == "cuad":
        retained_set = min(
            smallest_sufficient_rui_sets,
            key=lambda row: (len(row["rui_ids"]), tuple(row["rui_ids"])),
        )
        retained_rui_ids = set(retained_set["rui_ids"])
        rius = [rui for rui in rius if str(rui.get("rui_id")) in retained_rui_ids]
        smallest_sufficient_rui_sets = [retained_set]

    return {
        "schema": PRE_PARTITION_TASK_TUPLE_SCHEMA,
        "dataset": dataset,
        "instance_id": trial.get("instance_id"),
        "source_id": trial.get("source_id"),
        "candidate_index": trial.get("candidate_index"),
        "source_label": _source_label(trial, dataset=dataset),
        "question": public_task.get("question") or public_task.get("objective"),
        "task_context": public_task.get("contract_context") or public_task.get("context"),
        "decision_set": decision_set,
        "options": [
            {
                "decision_id": decision_id,
                "definition": definitions.get(decision_id, ""),
            }
            for decision_id in decision_set
        ],
        "oracle": {
            "answer": oracle.get("answer"),
            "rationale": oracle.get("rationale"),
            "decision_rule": oracle.get("decision_rule") or oracle.get("rule"),
            "features": oracle.get("features") or {},
        },
        "rius": rius,
        "smallest_sufficient_rui_sets": smallest_sufficient_rui_sets,
        "generation_metadata": _generation_metadata(trial),
    }


def _decision_set(public_task: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in public_task.get("decision_set") or []:
        if isinstance(item, dict):
            value = item.get("value") or item.get("decision_id") or item.get("id")
            if value:
                out.append(str(value))
        elif item is not None:
            out.append(str(item))
    return out


def _rui_units(trial: dict[str, Any]) -> list[dict[str, Any]]:
    plan = trial.get("plan") if isinstance(trial.get("plan"), dict) else {}
    units = plan.get("evidence_units")
    if isinstance(units, list) and units:
        return [_rui_from_plan_unit(unit) for unit in units if isinstance(unit, dict)]

    selection = trial.get("selection") if isinstance(trial.get("selection"), dict) else {}
    fallback_units = selection.get("selected_rui_basis") or selection.get("selected_span_basis") or []
    return [_rui_from_selection_unit(unit) for unit in fallback_units if isinstance(unit, dict)]


def _rui_from_plan_unit(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "rui_id": unit.get("unit_id"),
        "text": unit.get("text"),
        "summary": unit.get("summary"),
        "information_feature": unit.get("information_feature"),
        "criticality": unit.get("criticality"),
        "tags": unit.get("tags") or [],
        "source_ref": unit.get("source_ref") if isinstance(unit.get("source_ref"), dict) else {},
        "decision_relevance": unit.get("decision_relevance") or [],
    }


def _rui_from_selection_unit(unit: dict[str, Any]) -> dict[str, Any]:
    source_ref = {
        key: value
        for key, value in {
            "source_rui_id": unit.get("source_rui_id"),
            "source_span_index": unit.get("source_span_index"),
            "issue_group_id": unit.get("issue_group_id"),
        }.items()
        if value is not None
    }
    return {
        "rui_id": unit.get("unit_id"),
        "text": unit.get("text"),
        "summary": unit.get("summary"),
        "information_feature": unit.get("legal_feature") or unit.get("information_feature"),
        "criticality": unit.get("criticality"),
        "tags": unit.get("tags") or [],
        "source_ref": source_ref,
        "decision_relevance": unit.get("decision_relevance") or [],
    }


def _smallest_sufficient_sets(
    audit: dict[str, Any],
    *,
    retained_rui_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    raw_sets = audit.get("smallest_sufficient_rui_sets") or audit.get("smallest_sufficient_span_sets") or []
    out: list[dict[str, Any]] = []
    for row in raw_sets:
        if not isinstance(row, dict):
            continue
        rui_ids = [str(item) for item in row.get("unit_ids", [])]
        if retained_rui_ids is not None and not set(rui_ids) <= retained_rui_ids:
            continue
        out.append(
            {
                "rui_ids": rui_ids,
                "answer": row.get("answer"),
                "why_sufficient": row.get("why_sufficient"),
                "why_minimal": row.get("why_minimal"),
            }
        )
    return out


def _source_label(trial: dict[str, Any], *, dataset: str) -> dict[str, Any]:
    if dataset == "cuad":
        return {
            "label_family": trial.get("label_family"),
        }
    if dataset == "maud":
        return {
            "contract_name": trial.get("contract_name"),
            "category": trial.get("category"),
        }
    return {}


def _generation_metadata(trial: dict[str, Any]) -> dict[str, Any]:
    metadata = trial.get("generation_metadata")
    if not isinstance(metadata, dict):
        return {}
    return {
        key: metadata.get(key)
        for key in (
            "provider",
            "model",
            "endpoint",
            "min_accept_size",
            "target_min_rui_count",
            "max_rui_count",
            "acceptance_mode",
        )
        if key in metadata
    }
