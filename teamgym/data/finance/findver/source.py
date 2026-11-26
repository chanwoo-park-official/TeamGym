"""Load native FinDVer examples and their context-indexed SEC reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .integrity import sha256


def load_findver_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("FinDVer data file must contain a JSON array.")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"FinDVer row {index} is not an object.")
        copy = dict(row)
        copy["_source_row_index"] = index
        rows.append(copy)
    return rows


def load_financial_report(
    reports_root: str | Path,
    report_name: str,
) -> dict[str, Any]:
    root = Path(reports_root).expanduser().resolve()
    path = (root / report_name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Report path escapes reports root: {report_name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    contexts = payload.get("context")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"Report has no context array: {path}")
    ids = [row.get("id") for row in contexts if isinstance(row, dict)]
    if len(ids) != len(contexts) or any(not isinstance(value, int) for value in ids):
        raise ValueError(f"Report contexts need integer IDs: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Report contains duplicate context IDs: {path}")
    payload["_source_path"] = str(path.resolve())
    return payload


def findver_report_candidates(
    *,
    qa_path: str | Path,
    reports_root: str | Path,
    subset: str,
    source_split: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one filing with source QAs from one subset or every subset."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    loaded = load_findver_rows(qa_path)
    declared = {str(row.get("split") or "").strip().lower() for row in loaded} - {""}
    if source_split is None and len(declared) > 1:
        raise ValueError("Mixed source splits require an explicit source_split")
    seen_ids: set[str] = set()
    for row in loaded:
        row_split = str(row.get("split") or "").strip().lower()
        if source_split is not None and row_split and row_split != source_split:
            continue
        if source_split is not None and not row_split and source_split not in Path(qa_path).stem.lower():
            raise ValueError("Rows without split need a QA filename declaring the requested source split")
        row_subset = str(row.get("subset") or "").strip().lower()
        if subset != "all" and row_subset != subset:
            continue
        report_name = str(row.get("report") or "").strip()
        example_id = str(row.get("example_id") or "").strip()
        if example_id in seen_ids:
            raise ValueError(f"Duplicate FinDVer example_id: {example_id}")
        seen_ids.add(example_id)
        context_ids = _unique_ints(row.get("relevant_context") or [])
        if not report_name or not example_id or not context_ids:
            continue
        grouped.setdefault(report_name, []).append(row)

    for report_name, rows in grouped.items():
        report = load_financial_report(reports_root, report_name)
        contexts_by_id = {
            int(context["id"]): context
            for context in report["context"]
            if isinstance(context, dict)
        }
        source_qas: list[dict[str, Any]] = []
        for row in rows:
            context_ids = _unique_ints(row.get("relevant_context") or [])
            if any(context_id not in contexts_by_id for context_id in context_ids):
                continue
            contexts = [
                {
                    "context_id": context_id,
                    "type": str(
                        contexts_by_id[context_id].get("type") or "unknown"
                    ),
                    "text": str(
                        contexts_by_id[context_id].get("context") or ""
                    ),
                }
                for context_id in context_ids
            ]
            if any(not context["text"].strip() for context in contexts):
                continue
            source_qas.append(
                {
                    "qa_id": str(row.get("example_id") or ""),
                    "subset": str(row.get("subset") or "").strip().lower(),
                    "statement": row.get("statement"),
                    "entailment_label": row.get("entailment_label"),
                    "explanation": row.get("explanation")
                    or row.get("explaination"),
                    "python_calculation": row.get("python_calculation"),
                    "execution_result": row.get("execution_result"),
                    "construction_knowledge": list(row.get("knowledge") or []),
                    "relevant_context_ids": context_ids,
                    "relevant_contexts": contexts,
                }
            )
        if not source_qas:
            continue
        yield {
            "dataset": "findver",
            "source_id": f"{subset}_{Path(report_name).stem}",
            "subset": subset,
            "split": source_split or rows[0].get("split"),
            "source_provenance": {"report_sha256": sha256(report["_source_path"]),
                                  "qa_sha256": sha256(qa_path),
                                  "qa_row_indices": [row["_source_row_index"] for row in rows]},
            "report_name": report_name,
            "report_type": report.get("report_type"),
            "report_url": report.get("url"),
            "report_source_path": report.get("_source_path"),
            "source_qas": source_qas,
            "source_subsets": sorted(
                {
                    str(qa.get("subset") or "")
                    for qa in source_qas
                    if str(qa.get("subset") or "")
                }
            ),
            "full_report_contexts": [
                {
                    "context_id": int(context["id"]),
                    "type": str(context.get("type") or "unknown"),
                    "text": str(context.get("context") or ""),
                }
                for context in report["context"]
                if isinstance(context, dict)
            ],
        }


def _unique_ints(values: list[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number not in seen:
            seen.add(number)
            out.append(number)
    return out
