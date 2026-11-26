"""Question-history helpers for legal task generation funnels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_source_question_history(path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"question history file not found: {source}")
    rows = _read_rows(source)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        entry = summarize_question_attempt(row)
        source_id = str(entry.get("source_id") or "")
        if not source_id:
            continue
        grouped.setdefault(source_id, []).append(entry)
    return grouped


def summarize_question_attempt(row: dict[str, Any]) -> dict[str, Any]:
    plan = row.get("plan") if isinstance(row.get("plan"), dict) else {}
    public_task = plan.get("public_task") if isinstance(plan.get("public_task"), dict) else {}
    plan_summary = row.get("plan_summary") if isinstance(row.get("plan_summary"), dict) else {}
    decision_set = _decision_values(public_task.get("decision_set")) or _decision_values(plan_summary.get("decision_set"))
    oracle = plan.get("oracle") if isinstance(plan.get("oracle"), dict) else {}
    errors = row.get("errors") if isinstance(row.get("errors"), list) else []
    return {
        "source_id": row.get("source_id"),
        "instance_id": row.get("instance_id"),
        "candidate_index": row.get("candidate_index"),
        "status": row.get("status") or ("accepted" if plan else "unknown"),
        "question": public_task.get("objective") or plan_summary.get("question_objective"),
        "decision_set": decision_set,
        "oracle_answer": oracle.get("answer"),
        "errors": [str(error) for error in errors[:5]],
    }


def question_attempt_entry(
    *,
    source_id: str,
    instance_id: str,
    candidate_index: int,
    status: str,
    plan: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    public_task = plan.get("public_task") if isinstance(plan.get("public_task"), dict) else {}
    oracle = plan.get("oracle") if isinstance(plan.get("oracle"), dict) else {}
    return {
        "source_id": source_id,
        "instance_id": instance_id,
        "candidate_index": candidate_index,
        "status": status,
        "question": public_task.get("objective"),
        "decision_set": _decision_values(public_task.get("decision_set")),
        "oracle_answer": oracle.get("answer"),
        "errors": [str(error) for error in errors[:5]],
    }


def question_history_prompt(
    entries: list[dict[str, Any]],
    *,
    max_entries: int = 12,
    max_chars: int = 6000,
) -> str:
    clean = [
        {
            "status": entry.get("status"),
            "question": entry.get("question"),
            "decision_set": entry.get("decision_set") or [],
            "oracle_answer": entry.get("oracle_answer"),
            "errors": entry.get("errors") or [],
        }
        for entry in entries
        if entry.get("question") or entry.get("decision_set") or entry.get("errors")
    ]
    if not clean:
        return "None."
    payload = clean[-max_entries:]
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80].rstrip() + "\n... truncated older/source-history details ..."


def _read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        payload = json.loads(text)
        return [row for row in payload if isinstance(row, dict)]
    if stripped.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            if isinstance(payload.get("rows"), list):
                return [row for row in payload["rows"] if isinstance(row, dict)]
            return [payload]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _decision_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            item_value = str(item.get("value") or "").strip()
        else:
            item_value = str(item or "").strip()
        if item_value:
            out.append(item_value)
    return out
