"""Bind accepted medical exports to the payload and artifacts actually audited."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _payload_hash(payload: dict) -> str:
    frozen = {k: v for k, v in payload.items() if k != "export_certification"}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def certify_export(payload: dict, artifacts: dict[str, Path]) -> dict:
    return {"schema": "medical_export_certification_v1", "payload_sha256": _payload_hash(payload),
            "artifact_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in artifacts.items()}}


def validate_export_certification(payload: dict, artifacts: dict[str, Path]) -> None:
    certificate = payload.get("export_certification")
    if not isinstance(certificate, dict):
        raise ValueError("Medical export requires a completed construction certification")
    try:
        expected = certify_export(payload, artifacts)
    except FileNotFoundError as exc:
        raise ValueError("Medical export is missing a certified construction artifact") from exc
    if certificate != expected:
        raise ValueError("Medical export payload or artifacts changed after certification; rerun affected audits")


def require_passed_audit(audit: dict, label: str) -> None:
    if not isinstance(audit, dict) or audit.get("passed") is not True or audit.get("errors"):
        raise ValueError(f"Medical export requires a passed {label}")


def require_participant_audits(audit: dict, participants: list[dict], *, token_budget: int) -> None:
    require_passed_audit(audit, "participant final-answer audit")
    rows = audit.get("participants")
    expected = {p["participant_id"] for p in participants}
    if (not expected or not isinstance(rows, list) or len(rows) != len(expected)
            or any(not isinstance(row, dict) for row in rows)
            or {row.get("participant_id") for row in rows} != expected):
        raise ValueError("Medical export requires exactly one audit for every participant")
    for row in rows:
        require_passed_audit(row, f"participant {row['participant_id']}")
        if row.get("participant_context_token_budget") != token_budget:
            raise ValueError("Medical export budget differs from participant certification")


def require_boundary_cases(audit: dict, candidate_ids: list[str]) -> None:
    require_passed_audit(audit, "ALL/NONE/LOO boundary audit")
    rows = audit.get("cases")
    expected = {"all", "none", *(f"without_{cid}" for cid in candidate_ids)}
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or any(not isinstance(row, dict) for row in rows)
            or {row.get("case_id") for row in rows} != expected
            or any(row.get("errors") or row.get("outcome", {}).get("condition_ok") is not True for row in rows)):
        raise ValueError("Medical export requires complete successful ALL/NONE/LOO cases")


def positive_token_budget(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("participant_token_budget must be a positive integer")
    return value


def construction_artifacts(root, dataset):
    names = (['run_manifest.json', '00_source.json', '01_candidates.json', '02_task_rubric.json',
              '03_task_audit.json', '04_boundaries.json', '05_final_rius.hidden.json']
             if dataset == 'evidencebench' else
             ['run_manifest.json', '01_qa_mappings.json', '02_task.json', '03_grounding_audit.json', '04_boundary_audits.json'])
    return {name: Path(root) / name for name in names}
