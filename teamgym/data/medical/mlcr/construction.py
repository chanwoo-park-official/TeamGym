"""Deterministic contracts for MLCR QA mapping and composite task creation."""

from __future__ import annotations

import hashlib
from typing import Any


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def atom_catalog(case_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for document in case_index.get("documents") or []:
        for atom in document.get("atoms") or []:
            catalog[str(atom["atom_id"])] = {**atom, "document_id": document["document_id"]}
    return catalog


def validate_qa_mapping(mapping: Any, *, case_index: dict[str, Any], prompt_id: str,
                        available_atom_ids: set[str] | None = None) -> list[str]:
    if not isinstance(mapping, dict):
        return ["QA mapping must be an object"]
    errors: list[str] = []
    if str(mapping.get("prompt_id") or "") != prompt_id:
        errors.append("prompt_id does not match requested source QA")
    candidates = mapping.get("candidate_spans")
    if not isinstance(candidates, list):
        return errors + ["candidate_spans must be a list"]
    catalog = atom_catalog(case_index)
    candidate_ids: list[str] = []
    used_atoms: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            errors.append("candidate span must be an object")
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        candidate_ids.append(candidate_id)
        atom_ids = string_list(candidate.get("atom_ids"))
        if available_atom_ids is not None and not set(atom_ids) <= available_atom_ids:
            errors.append(f"{candidate_id}: references atoms not supplied to this mapping call")
        if not candidate_id or not atom_ids:
            errors.append("each candidate requires candidate_id and atom_ids")
            continue
        if not set(atom_ids) <= set(catalog):
            errors.append(f"{candidate_id}: unknown atom IDs")
            continue
        atoms = [catalog[atom_id] for atom_id in atom_ids]
        if len({atom["document_id"] for atom in atoms}) != 1:
            errors.append(f"{candidate_id}: candidate atoms cross source documents")
        indices = [int(atom["atom_index"]) for atom in atoms]
        if indices != sorted(set(indices)) or indices != list(range(indices[0], indices[-1] + 1)):
            errors.append(f"{candidate_id}: atom IDs must be ordered, unique, and contiguous")
        overlap = used_atoms & set(atom_ids)
        if overlap:
            errors.append(f"{candidate_id}: overlaps another candidate: {sorted(overlap)}")
        used_atoms.update(atom_ids)
        if not str(candidate.get("claim_contribution") or "").strip():
            errors.append(f"{candidate_id}: claim_contribution is required")
        if not str(candidate.get("context_restoration_reason") or "").strip():
            errors.append(f"{candidate_id}: context_restoration_reason is required")
    if any(not value for value in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        errors.append("candidate IDs must be nonempty and unique")
    if mapping.get("answer_reconstructable") is True and not candidates:
        errors.append("a reconstructable answer requires at least one candidate span")
    if not isinstance(mapping.get("answer_reconstructable"), bool):
        errors.append("answer_reconstructable must be boolean")
    return list(dict.fromkeys(errors))


def hydrate_candidates(mapping: dict[str, Any], *, case_index: dict[str, Any]) -> list[dict[str, Any]]:
    """Replace model-selected atom IDs with exact contiguous source slices."""

    catalog = atom_catalog(case_index)
    documents = {
        str(document["document_id"]): document for document in case_index.get("documents") or []
    }
    hydrated: list[dict[str, Any]] = []
    for candidate in mapping.get("candidate_spans") or []:
        atom_ids = string_list(candidate.get("atom_ids"))
        atoms = [catalog[atom_id] for atom_id in atom_ids]
        document_id = str(atoms[0]["document_id"])
        start, end = int(atoms[0]["raw_start"]), int(atoms[-1]["raw_end"])
        text = str(documents[document_id]["text"])[start:end]
        hydrated.append(
            {
                **candidate,
                "source_document_id": document_id,
                "raw_start": start,
                "raw_end": end,
                "text": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    return hydrated


def namespace_candidates(candidates: list[dict[str, Any]], *, prompt_id: str) -> list[dict[str, Any]]:
    """Make model-local candidate IDs stable and unique across source QAs."""

    namespaced: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates, start=1):
        namespaced.append(
            {
                **candidate,
                "model_candidate_id": str(candidate.get("candidate_id") or ""),
                "candidate_id": f"cand_{prompt_id}_{position:02d}",
                "source_prompt_id": prompt_id,
            }
        )
    return namespaced


def hydrate_task_evidence(task: dict[str, Any], *, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach authoritative exact source fields to selected task evidence units."""

    by_id = {str(row["candidate_id"]): row for row in candidates}
    hydrated_units: list[dict[str, Any]] = []
    for unit in task.get("evidence_units") or []:
        candidate_id = str(unit.get("candidate_id") or "")
        source = by_id[candidate_id]
        hydrated_units.append(
            {
                **unit,
                "source_document_id": source["source_document_id"],
                "source_prompt_id": source["source_prompt_id"],
                "atom_ids": list(source["atom_ids"]),
                "raw_start": source["raw_start"],
                "raw_end": source["raw_end"],
                "text": source["text"],
                "sha256": source["sha256"],
            }
        )
    return {**task, "evidence_units": hydrated_units}


def validate_task(
    task: Any,
    *,
    candidates: list[dict[str, Any]],
    valid_prompt_ids: set[str],
    min_source_questions: int = 2,
) -> list[str]:
    if not isinstance(task, dict):
        return ["task must be an object"]
    errors: list[str] = []
    candidate_ids = {str(row.get("candidate_id") or "") for row in candidates}
    selected = string_list(task.get("selected_candidate_ids"))
    source_qas = string_list(task.get("source_question_ids_used"))
    if len(selected) < 2 or len(selected) != len(set(selected)):
        errors.append("at least two unique candidates must be selected")
    if not set(selected) <= candidate_ids:
        errors.append("selected_candidate_ids contain unavailable candidates")
    if len(source_qas) < min_source_questions or len(source_qas) != len(set(source_qas)):
        errors.append(f"at least {min_source_questions} unique source questions must be used")
    if not set(source_qas) <= valid_prompt_ids:
        errors.append("source_question_ids_used are empty or invalid")
    public_task = task.get("public_task")
    question = str(public_task.get("question") or "") if isinstance(public_task, dict) else ""
    if not question.strip():
        errors.append("public question is required")
    lowered = question.casefold()
    if any(token in lowered for token in ("riu_", "candidate_", "gold evidence", "source span")):
        errors.append("public question leaks construction metadata")
    rubric = task.get("rubric")
    claims = rubric.get("required_claims") if isinstance(rubric, dict) else None
    if not isinstance(claims, list) or not claims:
        return errors + ["required_claims are required"]
    claim_ids = [str(row.get("claim_id") or "") for row in claims if isinstance(row, dict)]
    if len(claim_ids) != len(claims) or any(not value for value in claim_ids):
        errors.append("every claim requires a claim_id")
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("claim IDs must be unique")
    supporters_by_claim: dict[str, set[str]] = {}
    for claim in claims:
        if not isinstance(claim, dict):
            errors.append("claims must be objects")
            continue
        claim_id = str(claim.get("claim_id") or "")
        supporters = set(string_list(claim.get("supporting_candidate_ids")))
        supporters_by_claim[claim_id] = supporters
        if not supporters or not supporters <= set(selected):
            errors.append(f"{claim_id}: invalid supporting candidate IDs")
        for field in ("description", "expected_answer_element", "grading_rule"):
            if not str(claim.get(field) or "").strip():
                errors.append(f"{claim_id}: {field} is required")
    mapped = set().union(*supporters_by_claim.values()) if supporters_by_claim else set()
    if mapped != set(selected):
        errors.append("every selected candidate must support at least one required claim")
    candidate_prompt_by_id = {
        str(row.get("candidate_id") or ""): str(row.get("source_prompt_id") or "")
        for row in candidates
    }
    contributing_questions = {
        candidate_prompt_by_id[candidate_id]
        for candidate_id in selected
        if candidate_prompt_by_id.get(candidate_id)
    }
    if contributing_questions != set(source_qas):
        errors.append("every used source question must contribute selected evidence, and no other question may contribute")
    evidence_units = task.get("evidence_units")
    if not isinstance(evidence_units, list) or len(evidence_units) != len(selected):
        errors.append("evidence_units must cover selected candidates exactly once")
    else:
        by_candidate = {str(row.get("candidate_id") or ""): row for row in evidence_units}
        if set(by_candidate) != set(selected):
            errors.append("evidence_units do not match selected candidates")
        for candidate_id, unit in by_candidate.items():
            declared = set(string_list(unit.get("supports_claim_ids")))
            actual = {claim_id for claim_id, supporters in supporters_by_claim.items() if candidate_id in supporters}
            if declared != actual:
                errors.append(f"{candidate_id}: bidirectional claim mapping mismatch")
            source = next((row for row in candidates if str(row.get("candidate_id")) == candidate_id), None)
            if source is not None and "text" in unit:
                for field in ("source_document_id", "raw_start", "raw_end", "text", "sha256"):
                    if unit.get(field) != source.get(field):
                        errors.append(f"{candidate_id}: hydrated {field} differs from source candidate")
    oracle = task.get("oracle")
    if not isinstance(oracle, dict) or not str(oracle.get("reference_answer") or "").strip():
        errors.append("oracle.reference_answer is required")
    return list(dict.fromkeys(errors))


def validate_grounding_audit(value: Any, *, task: dict[str, Any]) -> list[str]:
    if not isinstance(value, dict):
        return ["grounding audit must be an object"]
    errors: list[str] = []
    claim_ids = {str(row["claim_id"]) for row in task["rubric"]["required_claims"]}
    audits = value.get("claim_audits")
    if not isinstance(audits, list):
        return ["claim_audits must be a list"]
    by_id = {str(row.get("claim_id") or ""): row for row in audits if isinstance(row, dict)}
    if len(audits) != len(claim_ids) or set(by_id) != claim_ids:
        errors.append("claim_audits must cover every required claim exactly once")
    for claim_id, audit in by_id.items():
        if audit.get("grounded") is not True:
            errors.append(f"{claim_id}: claim is not grounded")
        if audit.get("scope_preserved") is not True:
            errors.append(f"{claim_id}: source scope is not preserved")
    if value.get("reference_answer_fully_grounded") is not True:
        errors.append("reference answer is not fully grounded")
    if value.get("all_units_materially_necessary") is not True:
        errors.append("not every selected unit is materially necessary")
    if value.get("passed") is not True:
        errors.append("grounding audit did not pass")
    if value.get("issues") or value.get("errors"):
        errors.append("grounding audit reported issues or errors")
    return list(dict.fromkeys(errors))
