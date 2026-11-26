"""Blind public-only, all-RIU, and every leave-one-RIU-out audits for MLCR."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from .construction import string_list
from .llm import ResponseCache, structured_call


def validate_solver(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["solver response must be an object"]
    errors: list[str] = []
    answer = value.get("answer")
    if not isinstance(answer, str) and not (
        isinstance(answer, dict)
        and answer
        and all(isinstance(key, str) and isinstance(item, str) for key, item in answer.items())
    ):
        errors.append("solver answer must be text or a nonempty field-to-text object")
    if not isinstance(value.get("insufficient"), bool):
        errors.append("solver insufficient must be boolean")
    if not isinstance(value.get("missing_answer_elements"), list):
        errors.append("solver missing_answer_elements must be a list")
    return errors


def validate_judgment(value: Any, *, claim_ids: set[str], available_numbers: set[str] | None = None) -> list[str]:
    if not isinstance(value, dict):
        return ["judge response must be an object"]
    errors: list[str] = []
    if not isinstance(value.get("fully_correct"), bool):
        errors.append("fully_correct must be boolean")
    results = value.get("claim_results")
    if not isinstance(results, list):
        return errors + ["claim_results must be a list"]
    by_id = {
        str(row.get("claim_id") or ""): row
        for row in results
        if isinstance(row, dict)
    }
    if len(by_id) != len(results) or set(by_id) != claim_ids:
        errors.append("claim_results must cover every rubric claim exactly once")
    for row in by_id.values():
        for field in ("content_correct", "grounded_in_available_material", "satisfied"):
            if not isinstance(row.get(field), bool):
                errors.append(f"each claim result requires boolean {field}")
        if row.get("satisfied") is not (row.get("content_correct") is True and row.get("grounded_in_available_material") is True):
            errors.append("satisfied must equal content_correct AND grounded_in_available_material")
        cited = row.get("material_numbers")
        if not isinstance(cited, list) or not all(isinstance(n, str) for n in cited):
            errors.append("material_numbers must be an array of strings")
        elif available_numbers is not None and not set(cited) <= available_numbers:
            errors.append("judge cited unavailable materials")
        if row.get("grounded_in_available_material") is True and not cited:
            errors.append("grounded claims require actual material citations")
    incomplete = set(string_list(value.get("incomplete_claim_ids")))
    expected_incomplete = {
        claim_id for claim_id, row in by_id.items() if row.get("satisfied") is not True
    }
    if incomplete != expected_incomplete:
        errors.append("incomplete_claim_ids disagree with claim_results")
    if value.get("fully_correct") is True and incomplete:
        errors.append("fully_correct cannot be true with incomplete claims")
    prohibited = value.get("prohibited_errors_present")
    if not isinstance(prohibited, list):
        errors.append("prohibited_errors_present must be a list")
    if value.get("fully_correct") is True and prohibited:
        errors.append("fully_correct cannot be true with prohibited errors")
    expected_content = bool(by_id) and all(r.get("content_correct") is True for r in by_id.values())
    if value.get("content_complete") is not expected_content:
        errors.append("content_complete contradicts claim results")
    expected_full = bool(by_id) and not incomplete and not prohibited
    if value.get("fully_correct") is not expected_full:
        errors.append("fully_correct contradicts claim results and prohibited errors")
    if value.get("fully_correct") is False and not incomplete and not value.get("prohibited_errors_present"):
        errors.append("an incorrect answer needs an incomplete claim or prohibited error")
    return list(dict.fromkeys(errors))


def display_materials(evidence_units: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    ordered = [dict(row) for row in evidence_units]
    random.Random(seed).shuffle(ordered)
    display: list[dict[str, Any]] = []
    hidden: list[dict[str, str]] = []
    for position, row in enumerate(ordered, start=1):
        number = f"material_{position:02d}"
        display.append({"material_number": number, "material_type": "clinical_excerpt", "text": row["text"]})
        hidden.append({
            "material_number": number,
            "unit_id": str(row["unit_id"]),
            "candidate_id": str(row["candidate_id"]),
        })
    return display, hidden


def boundary_outcome(
    *,
    kind: str,
    judgment: dict[str, Any],
    rubric: dict[str, Any],
    omitted_candidate_id: str | None = None,
) -> dict[str, Any]:
    fully_correct = judgment.get("fully_correct") is True and not judgment.get("prohibited_errors_present")
    if kind == "all":
        return {"condition_ok": fully_correct, "all_answerable": fully_correct}
    if kind == "none":
        content_complete = judgment.get("content_complete") is True and not judgment.get("prohibited_errors_present")
        return {"condition_ok": not content_complete, "none_answerable": content_complete}
    if kind != "loo" or not omitted_candidate_id:
        raise ValueError(f"unsupported boundary kind: {kind}")
    affected = {
        str(claim["claim_id"])
        for claim in rubric.get("required_claims") or []
        if omitted_candidate_id in string_list(claim.get("supporting_candidate_ids"))
    }
    incomplete = set(string_list(judgment.get("incomplete_claim_ids")))
    broken = sorted(affected & incomplete)
    return {
        "condition_ok": not fully_correct and bool(broken),
        "loo_answerable": fully_correct,
        "omitted_candidate_id": omitted_candidate_id,
        "affected_claim_ids": sorted(affected),
        "broken_affected_claim_ids": broken,
    }


def run_boundaries(
    *,
    case_id: str,
    task: dict[str, Any],
    evidence_units: list[dict[str, Any]],
    client: Any,
    cache: ResponseCache,
    trace_root: Path,
    attempts: int,
    seed: int,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = [
        {"case_id": "all", "kind": "all", "available": evidence_units},
        {"case_id": "none", "kind": "none", "available": []},
    ]
    for unit in evidence_units:
        candidate_id = str(unit["candidate_id"])
        cases.append({
            "case_id": f"without_{candidate_id}",
            "kind": "loo",
            "omitted_candidate_id": candidate_id,
            "available": [row for row in evidence_units if row["candidate_id"] != candidate_id],
        })
    claim_ids = {str(row["claim_id"]) for row in task["rubric"]["required_claims"]}
    results: list[dict[str, Any]] = []
    for position, case in enumerate(cases):
        display, hidden = display_materials(case["available"], seed=seed + position)
        solver, solver_errors = structured_call(
            client=client,
            cache=cache,
            trace_root=trace_root,
            case_id=case_id,
            stage=f"boundary_solver.{case['case_id']}",
            system_prompt_name="boundary_solver_system.txt",
            user_prompt_name="04_boundary_solver_user.txt",
            variables={"PUBLIC_TASK_JSON": task["public_task"], "DISPLAY_MATERIALS_JSON": display},
            validator=validate_solver,
            attempts=attempts,
        )
        judgment: dict[str, Any] = {}
        judge_errors: list[str] = []
        if not solver_errors:
            judgment, judge_errors = structured_call(
                client=client,
                cache=cache,
                trace_root=trace_root,
                case_id=case_id,
                stage=f"boundary_judge.{case['case_id']}",
                system_prompt_name="boundary_judge_system.txt",
                user_prompt_name="05_boundary_judge_user.txt",
                variables={
                    "PUBLIC_TASK_JSON": task["public_task"],
                    "RUBRIC_JSON": task["rubric"],
                    "SOLVER_ANSWER_JSON": solver,
                    "DISPLAY_MATERIALS_JSON": display,
                    "MATERIAL_MAPPING_JSON": hidden,
                    "CONDITION_JSON": {"kind": case["kind"], "omitted_candidate_id": case.get("omitted_candidate_id")},
                },
                validator=lambda row: validate_judgment(row, claim_ids=claim_ids,
                    available_numbers={r["material_number"] for r in display}),
                attempts=attempts,
            )
        errors = solver_errors + judge_errors
        outcome = (
            boundary_outcome(
                kind=case["kind"],
                judgment=judgment,
                rubric=task["rubric"],
                omitted_candidate_id=case.get("omitted_candidate_id"),
            )
            if not errors else {"condition_ok": False}
        )
        results.append({
            "case_id": case["case_id"],
            "kind": case["kind"],
            "omitted_candidate_id": case.get("omitted_candidate_id"),
            "display_materials": display,
            "hidden_material_mapping": hidden,
            "solver": solver,
            "judgment": judgment,
            "outcome": outcome,
            "errors": errors,
        })
    return {"passed": all(row["outcome"].get("condition_ok") is True for row in results), "cases": results}


def audit_participant_answers(*, case_id, task, partition, client, cache, trace_root, attempts):
    """Check complete final answers from each actual local+common observation."""
    results = []
    from .evaluation import consumer_full_context_materials
    token_budget = partition.get('participant_token_budget', 40000)
    actual_materials = consumer_full_context_materials(partition, token_budget=token_budget)
    claim_ids = {str(row['claim_id']) for row in task['rubric']['required_claims']}
    for participant in partition['participants']:
        pid = participant['participant_id']
        display = [{'material_number': 'material_01', 'material_type': 'clinical_record_packet',
                    'text': actual_materials[pid]}]
        solver, errors = structured_call(
            client=client, cache=cache, trace_root=trace_root, case_id=case_id,
            stage=f'participant_final_answer.solver.{pid}', system_prompt_name='boundary_solver_system.txt',
            user_prompt_name='04_boundary_solver_user.txt',
            variables={'PUBLIC_TASK_JSON': task['public_task'], 'DISPLAY_MATERIALS_JSON': display},
            validator=validate_solver, attempts=attempts)
        judgment = {}
        if not errors:
            judgment, errors = structured_call(
                client=client, cache=cache, trace_root=trace_root, case_id=case_id,
                stage=f'participant_final_answer.judge.{pid}', system_prompt_name='boundary_judge_system.txt',
                user_prompt_name='05_boundary_judge_user.txt',
                variables={'PUBLIC_TASK_JSON': task['public_task'], 'RUBRIC_JSON': task['rubric'],
                           'SOLVER_ANSWER_JSON': solver, 'DISPLAY_MATERIALS_JSON': display,
                           'MATERIAL_MAPPING_JSON': [], 'CONDITION_JSON': {'kind': 'participant_local_observation'}},
                validator=lambda row: validate_judgment(row, claim_ids=claim_ids, available_numbers={'material_01'}),
                attempts=attempts)
        complete = judgment.get('content_complete') is True and not judgment.get('prohibited_errors_present')
        results.append({'participant_id': pid, 'display_materials': display,
                        'actual_consumer_material_sha256': __import__('hashlib').sha256(actual_materials[pid].encode()).hexdigest(),
                        'participant_context_token_budget': token_budget, 'solver': solver,
                        'judgment': judgment, 'errors': errors, 'passed': not errors and not complete})
    return {'passed': bool(results) and all(r['passed'] for r in results), 'participants': results,
            'limitation': 'finite solver/judge evidence, not proof over all possible answers'}
