"""ALL, NONE, and leave-one-RIU-out semantic boundary audits."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import random
from typing import Any

from teamgym.data.medical.evidencebench.llm import (
    ResponseCache,
    structured_call,
)
from teamgym.data.medical.evidencebench.validation import (
    string_list,
    validate_boundary_judgment,
    validate_solver,
)


def _candidate_materials(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for atom in candidate["evidence_atoms"]:
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "atom_id": atom["atom_id"],
                    "sentence_indices": list(atom["sentence_indices"]),
                    "text": atom["text"],
                }
            )
    return rows


def display_materials(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return neutral solver material and a construction-only hidden mapping."""

    ordered = [dict(row) for row in rows]
    random.Random(seed).shuffle(ordered)
    display: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        evidence_number = f"evidence_{index:02d}"
        display.append(
            {
                "evidence_number": evidence_number,
                "material_type": "paper_excerpt",
                "text": row["text"],
            }
        )
        mapping.append(
            {
                "evidence_number": evidence_number,
                "candidate_id": row["candidate_id"],
                "atom_id": row["atom_id"],
                "sentence_indices": row["sentence_indices"],
            }
        )
    return display, mapping


def boundary_case_outcome(
    *,
    kind: str,
    judgment: dict[str, Any],
    rubric: dict[str, Any],
    omitted_candidate_id: str | None = None,
) -> dict[str, Any]:
    """Apply case-specific semantic criteria without an exact-set shortcut."""

    results = {
        str(row.get("claim_id")): row
        for row in judgment.get("claim_results") or []
        if isinstance(row, dict)
    }
    critical_errors = list(judgment.get("critical_errors") or [])
    if kind == "all":
        passed = bool(
            judgment.get("content_complete") is True
            and judgment.get("grounded_complete") is True
            and results
            and all(row.get("satisfied") is True for row in results.values())
            and not critical_errors
        )
        return {"condition_ok": passed, "all_answerable": passed}
    if kind == "none":
        too_easy = judgment.get("content_complete") is True
        return {
            "condition_ok": not too_easy,
            "none_answerable": too_easy,
            "reason": "public task alone produced a complete answer" if too_easy else "complete answer was not produced",
        }
    if kind != "loo" or not omitted_candidate_id:
        raise ValueError(f"unsupported boundary kind: {kind}")
    omitted_claim_ids = [
        str(row["claim_id"])
        for row in rubric.get("required_claims") or []
        if omitted_candidate_id in string_list(row.get("supporting_candidate_ids"))
    ]
    broken = [
        claim_id
        for claim_id in omitted_claim_ids
        if results.get(claim_id, {}).get("satisfied") is not True
    ]
    condition_ok = bool(judgment.get("grounded_complete") is not True and broken)
    return {
        "condition_ok": condition_ok,
        "loo_answerable": judgment.get("content_complete") is True,
        "loo_grounded_complete": judgment.get("grounded_complete") is True,
        "omitted_candidate_id": omitted_candidate_id,
        "omitted_claim_ids": omitted_claim_ids,
        "broken_omitted_claim_ids": broken,
    }


def semantic_rubric(rubric: dict[str, Any]) -> dict[str, Any]:
    """Keep answer criteria and clinical scope, hiding proposed source ownership."""
    result = deepcopy(rubric)
    for claim in result.get("required_claims", []):
        for key in ("supporting_candidate_ids", "candidate_roles", "support_logic"):
            claim.pop(key, None)
    return result


def run_boundaries(
    *,
    instance_id: str,
    task_rubric: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
    client: Any,
    cache: ResponseCache,
    trace_root: Path,
    attempts: int,
    seed: int,
) -> dict[str, Any]:
    all_material = _candidate_materials(selected_candidates)
    cases: list[dict[str, Any]] = [
        {"case_id": "all", "kind": "all", "available": all_material},
        {"case_id": "none", "kind": "none", "available": []},
    ]
    for candidate in selected_candidates:
        candidate_id = str(candidate["candidate_id"])
        cases.append(
            {
                "case_id": f"without_{candidate_id}",
                "kind": "loo",
                "omitted_candidate_id": candidate_id,
                "available": [row for row in all_material if row["candidate_id"] != candidate_id],
            }
        )

    results: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        display, mapping = display_materials(case["available"], seed=seed + case_index)
        available_numbers = {str(row["evidence_number"]) for row in display}
        candidate_by_number = {
            str(row["evidence_number"]): str(row["candidate_id"])
            for row in mapping
        }
        solver, solver_errors = structured_call(
            client=client,
            cache=cache,
            trace_root=trace_root,
            instance_id=instance_id,
            stage=f"boundary_solver.{case['case_id']}",
            system_prompt_name="boundary_solver_system.txt",
            user_prompt_name="04_boundary_solver_user.txt",
            variables={
                "PUBLIC_TASK_JSON": task_rubric["public_task"],
                "AVAILABLE_MATERIAL_JSON": display,
            },
            validator=lambda row, allowed=available_numbers: validate_solver(row, available_numbers=allowed),
            attempts=attempts,
        )
        judgment: dict[str, Any] = {}
        judge_errors: list[str] = []
        if not solver_errors:
            judgment, judge_errors = structured_call(
                client=client,
                cache=cache,
                trace_root=trace_root,
                instance_id=instance_id,
                stage=f"boundary_judge.{case['case_id']}",
                system_prompt_name="boundary_judge_system.txt",
                user_prompt_name="05_boundary_judge_user.txt",
                variables={
                    "PUBLIC_TASK_JSON": task_rubric["public_task"],
                    "RUBRIC_JSON": semantic_rubric(task_rubric["rubric"]),
                    "CANONICAL_ANSWER_JSON": task_rubric["canonical_answer"],
                    "AVAILABLE_MATERIAL_JSON": display,
                    "SOLVER_RESPONSE_JSON": solver,
                },
                validator=lambda row, allowed=available_numbers: validate_boundary_judgment(
                    row,
                    rubric=task_rubric["rubric"],
                    available_numbers=allowed,
                    candidate_by_number=candidate_by_number,
                    enforce_support_logic=False,
                ),
                attempts=attempts,
            )
        outcome = (
            boundary_case_outcome(
                kind=case["kind"],
                judgment=judgment,
                rubric=task_rubric["rubric"],
                omitted_candidate_id=case.get("omitted_candidate_id"),
            )
            if not solver_errors and not judge_errors
            else {"condition_ok": False}
        )
        results.append(
            {
                "case_id": case["case_id"],
                "kind": case["kind"],
                "omitted_candidate_id": case.get("omitted_candidate_id"),
                "display_material": display,
                "hidden_material_mapping": mapping,
                "solver": solver,
                "judgment": judgment,
                "outcome": outcome,
                "errors": solver_errors + judge_errors,
            }
        )
    return {
        "passed": all(row["outcome"].get("condition_ok") is True for row in results),
        "cases": results,
    }


def audit_participant_answers(*, instance_id, task_rubric, partition, client,
                             cache, trace_root, attempts):
    """Test final-answer recoverability from each actual local+common packet."""
    contexts = {r["context_id"]: r for r in partition["contexts"]}
    from .evaluation import consumer_full_context_materials
    token_budget = partition.get("participant_token_budget", 40_000)
    actual_materials = consumer_full_context_materials(partition, token_budget=token_budget)
    results = []
    for participant in partition["participants"]:
        pid = participant["participant_id"]
        display = [{"evidence_number": "evidence_0001", "material_type": "complete_paper_packet",
                    "text": actual_materials[pid]}]
        numbers = {r["evidence_number"] for r in display}
        solver, errors = structured_call(
            client=client, cache=cache, trace_root=trace_root, instance_id=instance_id,
            stage=f"participant_final_answer.solver.{pid}", system_prompt_name="boundary_solver_system.txt",
            user_prompt_name="04_boundary_solver_user.txt",
            variables={"PUBLIC_TASK_JSON": task_rubric["public_task"], "AVAILABLE_MATERIAL_JSON": display},
            validator=lambda row: validate_solver(row, available_numbers=numbers), attempts=attempts)
        judgment = {}
        if not errors:
            judgment, errors = structured_call(
                client=client, cache=cache, trace_root=trace_root, instance_id=instance_id,
                stage=f"participant_final_answer.judge.{pid}", system_prompt_name="boundary_judge_system.txt",
                user_prompt_name="05_boundary_judge_user.txt",
                variables={"PUBLIC_TASK_JSON": task_rubric["public_task"], "RUBRIC_JSON": semantic_rubric(task_rubric["rubric"]),
                           "CANONICAL_ANSWER_JSON": task_rubric["canonical_answer"],
                           "AVAILABLE_MATERIAL_JSON": display,
                           "SOLVER_RESPONSE_JSON": solver},
                validator=lambda row: validate_boundary_judgment(row, rubric=task_rubric["rubric"],
                    available_numbers=numbers, candidate_by_number={}, enforce_support_logic=False), attempts=attempts)
        results.append({"participant_id": pid, "display_material": display,
                        "actual_consumer_material_sha256": __import__('hashlib').sha256(actual_materials[pid].encode()).hexdigest(),
                        "participant_context_token_budget": token_budget, "solver": solver,
                        "judgment": judgment, "errors": errors,
                        "passed": not errors and judgment.get("content_complete") is False})
    return {"passed": bool(results) and all(r["passed"] for r in results), "participants": results,
            "limitation": "finite solver/judge evidence, not proof over all possible answers"}


def final_rius(
    *,
    task_rubric: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
    boundaries: dict[str, Any],
) -> list[dict[str, Any]]:
    claims = task_rubric["rubric"]["required_claims"]
    loo = {row.get("omitted_candidate_id"): row for row in boundaries["cases"] if row["kind"] == "loo"}
    final: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected_candidates, start=1):
        candidate_id = str(candidate["candidate_id"])
        final.append(
            {
                "riu_id": f"riu_{index:02d}",
                "candidate_id": candidate_id,
                "hidden_claim": candidate["hidden_contribution"],
                "required_claim_ids": [
                    str(row["claim_id"])
                    for row in claims
                    if candidate_id in string_list(row.get("supporting_candidate_ids"))
                ],
                "evidence_atoms": candidate["evidence_atoms"],
                "loo_audit": loo.get(candidate_id, {}).get("outcome"),
            }
        )
    return final
