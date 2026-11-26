"""Independent MLCR mapping, composite-task, grounding, and boundary pipeline."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from typing import Any

from .boundary import run_boundaries, audit_participant_answers
from teamgym.data.medical.certification import certify_export, positive_token_budget, construction_artifacts
from .execution import export_artifacts
from .execution import assert_frozen_dependencies
from .construction import (
    atom_catalog,
    hydrate_candidates,
    hydrate_task_evidence,
    namespace_candidates,
    validate_grounding_audit,
    validate_qa_mapping,
    validate_task,
)
from .llm import ResponseCache, structured_call, atomic_text, content_hash, inference_identity, load_prompt
from teamgym.tokenization import count_tokens
from .partition import (
    audit_and_sanitize_partition,
    build_token_budgeted_partition,
    materialize_modes,
    merge_ownership_groups,
    validate_collaborative_ownership,
)


SCHEMA = "mlcr_riu_pipeline_v1"


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(target, json.dumps(value, indent=2, ensure_ascii=False) + os.linesep)


def mapping_source_supply(atoms, qa, *, token_budget):
    """Supply exact atoms, using recorded lexical retrieval only if needed."""
    if token_budget < 1:
        raise ValueError("mapping_source_token_budget must be positive")
    costs = [count_tokens(json.dumps(row, ensure_ascii=False)) for row in atoms]
    if sum(costs) <= token_budget:
        return atoms, {"policy": "complete_case", "supplied": len(atoms), "total": len(atoms),
                       "tokens": sum(costs), "budget": token_budget}
    terms = set(re.findall(r"[a-z0-9]{3,}", str(qa.get("question", "")).lower()))
    stop = {"the", "and", "was", "were", "what", "which", "with", "that", "this", "from", "patient", "claimant"}
    terms -= stop
    scored = sorted(range(len(atoms)), key=lambda i: (-len(terms & set(re.findall(r"[a-z0-9]{3,}", atoms[i]["text"].lower()))), i))
    selected, used = set(), 0
    for i in scored:
        # Include exact adjacent atoms within the same document for local scope.
        group = [j for j in range(max(0, i - 1), min(len(atoms), i + 2))
                 if atoms[j]["source_document_id"] == atoms[i]["source_document_id"] and j not in selected]
        cost = sum(costs[j] for j in group)
        if used + cost <= token_budget:
            selected.update(group)
            used += cost
    return [atoms[i] for i in sorted(selected)], {"policy": "question_lexical_atoms_with_neighbors",
        "supplied": len(selected), "total": len(atoms), "tokens": used, "budget": token_budget,
        "limitation": "unsupplied source may contain additional supporting evidence"}


def public_source_atoms(case_index: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "atom_id": atom_id,
            "source_document_id": row["document_id"],
            "source_order": row["atom_index"],
            "text": row["text"],
        }
        for atom_id, row in atom_catalog(case_index).items()
    ]


def run_case(
    *,
    case_index: dict[str, Any],
    output_root: str | Path,
    generation_client: Any,
    verification_client: Any,
    prompt_ids: list[str] | None = None,
    min_source_questions: int = 2,
    attempts: int = 2,
    generation_attempts: int = 3,
    participant_token_budget: int = 40_000,
    leakage_rounds: int = 8,
    ownership_repair_rounds: int = 3,
    seed: int = 20260913,
    qa_mapping_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
    strict_one_each: bool = False,
    construction_feedback: Any = "None",
    mapping_feedback: Any = "None",
    mapping_source_token_budget: int = 120_000,
    max_input_tokens: int = 180_000,
    construction_only: bool = False,
) -> dict[str, Any]:
    positive_token_budget(participant_token_budget)
    if min_source_questions < 1:
        raise ValueError("min_source_questions must be at least 1")
    case_id = str(case_index["case_uuid"])
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(output / "llm_cache.jsonl", max_input_tokens=max_input_tokens)
    traces = output / "_llm_traces"
    source_qas = {
        str(row["prompt_id"]): row for row in case_index.get("source_questions") or []
    }
    selected_prompt_ids = prompt_ids or list(source_qas)
    unknown = sorted(set(selected_prompt_ids) - set(source_qas))
    if unknown:
        raise ValueError(f"unknown prompt IDs for {case_id}: {unknown}")
    atoms = public_source_atoms(case_index)
    source_identity = content_hash({"case_id": case_id, "atoms": atoms, "source_qas": source_qas})
    source_lineage = {**case_index.get("source_lineage", {}), "case_id": case_id,
                      "case_content_sha256": source_identity}
    execution_identity = {"dependencies": cache.dependencies, "generation": inference_identity(generation_client),
                          "verification": inference_identity(verification_client), "participant_context_token_budget": participant_token_budget}
    package_root = Path(__file__).parent
    prompt_root = package_root.parents[2] / "prompts" / "medical" / "mlcr"
    import hashlib
    write_json(output / "run_manifest.json", {"schema": "mlcr_run_manifest_v2", "source_lineage": source_lineage, "execution_identity": execution_identity,
        "generation": inference_identity(generation_client), "verification": inference_identity(verification_client),
        "selected_prompt_ids": selected_prompt_ids, "min_source_questions": min_source_questions,
        "strict_one_each": strict_one_each, "seed": seed,
        "limits": {"generation_attempts": generation_attempts, "validation_attempts": attempts,
                   "leakage_repairs": leakage_rounds, "ownership_rounds": ownership_repair_rounds,
                   "participant_token_budget": participant_token_budget, "mapping_source_tokens": mapping_source_token_budget,
                   "max_input_tokens": max_input_tokens},
        "code_prompt_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [*package_root.glob("*.py"), *prompt_root.glob("*.txt")]}})
    mappings: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for prompt_id in selected_prompt_ids:
        qa = source_qas[prompt_id]
        supplied_atoms, supply = mapping_source_supply(atoms, qa, token_budget=mapping_source_token_budget)
        key = (case_id, prompt_id)
        identity = content_hash({"source": source_identity, "qa": qa, "supply": supplied_atoms,
                                 "inference": inference_identity(generation_client),
                                 "prompt": load_prompt("01_map_qa_to_source_user.txt") + load_prompt("construction_system.txt")})
        cached_entry = (qa_mapping_cache or {}).get(key) if mapping_feedback == "None" else None
        cached_mapping = (cached_entry.get("mapping") if isinstance(cached_entry, dict)
                          and cached_entry.get("identity") == identity else None)
        cached_errors = (validate_qa_mapping(cached_mapping, case_index=case_index,
                                             prompt_id=prompt_id, available_atom_ids={r["atom_id"] for r in supplied_atoms})
                         if cached_mapping is not None else ["not cached"])
        if cached_mapping is not None and not cached_errors:
            mapping, errors = deepcopy(cached_mapping), []
        else:
            mapping, errors = structured_call(
                client=generation_client,
                cache=cache,
                trace_root=traces,
                case_id=case_id,
                stage=f"map_qa.{prompt_id}",
                user_prompt_name="01_map_qa_to_source_user.txt",
                variables={"SOURCE_QA_JSON": qa, "SOURCE_ATOMS_JSON": supplied_atoms, "SOURCE_SUPPLY_JSON": supply},
                validator=lambda row, pid=prompt_id: validate_qa_mapping(
                    row, case_index=case_index, prompt_id=pid, available_atom_ids={r["atom_id"] for r in supplied_atoms}
                ),
                attempts=attempts,
                initial_feedback=mapping_feedback,
            )
            if qa_mapping_cache is not None and not errors:
                qa_mapping_cache[key] = {"identity": identity, "mapping": deepcopy(mapping)}
        mappings.append({"prompt_id": prompt_id, "response": mapping, "errors": errors,
                         "source_supply": supply, "input_identity": identity,
                         "mapping_cache_hit": cached_mapping is not None and not cached_errors})
        if errors or mapping.get("answer_reconstructable") is not True:
            continue
        hydrated = hydrate_candidates(mapping, case_index=case_index)
        candidates.extend(namespace_candidates(hydrated, prompt_id=prompt_id))
    write_json(output / "01_qa_mappings.json", mappings)
    contributing_questions = {str(row["source_prompt_id"]) for row in candidates}
    if (
        len(candidates) < 2
        or len(contributing_questions) < min_source_questions
    ):
        result = {
            "schema": SCHEMA,
            "case_id": case_id,
            "status": "insufficient_source_candidates",
            "candidate_count": len(candidates),
            "min_source_questions": min_source_questions,
            "contributing_question_ids": sorted(contributing_questions),
        }
        write_json(output / "run_report.json", result)
        return result

    task: dict[str, Any] = {}
    grounding: dict[str, Any] = {}
    boundaries: dict[str, Any] = {}
    feedback: Any = construction_feedback
    generation_history: list[dict[str, Any]] = []
    status = "task_generation_failed"
    for generation_attempt in range(1, max(1, generation_attempts) + 1):
        task, task_errors = structured_call(
            client=generation_client,
            cache=cache,
            trace_root=traces,
            case_id=case_id,
            stage=f"generate_composite_task.round_{generation_attempt}",
            user_prompt_name="02_generate_composite_task_user.txt",
            variables={
                "SOURCE_QAS_JSON": [source_qas[value] for value in selected_prompt_ids],
                "CANDIDATES_JSON": candidates,
                "MIN_SOURCE_QUESTIONS": min_source_questions,
            },
            validator=lambda row: validate_task(
                row, candidates=candidates, valid_prompt_ids=set(selected_prompt_ids),
                min_source_questions=min_source_questions,
            ),
            attempts=attempts,
            initial_feedback=feedback,
        )
        if not task_errors:
            task = hydrate_task_evidence(task, candidates=candidates)
            task_errors = validate_task(
                task, candidates=candidates, valid_prompt_ids=set(selected_prompt_ids),
                min_source_questions=min_source_questions,
            )
        round_record: dict[str, Any] = {
            "generation_attempt": generation_attempt,
            "task": task,
            "task_errors": task_errors,
        }
        if task_errors:
            round_record["status"] = "task_invalid"
            generation_history.append(round_record)
            feedback = {"stage": "task_validation", "errors": task_errors}
            continue

        grounding, grounding_errors = structured_call(
            client=verification_client,
            cache=cache,
            trace_root=traces,
            case_id=case_id,
            stage=f"grounding_audit.round_{generation_attempt}",
            user_prompt_name="03_grounding_audit_user.txt",
            variables={"TASK_JSON": task, "EVIDENCE_JSON": task["evidence_units"]},
            validator=lambda row: validate_grounding_audit(row, task=task),
            attempts=attempts,
        )
        round_record["grounding"] = grounding
        round_record["grounding_errors"] = grounding_errors
        if grounding_errors:
            round_record["status"] = "grounding_failed"
            generation_history.append(round_record)
            feedback = {
                "stage": "grounding_and_minimality",
                "errors": grounding_errors,
                "audit": grounding,
                "instruction": (
                    "Redesign the selected evidence and claim mapping. Remove redundant "
                    "units or use unique necessary contributions already present in the "
                    "available candidates; do not invent claims."
                ),
            }
            status = "grounding_failed"
            continue

        boundaries = run_boundaries(
            case_id=case_id,
            task=task,
            evidence_units=task["evidence_units"],
            client=verification_client,
            cache=cache,
            trace_root=traces,
            attempts=attempts,
            seed=seed + generation_attempt * 1000,
        )
        round_record["boundaries"] = boundaries
        round_record["status"] = "boundary_passed" if boundaries["passed"] else "boundary_failed"
        generation_history.append(round_record)
        if boundaries["passed"]:
            status = "boundary_passed"
            break
        status = "boundary_failed"
        feedback = {
            "stage": "answerability_and_necessity_boundaries",
            "failed_cases": [
                {
                    "case_id": row["case_id"],
                    "outcome": row["outcome"],
                    "errors": row["errors"],
                }
                for row in boundaries["cases"]
                if row["outcome"].get("condition_ok") is not True
            ],
            "instruction": (
                "Redesign using the same source candidates. ALL must be fully answerable, "
                "NONE must not be fully answerable, and removing each selected RIU must "
                "break at least one claim that RIU supports."
            ),
        }
    write_json(output / "02_generation_history.json", generation_history)
    write_json(output / "02_task.json", {"task": task, "errors": generation_history[-1].get("task_errors", [])})
    write_json(output / "03_grounding_audit.json", {
        "response": grounding,
        "errors": generation_history[-1].get("grounding_errors", []),
    })
    if boundaries:
        write_json(output / "04_boundary_audits.json", boundaries)
    partition: dict[str, Any] = {}
    leakage_audit: dict[str, Any] = {}
    modes: dict[str, Any] = {}
    participant_answers: dict[str, Any] = {}
    if status == "boundary_passed":
        assert_frozen_dependencies(cache.dependencies)
        qualified = {"status": "construction_qualified", "task": task, "units": task["evidence_units"],
                     "source_lineage": source_lineage, "execution_identity": execution_identity}
        qualified["export_certification"] = certify_export(qualified, construction_artifacts(output, "mlcr"))
        write_json(output / "qualified_construction.json", qualified)
        if construction_only:
            result = {"status": "construction_qualified", "case_id": case_id,
                "instance_id": f"mlcr_{case_id}_{content_hash({'source': source_identity, 'task': task})[:16]}",
                "task": task, "source_lineage": source_lineage, "execution_identity": execution_identity,
                "grounding_audit": grounding, "boundary_audits": boundaries}
            write_json(output / "run_report.json", result)
            return result
        unit_ids = [str(row["unit_id"]) for row in task["evidence_units"]]
        ownership_groups = [[unit_id] for unit_id in unit_ids]
        known_overlap_pairs: list[list[str]] = []
        ownership_history: list[dict[str, Any]] = []
        for ownership_round in range(1, max(1, ownership_repair_rounds) + 1):
            partition = build_token_budgeted_partition(
                case_index=case_index,
                evidence_units=task["evidence_units"],
                participant_token_budget=participant_token_budget,
                ownership_groups=ownership_groups,
            )
            partition, leakage_audit = audit_and_sanitize_partition(
                case_id=case_id,
                public_task=task["public_task"],
                evidence_units=task["evidence_units"],
                partition=partition,
                client=verification_client,
                cache=cache,
                trace_root=traces / f"ownership_round_{ownership_round}",
                attempts=attempts,
                max_rounds=leakage_rounds,
            )
            ownership_history.append({
                "ownership_round": ownership_round,
                "ownership_groups": ownership_groups,
                "audit": deepcopy(leakage_audit),
            })
            pairs = leakage_audit.get("ownership_merge_pairs") or []
            if not pairs:
                break
            if strict_one_each:
                leakage_audit = {
                    **leakage_audit,
                    "passed": False,
                    "errors": list(leakage_audit.get("errors") or [])
                    + ["canonical foreign-RIU leakage violates strict one-each ownership"],
                }
                break
            known_overlap_pairs.extend(pairs)
            repaired = merge_ownership_groups(unit_ids, known_overlap_pairs)
            if repaired == ownership_groups or len(repaired) < 2:
                leakage_audit = {
                    **leakage_audit,
                    "passed": False,
                    "errors": list(leakage_audit.get("errors") or [])
                    + ["canonical leakage cannot be repaired while preserving collaboration"],
                }
                break
            ownership_groups = repaired
        leakage_audit["ownership_history"] = ownership_history
        ownership_errors = validate_collaborative_ownership(task, partition)
        if ownership_errors:
            leakage_audit["passed"] = False
            leakage_audit["errors"] = list(leakage_audit.get("errors") or []) + ownership_errors
        modes = materialize_modes(task=task, partition=partition)
        write_json(output / "05_partition.json", partition)
        write_json(output / "06_packet_leakage_audit.json", leakage_audit)
        write_json(output / "07_modes.json", modes)
        status = "packet_leakage_failed"
        if leakage_audit.get("passed") is True:
            participant_answers = audit_participant_answers(
                case_id=case_id, task=task, partition=partition, client=verification_client,
                cache=cache, trace_root=traces, attempts=attempts)
            write_json(output / "08_participant_final_answer_audit.json", participant_answers)
            status = "accepted" if participant_answers["passed"] else "participant_final_answer_audit_failed"
    assert_frozen_dependencies(cache.dependencies)
    result = {
        "schema": SCHEMA,
        "execution_identity": execution_identity,
        "case_id": case_id,
        "instance_id": f"mlcr_{case_id}_{content_hash({'source': source_identity, 'task': task})[:16]}",
        "source_lineage": source_lineage,
        "status": status,
        "source_question_ids_considered": selected_prompt_ids,
        "min_source_questions": min_source_questions,
        "source_question_ids_used": task.get("source_question_ids_used", []),
        "candidate_count": len(candidates),
        "final_riu_count": len(task.get("evidence_units") or []),
        "ownership_policy": "strict_one_each" if strict_one_each else "one_each_with_repair_merges",
        "task": task,
        "grounding_audit": grounding,
        "boundary_audits": boundaries,
        "generation_history": generation_history,
        "partition": partition,
        "packet_leakage_audit": leakage_audit,
        "participant_final_answer_audit": participant_answers,
        "modes": modes,
    }
    if status == "accepted":
        result["export_certification"] = certify_export(result, export_artifacts(output))
    write_json(output / "run_report.json", result)
    return result
