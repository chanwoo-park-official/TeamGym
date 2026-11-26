"""Validation helpers for the FinDVer free-response generation funnel."""

from __future__ import annotations

import math
import re

from typing import Any, Iterable


ATOMIC_ANSWER_SPEC_TYPES = {"numeric", "text", "classification"}
ANSWER_SPEC_TYPES = {*ATOMIC_ANSWER_SPEC_TYPES, "composite"}
QA_BLOCK_ROLES = {
    "direct_verification",
    "calculation_input",
    "entity_or_segment_scope",
    "period_or_unit_scope",
    "condition_or_exception",
}
CONSOLIDATION_RELATION_TYPES = {
    "exact_duplicate",
    "fully_contained",
    "partial_overlap",
}


def normalize_qa_evidence_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    """Merge repeated references to one native block without losing its roles."""
    merged: dict[int, dict[str, Any]] = {}
    malformed: list[Any] = []
    for row in mapping.get("evidence_blocks") or []:
        if not isinstance(row, dict):
            malformed.append(row)
            continue
        context_id = _as_int(row.get("context_id"))
        if context_id is None:
            malformed.append(row)
            continue
        target = merged.setdefault(
            context_id,
            {
                "context_id": context_id,
                "roles": [],
                "contributions": [],
            },
        )
        roles = list(row.get("roles") or [])
        if row.get("role"):
            roles.insert(0, row["role"])
        contributions = list(row.get("contributions") or [])
        if row.get("contribution"):
            contributions.insert(0, row["contribution"])
        target["roles"] = list(
            dict.fromkeys(
                str(value) for value in target["roles"] + roles if str(value)
            )
        )
        target["contributions"] = list(
            dict.fromkeys(
                str(value).strip()
                for value in target["contributions"] + contributions
                if str(value).strip()
            )
        )

    evidence_blocks: list[Any] = []
    for row in merged.values():
        roles = row.pop("roles")
        contributions = row.pop("contributions")
        row["role"] = roles[0] if roles else ""
        row["roles"] = roles
        row["contribution"] = "; ".join(contributions)
        row["contributions"] = contributions
        evidence_blocks.append(row)
    mapping["evidence_blocks"] = evidence_blocks + malformed
    return mapping


def validate_qa_evidence_mapping(
    mapping: dict[str, Any],
    *,
    qa: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    qa_id = str(qa.get("qa_id") or "")
    if str(mapping.get("qa_id") or "") != qa_id:
        errors.append(f"mapping must reference QA {qa_id}")
    if mapping.get("answer_reconstructable") is not True:
        errors.append(
            "source label or numeric answer is not reconstructable from "
            "the annotated contexts"
        )
    available_ids = {
        int(row["context_id"])
        for row in qa.get("relevant_contexts") or []
        if isinstance(row, dict) and _as_int(row.get("context_id")) is not None
    }
    evidence_blocks = [
        row
        for row in mapping.get("evidence_blocks") or []
        if isinstance(row, dict)
    ]
    normalized = [_as_int(row.get("context_id")) for row in evidence_blocks]
    if not normalized or any(value is None for value in normalized):
        errors.append("evidence_blocks must contain integer context_id values")
    normalized = [value for value in normalized if value is not None]
    if len(normalized) != len(set(normalized)):
        errors.append("evidence_blocks contains duplicate context_id values")
    if not set(normalized) <= available_ids:
        errors.append("evidence_blocks references contexts outside the QA")
    for row in evidence_blocks:
        context_id = row.get("context_id")
        roles = list(row.get("roles") or [])
        if row.get("role"):
            roles.insert(0, row["role"])
        normalized_roles = {str(value) for value in roles if str(value)}
        if not normalized_roles or not normalized_roles <= QA_BLOCK_ROLES:
            errors.append(f"{context_id}: unknown QA evidence role")
        if not str(row.get("contribution") or "").strip():
            errors.append(f"{context_id}: contribution is missing")
    if not str(mapping.get("reconstruction_summary") or "").strip():
        errors.append("reconstruction_summary is required")
    return _dedupe(errors)


def normalize_evidence_consolidation(
    consolidation: dict[str, Any],
) -> dict[str, Any]:
    """Discard no-op exclusions where a block is claimed to cover itself."""
    retained: list[Any] = []
    issues = list(consolidation.get("issues") or [])
    for row in consolidation.get("excluded_contexts") or []:
        if not isinstance(row, dict):
            retained.append(row)
            continue
        context_id = _as_int(row.get("context_id"))
        covered_by = {
            value
            for value in (
                _as_int(item) for item in row.get("covered_by_context_ids") or []
            )
            if value is not None
        }
        if context_id is not None and covered_by == {context_id}:
            issues.append(
                f"Ignored no-op self-exclusion for context {context_id}; "
                "the native block remains retained."
            )
            continue
        retained.append(row)
    consolidation["excluded_contexts"] = retained
    consolidation["issues"] = list(dict.fromkeys(str(value) for value in issues))
    return consolidation


def validate_evidence_consolidation(
    consolidation: dict[str, Any],
    *,
    candidate_context_ids: Iterable[int],
) -> list[str]:
    errors: list[str] = []
    available = {int(value) for value in candidate_context_ids}
    if consolidation.get("valid") is not True:
        errors.append(
            f"evidence consolidation rejected candidate: "
            f"{consolidation.get('failed_reason') or ''}".strip()
        )

    merged_ids: set[int] = set()
    for index, row in enumerate(consolidation.get("merge_groups") or [], start=1):
        if not isinstance(row, dict):
            errors.append(f"merge group {index} must be an object")
            continue
        ids = [_as_int(value) for value in row.get("source_context_ids") or []]
        if len(ids) < 2 or any(value is None for value in ids):
            errors.append(f"merge group {index} needs at least two integer context IDs")
            continue
        normalized = [int(value) for value in ids if value is not None]
        if normalized != sorted(normalized) or any(
            right != left + 1 for left, right in zip(normalized, normalized[1:])
        ):
            errors.append(f"merge group {index} context IDs must be consecutive")
        if not set(normalized) <= available:
            errors.append(f"merge group {index} references unknown contexts")
        if merged_ids & set(normalized):
            errors.append(f"merge group {index} overlaps another merge group")
        merged_ids.update(normalized)
        if not str(row.get("reason") or "").strip():
            errors.append(f"merge group {index} requires reason")

    excluded_ids: set[int] = set()
    for index, row in enumerate(
        consolidation.get("excluded_contexts") or [], start=1
    ):
        if not isinstance(row, dict):
            errors.append(f"excluded context {index} must be an object")
            continue
        context_id = _as_int(row.get("context_id"))
        covered_by = [
            _as_int(value) for value in row.get("covered_by_context_ids") or []
        ]
        if context_id is None or context_id not in available:
            errors.append(f"excluded context {index} references an unknown context")
            continue
        if context_id in excluded_ids:
            errors.append(f"excluded context {context_id} is duplicated")
        excluded_ids.add(context_id)
        if context_id in merged_ids:
            errors.append(f"context {context_id} cannot be merged and excluded")
        normalized_cover = [
            int(value) for value in covered_by if value is not None
        ]
        if (
            not normalized_cover
            or any(value is None for value in covered_by)
            or not set(normalized_cover) <= available
            or context_id in normalized_cover
        ):
            errors.append(
                f"excluded context {context_id} needs valid covering context IDs"
            )
        if str(row.get("relation_type") or "") not in {
            "exact_duplicate",
            "fully_contained",
        }:
            errors.append(
                f"excluded context {context_id} must be exact_duplicate or fully_contained"
            )
        if not str(row.get("reason") or "").strip():
            errors.append(f"excluded context {context_id} requires reason")
    for row in consolidation.get("excluded_contexts") or []:
        if not isinstance(row, dict):
            continue
        covering = {
            value
            for value in (
                _as_int(item) for item in row.get("covered_by_context_ids") or []
            )
            if value is not None
        }
        if covering & excluded_ids:
            errors.append(
                f"excluded context {row.get('context_id')} must be covered by "
                "retained contexts"
            )

    for index, row in enumerate(
        consolidation.get("information_relations") or [], start=1
    ):
        if not isinstance(row, dict):
            errors.append(f"information relation {index} must be an object")
            continue
        relation_type = str(row.get("relation_type") or "")
        if relation_type not in CONSOLIDATION_RELATION_TYPES:
            errors.append(f"information relation {index} has invalid relation_type")
        provider_ids = {
            value
            for value in (
                _as_int(item) for item in row.get("provider_context_ids") or []
            )
            if value is not None
        }
        if len(provider_ids) < 2 or not provider_ids <= available:
            errors.append(
                f"information relation {index} needs at least two valid providers"
            )
        if not str(row.get("duplicated_information") or "").strip():
            errors.append(
                f"information relation {index} requires duplicated_information"
            )
    return _dedupe(errors)


def normalize_free_response_plan(
    plan: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    public = plan.get("public_task")
    public = public if isinstance(public, dict) else {}
    public["answer_format"] = {"type": "free_response"}
    plan["public_task"] = public
    oracle = plan.get("oracle")
    oracle = oracle if isinstance(oracle, dict) else {}
    claim_ids_by_riu: dict[str, list[str]] = {}
    for step in oracle.get("reasoning_steps") or []:
        if not isinstance(step, dict):
            continue
        claim_id = str(step.get("claim_id") or "")
        for riu_id in step.get("uses") or []:
            if claim_id:
                claim_ids_by_riu.setdefault(str(riu_id), []).append(claim_id)
    plan["evidence_units"] = [
        {
            "unit_id": str(row.get("unit_id") or ""),
            "text": row.get("text"),
            "information_feature": row.get("information_feature"),
            "entity_scope": row.get("entity_scope"),
            "period_scope": row.get("period_scope"),
            "source_ref": {
                "context_id": row.get("context_id"),
                "context_ids": list(row.get("source_context_ids") or []),
                "context_type": row.get("context_type"),
            },
            "supports_claim_ids": list(
                dict.fromkeys(
                    claim_ids_by_riu.get(str(row.get("unit_id") or ""), [])
                )
            ),
        }
        for row in selection.get("selected_contexts") or []
        if isinstance(row, dict)
    ]
    oracle["answer"] = "free_response"
    if not isinstance(oracle.get("required_decision_steps"), list):
        oracle["required_decision_steps"] = list(oracle.get("reasoning_steps") or [])
    plan["oracle"] = oracle
    return plan


def validate_free_response_plan(
    plan: dict[str, Any],
    selection: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    public = plan.get("public_task")
    public = public if isinstance(public, dict) else {}
    question = str(public.get("question") or "").strip()
    if not question:
        errors.append("public_task.question is required")
    source_questions = {
        _norm(qa.get("statement"))
        for qa in candidate.get("source_qas") or []
        if isinstance(qa, dict) and str(qa.get("statement") or "").strip()
    }
    if (
        _norm(question) == _norm(candidate.get("source_statement"))
        or _norm(question) in source_questions
    ):
        errors.append("public question must be newly constructed")
    if any(
        token in question.casefold()
        for token in ("riu", "context id", "annotation", "participant", "oracle")
    ):
        errors.append("public question leaks construction metadata")
    if str((public.get("answer_format") or {}).get("type") or "") != "free_response":
        errors.append("public_task.answer_format.type must be free_response")
    if not str(public.get("response_format_note") or "").strip():
        errors.append("public_task.response_format_note is required")

    selected_ids = {
        str(row.get("unit_id"))
        for row in selection.get("selected_contexts") or []
        if isinstance(row, dict) and row.get("unit_id")
    }
    oracle = plan.get("oracle")
    oracle = oracle if isinstance(oracle, dict) else {}
    if oracle.get("answer") != "free_response":
        errors.append("oracle.answer must be free_response")
    if not str(oracle.get("short_answer") or "").strip():
        errors.append("oracle.short_answer is required")
    if not str(oracle.get("canonical_response") or "").strip():
        errors.append("oracle.canonical_response is required")

    answer_spec = oracle.get("answer_spec")
    answer_spec = answer_spec if isinstance(answer_spec, dict) else {}
    answer_type = str(answer_spec.get("type") or "")
    composite_claim_ids: set[str] = set()
    if answer_type not in ANSWER_SPEC_TYPES:
        errors.append(f"answer_spec.type must be one of {sorted(ANSWER_SPEC_TYPES)}")
    if answer_type in ATOMIC_ANSWER_SPEC_TYPES:
        errors.extend(_validate_atomic_answer_spec(answer_spec, prefix="answer_spec"))
    elif answer_type == "composite":
        items = answer_spec.get("items")
        items = items if isinstance(items, list) else []
        if len(items) < 2:
            errors.append("composite answer_spec.items must contain at least two items")
        observed_item_ids: set[str] = set()
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                errors.append(f"composite answer item {index} must be an object")
                continue
            item_id = str(item.get("item_id") or "").strip()
            if not item_id:
                errors.append(f"composite answer item {index} requires item_id")
            elif item_id in observed_item_ids:
                errors.append(f"duplicate composite answer item_id {item_id}")
            else:
                observed_item_ids.add(item_id)
            item_claim_ids = {
                str(value).strip()
                for value in item.get("claim_ids") or []
                if str(value).strip()
            }
            if not item_claim_ids:
                errors.append(
                    f"composite answer item {item_id or index} requires claim_ids"
                )
            composite_claim_ids.update(item_claim_ids)
            item_type = str(item.get("type") or "")
            if item_type not in ATOMIC_ANSWER_SPEC_TYPES:
                errors.append(
                    f"composite answer item {item_id or index} has invalid type"
                )
            else:
                errors.extend(
                    _validate_atomic_answer_spec(
                        item,
                        prefix=f"composite answer item {item_id or index}",
                    )
                )

    steps = oracle.get("reasoning_steps")
    steps = steps if isinstance(steps, list) else []
    if not steps:
        errors.append("oracle.reasoning_steps must be non-empty")
    covered: set[str] = set()
    observed_numbers: set[int] = set()
    observed_claim_ids: set[str] = set()
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errors.append(f"reasoning step {index} must be an object")
            continue
        number = _as_int(step.get("step"))
        if number is None:
            errors.append(f"reasoning step {index} requires an integer step")
        else:
            observed_numbers.add(number)
        claim_id = str(step.get("claim_id") or "").strip()
        if not claim_id:
            errors.append(f"reasoning step {index} requires claim_id")
        elif claim_id in observed_claim_ids:
            errors.append(f"duplicate reasoning claim_id {claim_id}")
        else:
            observed_claim_ids.add(claim_id)
        uses = {str(value) for value in step.get("uses") or []}
        if not uses:
            errors.append(f"reasoning step {index} must use at least one RIU")
        if not uses <= selected_ids:
            errors.append(
                f"reasoning step {index} cites unknown RIUs {sorted(uses - selected_ids)}"
            )
        covered.update(uses)
        if not str(step.get("claim") or "").strip():
            errors.append(f"reasoning step {index} requires claim")
        if not str(step.get("judge_focus") or "").strip():
            errors.append(f"reasoning step {index} requires judge_focus")
        if not str(step.get("expected_answer_element") or "").strip():
            errors.append(
                f"reasoning step {index} requires expected_answer_element"
            )
        if not str(step.get("grading_rule") or "").strip():
            errors.append(f"reasoning step {index} requires grading_rule")
    if observed_numbers and observed_numbers != set(range(1, len(steps) + 1)):
        errors.append("reasoning step numbers must be consecutive from 1")
    if covered != selected_ids:
        errors.append(
            "reasoning_steps must use every selected RIU; "
            f"missing={sorted(selected_ids - covered)}"
        )
    if not composite_claim_ids <= observed_claim_ids:
        errors.append(
            "composite answer items cite unknown claims "
            f"{sorted(composite_claim_ids - observed_claim_ids)}"
        )

    rubric = oracle.get("judge_rubric")
    rubric = rubric if isinstance(rubric, dict) else {}
    criteria = rubric.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        errors.append("oracle.judge_rubric.criteria must be non-empty")
    if not isinstance(oracle.get("disqualifying_errors"), list):
        errors.append("oracle.disqualifying_errors must be a list")
    return _dedupe(errors)


def _validate_atomic_answer_spec(
    answer_spec: dict[str, Any],
    *,
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    answer_type = str(answer_spec.get("type") or "")
    if answer_type == "numeric":
        if not _is_number(answer_spec.get("value")):
            errors.append(f"{prefix} numeric value must be a number")
        if not str(answer_spec.get("unit") or "").strip():
            errors.append(f"{prefix} numeric unit is required")
        tolerance = answer_spec.get("absolute_tolerance", 0)
        if not _is_number(tolerance) or float(tolerance) < 0:
            errors.append(f"{prefix} numeric absolute_tolerance must be non-negative")
    elif answer_type in {"text", "classification"}:
        accepted = answer_spec.get("accepted_conclusions")
        if not isinstance(accepted, list) or not any(
            str(value).strip() for value in accepted
        ):
            errors.append(f"{prefix} accepted_conclusions must be non-empty")
    return errors


def validate_free_grounding_audit(
    audit: dict[str, Any],
    plan: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for key in (
        "valid",
        "question_answerable_from_full_riu_set",
        "short_answer_correct",
        "canonical_response_grounded",
        "all_selected_rius_necessary",
        "answer_spec_correct",
    ):
        if audit.get(key) is not True:
            errors.append(f"grounding audit did not confirm {key}")
    expected_steps = {
        int(step["step"])
        for step in (plan.get("oracle") or {}).get("reasoning_steps") or []
        if isinstance(step, dict) and _as_int(step.get("step")) is not None
    }
    expected_claim_by_step = {
        int(step["step"]): str(step.get("claim_id") or "")
        for step in (plan.get("oracle") or {}).get("reasoning_steps") or []
        if isinstance(step, dict) and _as_int(step.get("step")) is not None
    }
    rows = audit.get("step_results")
    rows = rows if isinstance(rows, list) else []
    by_step = {
        int(row["step"]): row
        for row in rows
        if isinstance(row, dict) and _as_int(row.get("step")) is not None
    }
    if len(by_step) != len(rows) or set(by_step) != expected_steps:
        errors.append("grounding audit must cover every reasoning step")
    for step, row in by_step.items():
        if str(row.get("claim_id") or "") != expected_claim_by_step.get(step):
            errors.append(f"reasoning step {step} claim_id does not match")
        if row.get("grounded") is not True:
            errors.append(f"reasoning step {step} is not grounded")
        if row.get("uses_match") is not True:
            errors.append(f"reasoning step {step} RIU mapping is invalid")
    if audit.get("unsupported_claims"):
        errors.append("grounding audit found unsupported claims")
    return _dedupe(errors)


def validate_free_solver_response(response: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not str(response.get("written_answer") or "").strip():
        errors.append("solver response requires written_answer")
    confidence = response.get("confidence")
    if not _is_number(confidence) or not 0 <= float(confidence) <= 1:
        errors.append("solver confidence must be between 0 and 1")
    return errors


def validate_free_response_judgment(
    judgment: dict[str, Any],
    *,
    oracle: dict[str, Any],
    available_riu_ids: Iterable[str],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(judgment.get("passes"), bool):
        errors.append("judge passes must be boolean")
    if not isinstance(judgment.get("final_answer_correct"), bool):
        errors.append("judge final_answer_correct must be boolean")
    if not isinstance(judgment.get("source_grounded"), bool):
        errors.append("judge source_grounded must be boolean")
    coverage = judgment.get("required_step_coverage")
    if not _is_number(coverage) or not 0 <= float(coverage) <= 1:
        errors.append("judge required_step_coverage must be between 0 and 1")
    expected_claim_ids = {
        str(step.get("claim_id") or "")
        for step in oracle.get("reasoning_steps") or []
        if isinstance(step, dict) and str(step.get("claim_id") or "")
    }
    available = {str(value) for value in available_riu_ids}
    rows = [
        row
        for row in judgment.get("claim_results") or []
        if isinstance(row, dict)
    ]
    returned_ids = [str(row.get("claim_id") or "") for row in rows]
    if (
        len(returned_ids) != len(set(returned_ids))
        or set(returned_ids) != expected_claim_ids
    ):
        errors.append("judge must grade every oracle claim exactly once")
    allowed_statuses = {"satisfied", "partial", "missing", "incorrect"}
    satisfied_count = 0
    for row in rows:
        claim_id = str(row.get("claim_id") or "")
        if row.get("status") not in allowed_statuses:
            errors.append(f"{claim_id}: invalid claim status")
        if not isinstance(row.get("content_correct"), bool):
            errors.append(f"{claim_id}: content_correct must be boolean")
        if not isinstance(row.get("grounded_in_available_evidence"), bool):
            errors.append(
                f"{claim_id}: grounded_in_available_evidence must be boolean"
            )
        cited = {str(value) for value in row.get("supporting_riu_ids") or []}
        if not cited <= available:
            errors.append(f"{claim_id}: judge cites unavailable RIUs")
        satisfied = (
            row.get("content_correct") is True
            and row.get("grounded_in_available_evidence") is True
        )
        if (row.get("status") == "satisfied") != satisfied:
            errors.append(
                f"{claim_id}: satisfied status must match correctness and grounding"
            )
        if satisfied:
            satisfied_count += 1
    expected_coverage = (
        satisfied_count / len(expected_claim_ids)
        if expected_claim_ids
        else 0.0
    )
    if _is_number(coverage) and abs(float(coverage) - expected_coverage) > 1e-6:
        errors.append("required_step_coverage does not match claim_results")
    should_pass = (
        bool(expected_claim_ids)
        and satisfied_count == len(expected_claim_ids)
        and judgment.get("final_answer_correct") is True
        and judgment.get("source_grounded") is True
        and not judgment.get("disqualifying_errors")
    )
    if judgment.get("passes") in (True, False) and bool(
        judgment.get("passes")
    ) != should_pass:
        errors.append("judge passes is inconsistent with claim-level results")
    return errors


def normalize_free_response_judgment(
    judgment: dict[str, Any],
    *,
    oracle: dict[str, Any],
) -> dict[str, Any]:
    """Compute judge summary fields from its atomic claim assessments."""
    expected_claim_ids = {
        str(step.get("claim_id") or "")
        for step in oracle.get("reasoning_steps") or []
        if isinstance(step, dict) and str(step.get("claim_id") or "")
    }
    satisfied_count = 0
    for row in judgment.get("claim_results") or []:
        if not isinstance(row, dict):
            continue
        correct = row.get("content_correct") is True
        grounded = row.get("grounded_in_available_evidence") is True
        if correct and grounded:
            row["status"] = "satisfied"
            satisfied_count += 1
        elif correct or grounded:
            row["status"] = "partial"
        elif row.get("status") not in {"missing", "incorrect"}:
            row["status"] = "missing"
    judgment["required_step_coverage"] = (
        satisfied_count / len(expected_claim_ids) if expected_claim_ids else 0.0
    )
    judgment["passes"] = (
        bool(expected_claim_ids)
        and satisfied_count == len(expected_claim_ids)
        and judgment.get("final_answer_correct") is True
        and judgment.get("source_grounded") is True
        and not judgment.get("disqualifying_errors")
    )
    return judgment


def validate_packet_riu_audit(
    audit: dict[str, Any],
    *,
    all_rui_ids: Iterable[str],
    assigned_or_shared_rui_ids: Iterable[str],
    packet_context_ids: Iterable[int] | None = None,
    expected_participant_id: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if (
        expected_participant_id is not None
        and str(audit.get("participant_id") or "") != expected_participant_id
    ):
        errors.append("packet audit participant_id does not match the input packet")
    all_ids = {str(value) for value in all_rui_ids}
    assigned_or_shared = {
        str(value) for value in assigned_or_shared_rui_ids
    }
    expected = all_ids - assigned_or_shared
    packet_ids = {
        value
        for value in (_as_int(item) for item in (packet_context_ids or []))
        if value is not None
    }
    rows = audit.get("unassigned_riu_results")
    rows = rows if isinstance(rows, list) else []
    by_rui = {
        str(row.get("riu_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("riu_id")
    }
    if len(by_rui) != len(rows):
        errors.append("packet audit contains duplicate or malformed RIU rows")
    if set(by_rui) != expected:
        errors.append("packet audit must cover every unassigned RIU exactly once")
    for riu_id, row in by_rui.items():
        if not isinstance(row.get("visible"), bool):
            errors.append(f"{riu_id}: visible must be boolean")
            continue
        context_ids = row.get("evidence_context_ids")
        if not isinstance(context_ids, list):
            errors.append(f"{riu_id}: evidence_context_ids must be a list")
            continue
        normalized_ids = [_as_int(value) for value in context_ids]
        if any(value is None for value in normalized_ids):
            errors.append(f"{riu_id}: evidence_context_ids must contain integers")
        if row.get("visible") is True and not normalized_ids and row.get("evidence_in_common") is not True:
            errors.append(f"{riu_id}: visible result requires evidence_context_ids")
        if row.get("visible") is False and normalized_ids:
            errors.append(f"{riu_id}: invisible result cannot cite evidence contexts")
        unknown = {
            value
            for value in normalized_ids
            if value is not None and packet_context_ids is not None and value not in packet_ids
        }
        if unknown:
            errors.append(
                f"{riu_id}: cited contexts are not in this packet: {sorted(unknown)}"
            )
        if not str(row.get("reason") or "").strip():
            errors.append(f"{riu_id}: reason is required")
    return _dedupe(errors)


def leaked_context_ids(
    audit: dict[str, Any],
) -> set[int]:
    out: set[int] = set()
    for row in audit.get("unassigned_riu_results") or []:
        if not isinstance(row, dict):
            continue
        if row.get("visible") is not True:
            continue
        for value in row.get("evidence_context_ids") or []:
            number = _as_int(value)
            if number is not None:
                out.add(number)
    return out


def _is_number(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool) or isinstance(value, float) and not value.is_integer():
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def public_answer_disclosure_errors(plan: dict[str, Any]) -> list[str]:
    """Catch explicit answer-with-unit disclosures without banning scope years.

    Semantic disclosures are checked independently by the model audit. This
    narrow deterministic check also prevents the observed cardinality leak from
    being accepted if an auditor overlooks it again.
    """
    public = " ".join(str(value) for key, value in (plan.get("public_task") or {}).items()
                      if key in {"title", "question", "response_format_note", "instruction"})
    spec = (plan.get("oracle") or {}).get("answer_spec") or {}
    items = spec.get("items") or [spec]
    words = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
    errors = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict) or item.get("type") != "numeric" or not _is_number(item.get("value")):
            continue
        value = float(item["value"])
        forms = {str(item["value"]), format(value, "g")}
        if value.is_integer():
            forms |= {str(int(value)), format(int(value), ",")}
            if int(value) in words:
                forms.add(words[int(value)])
        number = r"(?<![\w.])(?:" + "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True)) + r")(?![\w.])"
        unit = str(item.get("unit") or "").strip().lower()
        # Exact unit phrases catch count answers, not unrelated years or (4) item numbering.
        units = {unit}
        if unit.endswith('s'):
            units.add(unit[:-1])
        if unit in {'percent', 'percentage', 'percentage points'}:
            units |= {'%', 'percent', 'percentage points'}
        suffix = '(?:' + '|'.join(re.escape(u).replace(r"\ ", r"\s+") for u in units if u) + ')'
        disclosed = bool(unit and re.search(number + r"\s+" + suffix + r"\b", public, re.IGNORECASE))
        if unit in {'dollar', 'dollars', 'usd'}:
            disclosed |= bool(re.search(r"\$\s*" + number, public, re.IGNORECASE))
        if '%' in units:
            disclosed |= bool(re.search(number + r"\s*%", public, re.IGNORECASE))
        if disclosed:
            errors.append(f"public objective discloses numeric answer {item.get('item_id') or index} with its unit/cardinality")
    return errors


def validate_objective_disclosure_audit(audit: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    errors = public_answer_disclosure_errors(plan)
    if not isinstance(audit.get('valid'), bool) or not isinstance(audit.get('discloses_answer_information'), bool):
        errors.append('objective disclosure audit requires strict boolean decisions')
    findings = audit.get('findings')
    if not isinstance(findings, list):
        errors.append('objective disclosure audit findings must be a list')
        findings = []
    public = " ".join(str(value) for value in (plan.get("public_task") or {}).values())
    for finding in findings:
        if not isinstance(finding, dict):
            errors.append('objective disclosure finding must be an object')
            continue
        quote = finding.get('public_quote')
        if not isinstance(quote, str) or not quote.strip() or quote not in public:
            errors.append('objective disclosure finding must quote exact public text')
    if audit.get('valid') is not True or audit.get('discloses_answer_information') is not False or findings:
        errors.append('public objective discloses or may disclose answer information')
    return _dedupe(errors)
