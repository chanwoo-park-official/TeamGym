"""Deterministic validation for EvidenceBench construction artifacts."""

from __future__ import annotations

from typing import Any


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def validate_candidates(
    value: Any,
    *,
    record: dict[str, Any],
    anchors: list[dict[str, Any]],
) -> list[str]:
    if not isinstance(value, dict):
        return ["candidate normalization must be an object"]
    errors: list[str] = []
    if value.get("valid") is not True:
        errors.append("candidate normalization did not return valid=true")
    candidates = value.get("candidate_rius")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return errors + ["at least two evidence candidates are required"]
    if len(candidates) != len(anchors):
        errors.append("normalization must return exactly one candidate per source anchor")
    candidate_ids = [str(row.get("candidate_id") or "") for row in candidates if isinstance(row, dict)]
    if len(candidate_ids) != len(candidates) or any(not item for item in candidate_ids):
        errors.append("every candidate requires a candidate_id")
    if len(set(candidate_ids)) != len(candidate_ids):
        errors.append("candidate IDs must be unique")

    paper = list(record["paper_as_candidate_pool"])
    anchor_indices = {int(row["sentence_index"]) for row in anchors}
    anchor_by_id = {
        str(row.get("anchor_id") or f"anchor_{position:02d}"): int(row["sentence_index"])
        for position, row in enumerate(anchors, start=1)
    }
    allowed_indices = {
        int(item["sentence_index"])
        for anchor in anchors
        for item in anchor["adjacent_context"]
    }
    neighborhood_by_id = {
        str(anchor.get("anchor_id") or f"anchor_{position:02d}"):
            {int(item["sentence_index"]) for item in anchor["adjacent_context"]}
        for position, anchor in enumerate(anchors, 1)
    }
    observed_indices: list[int] = []
    observed_anchor_ids: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            errors.append("candidate rows must be objects")
            continue
        if candidate.get("anchor_preserved") is not True:
            errors.append(f"{candidate.get('candidate_id')}: anchor_preserved must be true")
        if not str(candidate.get("hidden_contribution") or "").strip():
            errors.append(f"{candidate.get('candidate_id')}: hidden contribution is required")
        source_anchor_ids = string_list(candidate.get("source_anchor_ids"))
        if len(source_anchor_ids) != 1:
            errors.append(f"{candidate.get('candidate_id')}: exactly one source_anchor_id is required")
        if not set(source_anchor_ids) <= set(anchor_by_id):
            errors.append(f"{candidate.get('candidate_id')}: source_anchor_ids contain unknown anchors")
        observed_anchor_ids.extend(source_anchor_ids)
        if not str(candidate.get("context_restoration_reason") or "").strip():
            errors.append(f"{candidate.get('candidate_id')}: context_restoration_reason is required")
        atoms = candidate.get("evidence_atoms")
        if not isinstance(atoms, list) or not atoms:
            errors.append(f"{candidate.get('candidate_id')}: evidence_atoms are required")
            continue
        candidate_atom_ids: list[str] = []
        candidate_indices: set[int] = set()
        for atom in atoms:
            if not isinstance(atom, dict):
                errors.append("evidence atom must be an object")
                continue
            atom_id = str(atom.get("atom_id") or "")
            candidate_atom_ids.append(atom_id)
            indices = atom.get("sentence_indices")
            if not isinstance(indices, list) or not all(type(i) is int for i in indices):
                errors.append(f"{atom_id}: sentence indices must be an integer array")
                continue
            if not atom_id or not indices:
                errors.append(f"{candidate.get('candidate_id')}: atom id and sentence indices are required")
                continue
            if indices != sorted(set(indices)):
                errors.append(f"{atom_id}: sentence indices must be sorted and unique")
            if indices and indices != list(range(indices[0], indices[-1] + 1)):
                errors.append(f"{atom_id}: sentence indices must be contiguous")
            if not set(indices) <= allowed_indices:
                errors.append(f"{atom_id}: used a sentence outside supplied local neighborhoods")
            if any(index < 0 or index >= len(paper) for index in indices):
                errors.append(f"{atom_id}: sentence index is outside the paper")
                continue
            expected = " ".join(str(paper[index]).strip() for index in indices)
            if str(atom.get("text") or "") != expected:
                errors.append(f"{atom_id}: atom text is not exact verbatim source text")
            observed_indices.extend(indices)
            candidate_indices.update(indices)
        ordered = sorted(candidate_indices)
        if ordered and ordered != list(range(ordered[0], ordered[-1] + 1)):
            errors.append(f"{candidate.get('candidate_id')}: complete candidate span must be contiguous")
        own_neighborhood = set().union(*(neighborhood_by_id.get(i, set()) for i in source_anchor_ids))
        if not candidate_indices <= own_neighborhood:
            errors.append(f"{candidate.get('candidate_id')}: span exceeds its own anchor neighborhood")
        missing_assigned_anchors = {
            anchor_id
            for anchor_id in source_anchor_ids
            if anchor_by_id.get(anchor_id) not in candidate_indices
        }
        if missing_assigned_anchors:
            errors.append(
                f"{candidate.get('candidate_id')}: assigned anchors absent from evidence atoms: "
                f"{sorted(missing_assigned_anchors)}"
            )
        foreign_anchor_indices = anchor_indices - {
            anchor_by_id[anchor_id]
            for anchor_id in source_anchor_ids
            if anchor_id in anchor_by_id
        }
        if candidate_indices & foreign_anchor_indices:
            errors.append(f"{candidate.get('candidate_id')}: evidence atoms include a foreign optimal anchor")
        if any(not item for item in candidate_atom_ids) or len(set(candidate_atom_ids)) != len(candidate_atom_ids):
            errors.append(f"{candidate.get('candidate_id')}: atom IDs must be nonempty and unique within the candidate")
    if sorted(observed_anchor_ids) != sorted(anchor_by_id):
        errors.append("every source anchor must be assigned to exactly one candidate")
    if len(set(observed_indices)) != len(observed_indices):
        errors.append("evidence atoms overlap")
    if not anchor_indices <= set(observed_indices):
        errors.append(f"not every optimal anchor is covered: {sorted(anchor_indices - set(observed_indices))}")
    return dedupe(errors)


def validate_task_rubric(value: Any, *, candidates: list[dict[str, Any]]) -> list[str]:
    if not isinstance(value, dict):
        return ["task generation must be an object"]
    errors: list[str] = []
    available = {str(row["candidate_id"]) for row in candidates}
    selected = string_list(value.get("selected_candidate_ids"))
    if len(selected) < 2 or len(selected) != len(set(selected)):
        errors.append("select at least two unique evidence candidates")
    if not set(selected) <= available:
        errors.append(f"unknown selected candidates: {sorted(set(selected) - available)}")
    dispositions = value.get("candidate_dispositions")
    disposition_by_id = {
        str(row.get("candidate_id") or ""): row
        for row in dispositions or []
        if isinstance(row, dict)
    }
    if not isinstance(dispositions, list) or len(dispositions) != len(available) or set(disposition_by_id) != available:
        errors.append("candidate dispositions must cover every normalized candidate")
    else:
        disposition_selected = {
            candidate_id
            for candidate_id, row in disposition_by_id.items()
            if row.get("included") is True
        }
        if disposition_selected != set(selected):
            errors.append("candidate dispositions do not match selected_candidate_ids")
        if any(not isinstance(row.get("included"), bool) for row in disposition_by_id.values()):
            errors.append("every candidate disposition requires a boolean included flag")
        if any(not str(row.get("reason") or "").strip() for row in disposition_by_id.values()):
            errors.append("every candidate disposition requires a reason")
    task = value.get("public_task")
    question = str(task.get("question") or "") if isinstance(task, dict) else ""
    if not question.strip():
        errors.append("public question is required")
    forbidden_tokens = ["cand_", "riu_", "source_unit_", "evidencebench", "gold evidence"]
    lowered = question.casefold()
    if any(token in lowered for token in forbidden_tokens):
        errors.append("public question leaks construction metadata")
    rubric = value.get("rubric")
    claims = rubric.get("required_claims") if isinstance(rubric, dict) else None
    if not isinstance(claims, list) or not claims:
        return errors + ["required_claims are required"]
    claim_ids = [str(row.get("claim_id") or "") for row in claims if isinstance(row, dict)]
    if len(claim_ids) != len(claims) or any(not item for item in claim_ids):
        errors.append("every rubric claim requires a claim_id")
    if len(set(claim_ids)) != len(claim_ids):
        errors.append("claim IDs must be unique")
    structurally_necessary: set[str] = set()
    claim_id_set = set(claim_ids)
    for claim in claims:
        if not isinstance(claim, dict):
            errors.append("rubric claims must be objects")
            continue
        supporters = string_list(claim.get("supporting_candidate_ids"))
        if not supporters or len(supporters) != len(set(supporters)) or not set(supporters) <= set(selected):
            errors.append(f"{claim.get('claim_id')}: invalid supporting candidate IDs")
        support_logic = str(claim.get("support_logic") or "")
        if support_logic not in {"ALL_REQUIRED", "ANY_SUFFICIENT"}:
            errors.append(f"{claim.get('claim_id')}: invalid support_logic")
        if support_logic == "ALL_REQUIRED":
            structurally_necessary.update(supporters)
        elif support_logic == "ANY_SUFFICIENT" and len(supporters) == 1:
            structurally_necessary.update(supporters)
        roles = claim.get("candidate_roles")
        if not isinstance(roles, dict) or set(map(str, roles)) != set(supporters):
            errors.append(f"{claim.get('claim_id')}: candidate_roles must cover all supporting candidates")
        elif any(not str(roles[candidate_id]).strip() for candidate_id in supporters):
            errors.append(f"{claim.get('claim_id')}: candidate roles must be nonempty")
        if not str(claim.get("criterion") or "").strip():
            errors.append(f"{claim.get('claim_id')}: criterion is required")
    missing_necessary = set(selected) - structurally_necessary
    if missing_necessary:
        errors.append(
            "selected candidates are not individually necessary under the declared support logic: "
            f"{sorted(missing_necessary)}"
        )
    if isinstance(dispositions, list):
        for candidate_id, row in disposition_by_id.items():
            mapped = set(string_list(row.get("mapped_claim_ids")))
            if not mapped <= claim_id_set:
                errors.append(f"{candidate_id}: disposition maps unknown claim IDs")
            if candidate_id in set(selected) and not mapped:
                errors.append(f"{candidate_id}: selected candidate has no mapped claim")
            if candidate_id not in set(selected) and mapped:
                errors.append(f"{candidate_id}: excluded candidate cannot map required claims")
            actual = {
                str(claim.get("claim_id"))
                for claim in claims
                if candidate_id in string_list(claim.get("supporting_candidate_ids"))
            }
            if mapped != actual:
                errors.append(f"{candidate_id}: disposition claim mapping disagrees with rubric")
    if not str(value.get("canonical_answer") or "").strip():
        errors.append("canonical_answer is required")
    return dedupe(errors)


def validate_task_audit(value: Any, *, task_rubric: dict[str, Any]) -> list[str]:
    if not isinstance(value, dict):
        return ["task audit must be an object"]
    errors: list[str] = []
    for field in (
        "valid",
        "question_natural",
        "question_answerable_from_selected_material",
        "rubric_grounded",
        "canonical_answer_grounded",
        "selected_candidates_relevant",
        "question_not_artificially_expanded",
        "no_construction_metadata_leakage",
        "medical_scope_preserved",
    ):
        if value.get(field) is not True:
            errors.append(f"task audit did not confirm {field}")
    expected = {
        str(row["claim_id"])
        for row in (task_rubric.get("rubric") or {}).get("required_claims") or []
    }
    results = value.get("claim_results")
    observed = {
        str(row.get("claim_id") or "")
        for row in results or []
        if isinstance(row, dict)
    }
    if not isinstance(results, list) or len(results) != len(expected) or observed != expected:
        errors.append("task audit claim coverage mismatch")
    elif any(
        row.get("grounded") is not True or row.get("support_logic_valid") is not True
        for row in results
        if isinstance(row, dict)
    ):
        errors.append("at least one required claim or support mapping is invalid")
    else:
        claim_by_id = {
            str(row.get("claim_id") or ""): row
            for row in (task_rubric.get("rubric") or {}).get("required_claims") or []
            if isinstance(row, dict)
        }
        for row in results:
            claim_id = str(row.get("claim_id") or "")
            expected_supporters = set(string_list(claim_by_id.get(claim_id, {}).get("supporting_candidate_ids")))
            if set(string_list(row.get("supporting_candidate_ids"))) != expected_supporters:
                errors.append(f"{claim_id}: task audit supporting-candidate mismatch")
    selected = set(string_list(task_rubric.get("selected_candidate_ids")))
    dispositions = {
        str(row.get("candidate_id") or ""): row
        for row in task_rubric.get("candidate_dispositions") or []
        if isinstance(row, dict)
    }
    candidate_results = value.get("candidate_results")
    candidate_by_id = {
        str(row.get("candidate_id") or ""): row
        for row in candidate_results or []
        if isinstance(row, dict)
    }
    if not isinstance(candidate_results, list) or len(candidate_results) != len(dispositions) or set(candidate_by_id) != set(dispositions):
        errors.append("task audit normalized-candidate coverage mismatch")
    else:
        for candidate_id, row in candidate_by_id.items():
            is_selected = candidate_id in selected
            if row.get("included_as_selected") is not is_selected:
                errors.append(f"{candidate_id}: audit inclusion flag disagrees with task")
            if is_selected:
                for field in ("question_relevant", "direct_support", "necessary_under_support_logic"):
                    if row.get(field) is not True:
                        errors.append(f"{candidate_id}: task audit did not confirm {field}")
            elif row.get("exclusion_justified") is not True:
                errors.append(f"{candidate_id}: task audit did not justify exclusion")
            mapped = set(string_list(row.get("mapped_claim_ids")))
            expected_mapped = {
                str(claim.get("claim_id"))
                for claim in (task_rubric.get("rubric") or {}).get("required_claims") or []
                if candidate_id in string_list(claim.get("supporting_candidate_ids"))
            }
            if mapped != expected_mapped:
                errors.append(f"{candidate_id}: task audit claim mapping mismatch")
    if value.get("errors"):
        errors.append("task audit reported errors")
    return dedupe(errors)


def validate_solver(value: Any, *, available_numbers: set[str]) -> list[str]:
    if not isinstance(value, dict):
        return ["solver response must be an object"]
    errors: list[str] = []
    if not str(value.get("answer") or "").strip():
        errors.append("solver answer is required")
    used = value.get("evidence_used")
    if not isinstance(used, list):
        errors.append("evidence_used must be a list")
    else:
        cited = {
            str(row.get("evidence_number") or "")
            for row in used
            if isinstance(row, dict)
        }
        if not cited <= available_numbers:
            errors.append(f"solver cited unavailable evidence: {sorted(cited - available_numbers)}")
    if not isinstance(value.get("unresolved_requirements"), list):
        errors.append("unresolved_requirements must be a list")
    return errors


def validate_boundary_judgment(
    value: Any,
    *,
    rubric: dict[str, Any],
    available_numbers: set[str],
    candidate_by_number: dict[str, str],
    enforce_support_logic: bool = True,
) -> list[str]:
    if not isinstance(value, dict):
        return ["boundary judgment must be an object"]
    errors: list[str] = []
    if not isinstance(value.get("content_complete"), bool):
        errors.append("content_complete must be boolean")
    if not isinstance(value.get("grounded_complete"), bool):
        errors.append("grounded_complete must be boolean")
    claims = value.get("claim_results")
    expected = {str(row["claim_id"]) for row in rubric.get("required_claims") or []}
    observed = {
        str(row.get("claim_id") or "")
        for row in claims or []
        if isinstance(row, dict)
    }
    if not isinstance(claims, list) or len(claims) != len(expected) or observed != expected:
        return errors + ["boundary judge claim coverage mismatch"]
    for row in claims:
        if not isinstance(row, dict):
            errors.append("claim results must be objects")
            continue
        for field in ("content_correct", "grounded_in_available_material", "satisfied"):
            if not isinstance(row.get(field), bool):
                errors.append(f"{row.get('claim_id')}: {field} must be boolean")
        expected_satisfied = bool(row.get("content_correct") and row.get("grounded_in_available_material"))
        if row.get("satisfied") is not expected_satisfied:
            errors.append(f"{row.get('claim_id')}: satisfied must equal content_correct AND grounded")
        cited = set(string_list(row.get("evidence_numbers")))
        if not cited <= available_numbers:
            errors.append(f"{row.get('claim_id')}: judge cited unavailable evidence")
        claim = next(
            (
                item
                for item in rubric.get("required_claims") or []
                if str(item.get("claim_id") or "") == str(row.get("claim_id") or "")
            ),
            {},
        )
        supporters = set(string_list(claim.get("supporting_candidate_ids")))
        cited_candidates = {
            candidate_by_number[number]
            for number in cited
            if number in candidate_by_number
        }
        support_logic = str(claim.get("support_logic") or "")
        support_satisfied = (
            supporters <= cited_candidates
            if support_logic == "ALL_REQUIRED"
            else bool(supporters & cited_candidates)
        )
        if enforce_support_logic and row.get("grounded_in_available_material") is True and not support_satisfied:
            errors.append(f"{row.get('claim_id')}: cited evidence does not satisfy support_logic")
    if not isinstance(value.get("unsupported_claims"), list):
        errors.append("unsupported_claims must be a list")
    if not isinstance(value.get("critical_errors"), list):
        errors.append("critical_errors must be a list")
    if value.get("content_complete") is not all(row.get("content_correct") is True for row in claims):
        errors.append("content_complete contradicts claim results")
    if value.get("grounded_complete") is not all(row.get("satisfied") is True for row in claims):
        errors.append("grounded_complete contradicts claim results")
    return dedupe(errors)


def validate_leakage_audit(
    value: Any,
    *,
    participant_id: str,
    foreign_riu_ids: set[str],
    packet_ids: set[str],
    canonical_ids: set[str],
) -> list[str]:
    if not isinstance(value, dict):
        return ["leakage audit must be an object"]
    errors: list[str] = []
    if str(value.get("participant_id") or "") != participant_id:
        errors.append("leakage audit participant mismatch")
    rows = value.get("foreign_riu_results")
    observed = {
        str(row.get("riu_id") or "")
        for row in rows or []
        if isinstance(row, dict)
    }
    if not isinstance(rows, list) or len(rows) != len(foreign_riu_ids) or observed != foreign_riu_ids:
        return errors + ["leakage audit foreign RIU coverage mismatch"]
    for row in rows:
        if not isinstance(row, dict):
            errors.append("leakage result must be an object")
            continue
        for field in ("reconstructable_from_packet", "canonical_overlap"):
            if not isinstance(row.get(field), bool):
                errors.append(f"{row.get('riu_id')}: {field} must be boolean")
        support = set(string_list(row.get("leakage_support_context_ids")))
        removable = set(string_list(row.get("removable_noncanonical_context_ids")))
        if not support <= packet_ids or not removable <= packet_ids:
            errors.append(f"{row.get('riu_id')}: leakage audit cited an unknown context")
        if removable & canonical_ids:
            errors.append(f"{row.get('riu_id')}: canonical contexts cannot be removable")
        if not removable <= support:
            errors.append(f"{row.get('riu_id')}: removable contexts must support the reconstruction")
        if row.get("canonical_overlap") is True and row.get("reconstructable_from_packet") is not True:
            errors.append(f"{row.get('riu_id')}: canonical overlap implies reconstruction")
        if row.get("reconstructable_from_packet") is False and (support or removable):
            errors.append(f"{row.get('riu_id')}: no reconstruction cannot have leakage support or removals")
        if row.get("canonical_overlap") is True and not support <= canonical_ids:
            errors.append(f"{row.get('riu_id')}: canonical_overlap requires canonical-only support")
        if row.get("reconstructable_from_packet") is True and not support:
            errors.append(f"{row.get('riu_id')}: reconstructable leakage requires support contexts")
    if value.get("errors"):
        errors.append("leakage audit reported errors")
    return dedupe(errors)
