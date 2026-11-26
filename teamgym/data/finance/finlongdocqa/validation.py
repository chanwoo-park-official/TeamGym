"""Validation for the FinLongDocQA free-response pipeline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .source import restrict_table_rows


def mapping_block_ids(mapping: dict[str, Any]) -> list[str]:
    """Return active evidence block IDs with legacy mapping compatibility."""

    values = mapping.get("evidence_block_ids")
    if values is None:
        values = mapping.get("required_block_ids")
    return [str(value) for value in values or [] if str(value)]


def validate_qa_evidence_mapping(
    mapping: dict[str, Any],
    *,
    qa: dict[str, Any],
    available_blocks: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    qa_id = str(qa.get("qa_id") or "")
    if str(mapping.get("qa_id") or "") != qa_id:
        errors.append(f"mapping must reference QA {qa_id}")
    if mapping.get("answer_reconstructable") is not True:
        errors.append("QA answer is not reconstructable from its annotated pages")
    available_ids = {
        str(block.get("block_id") or "") for block in available_blocks
    }
    evidence_ids = mapping_block_ids(mapping)
    if not evidence_ids:
        errors.append("evidence_block_ids must not be empty")
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("evidence_block_ids contains duplicates")
    if not set(evidence_ids) <= available_ids:
        errors.append("evidence_block_ids references blocks outside the QA pages")
    contributions = [
        row
        for row in (
            mapping.get("block_contributions")
            if mapping.get("block_contributions") is not None
            else mapping.get("block_roles")
        )
        or []
        if isinstance(row, dict)
    ]
    contribution_ids = [
        str(row.get("block_id") or "") for row in contributions
    ]
    if len(contribution_ids) != len(set(contribution_ids)) or set(
        contribution_ids
    ) != set(evidence_ids):
        errors.append(
            "block_contributions must cover every evidence block exactly once"
        )
    allowed_roles = {
        "direct_answer",
        "calculation_input",
        "definition_or_formula",
        "entity_or_segment_scope",
        "period_or_unit_scope",
        "condition_or_exception",
    }
    for row in contributions:
        block_id = str(row.get("block_id") or "")
        if str(row.get("role") or "") not in allowed_roles:
            errors.append(f"{block_id}: unknown evidence role")
        contribution = row.get("contribution")
        if contribution is None:
            contribution = row.get("why_required")
        if not str(contribution or "").strip():
            errors.append(f"{block_id}: contribution is missing")
    return _dedupe(errors)


def validate_evidence_consolidation(
    result: dict[str, Any],
    *,
    report_id: str,
    candidate_blocks: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if str(result.get("report_id") or "") != report_id:
        errors.append(f"consolidation must reference report {report_id}")
    candidate_ids = {
        str(block.get("block_id") or "") for block in candidate_blocks
    }
    retained = [
        str(value) for value in result.get("retained_block_ids") or []
    ]
    excluded = [
        str(value) for value in result.get("excluded_block_ids") or []
    ]
    if len(retained) != len(set(retained)):
        errors.append("retained_block_ids contains duplicates")
    if len(excluded) != len(set(excluded)):
        errors.append("excluded_block_ids contains duplicates")
    if set(retained) & set(excluded):
        errors.append("retained and excluded block IDs must be disjoint")
    if set(retained) | set(excluded) != candidate_ids:
        errors.append(
            "retained and excluded block IDs must partition the candidate pool"
        )
    if len(retained) < 2:
        errors.append("consolidation must retain at least two blocks")

    allowed_relations = {
        "exact_duplicate",
        "information_contained",
        "derived_equivalent",
        "partial_overlap",
    }
    excluded_justified: set[str] = set()
    for index, group in enumerate(
        result.get("overlap_groups") or [],
        start=1,
    ):
        if not isinstance(group, dict):
            errors.append(f"overlap group {index} must be an object")
            continue
        providers = {
            str(value)
            for value in group.get("provider_block_ids") or []
            if str(value)
        }
        preferred = str(group.get("preferred_block_id") or "")
        relation = str(group.get("relation") or "")
        if len(providers) < 2 or not providers <= candidate_ids:
            errors.append(
                f"overlap group {index} must reference at least two candidate blocks"
            )
        if preferred not in providers or preferred not in set(retained):
            errors.append(
                f"overlap group {index} preferred block must be retained"
            )
        if relation not in allowed_relations:
            errors.append(f"overlap group {index} has an invalid relation")
        removable = {
            str(value)
            for value in group.get("fully_subsumed_block_ids") or []
            if str(value)
        }
        if not removable <= providers - {preferred}:
            errors.append(
                f"overlap group {index} has invalid fully_subsumed_block_ids"
            )
        excluded_justified.update(removable)
    if not set(excluded) <= excluded_justified:
        errors.append(
            "every excluded block must be justified as fully subsumed"
        )
    return _dedupe(errors)



def consolidated_qa_mappings(mappings: list[dict[str, Any]], consolidation: dict[str, Any]) -> list[dict[str, Any]]:
    """Carry validated provider substitutions into QA links, retaining ancestry."""
    excluded = set(consolidation["excluded_block_ids"])
    replacements = {}
    for group in consolidation.get("overlap_groups") or []:
        for old in group.get("fully_subsumed_block_ids") or []:
            if old in excluded:
                target = group["preferred_block_id"]
                if old in replacements and replacements[old]["replacement_block_id"] != target:
                    raise ValueError("ambiguous consolidated evidence provider")
                replacements[old] = {"original_block_id": old, "replacement_block_id": target,
                                     "relation": group["relation"]}
    result = deepcopy(mappings)
    for mapping in result:
        original = mapping_block_ids(mapping)
        mapping["original_evidence_block_ids"] = original
        mapping["evidence_block_ids"] = list(dict.fromkeys(replacements.get(block_id, {}).get("replacement_block_id", block_id)
                                                           for block_id in original))
        mapping["consolidation_replacements"] = [deepcopy(replacements[block_id]) for block_id in original if block_id in replacements]
        # Keep the original role provenance; downstream task selection uses the
        # effective IDs, not annotations tied to a block no longer in the pool.
        mapping["original_block_contributions"] = deepcopy(mapping.get("block_contributions") or mapping.get("block_roles") or [])
        roles = {}
        for role in mapping["original_block_contributions"]:
            old = str(role.get("block_id") or "")
            target = replacements.get(old, {}).get("replacement_block_id", old)
            role = {**role, "block_id": target}
            if target not in roles:
                roles[target] = role
            else:
                roles[target]["contribution"] = str(roles[target].get("contribution") or "") + "; " + str(role.get("contribution") or "")
        mapping["block_contributions"] = list(roles.values())
    return result


def hydrate_task(
    task: dict[str, Any],
    candidate_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replace model-provided evidence text with immutable complete blocks."""

    hydrated = deepcopy(task)
    by_id = {
        str(block.get("block_id") or ""): block for block in candidate_blocks
    }
    units: list[dict[str, Any]] = []
    proposed_to_hydrated: dict[str, str] = {}
    hydrated_by_block: dict[str, str] = {}
    for index, proposed in enumerate(task.get("evidence_units") or [], start=1):
        if not isinstance(proposed, dict):
            continue
        block_id = str(proposed.get("source_block_id") or "")
        proposed_id = str(proposed.get("unit_id") or f"riu_{index:02d}")
        if block_id in hydrated_by_block:
            proposed_to_hydrated[proposed_id] = hydrated_by_block[block_id]
            continue
        block = by_id.get(block_id) or {}
        hydrated_id = f"riu_{len(units) + 1:02d}"
        proposed_to_hydrated[proposed_id] = hydrated_id
        hydrated_by_block[block_id] = hydrated_id
        restriction = None
        if proposed.get("source_row_numbers") is not None and block:
            restriction = restrict_table_rows(block, proposed["source_row_numbers"])
        units.append(
            {
                "unit_id": hydrated_id,
                "source_block_id": block_id,
                "information_feature": str(
                    proposed.get("information_feature")
                    or block.get("heading")
                    or block.get("block_type")
                    or "report block"
                ),
                "hidden_summary": str(proposed.get("hidden_summary") or ""),
                "supports_claim_ids": [
                    str(value)
                    for value in proposed.get("supports_claim_ids") or []
                    if str(value)
                ],
                "text": restriction["text"] if restriction else str(block.get("text") or ""),
                **({"source_row_numbers": restriction["source_row_numbers"]} if restriction else {}),
                "source_ref": {
                    "source_block_id": block_id,
                    "page_number": block.get("page_number"),
                    "block_type": block.get("block_type"),
                    "heading": block.get("heading"),
                    "raw_start": block.get("raw_start"),
                    "raw_end": block.get("raw_end"),
                    **({k: restriction[k] for k in ("source_row_numbers", "source_row_spans", "original_block_sha256")} if restriction else {}),
                },
            }
        )
    hydrated["evidence_units"] = units
    rubric = hydrated.get("rubric")
    if isinstance(rubric, dict):
        for claim in rubric.get("required_claims") or []:
            if not isinstance(claim, dict):
                continue
            claim["supporting_unit_ids"] = list(
                dict.fromkeys(
                    proposed_to_hydrated.get(str(value), str(value))
                    for value in claim.get("supporting_unit_ids") or []
                )
            )
        synchronize_task_claim_links(hydrated)
    return hydrated


def synchronize_task_claim_links(task: dict[str, Any]) -> None:
    """Make unit-to-claim links the reciprocal of rubric claim-to-unit links."""

    claims = [
        row
        for row in (task.get("rubric") or {}).get("required_claims") or []
        if isinstance(row, dict)
    ]
    for unit in task.get("evidence_units") or []:
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("unit_id") or "")
        unit["supports_claim_ids"] = [
            str(claim.get("claim_id") or "")
            for claim in claims
            if unit_id
            in {
                str(value)
                for value in claim.get("supporting_unit_ids") or []
            }
        ]


def validate_generated_task(
    task: dict[str, Any],
    *,
    candidate_blocks: list[dict[str, Any]],
    qa_mappings: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if task.get("schema") != "finlongdocqa_free_response_task_v1":
        errors.append("task schema must be finlongdocqa_free_response_task_v1")

    available_qa_ids = {
        str(mapping.get("qa_id") or "") for mapping in qa_mappings
    }
    used_qa_ids = [str(value) for value in task.get("source_qa_ids_used") or []]
    if not used_qa_ids:
        errors.append("source_qa_ids_used must contain at least one source QA")
    if len(used_qa_ids) != len(set(used_qa_ids)):
        errors.append("source_qa_ids_used contains duplicates")
    if not set(used_qa_ids) <= available_qa_ids:
        errors.append("source_qa_ids_used references an unavailable source QA")

    public = task.get("public_task") or {}
    if not str(public.get("title") or "").strip():
        errors.append("public_task.title is required")
    if not str(public.get("question") or "").strip():
        errors.append("public_task.question is required")
    requirements = [
        str(value).strip()
        for value in public.get("response_requirements") or []
        if str(value).strip()
    ]
    if not requirements:
        errors.append("public_task.response_requirements must not be empty")
    if public.get("decision_set") is not None:
        errors.append("a free-response task must not contain decision_set")

    pool = {
        str(block.get("block_id") or ""): block
        for block in candidate_blocks
    }
    units = [
        row for row in task.get("evidence_units") or [] if isinstance(row, dict)
    ]
    unit_ids = [str(row.get("unit_id") or "") for row in units]
    block_ids = [str(row.get("source_block_id") or "") for row in units]
    if len(units) < 2:
        errors.append("the task must use at least two evidence units")
    if any(not value for value in unit_ids) or len(unit_ids) != len(set(unit_ids)):
        errors.append("evidence unit IDs must be present and unique")
    if any(not value for value in block_ids) or len(block_ids) != len(
        set(block_ids)
    ):
        errors.append("selected source block IDs must be present and unique")
    if not set(block_ids) <= set(pool):
        errors.append("evidence units must select only from the mapped block pool")
    mapped_blocks_by_qa = {
        str(mapping.get("qa_id") or ""): set(mapping_block_ids(mapping))
        for mapping in qa_mappings
    }
    blocks_for_used_qas = set().union(
        *(mapped_blocks_by_qa.get(qa_id, set()) for qa_id in used_qa_ids)
    )
    if not set(block_ids) <= blocks_for_used_qas:
        errors.append(
            "each selected block must belong to at least one used source QA"
        )
    for qa_id in used_qa_ids:
        if not set(block_ids) & mapped_blocks_by_qa.get(qa_id, set()):
            errors.append(
                f"{qa_id}: every used source QA must contribute a selected block"
            )
    for unit in units:
        block_id = str(unit.get("source_block_id") or "")
        if block_id not in pool:
            continue
        block = pool[block_id]
        expected_text = str(block.get("text") or "")
        if unit.get("source_row_numbers") is not None:
            try:
                restriction = restrict_table_rows(block, unit["source_row_numbers"])
                expected_text = restriction["text"]
                if any((unit.get("source_ref") or {}).get(k) != restriction[k]
                       for k in ("source_row_numbers", "source_row_spans", "original_block_sha256")):
                    errors.append(f"{block_id}: restricted rows have invalid source provenance")
            except (KeyError, ValueError, TypeError) as exc:
                errors.append(f"{block_id}: invalid table row restriction: {exc}")
        if str(unit.get("text") or "") != expected_text:
            errors.append(f"{block_id}: evidence text must equal the complete source block or exact selected table rows")

    rubric = task.get("rubric") or {}
    if rubric.get("rubric_version") != "finlongdocqa_claim_rubric_v1":
        errors.append("rubric_version must be finlongdocqa_claim_rubric_v1")
    claims = [
        row
        for row in rubric.get("required_claims") or []
        if isinstance(row, dict)
    ]
    if not claims:
        errors.append("rubric.required_claims must not be empty")
    claim_ids = [str(row.get("claim_id") or "") for row in claims]
    claim_id_set = set(claim_ids)
    if any(not value for value in claim_ids) or len(claim_ids) != len(
        claim_id_set
    ):
        errors.append("rubric claim IDs must be present and unique")

    unit_id_set = set(unit_ids)
    claim_support: dict[str, set[str]] = {}
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        if not str(claim.get("description") or "").strip():
            errors.append(f"{claim_id}: description is required")
        if not str(claim.get("expected_answer_element") or "").strip():
            errors.append(f"{claim_id}: expected_answer_element is required")
        if not str(claim.get("grading_rule") or "").strip():
            errors.append(f"{claim_id}: grading_rule is required")
        supporting = {
            str(value)
            for value in claim.get("supporting_unit_ids") or []
            if str(value)
        }
        if not supporting:
            errors.append(f"{claim_id}: supporting_unit_ids must not be empty")
        if not supporting <= unit_id_set:
            errors.append(f"{claim_id}: supporting_unit_ids contains unknown units")
        claim_support[claim_id] = supporting

    for unit in units:
        unit_id = str(unit.get("unit_id") or "")
        linked_claims = {
            str(value)
            for value in unit.get("supports_claim_ids") or []
            if str(value)
        }
        if not linked_claims:
            errors.append(f"{unit_id}: supports_claim_ids must not be empty")
        if not linked_claims <= claim_id_set:
            errors.append(f"{unit_id}: supports_claim_ids contains unknown claims")
        expected = {
            claim_id
            for claim_id, supporting in claim_support.items()
            if unit_id in supporting
        }
        if linked_claims != expected:
            errors.append(
                f"{unit_id}: unit-to-claim links must agree with the rubric"
            )

    oracle = task.get("oracle") or {}
    if not str(oracle.get("reference_answer") or "").strip():
        errors.append("oracle.reference_answer is required")
    return _dedupe(errors)


def solver_evidence_view(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"evidence_number": index, "text": str(unit.get("text") or "")}
        for index, unit in enumerate(units, start=1)
    ]


def validate_judge_result(
    result: dict[str, Any],
    *,
    claim_ids: set[str],
    available_evidence_count: int,
) -> list[str]:
    errors: list[str] = []
    rows = [
        row
        for row in result.get("claim_results") or []
        if isinstance(row, dict)
    ]
    returned_ids = [str(row.get("claim_id") or "") for row in rows]
    if len(returned_ids) != len(set(returned_ids)) or set(
        returned_ids
    ) != claim_ids:
        errors.append("judge must grade every rubric claim exactly once")
    allowed_statuses = {"satisfied", "partial", "missing", "incorrect"}
    valid_numbers = set(range(1, available_evidence_count + 1))
    for row in rows:
        claim_id = str(row.get("claim_id") or "")
        if row.get("status") not in allowed_statuses:
            errors.append(f"{claim_id}: invalid judge status")
        if type(row.get("content_correct")) is not bool:
            errors.append(f"{claim_id}: content_correct must be boolean")
        if type(row.get("grounded_in_available_material")) is not bool:
            errors.append(
                f"{claim_id}: grounded_in_available_material must be boolean"
            )
        evidence_numbers = {
            int(value)
            for value in row.get("supporting_evidence_numbers") or []
            if isinstance(value, int) or str(value).isdigit()
        }
        if not evidence_numbers <= valid_numbers:
            errors.append(f"{claim_id}: judge cites unavailable evidence")
        satisfied = (
            row.get("content_correct") is True
            and row.get("grounded_in_available_material") is True
        )
        if (row.get("status") == "satisfied") != satisfied:
            errors.append(
                f"{claim_id}: satisfied status must match correctness and grounding"
            )
    if type(result.get("has_material_error")) is not bool:
        errors.append("judge.has_material_error must be boolean")
    if type(result.get("overall_complete_and_correct")) is not bool:
        errors.append("judge.overall_complete_and_correct must be boolean")
    expected_complete = bool(rows) and all(_claim_satisfied(row) for row in rows) and result.get("has_material_error") is False
    if type(result.get("overall_complete_and_correct")) is bool and result["overall_complete_and_correct"] != expected_complete:
        errors.append("judge overall_complete_and_correct disagrees with claim results or material error")
    return _dedupe(errors)


def summarize_free_response_audit(
    *,
    task: dict[str, Any],
    full_result: dict[str, Any],
    no_evidence_result: dict[str, Any],
    leave_one_out_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    claims = [
        row
        for row in (task.get("rubric") or {}).get("required_claims") or []
        if isinstance(row, dict)
    ]
    claim_ids = {str(row.get("claim_id") or "") for row in claims}
    claim_support = {
        str(row.get("claim_id") or ""): {
            str(value) for value in row.get("supporting_unit_ids") or []
        }
        for row in claims
    }
    unit_ids = {
        str(row.get("unit_id") or "")
        for row in task.get("evidence_units") or []
        if isinstance(row, dict)
    }
    errors: list[str] = []

    full_judge = full_result.get("judge_result") or {}
    errors.extend(
        f"full: {error}"
        for error in validate_judge_result(
            full_judge,
            claim_ids=claim_ids,
            available_evidence_count=len(unit_ids),
        )
    )
    if (
        full_judge.get("overall_complete_and_correct") is not True
        or full_judge.get("has_material_error") is True
        or not all(
            _claim_satisfied(row)
            for row in full_judge.get("claim_results") or []
        )
    ):
        errors.append("full evidence response did not satisfy every rubric claim")

    none_judge = no_evidence_result.get("judge_result") or {}
    errors.extend(
        f"none: {error}"
        for error in validate_judge_result(
            none_judge,
            claim_ids=claim_ids,
            available_evidence_count=0,
        )
    )
    if none_judge.get("overall_complete_and_correct") is True:
        errors.append("no-evidence response was judged complete and correct")
    if any(
        _claim_satisfied(row)
        for row in none_judge.get("claim_results") or []
    ):
        errors.append("no-evidence response grounded a report-specific claim")

    normalized = []
    seen: list[str] = []
    for row in leave_one_out_results:
        omitted = str(row.get("omitted_unit_id") or "")
        seen.append(omitted)
        judge = row.get("judge_result") or {}
        errors.extend(
            f"{omitted}: {error}"
            for error in validate_judge_result(
                judge,
                claim_ids=claim_ids,
                available_evidence_count=max(0, len(unit_ids) - 1),
            )
        )
        if judge.get("overall_complete_and_correct") is True:
            errors.append(f"{omitted}: leave-one-out response remained complete")
        linked_claims = {
            claim_id
            for claim_id, supporting in claim_support.items()
            if omitted in supporting
        }
        judged_by_id = {
            str(result.get("claim_id") or ""): result
            for result in judge.get("claim_results") or []
            if isinstance(result, dict)
        }
        broken_claims = sorted(
            claim_id
            for claim_id in linked_claims
            if not _claim_satisfied(judged_by_id.get(claim_id) or {})
        )
        if not linked_claims:
            errors.append(f"{omitted}: unit is not linked to a rubric claim")
        elif not broken_claims:
            errors.append(
                f"{omitted}: all linked claims remained correct and grounded"
            )
        normalized.append(
            {
                **row,
                "linked_claim_ids": sorted(linked_claims),
                "broken_claim_ids": broken_claims,
            }
        )
    if len(seen) != len(set(seen)) or set(seen) != unit_ids:
        errors.append("leave-one-out audit must cover every evidence unit once")
    return (
        {
            "schema": "finlongdocqa_free_response_audit_v1",
            "accept": not errors,
            "full_evidence": full_result,
            "no_evidence": no_evidence_result,
            "leave_one_out": normalized,
        },
        _dedupe(errors),
    )


def validate_packet_leakage_audit(
    audit: dict[str, Any],
    *,
    participant_id: str,
    packet_context_unit_ids: set[str],
    canonical_context_unit_ids: set[str],
    foreign_rui_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    if str(audit.get("participant_id") or "") != participant_id:
        errors.append(f"packet audit must reference participant {participant_id}")
    rows = [
        row
        for row in audit.get("foreign_rui_results") or []
        if isinstance(row, dict)
    ]
    row_ids = [str(row.get("rui_id") or "") for row in rows]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != foreign_rui_ids:
        errors.append("packet audit must cover every foreign RIU exactly once")
    for row in rows:
        rui_id = str(row.get("rui_id") or "")
        reconstructable = row.get("reconstructable_from_packet")
        if type(reconstructable) is not bool:
            errors.append(
                f"{rui_id}: reconstructable_from_packet must be boolean"
            )
        support_ids = [
            str(value)
            for value in row.get("minimal_support_context_ids") or []
        ]
        removable_ids = [
            str(value)
            for value in row.get("removable_noncanonical_context_ids") or []
        ]
        if len(support_ids) != len(set(support_ids)):
            errors.append(f"{rui_id}: support context IDs contain duplicates")
        if len(removable_ids) != len(set(removable_ids)):
            errors.append(f"{rui_id}: removable context IDs contain duplicates")
        if not set(support_ids) <= packet_context_unit_ids:
            errors.append(f"{rui_id}: support references context outside packet")
        if not set(removable_ids) <= set(support_ids):
            errors.append(
                f"{rui_id}: removable contexts must be part of minimal support"
            )
        if set(removable_ids) & canonical_context_unit_ids:
            errors.append(
                f"{rui_id}: removable contexts must be non-canonical"
            )
        extra_routes = row.get("additional_reconstruction_routes", [])
        if not isinstance(extra_routes, list):
            errors.append(f"{rui_id}: additional_reconstruction_routes must be an array")
            extra_routes = []
        if reconstructable is False and extra_routes:
            errors.append(f"{rui_id}: non-reconstructable result cannot have additional routes")
        seen_routes = {tuple(sorted(support_ids))}
        for route in extra_routes:
            if not isinstance(route, dict) or set(route) != {"minimal_support_context_ids", "removable_noncanonical_context_ids", "canonical_overlap", "reason"}:
                errors.append(f"{rui_id}: invalid additional reconstruction route")
                continue
            route_key = tuple(sorted(str(value) for value in route.get("minimal_support_context_ids") or []))
            if route_key in seen_routes:
                errors.append(f"{rui_id}: duplicate reconstruction route")
            seen_routes.add(route_key)
            errors.extend(validate_packet_leakage_audit(
                {"participant_id": participant_id, "foreign_rui_results": [{"rui_id": rui_id,
                    "reconstructable_from_packet": True, **route}]}, participant_id=participant_id,
                packet_context_unit_ids=packet_context_unit_ids,
                canonical_context_unit_ids=canonical_context_unit_ids, foreign_rui_ids={rui_id}))
        canonical_overlap = row.get("canonical_overlap")
        if type(canonical_overlap) is not bool:
            errors.append(f"{rui_id}: canonical_overlap must be boolean")
        if reconstructable is True and not support_ids:
            errors.append(
                f"{rui_id}: reconstructable result must identify support contexts"
            )
        if reconstructable is True and not removable_ids and canonical_overlap is not True:
            errors.append(
                f"{rui_id}: reconstructable result needs removable context or canonical overlap"
            )
        if reconstructable is False and (
            support_ids or removable_ids or canonical_overlap is True
        ):
            errors.append(
                f"{rui_id}: non-reconstructable result must not identify leakage"
            )
    return _dedupe(errors)


def validate_packet_leakage_candidates(
    result: dict[str, Any],
    *,
    participant_id: str,
    chunk_context_unit_ids: set[str],
    foreign_rui_ids: set[str],
    max_candidates_per_rui: int = 12,
) -> list[str]:
    errors: list[str] = []
    if str(result.get("participant_id") or "") != participant_id:
        errors.append(
            f"candidate scan must reference participant {participant_id}"
        )
    rows = [
        row
        for row in result.get("foreign_rui_candidates") or []
        if isinstance(row, dict)
    ]
    row_ids = [str(row.get("rui_id") or "") for row in rows]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != foreign_rui_ids:
        errors.append(
            "candidate scan must cover every foreign RIU exactly once"
        )
    for row in rows:
        rui_id = str(row.get("rui_id") or "")
        candidates = [
            str(value)
            for value in row.get("candidate_context_unit_ids") or []
        ]
        if len(candidates) != len(set(candidates)):
            errors.append(f"{rui_id}: candidate context IDs contain duplicates")
        if len(candidates) > max_candidates_per_rui:
            errors.append(
                f"{rui_id}: candidate scan returned more than "
                f"{max_candidates_per_rui} contexts"
            )
        if not set(candidates) <= chunk_context_unit_ids:
            errors.append(
                f"{rui_id}: candidate scan references context outside chunk"
            )
    return _dedupe(errors)


def _claim_satisfied(row: dict[str, Any]) -> bool:
    return (
        row.get("status") == "satisfied"
        and row.get("content_correct") is True
        and row.get("grounded_in_available_material") is True
    )


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))
