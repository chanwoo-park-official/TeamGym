from __future__ import annotations

from teamgym.data.medical.evidencebench.pipeline import (
    boundary_case_outcome,
    build_disjoint_packets,
    display_materials,
    validate_boundary_judgment,
    validate_candidates,
    validate_task_rubric,
)


def _record() -> dict:
    return {
        "instance_id": "evidencebench_test_case",
        "paper_id": "paper_1",
        "hypothesis": "Exposure changes two measured outcomes.",
        "paper_as_candidate_pool": [
            "Background sentence.",
            "Animals received one dose of the exposure.",
            "Outcome A increased relative to controls.",
            "Outcome B did not increase.",
            "Unrelated discussion sentence.",
        ],
        "sentence_types_in_candidate_pool": ["abstract", "methods", "results", "results", "discussion"],
        "evidence_retrieval_at_optimal_evaluation": {
            "one_selection_of_sentences": [1, 2, 3]
        },
    }


def _candidates() -> list[dict]:
    return [
        {
            "candidate_id": "cand_01",
            "source_anchor_ids": ["anchor_01"],
            "hidden_contribution": "exposure condition",
            "anchor_preserved": True,
            "context_restoration_reason": "none",
            "evidence_atoms": [
                {"atom_id": "atom_01", "sentence_indices": [1], "text": "Animals received one dose of the exposure."}
            ],
        },
        {
            "candidate_id": "cand_02",
            "source_anchor_ids": ["anchor_02"],
            "hidden_contribution": "positive outcome",
            "anchor_preserved": True,
            "context_restoration_reason": "none",
            "evidence_atoms": [
                {"atom_id": "atom_02", "sentence_indices": [2], "text": "Outcome A increased relative to controls."}
            ],
        },
        {
            "candidate_id": "cand_03",
            "source_anchor_ids": ["anchor_03"],
            "hidden_contribution": "negative outcome",
            "anchor_preserved": True,
            "context_restoration_reason": "none",
            "evidence_atoms": [
                {"atom_id": "atom_03", "sentence_indices": [3], "text": "Outcome B did not increase."}
            ],
        },
    ]


def _rubric() -> dict:
    return {
        "required_claims": [
            {
                "claim_id": "claim_01",
                "criterion": "state that outcome A increased while outcome B did not",
                "support_logic": "ALL_REQUIRED",
                "supporting_candidate_ids": ["cand_02", "cand_03"],
                "candidate_roles": {
                    "cand_02": "supports the positive outcome",
                    "cand_03": "supports the negative outcome",
                },
            },
        ],
        "scope_constraints": [],
        "forbidden_overclaims": [],
        "disqualifying_errors": [],
    }


def test_candidate_normalization_treats_optimal_sentences_as_anchors() -> None:
    record = _record()
    anchors = [
        {
            "sentence_index": index,
            "adjacent_context": [
                {"sentence_index": neighbor, "text": record["paper_as_candidate_pool"][neighbor]}
                for neighbor in range(max(0, index - 1), min(5, index + 2))
            ],
        }
        for index in (1, 2, 3)
    ]
    value = {"valid": True, "candidate_rius": _candidates()}
    assert validate_candidates(value, record=record, anchors=anchors) == []


def test_atom_ids_may_repeat_across_candidates() -> None:
    record = _record()
    candidates = _candidates()
    for candidate in candidates:
        candidate["evidence_atoms"][0]["atom_id"] = "atom_01"
    anchors = [
        {
            "sentence_index": index,
            "adjacent_context": [{"sentence_index": index, "text": record["paper_as_candidate_pool"][index]}],
        }
        for index in (1, 2, 3)
    ]
    assert validate_candidates({"valid": True, "candidate_rius": candidates}, record=record, anchors=anchors) == []


def test_source_anchors_cannot_be_merged_during_normalization() -> None:
    record = _record()
    anchors = [
        {
            "anchor_id": f"anchor_{position:02d}",
            "sentence_index": index,
            "adjacent_context": [{"sentence_index": index, "text": record["paper_as_candidate_pool"][index]}],
        }
        for position, index in enumerate((1, 2, 3), start=1)
    ]
    candidates = _candidates()
    candidates[0]["source_anchor_ids"] = ["anchor_01", "anchor_02"]
    candidates[0]["evidence_atoms"].append(candidates[1]["evidence_atoms"][0])
    value = {"valid": True, "candidate_rius": [candidates[0], candidates[2]]}
    errors = " ".join(validate_candidates(value, record=record, anchors=anchors))
    assert "exactly one candidate per source anchor" in errors
    assert "exactly one source_anchor_id" in errors


def test_task_allows_multiple_candidates_to_jointly_support_one_claim() -> None:
    task = {
        "selected_candidate_ids": ["cand_02", "cand_03"],
        "candidate_dispositions": [
            {"candidate_id": "cand_01", "included": False, "mapped_claim_ids": [], "reason": "not asked"},
            {"candidate_id": "cand_02", "included": True, "mapped_claim_ids": ["claim_01"], "reason": "outcome A"},
            {"candidate_id": "cand_03", "included": True, "mapped_claim_ids": ["claim_01"], "reason": "outcome B"},
        ],
        "public_task": {"question": "How did the two measured outcomes change?"},
        "rubric": _rubric(),
        "canonical_answer": "Outcome A increased while outcome B did not.",
    }
    assert validate_task_rubric(task, candidates=_candidates()) == []
    task["rubric"]["required_claims"][0]["support_logic"] = "ANY_SUFFICIENT"
    assert "not individually necessary" in " ".join(validate_task_rubric(task, candidates=_candidates()))


def test_boundary_conditions_use_semantic_completeness_not_exact_set() -> None:
    rubric = _rubric()
    all_judgment = {
        "content_complete": True,
        "grounded_complete": True,
        "claim_results": [
            {"claim_id": "claim_01", "satisfied": True},
        ],
        "critical_errors": [],
    }
    assert boundary_case_outcome(kind="all", judgment=all_judgment, rubric=rubric)["condition_ok"]
    assert not boundary_case_outcome(kind="none", judgment=all_judgment, rubric=rubric)["condition_ok"]

    loo_judgment = {
        "content_complete": True,
        "grounded_complete": False,
        "claim_results": [
            {"claim_id": "claim_01", "satisfied": False},
        ],
        "critical_errors": [],
    }
    outcome = boundary_case_outcome(
        kind="loo",
        judgment=loo_judgment,
        rubric=rubric,
        omitted_candidate_id="cand_03",
    )
    assert outcome["condition_ok"]
    assert outcome["loo_answerable"]
    assert not outcome["loo_grounded_complete"]
    assert outcome["broken_omitted_claim_ids"] == ["claim_01"]


def test_boundary_grounding_enforces_all_required_support() -> None:
    judgment = {
        "content_complete": True,
        "grounded_complete": True,
        "claim_results": [
            {
                "claim_id": "claim_01",
                "content_correct": True,
                "grounded_in_available_material": True,
                "satisfied": True,
                "evidence_numbers": ["evidence_01"],
            }
        ],
        "unsupported_claims": [],
        "critical_errors": [],
    }
    errors = validate_boundary_judgment(
        judgment,
        rubric=_rubric(),
        available_numbers={"evidence_01"},
        candidate_by_number={"evidence_01": "cand_02"},
    )
    assert "does not satisfy support_logic" in " ".join(errors)


def test_solver_material_uses_only_neutral_display_numbers() -> None:
    rows = [
        {
            "candidate_id": "cand_01",
            "atom_id": "atom_01",
            "sentence_indices": [1],
            "text": "Exact paper evidence.",
        }
    ]
    display, mapping = display_materials(rows, seed=1)
    assert display == [
        {"evidence_number": "evidence_01", "material_type": "paper_excerpt", "text": "Exact paper evidence."}
    ]
    assert "candidate_id" not in display[0]
    assert mapping[0]["candidate_id"] == "cand_01"


def test_full_paper_partition_is_exhaustive_and_disjoint() -> None:
    final_rius = [
        {
            "riu_id": "riu_01",
            "candidate_id": "cand_02",
            "evidence_atoms": _candidates()[1]["evidence_atoms"],
        },
        {
            "riu_id": "riu_02",
            "candidate_id": "cand_03",
            "evidence_atoms": _candidates()[2]["evidence_atoms"],
        },
    ]
    partition = build_disjoint_packets(record=_record(), final_rius=final_rius, seed=9)
    assigned = [
        context_id
        for participant in partition["participants"]
        for context_id in participant["context_ids"]
    ]
    assert len(assigned) == 5
    assert len(set(assigned)) == 5
    assert set(assigned) == {row["context_id"] for row in partition["contexts"]}
