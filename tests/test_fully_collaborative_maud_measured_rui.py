from __future__ import annotations

import random

from teamgym.data.legal.maud_measured_rui_funnel import (
    _normalize_decision_set,
    _normalize_plan_against_selection,
    _trim_to_smallest_sufficient_set,
    _validate_audit,
    _validate_plan,
    _validate_selection,
)
from teamgym.data.legal.maud_document_partition import (
    MAUDRUIRange,
    build_full_document_partition,
    _drop_overlapping_rui_ranges,
    _matched_rui_ranges_for_trial,
)


def _candidate() -> dict:
    return {
        "candidate_rui_units": [
            {
                "source_rui_id": "source_rui_001",
                "issue_group_id": "issue_group_0001",
                "text": "The termination fee tail period lasts twelve months after termination.",
            },
            {
                "source_rui_id": "source_rui_002",
                "issue_group_id": "issue_group_0002",
                "text": "The acquisition proposal must have been publicly disclosed before termination.",
            },
        ]
    }


def _selection() -> dict:
    return {
        "candidate_assessment": {"feasible": True, "rationale": "Two independent trigger features."},
        "source_rui_classification": [
            {"source_rui_id": "source_rui_001", "role": "used_for_rui", "reason": "Tail period."},
            {"source_rui_id": "source_rui_002", "role": "used_for_rui", "reason": "Disclosure condition."},
        ],
        "selected_rui_basis": [
            {
                "unit_id": "rui_01",
                "source_rui_id": "source_rui_001",
                "issue_group_id": "issue_group_0001",
                "text": "The termination fee tail period lasts twelve months after termination.",
                "legal_feature": "tail_period_length",
                "possible_decision_boundary": "distinguishes twelve months from a longer or shorter tail",
            },
            {
                "unit_id": "rui_02",
                "source_rui_id": "source_rui_002",
                "issue_group_id": "issue_group_0002",
                "text": "The acquisition proposal must have been publicly disclosed before termination.",
                "legal_feature": "public_disclosure_condition",
                "possible_decision_boundary": "distinguishes public-disclosure trigger from no disclosure requirement",
            },
        ],
    }


def test_maud_measured_rui_selection_validation_accepts_basis() -> None:
    assert _validate_selection(_selection(), _candidate(), min_accept_size=2, max_rui_count=4) == []
    assert _validate_selection(_selection(), _candidate(), min_accept_size=2, max_rui_count=None) == []


def test_maud_measured_rui_plan_and_audit_accept_all_required_units() -> None:
    selection = _selection()
    plan = _normalize_decision_set(
        {
            "candidate_assessment": {
                "selected_rui_ids": ["rui_01", "rui_02"],
                "coverage_rationale": "Together they define timing and disclosure.",
                "question_design_rationale": "Each RIU controls a separate profile component.",
            },
            "public_task": {
                "objective": "Which termination-fee tail-trigger profile is supported?",
                "decision_set": [
                    {
                        "value": "profile_alpha",
                        "definition": "A twelve-month tail applies and prior public disclosure is required.",
                    },
                    {
                        "value": "profile_beta",
                        "definition": "A twelve-month tail applies but prior public disclosure is not required.",
                    },
                    {
                        "value": "profile_gamma",
                        "definition": "A fifteen-month tail applies and prior public disclosure is required.",
                    },
                ],
            },
            "evidence_units": [
                {
                    "unit_id": "rui_01",
                    "text": "The termination fee tail period lasts twelve months after termination.",
                    "source_ref": {"source_rui_id": "source_rui_001", "issue_group_id": "issue_group_0001"},
                    "information_unit_type": "reference_information_unit",
                    "criticality": "critical",
                    "decision_relevance": [
                        {
                            "decision": "profile_alpha",
                            "relation": "required_for",
                            "reason": "The profile requires the twelve-month tail.",
                        }
                    ],
                },
                {
                    "unit_id": "rui_02",
                    "text": "The acquisition proposal must have been publicly disclosed before termination.",
                    "source_ref": {"source_rui_id": "source_rui_002", "issue_group_id": "issue_group_0002"},
                    "information_unit_type": "reference_information_unit",
                    "criticality": "critical",
                    "decision_relevance": [
                        {
                            "decision": "profile_alpha",
                            "relation": "required_for",
                            "reason": "The profile requires prior public disclosure.",
                        }
                    ],
                },
            ],
            "participants": [
                {"participant_id": "p1", "evidence_unit_ids": ["rui_01"]},
                {"participant_id": "p2", "evidence_unit_ids": ["rui_02"]},
            ],
            "oracle": {"answer": "profile_alpha", "rationale": "Both features are supported."},
        }
    )
    plan = _normalize_plan_against_selection(plan, selection)

    assert plan["public_task"]["decision_set"] == ["decision_a", "decision_b", "decision_c"]
    assert plan["oracle"]["answer"] == "decision_a"
    assert _validate_plan(plan, selection) == []

    audit = {
        "valid": True,
        "accept_under_threshold": True,
        "copied_from_maud_rui_blocks": True,
        "full_rui_set_sufficient": True,
        "minimum_sufficient_rui_set_size": 2,
        "smallest_sufficient_rui_sets": [
            {
                "unit_ids": ["rui_01", "rui_02"],
                "answer": "decision_a",
                "why_sufficient": "Both timing and disclosure are required.",
                "why_minimal": "Each singleton misses one boundary.",
            }
        ],
        "single_unit_shortcut": False,
        "single_participant_shortcut": False,
        "leave_one_out_results": [
            {
                "omitted_unit_id": "rui_01",
                "remaining_unit_ids": ["rui_02"],
                "can_determine_oracle_without_omitted": False,
            },
            {
                "omitted_unit_id": "rui_02",
                "remaining_unit_ids": ["rui_01"],
                "can_determine_oracle_without_omitted": False,
            },
        ],
        "non_required_unit_ids": [],
    }
    assert _validate_audit(audit, plan, min_accept_size=2) == []


def test_maud_measured_rui_plan_rejects_supporting_unit() -> None:
    selection = _selection()
    plan = {
        "public_task": {
            "objective": "Which profile applies?",
            "decision_set": ["decision_a", "decision_b"],
            "decision_option_definitions": {
                "decision_a": "A twelve-month tail applies and prior public disclosure is required.",
                "decision_b": "A twelve-month tail applies but prior public disclosure is not required.",
            },
        },
        "evidence_units": [
            {
                "unit_id": "rui_01",
                "text": "The termination fee tail period lasts twelve months after termination.",
                "source_ref": {"source_rui_id": "source_rui_001", "issue_group_id": "issue_group_0001"},
                "information_unit_type": "reference_information_unit",
                "criticality": "critical",
                "decision_relevance": [
                    {"decision": "decision_a", "relation": "required_for", "reason": "Needed."}
                ],
            },
            {
                "unit_id": "rui_02",
                "text": "The acquisition proposal must have been publicly disclosed before termination.",
                "source_ref": {"source_rui_id": "source_rui_002", "issue_group_id": "issue_group_0002"},
                "information_unit_type": "reference_information_unit",
                "criticality": "supporting",
                "decision_relevance": [
                    {"decision": "decision_a", "relation": "required_for", "reason": "Needed."}
                ],
            },
        ],
        "participants": [
            {"participant_id": "p1", "evidence_unit_ids": ["rui_01"]},
            {"participant_id": "p2", "evidence_unit_ids": ["rui_02"]},
        ],
        "oracle": {"answer": "decision_a"},
    }

    errors = _validate_plan(plan, selection)

    assert "rui_02: criticality must be critical." in errors


def test_maud_measured_rui_trim_keeps_smallest_sufficient_set() -> None:
    selection = {
        "candidate_assessment": {"feasible": True},
        "source_rui_classification": [],
        "selected_rui_basis": [
            {"unit_id": "rui_01", "source_rui_id": "source_rui_001", "issue_group_id": "issue_group_0001", "text": "A"},
            {"unit_id": "rui_02", "source_rui_id": "source_rui_002", "issue_group_id": "issue_group_0002", "text": "B"},
            {"unit_id": "rui_03", "source_rui_id": "source_rui_003", "issue_group_id": "issue_group_0003", "text": "C"},
        ],
    }
    plan = {
        "evidence_units": [
            {"unit_id": "rui_01", "text": "A"},
            {"unit_id": "rui_02", "text": "B"},
            {"unit_id": "rui_03", "text": "C"},
        ],
        "participants": [
            {"participant_id": "participant_1", "evidence_unit_ids": ["rui_01", "rui_03"]},
            {"participant_id": "participant_2", "evidence_unit_ids": ["rui_02"]},
        ],
    }
    audit = {
        "minimum_sufficient_rui_set_size": 2,
        "smallest_sufficient_rui_sets": [{"unit_ids": ["rui_01", "rui_02"], "answer": "decision_a"}],
    }

    trimmed_selection, trimmed_plan, metadata = _trim_to_smallest_sufficient_set(selection, plan, audit)

    assert metadata["trimmed"] is True
    assert metadata["removed_unit_ids"] == ["rui_03"]
    assert [row["unit_id"] for row in trimmed_selection["selected_rui_basis"]] == ["rui_01", "rui_02"]
    assert [row["unit_id"] for row in trimmed_plan["evidence_units"]] == ["rui_01", "rui_02"]
    assert trimmed_plan["participants"] == [
        {"participant_id": "participant_1", "evidence_unit_ids": ["rui_01"]},
        {"participant_id": "participant_2", "evidence_unit_ids": ["rui_02"]},
    ]


def test_maud_document_partition_drops_unmatched_rius() -> None:
    contract = "Section 1. Buyer consent shall not be unreasonably withheld. Section 2. Reasonable best efforts apply."
    trial = {
        "plan": {
            "evidence_units": [
                {
                    "unit_id": "rui_01",
                    "text": "Buyer consent shall not be unreasonably withheld.",
                    "source_ref": {"source_rui_id": "source_rui_001", "issue_group_id": "issue_group_0001"},
                },
                {
                    "unit_id": "rui_02",
                    "text": "This sentence is not present in the agreement.",
                    "source_ref": {"source_rui_id": "source_rui_002", "issue_group_id": "issue_group_0002"},
                },
            ]
        }
    }

    matched, dropped = _matched_rui_ranges_for_trial(trial, contract)

    assert [rui.unit_id for rui in matched] == ["rui_01"]
    assert [row["unit_id"] for row in dropped] == ["rui_02"]
    assert dropped[0]["reason"] == "no_full_agreement_offsets"


def test_maud_document_partition_drops_broader_overlapping_rui() -> None:
    ranges = [
        MAUDRUIRange(
            unit_id="rui_01",
            source_rui_id="source_rui_001",
            issue_group_id="issue_group_0001",
            clean_text="A long clause that contains the short clause.",
            raw_start=0,
            raw_end=100,
            raw_text="",
            legal_feature="long",
            summary="",
            section_key="",
            match_status="fuzzy_medium",
            match_score=0.9,
        ),
        MAUDRUIRange(
            unit_id="rui_02",
            source_rui_id="source_rui_002",
            issue_group_id="issue_group_0002",
            clean_text="short clause",
            raw_start=40,
            raw_end=60,
            raw_text="",
            legal_feature="short",
            summary="",
            section_key="",
            match_status="exact_raw",
            match_score=1.0,
        ),
    ]

    kept, dropped = _drop_overlapping_rui_ranges(ranges)

    assert [rui.unit_id for rui in kept] == ["rui_02"]
    assert [row["unit_id"] for row in dropped] == ["rui_01"]
    assert dropped[0]["reason"] == "overlaps_grounded_rui"


def test_maud_budgeted_document_partition_keeps_rius_without_full_coverage() -> None:
    rui_1 = "The termination fee tail period lasts twelve months after termination."
    rui_2 = "The acquisition proposal must have been publicly disclosed before termination."
    filler_a = " ".join(["Background covenant language appears here."] * 80)
    filler_b = " ".join(["Unrelated merger agreement boilerplate appears here."] * 80)
    filler_c = " ".join(["Additional tax and notice provisions appear here."] * 80)
    contract = "\n\n".join(
        [
            f"SECTION 1. Background\n{filler_a}",
            f"SECTION 2. Tail Period\n{rui_1} {filler_b}",
            f"SECTION 3. Interim Covenants\n{filler_c}",
            f"SECTION 4. Miscellaneous\n{filler_a}",
            f"SECTION 5. Disclosure Trigger\n{rui_2} {filler_b}",
            f"SECTION 6. Notices\n{filler_c}",
        ]
    )
    trial = {
        "instance_id": "maud_budgeted_test",
        "source_id": "contract_x__deal_protection",
        "contract_name": "contract_x",
        "category": "Deal Protection and Related Provisions",
        "candidate_index": 1,
        "plan": {
            "public_task": {
                "objective": "Which termination-fee trigger profile is supported?",
                "decision_set": ["decision_a", "decision_b"],
                "decision_option_definitions": {
                    "decision_a": "A twelve-month tail applies and prior public disclosure is required.",
                    "decision_b": "A twelve-month tail applies without any prior public disclosure condition.",
                },
            },
            "evidence_units": [
                {
                    "unit_id": "rui_01",
                    "text": rui_1,
                    "summary": "Tail period length.",
                    "information_feature": "tail period length",
                    "source_ref": {
                        "source_rui_id": "source_rui_001",
                        "issue_group_id": "issue_group_0001",
                    },
                },
                {
                    "unit_id": "rui_02",
                    "text": rui_2,
                    "summary": "Public disclosure condition.",
                    "information_feature": "public disclosure condition",
                    "source_ref": {
                        "source_rui_id": "source_rui_002",
                        "issue_group_id": "issue_group_0002",
                    },
                },
            ],
            "oracle": {"answer": "decision_a", "rationale": "Both RIUs are needed."},
        },
        "audit": {
            "smallest_sufficient_rui_sets": [{"unit_ids": ["rui_01", "rui_02"]}],
            "minimum_sufficient_rui_set_size": 2,
        },
    }

    partition = build_full_document_partition(
        trial,
        contract,
        rng=random.Random(5),
        seed=5,
        strategy="one_each",
        role_style="feature",
        target_words=45,
        min_words=10,
        soft_max_words=70,
        hard_max_words=100,
        min_grounded_rius=2,
        min_participants=4,
        context_mode="budgeted",
        context_budget_tokens=450,
        context_neighbor_chunks=0,
    )

    assert partition["context_mode"] == "budgeted"
    assert partition["validation"]["valid"] is True
    assert partition["common_datapoint"]["format_validation"]["valid"] is True
    assert (
        partition["common_datapoint"]["source"]["source_preservation"]
        == "budgeted_agreement_context_partitioned_into_private_chunks"
    )
    budgeting = partition["common_datapoint"]["dataset_specific"]["context_budgeting"]
    assert budgeting["selected_context_word_count"] < budgeting["original_context_word_count"]
    covered = {
        rui_id
        for unit in partition["document_units"]
        for rui_id in unit.get("hidden_contains_rui_ids", [])
    }
    assert covered == {"rui_01", "rui_02"}
