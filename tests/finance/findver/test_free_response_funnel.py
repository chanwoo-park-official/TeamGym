from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from teamgym.data.finance.findver import free_response_funnel

from teamgym.core.information_allocation import (
    PAPER_ALLOCATION_MODES,
)
from teamgym.data.finance.findver.free_response_funnel import (
    _prepare_report_candidate,
    _prune_nonessential_riu,
)
from teamgym.data.finance.findver.free_response_validation import (
    leaked_context_ids,
    normalize_evidence_consolidation,
    normalize_free_response_judgment,
    normalize_free_response_plan,
    normalize_qa_evidence_mapping,
    validate_free_response_plan,
    validate_free_response_judgment,
    validate_packet_riu_audit,
    validate_evidence_consolidation,
    validate_qa_evidence_mapping,
)
from teamgym.data.finance.findver.leakage import (
    packet_bm25_hints,
)
from teamgym.data.finance.findver.partition import (
    build_full_context_partition,
    remove_surrounding_contexts,
    validate_packet_context_integrity,
)
from teamgym.data.finance.findver.source import (
    findver_report_candidates,
)
from teamgym.data.finance.findver.validation import (
    normalize_selection,
)


def _candidate() -> dict:
    return {
        "source_id": "free-response-test",
        "subset": "numeric",
        "split": "test",
        "source_statement": "Old verification statement.",
        "source_entailment_label": True,
        "source_knowledge": [],
        "report_name": "TEST-10-Q.json",
        "report_type": "10-Q",
        "candidate_context_ids": [10, 20],
        "candidate_contexts": [
            {
                "context_id": 10,
                "type": "table",
                "text": "Current operating income was 12.4 million.",
            },
            {
                "context_id": 20,
                "type": "table",
                "text": "Prior operating income was 10.0 million.",
            },
        ],
        "full_report_contexts": [
            {
                "context_id": 10,
                "type": "table",
                "text": "Current operating income was 12.4 million.",
            },
            {
                "context_id": 15,
                "type": "paragraph",
                "text": "The current-period operating income figure was 12.4 million.",
            },
            {
                "context_id": 20,
                "type": "table",
                "text": "Prior operating income was 10.0 million.",
            },
        ],
    }


def _selection(candidate: dict) -> dict:
    return normalize_selection(
        {
            "valid": True,
            "joint_question_theme": "operating income change",
            "target_operation": "percentage change",
            "selected_contexts": [
                {
                    "unit_id": "riu_current",
                    "context_id": 10,
                    "information_feature": "current operating income",
                    "entity_scope": "company",
                    "period_scope": "current period",
                    "why_critical": "the numerator cannot be calculated without it",
                },
                {
                    "unit_id": "riu_prior",
                    "context_id": 20,
                    "information_feature": "prior operating income",
                    "entity_scope": "company",
                    "period_scope": "prior period",
                    "why_critical": "the comparison base cannot be calculated without it",
                },
            ],
        },
        candidate,
    )


def _plan(selection: dict) -> dict:
    return normalize_free_response_plan(
        {
            "public_task": {
                "title": "Operating income review",
                "question": "What was the percentage change in operating income?",
                "response_format_note": "State the result and concise calculation.",
            },
            "oracle": {
                "short_answer": "24%",
                "canonical_response": "(12.4 - 10.0) / 10.0 = 24%.",
                "answer_spec": {
                    "type": "numeric",
                    "value": 24.0,
                    "unit": "percent",
                    "absolute_tolerance": 0.1,
                },
                "reasoning_steps": [
                    {
                        "claim_id": "claim_01",
                        "step": 1,
                        "uses": ["riu_current", "riu_prior"],
                        "claim": "The percentage change is 24%.",
                        "expected_answer_element": "24%",
                        "grading_rule": (
                            "Accept 24 percent within 0.1 percentage points."
                        ),
                        "judge_focus": "percent_change(12.4,10.0,24.0)",
                    }
                ],
                "judge_rubric": {
                    "passing_score": 0.8,
                    "criteria": [
                        {
                            "name": "answer",
                            "weight": 1.0,
                            "description": "Correct result and derivation.",
                        }
                    ],
                },
                "disqualifying_errors": ["The result is incorrect."],
            },
        },
        selection,
    )


def _three_riu_fixture() -> tuple[dict, dict, dict]:
    candidate = _candidate()
    extra_context = {
        "context_id": 30,
        "type": "paragraph",
        "text": "The comparison uses prior operating income as its base.",
    }
    candidate["candidate_context_ids"].append(30)
    candidate["candidate_contexts"].append(extra_context)
    candidate["full_report_contexts"].append(extra_context)
    selection = _selection(candidate)
    selection["selected_contexts"].append(
        {
            "unit_id": "riu_method",
            "context_id": 30,
            "source_context_ids": [30],
            "source_context_types": ["paragraph"],
            "context_type": "paragraph",
            "text": extra_context["text"],
            "information_feature": "comparison calculation base",
            "entity_scope": "company",
            "period_scope": "current and prior periods",
            "why_critical": "the calculation basis cannot be identified without it",
        }
    )
    plan = _plan(selection)
    for key in ("reasoning_steps", "required_decision_steps"):
        uses = plan["oracle"][key][0]["uses"]
        if "riu_method" not in uses:
            uses.append("riu_method")
    return candidate, selection, plan


def test_free_response_plan_and_partition_validate() -> None:
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(selection)
    assert validate_free_response_plan(plan, selection, candidate) == []

    scenario = build_full_context_partition(
        candidate={
            **candidate,
            "context_policy": "sanitized_native_context_packets",
        },
        selection=selection,
        plan=plan,
        audits={"grounding": {}, "public_only": {}, "riu_boundary": {}},
        seed=17,
        allocation_mode="one_each",
        min_participants=2,
    )
    common = scenario["common_datapoint"]
    assert common["format_validation"]["valid"]
    assert common["decision_set"] == ["free_response"]
    assert common["oracle"]["short_answer"] == "24%"
    assert common["oracle"]["answer_spec"]["value"] == 24.0
    assert common["task_description"]["answer_format"]["type"] == "free_response"


@pytest.mark.parametrize("allocation_mode", PAPER_ALLOCATION_MODES)
def test_all_five_allocation_modes_build_and_validate_foreign_riu_packets(
    allocation_mode: str,
) -> None:
    candidate, selection, plan = _three_riu_fixture()
    scenario = build_full_context_partition(
        candidate=candidate,
        selection=selection,
        plan=plan,
        audits={"grounding": {}, "public_only": {}, "riu_boundary": {}},
        seed=17,
        allocation_mode=allocation_mode,
        min_participants=2,
    )
    common = scenario["common_datapoint"]
    allocation = common["generation_metadata"][
        "private_information_allocation"
    ]
    assert common["format_validation"]["valid"] is True
    assert allocation["mode"] == allocation_mode
    assert allocation["validation"]["valid"] is True

    participants = {
        row["participant_id"]: row for row in common["participants"]
    }
    shared = {
        row["rui_id"]
        for row in common["initial_common_information"].get(
            "common_reference_units", []
        )
    }
    all_rui_ids = {
        row["rui_id"] for row in common["reference_information_units"]
    }
    units = {
        row["context_unit_id"]: row
        for row in common["private_context_units"]
    }
    for participant_id, participant in participants.items():
        authorized = set(participant["owned_rui_ids"]) | shared
        packet_units = [
            units[unit_id]
            for unit_id in participant["private_context_unit_ids"]
        ]
        for unit in packet_units:
            assert set(unit.get("hidden_contains_rui_ids") or []) <= authorized
        foreign = all_rui_ids - authorized
        audit = {
            "participant_id": participant_id,
            "unassigned_riu_results": [
                {
                    "riu_id": rui_id,
                    "visible": False,
                    "evidence_context_ids": [],
                    "reason": "the foreign RIU is not visible in this packet",
                }
                for rui_id in sorted(foreign)
            ],
        }
        packet_context_ids = {
            int(unit["source_ref"]["context_id"])
            for unit in packet_units
        }
        assert validate_packet_riu_audit(
            audit,
            all_rui_ids=all_rui_ids,
            assigned_or_shared_rui_ids=authorized,
            packet_context_ids=packet_context_ids,
            expected_participant_id=participant_id,
        ) == []

    selected_context_ids = {
        context_id
        for row in selection["selected_contexts"]
        for context_id in row.get("source_context_ids") or [row["context_id"]]
    }
    assert validate_packet_context_integrity(
        scenario,
        selected_riu_context_ids=selected_context_ids,
        removed_context_ids=set(),
    ) == []


def test_partition_uses_oracle_smallest_sufficient_sets() -> None:
    candidate, selection, plan = _three_riu_fixture()
    plan["oracle"]["smallest_sufficient_rui_sets"] = [
        {"rui_ids": ["riu_current", "riu_prior"]},
        {"rui_ids": ["riu_current", "riu_method"]},
    ]
    scenario = build_full_context_partition(
        candidate=candidate,
        selection=selection,
        plan=plan,
        audits={"grounding": {}, "public_only": {}, "riu_boundary": {}},
        seed=17,
        allocation_mode="one_each",
        min_participants=2,
    )
    oracle = scenario["common_datapoint"]["oracle"]
    assert oracle["minimum_sufficient_rui_set_size"] == 2
    assert [
        row["rui_ids"] for row in oracle["smallest_sufficient_rui_sets"]
    ] == [
        ["riu_current", "riu_prior"],
        ["riu_current", "riu_method"],
    ]
    allocation = scenario["common_datapoint"]["generation_metadata"][
        "private_information_allocation"
    ]
    assert allocation["validation"]["metrics"][
        "smallest_sufficient_set_count"
    ] == 2


def test_partition_rejects_unknown_oracle_sufficient_set_riu() -> None:
    candidate, selection, plan = _three_riu_fixture()
    plan["oracle"]["smallest_sufficient_rui_sets"] = [
        {"rui_ids": ["riu_current", "riu_unknown"]}
    ]
    with pytest.raises(ValueError, match="references unknown RIUs"):
        build_full_context_partition(
            candidate=candidate,
            selection=selection,
            plan=plan,
            audits={"grounding": {}, "public_only": {}, "riu_boundary": {}},
            seed=17,
            allocation_mode="one_each",
            min_participants=2,
        )


def test_composite_free_response_answer_spec_validates_claim_links() -> None:
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(selection)
    plan["public_task"]["question"] = (
        "State current operating income and prior operating income."
    )
    plan["oracle"]["short_answer"] = "Current: 12.4 million; prior: 10.0 million."
    plan["oracle"]["canonical_response"] = (
        "1. Current operating income was 12.4 million. "
        "2. Prior operating income was 10.0 million."
    )
    plan["oracle"]["answer_spec"] = {
        "type": "composite",
        "items": [
            {
                "item_id": "item_01",
                "claim_ids": ["claim_01"],
                "type": "numeric",
                "value": 12.4,
                "unit": "million",
                "absolute_tolerance": 0.0,
            },
            {
                "item_id": "item_02",
                "claim_ids": ["claim_02"],
                "type": "numeric",
                "value": 10.0,
                "unit": "million",
                "absolute_tolerance": 0.0,
            },
        ],
    }
    plan["oracle"]["reasoning_steps"] = [
        {
            "claim_id": "claim_01",
            "step": 1,
            "uses": ["riu_current"],
            "claim": "Current operating income was 12.4 million.",
            "expected_answer_element": "12.4 million",
            "grading_rule": "Require 12.4 million.",
            "judge_focus": "current-period value",
        },
        {
            "claim_id": "claim_02",
            "step": 2,
            "uses": ["riu_prior"],
            "claim": "Prior operating income was 10.0 million.",
            "expected_answer_element": "10.0 million",
            "grading_rule": "Require 10.0 million.",
            "judge_focus": "prior-period value",
        },
    ]
    assert validate_free_response_plan(plan, selection, candidate) == []

    plan["oracle"]["answer_spec"]["items"][1]["claim_ids"] = ["claim_missing"]
    errors = validate_free_response_plan(plan, selection, candidate)
    assert any("unknown claims" in error for error in errors)


def test_bm25_hints_surface_repeated_riu_fact() -> None:
    packet = {
        "contexts": [
            {
                "source_ref": {"context_id": 15},
                "text": "The current-period operating income figure was 12.4 million.",
            }
        ]
    }
    ruis = [
        {
            "riu_id": "riu_current",
            "text": "Current operating income was 12.4 million.",
            "information_feature": "current operating income",
        }
    ]
    hints = packet_bm25_hints(
        packet,
        ruis,
        unauthorized_rui_ids={"riu_current"},
    )
    assert hints
    assert hints[0]["context_id"] == 15
    assert hints[0]["riu_id"] == "riu_current"


def test_packet_audit_collects_clear_unassigned_riu_context_ids() -> None:
    audit = {
        "participant_id": "participant_a",
        "unassigned_riu_results": [
            {
                "riu_id": "riu_current",
                "visible": True,
                "evidence_context_ids": [15],
                "reason": "context 15 directly repeats the current value",
            },
        ],
    }
    errors = validate_packet_riu_audit(
        audit,
        all_rui_ids={"riu_current", "riu_prior"},
        assigned_or_shared_rui_ids={"riu_prior"},
        packet_context_ids={15, 20},
        expected_participant_id="participant_a",
    )
    assert errors == []
    assert leaked_context_ids(audit) == {15}


def test_packet_audit_does_not_request_assigned_riu_duplicates() -> None:
    audit = {
        "participant_id": "participant_a",
        "unassigned_riu_results": [
            {
                "riu_id": "riu_prior",
                "visible": False,
                "evidence_context_ids": [],
                "reason": "no prior-period value appears",
            }
        ],
    }
    assert validate_packet_riu_audit(
        audit,
        all_rui_ids={"riu_current", "riu_prior"},
        assigned_or_shared_rui_ids={"riu_current"},
        packet_context_ids={10, 15},
        expected_participant_id="participant_a",
    ) == []
    assert leaked_context_ids(audit) == set()


def test_surrounding_context_removal_preserves_fixed_packet_assignment() -> None:
    candidate = _candidate()
    selection = _selection(candidate)
    scenario = build_full_context_partition(
        candidate=candidate,
        selection=selection,
        plan=_plan(selection),
        audits={"grounding": {}, "public_only": {}, "riu_boundary": {}},
        seed=17,
        allocation_mode="one_each",
        min_participants=2,
    )
    common = scenario["common_datapoint"]
    owner_before = next(
        row["assigned_to"]
        for row in common["private_context_units"]
        if row["source_ref"]["context_id"] == 15
    )
    result = remove_surrounding_contexts(
        scenario,
        source_context_ids={15},
        selected_riu_context_ids={10, 20},
    )
    assert owner_before
    assert result["removed_context_ids"] == [15]
    assert {
        row["source_ref"]["context_id"]
        for row in common["private_context_units"]
    } == {10, 20}
    assert validate_packet_context_integrity(
        scenario,
        selected_riu_context_ids={10, 20},
        removed_context_ids={15},
    ) == []


def test_claim_level_judgment_requires_available_evidence() -> None:
    oracle = _plan(_selection(_candidate()))["oracle"]
    judgment = {
        "passes": True,
        "final_answer_correct": True,
        "required_step_coverage": 1.0,
        "source_grounded": True,
        "claim_results": [
            {
                "claim_id": "claim_01",
                "status": "satisfied",
                "content_correct": True,
                "grounded_in_available_evidence": True,
                "supporting_riu_ids": ["riu_current", "riu_prior"],
                "reason": "both operands and the calculation are present",
            }
        ],
        "disqualifying_errors": [],
        "reason": "complete",
    }
    assert validate_free_response_judgment(
        judgment,
        oracle=oracle,
        available_riu_ids={"riu_current", "riu_prior"},
    ) == []
    errors = validate_free_response_judgment(
        judgment,
        oracle=oracle,
        available_riu_ids={"riu_current"},
    )
    assert any("unavailable RIUs" in error for error in errors)


def test_judgment_summary_fields_are_derived_from_atomic_results() -> None:
    oracle = _plan(_selection(_candidate()))["oracle"]
    judgment = {
        "passes": False,
        "final_answer_correct": True,
        "required_step_coverage": 0.0,
        "source_grounded": True,
        "claim_results": [
            {
                "claim_id": "claim_01",
                "status": "missing",
                "content_correct": True,
                "grounded_in_available_evidence": True,
                "supporting_riu_ids": ["riu_current", "riu_prior"],
                "reason": "both operands and the calculation are present",
            }
        ],
        "disqualifying_errors": [],
    }
    normalize_free_response_judgment(judgment, oracle=oracle)
    assert judgment["claim_results"][0]["status"] == "satisfied"
    assert judgment["required_step_coverage"] == 1.0
    assert judgment["passes"] is True
    assert validate_free_response_judgment(
        judgment,
        oracle=oracle,
        available_riu_ids={"riu_current", "riu_prior"},
    ) == []


def test_qa_mapping_requires_exact_annotated_contexts() -> None:
    qa = {
        "qa_id": "numeric-1",
        "relevant_contexts": [
            {"context_id": 10, "type": "table", "text": "Value 10"},
            {"context_id": 20, "type": "table", "text": "Value 20"},
        ],
    }
    mapping = {
        "qa_id": "numeric-1",
        "answer_reconstructable": True,
        "evidence_blocks": [
            {
                "context_id": 10,
                "role": "calculation_input",
                "contribution": "first operand",
            },
            {
                "context_id": 20,
                "role": "calculation_input",
                "contribution": "second operand",
            },
        ],
        "reconstruction_summary": "Both operands reconstruct the result.",
    }
    assert validate_qa_evidence_mapping(mapping, qa=qa) == []


def test_qa_mapping_merges_duplicate_block_roles() -> None:
    qa = {
        "qa_id": "numeric-1",
        "relevant_contexts": [
            {"context_id": 10, "type": "table", "text": "Value 10"}
        ],
    }
    mapping = {
        "qa_id": "numeric-1",
        "answer_reconstructable": True,
        "evidence_blocks": [
            {
                "context_id": 10,
                "role": "calculation_input",
                "contribution": "provides the operand",
            },
            {
                "context_id": 10,
                "role": "period_or_unit_scope",
                "contribution": "establishes that the value is in millions",
            },
        ],
        "reconstruction_summary": "The block supplies the value and unit.",
    }
    normalize_qa_evidence_mapping(mapping)
    assert len(mapping["evidence_blocks"]) == 1
    assert mapping["evidence_blocks"][0]["roles"] == [
        "calculation_input",
        "period_or_unit_scope",
    ]
    assert validate_qa_evidence_mapping(mapping, qa=qa) == []


def test_report_mapping_builds_union_candidate_pool() -> None:
    class MappingClient:
        cfg = SimpleNamespace(model="fake-gpt-5.4")
        last_request_meta = {}

        def generate_json(self, *, user_prompt, **kwargs):
            if "Consolidate the mapped evidence candidates" in user_prompt:
                return {
                    "valid": True,
                    "merge_groups": [],
                    "excluded_contexts": [],
                    "information_relations": [],
                    "issues": [],
                    "failed_reason": "",
                }
            qa_id = "qa-1" if '"qa_id": "qa-1"' in user_prompt else "qa-2"
            context_id = 10 if qa_id == "qa-1" else 20
            return {
                "qa_id": qa_id,
                "answer_reconstructable": True,
                "evidence_blocks": [
                    {
                        "context_id": context_id,
                        "role": "direct_verification",
                        "contribution": "required source fact",
                    }
                ],
                "reconstruction_summary": "The annotated block verifies the QA.",
                "issues": [],
            }

    candidate = {
        "source_id": "report-1",
        "subset": "ie",
        "report_name": "REPORT.json",
        "source_qas": [
            {
                "qa_id": "qa-1",
                "subset": "ie",
                "statement": "first",
                "entailment_label": True,
                "relevant_context_ids": [10],
                "relevant_contexts": [
                    {"context_id": 10, "type": "paragraph", "text": "first fact"}
                ],
                "public_knowledge": [],
            },
            {
                "qa_id": "qa-2",
                "subset": "numeric",
                "statement": "second",
                "entailment_label": True,
                "relevant_context_ids": [20],
                "relevant_contexts": [
                    {"context_id": 20, "type": "table", "text": "second fact"}
                ],
                "public_knowledge": [],
            },
        ],
        "full_report_contexts": [
            {"context_id": 10, "type": "paragraph", "text": "first fact"},
            {"context_id": 20, "type": "table", "text": "second fact"},
            {"context_id": 30, "type": "paragraph", "text": "background"},
        ],
    }
    prepared, record = _prepare_report_candidate(
        candidate=candidate,
        audit_client=MappingClient(),
        trace_root=Path("."),
        min_units=2,
    )
    assert prepared is not None
    assert record["errors"] == []
    assert prepared["candidate_context_ids"] == [10, 20]
    assert prepared["candidate_contexts"][0]["source_qa_ids"] == ["qa-1"]


def test_consolidation_accepts_adjacent_fragments_and_partial_overlap() -> None:
    consolidation = {
        "valid": True,
        "merge_groups": [
            {"source_context_ids": [10, 11], "reason": "one split table row"}
        ],
        "excluded_contexts": [],
        "information_relations": [
            {
                "relation_type": "partial_overlap",
                "duplicated_information": "same total",
                "provider_context_ids": [10, 20],
            }
        ],
    }
    assert validate_evidence_consolidation(
        consolidation,
        candidate_context_ids=[10, 11, 20],
    ) == []


def test_consolidation_ignores_noop_self_exclusion() -> None:
    consolidation = {
        "valid": True,
        "merge_groups": [],
        "excluded_contexts": [
            {
                "context_id": 74,
                "covered_by_context_ids": [74],
                "relation_type": "exact_duplicate_within_block",
                "reason": "duplicate rows within the same native block",
            }
        ],
        "information_relations": [],
        "issues": [],
    }
    normalize_evidence_consolidation(consolidation)
    assert consolidation["excluded_contexts"] == []
    assert "no-op self-exclusion" in consolidation["issues"][0]
    assert validate_evidence_consolidation(
        consolidation,
        candidate_context_ids=[74],
    ) == []


def test_partition_keeps_merged_candidate_fragments_with_one_riu_owner() -> None:
    candidate = _candidate()
    candidate["candidate_contexts"][0]["candidate_block_id"] = "merged_000010_000015"
    candidate["candidate_contexts"][0]["source_context_ids"] = [10, 15]
    candidate["candidate_contexts"][0]["source_context_types"] = [
        "table",
        "paragraph",
    ]
    candidate["candidate_contexts"][0]["text"] = (
        "Current operating income was 12.4 million.\n\n"
        "The current-period operating income figure was 12.4 million."
    )
    selection = _selection(candidate)
    scenario = build_full_context_partition(
        candidate=candidate,
        selection=selection,
        plan=_plan(selection),
        audits={"grounding": {}, "public_only": {}, "riu_boundary": {}},
        seed=17,
        allocation_mode="one_each",
        min_participants=2,
    )
    contexts = scenario["common_datapoint"]["private_context_units"]
    owners = {
        row["source_ref"]["context_id"]: row["assigned_to"]
        for row in contexts
        if row["source_ref"]["context_id"] in {10, 15}
    }
    assert owners[10] == owners[15]


def test_successful_loo_prunes_and_relinks_claim() -> None:
    candidate = _candidate()
    candidate["candidate_contexts"].append(
        {
            "context_id": 30,
            "type": "paragraph",
            "text": "A duplicate prior operating income was 10.0 million.",
        }
    )
    selection = normalize_selection(
        {
            "valid": True,
            "selected_contexts": [
                {
                    "unit_id": "riu_current",
                    "context_id": 10,
                    "information_feature": "current income",
                    "entity_scope": "company",
                    "period_scope": "current",
                    "why_critical": "current operand",
                },
                {
                    "unit_id": "riu_prior",
                    "context_id": 20,
                    "information_feature": "prior income",
                    "entity_scope": "company",
                    "period_scope": "prior",
                    "why_critical": "prior operand",
                },
                {
                    "unit_id": "riu_duplicate",
                    "context_id": 30,
                    "information_feature": "duplicated prior income",
                    "entity_scope": "company",
                    "period_scope": "prior",
                    "why_critical": "claimed duplicate operand",
                },
            ],
        },
        candidate,
    )
    plan = _plan(selection)
    plan["oracle"]["reasoning_steps"][0]["uses"] = [
        "riu_current",
        "riu_prior",
        "riu_duplicate",
    ]
    panel = {
        "case_results": [
            {
                "case_id": "leave_out_riu_duplicate",
                "is_full_set": False,
                "omitted_riu_id": "riu_duplicate",
                "pass_count": 1,
                "solver_results": [
                    {
                        "passes": True,
                        "judgment": {
                            "claim_results": [
                                {
                                    "claim_id": "claim_01",
                                    "status": "satisfied",
                                    "supporting_riu_ids": [
                                        "riu_current",
                                        "riu_prior",
                                    ],
                                }
                            ]
                        },
                    }
                ],
            }
        ]
    }
    repaired = _prune_nonessential_riu(
        selection=selection,
        plan=plan,
        boundary_panel=panel,
        boundary_errors=[
            "leave_out_riu_duplicate: answer recovered by 1/1 solvers"
        ],
        min_units=2,
    )
    assert repaired is not None
    repaired_selection, repaired_plan, record = repaired
    assert record["removed_riu_id"] == "riu_duplicate"
    assert {
        row["unit_id"] for row in repaired_selection["selected_contexts"]
    } == {"riu_current", "riu_prior"}
    assert repaired_plan["oracle"]["reasoning_steps"][0]["uses"] == [
        "riu_current",
        "riu_prior",
    ]


def test_report_loader_can_mix_all_subsets(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "R.json").write_text(
        json.dumps(
            {
                "context": [
                    {"id": 10, "type": "paragraph", "context": "IE fact"},
                    {"id": 20, "type": "table", "context": "Numeric fact"},
                ]
            }
        ),
        encoding="utf-8",
    )
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(
        json.dumps(
            [
                {
                    "example_id": "ie-1",
                    "subset": "ie",
                    "report": "R.json",
                    "relevant_context": [10],
                },
                {
                    "example_id": "numeric-1",
                    "subset": "numeric",
                    "report": "R.json",
                    "relevant_context": [20],
                },
            ]
        ),
        encoding="utf-8",
    )
    candidates = list(
        findver_report_candidates(
            qa_path=qa_path,
            reports_root=reports,
            subset="all",
        )
    )
    assert len(candidates) == 1
    assert candidates[0]["source_subsets"] == ["ie", "numeric"]
    assert {
        qa["subset"] for qa in candidates[0]["source_qas"]
    } == {"ie", "numeric"}



@pytest.mark.parametrize("candidate_limit", [0, 1])
def test_report_limit_does_not_read_an_unrequested_report(tmp_path, monkeypatch, candidate_limit):
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps([
        {"report": name, "example_id": name, "subset": "numeric", "relevant_context": [1]}
        for name in ("first.json", "second.json")
    ]))
    if candidate_limit:
        (tmp_path / "first.json").write_text(json.dumps({
            "context": [{"id": 1, "type": "paragraph", "context": "Reported revenue was 10."}],
        }))
    monkeypatch.setattr(
        free_response_funnel, "_prepare_report_candidate",
        lambda **_: (None, {"errors": ["No model calls in this source-loading check"]}),
    )
    client = SimpleNamespace(cfg=SimpleNamespace(model="mock"))

    report = free_response_funnel.build_dataset(
        qa_path=qa_path, reports_root=tmp_path, output_root=tmp_path / "output",
        generator_client=client, audit_client=client,
        blind_clients=[("solver", client)], subset="numeric",
        candidate_limit=candidate_limit,
    )

    assert report["candidate_count"] == candidate_limit
    assert report["funnel_counts"] == (
        {"candidate_pool_rejected": 1} if candidate_limit else {}
    )


def test_findver_cli_initializes_the_construction_client(tmp_path, monkeypatch):
    env_file = tmp_path / "model.env"
    env_file.write_text("FINDVER_TEST_API_KEY=placeholder\n", encoding="utf-8")
    monkeypatch.delenv("FINDVER_TEST_API_KEY", raising=False)
    monkeypatch.setattr("openai.OpenAI", lambda **_: object())

    def build(**inputs):
        assert inputs["generator_client"].api_key == "placeholder"
        assert inputs["audit_client"].cfg.env_file == str(env_file)
        return {"accepted_count": 0}

    monkeypatch.setattr(free_response_funnel, "build_dataset", build)
    monkeypatch.setattr("sys.argv", [
        "teamgym-generate findver",
        "--qa-path", str(tmp_path / "source.json"),
        "--reports-root", str(tmp_path / "reports"),
        "--output-root", str(tmp_path / "out"),
        "--env-file", str(env_file),
        "--api-key-env", "FINDVER_TEST_API_KEY",
    ])
    with pytest.raises(SystemExit) as failure:
        free_response_funnel.main()
    assert failure.value.code == 2
