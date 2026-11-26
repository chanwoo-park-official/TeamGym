from __future__ import annotations

import json

from teamgym.data.legal.contracts.generation_records import read_generation_trials, split_generation_byproducts
from teamgym.data.legal.cuad.generation import (
    _measured_feedback_from_errors,
    _normalize_decision_set,
    _trim_to_smallest_sufficient_span_set,
)


def test_contract_trial_reader_accepts_both_formats_and_source_line_separators(tmp_path):
    rows = [{"source": "one\u2028two"}, {"source": "three"}]
    array = tmp_path / "trials.json"
    lines = tmp_path / "trials.jsonl"
    array.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    lines.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    assert read_generation_trials(array) == rows
    assert read_generation_trials(lines) == rows


def test_measured_cuad_decision_labels_normalize_to_paper_terms() -> None:
    plan = _normalize_decision_set(
        {
            "public_task": {
                "decision_set": [
                    {"value": "profile_a", "definition": "First legal profile."},
                    {"value": "profile_b", "definition": "Second legal profile."},
                ]
            },
            "evidence_units": [
                {
                    "unit_id": "span_01",
                    "decision_relevance": [
                        {"decision": "profile_b", "relation": "required_for", "reason": "Needed."}
                    ],
                }
            ],
            "oracle": {"answer": "profile_b"},
        }
    )

    assert plan["public_task"]["decision_set"] == ["decision_a", "decision_b"]
    assert plan["public_task"]["decision_option_definitions"] == {
        "decision_a": "First legal profile.",
        "decision_b": "Second legal profile.",
    }
    assert plan["oracle"]["answer"] == "decision_b"
    assert plan["evidence_units"][0]["decision_relevance"][0]["decision"] == "decision_b"


def test_generation_byproducts_are_separated_from_accepted_rows() -> None:
    accepted_rows, byproducts = split_generation_byproducts(
        [
            {
                "instance_id": "cuad_001",
                "generation_metadata": {"method": "unit_test"},
                "_generation_byproduct": {
                    "schema": "cuad_measured_span_generation_byproduct_v1",
                    "instance_id": "cuad_001",
                    "rejection_history": [{"stage": "planner_local_validation"}],
                },
            }
        ],
        min_accept_size=4,
        save_generation_byproducts=True,
    )

    assert "_generation_byproduct" not in accepted_rows[0]
    assert accepted_rows[0]["generation_byproduct_ref"] == {
        "path": "generation_byproducts_minaccept4.jsonl",
        "record_id": "cuad_001",
    }
    assert byproducts[0]["rejection_history"] == [{"stage": "planner_local_validation"}]


def test_measured_cuad_retains_one_smallest_sufficient_span_set() -> None:
    selection = {
        "selected_span_basis": [
            {"unit_id": "span_01", "source_span_index": 1, "text": "A"},
            {"unit_id": "span_02", "source_span_index": 2, "text": "B"},
            {"unit_id": "span_03", "source_span_index": 3, "text": "C"},
        ]
    }
    plan = {
        "evidence_units": [
            {"unit_id": "span_01", "text": "A"},
            {"unit_id": "span_02", "text": "B"},
            {"unit_id": "span_03", "text": "C"},
        ],
        "participants": [
            {"participant_id": "participant_1", "evidence_unit_ids": ["span_01", "span_03"]},
            {"participant_id": "participant_2", "evidence_unit_ids": ["span_02"]},
        ],
    }
    audit = {
        "minimum_sufficient_span_set_size": 2,
        "smallest_sufficient_span_sets": [
            {
                "unit_ids": ["span_01", "span_02"],
                "answer": "decision_a",
                "why_minimal": "Removing either span breaks the reference derivation.",
            }
        ],
    }

    final_selection, final_plan, metadata = _trim_to_smallest_sufficient_span_set(
        selection,
        plan,
        audit,
    )

    assert metadata["trimmed"] is True
    assert metadata["removed_unit_ids"] == ["span_03"]
    assert [row["unit_id"] for row in final_selection["selected_span_basis"]] == [
        "span_01",
        "span_02",
    ]
    assert [row["unit_id"] for row in final_plan["evidence_units"]] == [
        "span_01",
        "span_02",
    ]
    assert final_plan["participants"] == [
        {"participant_id": "participant_1", "evidence_unit_ids": ["span_01"]},
        {"participant_id": "participant_2", "evidence_unit_ids": ["span_02"]},
    ]


def test_cuad_retry_feedback_includes_previous_question_and_decision_set() -> None:
    plan = {
        "public_task": {
            "objective": "Which insurance profile applies?",
            "decision_set": ["decision_a", "decision_b"],
        },
        "evidence_units": [{"unit_id": "span_01", "source_ref": {"span_index": 1}}],
        "oracle": {"answer": "decision_b"},
    }

    feedback = _measured_feedback_from_errors(["span_02 was not necessary"], plan=plan)
    payload = json.loads(feedback.split("\n", 1)[1])
    previous = payload["previous_attempt"]

    assert previous["objective"] == plan["public_task"]["objective"]
    assert previous["decision_set"] == ["decision_a", "decision_b"]
    assert previous["oracle_answer"] == "decision_b"
    assert previous["evidence_units"][0]["span_index"] == 1
    assert payload["validation_errors"] == ["span_02 was not necessary"]
