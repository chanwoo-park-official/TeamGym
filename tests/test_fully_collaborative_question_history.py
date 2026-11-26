from __future__ import annotations

import json

from teamgym.data.legal.question_history import (
    load_source_question_history,
    question_attempt_entry,
    question_history_prompt,
)


def test_question_history_groups_by_source_and_formats_prompt(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    rows = [
        {
            "source_id": "source_a",
            "instance_id": "old_1",
            "status": "accepted",
            "plan": {
                "public_task": {
                    "objective": "Which insurance profile applies?",
                    "decision_set": [{"value": "decision_a"}, {"value": "decision_b"}],
                },
                "oracle": {"answer": "decision_a"},
            },
        },
        {
            "source_id": "source_b",
            "instance_id": "old_2",
            "status": "plan_local_failed",
            "errors": ["option design too obvious"],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    grouped = load_source_question_history(path)
    prompt = question_history_prompt(grouped["source_a"])
    rejected = question_attempt_entry(
        source_id="source_a",
        instance_id="new_1",
        candidate_index=3,
        status="rejected_planner_local_validation",
        plan={
            "public_task": {
                "objective": "Which notice profile applies?",
                "decision_set": ["decision_a", "decision_b"],
            },
            "oracle": {"answer": "decision_b"},
        },
        errors=["single participant shortcut"],
    )

    assert set(grouped) == {"source_a", "source_b"}
    assert "Which insurance profile applies?" in prompt
    assert "decision_a" in prompt
    assert "Which notice profile applies?" in question_history_prompt([rejected])
    assert "single participant shortcut" in question_history_prompt([rejected])


def test_question_history_loader_accepts_funnel_report_object(tmp_path) -> None:
    path = tmp_path / "funnel.json"
    path.write_text(
        json.dumps(
            {
                "schema": "test_report",
                "rows": [
                    {
                        "source_id": "source_a",
                        "instance_id": "row_1",
                        "status": "accepted",
                        "plan_summary": {
                            "question_objective": "Which termination profile applies?",
                            "decision_set": ["decision_a", "decision_b"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    grouped = load_source_question_history(path)

    assert grouped["source_a"][0]["question"] == "Which termination profile applies?"
