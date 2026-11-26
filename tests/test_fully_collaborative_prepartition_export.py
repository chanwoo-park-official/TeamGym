from __future__ import annotations

import json

from teamgym.data.legal.prepartition_export import (
    pre_partition_task_tuple,
    write_pre_partition_task_tuples,
)


def _accepted_trial() -> dict:
    return {
        "instance_id": "trial_001",
        "source_id": "source_001",
        "candidate_index": 1,
        "label_family": "Insurance",
        "plan": {
            "public_task": {
                "objective": "Which insurance profile is supported?",
                "contract_context": "A contract excerpt describes insurance obligations.",
                "decision_set": ["decision_a", "decision_b"],
                "decision_option_definitions": {
                    "decision_a": "The required coverage limit is five million dollars.",
                    "decision_b": "The required coverage limit is one million dollars.",
                },
            },
            "evidence_units": [
                {
                    "unit_id": "rui_01",
                    "text": "Alpha must maintain insurance coverage with a five million dollar limit.",
                    "summary": "Insurance limit.",
                    "information_feature": "coverage limit",
                    "criticality": "critical",
                    "source_ref": {"source_span_index": 1},
                    "decision_relevance": [
                        {
                            "decision": "decision_a",
                            "relation": "required_for",
                            "reason": "States the five million dollar limit.",
                        }
                    ],
                }
            ],
            "oracle": {"answer": "decision_a", "rationale": "The RIU states the limit."},
        },
        "audit": {
            "smallest_sufficient_span_sets": [
                {
                    "unit_ids": ["rui_01"],
                    "answer": "decision_a",
                    "why_sufficient": "The RIU states the limit.",
                    "why_minimal": "No smaller set exists.",
                },
                {
                    "unit_ids": ["rui_01", "rui_removed"],
                    "answer": "decision_a",
                    "why_sufficient": "An alternative pre-trim set.",
                    "why_minimal": "Recorded before post-audit trimming.",
                },
            ]
        },
    }


def test_pre_partition_task_tuple_flattens_rius_question_and_options() -> None:
    row = pre_partition_task_tuple(_accepted_trial(), dataset="cuad")

    assert row["schema"] == "legal_pre_partition_task_tuple_v1"
    assert row["question"] == "Which insurance profile is supported?"
    assert row["options"] == [
        {
            "decision_id": "decision_a",
            "definition": "The required coverage limit is five million dollars.",
        },
        {
            "decision_id": "decision_b",
            "definition": "The required coverage limit is one million dollars.",
        },
    ]
    assert row["oracle"]["answer"] == "decision_a"
    assert row["rius"][0]["rui_id"] == "rui_01"
    assert [item["rui_ids"] for item in row["smallest_sufficient_rui_sets"]] == [["rui_01"]]


def test_write_pre_partition_task_tuples_writes_jsonl_and_pretty_json(tmp_path) -> None:
    paths = write_pre_partition_task_tuples(
        output_root=tmp_path,
        dataset="cuad",
        min_accept_size=2,
        accepted_trials=[_accepted_trial()],
    )

    jsonl_path = tmp_path / "pre_partition_task_tuples_minaccept2.jsonl"
    pretty_path = tmp_path / "pre_partition_task_tuples_minaccept2.pretty.json"
    assert paths == {"jsonl": str(jsonl_path), "pretty_json": str(pretty_path)}
    assert json.loads(jsonl_path.read_text(encoding="utf-8"))["instance_id"] == "trial_001"
    assert json.loads(pretty_path.read_text(encoding="utf-8"))[0]["question"] == (
        "Which insurance profile is supported?"
    )
