from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from teamgym.rl.preparation.prepare_written_answer_legal import (
    SCHEMA_VERSION,
    SourceTask,
    ThreadLocalClient,
    cache_path,
    deterministic_failures,
    generate_task,
    transform_datapoint,
)


def _source_task() -> SourceTask:
    datapoint = {
        "schema_version": "teamgym_common_datapoint_v2",
        "instance_id": "budget_case",
        "dataset": "cuad",
        "scenario_family": "Budget",
        "source": {"source_id": "budget_case"},
        "task_description": {
            "title": "Project decision",
            "objective": "Which profile best matches the project?",
        },
        "decision_set": ["approve", "reject"],
        "decision_option_definitions": {
            "approve": "Cost is within budget.",
            "reject": "Cost exceeds budget.",
        },
        "initial_common_information": {
            "source_context": {"contract_title": "Project authorization"},
            "decision_options": {
                "approve": "Cost is within budget.",
                "reject": "Cost exceeds budget.",
            },
        },
        "participant_directory": [
            {"participant_id": "participant_a", "role_name": "Cost reviewer"},
            {"participant_id": "participant_b", "role_name": "Budget reviewer"},
        ],
        "participants": [
            {
                "participant_id": "participant_a",
                "role_name": "Cost reviewer",
                "private_context_unit_ids": ["ctx_cost"],
                "owned_rui_ids": ["rui_cost"],
            },
            {
                "participant_id": "participant_b",
                "role_name": "Budget reviewer",
                "private_context_unit_ids": ["ctx_budget"],
                "owned_rui_ids": ["rui_budget"],
            },
        ],
        "private_context_units": [
            {
                "context_unit_id": "ctx_cost",
                "text": "The signed estimate states a total cost of 40 dollars.",
                "assigned_to": ["participant_a"],
                "hidden_contains_rui_ids": ["rui_cost"],
            },
            {
                "context_unit_id": "ctx_budget",
                "text": "The authorization states a budget of 50 dollars.",
                "assigned_to": ["participant_b"],
                "hidden_contains_rui_ids": ["rui_budget"],
            },
        ],
        "reference_information_units": [
            {
                "rui_id": "rui_cost",
                "text": "The total project cost is 40 dollars.",
                "summary": "Project cost is 40 dollars.",
            },
            {
                "rui_id": "rui_budget",
                "text": "The authorized budget is 50 dollars.",
                "summary": "Authorized budget is 50 dollars.",
            },
        ],
        "oracle": {
            "answer": "approve",
            "smallest_sufficient_rui_sets": [
                {"rui_ids": ["rui_cost", "rui_budget"], "answer": "approve"}
            ],
        },
        "generation_metadata": {},
        "dataset_specific": {},
    }
    return SourceTask(
        split="validation",
        row={},
        datapoint=datapoint,
        required_rui_ids=("rui_cost", "rui_budget"),
        source_hash="source-hash",
        task_uid="budget_case",
        source_instance_id="budget_case",
    )


def _proposal() -> dict:
    return {
        "question": (
            "Determine whether the project should proceed by comparing the "
            "privately documented total cost with the separately authorized budget."
        ),
        "canonical_response": (
            "The total project cost is 40 dollars and the authorized budget is "
            "50 dollars. Because the cost does not exceed the budget, the project "
            "should be approved."
        ),
        "required_decision_steps": [
            {
                "step_id": "cost",
                "claim": "Establish that total project cost is 40 dollars.",
                "uses": ["rui_cost"],
                "step_type": "evidence",
            },
            {
                "step_id": "budget",
                "claim": "Establish that the authorized budget is 50 dollars.",
                "uses": ["rui_budget"],
                "step_type": "evidence",
            },
            {
                "step_id": "synthesis",
                "claim": "Compare cost to budget and conclude that approval follows.",
                "uses": ["rui_cost", "rui_budget"],
                "step_type": "synthesis",
            },
        ],
        "disqualifying_errors": [
            "States that the project cost exceeds the budget.",
            "States that the project must be rejected.",
        ],
        "judge_rubric": {
            "require_all_steps_for_pass": True,
            "passing_score": 1.0,
            "evaluation_criteria": [
                "State the project cost.",
                "State the authorized budget.",
                "Compare them and reach the supported conclusion.",
            ],
            "bare_label_fails": True,
            "vague_answer_fails": True,
            "unsupported_material_claim_fails": True,
        },
        "difficulty_audit": {
            "option_free": True,
            "question_does_not_reveal_answer": True,
            "every_required_rui_is_necessary": True,
            "single_rui_is_insufficient": True,
            "cross_participant_synthesis_required": True,
            "canonical_response_fully_grounded": True,
        },
    }


def test_deterministic_validator_accepts_strict_complete_proposal() -> None:
    assert deterministic_failures(_source_task(), _proposal()) == []


def test_cached_only_requires_an_accepted_record_without_opening_a_client(
    tmp_path: Path,
) -> None:
    task = _source_task()
    args = Namespace(output_root=tmp_path, cached_only=True)
    clients = Mock(spec=ThreadLocalClient)
    with pytest.raises(ValueError, match="no matching accepted generation record"):
        generate_task(task, args=args, clients=clients)

    record = {
        "schema_version": SCHEMA_VERSION,
        "source_hash": task.source_hash,
        "instance_id": task.task_uid,
        "accepted": True,
        "proposal": _proposal(),
    }
    path = cache_path(tmp_path, task.task_uid)
    path.parent.mkdir()
    path.write_text(json.dumps(record), encoding="utf-8")
    assert generate_task(task, args=args, clients=clients) == record
    clients.get.assert_not_called()


def test_deterministic_validator_rejects_disguised_multiple_choice() -> None:
    proposal = _proposal()
    proposal["question"] = (
        "Which profile best matches the project after comparing its cost and budget?"
    )
    failures = deterministic_failures(_source_task(), proposal)
    assert any("multiple-choice language" in failure for failure in failures)


def test_transform_removes_source_options_and_enables_strict_judge() -> None:
    task = _source_task()
    source = deepcopy(task.datapoint)
    source["generation_metadata"]["decision_option_permutation"] = {
        "old_answer": "approve"
    }
    record = {
        "instance_id": "budget_case",
        "proposal": _proposal(),
        "required_rui_ids": list(task.required_rui_ids),
        "source_hash": task.source_hash,
        "accepted_revision": 0,
        "generator_request_meta": {"model": "openai.gpt-5.5"},
        "critic_request_meta": {"model": "openai.gpt-5.5"},
    }

    transformed = transform_datapoint(source, record=record)

    assert transformed["task_description"]["answer_type"] == "short_answer_judged"
    assert transformed["decision_set"] == ["written_response"]
    assert transformed["oracle"]["pass_policy"] == "all_required_decision_steps"
    assert transformed["oracle"]["judge_rubric"]["require_all_steps_for_pass"] is True
    assert "decision_option_permutation" not in transformed["generation_metadata"]
    assert transformed["format_validation"]["valid"] is True
