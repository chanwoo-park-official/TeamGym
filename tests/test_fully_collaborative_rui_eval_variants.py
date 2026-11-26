from __future__ import annotations

from argparse import Namespace
import json
import sys

import pytest

from teamgym.core.schema import build_common_datapoint
from teamgym.interaction import run_evaluation as team
from teamgym.interaction import run_single_agent_rui_evaluation as single_agent
from teamgym.interaction import run_single_participant_evaluation as single_participant
from teamgym.interaction.evaluation import evaluate_episode
from teamgym.interaction.llm_actors import PromptTraceRecorder
from teamgym.interaction.rui_context import rui_only_datapoint
from teamgym.interaction.run_single_agent_rui_evaluation import run_single_agent_rui_episode
from teamgym.interaction.run_single_agent_full_context_evaluation import (
    full_source_context_text,
    full_source_context_units,
    run_single_agent_full_context_episode,
)


class FakeClient:
    last_request_meta = {"provider": "fake", "model": "fake-model"}

    def generate_json(self, **_: object) -> dict:
        return {
            "answer": "decision_a",
            "rationale": "RIU r1 and RIU r2 jointly support decision_a.",
            "confidence": 0.9,
        }


def _datapoint() -> dict:
    return build_common_datapoint(
        instance_id="sample_001",
        dataset="cuad",
        source={"source_id": "source_001"},
        scenario_family="test",
        task_description={"objective": "Choose the supported profile."},
        decision_set=["decision_a", "decision_b"],
        decision_option_definitions={
            "decision_a": "Both required terms are present.",
            "decision_b": "Only one required term is present.",
        },
        initial_common_information={"source_context": {"document_kind": "contract"}},
        participants=[
            {
                "participant_id": "participant_a",
                "role_name": "first reviewer",
                "private_context_unit_ids": ["du_1"],
                "private_word_count": 9,
                "owned_rui_ids": ["r1"],
            },
            {
                "participant_id": "participant_b",
                "role_name": "second reviewer",
                "private_context_unit_ids": ["du_2"],
                "private_word_count": 9,
                "owned_rui_ids": ["r2"],
            },
            {
                "participant_id": "participant_c",
                "role_name": "context reviewer",
                "private_context_unit_ids": ["du_3"],
                "private_word_count": 5,
                "owned_rui_ids": [],
            },
        ],
        private_context_units=[
            {
                "context_unit_id": "du_1",
                "text": "Long full contract chunk containing r1 plus unrelated text.",
                "assigned_to": ["participant_a"],
                "word_count": 9,
                "hidden_contains_rui_ids": ["r1"],
            },
            {
                "context_unit_id": "du_2",
                "text": "Long full contract chunk containing r2 plus unrelated text.",
                "assigned_to": ["participant_b"],
                "word_count": 9,
                "hidden_contains_rui_ids": ["r2"],
            },
            {
                "context_unit_id": "du_3",
                "text": "Unrelated background context.",
                "assigned_to": ["participant_c"],
                "word_count": 3,
                "hidden_contains_rui_ids": [],
            },
        ],
        reference_information_units=[
            {
                "rui_id": "r1",
                "text": "The first required term is present.",
                "summary": "First term.",
                "information_feature": "first term",
            },
            {
                "rui_id": "r2",
                "text": "The second required term is present.",
                "summary": "Second term.",
                "information_feature": "second term",
            },
        ],
        oracle={
            "answer": "decision_a",
            "smallest_sufficient_rui_sets": [{"rui_ids": ["r1", "r2"], "answer": "decision_a"}],
        },
        generation_metadata={"method": "test"},
    )


def test_rui_only_datapoint_replaces_full_context_with_owned_rius() -> None:
    transformed = rui_only_datapoint(_datapoint())

    assert transformed["format_validation"]["valid"] is True
    assert transformed["dataset_specific"]["interaction_context_variant"] == "rui_only"
    assert [unit["hidden_contains_rui_ids"] for unit in transformed["private_context_units"]] == [["r1"], ["r2"]]
    assert all("Long full contract chunk" not in unit["text"] for unit in transformed["private_context_units"])
    assert transformed["participants"][0]["private_context_unit_ids"] == ["rui_only_participant_a_r1"]
    assert transformed["participants"][2]["private_context_unit_ids"] == []


def test_single_agent_rui_episode_evaluates_with_all_rius_disclosed() -> None:
    datapoint = _datapoint()
    args = Namespace(
        context_token_budget=20000,
        final_rationale_token_budget=120,
        final_reasoning_token_budget=256,
    )

    episode = run_single_agent_rui_episode(
        datapoint,
        client=FakeClient(),
        recorder=PromptTraceRecorder(None),
        args=args,
    )
    evaluation = evaluate_episode(episode, datapoint)

    assert episode["communication_structure"]["name"] == "single_agent_all_rius"
    assert episode["final_decision"]["answer"] == "decision_a"
    assert set(episode["public_rui_ids"]) == {"r1", "r2"}
    assert evaluation["primary_eval"]["task_success"] is True
    assert evaluation["primary_eval"]["informed_success"] is True


@pytest.mark.parametrize(
    ("runner", "schema"),
    (
        (team, "teamgym_interaction_eval_run"),
        (single_agent, "teamgym_single_agent_eval_run"),
        (single_participant, "teamgym_single_participant_eval_run"),
    ),
)
def test_evaluation_cli_reports_skipped_rows_and_uses_teamgym_ids(
    tmp_path, monkeypatch, capsys, runner, schema
) -> None:
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(
        "\n" + json.dumps(_datapoint()) + "\n{}\n[]\n"
        + json.dumps({**_datapoint(), "instance_id": "beyond_limit"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "runs"
    argv = [
        "evaluate", "--input-root", str(inputs), "--output-dir", str(output),
        "--no-trace-prompts", "--limit", "1",
    ]
    if runner is not single_participant:
        argv.append("--no-render-html")
    if runner is team:
        argv.extend(["--communication-structure", "no_coordinator_round_robin__final_synthesis", "--turn-budget", "1"])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        runner, "_make_llm_client" if runner is team else "_make_client", lambda *_: FakeClient()
    )

    runner.main()

    run, = output.iterdir()
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "eval_summary.json").read_text(encoding="utf-8"))
    diagnostics = {
        "normalized_row_count": 2,
        "selected_row_count": 1,
        "skipped_row_count": 2,
        "skipped_rows_by_file": [{"path": str(inputs), "row_count": 2, "line_numbers": [3, 4]}],
    }
    assert manifest["schema_version"] == schema
    assert manifest["input_diagnostics"] == summary["input_diagnostics"] == diagnostics
    assert json.loads(capsys.readouterr().out)["input_diagnostics"] == diagnostics
    assert ("participant_response_mode" in manifest["llm"]) is (runner is team)
    for trajectory in (run / "trajectories").glob("*.json"):
        payload = json.loads(trajectory.read_text(encoding="utf-8"))
        assert payload["episode"]["schema_version"] == "teamgym_episode"
        assert payload["evaluation"]["schema_version"] == "teamgym_episode_evaluation"


def test_full_source_context_is_ordered_union_without_hidden_metadata() -> None:
    datapoint = _datapoint()
    datapoint["participants"][1]["private_context_unit_ids"].append("du_1")
    datapoint["private_context_units"].append(
        {
            "context_unit_id": "unassigned_secret",
            "text": "This material is not available to any participant.",
            "assigned_to": [],
            "hidden_contains_rui_ids": ["secret_rui"],
        }
    )

    units = full_source_context_units(datapoint)
    text, unit_count, token_count = full_source_context_text(
        datapoint,
        context_token_budget=20000,
    )

    assert [unit["context_unit_id"] for unit in units] == ["du_1", "du_2", "du_3"]
    assert text.count("Long full contract chunk containing r1") == 1
    assert text.index("containing r1") < text.index("containing r2") < text.index("Unrelated")
    assert "not available to any participant" not in text
    assert "hidden_contains_rui_ids" not in text
    assert unit_count == 3
    assert token_count > 0


def test_single_agent_full_context_prompt_uses_source_without_rui_annotations() -> None:
    datapoint = _datapoint()
    args = Namespace(
        context_token_budget=20000,
        final_rationale_token_budget=120,
        final_reasoning_token_budget=256,
    )
    recorder = PromptTraceRecorder(None)

    episode = run_single_agent_full_context_episode(
        datapoint,
        client=FakeClient(),
        recorder=recorder,
        args=args,
    )

    prompt = recorder.prompt_records[0]["user_prompt"]
    assert episode["communication_structure"]["name"] == "single_agent_full_context"
    assert episode["public_rui_ids"] == []
    assert episode["messages"][0]["disclosed_rui_ids"] == []
    assert "Long full contract chunk containing r1" in prompt
    assert "Unrelated background context" in prompt
    assert "reference_information_units" not in prompt
    assert "hidden_contains_rui_ids" not in prompt
