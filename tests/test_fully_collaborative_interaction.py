from __future__ import annotations

import json
from pathlib import Path

import pytest

from teamgym.interaction import (
    BudgetSpec,
    CommunicationStructure,
    communication_structure_preset,
    evaluate_episode,
    iter_common_datapoints,
    run_episode,
)
from teamgym.interaction.llm_actors import (
    LLMCoordinatorActor,
    LLMFinalDecisionActor,
    LLMParticipantActor,
    LLMTwoStageParticipantActor,
    PromptTraceRecorder,
    _private_context_payload,
)
from teamgym.interaction.actor_utils import count_tokens, slice_tokens
from teamgym.interaction.public_reading import ParticipantPublicReader
from teamgym.visualization.interaction_trajectory_report import build_report
from teamgym.core.io import write_json
from teamgym.core.schema import validate_common_datapoint
from teamgym.data.long_context_reasoning.seed_builder import build_from_seed_specs


def _toy_datapoint() -> dict:
    return {
        "schema_version": "teamgym_common_datapoint_v2",
        "instance_id": "toy_001",
        "dataset": "toy",
        "scenario_family": "unit_test",
        "source": {"source_id": "toy"},
        "task_description": {"objective": "Choose the supported decision."},
        "decision_set": ["decision_a", "decision_b"],
        "decision_option_definitions": {
            "decision_a": "Wrong option.",
            "decision_b": "Supported option.",
        },
        "initial_common_information": {
            "source_context": {
                "dataset": "toy",
                "document_kind": "unit_test",
                "common_fact": "Shared common fact.",
            }
        },
        "participant_directory": [
            {"participant_id": "participant_a", "role_name": "alpha reviewer"},
            {"participant_id": "participant_b", "role_name": "beta reviewer"},
        ],
        "participants": [
            {
                "participant_id": "participant_a",
                "role_name": "alpha reviewer",
                "private_context_unit_ids": ["du_001", "du_003"],
                "owned_rui_ids": ["rui_001"],
            },
            {
                "participant_id": "participant_b",
                "role_name": "beta reviewer",
                "private_context_unit_ids": ["du_002"],
                "owned_rui_ids": ["rui_002"],
            },
        ],
        "private_context_units": [
            {"context_unit_id": "du_001", "text": "Alpha fact.", "assigned_to": ["participant_a"]},
            {"context_unit_id": "du_002", "text": "Beta fact.", "assigned_to": ["participant_b"]},
            {"context_unit_id": "du_003", "text": "Alpha second assigned document fact.", "assigned_to": ["participant_a"]},
        ],
        "reference_information_units": [
            {"rui_id": "rui_001", "summary": "Alpha fact matters."},
            {"rui_id": "rui_002", "summary": "Beta fact matters."},
        ],
        "oracle": {
            "answer": "decision_b",
            "smallest_sufficient_rui_sets": [{"rui_ids": ["rui_001", "rui_002"], "answer": "decision_b"}],
        },
        "generation_metadata": {"method": "unit_test"},
        "dataset_specific": {},
        "format_validation": {"valid": True},
    }


@pytest.mark.parametrize(
    ("corruption", "check_name", "prompt_error"),
    [
        ("foreign", "private_context_assignments_match", "not assigned"),
        ("duplicate", "private_context_unit_ids_nonempty_and_unique", "Duplicate private context"),
    ],
)
def test_invalid_context_assignments_cannot_reach_participant_prompts(
    corruption, check_name, prompt_error
) -> None:
    datapoint = _toy_datapoint()
    participant = datapoint["participants"][0]
    if corruption == "foreign":
        participant["private_context_unit_ids"].append("du_002")
    else:
        datapoint["private_context_units"].append(dict(datapoint["private_context_units"][0]))
    result = validate_common_datapoint(datapoint)
    assert result["valid"] is False
    assert {"name": check_name, "passed": False} in result["checks"]
    with pytest.raises(ValueError, match=prompt_error):
        _private_context_payload(datapoint, participant, BudgetSpec())


def test_written_episode_reports_decision_label_without_correctness() -> None:
    datapoint = _toy_datapoint()
    datapoint["task_description"]["answer_type"] = "short_answer_judged"
    datapoint["oracle"]["answer"] = "free_response"
    result = evaluate_episode({"final_decision": {"answer": "free_response"}}, datapoint)
    assert result["primary_eval"]["decision_label_matches"] is True
    assert "task_success" not in result["primary_eval"]


def test_coordinator_select_one_episode_reaches_informed_success() -> None:
    datapoint = _toy_datapoint()
    episode = _run_fake_llm_episode(
        datapoint,
        communication_structure_preset("coordinator_select_one__coordinator_decision"),
        budgets=BudgetSpec(turn_budget=4),
    )
    evaluation = evaluate_episode(episode, datapoint)

    assert episode["communication_structure"]["turn_allocation_rule"] == "coordinator_selects_one"
    assert episode["final_decision"]["answer"] == "decision_b"
    assert evaluation["primary_eval"]["task_success"] is True
    assert evaluation["primary_eval"]["informed_success"] is True
    assert set(episode["public_rui_ids"]) == set()
    assert set(evaluation["reference_information_matching"]["lexically_matched_rui_ids"]) == {"rui_001", "rui_002"}


def test_round_robin_runs_without_coordinator() -> None:
    datapoint = _toy_datapoint()
    episode = _run_fake_llm_episode(
        datapoint,
        communication_structure_preset("no_coordinator_round_robin__final_synthesis"),
        budgets=BudgetSpec(turn_budget=4),
    )

    assert episode["communication_structure"]["coordinator_present"] is False
    assert all(message["speaker_role"] == "participant" for message in episode["messages"])
    assert episode["final_decision"]["actor_role"] == "final_synthesis"


def test_round_robin_designated_participant_decides_with_private_context(tmp_path: Path) -> None:
    datapoint = _toy_datapoint()
    client = _FakeJsonClient()
    recorder = PromptTraceRecorder(tmp_path / "prompts")
    episode = run_episode(
        datapoint,
        communication_structure_preset("no_coordinator_round_robin__designated_participant"),
        budgets=BudgetSpec(turn_budget=3),
        participant_actor=LLMParticipantActor(client, recorder),
        coordinator_actor=LLMCoordinatorActor(client, recorder),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )

    assert episode["final_decision"]["actor_id"] == "participant_a"
    assert episode["final_decision"]["actor_role"] == "participant"
    final_record = next(
        row
        for row in recorder.prompt_records
        if row["actor_role"] == "participant" and row["prompt_kind"] == "final_synthesis"
    )
    final_payload = json.loads(final_record["user_prompt"])
    assert "Alpha fact." in final_payload["private_source_material"]
    assert "Beta fact." not in final_payload["private_source_material"]
    assert "designated participant" in final_record["system_prompt"]


def test_coordinator_ask_everybody_requests_every_participant() -> None:
    datapoint = _toy_datapoint()
    episode = _run_fake_llm_episode(
        datapoint,
        communication_structure_preset("coordinator_ask_everybody__coordinator_decision"),
        budgets=BudgetSpec(turn_budget=2),
    )

    first_turn = episode["turns"][0]
    assert first_turn["actions"][0]["target_ids"] == ["participant_a", "participant_b"]
    assert [message["speaker_id"] for message in first_turn["messages"]] == [
        "coordinator",
        "participant_a",
        "participant_b",
    ]
    participant_prompts = [
        json.loads(row["user_prompt"])
        for row in episode["prompt_records"]
        if row["prompt_kind"] == "participant_response"
    ]
    assert len(participant_prompts) == 2
    assert all(prompt["chat_history"][-1]["target_ids"] == ["participant_a", "participant_b"] for prompt in participant_prompts)


def test_coordinator_select_multiple_can_target_subset() -> None:
    datapoint = _toy_datapoint()
    datapoint["participant_directory"].append({"participant_id": "participant_c", "role_name": "gamma reviewer"})
    datapoint["participants"].append(
        {
            "participant_id": "participant_c",
            "role_name": "gamma reviewer",
            "private_context_unit_ids": ["du_004"],
            "owned_rui_ids": [],
        }
    )
    datapoint["private_context_units"].append(
        {"context_unit_id": "du_004", "text": "Gamma fact.", "assigned_to": ["participant_c"]}
    )
    episode = _run_fake_llm_episode(
        datapoint,
        communication_structure_preset("coordinator_select_multiple__coordinator_decision"),
        budgets=BudgetSpec(turn_budget=2),
    )

    first_turn = episode["turns"][0]
    assert first_turn["actions"][0]["target_ids"] == ["participant_a", "participant_b"]
    assert [message["speaker_id"] for message in first_turn["messages"]] == [
        "coordinator",
        "participant_a",
        "participant_b",
    ]


def test_participant_handoff_prompt_allows_next_actor(tmp_path: Path) -> None:
    datapoint = _toy_datapoint()
    client = _FakeJsonClient()
    recorder = PromptTraceRecorder(tmp_path / "prompts")

    episode = run_episode(
        datapoint,
        communication_structure_preset("no_coordinator_handoff__final_synthesis"),
        budgets=BudgetSpec(turn_budget=3),
        participant_actor=LLMParticipantActor(client, recorder),
        coordinator_actor=LLMCoordinatorActor(client, recorder),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )

    participant_prompts = [
        row
        for row in recorder.prompt_records
        if row["prompt_kind"] == "participant_response"
    ]
    first_payload = json.loads(participant_prompts[0]["user_prompt"])
    assert 'input JSON has a "chat_history" key' in participant_prompts[0]["system_prompt"]
    assert "next public response for that chat" in participant_prompts[0]["system_prompt"]
    assert "one of 2 participants working together" in participant_prompts[0]["system_prompt"]
    assert first_payload["participant_mode"] == "scheduled_turn"
    assert first_payload["turn_instruction"] == (
        "It is your turn to speak. Provide a public response using private_source_material and chat_history."
    )
    assert first_payload["response_format"]["utterance"] == "string public message"
    assert first_payload["response_format"]["recommended_next_actor_id"] == (
        "optional participant id from allowed_participant_ids"
    )
    assert first_payload["allowed_participant_ids"] == ["participant_a", "participant_b"]
    assert episode["messages"][0]["recommended_next_actor_id"] == "participant_b"


def test_run_episode_requires_explicit_actors() -> None:
    datapoint = _toy_datapoint()
    try:
        run_episode(
            datapoint,
            communication_structure_preset("coordinator_select_one__coordinator_decision"),
            budgets=BudgetSpec(turn_budget=3),
        )
    except ValueError as exc:
        assert "participant_actor" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("run_episode should require explicit actors")


def test_generated_common_datapoints_are_loadable(tmp_path: Path) -> None:
    spans = ["Condition A is satisfied.", "Condition B is satisfied.", "Condition C is satisfied."]
    seeds = [
        {
            "instance_id": f"generated_loader_{index}",
            "source_family": "loader_fixture",
            "source_id": f"source_{index}",
            "source_text": " ".join(spans),
            "question": "Are all three conditions satisfied?",
            "answer": "yes",
            "rius": [
                {"rui_id": f"rui_{unit}", "source_span": span, "normalized_claim": span, "rui_type": "fact"}
                for unit, span in enumerate(spans)
            ],
            "required_rui_ids": ["rui_0", "rui_1", "rui_2"],
        }
        for index in range(3)
    ]
    build_from_seed_specs(seed_specs=seeds, output_root=tmp_path, allocation_mode="one_each",
                          target_context_tokens=128, min_participants=3, seed=11)
    datapoints = list(iter_common_datapoints(tmp_path))
    assert len(datapoints) == 3
    assert {row["instance_id"] for row in datapoints} == {seed["instance_id"] for seed in seeds}
    assert all(datapoint["schema_version"] == "teamgym_common_datapoint_v2" for datapoint in datapoints)
    assert all(datapoint["participant_directory"] for datapoint in datapoints)
    assert all(datapoint["format_validation"]["valid"] for datapoint in datapoints)


def test_llm_actor_prompts_do_not_expose_hidden_rius_or_oracle(tmp_path: Path) -> None:
    datapoint = _toy_datapoint()
    client = _FakeJsonClient()
    recorder = PromptTraceRecorder(tmp_path / "prompts")

    episode = run_episode(
        datapoint,
        communication_structure_preset("coordinator_select_one__coordinator_decision"),
        budgets=BudgetSpec(
            turn_budget=3,
            participant_utterance_token_budget=41,
            coordinator_utterance_token_budget=42,
            final_rationale_token_budget=43,
            participant_reasoning_token_budget=21,
            coordinator_reasoning_token_budget=22,
            final_reasoning_token_budget=23,
        ),
        participant_actor=LLMParticipantActor(client, recorder),
        coordinator_actor=LLMCoordinatorActor(client, recorder),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )
    evaluation = evaluate_episode(episode, datapoint)

    assert len([row for row in episode["messages"] if row["speaker_role"] == "participant"]) == 2
    assert set(episode["public_rui_ids"]) == set()
    assert set(evaluation["reference_information_matching"]["lexically_matched_rui_ids"]) == {"rui_001", "rui_002"}
    assert episode["prompt_records"]
    assert list((tmp_path / "prompts").glob("*.prompt_record.json"))

    all_prompt_text = "\n".join(
        f"{row['system_prompt']}\n{row['user_prompt']}"
        for row in episode["prompt_records"]
    )
    hidden_tokens = [
        "rui_001",
        "rui_002",
        "owned_rui_ids",
        "reference_information_units",
        "smallest_sufficient_rui_sets",
        "\"oracle\"",
    ]
    for token in hidden_tokens:
        assert token not in all_prompt_text

    for record in episode["prompt_records"]:
        payload = json.loads(record["user_prompt"])
        task = payload["task"]
        assert "instance_id" not in task
        assert "dataset" not in task
        assert "scenario_family" not in task
        assert "dataset" not in json.dumps(task.get("initial_common_information"), ensure_ascii=False)
        assert "Shared common fact." in json.dumps(task.get("initial_common_information"), ensure_ascii=False)
        assert "coordinator_mode" not in payload
        assert "max_targets" not in payload
        if record["prompt_kind"] == "coordinator_query":
            assert "public_transcript" not in payload
            assert "chat_history" in payload
            assert all("role_name" not in row for row in payload["chat_history"])
            assert payload["response_format"]["action"] == "turn_choose"
            assert "participant_a | participant_b" in payload["response_format"]["target_ids"]
            assert 'Example: ["participant_a"]' in payload["response_format"]["target_ids"]
            assert "It is your turn to speak." in payload["turn_instruction"]
            assert "Coordinate all participants through the public chat" in payload["turn_instruction"]
            assert "Provide a public response to the last message in chat_history" in payload["turn_instruction"]
            assert "ask target participant(s) to respond next" in payload["turn_instruction"]
            assert "specific source information" not in payload["turn_instruction"]
            assert "bare final decision label" not in payload["turn_instruction"]
            assert payload["response_format"]["utterance"] == "string public coordinator message"
            assert all("role_name" not in row for row in payload["participant_directory"])
            assert "Current mode:" not in record["system_prompt"]
            assert 'input JSON has a "chat_history" key' in record["system_prompt"]
            assert "public chat you are coordinating" in record["system_prompt"]
            assert "next public coordinator response for that chat" in record["system_prompt"]
            assert "Use at most 22 private reasoning tokens" in record["system_prompt"]
            assert "under 42 tokens" in record["system_prompt"]
            assert "follow response_format exactly" in record["system_prompt"]
        elif record["prompt_kind"] == "coordinator_final_decision":
            assert payload["response_format"]["action"] == "final_decision"
            assert "turn_instruction" in payload
            assert "Current mode:" not in record["system_prompt"]
            assert "Use at most 23 private reasoning tokens" in record["system_prompt"]
            assert "under 43 tokens" in record["system_prompt"]
            assert "follow response_format exactly" in record["system_prompt"]
        elif record["prompt_kind"] == "participant_response":
            assert payload["participant_mode"] == "respond_to_request"
            assert "turn_instruction" in payload
            assert "public_transcript" not in payload
            assert "request" not in payload
            assert "allowed_participant_ids" not in payload
            assert all("role_name" not in row for row in payload["chat_history"])
            assert payload["chat_history"][-1]["speaker_id"] == "coordinator"
            assert payload["chat_history"][-1]["speaker_role"] == "coordinator"
            assert payload["chat_history"][-1]["target_ids"]
            assert "role_name" not in payload["participant"]
            assert isinstance(payload["private_source_material"], str)
            assert "source_unit_handle" not in payload["private_source_material"]
            assert "recommended_decision" not in payload["response_format"]
            assert "recommended_next_actor_id" not in payload["response_format"]
            assert payload["turn_instruction"] == (
                "It is your turn to speak. Provide a public response to the last message in chat_history."
            )
            assert payload["response_format"]["utterance"] == "string public message"
            assert "public_reasoning" not in payload["response_format"]
            assert "confidence" not in payload["response_format"]
            assert "last message in chat_history" in payload["turn_instruction"]
            assert "It is your turn to speak" in payload["turn_instruction"]
            assert 'input JSON has a "chat_history" key' in record["system_prompt"]
            assert "next public response for that chat" in record["system_prompt"]
            assert "Share concise, source-grounded information" not in record["system_prompt"]
            assert "working together through the public chat" not in record["system_prompt"]
            assert "Use at most 21 private reasoning tokens" in record["system_prompt"]
            assert "under 41 tokens" in record["system_prompt"]
            assert "follow response_format exactly" in record["system_prompt"]

    participant_a_prompt = next(
        row["user_prompt"]
        for row in episode["prompt_records"]
        if row["actor_id"] == "participant_a"
    )
    assert "Alpha fact." in participant_a_prompt
    assert "Alpha second assigned document fact." in participant_a_prompt
    assert "Beta fact." not in participant_a_prompt
    assert "Shared common fact." in participant_a_prompt
    assert "alpha reviewer" not in participant_a_prompt
    assert "beta reviewer" not in participant_a_prompt

    first_coordinator_prompt = next(
        row["user_prompt"]
        for row in episode["prompt_records"]
        if row["prompt_kind"] == "coordinator_query"
    )
    assert "Alpha fact." not in first_coordinator_prompt
    assert "Beta fact." not in first_coordinator_prompt


def test_participant_confidence_is_opt_in(tmp_path: Path) -> None:
    datapoint = _toy_datapoint()
    client = _FakeJsonClient()
    recorder = PromptTraceRecorder(tmp_path / "prompts")

    episode = run_episode(
        datapoint,
        communication_structure_preset("coordinator_select_one__coordinator_decision"),
        budgets=BudgetSpec(turn_budget=2),
        participant_actor=LLMParticipantActor(client, recorder, include_confidence=True),
        coordinator_actor=LLMCoordinatorActor(client, recorder),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )

    participant_prompt = next(
        row["user_prompt"]
        for row in recorder.prompt_records
        if row["prompt_kind"] == "participant_response"
    )
    assert "confidence" in json.loads(participant_prompt)["response_format"]
    assert any(message.get("confidence") == "high" for message in episode["messages"] if message["speaker_role"] == "participant")


def test_participant_bare_decision_response_is_kept_and_annotated(tmp_path: Path) -> None:
    datapoint = _toy_datapoint()
    client = _BareDecisionClient()
    recorder = PromptTraceRecorder(tmp_path / "prompts")

    episode = run_episode(
        datapoint,
        communication_structure_preset("coordinator_select_one__coordinator_decision"),
        budgets=BudgetSpec(turn_budget=2),
        participant_actor=LLMParticipantActor(client, recorder),
        coordinator_actor=LLMCoordinatorActor(client, recorder),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )

    participant_message = next(message for message in episode["messages"] if message["speaker_role"] == "participant")
    assert participant_message["content"] == "decision_b"
    assert participant_message["utterance_analysis"] == {"kind": "decision_id_only"}
    assert [row["prompt_kind"] for row in recorder.prompt_records if row["actor_role"] == "participant"] == [
        "participant_response",
    ]


def test_role_name_mode_can_expose_participant_ids(tmp_path: Path) -> None:
    datapoint = _toy_datapoint()
    client = _FakeJsonClient()
    recorder = PromptTraceRecorder(tmp_path / "prompts")

    run_episode(
        datapoint,
        communication_structure_preset("coordinator_select_one__coordinator_decision"),
        budgets=BudgetSpec(turn_budget=3),
        participant_actor=LLMParticipantActor(client, recorder, role_name_mode="participant_id"),
        coordinator_actor=LLMCoordinatorActor(client, recorder, role_name_mode="participant_id"),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )

    for record in recorder.prompt_records:
        payload = json.loads(record["user_prompt"])
        if record["prompt_kind"] == "coordinator_query":
            assert all(row["role_name"] == row["participant_id"] for row in payload["participant_directory"])
        elif record["prompt_kind"] == "participant_response":
            assert payload["participant"]["role_name"] == payload["participant"]["participant_id"]


def test_private_source_truncation_is_distributed_across_original_order(tmp_path: Path) -> None:
    datapoint = _toy_datapoint()
    datapoint["participants"][0]["private_context_unit_ids"] = ["du_long"]
    datapoint["private_context_units"] = [
        {
            "context_unit_id": "du_long",
            "text": " ".join(
                [
                    "FRONT_MARKER",
                    *["front filler"] * 160,
                    "MIDDLE_MARKER",
                    *["middle filler"] * 160,
                    "END_MARKER",
                ]
            ),
            "assigned_to": ["participant_a"],
        },
        {"context_unit_id": "du_002", "text": "Beta fact.", "assigned_to": ["participant_b"]},
    ]
    client = _FakeJsonClient()
    recorder = PromptTraceRecorder(tmp_path / "prompts")

    run_episode(
        datapoint,
        communication_structure_preset("coordinator_select_one__coordinator_decision"),
        budgets=BudgetSpec(turn_budget=2, participant_context_token_budget=90),
        participant_actor=LLMParticipantActor(client, recorder),
        coordinator_actor=LLMCoordinatorActor(client, recorder),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )

    participant_a_prompt = next(
        row["user_prompt"]
        for row in recorder.prompt_records
        if row["actor_id"] == "participant_a"
    )
    private_source = json.loads(participant_a_prompt)["private_source_material"]
    assert private_source.index("FRONT_MARKER") < private_source.index("MIDDLE_MARKER")
    assert private_source.index("MIDDLE_MARKER") < private_source.index("END_MARKER")
    assert "front filler " * 120 not in private_source


def test_private_source_preserves_rui_containing_original_context_unit(tmp_path: Path) -> None:
    datapoint = _toy_datapoint()
    datapoint["participants"][0]["private_context_unit_ids"] = ["du_front", "du_protected", "du_back"]
    datapoint["private_context_units"] = [
        {
            "context_unit_id": "du_front",
            "text": "FRONT_CONTEXT " + "front filler " * 120,
            "assigned_to": ["participant_a"],
        },
        {
            "context_unit_id": "du_protected",
            "text": "PROTECTED_ORIGINAL_SOURCE_CONTEXT " + "protected source words " * 20,
            "assigned_to": ["participant_a"],
            "hidden_contains_rui_ids": ["rui_secret"],
        },
        {
            "context_unit_id": "du_back",
            "text": "back filler " * 120 + "BACK_CONTEXT",
            "assigned_to": ["participant_a"],
        },
        {"context_unit_id": "du_002", "text": "Beta fact.", "assigned_to": ["participant_b"]},
    ]
    client = _FakeJsonClient()
    recorder = PromptTraceRecorder(tmp_path / "prompts")

    run_episode(
        datapoint,
        communication_structure_preset("coordinator_select_one__coordinator_decision"),
        budgets=BudgetSpec(turn_budget=2, participant_context_token_budget=120),
        participant_actor=LLMParticipantActor(client, recorder),
        coordinator_actor=LLMCoordinatorActor(client, recorder),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )

    participant_a_prompt = next(
        row["user_prompt"]
        for row in recorder.prompt_records
        if row["actor_id"] == "participant_a"
    )
    private_source = json.loads(participant_a_prompt)["private_source_material"]
    assert "PROTECTED_ORIGINAL_SOURCE_CONTEXT" in private_source
    assert "protected source words " * 10 in private_source
    assert "rui_secret" not in participant_a_prompt
    assert private_source.index("FRONT_CONTEXT") < private_source.index("PROTECTED_ORIGINAL_SOURCE_CONTEXT")


@pytest.mark.parametrize("historical_ids", [False, True])
def test_interaction_report_renders_trajectory_html(tmp_path: Path, historical_ids: bool) -> None:
    datapoint = _toy_datapoint()
    structure = communication_structure_preset("coordinator_select_one__coordinator_decision")
    historical_structure_id = "collabgym_fc_communication_structure_v1"
    if historical_ids:
        saved_structure = {**structure.to_dict(), "schema_version": historical_structure_id}
        structure = CommunicationStructure(**saved_structure)
    episode = _run_fake_llm_episode(
        datapoint,
        structure,
        budgets=BudgetSpec(turn_budget=3),
    )
    evaluation = evaluate_episode(episode, datapoint)
    assert episode["schema_version"] == "teamgym_episode"
    assert episode["communication_structure"]["schema_version"] == "teamgym_communication_structure"
    assert evaluation["schema_version"] == "teamgym_episode_evaluation"
    if historical_ids:
        episode["schema_version"] = "collabgym_fc_episode_v1"
        episode["communication_structure"]["schema_version"] = historical_structure_id
        evaluation["schema_version"] = "collabgym_fc_episode_evaluation_v1"
    run_dir = tmp_path / "run"
    trajectories_dir = run_dir / "trajectories"
    trajectories_dir.mkdir(parents=True)
    write_json(
        run_dir / "run_manifest.json",
        {
            "actor_runtime": "llm",
            "llm": {"model": "fake-json"},
            "budgets": {"turn_budget": 3},
            "prompt_tracing_enabled": True,
        },
    )
    write_json(
        run_dir / "eval_summary.json",
        {"aggregate": {"episode_count": 1, "mean_task_success": 1.0}},
    )
    write_json(
        trajectories_dir / "toy.json",
        {"episode": episode, "evaluation": evaluation},
    )

    report_path = build_report(input_path=run_dir)
    html = report_path.read_text(encoding="utf-8")

    assert report_path.name == "interaction_report.html"
    assert "Fully Collaborative Interaction Report" in html
    assert "Alpha fact matters." in html
    assert "Evaluator-Only RIU Matching" in html
    assert "Prompt Records" in html
    assert 'id="trajectoryFilter"' in html
    assert "coordinator_select_one__coordinator_decision" in html


def test_two_stage_participant_keeps_think_private_and_caps_both_fields() -> None:
    datapoint = _toy_datapoint()
    client = _TwoStageFakeJsonClient()
    recorder = PromptTraceRecorder()
    budgets = BudgetSpec(
        turn_budget=3,
        participant_reasoning_token_budget=4,
        participant_utterance_token_budget=5,
    )
    episode = run_episode(
        datapoint,
        communication_structure_preset("coordinator_select_one__coordinator_decision"),
        budgets=budgets,
        participant_actor=LLMTwoStageParticipantActor(client, recorder),
        coordinator_actor=LLMCoordinatorActor(client, recorder),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )

    think_records = [row for row in recorder.prompt_records if row["prompt_kind"] == "participant_private_think"]
    message_records = [row for row in recorder.prompt_records if row["prompt_kind"] == "participant_public_message"]
    assert len(think_records) == 2
    assert len(message_records) == 2
    for row in message_records:
        payload = json.loads(row["user_prompt"])
        assert "private_source_material" not in payload
        assert count_tokens(payload["private_think"]) <= budgets.participant_reasoning_token_budget
    for message in [row for row in episode["messages"] if row["speaker_role"] == "participant"]:
        assert "private_think" not in message
        assert message["private_think_token_count"] <= budgets.participant_reasoning_token_budget
        assert count_tokens(message["content"]) <= budgets.participant_utterance_token_budget


def test_fifo_public_reader_consumes_oldest_unread_tokens() -> None:
    reader = ParticipantPublicReader()
    content = "Alpha evidence first, then beta evidence, then gamma evidence."
    messages = [
        {
            "turn": 1,
            "speaker_id": "participant_b",
            "speaker_role": "participant",
            "content": content,
        }
    ]
    history, metrics = reader.observe(
        datapoint_id="toy",
        participant_id="participant_a",
        messages=messages,
        request=None,
        policy="fifo",
        token_budget=4,
        seed=0,
    )

    assert metrics["new_public_tokens_read"] == 4
    assert slice_tokens(content, 0, 4).strip() in history[0]["content"]
    assert history[0]["partially_observed"] is True


def test_stochastic_public_reader_is_reproducible_and_contiguous() -> None:
    content = " ".join(f"token{index}" for index in range(40))
    messages = [
        {
            "turn": 1,
            "speaker_id": "participant_b",
            "speaker_role": "participant",
            "content": content,
        }
    ]
    outputs = []
    for _ in range(2):
        reader = ParticipantPublicReader()
        outputs.append(
            reader.observe(
                datapoint_id="toy",
                participant_id="participant_a",
                messages=messages,
                request=None,
                policy="stochastic_contiguous",
                token_budget=8,
                seed=17,
            )
        )

    assert outputs[0] == outputs[1]
    visible = outputs[0][0][0]["content"]
    observed = "".join(part for part in visible.splitlines() if part != "[...]")
    assert count_tokens(observed) <= 8
    assert outputs[0][1]["new_public_tokens_read"] == 8


def test_participant_read_limit_does_not_limit_coordinator_transcript() -> None:
    datapoint = _toy_datapoint()
    client = _FakeJsonClient()
    recorder = PromptTraceRecorder()
    episode = run_episode(
        datapoint,
        communication_structure_preset("coordinator_select_one__coordinator_decision"),
        budgets=BudgetSpec(
            turn_budget=3,
            participant_public_read_policy="fifo",
            participant_public_read_token_budget=1,
            public_read_charge_current_message=True,
        ),
        participant_actor=LLMParticipantActor(client, recorder),
        coordinator_actor=LLMCoordinatorActor(client, recorder),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )

    final_record = next(
        row for row in recorder.prompt_records if row["prompt_kind"] == "coordinator_final_decision"
    )
    final_payload = json.loads(final_record["user_prompt"])
    assert [row["content"] for row in final_payload["public_transcript"]] == [
        row["content"] for row in episode["messages"]
    ]
    participant_messages = [
        row for row in episode["messages"] if row["speaker_role"] == "participant"
    ]
    assert all(row["public_read_policy"] == "fifo" for row in participant_messages)
    assert all(row["priority_public_tokens_read"] == 1 for row in participant_messages)
    assert "new_public_tokens_read" not in episode["final_decision"]


def test_strict_reader_charges_current_request_before_backlog() -> None:
    reader = ParticipantPublicReader()
    request_content = "Current directed question contains details that consume the reading allowance."
    history, metrics = reader.observe(
        datapoint_id="toy",
        participant_id="participant_a",
        messages=[
            {
                "turn": 1,
                "speaker_id": "participant_b",
                "speaker_role": "participant",
                "content": "Older public evidence should remain unread when the request uses the budget.",
            }
        ],
        request={
            "turn": 2,
            "speaker_id": "coordinator",
            "speaker_role": "coordinator",
            "content": request_content,
            "target_ids": ["participant_a"],
        },
        policy="fifo",
        token_budget=4,
        seed=0,
        charge_request=True,
    )

    assert len(history) == 1
    assert history[0]["speaker_id"] == "coordinator"
    assert slice_tokens(request_content, 0, 4).strip() in history[0]["content"]
    assert metrics["new_public_tokens_read"] == 4
    assert metrics["priority_public_tokens_read"] == 4


def test_all_actor_reading_limits_coordinator_queries_and_final_decision() -> None:
    datapoint = _toy_datapoint()
    client = _FakeJsonClient()
    recorder = PromptTraceRecorder()
    episode = run_episode(
        datapoint,
        communication_structure_preset("coordinator_select_one__coordinator_decision"),
        budgets=BudgetSpec(
            turn_budget=3,
            participant_public_read_policy="fifo",
            participant_public_read_token_budget=1,
            public_read_scope="all_actors",
            public_read_charge_current_message=True,
        ),
        participant_actor=LLMParticipantActor(client, recorder),
        coordinator_actor=LLMCoordinatorActor(client, recorder),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )

    coordinator_messages = [
        row for row in episode["messages"] if row["speaker_role"] == "coordinator"
    ]
    participant_messages = [
        row for row in episode["messages"] if row["speaker_role"] == "participant"
    ]
    assert coordinator_messages[0]["new_public_tokens_read"] == 0
    assert coordinator_messages[1]["new_public_tokens_read"] == 1
    assert all(row["priority_public_tokens_read"] == 1 for row in participant_messages)
    assert episode["final_decision"]["new_public_tokens_read"] == 1

    final_record = next(
        row for row in recorder.prompt_records if row["prompt_kind"] == "coordinator_final_decision"
    )
    final_transcript = json.loads(final_record["user_prompt"])["public_transcript"]
    assert any(
        row.get("speaker_role") == "participant" and row.get("partially_observed")
        for row in final_transcript
    )
    assert [row["content"] for row in final_transcript] != [
        row["content"] for row in episode["messages"]
    ]


def _run_fake_llm_episode(datapoint: dict, structure, *, budgets: BudgetSpec) -> dict:
    client = _FakeJsonClient()
    recorder = PromptTraceRecorder()
    return run_episode(
        datapoint,
        structure,
        budgets=budgets,
        participant_actor=LLMParticipantActor(client, recorder),
        coordinator_actor=LLMCoordinatorActor(client, recorder),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )


class _FakeJsonClient:
    def __init__(self) -> None:
        self.last_request_meta = {}

    def generate_json(
        self,
        *,
        prompt_id: str,
        system_prompt: str,
        user_prompt: str,
        trace_dir: str | Path | None = None,
    ) -> dict:
        self.last_request_meta = {"provider": "fake", "model": "fake-json", "prompt_id": prompt_id}
        if "coordinator_query" in prompt_id:
            if "one to 3 participant ids" in user_prompt:
                target_ids = ["participant_a", "participant_b"]
            elif "one to 2 participant ids" in user_prompt:
                target_ids = ["participant_a", "participant_b"]
            else:
                target_ids = ["participant_b" if "Alpha fact matters" in user_prompt else "participant_a"]
            return {
                "action": "turn_choose",
                "target_ids": target_ids,
                "utterance": "Please share the source material most relevant to the decision.",
                "reason": "Need public evidence.",
            }
        if "participant_a.participant_response" in prompt_id:
            return {
                "utterance": "Alpha fact matters.",
                "recommended_next_actor_id": "participant_b",
                "confidence": "high",
            }
        if "participant_b.participant_response" in prompt_id:
            return {
                "utterance": "Beta fact matters.",
                "confidence": "high",
            }
        return {
            "action": "final_decision",
            "answer": "decision_b",
            "rationale": "Alpha fact matters and Beta fact matters.",
            "confidence": 0.8,
        }


class _TwoStageFakeJsonClient(_FakeJsonClient):
    def generate_json(
        self,
        *,
        prompt_id: str,
        system_prompt: str,
        user_prompt: str,
        trace_dir: str | Path | None = None,
    ) -> dict:
        if "participant_private_think" in prompt_id:
            self.last_request_meta = {"provider": "fake", "model": "fake-json", "prompt_id": prompt_id}
            return {
                "think": "Alpha fact matters. Beta fact matters. Additional private analysis that must be truncated."
            }
        if "participant_public_message" in prompt_id:
            self.last_request_meta = {"provider": "fake", "model": "fake-json", "prompt_id": prompt_id}
            payload = json.loads(user_prompt)
            return {"utterance": f"Evidence from my note: {payload['private_think']} with extra public detail."}
        return super().generate_json(
            prompt_id=prompt_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_dir=trace_dir,
        )


class _BareDecisionClient:
    def __init__(self) -> None:
        self.last_request_meta = {}

    def generate_json(
        self,
        *,
        prompt_id: str,
        system_prompt: str,
        user_prompt: str,
        trace_dir: str | Path | None = None,
    ) -> dict:
        self.last_request_meta = {"provider": "fake", "model": "fake-json", "prompt_id": prompt_id}
        if "coordinator_query" in prompt_id:
            return {
                "action": "turn_choose",
                "target_ids": ["participant_a"],
                "utterance": "Please share the source material most relevant to the decision.",
                "reason": "Need public evidence.",
            }
        if "participant_a.participant_response" in prompt_id:
            return {"utterance": "decision_b"}
        return {
            "action": "final_decision",
            "answer": "decision_b",
            "rationale": "Alpha fact matters.",
            "confidence": 0.8,
        }
