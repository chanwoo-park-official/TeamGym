from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from teamgym.core.schema import SUPPORTED_COMMON_DATAPOINT_SCHEMA_VERSIONS
from teamgym.interaction.environment import run_episode
from teamgym.interaction.evaluation import evaluate_episode
from teamgym.interaction.llm_actors import (
    LLMCoordinatorActor,
    LLMFinalDecisionActor,
    LLMParticipantActor,
    PromptTraceRecorder,
)
from teamgym.interaction.structures import (
    communication_structure_preset,
)
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.workflows.workflow import Workflow

from teamgym.rl.data import (
    canonical_task_id,
    load_prepared_dataset,
    prepare_data,
)
from teamgym.rl.rollout.fixed_actors import (
    ReplayParticipantActor,
    SourceExcerptParticipantActor,
)
from teamgym.rl.rollout.profiles import get_rollout_profile
from teamgym.rl.preparation.prepare_no_redundant_legal import (
    participants_required_by_every_sufficient_set,
    prune_empty_participants,
)
from teamgym.rl.rollout.rewards import format_scores, scalarize_evaluation
from teamgym.rl.rollout.workflow import (
    FrozenOpenAIJsonClient,
    TeamGymWorkflow,
    _datapoint_from_task,
    _action_schema_error,
    _parse_exact_json_object,
    _repair_compact_action,
    _repair_participant_format_counterfactual,
)
from teamgym.rl.training.trainer_compat import (
    FORMAT_ADVANTAGE_OVERRIDE_KEY,
    _apply_format_advantage_overrides,
)


FIXTURE = Path("teamgym/rl/fixtures/tiny_normalized.jsonl")


@pytest.mark.parametrize("schema_version", SUPPORTED_COMMON_DATAPOINT_SCHEMA_VERSIONS)
def test_workflow_accepts_current_and_historical_datapoints(schema_version) -> None:
    with FIXTURE.open(encoding="utf-8") as handle:
        datapoint = json.loads(next(handle))
    datapoint["schema_version"] = schema_version
    assert _datapoint_from_task(datapoint) is datapoint
    assert _datapoint_from_task({"datapoint": datapoint}) is datapoint
    assert _datapoint_from_task({"teamgym_datapoint_json": json.dumps(datapoint)}) == datapoint


class _AsyncScriptedEngine(RolloutEngine):
    def __init__(self) -> None:
        self.calls = 0
        self.request_kwargs: list[dict] = []

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        self.request_kwargs.append(kwargs)
        self.calls += 1
        system = messages[0]["content"]
        user = messages[1]["content"]
        if "Choose which participant" in system:
            target = "participant_a" if self.calls == 1 else "participant_b"
            payload = {
                "action": "turn_choose",
                "target_ids": [target],
                "utterance": "Please provide the relevant private record.",
                "reason": "collect both records",
            }
        else:
            assert "public transcript" in system
            assert "75 dollars" in user and "60 dollars" in user
            payload = {
                "action": "final_decision",
                "answer": "reject",
                "rationale": "The cost of 75 dollars exceeds the budget of 60 dollars.",
                "confidence": 1.0,
            }
        content = json.dumps(payload)
        return ModelOutput(
            text=content,
            content="",
            reasoning=content,
            prompt_ids=[1, 2],
            completion_ids=list(range(3, 3 + len(content.split()))),
            prompt_length=2,
            completion_length=len(content.split()),
            finish_reason="stop",
        )


class _ModeRoutingScriptedEngine(RolloutEngine):
    """Return valid actor outputs while retaining request-routing metadata."""

    def __init__(self) -> None:
        self.request_kwargs: list[dict] = []

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        self.request_kwargs.append(kwargs)
        application_id = str(kwargs["application_id"])
        user = messages[1]["content"]
        if application_id.endswith("coordinator_query:retry_0"):
            target = (
                "participant_a" if ".turn_1." in application_id else "participant_b"
            )
            payload = {
                "action": "turn_choose",
                "target_ids": [target],
                "utterance": "Please report the relevant private source fact.",
                "reason": "collect the needed records",
            }
        elif ".participant_response:retry_0" in application_id:
            if "75 dollars" in user:
                utterance = "The signed project estimate gives a cost of 75 dollars."
            else:
                utterance = "The authorization memo gives a budget of 60 dollars."
            payload = {"utterance": utterance}
        else:
            payload = {
                "action": "final_decision",
                "answer": "reject",
                "rationale": "The reported cost of 75 exceeds the budget of 60.",
                "confidence": 1.0,
            }
        content = json.dumps(payload)
        return ModelOutput(
            text=content,
            content="",
            reasoning=content,
            prompt_ids=[1, 2],
            completion_ids=list(range(3, 3 + len(content.split()))),
            prompt_length=2,
            completion_length=len(content.split()),
            finish_reason="stop",
        )


class _WrittenScriptedEngine(RolloutEngine):
    def __init__(self) -> None:
        self.request_kwargs: list[dict] = []

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        self.request_kwargs.append(kwargs)
        application_id = str(kwargs["application_id"])
        user = messages[1]["content"]
        if application_id.endswith("coordinator_query:retry_0"):
            target = (
                "participant_a" if ".turn_1." in application_id else "participant_b"
            )
            payload = {
                "target": target,
                "ask": "Report your controlling private-source fact.",
            }
        elif ".participant_response:retry_0" in application_id:
            payload = {
                "utterance": (
                    "The signed project estimate gives a cost of 75 dollars."
                    if "75 dollars" in user
                    else "The authorization memo gives a budget of 60 dollars."
                )
            }
        else:
            payload = {
                "action": "final_decision",
                "answer": "written_response",
                "written_answer": (
                    "The cost is 75 dollars and exceeds the 60-dollar budget, "
                    "so the project should be rejected."
                ),
                "confidence": 1.0,
            }
        content = json.dumps(payload)
        return ModelOutput(
            text=content,
            content="",
            reasoning=content,
            prompt_ids=[1, 2],
            completion_ids=list(range(3, 3 + len(content.split()))),
            prompt_length=2,
            completion_length=len(content.split()),
            finish_reason="stop",
        )


class _WrittenRetryScriptedEngine(RolloutEngine):
    """Emit an invalid participant first attempt and an exact-JSON retry."""

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = str(kwargs["application_id"])
        user = messages[1]["content"]
        if application_id.endswith("coordinator_query:retry_0"):
            target = (
                "participant_a" if ".turn_1." in application_id else "participant_b"
            )
            content = json.dumps(
                {
                    "target": target,
                    "ask": "Report your controlling private-source fact.",
                }
            )
        elif (
            ".participant_response:retry_0" in application_id
        ):
            content = json.dumps("not an object")
        elif ".participant_response:retry_1" in application_id:
            content = json.dumps(
                {
                    "utterance": (
                        "The signed estimate gives a cost of 75 dollars."
                        if "75 dollars" in user
                        else "The authorization memo gives a budget of 60 dollars."
                    )
                }
            )
        else:
            content = json.dumps(
                {
                    "action": "final_decision",
                    "answer": "written_response",
                    "written_answer": (
                        "The cost is 75 dollars and exceeds the 60-dollar budget, "
                        "so the project should be rejected."
                    ),
                    "confidence": 1.0,
                }
            )
        return ModelOutput(
            text=content,
            content="",
            reasoning=content,
            prompt_ids=[1, 2],
            completion_ids=list(range(3, 3 + len(content.split()))),
            prompt_length=2,
            completion_length=len(content.split()),
            finish_reason="stop",
        )


class _SyncScriptedClient:
    def __init__(self) -> None:
        self.engine = _AsyncScriptedEngine()
        self.last_request_meta = {}

    def generate_json(self, *, prompt_id, system_prompt, user_prompt, trace_dir=None):
        del prompt_id, trace_dir
        output = asyncio.run(
            self.engine.get_model_response(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        )
        return json.loads(output.text or output.content or "")


class _PromptCaptureClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.last_request_meta = {}

    def generate_json(
        self, *, prompt_id, system_prompt, user_prompt, trace_dir=None
    ):
        del trace_dir
        self.calls.append(
            {
                "prompt_id": prompt_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if prompt_id.endswith("coordinator_query"):
            return {
                "target": "participant_a",
                "ask": "What exact project cost does your private source state?",
            }
        return {"utterance": "The signed estimate states a cost of 40 dollars."}


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()
    ]


def test_profiles_resolve_paper_and_smoke_budgets() -> None:
    paper = get_rollout_profile("paper_main_full_context")
    assert paper.paper_compatible is True
    assert paper.budgets.turn_budget == 16
    assert paper.budgets.participant_context_token_budget == 40_000
    assert paper.budgets.participant_utterance_token_budget == 600
    assert paper.budgets.final_reasoning_token_budget == 1_800
    matched = get_rollout_profile(
        "paper_matched_participant_opportunities", participant_count=5
    )
    assert matched.budgets.turn_budget == 6
    smoke = get_rollout_profile("short_context_smoke")
    assert get_rollout_profile("smoke_h200_qwen25") == smoke
    assert smoke.paper_compatible is False
    assert smoke.budgets.turn_budget == 3
    written = get_rollout_profile(
        "written_64_query_private_full_context", participant_count=5
    )
    assert written.budgets.turn_budget == 6
    assert written.budgets.participant_utterance_token_budget == 48
    assert written.budgets.final_rationale_token_budget == 256
    assert written.model_max_output_tokens == 384
    assert written.participant_model_max_output_tokens == 64
    assert written.coordinator_query_compact_output is True
    assert written.participant_observation_mode == "query_private_only"
    efficient = get_rollout_profile("efficient_64_coordinator", participant_count=5)
    assert efficient.budgets.turn_budget == 6
    assert efficient.budgets.coordinator_reasoning_token_budget == 0
    assert efficient.budgets.final_reasoning_token_budget == 0
    assert efficient.budgets.final_rationale_token_budget == 64
    assert efficient.model_max_output_tokens == 64
    assert efficient.participant_response_mode == "direct"
    full_context = get_rollout_profile(
        "efficient_64_full_context", participant_count=5
    )
    assert full_context.budgets.turn_budget == 6
    assert full_context.budgets.participant_context_token_budget == 28_000
    assert full_context.model_max_output_tokens == 64
    assert full_context.participant_response_mode == "direct"
    query_private = get_rollout_profile(
        "efficient_64_query_private_full_context", participant_count=5
    )
    assert query_private.budgets.turn_budget == 6
    assert query_private.budgets.participant_context_token_budget == 28_000
    assert query_private.participant_observation_mode == "query_private_only"
    participant_32 = get_rollout_profile(
        "efficient_32_participant_query_private_full_context",
        participant_count=5,
    )
    assert participant_32.budgets.turn_budget == 6
    assert participant_32.budgets.participant_utterance_token_budget == 32
    assert participant_32.budgets.coordinator_utterance_token_budget == 64
    assert participant_32.model_max_output_tokens == 64
    assert participant_32.participant_model_max_output_tokens == 32
    assert participant_32.participant_observation_mode == "query_private_only"
    retrieval_query_private = get_rollout_profile(
        "retrieval_probe_query_private_full_context", participant_count=2
    )
    assert retrieval_query_private.budgets.turn_budget == 3
    assert retrieval_query_private.model_max_output_tokens == 256
    assert (
        retrieval_query_private.participant_observation_mode
        == "query_private_only"
    )


def test_query_private_protocol_hides_task_options_and_transcript_from_participant() -> None:
    datapoint = _fixture_rows()[0]
    profile = get_rollout_profile(
        "efficient_64_query_private_full_context", participant_count=2
    )
    client = _PromptCaptureClient()
    recorder = PromptTraceRecorder()
    coordinator = LLMCoordinatorActor(
        client,
        recorder,
        compact_output=True,
        participants_query_private_only=True,
        include_compact_envelope_example=True,
    )
    state = {
        "turn": 1,
        "messages": [
            {
                "speaker_id": "participant_b",
                "speaker_role": "participant",
                "content": "This prior public message must remain hidden.",
            }
        ],
    }
    query = coordinator.choose_targets(
        datapoint=datapoint,
        state=state,
        max_targets=1,
        budgets=profile.budgets,
    )
    coordinator_call = client.calls[-1]
    coordinator_payload = json.loads(coordinator_call["user_prompt"])
    assert coordinator_payload["task"]["decision_option_definitions"]
    assert "Never refer to an option or decision label" in coordinator_call[
        "system_prompt"
    ]
    assert (
        '{"target":"<id from participant_directory>",'
        '"ask":"<self-contained source question>"}'
        in coordinator_call["system_prompt"]
    )
    assert '"target":"participant_a"' not in coordinator_call["system_prompt"]

    external_coordinator = LLMCoordinatorActor(
        client,
        recorder,
        compact_output=True,
        participants_query_private_only=True,
    )
    external_coordinator.choose_targets(
        datapoint=datapoint,
        state=state,
        max_targets=1,
        budgets=profile.budgets,
    )
    assert "Required envelope example" not in client.calls[-1]["system_prompt"]

    participant = LLMParticipantActor(
        client,
        recorder,
        observation_mode="query_private_only",
    )
    message = participant.respond(
        datapoint=datapoint,
        participant_id="participant_a",
        state=state,
        request=query,
        budgets=profile.budgets,
    )
    participant_call = client.calls[-1]
    participant_payload = json.loads(participant_call["user_prompt"])
    assert set(participant_payload) == {
        "participant_mode",
        "current_question",
        "private_source_material",
        "response_format",
    }
    assert "40 dollars" in participant_payload["private_source_material"]
    assert "approve" not in participant_call["user_prompt"]
    assert "prior public message" not in participant_call["user_prompt"]
    assert "cannot see the task, answer choices" in participant_call[
        "system_prompt"
    ]
    assert message["content"] == "The signed estimate states a cost of 40 dollars."


def test_compact_action_repair_requires_an_explicit_labeled_field() -> None:
    parsed, method = _repair_compact_action(
        '"answer": "decision_c", "Why": "source evidence agrees"',
        prompt_kind="coordinator_final_decision",
    )
    assert parsed == {"answer": "decision_c", "why": "source evidence agrees"}
    assert method == "wrapped_json"

    parsed, method = _repair_compact_action(
        '"target": "participant_b", "ask": "quote the clause',
        prompt_kind="coordinator_query",
    )
    assert parsed == {"target": "participant_b"}
    assert method == "labeled_fields"

    assert _repair_compact_action(
        "I think decision_c is best", prompt_kind="coordinator_final_decision"
    ) == (None, None)


def test_participant_format_counterfactual_preserves_raw_content() -> None:
    raw = "The source requires written consent, but the model omitted JSON."
    assert _repair_participant_format_counterfactual(
        raw, prompt_kind="participant_response"
    ) == {"utterance": raw}
    assert (
        _repair_participant_format_counterfactual(
            raw, prompt_kind="coordinator_final_decision"
        )
        is None
    )
    assert (
        _repair_participant_format_counterfactual(
            "   ", prompt_kind="participant_response"
        )
        is None
    )


def test_participant_format_counterfactual_extracts_suffix_token_budget() -> None:
    class WhitespaceTokenizer:
        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[str]:
            assert add_special_tokens is False
            return text.split()

        @staticmethod
        def decode(
            token_ids: list[str], skip_special_tokens: bool = True
        ) -> str:
            assert skip_special_tokens is True
            return " ".join(token_ids)

    raw = (
        '{"response_format":{"utterance":"zero one two three four five'
    )
    assert _repair_participant_format_counterfactual(
        raw,
        prompt_kind="participant_response",
        strategy="extract_utterance_suffix",
        max_utterance_tokens=3,
        tokenizer=WhitespaceTokenizer(),
    ) == {"utterance": "three four five"}


def test_action_schema_distinguishes_json_syntax_from_protocol_format() -> None:
    participant_payload = {
        "response_format": {"utterance": "string public message"}
    }
    assert (
        _action_schema_error(
            {"utterance": "The source requires written consent."},
            prompt_kind="participant_response",
            prompt_payload=participant_payload,
        )
        is None
    )
    assert "outside response_format" in str(
        _action_schema_error(
            {
                "response_format": {
                    "utterance": "The source requires written consent."
                }
            },
            prompt_kind="participant_response",
            prompt_payload=participant_payload,
        )
    )
    assert "non-empty" in str(
        _action_schema_error(
            {"utterance": ""},
            prompt_kind="participant_response",
            prompt_payload=participant_payload,
        )
    )
    assert "outside response_format" in str(
        _action_schema_error(
            {
                "utterance": "The source requires written consent.",
                "response_format": {"utterance": "string public message"},
            },
            prompt_kind="participant_response",
            prompt_payload=participant_payload,
        )
    )


def test_action_schema_enforces_exact_compact_protocol_values() -> None:
    coordinator_payload = {
        "participant_directory": [
            {"participant_id": "participant_a"},
            {"participant_id": "participant_b"},
        ],
        "response_format": {
            "target": "one id from participant_directory",
            "ask": "public question in at most 12 words",
        },
    }
    assert (
        _action_schema_error(
            {"target": "participant_a", "ask": "Quote the controlling clause."},
            prompt_kind="coordinator_query",
            prompt_payload=coordinator_payload,
        )
        is None
    )
    assert "participant_directory" in str(
        _action_schema_error(
            {"target": "participant_c", "ask": "Quote the controlling clause."},
            prompt_kind="coordinator_query",
            prompt_payload=coordinator_payload,
        )
    )

    final_payload = {
        "task": {"decision_set": ["decision_a", "decision_b"]},
        "response_format": {
            "answer": "one id from decision_set",
            "why": "rationale in at most 12 words",
        },
    }
    assert "decision_set" in str(
        _action_schema_error(
            {"answer": "decision_c", "why": "The evidence matches."},
            prompt_kind="coordinator_final_decision",
            prompt_payload=final_payload,
        )
    )
    assert "missing" in str(
        _action_schema_error(
            {"answer": "decision_a"},
            prompt_kind="coordinator_final_decision",
            prompt_payload=final_payload,
        )
    )


def test_exact_json_parser_rejects_recoverable_envelope_errors() -> None:
    expected = {"utterance": "The source requires consent."}
    assert _parse_exact_json_object(json.dumps(expected)) == expected
    with pytest.raises(json.JSONDecodeError):
        _parse_exact_json_object(
            "```json\n"
            + json.dumps(expected)
            + "\n```"
        )
    with pytest.raises(json.JSONDecodeError):
        _parse_exact_json_object(
            "Here is the requested JSON: " + json.dumps(expected)
        )
    with pytest.raises(ValueError, match="top-level JSON object"):
        _parse_exact_json_object(json.dumps([expected]))


def test_frozen_client_retries_schema_errors_without_reward_penalty(
    monkeypatch,
) -> None:
    class _FakeOpenAIClient:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.last_request_meta: dict = {}

        def generate_json(
            self, *, prompt_id, system_prompt, user_prompt, trace_dir=None
        ) -> dict:
            del prompt_id, user_prompt, trace_dir
            self.calls.append(system_prompt)
            self.last_request_meta = {"call": len(self.calls)}
            if len(self.calls) == 1:
                return {
                    "target": "participant_a",
                    "ask": "Report the source fact.",
                    "extra": "not allowed",
                }
            return {
                "target": "participant_a",
                "ask": "Report the source fact.",
            }

    fake = _FakeOpenAIClient()
    monkeypatch.setattr(
        "teamgym.rl.rollout.frozen_client.OpenAIJsonClient",
        lambda config: fake,
    )
    records: list[dict] = []
    client = FrozenOpenAIJsonClient(
        base_url="http://127.0.0.1:18080/v1",
        model="Qwen/Qwen2.5-3B-Instruct",
        api_key="test",
        max_output_tokens=64,
        temperature=0.0,
        timeout_sec=30,
        attempt_records=records,
        max_parse_retries=1,
    )
    response = client.generate_json(
        prompt_id="task.turn_1.coordinator_query",
        system_prompt="Return JSON.",
        user_prompt=json.dumps(
            {
                "response_format": {
                    "target": "participant_a",
                    "ask": "question",
                }
            }
        ),
    )

    assert response == {
        "target": "participant_a",
        "ask": "Report the source fact.",
    }
    assert [row["parse_status"] for row in records] == [
        "invalid_schema",
        "ok",
    ]
    assert all(row["train_mask"] is False for row in records)
    assert all(row["local_reward_penalty"] == 0.0 for row in records)
    assert "violated response_format" in fake.calls[1]


def test_frozen_client_runtime_unwrap_keeps_format_failure_visible(
    monkeypatch,
) -> None:
    nested = {
        "action": "final_decision",
        "answer": "written_response",
        "confidence": 1.0,
        "written_answer": "The reported evidence supports the conclusion.",
    }

    class _WrappedOpenAIClient:
        last_request_meta: dict = {}

        def generate_json(self, **kwargs) -> dict:
            del kwargs
            return {"response_format": nested}

    monkeypatch.setattr(
        "teamgym.rl.rollout.frozen_client.OpenAIJsonClient",
        lambda config: _WrappedOpenAIClient(),
    )
    records: list[dict] = []
    client = FrozenOpenAIJsonClient(
        base_url="http://127.0.0.1:18080/v1",
        model="Qwen/Qwen2.5-3B-Instruct",
        api_key="test",
        max_output_tokens=384,
        temperature=0.0,
        timeout_sec=30,
        attempt_records=records,
        max_parse_retries=1,
    )
    response = client.generate_json(
        prompt_id="task.turn_3.coordinator.coordinator_final_decision",
        system_prompt="Return JSON.",
        user_prompt=json.dumps(
            {
                "task": {"decision_set": ["written_response"]},
                "response_format": {
                    "action": "final_decision",
                    "answer": "one decision id from decision_set",
                    "confidence": "number between 0 and 1",
                    "written_answer": "short conclusion",
                },
            }
        ),
    )

    assert response == nested
    assert [row["parse_status"] for row in records] == [
        "invalid_schema",
        "invalid_schema",
    ]
    assert records[-1]["runtime_repair_method"] == "unwrap_response_format"
    assert all(row["local_reward_penalty"] == 0.0 for row in records)


def test_canonical_identity_ignores_allocation_suffix() -> None:
    row = _fixture_rows()[0]
    variant = deepcopy(row)
    variant["instance_id"] = "teamgym_smoke_budget_approve__partition_common_anchor"
    variant["generation_metadata"]["allocation_mode"] = "common_anchor"
    assert canonical_task_id(row) == canonical_task_id(variant)


def test_no_redundant_transform_prunes_empty_participant_and_revalidates() -> None:
    datapoint = deepcopy(_fixture_rows()[0])
    datapoint["participants"].append(
        {
            "participant_id": "participant_c",
            "role_name": "Redundant reviewer",
            "private_context_unit_ids": [],
            "owned_rui_ids": [],
        }
    )
    datapoint["participant_directory"].append(
        {
            "participant_id": "participant_c",
            "role_name": "Redundant reviewer",
        }
    )

    transformed, removed = prune_empty_participants(datapoint)

    assert removed == 1
    assert {
        row["participant_id"] for row in transformed["participants"]
    } == {"participant_a", "participant_b"}
    assert {
        row["participant_id"] for row in transformed["participant_directory"]
    } == {"participant_a", "participant_b"}
    assert all(row["owned_rui_ids"] for row in transformed["participants"])
    assert transformed["format_validation"]["valid"] is True
    assert participants_required_by_every_sufficient_set(transformed)


def test_no_redundant_transform_rejects_bypassable_participant() -> None:
    datapoint = deepcopy(_fixture_rows()[0])
    participant_a_rui = datapoint["participants"][0]["owned_rui_ids"][0]
    datapoint["oracle"]["smallest_sufficient_rui_sets"] = [
        {"rui_ids": [participant_a_rui], "answer": datapoint["oracle"]["answer"]}
    ]
    assert not participants_required_by_every_sufficient_set(datapoint)


def test_runtime_query_schedule_requires_each_target_exactly_once() -> None:
    datapoint = deepcopy(_fixture_rows()[0])
    participant_ids = [
        row["participant_id"] for row in datapoint["participant_directory"]
    ]
    assert participant_ids == ["participant_a", "participant_b"]
    episode = {
        "communication_structure": {"name": "test"},
        "turns": [
            {
                "actions": [
                    {
                        "type": "query",
                        "actor_id": "coordinator",
                        "target_ids": ["participant_b"],
                    }
                ]
            },
            {
                "actions": [
                    {
                        "type": "query",
                        "actor_id": "coordinator",
                        "target_ids": ["participant_a"],
                    }
                ]
            },
        ],
        "messages": [],
        "final_decision": {"answer": datapoint["oracle"]["answer"]},
    }
    evaluation = evaluate_episode(episode, datapoint)
    assert evaluation["participation_eval"]["query_target_ids"] == [
        "participant_b",
        "participant_a",
    ]
    assert evaluation["participation_eval"]["complete_query_schedule"] is True
    assert evaluation["participation_eval"]["complete_participant_probe"] is False

    episode["turns"][1]["actions"][0]["target_ids"] = ["participant_b"]
    evaluation = evaluate_episode(episode, datapoint)
    assert evaluation["participation_eval"]["complete_query_schedule"] is False


def test_prepare_groups_variants_without_split_leakage(tmp_path: Path) -> None:
    rows = _fixture_rows()
    variant = deepcopy(rows[0])
    variant["instance_id"] = "teamgym_smoke_budget_approve__partition_common_anchor"
    variant["generation_metadata"]["allocation_mode"] = "common_anchor"
    source = tmp_path / "input.jsonl"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in [*rows, variant]), encoding="utf-8"
    )
    result = prepare_data(
        input_paths=source,
        output_dir=tmp_path / "prepared",
        seed=17,
        validation_fraction=0.25,
        test_fraction=0.25,
    )
    assert result["canonical_task_counts"] == {"train": 2, "validation": 1, "test": 1}
    manifest = [
        json.loads(line)
        for line in Path(result["manifest_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    variant_splits = {
        row["split"]
        for row in manifest
        if row["canonical_task_id"] == "teamgym_smoke_budget_approve"
    }
    assert len(variant_splits) == 1
    audit = json.loads(Path(result["leakage_audit_path"]).read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert not any(audit["overlaps"].values())


def test_prepare_source_groups_interleaves_and_preserves_unicode_lines(
    tmp_path: Path,
) -> None:
    rows = _fixture_rows()
    for index, row in enumerate(rows):
        dataset = "cuad" if index % 2 == 0 else "maud"
        row["dataset_specific"]["benchmark_dataset"] = dataset
        row["_cross"] = {
            "canonical_task_id": f"{dataset}_task_{index}",
            "source_group_id": f"source_{dataset}",
        }
    rows[0]["private_context_units"][0]["text"] += "\u2028retained separator"
    source = tmp_path / "source_group_input.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = prepare_data(
        input_paths=source,
        output_dir=tmp_path / "source_group_prepared",
        seed=17,
        validation_fraction=0.0,
        test_fraction=0.0,
        split_group="source_group",
        interleave_datasets=True,
    )
    assert result["split_group_counts"] == {
        "train": 2,
        "validation": 0,
        "test": 0,
    }
    loaded = load_prepared_dataset(result["output_dir"], "train")
    assert [row["dataset"] for row in loaded] == ["cuad", "maud", "cuad", "maud"]
    decoded = [json.loads(row["teamgym_datapoint_json"]) for row in loaded]
    assert any(
        "\u2028retained separator" in context["text"]
        for row in decoded
        for context in row["private_context_units"]
    )


def test_rl_bridge_matches_direct_paper_runtime(tmp_path: Path, monkeypatch) -> None:
    datapoint = _fixture_rows()[1]
    task = {
        "instance_id": datapoint["instance_id"],
        "canonical_task_id": canonical_task_id(datapoint),
        "teamgym_datapoint_json": json.dumps(datapoint),
    }
    engine = _AsyncScriptedEngine()
    monkeypatch.setenv("HYDRA_RUN_DIR", str(tmp_path / "hydra"))
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = TeamGymWorkflow(
            rollout_engine=engine,
            executor=executor,
            rollout_profile="short_context_smoke",
            role_training_mode="coordinator_only",
            rollout_lora_name="default",
            participant_backend="scripted",
            reward_config="task_success_plus_informed_success",
            artifact_dir=str(tmp_path / "artifacts"),
        )
        rl_episode = asyncio.run(workflow.run(task, "task:0"))

    client = _SyncScriptedClient()
    recorder = PromptTraceRecorder()
    profile = get_rollout_profile("short_context_smoke")
    direct = run_episode(
        datapoint,
        communication_structure_preset(profile.communication_structure),
        budgets=profile.budgets,
        participant_actor=SourceExcerptParticipantActor(),
        coordinator_actor=LLMCoordinatorActor(
            client, recorder, role_name_mode="hidden"
        ),
        final_actor=LLMFinalDecisionActor(client, recorder),
    )
    bridged = rl_episode.info["teamgym"]["paper_episode"]
    direct.pop("created_at_utc")
    bridged.pop("created_at_utc")
    for episode in (direct, bridged):
        for record in episode["prompt_records"]:
            record["request_meta"] = {}
    assert bridged == direct
    assert rl_episode.is_correct is True
    assert rl_episode.metrics["teamgym_trainable_step_count"] == 3.0
    assert all(
        request["lora_request"]["lora_path"]
        == str((tmp_path / "hydra" / "default").resolve())
        for request in engine.request_kwargs
    )
    assert all(
        step.info["train_mask"] is True for step in rl_episode.trajectories[0].steps
    )
    prompt_text = "\n".join(
        record["system_prompt"] + record["user_prompt"]
        for record in rl_episode.info["teamgym"]["generation_attempts"]
    ).casefold()
    assert '"oracle"' not in prompt_text
    assert "rui_cost" not in prompt_text
    assert "smallest_sufficient" not in prompt_text
    artifact = next((tmp_path / "artifacts").glob("*.json"))
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["paper_episode"]["schema_version"] == "teamgym_episode"
    assert saved["evaluation"] == rl_episode.info["teamgym"]["evaluation"]


def test_participant_only_routes_fixed_coordinator_to_base(
    tmp_path: Path, monkeypatch
) -> None:
    datapoint = _fixture_rows()[1]
    task = {
        "instance_id": datapoint["instance_id"],
        "canonical_task_id": canonical_task_id(datapoint),
        "teamgym_datapoint_json": json.dumps(datapoint),
    }
    engine = _ModeRoutingScriptedEngine()
    monkeypatch.setenv("HYDRA_RUN_DIR", str(tmp_path / "hydra"))
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = TeamGymWorkflow(
            rollout_engine=engine,
            executor=executor,
            rollout_profile="short_context_smoke",
            role_training_mode="participant_only",
            rollout_lora_name="default",
            participant_backend="rollout",
            fixed_coordinator_backend="base_model",
            reward_config="task_success_plus_informed_success",
        )
        episode = asyncio.run(workflow.run(task, "participant-only:0"))

    assert episode.is_correct is True
    assert episode.metrics["teamgym_trainable_step_count"] == 2.0
    assert episode.metrics["teamgym_frozen_model_step_count"] == 3.0
    trainable, fixed = episode.trajectories
    assert trainable.name == "default"
    assert fixed.name == "__base__"
    assert all(step.info["actor_role"] == "participant" for step in trainable.steps)
    assert all(step.info["train_mask"] is True for step in trainable.steps)
    assert all(step.info["actor_role"] == "coordinator" for step in fixed.steps)
    assert all(step.info["train_mask"] is False for step in fixed.steps)
    assert all(step.info["lora_name"] == "__base__" for step in fixed.steps)
    attempts = episode.info["teamgym"]["generation_attempts"]
    for request, attempt in zip(engine.request_kwargs, attempts, strict=True):
        if attempt["actor_role"] == "participant":
            assert "lora_request" in request
        else:
            assert "lora_request" not in request


def test_participant_32_profile_caps_only_participant_completions(
    tmp_path: Path, monkeypatch
) -> None:
    datapoint = _fixture_rows()[1]
    task = {
        "instance_id": datapoint["instance_id"],
        "canonical_task_id": canonical_task_id(datapoint),
        "teamgym_datapoint_json": json.dumps(datapoint),
    }
    engine = _ModeRoutingScriptedEngine()
    monkeypatch.setenv("HYDRA_RUN_DIR", str(tmp_path / "hydra"))
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = TeamGymWorkflow(
            rollout_engine=engine,
            executor=executor,
            rollout_profile=(
                "efficient_32_participant_query_private_full_context"
            ),
            role_training_mode="participant_only",
            rollout_lora_name="default",
            participant_backend="rollout",
            fixed_coordinator_backend="base_model",
            reward_config="task_success",
        )
        episode = asyncio.run(workflow.run(task, "participant-32:0"))

    attempts = episode.info["teamgym"]["generation_attempts"]
    assert len(engine.request_kwargs) == len(attempts)
    for request, attempt in zip(engine.request_kwargs, attempts, strict=True):
        assert request["max_tokens"] == (
            32 if attempt["actor_role"] == "participant" else 64
        )


def test_full_model_route_uses_no_rollout_adapter(tmp_path: Path) -> None:
    datapoint = _fixture_rows()[1]
    task = {
        "instance_id": datapoint["instance_id"],
        "canonical_task_id": canonical_task_id(datapoint),
        "teamgym_datapoint_json": json.dumps(datapoint),
    }
    engine = _AsyncScriptedEngine()
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = TeamGymWorkflow(
            rollout_engine=engine,
            executor=executor,
            rollout_profile="short_context_smoke",
            role_training_mode="coordinator_only",
            rollout_lora_name=None,
            participant_backend="scripted",
            reward_config="task_success_plus_informed_success",
            artifact_dir=str(tmp_path / "artifacts"),
        )
        episode = asyncio.run(workflow.run(task, "full-model:0"))

    assert episode.is_correct is True
    assert episode.trajectories[0].name == "__full__"
    assert all(
        step.info["policy_route"] == "trainable_full_model"
        for step in episode.trajectories[0].steps
    )
    assert all(step.info["lora_name"] == "__full__" for step in episode.trajectories[0].steps)
    assert all("lora_request" not in request for request in engine.request_kwargs)
    assert all("lora_name" not in request for request in engine.request_kwargs)


def test_full_model_participant_only_rejects_false_inprocess_frozen_route() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ValueError, match="external_openai"):
            TeamGymWorkflow(
                rollout_engine=_ModeRoutingScriptedEngine(),
                executor=executor,
                rollout_profile="short_context_smoke",
                role_training_mode="participant_only",
                rollout_lora_name=None,
                participant_backend="rollout",
                fixed_coordinator_backend="base_model",
                reward_config="task_success_plus_informed_success",
            )


def test_full_model_participant_only_uses_external_frozen_coordinator_and_final(
    monkeypatch,
) -> None:
    datapoint = _fixture_rows()[1]
    task = {
        "instance_id": datapoint["instance_id"],
        "canonical_task_id": canonical_task_id(datapoint),
        "teamgym_datapoint_json": json.dumps(datapoint),
    }

    class _FakeFrozenClient:
        def __init__(self, *, attempt_records, **kwargs) -> None:
            del kwargs
            self.attempt_records = attempt_records
            self.last_request_meta: dict = {}

        def generate_json(
            self, *, prompt_id, system_prompt, user_prompt, trace_dir=None
        ) -> dict:
            del system_prompt, trace_dir
            if prompt_id.endswith("coordinator_query"):
                target = (
                    "participant_a"
                    if ".turn_1." in prompt_id
                    else "participant_b"
                )
                response = {
                    "target": target,
                    "ask": "Report the relevant private source fact.",
                }
            else:
                response = {
                    "answer": "reject",
                    "why": "The reported cost exceeds the reported budget.",
                }
            self.attempt_records.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_kind": prompt_id.rsplit(".", 1)[-1],
                    "actor_role": "coordinator",
                    "parse_status": "ok",
                    "train_mask": False,
                    "policy_route": "frozen_external_base",
                    "raw_output": json.dumps(response),
                    "system_prompt": "",
                    "user_prompt": user_prompt,
                    "parsed_response": response,
                }
            )
            return response

    monkeypatch.setattr(
        "teamgym.rl.rollout.workflow.FrozenOpenAIJsonClient",
        _FakeFrozenClient,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = TeamGymWorkflow(
            rollout_engine=_ModeRoutingScriptedEngine(),
            executor=executor,
            rollout_profile="short_context_smoke",
            role_training_mode="participant_only",
            rollout_lora_name=None,
            participant_backend="rollout",
            fixed_coordinator_backend="external_openai",
            fixed_coordinator_base_url="http://127.0.0.1:18080/v1",
            fixed_coordinator_model="Qwen/Qwen2.5-3B-Instruct",
            reward_config="task_success_plus_informed_success",
        )
        episode = asyncio.run(workflow.run(task, "external-participant-only:0"))

    assert episode.is_correct is True
    assert len(episode.trajectories) == 1
    assert episode.trajectories[0].name == "__full__"
    assert {
        step.info["actor_role"] for step in episode.trajectories[0].steps
    } == {"participant"}
    attempts = episode.info["teamgym"]["generation_attempts"]
    external = [
        row for row in attempts if row.get("policy_route") == "frozen_external_base"
    ]
    assert len(external) == 3
    assert any(
        row.get("prompt_kind") == "coordinator_final_decision"
        for row in external
    )
    assert all(row.get("train_mask") is False for row in external)


def test_full_model_coordinator_only_uses_external_frozen_participants(
    monkeypatch,
) -> None:
    datapoint = _fixture_rows()[1]
    task = {
        "instance_id": datapoint["instance_id"],
        "canonical_task_id": canonical_task_id(datapoint),
        "teamgym_datapoint_json": json.dumps(datapoint),
    }

    class _FakeFrozenParticipantClient:
        def __init__(self, *, attempt_records, policy_route, **kwargs) -> None:
            del kwargs
            assert policy_route == "frozen_external_participant"
            self.attempt_records = attempt_records
            self.last_request_meta: dict = {}

        def generate_json(
            self, *, prompt_id, system_prompt, user_prompt, trace_dir=None
        ) -> dict:
            del system_prompt, trace_dir
            utterance = (
                "The signed project estimate gives a cost of 75 dollars."
                if "75 dollars" in user_prompt
                else "The authorization memo gives a budget of 60 dollars."
            )
            response = {"utterance": utterance}
            self.attempt_records.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_kind": "participant_response",
                    "actor_role": "participant",
                    "parse_status": "ok",
                    "train_mask": False,
                    "policy_route": "frozen_external_participant",
                    "raw_output": json.dumps(response),
                    "system_prompt": "",
                    "user_prompt": user_prompt,
                    "parsed_response": response,
                }
            )
            return response

    monkeypatch.setattr(
        "teamgym.rl.rollout.workflow.FrozenOpenAIJsonClient",
        _FakeFrozenParticipantClient,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = TeamGymWorkflow(
            rollout_engine=_ModeRoutingScriptedEngine(),
            executor=executor,
            rollout_profile="short_context_smoke",
            role_training_mode="coordinator_only",
            rollout_lora_name=None,
            participant_backend="external_openai",
            fixed_participant_base_url="http://127.0.0.1:18080/v1",
            fixed_participant_model="/scratch/stage-p-hf",
            reward_config="task_success_plus_informed_success",
        )
        episode = asyncio.run(workflow.run(task, "external-coordinator-only:0"))

    assert episode.is_correct is True
    assert len(episode.trajectories) == 1
    assert {
        step.info["actor_role"] for step in episode.trajectories[0].steps
    } == {"coordinator"}
    external = [
        row
        for row in episode.info["teamgym"]["generation_attempts"]
        if row.get("policy_route") == "frozen_external_participant"
    ]
    assert len(external) == 2
    assert all(row.get("train_mask") is False for row in external)


def test_joint_routes_both_roles_to_one_trainable_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    datapoint = _fixture_rows()[1]
    task = {
        "instance_id": datapoint["instance_id"],
        "canonical_task_id": canonical_task_id(datapoint),
        "teamgym_datapoint_json": json.dumps(datapoint),
    }
    engine = _ModeRoutingScriptedEngine()
    monkeypatch.setenv("HYDRA_RUN_DIR", str(tmp_path / "hydra"))
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = TeamGymWorkflow(
            rollout_engine=engine,
            executor=executor,
            rollout_profile="short_context_smoke",
            role_training_mode="joint",
            rollout_lora_name="default",
            participant_backend="rollout",
            reward_config="task_success_plus_informed_success",
        )
        episode = asyncio.run(workflow.run(task, "joint:0"))

    assert episode.is_correct is True
    assert episode.metrics["teamgym_trainable_step_count"] == 5.0
    assert episode.metrics["teamgym_frozen_model_step_count"] == 0.0
    assert len(episode.trajectories) == 1
    assert {step.info["actor_role"] for step in episode.trajectories[0].steps} == {
        "coordinator",
        "participant",
    }
    assert all("lora_request" in request for request in engine.request_kwargs)


def test_reward_scalarization_preserves_vector() -> None:
    evaluation = {
        "primary_eval": {"task_success": True, "informed_success": False},
        "diagnostic": {"untouched": 7},
    }
    before = deepcopy(evaluation)
    reward, components = scalarize_evaluation(
        evaluation, "task_success_plus_informed_success"
    )
    assert reward == 0.5
    assert components == {"outcome_success": 1.0, "informed_success": 0.0}
    assert evaluation == before


def test_written_holistic_score_is_the_base_reward() -> None:
    evaluation = {"primary_eval": {"task_success": True}}
    judgment = {"judgment": {"score": 0.63, "holistic_passed": False}}
    reward, components = scalarize_evaluation(
        evaluation,
        "written_holistic_score",
        written_judgment=judgment,
    )
    assert reward == 0.63
    assert components == {"holistic_score": 0.63}


def test_format_score_uses_exact_schema_success_after_retry() -> None:
    attempts = [
        {
            "prompt_id": "participant_a",
            "retry_number": 0,
            "parse_status": "invalid_json",
            "train_mask": True,
        },
        {
            "prompt_id": "participant_a",
            "retry_number": 1,
            "parse_status": "ok",
            "train_mask": True,
        },
        {
            "prompt_id": "participant_b",
            "retry_number": 0,
            "parse_status": "invalid_schema",
            "train_mask": True,
        },
        {
            "prompt_id": "fixed_coordinator",
            "retry_number": 0,
            "parse_status": "ok",
            "train_mask": False,
        },
    ]
    assert format_scores(attempts, trainable_only=True) == {
        "invocation_count": 2.0,
        "format_score": 0.5,
        "first_pass_format_score": 0.0,
    }
    assert format_scores(attempts, trainable_only=False) == {
        "invocation_count": 3.0,
        "format_score": pytest.approx(2 / 3),
        "first_pass_format_score": pytest.approx(1 / 3),
    }


def test_per_attempt_format_override_replaces_only_invalid_advantage() -> None:
    class _Batch:
        batch = {
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "advantages": torch.tensor(
                [[0.7, 0.7, 0.0], [-0.4, -0.4, -0.4]]
            ),
            "returns": torch.tensor(
                [[0.7, 0.7, 0.0], [-0.4, -0.4, -0.4]]
            ),
            FORMAT_ADVANTAGE_OVERRIDE_KEY: torch.tensor(
                [-0.1, float("nan")]
            ),
        }

    batch = _Batch()
    _apply_format_advantage_overrides(batch)

    torch.testing.assert_close(
        batch.batch["advantages"],
        torch.tensor([[-0.1, -0.1, 0.0], [-0.4, -0.4, -0.4]]),
    )
    torch.testing.assert_close(
        batch.batch["returns"],
        torch.tensor([[-0.1, -0.1, 0.0], [-0.4, -0.4, -0.4]]),
    )


def test_written_workflow_uses_judge_score_and_logs_format(
    tmp_path: Path, monkeypatch
) -> None:
    datapoint = deepcopy(_fixture_rows()[1])
    datapoint["task_description"] = {
        "title": "Written budget characterization",
        "question": "Compare the private cost and budget and state the result.",
        "objective": "Provide a grounded written conclusion.",
        "answer_type": "short_answer_judged",
    }
    datapoint["decision_set"] = ["written_response"]
    datapoint["decision_option_definitions"] = {
        "written_response": "Compatibility transport value."
    }
    datapoint["oracle"] = {
        "answer": "written_response",
        "canonical_response": (
            "The 75-dollar cost exceeds the 60-dollar budget, so reject."
        ),
        "required_decision_steps": [
            {"step_id": "cost", "claim": "Cost is 75.", "uses": ["rui_cost"]},
            {"step_id": "budget", "claim": "Budget is 60.", "uses": ["rui_budget"]},
            {
                "step_id": "synthesis",
                "claim": "Cost exceeds budget, so reject.",
                "uses": ["rui_cost", "rui_budget"],
            },
        ],
        "judge_rubric": {"require_all_steps_for_pass": True},
    }
    task = {
        "instance_id": datapoint["instance_id"],
        "canonical_task_id": canonical_task_id(datapoint),
        "teamgym_datapoint_json": json.dumps(datapoint),
    }

    def fake_judge(**kwargs):
        assert kwargs["episode"]["final_decision"]["written_answer"]
        return {
            "judgment": {
                "score": 0.63,
                "holistic_passed": False,
                "strict_all_step_passed": False,
                "conclusion_correct": True,
                "covered_step_count": 3,
                "total_step_count": 3,
                "disqualifying_errors_found": [],
            },
            "cache_hit": False,
        }

    monkeypatch.setattr(
        "teamgym.rl.rollout.workflow.judge_written_episode", fake_judge
    )
    engine = _WrittenScriptedEngine()
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = TeamGymWorkflow(
            rollout_engine=engine,
            executor=executor,
            rollout_profile="written_64_query_private_full_context",
            role_training_mode="joint",
            rollout_lora_name="default",
            participant_backend="rollout",
            reward_config="written_holistic_score",
        )
        episode = asyncio.run(workflow.run(task, "written:0"))

    assert episode.trajectories[0].reward == 0.63
    assert episode.metrics["teamgym_holistic_score"] == 0.63
    assert episode.metrics["teamgym_written_judge_api_error"] == 0.0
    assert episode.metrics["teamgym_required_step_coverage"] == 1.0
    assert episode.metrics["teamgym_task_success"] == 0.0
    assert episode.metrics["teamgym_strict_success"] == 0.0
    assert episode.metrics["teamgym_format_score"] == 1.0
    assert episode.metrics["teamgym_first_pass_format_score"] == 1.0
    assert episode.info["teamgym"]["written_answer_judgment"]["judgment"][
        "score"
    ] == 0.63

    def failed_judge(**kwargs):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(
        "teamgym.rl.rollout.workflow.judge_written_episode", failed_judge
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = TeamGymWorkflow(
            rollout_engine=_WrittenScriptedEngine(),
            executor=executor,
            rollout_profile="written_64_query_private_full_context",
            role_training_mode="joint",
            rollout_lora_name="default",
            participant_backend="rollout",
            reward_config="written_holistic_score",
            written_judge_fail_on_error=True,
        )
        with pytest.raises(
            RuntimeError, match="Written-answer judge failed after configured retries"
        ):
            asyncio.run(workflow.run(task, "written:judge-failure"))


def test_written_workflow_assigns_invalid_attempt_and_retry_separately(
    tmp_path: Path, monkeypatch
) -> None:
    datapoint = deepcopy(_fixture_rows()[1])
    datapoint["task_description"] = {
        "title": "Written budget characterization",
        "question": "Compare the private cost and budget and state the result.",
        "objective": "Provide a grounded written conclusion.",
        "answer_type": "short_answer_judged",
    }
    datapoint["decision_set"] = ["written_response"]
    datapoint["decision_option_definitions"] = {
        "written_response": "Compatibility transport value."
    }
    datapoint["oracle"] = {
        "answer": "written_response",
        "canonical_response": (
            "The 75-dollar cost exceeds the 60-dollar budget, so reject."
        ),
        "required_decision_steps": [
            {"step_id": "cost", "claim": "Cost is 75.", "uses": ["rui_cost"]},
            {"step_id": "budget", "claim": "Budget is 60.", "uses": ["rui_budget"]},
            {
                "step_id": "synthesis",
                "claim": "Cost exceeds budget, so reject.",
                "uses": ["rui_cost", "rui_budget"],
            },
        ],
        "judge_rubric": {"require_all_steps_for_pass": True},
    }
    task = {
        "instance_id": datapoint["instance_id"],
        "canonical_task_id": canonical_task_id(datapoint),
        "teamgym_datapoint_json": json.dumps(datapoint),
    }

    monkeypatch.setattr(
        "teamgym.rl.rollout.workflow.judge_written_episode",
        lambda **kwargs: {
            "judgment": {
                "score": 0.63,
                "holistic_passed": False,
                "conclusion_correct": True,
                "covered_step_count": 2,
                "total_step_count": 3,
                "disqualifying_errors_found": [],
            },
            "cache_hit": False,
        },
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = TeamGymWorkflow(
            rollout_engine=_WrittenRetryScriptedEngine(),
            executor=executor,
            rollout_profile="written_64_query_private_full_context",
            role_training_mode="joint",
            rollout_lora_name="default",
            participant_backend="rollout",
            reward_config="written_holistic_score",
            max_parse_retries=1,
            invalid_json_penalty=-0.1,
            invalid_schema_penalty=-0.1,
            format_credit_assignment="per_attempt",
            artifact_dir=str(tmp_path),
        )
        episode = asyncio.run(workflow.run(task, "written-retry:0"))

    trainable_steps = [
        step
        for trajectory in episode.trajectories
        for step in trajectory.steps
        if step.info.get("train_mask") is not False
    ]
    invalid_steps = [
        step
        for step in trainable_steps
        if step.info["parse_status"] in {"invalid_json", "invalid_schema"}
    ]
    retry_steps = [
        step
        for step in trainable_steps
        if step.info["retry_number"] == 1 and step.info["parse_status"] == "ok"
    ]
    assert invalid_steps
    assert retry_steps
    assert all(step.reward == -0.1 for step in invalid_steps)
    assert all(step.reward == 0.63 for step in retry_steps)
    assert episode.trajectories[0].reward == 0.63
    assert episode.metrics["teamgym_reward"] == 0.63
    assert episode.metrics["teamgym_local_penalty"] == pytest.approx(
        -0.1 * len(invalid_steps)
    )
    assert episode.metrics["teamgym_penalized_score"] == pytest.approx(
        0.63 - 0.1 * len(invalid_steps)
    )
    assert episode.info["teamgym"]["format_credit_assignment"] == "per_attempt"


def test_complete_probe_reward_requires_correctness_and_exact_coverage() -> None:
    evaluation = {
        "primary_eval": {"task_success": True, "informed_success": False},
        "participation_eval": {"complete_participant_probe": True},
    }
    reward, components = scalarize_evaluation(
        evaluation, "task_success_plus_complete_probe_success"
    )
    assert reward == 1.0
    assert components == {
        "outcome_success": 1.0,
        "complete_probe_success": 1.0,
    }

    evaluation["primary_eval"]["task_success"] = False
    reward, components = scalarize_evaluation(
        evaluation, "task_success_plus_complete_probe_success"
    )
    assert reward == 0.0
    assert components["complete_probe_success"] == 0.0


def test_replay_participant_returns_frozen_message(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "messages": {
                    "task_1": {
                        "participant_a": "Payment is due within thirty days."
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    actor = ReplayParticipantActor(replay_path)
    message = actor.respond(
        datapoint={"instance_id": "task_1"},
        participant_id="participant_a",
        state={},
        request=None,
        budgets=get_rollout_profile("efficient_64_full_context").budgets,
    )

    assert message["content"] == "Payment is due within thirty days."
    assert message["fixed_actor_kind"] == "participant_replay_v1"


def test_workflow_executor_falls_back_when_shared_pool_is_closed() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    executor.shutdown()
    workflow_stub = type("_WorkflowStub", (), {"executor": executor})()

    result = asyncio.run(
        Workflow.run_in_executor(workflow_stub, lambda value: value + 1, 6)
    )

    assert result == 7
