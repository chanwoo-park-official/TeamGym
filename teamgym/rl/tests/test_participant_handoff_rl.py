from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine

from teamgym.rl.data import canonical_task_id
from teamgym.rl.rollout.profiles import get_rollout_profile
from teamgym.rl.rollout.workflow import TeamGymWorkflow, _action_schema_error


FIXTURE = Path("teamgym/rl/fixtures/tiny_normalized.jsonl")


def _datapoint() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])


def _participant_prompt_payload() -> dict:
    return {
        "allowed_participant_ids": ["participant_a", "participant_b"],
        "response_format": {
            "action": "handoff or finalize",
            "utterance": "string public message",
            "next_actor_id": (
                "optional participant id; required exactly when action is handoff"
            ),
        },
    }


def test_strict_participant_handoff_schema() -> None:
    payload = _participant_prompt_payload()
    assert _action_schema_error(
        {
            "action": "handoff",
            "utterance": "Please check the budget.",
            "next_actor_id": "participant_b",
        },
        prompt_kind="participant_response",
        prompt_payload=payload,
    ) is None
    missing = _action_schema_error(
        {"action": "handoff", "utterance": "Please check the budget."},
        prompt_kind="participant_response",
        prompt_payload=payload,
    )
    assert missing and "next_actor_id" in missing
    unknown = _action_schema_error(
        {
            "action": "handoff",
            "utterance": "Please check the budget.",
            "next_actor_id": "not_a_participant",
        },
        prompt_kind="participant_response",
        prompt_payload=payload,
    )
    assert unknown and "allowed_participant_ids" in unknown
    assert _action_schema_error(
        {"action": "finalize", "utterance": "We have enough information."},
        prompt_kind="participant_response",
        prompt_payload=payload,
    ) is None


class _HandoffEngine(RolloutEngine):
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.prompts: list[dict] = []

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        self.requests.append(kwargs)
        payload = json.loads(messages[1]["content"])
        self.prompts.append(payload)
        application_id = str(kwargs["application_id"])
        if ".turn_1." in application_id and application_id.endswith(
            "participant_response:retry_0"
        ):
            response = {
                "action": "handoff",
                "utterance": "I found my value; ask the other reviewer.",
                "next_actor_id": "unknown_participant",
            }
        elif ".turn_1." in application_id:
            current = str(payload["participant"]["participant_id"])
            other = next(
                row["participant_id"]
                for row in payload["participant_directory"]
                if row["participant_id"] != current
            )
            response = {
                "action": "handoff",
                "utterance": "I found my value; ask the other reviewer.",
                "next_actor_id": other,
            }
        elif application_id.endswith("participant_response:retry_0"):
            response = {
                "action": "finalize",
                "utterance": "Both values show that the project is within budget.",
            }
        else:
            response = {
                "action": "final_decision",
                "answer": "approve",
                "rationale": "The cost of 40 dollars is within the 50-dollar budget.",
                "confidence": 1.0,
            }
        text = json.dumps(response)
        return ModelOutput(
            text=text,
            content="",
            reasoning=text,
            prompt_ids=[1, 2],
            completion_ids=list(range(len(text.split()))),
            prompt_length=2,
            completion_length=len(text.split()),
            finish_reason="stop",
        )


def test_participant_only_handoff_workflow_penalizes_invalid_actor_and_trains_final(
    tmp_path: Path,
) -> None:
    datapoint = _datapoint()
    task = {
        "instance_id": datapoint["instance_id"],
        "canonical_task_id": canonical_task_id(datapoint),
        "teamgym_datapoint_json": json.dumps(datapoint),
    }
    engine = _HandoffEngine()
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = TeamGymWorkflow(
            rollout_engine=engine,
            executor=executor,
            rollout_profile="finance_riu_participant_handoff_15",
            role_training_mode="participant_only",
            rollout_lora_name=None,
            participant_backend="rollout",
            fixed_coordinator_backend="base_model",
            max_parse_retries=1,
            invalid_schema_penalty=-0.1,
            format_credit_assignment="per_attempt",
            reward_config="task_success",
            artifact_dir=str(tmp_path / "artifacts"),
        )
        episode = asyncio.run(workflow.run(task, "handoff:0"))

    assert episode.is_correct is True
    assert len(episode.trajectories) == 1
    steps = episode.trajectories[0].steps
    assert len(steps) == 4
    assert all(step.info["actor_role"] == "participant" for step in steps)
    assert all(step.info["train_mask"] is True for step in steps)
    assert steps[0].info["parse_status"] == "invalid_schema"
    assert steps[0].reward == -0.1
    assert steps[-1].info["prompt_kind"] == "participant_final_decision"
    assert all(request["max_tokens"] == 750 for request in engine.requests)

    view = episode.info["teamgym"]
    paper_episode = view["paper_episode"]
    assert paper_episode["stopped_reason"] == "participant_requested_finalize"
    assert len(paper_episode["messages"]) == 2
    assert paper_episode["final_decision"]["actor_id"] == paper_episode["messages"][-1]["speaker_id"]
    assert view["participant_handoff_dynamics"]["invalid_next_actor_attempt_count"] == 1
    assert episode.metrics["teamgym_invalid_next_actor_count"] == 1.0
    for prompt in engine.prompts[:3]:
        assert {row["role_name"] for row in prompt["participant_directory"]} == {
            "Cost reviewer",
            "Budget reviewer",
        }
    second_turn = next(
        prompt
        for prompt in engine.prompts
        if prompt.get("chat_history")
    )
    assert second_turn["chat_history"][0]["role_name"] in {
        "Cost reviewer",
        "Budget reviewer",
    }


def test_finance_handoff_profile_reserves_fifteen_communication_turns() -> None:
    profile = get_rollout_profile(
        "finance_riu_participant_handoff_15", participant_count=4
    )
    assert profile.budgets.turn_budget == 16
    assert profile.participant_role_name_mode == "public"
    assert profile.initial_speaker_mode == "episode_hash"
    assert profile.model_max_output_tokens == 750
    assert profile.participant_model_max_output_tokens == 750
