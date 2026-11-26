from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from rllm.agents.agent import Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine

from teamgym.rl.rollout import action_repair, policy_routing, rollout_client


def test_workflow_preserves_rollout_repair_and_routing_imports() -> None:
    from teamgym.rl.rollout import workflow

    assert workflow.RolloutEngineJsonClient is rollout_client.RolloutEngineJsonClient
    for name in ("_UTTERANCE_FIELD_RE", "_repair_compact_action", "_nested_utterance",
                 "_partial_utterance_value", "_participant_counterfactual_utterance",
                 "_repair_participant_format_counterfactual"):
        assert getattr(workflow, name) is getattr(action_repair, name)
    for name in ("FULL_MODEL_NAME", "VLLM_LORA_INT_ID", "VLLM_LORA_NAME", "_is_trainable"):
        assert getattr(workflow, name) is getattr(policy_routing, name)


def test_format_repair_imports_without_the_workflow_or_training_framework() -> None:
    subprocess.run(
        [
            sys.executable, "-c",
            "import sys; import teamgym.rl.rollout.action_repair; "
            "blocked = ('teamgym.rl.rollout.workflow', 'rllm', 'verl', 'torch'); "
            "assert not [name for name in sys.modules "
            "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]",
        ],
        cwd=Path(__file__).resolve().parents[3], check=True,
    )


def test_rollout_adapter_imports_without_episode_orchestration() -> None:
    subprocess.run(
        [
            sys.executable, "-c",
            "import sys; import teamgym.rl.rollout.rollout_client; "
            "assert 'teamgym.rl.rollout.workflow' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[3], check=True,
    )


def test_standalone_rollout_adapter_preserves_tokens_masks_and_retry_penalties() -> None:
    class Engine(RolloutEngine):
        def __init__(self):
            self.requests = []

        async def get_model_response(self, messages, **kwargs):
            self.requests.append({"messages": deepcopy(messages), "options": deepcopy(kwargs)})
            raw = "invalid JSON" if len(self.requests) == 1 else json.dumps({"utterance": "Source evidence."})
            return ModelOutput(text=raw, content="", reasoning=raw, prompt_ids=[1, 2],
                               completion_ids=[11, 12], logprobs=[-0.1, -0.2],
                               prompt_length=2, completion_length=2, finish_reason="stop")

    async def run():
        engine = Engine()
        trainable = Trajectory(uid="trainable", name="trainable")
        fixed = Trajectory(uid="fixed", name="fixed")
        client = rollout_client.RolloutEngineJsonClient(
            rollout_engine=engine, event_loop=asyncio.get_running_loop(), uid="case:0",
            trainable_trajectory=trainable, fixed_trajectory=fixed,
            training_mode="participant_only", rollout_lora_name=None,
            max_output_tokens=64, participant_max_output_tokens=32, max_parse_retries=1,
            invalid_json_penalty=-0.1, invalid_schema_penalty=-0.2, request_timeout_sec=10,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = await asyncio.get_running_loop().run_in_executor(
                executor,
                lambda: client.generate_json(prompt_id="case.turn_1.participant_a.participant_response",
                                             system_prompt="Return JSON.",
                                             user_prompt=json.dumps({"response_format": {"utterance": "message"}})),
            )
        return response, client, engine, trainable, fixed

    response, client, engine, trainable, fixed = asyncio.run(run())
    assert response == {"utterance": "Source evidence."}
    assert len(engine.requests) == 2
    assert all(request["options"]["max_tokens"] == 32 for request in engine.requests)
    assert all("lora_request" not in request["options"] for request in engine.requests)
    assert "32 tokens" in engine.requests[1]["messages"][-1]["content"]
    assert [step.info["parse_status"] for step in trainable.steps] == ["invalid_json", "ok"]
    assert all(step.info["train_mask"] is True for step in trainable.steps)
    assert [step.reward for step in trainable.steps] == [-0.1, 0.0]
    assert client.local_penalty_total == -0.1 and fixed.steps == []
    assert all(step.model_output.prompt_ids == [1, 2] for step in trainable.steps)
    assert all(step.model_output.completion_ids == [11, 12] for step in trainable.steps)
    assert all(step.model_output.logprobs == [-0.1, -0.2] for step in trainable.steps)
    assert all(step.info["lora_name"] == "__full__" for step in trainable.steps)
