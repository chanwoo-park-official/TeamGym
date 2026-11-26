from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from teamgym.rl.rollout import frozen_client


def test_frozen_client_imports_without_the_workflow_or_rl_framework() -> None:
    subprocess.run(
        [
            sys.executable, "-c",
            "import sys; import teamgym.rl.rollout.frozen_client; "
            "blocked = ('teamgym.rl.rollout.workflow', 'rllm', 'verl', 'torch'); "
            "assert not [name for name in sys.modules "
            "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]",
        ],
        cwd=Path(__file__).resolve().parents[3], check=True,
    )


def test_standalone_client_preserves_transport_options_and_frozen_logs(monkeypatch) -> None:
    class Backend:
        last_request_meta = {"request": "mock"}

        def __init__(self, config):
            self.config = asdict(config)
            self.calls = []

        def generate_json(self, **request):
            self.calls.append(request)
            return {"utterance": "Source evidence."}

    monkeypatch.setattr(frozen_client, "OpenAIJsonClient", Backend)
    records = []
    client = frozen_client.FrozenOpenAIJsonClient(
        base_url="http://offline.test/v1", model="fixed-policy", api_key="mock-key",
        max_output_tokens=750, temperature=0.0, timeout_sec=0.2,
        attempt_records=records, max_parse_retries=-1, policy_route="frozen_external_participant",
    )
    response = client.generate_json(prompt_id="case.turn_1.participant_a.participant_response",
                                    system_prompt="Return JSON.",
                                    user_prompt=json.dumps({"response_format": {"utterance": "message"}}))
    assert response == {"utterance": "Source evidence."}
    assert client.max_parse_retries == 0
    config = client.client.config
    assert config["provider"] == "openai_compatible"
    assert config["endpoint"] == "/chat/completions"
    assert config["timeout_sec"] == 1
    assert config["request_retry_attempts"] == 2
    assert config["model"] == "fixed-policy" and config["max_output_tokens"] == 750
    assert config["temperature"] == 0.0
    assert config["log_prompts"] is False and config["log_responses"] is False
    assert len(records) == 1 and records[0]["actor_role"] == "participant"
    assert records[0]["train_mask"] is False and records[0]["local_reward_penalty"] == 0.0
    assert records[0]["policy_route"] == "frozen_external_participant"
    assert records[0]["lora_name"] == "__base__"
    assert client.last_request_meta == {"request": "mock"}


def test_transport_errors_are_logged_and_propagated_without_format_retries(monkeypatch) -> None:
    class FailingBackend:
        def __init__(self, _config):
            self.last_request_meta = {"request": "failed"}
            self.calls = 0

        def generate_json(self, **_request):
            self.calls += 1
            raise RuntimeError("mocked transport failure")

    monkeypatch.setattr(frozen_client, "OpenAIJsonClient", FailingBackend)
    records = []
    client = frozen_client.FrozenOpenAIJsonClient(
        base_url="http://offline.test/v1", model="fixed-policy", api_key="mock-key",
        max_output_tokens=64, temperature=0.0, timeout_sec=30,
        attempt_records=records, max_parse_retries=3,
    )
    with pytest.raises(RuntimeError, match="mocked transport failure"):
        client.generate_json(prompt_id="case.turn_1.coordinator.coordinator_query",
                             system_prompt="Return JSON.", user_prompt="{}")
    assert client.client.calls == 1
    assert len(records) == 1 and records[0]["parse_status"] == "request_error"
    assert records[0]["parse_error"] == "mocked transport failure"
    assert records[0]["train_mask"] is False and records[0]["local_reward_penalty"] == 0.0
    assert client.last_request_meta == {"request": "failed"}
