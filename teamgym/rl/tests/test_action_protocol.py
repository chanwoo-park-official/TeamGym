from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from teamgym.rl.rollout import action_protocol as protocol


def test_workflow_keeps_existing_protocol_imports() -> None:
    from teamgym.rl.rollout import workflow

    for name in (
        "_prompt_kind", "_prompt_payload", "_parse_exact_json_object",
        "_action_schema_error", "_compact_action_has_required_field",
    ):
        assert getattr(workflow, name) is getattr(protocol, name)


@pytest.mark.parametrize("module", [
    "teamgym.rl.rollout.action_protocol",
])
def test_protocol_import_without_the_training_framework(module) -> None:
    subprocess.run(
        [
            sys.executable, "-c",
            "import importlib, sys; importlib.import_module(sys.argv[1]); "
            "blocked = ('teamgym.rl.rollout.workflow', 'rllm', 'verl', 'torch'); "
            "assert not [name for name in sys.modules "
            "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]",
            module,
        ],
        cwd=Path(__file__).resolve().parents[3], check=True,
    )


def test_prompt_decoding_and_compact_field_checks_keep_existing_behavior() -> None:
    payload = {"response_format": {"utterance": "public message"}}
    assert protocol._prompt_payload(json.dumps(payload)) == payload
    for value in (None, "not JSON", "[]", '"quoted text"'):
        assert protocol._prompt_payload(value) == {}
    assert protocol._prompt_kind("task.turn_1.participant_response") == "participant_response"
    assert protocol._compact_action_has_required_field({"answer": "decision_a"}, "participant_final_decision")
    assert not protocol._compact_action_has_required_field({"answer": " "}, "participant_final_decision")
    assert protocol._compact_action_has_required_field({"target": "participant_a"}, "coordinator_query")
    assert not protocol._compact_action_has_required_field({"answer": "decision_a"}, "coordinator_query")
