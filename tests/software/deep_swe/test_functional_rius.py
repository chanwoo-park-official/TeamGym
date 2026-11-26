from __future__ import annotations

import pytest
from types import SimpleNamespace

from teamgym.data.software.deep_swe.candidate import (
    assemble_functional_candidate,
    leave_one_out_conditions,
)
from teamgym.data.software.deep_swe.construction import (
    FunctionalRIUConstructionError,
    validate_plan,
    construct_in_stages,
)
from teamgym.data.software.deep_swe.stages.patch_runtime import Editor
from teamgym.data.software.deep_swe.stages.patch_episode import (
    is_test_path,
    selected_check_ids,
)


def _pool() -> dict:
    return {
        "task_id": "example",
        "instruction": "Implement the requested behavior.",
        "image_id": "example:latest",
        "fragments": [
            {
                "id": "frag-0001", "path": "a.py", "line_start": 1, "line_end": 1,
                "text": "def parse(): pass\n", "sha256": "a", "kind": "function",
                "name": "parse", "tokens": 5,
            },
            {
                "id": "frag-0002", "path": "b.py", "line_start": 1, "line_end": 1,
                "text": "def close(): pass\n", "sha256": "b", "kind": "function",
                "name": "close", "tokens": 5,
            },
        ],
        "oversized": [],
    }


def _observations() -> list[dict]:
    return [
        {"id": "observation-01", "argv": ["python", "-c", "print(1)"], "stdout": "1\n"}
    ]


def test_functional_candidate_preserves_constructor_boundary_but_not_in_allocation() -> None:
    plan = validate_plan(
        {
            "rius": [
                {
                    "functional_boundary": "parse behavior",
                    "code_fragment_ids": ["frag-0001"],
                    "observation_ids": ["observation-01"],
                    "rationale": "needed for parsing",
                    "coverage": ["parse failure"],
                },
                {
                    "functional_boundary": "cleanup behavior",
                    "code_fragment_ids": ["frag-0002"],
                    "observation_ids": [],
                    "rationale": "needed for cleanup",
                    "coverage": ["close failure"],
                },
            ]
        },
        pool=_pool(),
        observations=_observations(),
    )
    candidate = assemble_functional_candidate(
        _pool(), rius=plan["rius"], observations=_observations()
    )

    assert [row["code_atom_ids"] for row in candidate["allocation_candidates"]] == [
        ["atom-01"], ["atom-02"]
    ]
    assert "functional_boundary" not in candidate["allocation_candidates"][0]
    assert candidate["constructor_metadata"]["rius"][0]["functional_boundary"] == "parse behavior"
    assert [row["label"] for row in leave_one_out_conditions(candidate, "riu")] == [
        "baseline", "riu-01", "riu-02"
    ]


def test_functional_plan_rejects_unit_owned_by_two_rius() -> None:
    with pytest.raises(FunctionalRIUConstructionError, match="overlaps"):
        validate_plan(
            {
                "rius": [
                    {
                        "functional_boundary": "first", "code_fragment_ids": ["frag-0001"],
                        "observation_ids": [], "rationale": "first reason",
                    },
                    {
                        "functional_boundary": "second", "code_fragment_ids": ["frag-0001"],
                        "observation_ids": [], "rationale": "second reason",
                    },
                ]
            },
            pool=_pool(),
            observations=_observations(),
        )


def test_editor_write_cannot_overwrite_a_partially_visible_existing_file() -> None:
    source = "first\nsecond\n"

    def read_base(path: str) -> dict:
        if path == "existing.txt":
            return {"kind": "file", "text": source}
        return {"kind": "absent"}

    editor = Editor(
        [{"path": "existing.txt", "line_start": 1, "line_end": 1, "text": "first\n"}],
        read_base,
    )
    assert editor.edit([
        {"op": "write", "path": "existing.txt", "text": "replacement\n"}
    ]) == {"status": "rejected"}
    assert editor.last_reason == "cannot_overwrite_nonvisible_existing_file"
    assert editor.edit([
        {"op": "write", "path": "new.txt", "text": "new content\n"}
    ]) == {"status": "applied"}


def test_solver_selects_only_allowlisted_public_checks() -> None:
    catalog = {"compile": ["go", "test", "./pkg", "-run", "^$"]}

    assert selected_check_ids(
        {"checks": ["compile", "compile"]}, catalog
    ) == ["compile"]
    with pytest.raises(ValueError, match="unknown_public_check"):
        selected_check_ids({"checks": ["hidden"]}, catalog)


def test_solver_cannot_edit_test_files() -> None:
    assert is_test_path("pkg/parser_test.go")
    assert is_test_path("tests/test_parser.py")
    assert is_test_path("src/parser.spec.ts")
    assert not is_test_path("src/parser.py")


class _ConstructorClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.cfg = SimpleNamespace(max_output_tokens=16000)
        self.last_request_meta = {"usage": {"output_tokens": 10}}
        self.requests = []

    def generate_json(self, **kwargs):
        import json
        self.requests.append(json.loads(kwargs["user_prompt"]))
        return next(self.responses)


def _proposal(fragment_ids):
    return {"action": "propose", "rius": [{
        "functional_boundary": "parser behavior",
        "rationale": "base parser contract",
        "code_fragment_ids": fragment_ids, "observation_ids": [],
    }]}


def test_staged_constructor_returns_empty_riu_error_for_repair(tmp_path) -> None:
    client = _ConstructorClient([
        {"action": "read", "fragment_ids": ["frag-0001"]},
        _proposal([]), _proposal(["frag-0001"]),
    ])
    plan, _, stats = construct_in_stages(
        client=client,
        request={"source_fragments": _pool()["fragments"]},
        system_prompt="Construct JSON RIUs", pool=_pool(), observations=[],
        trace_dir=tmp_path,
    )
    assert "text" not in client.requests[0]["source_fragments"][0]
    assert client.requests[0]["read_source_fragments"] == []
    assert client.requests[1]["read_source_fragments"][0]["text"] == "def parse(): pass\n"
    assert client.requests[2]["constructor_feedback"]["error"] == "RIU 1 is empty"
    assert plan["rius"][0]["code_fragment_ids"] == ["frag-0001"]
    assert stats["validation_errors"] == 1


def test_staged_constructor_reports_over_budget_reads_and_stops_invalid_plans(tmp_path) -> None:
    client = _ConstructorClient([
        {"action": "read", "fragment_ids": ["frag-0001"]},
        _proposal(["frag-0001"]), _proposal(["frag-0001"]),
    ])
    with pytest.raises(FunctionalRIUConstructionError, match="retries exhausted"):
        construct_in_stages(
            client=client, request={"source_fragments": _pool()["fragments"]},
            system_prompt="Construct JSON RIUs", pool=_pool(), observations=[],
            trace_dir=tmp_path, round_source_tokens=1, validation_retries=1,
        )
    assert client.requests[1]["constructor_feedback"]["dropped_over_source_budget"] == ["frag-0001"]
    assert client.requests[1]["read_source_fragments"] == []
    assert "read first" in client.requests[2]["constructor_feedback"]["error"]


def test_constructor_uses_one_real_tool_call_and_unique_traces(tmp_path):
    import json
    from types import SimpleNamespace
    from teamgym.data.software.deep_swe.construction import ConstructorActionClient
    bodies = []
    def create(**body):
        bodies.append(body)
        return SimpleNamespace(model_dump=lambda: {'status': 'completed', 'usage': {'output_tokens': 12},
            'output': [{'type': 'function_call', 'name': 'constructor_action',
                        'arguments': json.dumps({'action': 'read', 'fragment_ids': ['frag-0001']})}]})
    wrapped = SimpleNamespace(cfg=SimpleNamespace(model='gpt-5.4', max_output_tokens=32000,
        reasoning_effort='medium'), _client=SimpleNamespace(responses=SimpleNamespace(create=create)),
        _validate_response_status=lambda response: None)
    client = ConstructorActionClient(wrapped)
    for _ in range(2):
        assert client.generate_json(prompt_id='same', system_prompt='system', user_prompt='user',
                                    trace_dir=tmp_path)['action'] == 'read'
    assert len(list(tmp_path.glob('*.response.json'))) == 2
    assert bodies[0]['tool_choice'] == {'type': 'function', 'name': 'constructor_action'}
    assert bodies[0]['parallel_tool_calls'] is False
    assert client.last_request_meta['usage']['output_tokens'] == 12


@pytest.mark.parametrize('running,expected', [(True, True), (False, False)])
def test_verifier_timeout_counts_only_started_docker_as_patch_failure(tmp_path, monkeypatch, running, expected):
    import json
    import subprocess
    from types import SimpleNamespace
    from teamgym.data.software.deep_swe.stages.patch_runtime import verify
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == 'run':
            raise subprocess.TimeoutExpired(cmd, 2)
        if cmd[1] == 'inspect':
            return SimpleNamespace(returncode=0, stdout=json.dumps([{'State': {'Running': running,
                                        'OOMKilled': False, 'Error': ''}}]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', fake_run)
    result = verify(SimpleNamespace(runtime=SimpleNamespace(verifier_timeout_sec=2)),
                    'image', tmp_path, tmp_path/'model.patch')
    assert result['verification_valid'] is expected
    assert result['failure_kind'] == ('verifier_timeout' if expected else 'docker_error')
    assert result['reward']['reward'] == 0
    assert (tmp_path/'verifier/timeout.json').exists()
    assert calls[-1][1:3] == ['rm', '-f']
