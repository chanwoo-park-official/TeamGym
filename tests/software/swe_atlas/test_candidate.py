from __future__ import annotations

import base64
import subprocess
import pytest

from teamgym.data.software.swe_atlas.candidate import (
    CandidateValidationError,
    DockerEvidenceCollector,
    _artifact_for_prompt,
    _source_first_evidence_mode_constraints,
    _validate_runtime_command,
    validate_candidates,
    validate_plan,
)
from teamgym.data.software.swe_atlas.schema import Rubric


RUNTIME = Rubric("runtime", "reports observed output", "positive", "must have", {})
STATIC = Rubric("static", "explains code path", "positive", "must have", {})


def test_synthesis_runtime_artifact_hides_command() -> None:
    artifact = {
        "artifact_id": "run",
        "kind": "runtime",
        "command": "echo SECRET_COLLECTOR_COMMAND",
        "content": "__COMMAND__=echo SECRET_COLLECTOR_COMMAND\nOBSERVED=1",
        "observed_content": "OBSERVED=1",
        "verified": True,
    }
    prompt_artifact = _artifact_for_prompt(
        artifact, include_runtime_command=False
    )
    assert "command" not in prompt_artifact
    assert prompt_artifact["content"] == "OBSERVED=1"


def test_plan_requires_source_anchor_for_hybrid_rubric() -> None:
    plan = {
        "rubric_requirements": [
            {"rubric_id": "runtime", "evidence_mode": "hybrid"},
            {"rubric_id": "static", "evidence_mode": "source"},
        ],
        "evidence_requests": [
            {
                "request_id": "run",
                "kind": "runtime",
                "command": "go test ./pkg -run TestThing -v",
                "expected_output_substrings": ["PASS"],
                "supports_rubric_ids": ["runtime"],
            },
            {
                "request_id": "src",
                "kind": "source",
                "path": "pkg/x.go",
                "supports_rubric_ids": ["static"],
            },
        ],
    }
    with pytest.raises(CandidateValidationError, match="no source anchor"):
        validate_plan(plan, (RUNTIME, STATIC), {"pkg/x.go"})


def test_plan_allows_at_most_one_integrated_runtime_request() -> None:
    plan = {
        "rubric_requirements": [
            {"rubric_id": "runtime", "evidence_mode": "hybrid"},
            {"rubric_id": "static", "evidence_mode": "source"},
        ],
        "evidence_requests": [
            {
                "request_id": "src",
                "kind": "source",
                "path": "pkg/x.go",
                "supports_rubric_ids": ["runtime", "static"],
            },
            {
                "request_id": "run1",
                "kind": "runtime",
                "command": "echo ONE",
                "expected_output_substrings": ["ONE"],
                "supports_rubric_ids": ["runtime"],
            },
            {
                "request_id": "run2",
                "kind": "runtime",
                "command": "echo TWO",
                "expected_output_substrings": ["TWO"],
                "supports_rubric_ids": ["runtime"],
            },
        ],
    }
    with pytest.raises(CandidateValidationError, match="at most one integrated runtime"):
        validate_plan(plan, (RUNTIME, STATIC), {"pkg/x.go"})


def test_source_first_mode_only_reserves_irreducible_output_rubrics() -> None:
    startup = Rubric(
        "startup", "Reports the process entry point observed at runtime", "positive", "must have", {}
    )
    logs = Rubric(
        "logs", "Reports log messages indicating the application is ready", "positive", "must have", {}
    )
    assert _source_first_evidence_mode_constraints((startup, logs)) == {
        "startup": "source",
        "logs": "hybrid",
    }


def test_source_first_mode_reserves_concrete_execution_artifacts() -> None:
    test_output = Rubric(
        "test-output",
        "Reports verbose Go test output markers for the SMTP endpoint test run",
        "positive",
        "must have",
        {},
    )
    wire_bytes = Rubric(
        "wire-bytes",
        "Reports first 20 bytes of serialized IPv4 header in hexadecimal format, "
        "demonstrating reliance on actual on-the-wire bytes",
        "positive",
        "must have",
        {},
    )
    source_fact = Rubric(
        "source-fact",
        "Identifies IPField as the field class used to represent IPv4 values",
        "positive",
        "must have",
        {},
    )
    assert _source_first_evidence_mode_constraints(
        (test_output, wire_bytes, source_fact)
    ) == {
        "test-output": "hybrid",
        "wire-bytes": "hybrid",
        "source-fact": "source",
    }


def test_runtime_request_cannot_absorb_source_constrained_rubric() -> None:
    plan = {
        "rubric_requirements": [
            {"rubric_id": "runtime", "evidence_mode": "hybrid"},
            {"rubric_id": "static", "evidence_mode": "source"},
        ],
        "evidence_requests": [
            {
                "request_id": "src",
                "kind": "source",
                "path": "pkg/x.go",
                "supports_rubric_ids": ["runtime", "static"],
            },
            {
                "request_id": "run",
                "kind": "runtime",
                "command": "echo READY",
                "expected_output_substrings": ["READY"],
                "supports_rubric_ids": ["runtime", "static"],
            },
        ],
    }
    with pytest.raises(CandidateValidationError, match="source-only rubrics"):
        validate_plan(
            plan,
            (RUNTIME, STATIC),
            {"pkg/x.go"},
            evidence_mode_constraints={"runtime": "hybrid", "static": "source"},
        )


def test_preflight_combines_inventory_tools_and_config_keys(monkeypatch) -> None:
    collector = DockerEvidenceCollector("image", "abc", timeout_sec=1)
    monkeypatch.setattr(
        collector,
        "_run",
        lambda _script: {
            "runner_exit_code": 0,
            "content": (
                "__COMMIT__=abc\n__RUNTIME_UID__=0\n__RUNTIME_USER__=root\n"
                "__PATHS_BEGIN__\nserver.py\ntests/test.env\n"
                "__PATHS_END__\n__TOOLS_BEGIN__\n/app/venv/bin/python\n"
                "__TOOLS_END__\n__CONFIG_BEGIN__\n__CONFIG_FILE__=tests/test.env\n"
                "__CONFIG_KEY__=tests/test.env:URL\n__CONFIG_KEY__=tests/test.env:DB_URI\n"
                "__CONFIG_END__\n"
            ),
        },
    )
    preflight = collector.preflight()
    assert preflight["paths"] == ["server.py", "tests/test.env"]
    assert preflight["runtime_uid"] == 0
    assert preflight["tool_paths"]["repository_python"] == "/app/venv/bin/python"
    assert preflight["config_files"] == [
        {"path": "tests/test.env", "keys": ["DB_URI", "URL"]}
    ]
    assert collector.inventory()["commit"] == "abc"


def test_candidate_rejects_bare_runtime_output() -> None:
    response = {
        "candidates": [
            {
                "candidate_id": "c1",
                "claim": "it passed",
                "supports_rubric_ids": ["runtime"],
                "refutes_rubric_ids": [],
                "evidence": [{"artifact_id": "run", "quote": "PASS"}],
            }
        ]
    }
    artifacts = [{"artifact_id": "run", "kind": "runtime", "content": "PASS", "verified": True}]
    requirements = [{"rubric_id": "runtime", "evidence_mode": "hybrid"}]
    with pytest.raises(CandidateValidationError, match="bare runtime"):
        validate_candidates(response, (RUNTIME,), requirements, artifacts)


def test_runtime_validator_ignores_comparison_inside_heredoc() -> None:
    _validate_runtime_command(
        "python3 - <<'PY'\n"
        "import sys\n"
        "print(len(sys.argv) > 2)\n"
        "PY"
    )


def test_runtime_validator_allows_tmp_mktemp_variable_redirect() -> None:
    _validate_runtime_command(
        "work=$(mktemp -d /tmp/swe-test.XXXXXX); echo ok > \"$work/out.txt\""
    )


def test_runtime_validator_allows_append_to_tmp_mktemp_variable() -> None:
    _validate_runtime_command(
        'work=$(mktemp -d /tmp/swe-test.XXXXXX); echo ok >> "$work/out.txt"'
    )


def test_runtime_validator_allows_redirect_to_tmp_derived_variable() -> None:
    _validate_runtime_command(
        'work=$(mktemp -d /tmp/swe-test.XXXXXX); out="$work/server.log"; echo ok > "$out"'
    )


def test_runtime_validator_allows_tmp_variable_in_command_substitution() -> None:
    _validate_runtime_command(
        "OUT=/tmp/audit.txt; PAYLOAD=\"$(echo ok > $OUT)\""
    )


def test_runtime_validator_rejects_non_tmp_shell_redirect() -> None:
    with pytest.raises(CandidateValidationError, match="only write under /tmp"):
        _validate_runtime_command("echo unsafe > repository-file.txt")


def test_runtime_validator_rejects_nested_bash_lc() -> None:
    with pytest.raises(CandidateValidationError, match="nested bash -lc"):
        _validate_runtime_command("bash -lc 'echo observed'")


def test_runtime_timeout_becomes_repairable_failed_artifact(monkeypatch) -> None:
    collector = DockerEvidenceCollector("image", "commit", timeout_sec=1)

    def fail(_script: str):
        raise CandidateValidationError("Docker evidence command timed out after 1s")

    monkeypatch.setattr(collector, "_run", fail)
    artifact = collector._runtime(
        {
            "request_id": "run", "command": "go test ./pkg", "expected_exit_code": 0,
            "purpose": "test", "supports_rubric_ids": ["runtime"],
        }
    )
    assert artifact["verified"] is False
    assert "timed out" in artifact["collection_error"]


def test_timed_out_docker_command_removes_its_own_container(monkeypatch) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    collector = DockerEvidenceCollector("image", "commit", timeout_sec=1)
    with pytest.raises(CandidateValidationError, match="timed out after 1s"):
        collector._run("sleep 10")
    started, cleanup = calls
    name = started[started.index("--name") + 1]
    assert name.startswith("teamgym-atlas-evidence-")
    assert cleanup == ["docker", "rm", "--force", name]


def test_runtime_expected_output_is_mechanically_required(monkeypatch) -> None:
    collector = DockerEvidenceCollector("image", "commit", timeout_sec=1)
    monkeypatch.setattr(
        collector,
        "_run",
        lambda _script: {
            "runner_exit_code": 0,
            "content": (
                "__CAPTURE_SCHEMA__=swe_atlas_runtime_capture_v1\n"
                "__COMMIT__=commit\n__DIRTY_BEFORE__=0\n__DIRTY_AFTER__=0\n"
                "__OUTPUT_BASE64__=" + base64.b64encode(b"ready_seen=False\n").decode() + "\n"
            ),
        },
    )
    artifact = collector._runtime(
        {
            "request_id": "run",
            "command": "echo ready_seen=False",
            "expected_exit_code": 0,
            "expected_output_substrings": ["ready_seen=True"],
            "purpose": "readiness",
            "supports_rubric_ids": ["runtime"],
        }
    )
    assert artifact["verified"] is False
    assert artifact["missing_expected_output_substrings"] == ["ready_seen=True"]
