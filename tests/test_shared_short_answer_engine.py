from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from teamgym.core.schema import COMMON_DATAPOINT_SCHEMA_VERSION
from teamgym.data.long_context_reasoning import judge as batch
from teamgym.scoring import short_answers as engine


@pytest.mark.parametrize("score", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_scores_are_grading_errors(score):
    with pytest.raises(ValueError, match="Judge score must be finite"):
        engine.normalize_judgment({"score": score}, required_steps=[])
    with pytest.raises(ValueError, match="Judge score must be finite"):
        engine.normalize_judgment(
            {"score": 0.5, "step_scores": [{"step": 1, "covered": True, "score": score}]},
            required_steps=[{"step": 1}],
        )


class MockJudgeClient:
    def __init__(self, *, fail: bool = False, passed: bool = True):
        self.fail = fail
        self.passed = passed
        self.calls = []
        self.last_request_meta = {"provider": "mock", "model": "mock-judge"}

    def generate_json(self, **request):
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("mocked judge failure")
        return {
            "passed": self.passed, "score": 0.8, "conclusion_correct": True,
            "covered_step_count": 1, "total_step_count": 1,
            "step_scores": [{"step": 1, "covered": True, "score": 1.0}],
            "missing_steps": [], "incorrect_claims": [], "disqualifying_errors_found": [],
        }


def _input(path: Path) -> dict:
    row = {
        "schema_version": COMMON_DATAPOINT_SCHEMA_VERSION,
        "instance_id": "batch_case", "dataset": "cuad",
        "task_description": {"question": "Is the action permitted?", "answer_type": "short_answer_judged"},
        "oracle": {
            "canonical_response": "The agreement permits the action.",
            "required_decision_steps": [{"step": 1, "uses": ["r1"], "claim": "Apply the permission."}],
        },
        "reference_information_units": [{"rui_id": "r1", "text": "The action is permitted."}],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return row


@pytest.mark.parametrize("flag", ["false", "true", 0, 1, None])
def test_invalid_rubric_flag_fails_before_a_model_request(tmp_path, flag):
    row = _input(tmp_path / "input.jsonl")
    row["oracle"]["judge_rubric"] = {"require_all_steps_for_pass": True}
    row["dataset_specific"] = {"judge_rubric": {"require_all_steps_for_pass": flag}}
    client = MockJudgeClient()
    with pytest.raises(ValueError, match=r"dataset_specific\.judge_rubric\..*JSON boolean"):
        engine.judge_one_answer(
            datapoint=row, client=client, answer_row={"answer_text": "yes"}, trace_dir=None, index=1
        )
    assert client.calls == []


@pytest.mark.parametrize("flag", [False, True])
def test_rubric_policy_honors_json_boolean(flag):
    assert engine.requires_all_steps_for_pass(
        {"oracle": {"judge_rubric": {"require_all_steps_for_pass": flag}}}
    ) is flag


def test_legacy_judge_reexports_the_shared_grading_engine() -> None:
    for name in (
        "JUDGE_SCHEMA", "SYSTEM_PROMPT", "judge_one_answer", "normalize_judgment",
        "judge_reference_information_units", "deterministic_numeric_evaluation",
        "pass_policy_for_datapoint", "strict_judgment_passed",
    ):
        assert getattr(batch, name) is getattr(engine, name), name


def test_grading_engine_imports_without_batch_construction_or_training() -> None:
    subprocess.run(
        [
            sys.executable, "-c",
            "import sys; import teamgym.scoring.short_answers; "
            "blocked = ('teamgym.data', "
            "'teamgym.interaction', 'teamgym.rl', 'rllm', 'verl', 'torch'); "
            "assert not [name for name in sys.modules "
            "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]",
        ],
        cwd=Path(__file__).resolve().parents[1], check=True,
    )


def test_batch_oracle_self_check_still_writes_judgments_and_reports(tmp_path) -> None:
    source = tmp_path / "input.jsonl"
    row = _input(source)
    output = tmp_path / "output"
    client = MockJudgeClient()
    report = batch.judge_short_answers(input_root=source, output_root=output, use_oracle_as_answer=True, client=client)
    assert report["aggregate"]["judged_count"] == 1
    assert report["aggregate"]["error_count"] == 0
    assert report["aggregate"]["mean_score"] == 0.8
    assert report["aggregate"]["passed_count"] == 1
    assert len(client.calls) == 1
    assert client.calls[0]["system_prompt"] == engine.SYSTEM_PROMPT
    assert json.loads(client.calls[0]["user_prompt"])["candidate_answer"] == row["oracle"]["canonical_response"]
    judgment = json.loads((output / "judgments.jsonl").read_text())
    assert judgment["schema"] == engine.JUDGE_SCHEMA
    assert judgment["answer_source"] == "oracle_self_check"
    assert json.loads((output / "judge_report.json").read_text()) == report
    assert (output / "judge_report.md").is_file()
    assert (output / "judge_report.html").is_file()


@pytest.mark.parametrize("require_all_steps", [False, True])
def test_legal_judge_failure_is_counted_as_failure_with_full_coverage(tmp_path, require_all_steps) -> None:
    source = tmp_path / "input.jsonl"
    row = _input(source)
    row["oracle"]["judge_rubric"] = {"require_all_steps_for_pass": require_all_steps}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "output"
    report = batch.judge_short_answers(
        input_root=source, output_root=output, use_oracle_as_answer=True,
        client=MockJudgeClient(passed=False),
    )

    judgment = json.loads((output / "judgments.jsonl").read_text())
    assert judgment["judgment"]["passed"] is False
    assert report["aggregate"]["mean_covered_step_rate"] == 1.0
    assert report["aggregate"]["passed_count"] == 0
    assert report["aggregate"]["strict_all_step_passed_count"] == 0

    # Existing v2 cache entries lack the separate strict result.
    judgment["judgment"].pop("strict_all_step_passed")
    assert batch.strict_all_step_passed(judgment) is False


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_batch_judge_keeps_error_logging_and_continuation_behavior(tmp_path, continue_on_error) -> None:
    source = tmp_path / "input.jsonl"
    _input(source)
    output = tmp_path / "output"
    arguments = dict(input_root=source, output_root=output, use_oracle_as_answer=True,
                     client=MockJudgeClient(fail=True), continue_on_error=continue_on_error)
    if continue_on_error:
        report = batch.judge_short_answers(**arguments)
        assert report["aggregate"]["judged_count"] == 0
        assert report["aggregate"]["error_count"] == 1
    else:
        with pytest.raises(RuntimeError, match="mocked judge failure"):
            batch.judge_short_answers(**arguments)
    error = json.loads((output / "judge_errors.jsonl").read_text())
    assert error["error_type"] == "RuntimeError"
    assert error["error_message"] == "mocked judge failure"
