from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from teamgym.rl import launch


def _arguments(tmp_path: Path, *extra: str) -> list[str]:
    return ["--prepared-data-dir", str(tmp_path / "data with spaces"),
            "--profile", "short_context_smoke", "--reward-config", "task_success",
            "--model", str(tmp_path / "model with spaces"),
            "--run-dir", str(tmp_path / "fresh run"), *extra]


def test_historical_smoke_profile_name_produces_the_same_plan(tmp_path):
    parser = launch.make_parser()
    current = launch.build_plan(parser.parse_args(_arguments(tmp_path)))
    historical = launch.build_plan(
        parser.parse_args(_arguments(tmp_path, "--profile", "smoke_h200_qwen25"))
    )

    assert historical == current
    assert current["profile"]["name"] == "short_context_smoke"
    assert current["environment"]["ROLLOUT_PROFILE"] == "short_context_smoke"
    assert "short_context_smoke" in parser.format_help()
    assert "smoke_h200_qwen25" not in parser.format_help()


@pytest.mark.parametrize(("extra", "mode", "backend"), [
    ([], "joint", "rollout"),
    (["--training-mode", "coordinator_only", "--participant-backend", "scripted"],
     "coordinator_only", "scripted"),
    (["--training-mode", "coordinator_only", "--fixed-participant-base-url", "http://localhost:9001/v1",
      "--fixed-participant-model", "frozen-participant"], "coordinator_only", "external_openai"),
    (["--training-mode", "participant_only", "--fixed-coordinator-base-url", "http://localhost:9002/v1",
      "--fixed-coordinator-model", "frozen-coordinator"], "participant_only", "rollout"),
    (["--training-mode", "participant_only", "--profile", "finance_riu_participant_handoff_15"],
     "participant_only", "rollout"),
])
def test_generic_role_plans_reuse_the_backend_without_paper_data_rules(tmp_path, extra, mode, backend):
    args = launch.make_parser().parse_args(_arguments(tmp_path, *extra))
    plan = launch.build_plan(args)
    env = plan["environment"]
    assert env["TRAINING_MODE"] == mode
    assert env["PARTICIPANT_BACKEND"] == backend
    assert env["N_GPUS_PER_NODE"] == "1"
    assert env["PPO_MINI_BATCH_SIZE"] == "4"
    assert env["TOTAL_TRAINING_STEPS"] == "3"
    assert Path(plan["command"][1]).is_file()
    assert plan["paper_recipe"] is False
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("extra", [
    ["--devices", "0,0"],
    ["--devices", "0,,1"],
    ["--tensor-parallel-size", "2"],
    ["--rollouts", "1"],
    ["--updates", "0"],
    ["--learning-rate", "nan"],
    ["--validation-split", "train"],
    ["--train-split", "../outside"],
    ["--profile", "finance_riu_participant_handoff_15"],
    ["--training-mode", "participant_only"],
    ["--training-mode", "coordinator_only"],
    ["--fixed-coordinator-model", "unused-model"],
    ["--max-response-length", "64"],
])
def test_invalid_plans_fail_before_reading_inputs_or_starting_jobs(tmp_path, extra):
    args = launch.make_parser().parse_args(_arguments(tmp_path, *extra))
    with pytest.raises(ValueError):
        launch.build_plan(args)
    assert not list(tmp_path.iterdir())


def test_dry_run_has_no_data_reads_writes_or_training_imports(tmp_path):
    code = (
        "import sys; from teamgym.rl.launch import main; main(sys.argv[1:]); "
        "blocked = ('torch', 'pandas', 'numpy', 'openai', 'verl', 'rllm', "
        "'teamgym.rl.train', 'teamgym.rl.preparation.check_training_data'); "
        "assert not [name for name in sys.modules "
        "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]"
    )
    result = subprocess.run([sys.executable, "-c", code, *_arguments(tmp_path, "--dry-run")],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["environment"]["MODEL_PATH"] == str(tmp_path / "model with spaces")
    assert not list(tmp_path.iterdir())


def test_data_only_action_needs_no_model_or_run_directory(tmp_path, monkeypatch, capsys):
    def check(root, **settings):
        assert root == tmp_path
        assert settings["train_split"] == "learn"
        return {"checked": True}

    monkeypatch.setitem(sys.modules, "teamgym.rl.preparation.check_training_data",
                        SimpleNamespace(check_prepared_splits=check))
    launch.main(["--prepared-data-dir", str(tmp_path), "--train-split", "learn",
                 "--profile", "short_context_smoke", "--reward-config", "task_success", "--check-data"])
    assert json.loads(capsys.readouterr().out) == {"checked": True}
    assert not list(tmp_path.iterdir())
