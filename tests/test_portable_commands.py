"""Check command paths and orchestration with local stubs, without models or jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


def _env(**values):
    cleared = {
        "SOURCE_CHECKPOINT", "SOURCE_ACTOR_DIR", "TARGET_DIR", "WANDB_DIR",
        "CASEBOOK_BASE_SOURCE", "MSLR_INPUT_ROOT", "MULTILEXSUM_INPUT_ROOT",
        "INPUT_JSONL", "SHARD_ROOT", "PYTHONPATH", "PYTHONHOME",
    }
    env = {
        key: value for key, value in os.environ.items()
        if key not in cleared and not key.startswith(("TEAMGYM_", "SARA_", "SLURM_"))
    }
    return {**env, **{key: str(value) for key, value in values.items()}}


def _executable(path, body):
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o700)
    return path


@pytest.mark.parametrize("joint", [False, True])
def test_checkpoint_export_uses_active_python_and_script_relative_helpers(tmp_path, joint):
    checkpoint = tmp_path / "source run" / "global_step_10"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    if joint:
        (checkpoint / "_SUCCESS").write_text("complete\n")
        for role in ("coordinator", "participant"):
            (actor / role).mkdir()
    bin_dir = tmp_path / "active environment"
    bin_dir.mkdir()
    _executable(bin_dir / "python", """
import json
from pathlib import Path
import sys
args = sys.argv[1:]
if args == ['-']:
    exec(sys.stdin.read())
else:
    assert args[:3] == ['-m', 'verl.model_merger', 'merge'], args
    assert Path(args[args.index('--local_dir') + 1]).is_dir()
    target = Path(args[args.index('--target_dir') + 1])
    target.mkdir()
    (target / 'config.json').write_text('{}')
    (target / 'model.safetensors').write_bytes(b'stub')
""")
    outside = tmp_path / "outside checkout"
    outside.mkdir()
    merged = outside / "merged models"
    script = "merge_joint_checkpoint_hf.sh" if joint else "merge_full_checkpoint_hf.sh"
    subprocess.run(
        ["bash", str(REPO / "teamgym/rl" / script)], cwd=outside,
        env=_env(
            SOURCE_CHECKPOINT=checkpoint, TEAMGYM_MERGED_ROOT=merged,
            PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        ), check=True, capture_output=True, text=True,
    )
    roles = ("coordinator", "participant") if joint else (None,)
    for role in roles:
        suffix = f"-{role}" if role else ""
        target = merged / f"source run-step10{suffix}-hf"
        marker = json.loads((target / "_MERGE_SUCCESS.json").read_text())
        assert Path(marker["source_actor_dir"]) == (actor / role if role else actor)
        assert marker["weight_files"] == {"model.safetensors": 4}
    assert not list(merged.glob("*.partial-*"))


def _scheduler(tmp_path):
    bin_dir = tmp_path / "scheduler"
    bin_dir.mkdir()
    _executable(bin_dir / "sbatch", """
import json, os, sys
from pathlib import Path
script = Path(sys.argv[-1])
assert script.is_file(), f"Missing batch script: {script}"
capture = Path(os.environ['CAPTURE'])
records = json.loads(capture.read_text()) if capture.exists() else []
records.append({'args': sys.argv[1:], 'env': {
    name: os.environ.get(name) for name in (
        'TEAMGYM_REPO_ROOT', 'TEAMGYM_DATAGEN_ROOT', 'CASEBOOK_BASE_SOURCE',
        'MSLR_INPUT_ROOT', 'MULTILEXSUM_INPUT_ROOT',
    )
}})
capture.write_text(json.dumps(records))
print(len(records))
""")
    return bin_dir


@pytest.mark.parametrize("run_root", [None, "generated/0712_datagen_final"], ids=["default", "legacy-override"])
def test_construction_and_audit_submissions_keep_caller_relative_paths(tmp_path, run_root):
    outside = tmp_path / "outside checkout"
    outside.mkdir()
    (outside / "casebook.txt").write_text("source")
    (outside / "mslr").mkdir()
    (outside / "tasks.jsonl").write_text("{}\n")
    scheduler = _scheduler(tmp_path)
    capture = tmp_path / "submitted.json"
    env = _env(
        CASEBOOK_BASE_SOURCE="casebook.txt", MSLR_INPUT_ROOT="mslr",
        MULTILEXSUM_INPUT_ROOT="tasks.jsonl", CAPTURE=capture,
        PATH=str(scheduler) + os.pathsep + os.environ["PATH"],
    )
    if run_root is not None:
        env["TEAMGYM_DATAGEN_ROOT"] = run_root
    expected_root = outside / (run_root or "generated/legal_construction")
    subprocess.run(
        ["bash", str(REPO / "scripts/data_generation/slurm/legal_written_construction/submit.sh")],
        cwd=outside, env=env, check=True, capture_output=True, text=True,
    )
    records = json.loads(capture.read_text())
    assert len(records) == 4
    assert "--dependency=afterok:1" in records[1]["args"]
    assert "--dependency=afterok:2:3" in records[3]["args"]
    for record in records:
        job_env = record["env"]
        assert job_env["TEAMGYM_REPO_ROOT"] == str(REPO)
        assert job_env["TEAMGYM_DATAGEN_ROOT"] == str(expected_root)
        for name, path in (
            ("CASEBOOK_BASE_SOURCE", "casebook.txt"), ("MSLR_INPUT_ROOT", "mslr"),
            ("MULTILEXSUM_INPUT_ROOT", "tasks.jsonl"),
        ):
            assert job_env[name] == str(outside / path)

    final_input = expected_root / "final_qualified/common_datapoints.jsonl"
    final_input.parent.mkdir(parents=True)
    final_input.write_text("{}\n")
    audit = subprocess.run(
        ["bash", str(REPO / "scripts/data_generation/slurm/legal_postqualification_audit/submit.sh")],
        cwd=outside, env=env, check=True, capture_output=True, text=True,
    )
    audit_job = json.loads(capture.read_text())[-1]
    assert audit_job["env"]["TEAMGYM_DATAGEN_ROOT"] == str(expected_root)
    assert f"output={expected_root / 'postqualification_audit'}" in audit.stdout.splitlines()


def test_slurm_spooled_worker_keeps_repository_and_selected_python(tmp_path):
    spool = tmp_path / "slurm spool"
    spool.mkdir()
    script = spool / "slurm_script"
    shutil.copy2(REPO / "scripts/data_generation/slurm/legal_written_construction/seeds.sbatch", script)
    source = tmp_path / "casebook source.txt"
    source.write_text("source")
    mslr = tmp_path / "mslr source"
    mslr.mkdir()
    capture = tmp_path / "commands.jsonl"
    python = _executable(tmp_path / "selected python", """
import json, os, sys
with open(os.environ['CAPTURE'], 'a') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
""")
    subprocess.run(
        ["bash", str(script)], cwd=spool, env=_env(
            TEAMGYM_REPO_ROOT=REPO, SLURM_SUBMIT_DIR=spool, TEAMGYM_PYTHON=python,
            TEAMGYM_DATAGEN_ROOT=tmp_path / "output data", CAPTURE=capture,
            CASEBOOK_BASE_SOURCE=source, MSLR_INPUT_ROOT=mslr,
        ), check=True, capture_output=True, text=True,
    )
    commands = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(commands) == 2
    assert commands[0][:3] == ["-m", "teamgym.data.cli", "legal/casebook"]
    assert commands[1][:3] == ["-m", "teamgym.data.cli", "legal/mslr"]
    assert commands[0][commands[0].index("--base-source") + 1] == str(source)
    assert commands[1][commands[1].index("--input-root") + 1] == str(mslr)
    assert commands[1][commands[1].index("--limit") + 1] == "100"
    assert commands[1][commands[1].index("--candidate-scan-limit") + 1] == "1389"


@pytest.mark.parametrize(("partition", "output"), [
    (None, None),
    ("local_queue", "generated/sara_target_grounded100_gpt54_0714_v10"),
    ("env_file", "sara output"),
])
def test_sara_submission_keeps_recipe_and_uses_optional_local_partition(tmp_path, partition, output):
    outside = tmp_path / "outside checkout"
    audited = outside / "audit source" / "metadata"
    audited.mkdir(parents=True)
    (audited / "audited_cases.json").write_text("[{}, {}]")
    (audited / "selection_report.json").write_text("{}")
    scheduler = _scheduler(tmp_path)
    capture = tmp_path / "submitted.json"
    env = _env(
        SARA_TARGET_SLICE_AUDIT_SOURCE="audit source",
        CAPTURE=capture, PATH=str(scheduler) + os.pathsep + os.environ["PATH"],
    )
    if output is not None:
        env["SARA_TARGET_SLICE_OUTPUT"] = output
    expected_output = outside / (output or "generated/sara_target_sliced")
    if partition == "env_file":
        setup = outside / "local setup.sh"
        setup.write_text('export SARA_TARGET_SLICE_PARTITION="${SARA_TARGET_SLICE_PARTITION:-profile_queue}"\n')
        env["TEAMGYM_ENV_FILE"] = setup.name
        partition = "profile_queue"
    elif partition:
        env["SARA_TARGET_SLICE_PARTITION"] = partition
    subprocess.run(
        ["bash", str(REPO / "scripts/data_generation/slurm/sara_target_sliced/submit_target_sliced_full.sh")],
        cwd=outside, env=env, check=True, capture_output=True, text=True,
    )
    records = json.loads(capture.read_text())
    assert len(records) == 2
    assert "--array=0-1%40" in records[0]["args"]
    assert "--cpus-per-task=50" in records[1]["args"]
    assert "--dependency=afterany:1" in records[1]["args"]
    for record in records:
        assert [arg for arg in record["args"] if arg.startswith("--partition=")] == (
            [f"--partition={partition}"] if partition else []
        )
        assert any(
            f"SARA_SUBPROOF_OUTPUT={expected_output}" in arg.split(",")
            for arg in record["args"] if arg.startswith("--export=")
        )
    assert (expected_output / "metadata/submission_jobs.txt").is_file()


@pytest.mark.parametrize("setting,value,expected", [
    (None, None, "data/legal/collections/all_task_allocations_1607.jsonl"),
    ("TEAMGYM_LEGAL_DATA_ROOT", "provided data", "provided data/collections/all_task_allocations_1607.jsonl"),
    ("TEAMGYM_LEGAL_DATASET", "provided.jsonl", "provided.jsonl"),
])
def test_shared_loader_default_uses_portable_paths_and_explicit_overrides(setting, value, expected):
    env = _env(PYTHONPATH=REPO)
    if setting:
        env[setting] = value
    result = subprocess.run(
        [sys.executable, "-c", "from teamgym.core.dataset import DEFAULT_LEGAL_DATASET_PATH; print(DEFAULT_LEGAL_DATASET_PATH)"],
        env=env, check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == expected
