"""Exercise the built artifact in a fresh environment outside the checkout."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import venv
from zipfile import ZipFile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    root = tmp_path_factory.mktemp("wheel-install")
    wheels = root / "wheels"
    outside = root / "outside-checkout"
    outside.mkdir()
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
         "--no-build-isolation", "--wheel-dir", str(wheels), str(REPO_ROOT)],
        cwd=outside, env=env, check=True, capture_output=True, text=True,
    )
    wheel, = wheels.glob("rllm-*.whl")
    environment = root / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    python = environment / "bin" / "python"
    subprocess.run(
        [sys.executable, "-m", "pip", "--python", str(python), "install",
         "--no-index", "--no-deps", str(wheel)],
        cwd=outside, env=env, check=True, capture_output=True, text=True,
    )
    return dict(wheel=wheel, python=python, bin=environment / "bin", cwd=outside, env=env)


def _run(installed, *command):
    return subprocess.run(
        list(map(str, command)), cwd=installed["cwd"], env=installed["env"],
        check=True, capture_output=True, text=True,
    )


def test_wheel_contains_code_and_exact_runtime_assets_without_datasets(installed):
    with ZipFile(installed["wheel"]) as wheel:
        members = set(wheel.namelist())
        for root in ("rllm", "teamgym"):
            for source in (REPO_ROOT / root).rglob("*.py"):
                relative = source.relative_to(REPO_ROOT)
                if set(relative.parts) & {"tests", "analysis", "datasets", "run_debug", "__pycache__"}:
                    continue
                assert wheel.read(relative.as_posix()) == source.read_bytes()
        assets = list((REPO_ROOT / "teamgym/prompts").rglob("*.txt"))
        assets.extend((REPO_ROOT / "teamgym/data/software/deep_swe/runtime_repairs").glob("*.Dockerfile"))
        for pattern in ("*.sh", "*.sbatch", "*.json", "fixtures/*.jsonl"):
            assets.extend((REPO_ROOT / "teamgym/rl").glob(pattern))
        assets.extend((REPO_ROOT / "rllm").rglob("*.yaml"))
        for source in assets:
            assert wheel.read(source.relative_to(REPO_ROOT).as_posix()) == source.read_bytes()
        source_launchers = {
            source.relative_to(REPO_ROOT).as_posix()
            for pattern in ("*.sh", "*.sbatch")
            for source in (REPO_ROOT / "teamgym/rl").glob(pattern)
        }
        packaged_launchers = {
            name for name in members
            if name.startswith("teamgym/rl/") and name.endswith((".sh", ".sbatch"))
        }
        assert packaged_launchers == source_launchers
        source_modules = {
            source.relative_to(REPO_ROOT).as_posix()
            for source in (REPO_ROOT / "teamgym/rl").rglob("*.py")
            if not set(source.relative_to(REPO_ROOT).parts) & {"tests", "analysis", "__pycache__"}
        }
        packaged_modules = {
            name for name in members
            if name.startswith("teamgym/rl/") and name.endswith(".py")
        }
        assert packaged_modules == source_modules
        assert not any(
            name.startswith("teamgym/rl/") and (
                (Path(name).name.startswith("campaign_") and name.endswith(".json"))
                or Path(name).name in {"CUAD_WRITTEN_DATASET_LOG.md", "QUERY_PRIVATE_EXPERIMENT_LOG.md"}
            )
            for name in members
        )
        assert "rllm/registry/dataset_registry.json" not in members
        assert not any(set(Path(name).parts) & {"tests", "analysis", "datasets", "run_debug"} for name in members)
        assert not any(name.startswith("verl/") for name in members)
        assert not any(name.startswith("examples_rl/") for name in members)
        assert not any(name.startswith("collabgym_fully_collaborative/") for name in members)
        assert not any(name.endswith((".parquet", ".pt", ".pth", ".safetensors", ".log", ".ipynb")) for name in members)


def test_installed_imports_and_prompt_lookup_use_only_the_wheel(installed):
    code = """
from importlib import import_module, util
from pathlib import Path
import sys
assert util.find_spec('collabgym_fully_collaborative') is None
from teamgym.core.io import prompt_text
for name in ('teamgym.core.schema',
             'teamgym.core.dataset',
             'teamgym.core.io',
             'teamgym.core.information_allocation',
             'teamgym.core.participant_roles',
             'teamgym.data.registry',
             'teamgym.scoring.short_answers',
             'teamgym.scoring.written_answers',
             'teamgym.rl.paper', 'teamgym.rl.rollout.action_protocol'):
    assert Path(import_module(name).__file__).is_relative_to(sys.prefix), name
spec = util.find_spec('rllm')
assert Path(spec.origin).is_relative_to(sys.prefix)
assert (Path(spec.origin).parent / 'trainer/config/agent_ppo_trainer.yaml').is_file()
import teamgym as package
prompts = Path(package.__file__).parent / 'prompts'
for source in prompts.rglob('*.txt'):
    assert prompt_text(source.relative_to(prompts).as_posix()) == source.read_text()
assert 'torch' not in sys.modules
assert 'openai' not in sys.modules
"""
    _run(installed, installed["python"], "-I", "-c", code)


def test_installed_finance_role_helpers_import_without_legal_or_optional_dependencies(installed):
    code = """
from pathlib import Path
import sys
from teamgym.core import participant_roles
from teamgym.data.long_context_reasoning import finance_participant_roles
for module in (participant_roles, finance_participant_roles):
    assert Path(module.__file__).is_relative_to(sys.prefix)
assert finance_participant_roles.role_name_preview is participant_roles.role_name_preview
assert participant_roles.role_name_from_information_features(['cash resources'], 1) == 'cash resources reviewer'
blocked = ('teamgym.data.legal', 'rllm', 'verl', 'torch', 'pandas', 'openai', 'tiktoken')
assert not [name for name in sys.modules
            if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]
"""
    _run(installed, installed["python"], "-I", "-c", code)


def test_installed_sara_proof_helpers_work_without_loading_generation(installed):
    code = """
from pathlib import Path
import sys
from teamgym.data.legal.sara import prolog, proof_slicing
for module in (prolog, proof_slicing):
    assert Path(module.__file__).is_relative_to(sys.prefix)
assert 'teamgym.data.legal.sara.subproof_generation' not in sys.modules
assert 'teamgym.llm.openai_client' not in sys.modules
program = 'income(alice,100). income(bob,200). tax(P,T) :- income(P,I), T is I/10.'
proof = proof_slicing.target_proof_slice(
    {'trace': ['income(alice,100)', 'income(bob,200)', 'tax(alice,10)']},
    {'goal': 'tax(alice,10)', 'trace_position': 2}, program,
)
assert proof['successful_trace_goals'] == ['income(alice,100)', 'tax(alice,10)']
from teamgym.data.legal.sara import subproof_generation
assert subproof_generation.parse_prolog_clauses is prolog.parse_prolog_clauses
assert subproof_generation.target_proof_slice is proof_slicing.target_proof_slice
assert not {'torch', 'ray', 'vllm', 'openai', 'pandas'} & sys.modules.keys()
"""
    _run(installed, installed["python"], "-I", "-c", code)


@pytest.mark.parametrize("command", [
    "teamgym-generate", "teamgym-evaluate", "teamgym-judge",
    "teamgym-validate-frozen", "teamgym-paper", "teamgym-train",
])
def test_installed_console_help_without_optional_dependencies(installed, command):
    result = _run(installed, installed["bin"] / command, "--help")
    assert "usage:" in result.stdout


@pytest.mark.parametrize("dataset", [
    "casebook", "mslr", "sara", "multilexsum",
    "legal/casebook", "legal/mslr", "legal/sara", "legal/multilexsum",
    "finance/findver", "finance/finlongdocqa", "medical/evidencebench", "medical/mlcr",
    "swe/swe-atlas", "swe/deep-swe",
])
def test_installed_construction_help_without_optional_dependencies(installed, dataset):
    result = _run(installed, installed["bin"] / "teamgym-generate", dataset, "--help")
    assert "usage:" in result.stdout


def test_installed_generation_catalog_lists_four_domains_without_optional_dependencies(installed):
    result = _run(installed, installed["bin"] / "teamgym-generate", "--list")
    for label in ("Legal (legal):", "Finance (finance):", "Medical (medical):", "Software Engineering (swe):"):
        assert label in result.stdout
    assert "legal/sara (aliases: sara)" in result.stdout


@pytest.mark.parametrize("regime", ["legal-participant", "legal-coordinator", "legal-joint", "finance-handoff"])
@pytest.mark.parametrize("evaluate", [False, True])
def test_installed_paper_dry_run_from_another_directory(installed, regime, evaluate):
    command = [installed["bin"] / "teamgym-paper", "--regime", regime,
               "--prepared-data-dir", "missing-data", "--model", "missing-model",
               "--run-dir", "missing-run", "--dry-run"]
    if regime == "legal-coordinator":
        command += ["--fixed-participant-model", "missing-participant"]
    if evaluate:
        command.append("--evaluate")
    plan = json.loads(_run(installed, *command).stdout)
    assert plan["cwd"] == str(installed["cwd"])
    assert Path(plan["command"][1]).is_relative_to(installed["python"].parents[1])
    assert Path(plan["command"][1]).is_file()
    assert plan["environment"]["TOTAL_TRAINING_STEPS"] == "81"
    assert plan["optimizer_updates"] == (0 if evaluate else 80)
    assert not list(installed["cwd"].iterdir())


@pytest.mark.parametrize("regime", ["legal-joint", "finance-handoff"])
def test_installed_shell_chain_reaches_training_without_starting_a_model(installed, tmp_path, regime):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    plan = json.loads(_run(
        installed, installed["bin"] / "teamgym-paper", "--regime", regime,
        "--prepared-data-dir", prepared, "--model", tmp_path / "model",
        "--run-dir", tmp_path / "run", "--dry-run",
    ).stdout)
    capture = tmp_path / "arguments.json"
    stub = tmp_path / "python-stub"
    stub.write_text(
        f"#!{installed['python']}\nimport json, sys\n"
        "assert sys.argv[1] == '-m'\n"
        "assert sys.argv[2] in ('teamgym.rl.train', 'teamgym.rl.tools.plot_reward')\n"
        "if sys.argv[2] == 'teamgym.rl.train':\n"
        f"    open({str(capture)!r}, 'w').write(json.dumps(sys.argv[3:]))\n"
    )
    stub.chmod(0o700)
    subprocess.run(
        plan["command"], cwd=installed["cwd"],
        env={**installed["env"], **plan["environment"], "TEAMGYM_PYTHON": str(stub)},
        check=True, capture_output=True, text=True,
    )
    arguments = json.loads(capture.read_text())
    assert "trainer.total_training_steps=81" in arguments
    assert "trainer.n_gpus_per_node=4" in arguments
    assert "data.train_batch_size=8" in arguments
    assert f"actor_rollout_ref.rollout.n={16 if regime == 'finance-handoff' else 8}" in arguments
    assert f"+teamgym.prepared_data_dir={prepared}" in arguments


@pytest.mark.parametrize(("mode", "profile"), [
    ("joint", "short_context_smoke"),
    ("participant_only", "finance_riu_participant_handoff_15"),
])
def test_installed_generic_training_dry_run_without_optional_dependencies(installed, mode, profile):
    args = ["--prepared-data-dir", "missing-data", "--model", "missing-model",
            "--run-dir", "missing-run", "--profile", profile, "--reward-config", "task_success",
            "--training-mode", mode, "--dry-run"]
    plan = json.loads(_run(installed, installed["bin"] / "teamgym-train", *args).stdout)
    assert plan["cwd"] == str(installed["cwd"])
    assert Path(plan["command"][1]).is_relative_to(installed["python"].parents[1])
    assert plan["environment"]["TRAINING_MODE"] == mode
    assert not list(installed["cwd"].iterdir())
    code = (
        "import sys; from teamgym.rl.launch import main; main(sys.argv[1:]); "
        "blocked = ('torch', 'pandas', 'numpy', 'openai', 'rllm', 'verl', "
        "'teamgym.rl.train', 'teamgym.rl.preparation.check_training_data'); "
        "assert not [name for name in sys.modules "
        "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]"
    )
    _run(installed, installed["python"], "-I", "-c", code, *args)


def test_installed_generic_launcher_forwards_custom_settings_to_the_trainer(installed, tmp_path):
    prepared = tmp_path / "custom data"
    model = tmp_path / "custom model"
    run = tmp_path / "custom run"
    plan = json.loads(_run(
        installed, installed["bin"] / "teamgym-train",
        "--prepared-data-dir", prepared, "--model", model, "--run-dir", run,
        "--profile", "short_context_smoke", "--reward-config", "task_success",
        "--train-split", "learn", "--validation-split", "selection",
        "--devices", "2,3", "--tensor-parallel-size", "2", "--batch-size", "2",
        "--updates", "5", "--epochs", "9", "--dry-run",
    ).stdout)
    capture = tmp_path / "generic_arguments.json"
    stub = tmp_path / "python-stub"
    stub.write_text(
        f"#!{installed['python']}\nimport json, sys\n"
        "assert sys.argv[1:3] == ['-m', 'teamgym.rl.train']\n"
        f"open({str(capture)!r}, 'w').write(json.dumps(sys.argv[3:]))\n"
    )
    stub.chmod(0o700)
    subprocess.run(plan["command"], cwd=installed["cwd"],
                   env={**installed["env"], **plan["environment"], "TEAMGYM_PYTHON": str(stub)},
                   check=True, capture_output=True, text=True)
    arguments = json.loads(capture.read_text())
    for expected in (
        "trainer.total_training_steps=6", "trainer.total_epochs=9",
        "trainer.n_gpus_per_node=2", "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8", "actor_rollout_ref.rollout.n=4",
        "+teamgym.train_split=learn", "+teamgym.validation_split=selection",
        "+teamgym.rollout_profile=short_context_smoke",
        "+teamgym.training_mode=joint", "+teamgym.reward_config=task_success",
        f"+teamgym.prepared_data_dir={prepared}",
    ):
        assert expected in arguments
