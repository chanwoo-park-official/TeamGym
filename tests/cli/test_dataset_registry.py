from __future__ import annotations

from tests._paths import REPO_ROOT

from importlib import import_module
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from teamgym.data.cli import main
from teamgym.data.registry import (
    DOMAIN_LABELS, PIPELINE_REGISTRY, PIPELINE_SPECS, PipelineSpec, build_registry,
)


def test_domain_keys_match_the_four_paper_categories():
    assert DOMAIN_LABELS == {
        "legal": "Legal", "finance": "Finance", "medical": "Medical",
        "swe": "Software Engineering",
    }


@pytest.mark.parametrize("dataset", ["cuad", "maud", "casebook", "mslr", "sara", "multilexsum"])
def test_legacy_alias_and_qualified_command_use_the_existing_constructor(dataset):
    legacy = PIPELINE_REGISTRY[dataset]
    assert PIPELINE_REGISTRY[f"legal/{dataset}"] is legacy
    assert legacy.module == f"teamgym.data.legal.{dataset}.pipeline"


@pytest.mark.parametrize("dataset", ["cuad", "maud"])
def test_contract_generation_help_advertises_partition(dataset, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([f"legal/{dataset}", "--help"])

    assert exit_info.value.code == 0
    assert f"teamgym-generate legal/{dataset} partition --help" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["legal/cuad", "cuad", "legal/maud", "maud"])
def test_contract_partition_stage_passes_existing_options(command, monkeypatch):
    dataset = command.rsplit("/", 1)[-1]
    module = import_module(f"teamgym.data.legal.{dataset}.partition")
    run_partition = Mock(return_value={})
    monkeypatch.setattr(module, "partition_accepted_trials", run_partition)
    arguments = [command, "partition", "--accepted-path", "accepted.jsonl", "--seed", "17"]
    if dataset == "maud":
        arguments += ["--context-mode", "budgeted", "--context-budget-tokens", "16000"]

    main(arguments)

    run_partition.assert_called_once()
    assert run_partition.call_args.kwargs["accepted_path"] == "accepted.jsonl"
    assert run_partition.call_args.kwargs["seed"] == 17
    if dataset == "maud":
        assert run_partition.call_args.kwargs["context_mode"] == "budgeted"
        assert run_partition.call_args.kwargs["context_budget_tokens"] == 16000


def test_datasets_in_different_domains_can_share_a_name_without_alias_collisions():
    legal = PipelineSpec("legal", "example", "example.legal", aliases=("example",))
    finance = PipelineSpec("finance", "example", "example.finance")
    registry = build_registry([legal, finance])
    assert registry["legal/example"] is legal
    assert registry["finance/example"] is finance
    assert registry["example"] is legal


@pytest.mark.parametrize("specs", [
    [PipelineSpec("legal", "example", "example.first"), PipelineSpec("legal", "example", "example.second")],
    [PipelineSpec("legal", "first", "example.first", ("shared",)), PipelineSpec("finance", "second", "example.second", ("shared",))],
])
def test_ambiguous_registrations_are_rejected_before_dispatch(specs):
    with pytest.raises(ValueError, match="Duplicate pipeline command"):
        build_registry(specs)


def test_new_constructors_use_one_of_the_paper_domain_keys():
    with pytest.raises(ValueError, match="Unknown domain"):
        build_registry([PipelineSpec("unknown", "example", "example.pipeline")])


def test_catalog_contains_only_constructors_present_in_this_checkout():
    root = REPO_ROOT
    for spec in PIPELINE_SPECS:
        assert (root / (spec.module.replace(".", "/") + ".py")).is_file()


def test_registry_import_does_not_load_pipelines_or_optional_dependencies():
    code = """
import sys
from teamgym.data.registry import DOMAIN_LABELS, PIPELINE_REGISTRY
assert len(DOMAIN_LABELS) == 4
assert 'legal/cuad' in PIPELINE_REGISTRY
blocked = ('teamgym.data.legal', 'openai', 'pandas', 'torch', 'rllm', 'verl')
assert not [name for name in sys.modules if any(name == p or name.startswith(p + '.') for p in blocked)]
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True,
    )
