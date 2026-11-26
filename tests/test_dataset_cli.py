from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from teamgym.data import cli


@pytest.mark.parametrize("dataset", cli.PIPELINES)
def test_dispatch_preserves_pipeline_arguments_and_restores_argv(dataset, monkeypatch):
    original = ["teamgym-generate", dataset, "generate", "--output-root", "a path"]
    monkeypatch.setattr(sys, "argv", original)
    imported = []
    received = []

    def load(module):
        imported.append(module)
        return SimpleNamespace(main=lambda: received.append(list(sys.argv)))

    monkeypatch.setattr(cli, "import_module", load)
    cli.main()
    assert imported == [cli.PIPELINES[dataset]]
    assert received == [[f"teamgym-generate {dataset}", *original[2:]]]
    assert sys.argv is original


def test_dispatch_restores_argv_when_pipeline_exits(monkeypatch):
    original = sys.argv

    def exit_pipeline():
        sys.argv = ["changed by pipeline"]
        raise SystemExit(7)

    monkeypatch.setattr(cli, "import_module", lambda _: SimpleNamespace(main=exit_pipeline))
    with pytest.raises(SystemExit) as error:
        cli.main(["sara", "--help"])
    assert error.value.code == 7
    assert sys.argv is original


def test_help_does_not_import_optional_generation_dependencies(monkeypatch, capsys):
    def unexpected_import(_):
        pytest.fail("root help must not import a construction pipeline")

    monkeypatch.setattr(cli, "import_module", unexpected_import)
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "multilexsum" in output
    for label in ("Legal", "Finance", "Medical", "Software Engineering"):
        assert label in " ".join(output.split())


def test_listing_shows_domain_groups_and_available_commands_without_importing_pipelines(monkeypatch, capsys):
    def unexpected_import(_):
        pytest.fail("listing must not import a construction pipeline")

    monkeypatch.setattr(cli, "import_module", unexpected_import)
    original = sys.argv
    cli.main(["--list"])
    assert sys.argv is original
    output = capsys.readouterr().out
    for domain, label in cli.DOMAIN_LABELS.items():
        assert f"{label} ({domain}):" in output
    assert "legal/cuad (aliases: cuad)" in output
    for command in ("finance/findver", "finance/finlongdocqa", "medical/evidencebench",
                    "medical/mlcr", "swe/swe-atlas", "swe/deep-swe"):
        assert command in output
    assert "No construction commands available" not in output


@pytest.mark.parametrize("domain", ["finance", "medical", "swe"])
def test_domain_catalog_does_not_register_unavailable_construction_commands(domain, monkeypatch):
    def unexpected_import(_):
        pytest.fail("unregistered commands must fail before importing a module")

    monkeypatch.setattr(cli, "import_module", unexpected_import)
    with pytest.raises(SystemExit) as error:
        cli.main([f"{domain}/unregistered"])
    assert error.value.code == 2


def test_missing_dataset_fails_without_importing_a_pipeline(monkeypatch):
    monkeypatch.setattr(cli, "import_module", lambda _: pytest.fail("no pipeline selected"))
    with pytest.raises(SystemExit) as error:
        cli.main([])
    assert error.value.code == 2
