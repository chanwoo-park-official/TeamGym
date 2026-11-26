from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from teamgym.rl import launch
from teamgym.rl.preparation.check_training_data import check_prepared_splits
from teamgym.rl.data import prepare_data, stable_json_hash
from teamgym.core.schema import LEGACY_COMMON_DATAPOINT_SCHEMA_VERSION


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/tiny_normalized.jsonl"


def _prepared(
    tmp_path: Path, *, written: bool = False, legacy: bool = False, by_source: bool = True
) -> Path:
    with FIXTURE.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        row["private_context_units"][0]["text"] += "\u2028source text"
        if by_source:
            row["source"]["source_group_id"] = row["source"]["source_id"]
        if legacy:
            row["schema_version"] = LEGACY_COMMON_DATAPOINT_SCHEMA_VERSION
        if written:
            original = row["oracle"]["answer"]
            row["task_description"]["answer_type"] = "short_answer_judged"
            row["decision_set"] = ["free_response"]
            row["decision_option_definitions"] = {"free_response": "Write the conclusion and reasoning."}
            row["oracle"]["answer"] = "free_response"
            row["oracle"]["canonical_response"] = f"The correct decision is {original}."
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "prepared"
    prepare_data(input_paths=source, output_dir=output, seed=17, validation_fraction=0.25, test_fraction=0.25,
                 split_group="source_group" if by_source else "canonical_task")
    for original, renamed in (("train", "learn"), ("validation", "selection"), ("test", "heldout")):
        for suffix in (".jsonl", ".parquet", "_verl.parquet"):
            (output / (original + suffix)).rename(output / (renamed + suffix))
    return output


def _read(root: Path, split: str) -> list[dict]:
    with (root / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(root: Path, split: str, rows: list[dict]) -> None:
    (root / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    pd.DataFrame(rows).to_parquet(root / f"{split}.parquet", index=False)
    pd.DataFrame({"extra_info": rows}).to_parquet(root / f"{split}_verl.parquet", index=False)


def _check(root: Path, reward: str = "task_success") -> dict:
    return check_prepared_splits(root, train_split="learn", validation_split="selection",
                                 test_split="heldout", reward_config=reward)


@pytest.mark.parametrize("legacy", [False, True])
def test_custom_split_names_and_historical_schemas_are_supported(tmp_path, legacy):
    root = _prepared(tmp_path, legacy=legacy)
    report = _check(root)
    assert {name: entry["rows"] for name, entry in report["splits"].items()} == {
        "learn": 2, "selection": 1, "heldout": 1,
    }
    assert report["splits"]["selection"]["stored_split_labels"] == ["validation"]
    assert len(report["file_sha256"]) == 9


def test_written_answers_require_the_substantive_judge_reward(tmp_path):
    root = _prepared(tmp_path, written=True)
    assert _check(root, "written_holistic_score")["source_groups_disjoint"]
    with pytest.raises(ValueError, match="written answers require"):
        _check(root)


def test_source_split_requires_explicit_groups_and_task_only_checks_report_unknown(tmp_path):
    root = _prepared(tmp_path, by_source=False)
    report = _check(root)
    assert report["source_groups_disjoint"] is None
    assert report["rows_without_explicit_source_group"] == 4
    with pytest.raises(ValueError, match="requires an explicit source_group_id"):
        prepare_data(
            input_paths=tmp_path / "input.jsonl",
            output_dir=tmp_path / "rejected",
            split_group="source_group",
        )
    assert not (tmp_path / "rejected").exists()


def test_finite_choice_is_not_silently_sent_to_the_written_reward(tmp_path):
    with pytest.raises(ValueError, match="finite-choice answers require"):
        _check(_prepared(tmp_path), "written_holistic_score")


@pytest.mark.parametrize(("corruption", "message"), [
    ("checksum", "checksum mismatch"),
    ("source_group_id", "source_group_id overlaps"),
    ("canonical_task_id", "canonical_task_id overlaps"),
    ("instance_id", "duplicate instance ID"),
])
def test_inconsistent_payloads_and_split_leaks_are_rejected(tmp_path, corruption, message):
    root = _prepared(tmp_path)
    training = _read(root, "learn")[0]
    values = _read(root, "selection")
    row = values[0]
    payload = json.loads(row["teamgym_datapoint_json"])
    if corruption == "checksum":
        payload["task_description"]["question"] += " Changed."
    elif corruption == "instance_id":
        row["instance_id"] = payload["instance_id"] = training["instance_id"]
    else:
        payload.setdefault("_cross", {})[corruption] = row[corruption] = training[corruption]
    row["teamgym_datapoint_json"] = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    if corruption != "checksum":
        row["datapoint_sha256"] = stable_json_hash(payload)
    _write(root, "selection", values)
    with pytest.raises(ValueError, match=message):
        _check(root)


def test_parquet_disagreement_is_rejected(tmp_path):
    root = _prepared(tmp_path)
    rows = _read(root, "selection")
    rows[0]["dataset"] = "different"
    (root / "selection.jsonl").write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="inconsistent dataset"):
        _check(root)


def test_missing_reference_and_missing_format_are_rejected(tmp_path):
    root = _prepared(tmp_path, written=True)
    rows = _read(root, "selection")
    payload = json.loads(rows[0]["teamgym_datapoint_json"])
    payload["oracle"].pop("canonical_response")
    rows[0]["teamgym_datapoint_json"] = json.dumps(payload)
    rows[0]["datapoint_sha256"] = stable_json_hash(payload)
    _write(root, "selection", rows)
    with pytest.raises(ValueError, match="reference is missing"):
        _check(root, "written_holistic_score")
    (root / "selection.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        _check(root, "written_holistic_score")


def test_cli_checks_data_without_a_model_or_job(tmp_path, monkeypatch, capsys):
    root = _prepared(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("A data-only check must not start a subprocess")
    monkeypatch.setattr(launch.subprocess, "run", forbidden)
    launch.main(["--prepared-data-dir", str(root), "--train-split", "learn",
                 "--validation-split", "selection", "--test-split", "heldout",
                 "--profile", "short_context_smoke", "--reward-config", "task_success", "--check-data"])
    assert json.loads(capsys.readouterr().out)["canonical_tasks_disjoint"]


def test_training_launch_records_the_checked_plan_before_calling_backend(tmp_path, monkeypatch, capsys):
    root = _prepared(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    run = tmp_path / "run"
    called = []
    def backend(command, **settings):
        assert (run / "launch_plan.json").is_file()
        called.append((command, settings))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launch.subprocess, "run", backend)
    arguments = ["--prepared-data-dir", str(root), "--train-split", "learn", "--validation-split", "selection",
                 "--profile", "short_context_smoke", "--reward-config", "task_success",
                 "--model", str(model), "--run-dir", str(run)]
    with pytest.raises(SystemExit) as result:
        launch.main(arguments)
    assert result.value.code == 0
    assert len(called) == 1
    assert called[0][1]["env"]["TRAIN_SPLIT"] == "learn"
    assert called[0][1]["env"]["VALIDATION_SPLIT"] == "selection"
    assert json.loads(capsys.readouterr().out)["data_check"]["source_groups_disjoint"]
    with pytest.raises(SystemExit) as result:
        launch.main(arguments)
    assert result.value.code == 2 and len(called) == 1
