from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from teamgym.rl.preparation.check_paper_data import check_prepared_data
from teamgym.rl.data import stable_json_hash
from teamgym.rl.paper import REPO_ROOT, build_plan, main
from rllm.parser.chat_template_parser import LlamaChatTemplateParser


def _args(tmp_path: Path, regime: str = "legal-joint", **overrides) -> argparse.Namespace:
    values = dict(
        regime=regime, prepared_data_dir=tmp_path / "prepared data",
        run_dir=tmp_path / "new run", model=str(tmp_path / "base-model"),
        fixed_participant_model=None, seed=29, updates=80, devices="0,1,2,3",
        evaluate=False, eval_checkpoint=None, eval_split=None,
    )
    return argparse.Namespace(**{**values, **overrides})


def test_paper_recipes_keep_independent_legal_policies_and_final_finance_caps(tmp_path):
    legal = build_plan(_args(tmp_path))
    finance = build_plan(_args(tmp_path, "finance-handoff"))
    assert "+teamgym.two_policy=True" in legal["command"]
    assert legal["environment"]["USE_KL_LOSS"] == "True"
    assert finance["environment"]["USE_KL_LOSS"] == "False"
    assert finance["environment"]["DATA_MAX_RESPONSE_LENGTH"] == "750"
    assert finance["environment"]["PPO_MINI_BATCH_SIZE"] == "128"
    assert finance["environment"]["ACTOR_MODEL_DTYPE"] == "bfloat16"
    assert "findver" in finance["command"][-1]
    for plan in (legal, finance):
        assert plan["environment"]["TOTAL_TRAINING_STEPS"] == "81"
        assert plan["environment"]["TRAIN_BATCH_SIZE"] == "8"
        assert plan["environment"]["FORMAT_CREDIT_ASSIGNMENT"] == "per_attempt"
        assert plan["environment"]["WRITTEN_JUDGE_FAIL_ON_ERROR"] == "True"


def test_stage_c_requires_and_routes_immutable_selected_participant(tmp_path):
    with pytest.raises(ValueError, match="fixed-participant-model"):
        build_plan(_args(tmp_path, "legal-coordinator"))
    plan = build_plan(_args(tmp_path, "legal-coordinator", fixed_participant_model="/selected-stage-b", devices="4,5,6,7"))
    assert plan["environment"]["PARTICIPANT_BACKEND"] == "external_openai"
    assert plan["environment"]["FIXED_PARTICIPANT_MODEL"] == "/selected-stage-b"
    assert plan["environment"]["FROZEN_PARTICIPANT_GPU"] == "7"
    participant = build_plan(_args(tmp_path, "legal-participant"))
    assert participant["environment"]["FIXED_COORDINATOR_TEMPERATURE"] == "0.0"


@pytest.mark.parametrize("regime,split", [("legal-joint", "validation_maud"), ("finance-handoff", "test")])
def test_endpoint_evaluation_only_loads_weights(tmp_path, regime, split):
    checkpoint = tmp_path / "global_step_70"
    plan = build_plan(_args(tmp_path, regime, evaluate=True, eval_checkpoint=checkpoint))
    env = plan["environment"]
    assert env["VAL_ONLY"] == "True"
    assert env["VALIDATION_SPLIT"] == split
    assert env["RESUME_FROM_PATH"] == str(checkpoint)
    assert env["ROLLOUT_N"] == "1"
    assert env["SAVE_FREQ"] == "-1"
    assert plan["optimizer_updates"] == 0
    assert "actor_rollout_ref.actor.checkpoint.load_contents=['model']" in plan["command"]
    assert not any("retain_" in arg for arg in plan["command"])


@pytest.mark.parametrize("overrides", [
    {"eval_split": "validation_maud"},
    {"eval_checkpoint": Path("global_step_80")},
    {"evaluate": True, "eval_split": "test"},
    {"updates": 81}, {"devices": "0,1,2,2"},
])
def test_rejects_ambiguous_or_wrong_regime_settings(tmp_path, overrides):
    with pytest.raises(ValueError):
        build_plan(_args(tmp_path, **overrides))


def test_dry_run_needs_no_data_model_or_training_import(tmp_path, capsys):
    main(["--regime", "finance-handoff", "--prepared-data-dir", str(tmp_path / "missing-data"),
          "--model", str(tmp_path / "missing-model"), "--run-dir", str(tmp_path / "missing-run"), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["environment"]["DATA_MAX_RESPONSE_LENGTH"] == "750"
    assert list(tmp_path.iterdir()) == []
    # In a fresh interpreter, neither torch nor the trainer should be imported.
    code = "import sys; from teamgym.rl.paper import main; main(sys.argv[1:]); assert 'torch' not in sys.modules; assert 'teamgym.rl.train' not in sys.modules"
    subprocess.run([sys.executable, "-c", code, "--regime", "legal-joint", "--prepared-data-dir", str(tmp_path / "none"), "--model", "/none", "--run-dir", str(tmp_path / "run"), "--dry-run"], cwd=REPO_ROOT, stdout=subprocess.DEVNULL, check=True)


def _write_split(root: Path, split: str, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(rows).to_parquet(root / f"{split}.parquet", index=False)
    pd.DataFrame({"extra_info": rows}).to_parquet(root / f"{split}_verl.parquet", index=False)


def _prepared(root: Path, finance: bool = False) -> dict[str, list[dict]]:
    splits = ("train", "validation", "test") if finance else ("train", "validation_cuad", "validation_maud")
    result = {}
    for split in splits:
        dataset = ("finlongdocqa" if split == "test" else "findver") if finance else ("maud" if split == "validation_maud" else "cuad")
        count = 4 if finance and split == "test" else 2
        payload = {
            "instance_id": split + "_task", "dataset": dataset,
            "_cross": {"canonical_task_id": split + "_task", "source_group_id": split + "_document"},
            "task_description": {"answer_type": "short_answer_judged"},
            "participants": [{"participant_id": f"p{i}", "owned_rui_ids": [f"r{i}"], "private_context_unit_ids": [f"c{i}"]} for i in range(count)],
            "private_context_units": [{"context_unit_id": f"c{i}", "text": "source\u2028text", "hidden_contains_rui_ids": [f"r{i}"]} for i in range(count)],
        }
        row = {
            "instance_id": payload["instance_id"], "dataset": dataset,
            "canonical_task_id": payload["_cross"]["canonical_task_id"],
            "source_group_id": payload["_cross"]["source_group_id"],
            "context_variant": "riu_only" if finance else "full_context",
            "participant_count": count, "answer_type": "short_answer_judged",
            "split": "validation" if split.startswith("validation_") else split,
            "datapoint_sha256": stable_json_hash(payload),
            "teamgym_datapoint_json": json.dumps(payload, ensure_ascii=False),
        }
        result[split] = [row]
        _write_split(root, split, [row])
    return result


@pytest.mark.parametrize("finance", [False, True])
def test_checks_all_prepared_formats_without_opening_models(tmp_path, finance):
    _prepared(tmp_path, finance)
    report = check_prepared_data(tmp_path, finance=finance)
    assert report["canonical_tasks_disjoint"] and report["source_groups_disjoint"]
    assert len(report["file_sha256"]) == 9
    assert not report["matches_paper_split_counts"]  # Tiny fixtures are identified.


def test_existing_run_is_preserved_before_any_training_launch(tmp_path, capsys):
    prepared = tmp_path / "prepared"
    _prepared(prepared)
    run = tmp_path / "existing-run"
    run.mkdir()
    marker = run / "keep.txt"
    marker.write_text("existing artifact")
    with pytest.raises(SystemExit) as error:
        main(["--regime", "legal-joint", "--prepared-data-dir", str(prepared),
              "--model", "/missing-model", "--run-dir", str(run)])
    assert error.value.code == 2
    assert "already exists" in capsys.readouterr().err
    assert marker.read_text() == "existing artifact"


@pytest.mark.parametrize("damage", ["checksum", "parquet", "source_overlap", "task_overlap", "finite_choice"])
def test_prepared_data_rejects_corruption_or_leakage(tmp_path, damage):
    rows = _prepared(tmp_path)
    split = "validation_cuad"
    row = deepcopy(rows[split][0])
    if damage == "parquet":
        row["instance_id"] = "wrong-parquet-id"
        pd.DataFrame([row]).to_parquet(tmp_path / f"{split}.parquet", index=False)
    else:
        payload = json.loads(row["teamgym_datapoint_json"])
        if damage == "checksum":
            payload["changed"] = True
        elif damage == "finite_choice":
            payload["task_description"] = {}
            row["datapoint_sha256"] = stable_json_hash(payload)
        else:
            field = "source_group_id" if damage == "source_overlap" else "canonical_task_id"
            row[field] = rows["train"][0][field]
            payload["_cross"][field] = row[field]
            row["datapoint_sha256"] = stable_json_hash(payload)
        row["teamgym_datapoint_json"] = json.dumps(payload)
        _write_split(tmp_path, split, [row])
    with pytest.raises(ValueError):
        check_prepared_data(tmp_path, finance=False)


@pytest.mark.parametrize("regime", ["legal-joint", "finance-handoff"])
def test_shell_backends_receive_the_public_recipe_without_models(tmp_path, regime, monkeypatch):
    args = _args(tmp_path, regime)
    args.prepared_data_dir.mkdir()
    outside = tmp_path / "outside-checkout"
    outside.mkdir()
    monkeypatch.chdir(outside)
    plan = build_plan(args)
    assert plan["cwd"] == str(outside)
    capture = tmp_path / "captured.json"
    stub = tmp_path / "fake-python"
    stub.write_text(f"#!{sys.executable}\nimport json, os, sys\nif sys.argv[1:3] == ['-m', 'teamgym.rl.train']:\n    open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[3:]))\n")
    stub.chmod(0o700)
    env = {**os.environ, **plan["environment"], "TEAMGYM_PYTHON": str(stub), "CAPTURE": str(capture)}
    subprocess.run(plan["command"], cwd=plan["cwd"], env=env, check=True, capture_output=True, text=True)
    received = json.loads(capture.read_text())
    assert "trainer.total_training_steps=81" in received
    assert f"+teamgym.prepared_data_dir={args.prepared_data_dir}" in received
    assert "trainer.logger=['console']" in received
    assert "actor_rollout_ref.model.lora_rank=0" in received
    assert "actor_rollout_ref.actor.clip_ratio_c=3.0" in received
    if regime == "legal-joint":
        assert "+teamgym.two_policy=True" in received
        assert "actor_rollout_ref.actor.checkpoint.save_contents=['model']" in received
    else:
        assert "data.max_response_length=750" in received
        assert "actor_rollout_ref.rollout.n=16" in received


@pytest.mark.parametrize("text", ['{"action":"finalize","utterance":"done"}', '{"action":', "", "<think>unchanged</think> {bad JSON}"])
def test_llama_finance_parser_preserves_valid_and_malformed_completions(text):
    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            assert ids == [128006, 1, 128009]
            assert skip_special_tokens is True
            return text

    parser = LlamaChatTemplateParser.__new__(LlamaChatTemplateParser)
    parser.tokenizer = Tokenizer()
    ids = [128006, 1, 128009]
    assert parser.parse_completion(ids) == {"content": text, "reasoning": "", "tool_calls": []}
    assert ids == [128006, 1, 128009]
