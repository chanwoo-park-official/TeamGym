from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from teamgym.rl.training.checkpoint_retention import (
    prune_model_checkpoints,
    verify_model_checkpoint,
    write_json_atomic,
)
from teamgym.rl.training.trainer_compat import TeamGymAgentWorkflowPPOTrainer


def _write_checkpoint(
    root: Path,
    *,
    step: int,
    world_size: int = 4,
    two_policy: bool = False,
) -> Path:
    checkpoint = root / f"global_step_{step}"
    role_dirs = (
        [checkpoint / "actor" / "coordinator", checkpoint / "actor" / "participant"]
        if two_policy
        else [checkpoint / "actor"]
    )
    for role_dir in role_dirs:
        role_dir.mkdir(parents=True)
        for rank in range(world_size):
            (role_dir / f"model_world_size_{world_size}_rank_{rank}.pt").write_bytes(
                f"rank-{rank}".encode()
            )
    (checkpoint / "data.pt").write_bytes(b"data")
    return checkpoint


@pytest.mark.parametrize("two_policy", [False, True])
def test_verify_model_checkpoint_accepts_complete_shards(
    tmp_path: Path, two_policy: bool
) -> None:
    checkpoint = _write_checkpoint(tmp_path, step=20, two_policy=two_policy)

    result = verify_model_checkpoint(
        checkpoint, world_size=4, two_policy=two_policy
    )

    assert result["step"] == 20
    assert result["two_policy"] is two_policy
    assert len(result["model_shards"]) == (8 if two_policy else 4)


def test_verify_model_checkpoint_rejects_missing_role_shard(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path, step=30, two_policy=True)
    (checkpoint / "actor" / "participant" / "model_world_size_4_rank_3.pt").unlink()

    with pytest.raises(RuntimeError, match="Incomplete participant checkpoint"):
        verify_model_checkpoint(checkpoint, world_size=4, two_policy=True)


def test_prune_model_checkpoints_only_removes_unprotected_step_dirs(
    tmp_path: Path,
) -> None:
    for step in (10, 20, 30):
        _write_checkpoint(tmp_path, step=step)
    unrelated = tmp_path / "analysis"
    unrelated.mkdir()
    marker = tmp_path / "latest_checkpointed_iteration.txt"
    marker.write_text("30", encoding="utf-8")

    removed = prune_model_checkpoints(tmp_path, {10, 30})

    assert [path.name for path in removed] == ["global_step_20"]
    assert (tmp_path / "global_step_10").is_dir()
    assert (tmp_path / "global_step_30").is_dir()
    assert unrelated.is_dir()
    assert marker.is_file()


def test_write_json_atomic_replaces_payload_without_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    write_json_atomic(path, {"best_step": 10})
    write_json_atomic(path, {"best_step": 30})

    assert json.loads(path.read_text(encoding="utf-8")) == {"best_step": 30}
    assert list(tmp_path.glob(".selection.json.tmp-*")) == []


def test_selection_resume_ignores_uncommitted_best_checkpoint(tmp_path: Path) -> None:
    complete = _write_checkpoint(tmp_path, step=20)
    (complete / "_SUCCESS").write_text("{}\n", encoding="utf-8")
    partial = _write_checkpoint(tmp_path, step=30)
    (partial / "actor" / "model_world_size_4_rank_3.pt").unlink()
    write_json_atomic(
        tmp_path / "checkpoint_selection.json",
        {
            "best_step": 30,
            "best_score": 0.7,
            "latest_checkpoint_complete": False,
            "scores": {"20": 0.5, "30": 0.7},
        },
    )
    trainer = object.__new__(TeamGymAgentWorkflowPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "teamgym": {"retain_cuad_best_and_latest": True},
            "trainer": {"default_local_dir": str(tmp_path)},
        }
    )
    trainer._teamgym_checkpoint_scores = {}
    trainer._teamgym_best_checkpoint_step = None
    trainer._teamgym_best_checkpoint_score = None

    trainer._load_teamgym_checkpoint_selection()

    assert trainer._teamgym_best_checkpoint_step == 20
    assert trainer._teamgym_best_checkpoint_score == 0.5


def test_selection_ignores_final_validation_without_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = TeamGymAgentWorkflowPPOTrainer.__mro__[1]
    metric = "val/in/unknown/teamgym_holistic_score_cuad"
    monkeypatch.setattr(
        parent,
        "_validate_agent",
        lambda self, epoch=0: {metric: 0.9},
    )
    trainer = object.__new__(TeamGymAgentWorkflowPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "teamgym": {
                "retain_cuad_best_and_latest": True,
                "checkpoint_selection_metric": metric,
            },
            "trainer": {
                "default_local_dir": str(tmp_path),
                "save_freq": 10,
            },
        }
    )
    trainer.global_steps = 81
    trainer._teamgym_checkpoint_scores = {80: 0.4}
    trainer._teamgym_best_checkpoint_step = 80
    trainer._teamgym_best_checkpoint_score = 0.4
    trainer._teamgym_previous_best_checkpoint_step = None

    result = trainer._validate_agent(epoch=0)

    assert result == {metric: 0.9}
    assert trainer._teamgym_checkpoint_scores == {80: 0.4}
    assert trainer._teamgym_best_checkpoint_step == 80
    assert not (tmp_path / "checkpoint_selection.json").exists()
