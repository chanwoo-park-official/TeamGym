from __future__ import annotations

import json
from pathlib import Path

import pytest

from teamgym.rl.tools.validate_written_rollouts import (
    RolloutValidationError,
    validate_rollout_set,
)


def _write_dataset(path: Path, instance_ids: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"instance_id": value}) + "\n" for value in instance_ids),
        encoding="utf-8",
    )


def _write_rollout(
    rollout_dir: Path,
    instance_id: str,
    *,
    episode_id: str | None = None,
    evaluation_instance_id: str | None = None,
) -> None:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_id": episode_id or f"episode-{instance_id}",
        "normalized_datapoint": {"instance_id": instance_id},
        "evaluation": {
            "instance_id": evaluation_instance_id or instance_id,
        },
    }
    (rollout_dir / f"{payload['episode_id']}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


@pytest.mark.parametrize("count", [26, 36])
def test_validate_rollout_set_derives_split_size(
    tmp_path: Path, count: int
) -> None:
    dataset = tmp_path / "validation.jsonl"
    rollout_dir = tmp_path / "rollouts"
    instance_ids = [f"task-{index:02d}" for index in range(count)]
    _write_dataset(dataset, instance_ids)
    for instance_id in instance_ids:
        _write_rollout(rollout_dir, instance_id)

    result = validate_rollout_set(dataset, rollout_dir)

    assert result["expected_count"] == count
    assert result["rollout_count"] == count
    assert result["instance_ids_match"] is True


def test_validate_rollout_set_rejects_missing_and_extra_tasks(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "validation.jsonl"
    rollout_dir = tmp_path / "rollouts"
    _write_dataset(dataset, ["task-a", "task-b"])
    _write_rollout(rollout_dir, "task-a")
    _write_rollout(rollout_dir, "task-c")

    with pytest.raises(RolloutValidationError, match="task identities differ"):
        validate_rollout_set(dataset, rollout_dir)


def test_validate_rollout_set_rejects_duplicate_instance_ids(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "validation.jsonl"
    rollout_dir = tmp_path / "rollouts"
    _write_dataset(dataset, ["task-a", "task-b"])
    _write_rollout(rollout_dir, "task-a", episode_id="episode-a")
    _write_rollout(rollout_dir, "task-a", episode_id="episode-b")

    with pytest.raises(RolloutValidationError, match="duplicate instance_ids"):
        validate_rollout_set(dataset, rollout_dir)


def test_validate_rollout_set_rejects_malformed_json(tmp_path: Path) -> None:
    dataset = tmp_path / "validation.jsonl"
    rollout_dir = tmp_path / "rollouts"
    _write_dataset(dataset, ["task-a"])
    rollout_dir.mkdir()
    (rollout_dir / "broken.json").write_text("{", encoding="utf-8")

    with pytest.raises(RolloutValidationError, match="Invalid rollout JSON"):
        validate_rollout_set(dataset, rollout_dir)


def test_validate_rollout_set_rejects_conflicting_embedded_ids(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "validation.jsonl"
    rollout_dir = tmp_path / "rollouts"
    _write_dataset(dataset, ["task-a"])
    _write_rollout(
        rollout_dir,
        "task-a",
        evaluation_instance_id="task-b",
    )

    with pytest.raises(RolloutValidationError, match="Conflicting instance_ids"):
        validate_rollout_set(dataset, rollout_dir)
