#!/usr/bin/env python3
"""Validate that a written-answer evaluation covers its exact dataset split."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


class RolloutValidationError(ValueError):
    """Raised when an evaluation directory is incomplete or inconsistent."""


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def load_expected_instance_ids(dataset_jsonl: Path) -> list[str]:
    if not dataset_jsonl.is_file():
        raise RolloutValidationError(f"Missing dataset JSONL: {dataset_jsonl}")

    instance_ids: list[str] = []
    with dataset_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RolloutValidationError(
                    f"Invalid dataset JSON at {dataset_jsonl}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RolloutValidationError(
                    f"Dataset row {line_number} is not a JSON object"
                )
            instance_id = str(row.get("instance_id") or "").strip()
            if not instance_id:
                raise RolloutValidationError(
                    f"Dataset row {line_number} has no instance_id"
                )
            instance_ids.append(instance_id)

    if not instance_ids:
        raise RolloutValidationError(f"Dataset contains no rows: {dataset_jsonl}")
    duplicates = _duplicates(instance_ids)
    if duplicates:
        raise RolloutValidationError(
            f"Dataset has duplicate instance_ids: {duplicates[:5]}"
        )
    return instance_ids


def _rollout_instance_id(payload: dict[str, Any], path: Path) -> str:
    normalized_id = str(
        (payload.get("normalized_datapoint") or {}).get("instance_id") or ""
    ).strip()
    evaluation_id = str(
        (payload.get("evaluation") or {}).get("instance_id") or ""
    ).strip()
    if normalized_id and evaluation_id and normalized_id != evaluation_id:
        raise RolloutValidationError(
            f"Conflicting instance_ids in {path}: "
            f"normalized={normalized_id!r}, evaluation={evaluation_id!r}"
        )
    instance_id = normalized_id or evaluation_id
    if not instance_id:
        raise RolloutValidationError(f"Rollout has no instance_id: {path}")
    return instance_id


def load_rollout_instance_ids(rollout_dir: Path) -> list[str]:
    if not rollout_dir.is_dir():
        raise RolloutValidationError(f"Missing rollout directory: {rollout_dir}")

    files = sorted(rollout_dir.glob("*.json"))
    if not files:
        raise RolloutValidationError(f"No rollout JSON files in {rollout_dir}")

    instance_ids: list[str] = []
    episode_ids: list[str] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RolloutValidationError(
                f"Invalid rollout JSON in {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RolloutValidationError(f"Rollout is not a JSON object: {path}")
        instance_ids.append(_rollout_instance_id(payload, path))
        episode_id = str(payload.get("episode_id") or "").strip()
        if not episode_id:
            raise RolloutValidationError(f"Rollout has no episode_id: {path}")
        episode_ids.append(episode_id)

    duplicate_instances = _duplicates(instance_ids)
    if duplicate_instances:
        raise RolloutValidationError(
            f"Rollouts have duplicate instance_ids: {duplicate_instances[:5]}"
        )
    duplicate_episodes = _duplicates(episode_ids)
    if duplicate_episodes:
        raise RolloutValidationError(
            f"Rollouts have duplicate episode_ids: {duplicate_episodes[:5]}"
        )
    return instance_ids


def validate_rollout_set(dataset_jsonl: Path, rollout_dir: Path) -> dict[str, Any]:
    expected = load_expected_instance_ids(dataset_jsonl)
    observed = load_rollout_instance_ids(rollout_dir)
    expected_set = set(expected)
    observed_set = set(observed)
    if expected_set != observed_set:
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        raise RolloutValidationError(
            "Rollout task identities differ from the dataset: "
            f"expected={len(expected)}, observed={len(observed)}, "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return {
        "dataset_jsonl": str(dataset_jsonl),
        "rollout_dir": str(rollout_dir),
        "expected_count": len(expected),
        "rollout_count": len(observed),
        "instance_ids_match": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = validate_rollout_set(
            args.dataset_jsonl.expanduser().resolve(),
            args.rollout_dir.expanduser().resolve(),
        )
    except RolloutValidationError as exc:
        raise SystemExit(f"Written rollout validation failed: {exc}") from exc
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
