"""Offline checks for arbitrary prepared TeamGym training splits."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


def check_prepared_splits(
    root: str | Path,
    *,
    train_split: str,
    validation_split: str,
    reward_config: str,
    test_split: str | None = None,
) -> dict:
    """Check local format agreement, references, reward compatibility, and splits."""
    import pandas as pd

    from teamgym.core.schema import validate_common_datapoint
    from ..data import (
        answer_type, canonical_dataset_name, canonical_task_id, file_sha256,
        explicit_source_group_id, source_group_id, stable_json_hash,
    )
    from ..rollout.rewards import REWARD_CONFIGS

    if reward_config not in REWARD_CONFIGS:
        raise ValueError(f"Unknown reward config: {reward_config}")
    splits = [train_split, validation_split] + ([test_split] if test_split else [])
    if len(splits) != len(set(splits)):
        raise ValueError("Train, validation, and optional test split names must differ")
    root = Path(root).expanduser().resolve()
    report = {"prepared_data_dir": str(root), "reward_config": reward_config,
              "splits": {}, "file_sha256": {}}
    seen_instances: set[str] = set()
    seen_groups: dict[tuple[str, str, str], str] = {}
    missing_source_groups = 0
    fields = (
        "instance_id", "dataset", "canonical_task_id", "source_group_id",
        "context_variant", "participant_count", "split", "answer_type",
        "datapoint_sha256", "teamgym_datapoint_json",
    )
    for split in splits:
        paths = [root / f"{split}{suffix}" for suffix in (".jsonl", ".parquet", "_verl.parquet")]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            report["file_sha256"][path.name] = file_sha256(path)
        with paths[0].open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        parquet = pd.read_parquet(paths[1]).to_dict("records")
        verl_frame = pd.read_parquet(paths[2])
        if "extra_info" not in verl_frame:
            raise ValueError(f"{split}: VERL parquet is missing extra_info")
        verl = verl_frame["extra_info"].tolist()
        if not rows or len(rows) != len(parquet) or len(rows) != len(verl):
            raise ValueError(f"{split}: empty split or inconsistent prepared row counts")
        datasets: Counter[str] = Counter()
        answer_types: Counter[str] = Counter()
        labels: set[str] = set()
        for index, (row, pq_row, verl_row) in enumerate(zip(rows, parquet, verl), 1):
            label = f"{split}:{index}"
            if not all(isinstance(value, dict) for value in (row, pq_row, verl_row)):
                raise ValueError(f"{label}: prepared rows must be objects")
            for field in fields:
                if field not in row or any(peer.get(field) != row[field] for peer in (pq_row, verl_row)):
                    raise ValueError(f"{label}: inconsistent {field} across prepared formats")
            if not isinstance(row["teamgym_datapoint_json"], str):
                raise ValueError(f"{label}: embedded datapoint must be a JSON string")
            payload = json.loads(row["teamgym_datapoint_json"])
            if not isinstance(payload, dict) or stable_json_hash(payload) != row["datapoint_sha256"]:
                raise ValueError(f"{label}: embedded datapoint checksum mismatch")
            try:
                valid = validate_common_datapoint(payload)["valid"]
            except (TypeError, AttributeError) as error:
                raise ValueError(f"{label}: invalid common datapoint structure") from error
            if not valid:
                raise ValueError(f"{label}: common datapoint validation failed")
            dataset = canonical_dataset_name(payload)
            if not isinstance(payload.get("instance_id"), str) or not payload["instance_id"].strip():
                raise ValueError(f"{label}: instance ID must be a nonempty string")
            if row["dataset"] != dataset or row["instance_id"] != payload["instance_id"]:
                raise ValueError(f"{label}: embedded datapoint identity mismatch")
            if row["instance_id"] in seen_instances:
                raise ValueError(f"{label}: duplicate instance ID across selected splits")
            seen_instances.add(row["instance_id"])
            if explicit_source_group_id(payload) is None:
                missing_source_groups += 1
            for field, resolver in (("canonical_task_id", canonical_task_id), ("source_group_id", source_group_id)):
                if row[field] != resolver(payload):
                    raise ValueError(f"{label}: embedded {field} mismatch")
                key = (field, dataset, row[field])
                previous = seen_groups.setdefault(key, split)
                if previous != split:
                    raise ValueError(f"{label}: {field} overlaps {previous}")
            kind = answer_type(payload)
            if row["participant_count"] != len(payload["participants"]) or row["answer_type"] != kind:
                raise ValueError(f"{label}: participant count or answer type mismatch")
            written = kind == "short_answer_judged"
            if written != (reward_config == "written_holistic_score"):
                raise ValueError(
                    f"{label}: written answers require written_holistic_score; "
                    "finite-choice answers require a task_success reward"
                )
            if written:
                oracle = payload.get("oracle") or {}
                specific = payload.get("dataset_specific") or {}
                reference = (oracle.get("canonical_response") or oracle.get("short_answer")
                             or oracle.get("answer_text") or specific.get("canonical_response"))
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError(f"{label}: written-answer reference is missing")
            datasets[dataset] += 1
            answer_types[kind] += 1
            labels.add(str(row["split"]))
        report["splits"][split] = {"rows": len(rows), "datasets": dict(sorted(datasets.items())),
                                   "answer_types": dict(sorted(answer_types.items())),
                                   "stored_split_labels": sorted(labels)}
    report.update(
        canonical_tasks_disjoint=True,
        source_groups_disjoint=None if missing_source_groups else True,
        rows_without_explicit_source_group=missing_source_groups,
    )
    return report
