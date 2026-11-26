#!/usr/bin/env python3
"""Merge and validate outputs from sharded finance role generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def instance_key(row: dict[str, Any]) -> tuple[str, str]:
    dataset = str(row.get("dataset") or "")
    instance_id = str(
        row.get("problem_instance_id") or row.get("instance_id") or ""
    )
    if not dataset or not instance_id:
        raise ValueError(f"Missing dataset or problem instance ID: {row}")
    return dataset, instance_id


def unique_sorted(
    rows: list[dict[str, Any]],
    *,
    key: Callable[[dict[str, Any]], tuple[str, ...]],
    artifact: str,
) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        row_key = key(row)
        if row_key in keyed:
            raise ValueError(f"Duplicate {artifact} key: {row_key}")
        keyed[row_key] = row
    return [keyed[row_key] for row_key in sorted(keyed)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-shards", required=True, type=int)
    parser.add_argument("--expected-problem-instances", required=True, type=int)
    args = parser.parse_args()

    shard_root = args.shard_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shard_dirs = sorted(path for path in shard_root.glob("shard_*") if path.is_dir())
    if len(shard_dirs) != args.expected_shards:
        raise ValueError(
            f"Expected {args.expected_shards} shard directories, found {len(shard_dirs)}"
        )

    reports: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    augmented_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        reports.append(json.loads((shard_dir / "report.json").read_text()))
        sidecar_rows.extend(read_jsonl(shard_dir / "information_features.jsonl"))
        preview_rows.extend(read_jsonl(shard_dir / "participant_role_previews.jsonl"))
        augmented_rows.extend(read_jsonl(shard_dir / "role_augmented.jsonl"))
        errors.extend(read_jsonl(shard_dir / "errors.jsonl"))

    sidecar_rows = unique_sorted(
        sidecar_rows,
        key=lambda row: (*instance_key(row), str(row.get("rui_id") or "")),
        artifact="information feature",
    )
    preview_rows = unique_sorted(
        preview_rows,
        key=instance_key,
        artifact="participant role preview",
    )
    augmented_rows = unique_sorted(
        augmented_rows,
        key=instance_key,
        artifact="role-augmented datapoint",
    )

    completed = sum(int(report["completed_problem_instances"]) for report in reports)
    assigned = sum(int(report["problem_instances_in_shard"]) for report in reports)
    reused = sum(int(report["reused_problem_instances"]) for report in reports)
    validation_errors: list[str] = []
    if assigned != args.expected_problem_instances:
        validation_errors.append(
            f"assigned problem instances: expected {args.expected_problem_instances}, "
            f"found {assigned}"
        )
    if completed != args.expected_problem_instances:
        validation_errors.append(
            f"completed problem instances: expected {args.expected_problem_instances}, "
            f"found {completed}"
        )
    if len(augmented_rows) != args.expected_problem_instances:
        validation_errors.append(
            f"derived rows: expected {args.expected_problem_instances}, "
            f"found {len(augmented_rows)}"
        )
    if len(preview_rows) != args.expected_problem_instances:
        validation_errors.append(
            f"preview rows: expected {args.expected_problem_instances}, "
            f"found {len(preview_rows)}"
        )
    if errors:
        validation_errors.append(f"generation errors: {len(errors)}")

    write_jsonl(output_root / "information_features.jsonl", sidecar_rows)
    write_jsonl(output_root / "participant_role_previews.jsonl", preview_rows)
    write_jsonl(output_root / "role_augmented.jsonl", augmented_rows)
    write_jsonl(output_root / "errors.jsonl", errors)
    merged_report = {
        "schema": "finance_participant_role_merged_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": not validation_errors,
        "expected_shards": args.expected_shards,
        "expected_problem_instances": args.expected_problem_instances,
        "assigned_problem_instances": assigned,
        "completed_problem_instances": completed,
        "reused_problem_instances": reused,
        "feature_count": len(sidecar_rows),
        "role_preview_count": len(preview_rows),
        "derived_row_count": len(augmented_rows),
        "error_count": len(errors),
        "validation_errors": validation_errors,
        "shard_reports": reports,
    }
    write_json(output_root / "report.json", merged_report)
    print(json.dumps(merged_report, indent=2, ensure_ascii=False))
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
