#!/usr/bin/env python3
"""Restore full-source MultiLexSum packets for qualified evaluation tasks."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from teamgym.core.release import FULL_CONTEXT, write_release_datapoints  # noqa: E402
from teamgym.core.schema import normalize_release_datapoint, validate_common_datapoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified", required=True, type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--combined", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    qualified = read_jsonl(args.qualified)
    original = {str(row.get("instance_id")): row for row in read_jsonl(args.original)}
    restored = []
    for qualified_row in qualified:
        instance_id = str(qualified_row.get("instance_id") or "")
        if instance_id not in original:
            raise RuntimeError(f"No original full-context row for {instance_id}")
        source = deepcopy(original[instance_id])
        assert_equal(source, qualified_row, "task_description")
        assert_equal(source, qualified_row, "decision_set")
        assert_equal(source, qualified_row, "decision_option_definitions")
        assert_equal(source, qualified_row, "oracle")
        source["dataset_specific"] = {
            **(source.get("dataset_specific") or {}),
            "benchmark_dataset": "multilexsum",
            "pilot_case_id": (qualified_row.get("dataset_specific") or {}).get("pilot_case_id", instance_id),
            "qualification_survivor": True,
            "context_variant": "full_original_source_packets",
        }
        source["generation_metadata"] = {
            **(source.get("generation_metadata") or {}),
            "context_variant": "full_original_source_packets",
            "qualification_source": "0712_datagen_final/all_riu_qualified",
        }
        validation = validate_common_datapoint(source)
        if not validation.get("valid"):
            raise RuntimeError(f"Restored row failed validation: {instance_id}: {validation}")
        source["format_validation"] = validation
        restored.append(source)

    if len(restored) != 29 or len({row["instance_id"] for row in restored}) != 29:
        raise RuntimeError(f"Expected 29 unique restored tasks, found {len(restored)}")

    restored_by_id = {row["instance_id"]: row for row in restored}
    combined = []
    replaced = 0
    for row in read_jsonl(args.combined):
        instance_id = str(row.get("instance_id") or "")
        if instance_id in restored_by_id:
            combined.append(deepcopy(restored_by_id[instance_id]))
            replaced += 1
        else:
            combined.append(row)
    if len(combined) != 142 or replaced != 29:
        raise RuntimeError(f"Expected a 142-task matrix with 29 replacements; found {len(combined)} and {replaced}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_release_datapoints(
        args.output_dir / "multilexsum_full_context_qualified29.jsonl", restored, evidence_condition=FULL_CONTEXT
    )
    write_release_datapoints(args.output_dir / "combined_full_context_corrected142.jsonl", combined)
    summary = {
        "schema": "multilexsum_full_context_restoration_v1",
        "qualified_tasks": len(restored),
        "combined_tasks": len(combined),
        "participant_count_min": min(len(row["participants"]) for row in restored),
        "participant_count_max": max(len(row["participants"]) for row in restored),
        "full_context_unit_count": sum(len(row["private_context_units"]) for row in restored),
    }
    (args.output_dir / "restoration_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def assert_equal(left: dict[str, Any], right: dict[str, Any], key: str) -> None:
    if left.get(key) != right.get(key):
        raise RuntimeError(f"Qualified and original rows differ in {key}: {left.get('instance_id')}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [normalize_release_datapoint(json.loads(line)) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
