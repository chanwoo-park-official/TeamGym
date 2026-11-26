#!/usr/bin/env python3
"""Merge accounted leakage survivors and build the all-PIU task manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from teamgym.core.io import read_jsonl, write_json, write_jsonl


TaskKey = tuple[str, str, int]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--validation-dir", default="leakage_validation")
    parser.add_argument("--output-dir", default="leakage_survivors")
    parser.add_argument("--manifest", help="Defaults to RUN_ROOT/validation_manifest.tsv.")
    args = parser.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    validation_root = root / args.validation_dir
    output_root = root / args.output_dir
    manifest_path = Path(args.manifest) if args.manifest else root / "validation_manifest.tsv"
    expected_order = read_manifest(manifest_path)
    expected = set(expected_order)

    survivor_files = sorted(validation_root.glob("*/*/row_*/01_leakage_deletion/leakage_deleted_common_datapoints.jsonl"))
    rejected_files = sorted(validation_root.glob("*/*/row_*/candidate_status.json"))
    survivor_by_key: dict[TaskKey, dict] = {}
    for path in survivor_files:
        rows = read_jsonl(path)
        if not rows:
            continue
        if len(rows) != 1:
            raise ValueError(f"Expected one leakage survivor in {path}, found {len(rows)}")
        survivor_by_key[task_key(path, validation_root)] = rows[0]

    rejected = {task_key(path, validation_root) for path in rejected_files}
    processed = set(survivor_by_key) | rejected
    missing = sorted(expected - processed)
    overlap = sorted(set(survivor_by_key) & rejected)
    unexpected = sorted(processed - expected)

    survivors: list[dict] = []
    qualification_rows: list[tuple[int, str, str, int, str]] = []
    counts: Counter[tuple[str, str]] = Counter()
    for key in expected_order:
        row = survivor_by_key.get(key)
        if row is None:
            continue
        qualification_index = len(survivors)
        survivors.append(row)
        dataset, mode, source_row_index = key
        counts[(dataset, mode)] += 1
        qualification_rows.append(
            (qualification_index, dataset, mode, source_row_index, str(row.get("instance_id") or ""))
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "0712_leakage_survivor_merge_v1",
        "manifest_row_count": len(expected),
        "processed_row_count": len(processed),
        "leakage_survivor_count": len(survivors),
        "leakage_rejected_count": len(rejected),
        "missing_row_count": len(missing),
        "overlap_row_count": len(overlap),
        "unexpected_row_count": len(unexpected),
        "missing_rows": key_dicts(missing),
        "overlap_rows": key_dicts(overlap),
        "unexpected_rows": key_dicts(unexpected),
        "counts": [
            {"dataset": dataset, "partition_mode": mode, "count": count}
            for (dataset, mode), count in sorted(counts.items())
        ],
    }
    write_json(output_root / "summary.json", summary)
    if missing or overlap or unexpected:
        raise RuntimeError("Leakage outputs do not account cleanly for the validation manifest")
    write_jsonl(output_root / "common_datapoints.jsonl", survivors)
    with (output_root / "qualification_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(qualification_rows)
    print(f"Merged {len(survivors)} leakage survivors into {output_root}")


def read_manifest(path: Path) -> list[TaskKey]:
    rows: list[TaskKey] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for dataset, mode, row_index, _source in csv.reader(handle, delimiter="\t"):
            rows.append((dataset, mode, int(row_index)))
    if len(rows) != len(set(rows)):
        raise ValueError(f"Duplicate task keys in {path}")
    return rows


def task_key(path: Path, validation_root: Path) -> TaskKey:
    dataset, mode, row_name = path.relative_to(validation_root).parts[:3]
    return dataset, mode, int(row_name.removeprefix("row_"))


def key_dicts(keys: list[TaskKey]) -> list[dict[str, object]]:
    return [
        {"dataset": dataset, "partition_mode": mode, "row_index": row_index}
        for dataset, mode, row_index in keys
    ]


if __name__ == "__main__":
    main()
