#!/usr/bin/env python3
"""Merge strict all-step single-agent all-PIU qualification results."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from teamgym.core.release import write_release_datapoints
from teamgym.core.io import read_jsonl, write_json, write_jsonl
from teamgym.data.long_context_reasoning.judge import strict_all_step_passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--survivor-dir", default="leakage_survivors")
    parser.add_argument("--qualification-dir", default="all_piu_qualification")
    parser.add_argument("--output-dir", default="all_riu_qualified")
    args = parser.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    survivor_root = root / args.survivor_dir
    qualification_root = root / args.qualification_dir
    output_root = root / args.output_dir
    source_rows = read_jsonl(survivor_root / "common_datapoints.jsonl")
    manifest = read_manifest(survivor_root / "qualification_manifest.tsv")
    if len(source_rows) != len(manifest):
        raise ValueError("Qualification source row count does not match manifest")

    passed_rows: list[dict] = []
    failed_rows: list[dict] = []
    result_rows: list[dict] = []
    missing: list[dict] = []
    counts: Counter[tuple[str, str, str]] = Counter()
    for qualification_index, dataset, mode, source_row_index, instance_id in manifest:
        shard = qualification_root / dataset / mode / f"row_{source_row_index:04d}"
        judgment_path = shard / "judge" / "judgments.jsonl"
        judgments = read_jsonl(judgment_path) if judgment_path.is_file() else []
        if len(judgments) != 1:
            missing.append(
                {
                    "qualification_index": qualification_index,
                    "dataset": dataset,
                    "partition_mode": mode,
                    "source_row_index": source_row_index,
                    "instance_id": instance_id,
                    "judgment_count": len(judgments),
                }
            )
            continue
        judgment_row = judgments[0]
        strict_passed = strict_all_step_passed(judgment_row)
        result = {
            "qualification_index": qualification_index,
            "dataset": dataset,
            "partition_mode": mode,
            "source_row_index": source_row_index,
            "instance_id": instance_id,
            "strict_all_step_passed": strict_passed,
            "judgment": judgment_row.get("judgment"),
        }
        result_rows.append(result)
        counts[(dataset, mode, "passed" if strict_passed else "failed")] += 1
        source_row = source_rows[qualification_index]
        if strict_passed:
            passed_rows.append(source_row)
        else:
            failed_rows.append(source_row)

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "0712_all_riu_qualification_merge_v1",
        "manifest_row_count": len(manifest),
        "judged_row_count": len(result_rows),
        "strict_passed_count": len(passed_rows),
        "strict_failed_count": len(failed_rows),
        "missing_row_count": len(missing),
        "missing_rows": missing,
        "counts": [
            {"dataset": dataset, "partition_mode": mode, "status": status, "count": count}
            for (dataset, mode, status), count in sorted(counts.items())
        ],
    }
    write_json(output_root / "summary.json", summary)
    write_jsonl(output_root / "qualification_results.jsonl", result_rows)
    if missing:
        raise RuntimeError(f"Qualification is incomplete: {len(missing)} judgments are missing")
    write_release_datapoints(output_root / "common_datapoints.jsonl", passed_rows)
    write_jsonl(output_root / "failed_common_datapoints.jsonl", failed_rows)
    print(f"Strict all-step pass: {len(passed_rows)} / {len(manifest)}")


def read_manifest(path: Path) -> list[tuple[int, str, str, int, str]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, dataset, mode, source_index, instance_id in csv.reader(handle, delimiter="\t"):
            rows.append((int(index), dataset, mode, int(source_index), instance_id))
    return rows


if __name__ == "__main__":
    main()
