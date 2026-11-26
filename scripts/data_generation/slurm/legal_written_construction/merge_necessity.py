#!/usr/bin/env python3
"""Retain tasks for which every leave-one-PIU-out variant fails strictly."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from teamgym.core.io import read_jsonl, write_json, write_jsonl
from teamgym.data.long_context_reasoning.judge import strict_all_step_passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--all-riu-dir", default="all_riu_qualified")
    parser.add_argument("--variants-dir", default="necessity_variants")
    parser.add_argument("--necessity-dir", default="leave_one_piu_out")
    parser.add_argument("--output-dir", default="final_qualified")
    args = parser.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    source_rows = read_jsonl(root / args.all_riu_dir / "common_datapoints.jsonl")
    manifest = read_manifest(root / args.variants_dir / "necessity_manifest.tsv")
    necessity_root = root / args.necessity_dir
    output_root = root / args.output_dir

    by_source: dict[int, list[dict]] = defaultdict(list)
    missing: list[dict] = []
    result_rows: list[dict] = []
    for variant_index, source_index, dataset, mode, source_instance, ablated_instance, missing_rui in manifest:
        shard = necessity_root / dataset / mode / f"variant_{variant_index:05d}"
        judgment_path = shard / "judge" / "judgments.jsonl"
        judgments = read_jsonl(judgment_path) if judgment_path.is_file() else []
        if len(judgments) != 1:
            missing.append(
                {
                    "variant_index": variant_index,
                    "source_row_index": source_index,
                    "dataset": dataset,
                    "partition_mode": mode,
                    "missing_rui_id": missing_rui,
                    "judgment_count": len(judgments),
                }
            )
            continue
        strict_passed = strict_all_step_passed(judgments[0])
        result = {
            "variant_index": variant_index,
            "source_row_index": source_index,
            "dataset": dataset,
            "partition_mode": mode,
            "source_instance_id": source_instance,
            "ablated_instance_id": ablated_instance,
            "missing_rui_id": missing_rui,
            "ablated_strict_all_step_passed": strict_passed,
            "necessity_demonstrated": not strict_passed,
            "judgment": judgments[0].get("judgment"),
        }
        result_rows.append(result)
        by_source[source_index].append(result)

    expected_variant_counts = Counter(source_index for _, source_index, *_rest in manifest)
    kept_rows: list[dict] = []
    rejected_rows: list[dict] = []
    counts: Counter[tuple[str, str, str]] = Counter()
    for source_index, row in enumerate(source_rows):
        variants = by_source.get(source_index, [])
        complete = len(variants) == expected_variant_counts[source_index]
        all_necessary = complete and bool(variants) and all(item["necessity_demonstrated"] for item in variants)
        dataset = benchmark_dataset(row)
        mode = partition_mode(row)
        counts[(dataset, mode, "passed" if all_necessary else "failed")] += 1
        if all_necessary:
            kept = deepcopy(row)
            kept["dataset_specific"] = {
                **(kept.get("dataset_specific") or {}),
                "benchmark_dataset": dataset,
                "qualification_status": "all_riu_pass_and_every_leave_one_rui_out_fail",
            }
            kept_rows.append(kept)
        else:
            rejected_rows.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "0712_leave_one_rui_out_qualification_v1",
        "source_row_count": len(source_rows),
        "variant_manifest_count": len(manifest),
        "judged_variant_count": len(result_rows),
        "missing_variant_count": len(missing),
        "necessity_passed_source_count": len(kept_rows),
        "necessity_failed_source_count": len(rejected_rows),
        "missing_variants": missing,
        "counts": [
            {"dataset": dataset, "partition_mode": mode, "status": status, "count": count}
            for (dataset, mode, status), count in sorted(counts.items())
        ],
    }
    write_json(output_root / "summary.json", summary)
    write_jsonl(output_root / "necessity_results.jsonl", result_rows)
    if missing:
        raise RuntimeError(f"Necessity qualification is incomplete: {len(missing)} judgments are missing")
    write_jsonl(output_root / "common_datapoints.jsonl", kept_rows)
    write_jsonl(output_root / "failed_common_datapoints.jsonl", rejected_rows)
    print(f"Leave-one-RIU-out necessity pass: {len(kept_rows)} / {len(source_rows)}")


def read_manifest(path: Path) -> list[tuple[int, int, str, str, str, str, str]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            variant_index, source_index, dataset, mode, source_instance, ablated_instance, missing_rui = fields
            rows.append(
                (int(variant_index), int(source_index), dataset, mode, source_instance, ablated_instance, missing_rui)
            )
    return rows


def partition_mode(row: dict) -> str:
    generation = row.get("generation_metadata") if isinstance(row.get("generation_metadata"), dict) else {}
    dataset_specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    return str(generation.get("allocation_mode") or dataset_specific.get("partition_mode") or "unknown")


def benchmark_dataset(row: dict) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    dataset_specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    family = str(source.get("source_family") or dataset_specific.get("source_family") or "").lower()
    if family.startswith("mslr_"):
        return "mslr"
    if family.startswith("sara_"):
        return "sara"
    if family.startswith("ecfr_casebook"):
        return "casebook"
    if "multi-lexsum" in family or "multilexsum" in family:
        return "multilexsum"
    return str(dataset_specific.get("benchmark_dataset") or row.get("dataset") or "unknown")


if __name__ == "__main__":
    main()
