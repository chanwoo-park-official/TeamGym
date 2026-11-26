#!/usr/bin/env python3
"""Build matched CUAD+MAUD full-context and PIU-only TeamGym splits."""

from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import zip_longest
import json
from pathlib import Path
from typing import Any, Iterator

from teamgym.core.schema import validate_common_datapoint

from ..data import prepare_data


DATASETS = ("cuad", "maud")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()

    archive_root = Path(args.archive_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    normalized_root = output_root / "normalized"
    normalized_root.mkdir(parents=True, exist_ok=True)
    heldout_path = output_root / "paper_test_source_groups.jsonl"

    full_path = normalized_root / "cuad_maud_full_context.jsonl"
    riu_path = normalized_root / "cuad_maud_riu_only.jsonl"
    counts = {dataset: 0 for dataset in DATASETS}
    paper_source_groups: set[tuple[str, str]] = set()
    with full_path.open("w", encoding="utf-8") as full_handle, riu_path.open(
        "w", encoding="utf-8"
    ) as riu_handle:
        for dataset in DATASETS:
            for index, row in _archive_rows(archive_root, dataset):
                cross = {
                    "archive_instance_id": str(index["archive_instance_id"]),
                    "canonical_task_id": str(index["canonical_task_id"]),
                    "source_group_id": str(index["source_group_id"]),
                    "partition_mode": str(index["allocation_mode"]),
                    "paper_standard_163": bool(index.get("paper_standard_163")),
                    "context_variant": "full_context",
                }
                full_row = deepcopy(row)
                full_row["_cross"] = cross
                full_handle.write(json.dumps(full_row, ensure_ascii=False) + "\n")

                riu_row = to_riu_only(full_row)
                validation = validate_common_datapoint(riu_row)
                if not validation["valid"]:
                    raise ValueError(
                        f"RIU-only validation failed for {row.get('instance_id')}: "
                        f"{validation.get('errors', validation)}"
                    )
                riu_handle.write(json.dumps(riu_row, ensure_ascii=False) + "\n")
                counts[dataset] += 1
                if index.get("paper_standard_163"):
                    paper_source_groups.add(
                        (dataset, str(index["source_group_id"]))
                    )

    heldout_path.write_text(
        "".join(
            json.dumps({"dataset": dataset, "source_group_id": source_group_id})
            + "\n"
            for dataset, source_group_id in sorted(paper_source_groups)
        ),
        encoding="utf-8",
    )

    results: dict[str, Any] = {}
    for variant, input_path in (
        ("full_context", full_path),
        ("riu_only", riu_path),
    ):
        results[variant] = prepare_data(
            input_paths=input_path,
            output_dir=output_root / variant,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=0.0,
            paper_heldout_input=heldout_path,
            split_group="source_group",
            interleave_datasets=True,
        )

    _assert_matched_splits(results)
    report = {
        "schema_version": "teamgym_legal_mixed_preparation_v1",
        "archive_root": str(archive_root),
        "datasets": list(DATASETS),
        "input_row_counts": counts,
        "paper_test_source_group_counts": {
            dataset: sum(1 for item in paper_source_groups if item[0] == dataset)
            for dataset in DATASETS
        },
        "split_unit": "source_group_id",
        "dataset_order": "deterministically shuffled within dataset, then round-robin interleaved",
        "variants": results,
    }
    report_path = output_root / "preparation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({**report, "report_path": str(report_path)}, indent=2))


def _archive_rows(
    archive_root: Path, dataset: str
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    data_root = archive_root / "data" / dataset
    index_path = data_root / "index.jsonl"
    instances_path = data_root / "instances.jsonl"
    if not index_path.is_file() or not instances_path.is_file():
        raise FileNotFoundError(f"Incomplete archive dataset: {data_root}")
    with index_path.open(encoding="utf-8") as index_handle, instances_path.open(
        encoding="utf-8"
    ) as instance_handle:
        for line_number, pair in enumerate(
            zip_longest(index_handle, instance_handle), start=1
        ):
            index_line, instance_line = pair
            if index_line is None or instance_line is None:
                raise ValueError(f"Index/instance length mismatch in {data_root}")
            index = json.loads(index_line)
            row = json.loads(instance_line)
            if str(index.get("stored_instance_id")) != str(row.get("instance_id")):
                raise ValueError(
                    f"Index mismatch in {dataset} line {line_number}: "
                    f"{index.get('stored_instance_id')} != {row.get('instance_id')}"
                )
            yield index, row


def to_riu_only(source: dict[str, Any]) -> dict[str, Any]:
    """Replace full private documents with exactly the already-owned RIU text."""

    row = deepcopy(source)
    rui_by_id = {
        str(item["rui_id"]): item
        for item in row.get("reference_information_units", [])
        if isinstance(item, dict) and item.get("rui_id")
    }
    contexts: list[dict[str, Any]] = []
    for participant in row.get("participants", []):
        participant_id = str(participant["participant_id"])
        owned = [
            str(item)
            for item in participant.get("owned_rui_ids", [])
            if str(item) in rui_by_id
        ]
        context_id = f"{participant_id}_riu_only_context"
        text = "\n\n".join(
            str(rui_by_id[rui_id].get("text") or "").strip()
            for rui_id in owned
        ).strip()
        participant["private_context_unit_ids"] = [context_id]
        participant["private_word_count"] = len(text.split())
        contexts.append(
            {
                "context_unit_id": context_id,
                "assigned_to": [participant_id],
                "text": text,
                "word_count": len(text.split()),
                "hidden_contains_rui_ids": owned,
                "metadata": {"context_kind": "riu_only_owned_evidence"},
            }
        )
    row["private_context_units"] = contexts
    row.setdefault("generation_metadata", {})["context_variant"] = "riu_only"
    row.setdefault("dataset_specific", {})["context_variant"] = "riu_only"
    row["_cross"]["context_variant"] = "riu_only"
    return row


def _assert_matched_splits(results: dict[str, Any]) -> None:
    assignments: dict[str, dict[tuple[str, str], str]] = {}
    for variant, result in results.items():
        rows = [
            json.loads(line)
            for line in Path(result["manifest_path"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assignments[variant] = {
            (str(row["dataset"]), str(row["source_group_id"])): str(row["split"])
            for row in rows
        }
    if assignments["full_context"] != assignments["riu_only"]:
        raise RuntimeError("Full-context and RIU-only source-group splits differ")


if __name__ == "__main__":
    main()
