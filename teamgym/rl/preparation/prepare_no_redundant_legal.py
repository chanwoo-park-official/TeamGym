#!/usr/bin/env python3
"""Build matched CUAD/MAUD views with no empty or semantically redundant players."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from teamgym.core.schema import (
    validate_common_datapoint,
)

from ..data import file_sha256, stable_json_hash


VARIANTS = ("riu_only", "full_context")
SPLITS = ("train", "validation", "test")
EXCLUDED_ALLOCATION = "redundant_player"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-dataset", default="cuad")
    parser.add_argument(
        "--evaluation-dataset",
        action="append",
        default=None,
        help="Dataset retained in validation/test; repeat as needed (default: cuad, maud)",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def identity(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row["dataset"]),
        str(row["instance_id"]),
        str(row["canonical_task_id"]),
        str(row["source_group_id"]),
        str(row["allocation"]),
    )


def prune_empty_participants(datapoint: dict[str, Any]) -> tuple[dict[str, Any], int]:
    participants = [
        participant
        for participant in datapoint.get("participants", [])
        if isinstance(participant, dict)
    ]
    retained = [
        participant
        for participant in participants
        if any(str(value) for value in participant.get("owned_rui_ids", []))
    ]
    retained_ids = {str(participant["participant_id"]) for participant in retained}
    if len(retained_ids) < 2:
        raise ValueError(
            f"{datapoint.get('instance_id')}: fewer than two nonempty participants"
        )

    directory = [
        row
        for row in datapoint.get("participant_directory", [])
        if isinstance(row, dict) and str(row.get("participant_id")) in retained_ids
    ]
    if {str(row["participant_id"]) for row in directory} != retained_ids:
        raise ValueError(
            f"{datapoint.get('instance_id')}: participant directory mismatch after pruning"
        )

    contexts: list[dict[str, Any]] = []
    for source_unit in datapoint.get("private_context_units", []):
        if not isinstance(source_unit, dict):
            continue
        assigned = [
            str(value)
            for value in source_unit.get("assigned_to", [])
            if str(value) in retained_ids
        ]
        if not assigned:
            continue
        unit = dict(source_unit)
        unit["assigned_to"] = assigned
        contexts.append(unit)
    context_ids = {str(unit.get("context_unit_id")) for unit in contexts}
    assigned_contexts = {
        participant_id: {
            str(unit.get("context_unit_id"))
            for unit in contexts
            if participant_id in unit.get("assigned_to", [])
        }
        for participant_id in retained_ids
    }

    normalized_participants: list[dict[str, Any]] = []
    for source_participant in retained:
        participant = dict(source_participant)
        participant_id = str(participant["participant_id"])
        declared = {
            str(value)
            for value in participant.get("private_context_unit_ids", [])
            if str(value) in context_ids
        }
        effective = declared | assigned_contexts[participant_id]
        if not effective:
            raise ValueError(
                f"{datapoint.get('instance_id')}: retained {participant_id} has no private context"
            )
        participant["private_context_unit_ids"] = sorted(effective)
        normalized_participants.append(participant)

    ruis: list[dict[str, Any]] = []
    for source_rui in datapoint.get("reference_information_units", []):
        if not isinstance(source_rui, dict):
            continue
        rui = dict(source_rui)
        if isinstance(rui.get("owned_by"), list):
            rui["owned_by"] = [
                str(value)
                for value in rui["owned_by"]
                if str(value) in retained_ids
            ]
        ruis.append(rui)

    result = dict(datapoint)
    result["participants"] = normalized_participants
    result["participant_directory"] = directory
    result["private_context_units"] = contexts
    result["reference_information_units"] = ruis
    result["format_validation"] = validate_common_datapoint(result)
    if not result["format_validation"].get("valid"):
        raise ValueError(
            f"{datapoint.get('instance_id')}: transformed datapoint failed validation: "
            f"{result['format_validation']}"
        )
    return result, len(participants) - len(normalized_participants)


def participants_required_by_every_sufficient_set(
    datapoint: dict[str, Any],
) -> bool:
    """Whether every player contributes to every annotated minimal proof.

    Merely owning an RIU does not make a participant necessary.  We use the
    oracle's smallest sufficient RIU sets and require each retained
    participant to own at least one RIU in every such set.  This deliberately
    excludes tasks with alternative proofs that can bypass a player.
    """

    participant_ids = {
        str(row["participant_id"])
        for row in datapoint.get("participant_directory", [])
        if isinstance(row, dict) and isinstance(row.get("participant_id"), str)
    }
    if len(participant_ids) < 2:
        return False
    owners_by_rui: dict[str, set[str]] = {}
    for participant in datapoint.get("participants", []):
        if not isinstance(participant, dict):
            continue
        participant_id = participant.get("participant_id")
        if not isinstance(participant_id, str):
            continue
        for rui_id in participant.get("owned_rui_ids", []):
            if isinstance(rui_id, str):
                owners_by_rui.setdefault(rui_id, set()).add(participant_id)

    oracle = (
        datapoint.get("oracle")
        if isinstance(datapoint.get("oracle"), dict)
        else {}
    )
    sufficient_sets = [
        row
        for row in oracle.get("smallest_sufficient_rui_sets", [])
        if isinstance(row, dict)
    ]
    if not sufficient_sets:
        return False
    for sufficient_set in sufficient_sets:
        represented: set[str] = set()
        for rui_id in sufficient_set.get("rui_ids", []):
            if isinstance(rui_id, str):
                represented.update(owners_by_rui.get(rui_id, set()))
        if not participant_ids <= represented:
            return False
    return True


def transform_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    train_dataset: str,
    evaluation_datasets: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    dropped_allocation = Counter()
    dropped_dataset = Counter()
    dropped_empty_participants = 0
    dropped_semantically_redundant = Counter()
    for source_row in rows:
        dataset = str(source_row["dataset"]).casefold()
        allowed = (
            dataset == train_dataset
            if split == "train"
            else dataset in evaluation_datasets
        )
        if not allowed:
            dropped_dataset[dataset] += 1
            continue
        allocation = str(source_row["allocation"]).casefold()
        if allocation == EXCLUDED_ALLOCATION:
            dropped_allocation[dataset] += 1
            continue
        datapoint = json.loads(source_row["teamgym_datapoint_json"])
        datapoint, removed = prune_empty_participants(datapoint)
        dropped_empty_participants += removed
        if not participants_required_by_every_sufficient_set(datapoint):
            dropped_semantically_redundant[dataset] += 1
            continue
        row = dict(source_row)
        row["participant_count"] = len(datapoint["participant_directory"])
        row["datapoint_sha256"] = stable_json_hash(datapoint)
        row["teamgym_datapoint_json"] = json.dumps(
            datapoint, sort_keys=True, ensure_ascii=False
        )
        kept.append(row)
    if not kept:
        raise ValueError(f"No rows retained for split {split}")
    return kept, {
        "input_rows": len(rows),
        "output_rows": len(kept),
        "dropped_dataset_rows": dict(sorted(dropped_dataset.items())),
        "dropped_redundant_player_rows": dict(sorted(dropped_allocation.items())),
        "dropped_semantically_redundant_rows": dict(
            sorted(dropped_semantically_redundant.items())
        ),
        "removed_empty_participant_slots": dropped_empty_participants,
    }


def write_split(root: Path, split: str, rows: list[dict[str, Any]]) -> list[Path]:
    jsonl_path = root / f"{split}.jsonl"
    parquet_path = root / f"{split}.parquet"
    verl_path = root / f"{split}_verl.parquet"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    frame = pd.DataFrame(rows)
    frame.to_parquet(parquet_path, index=False)
    pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "TeamGym workflow task"}],
                "reward_model": {"style": "rule", "ground_truth": None},
                "extra_info": row,
            }
            for row in rows
        ]
    ).to_parquet(verl_path, index=False)
    return [jsonl_path, parquet_path, verl_path]


def identity_sets(
    rows_by_split: dict[str, list[dict[str, Any]]], field: str
) -> dict[str, set[tuple[str, str]]]:
    return {
        split: {
            (str(row["dataset"]), str(row[field]))
            for row in rows
        }
        for split, rows in rows_by_split.items()
    }


def assert_no_leakage(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    for field in ("source_group_id", "canonical_task_id"):
        identities = identity_sets(rows_by_split, field)
        for index, left in enumerate(SPLITS):
            for right in SPLITS[index + 1 :]:
                overlap = identities[left] & identities[right]
                if overlap:
                    raise ValueError(
                        f"{field} leakage between {left} and {right}: "
                        f"{sorted(overlap)[:3]}"
                    )


def count_rows(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def verify_no_empty_participants(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        datapoint = json.loads(row["teamgym_datapoint_json"])
        for participant in datapoint["participants"]:
            if not participant.get("owned_rui_ids"):
                raise ValueError(
                    f"{row['instance_id']}: empty participant survived transformation"
                )
        if not participants_required_by_every_sufficient_set(datapoint):
            raise ValueError(
                f"{row['instance_id']}: a participant can be bypassed by a "
                "smallest sufficient RIU set"
            )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    train_dataset = args.train_dataset.strip().casefold()
    evaluation_datasets = {
        value.strip().casefold()
        for value in (args.evaluation_dataset or ["cuad", "maud"])
        if value.strip()
    }
    if train_dataset not in evaluation_datasets:
        raise ValueError("The training dataset must also be retained for evaluation")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output root {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    variant_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    report: dict[str, Any] = {
        "schema_version": "teamgym_no_redundant_legal_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "train_dataset": train_dataset,
        "evaluation_datasets": sorted(evaluation_datasets),
        "excluded_allocation": EXCLUDED_ALLOCATION,
        "drop_empty_participants": True,
        "require_every_participant_in_every_smallest_sufficient_set": True,
        "variants": {},
    }
    for variant in VARIANTS:
        source_variant = source_root / variant
        output_variant = output_root / variant
        output_variant.mkdir(parents=True, exist_ok=False)
        source_paths = {split: source_variant / f"{split}.jsonl" for split in SPLITS}
        missing = [str(path) for path in source_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing source split files: {missing}")
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        split_reports: dict[str, Any] = {}
        for split in SPLITS:
            rows_by_split[split], split_reports[split] = transform_split(
                load_jsonl(source_paths[split]),
                split=split,
                train_dataset=train_dataset,
                evaluation_datasets=evaluation_datasets,
            )
            verify_no_empty_participants(rows_by_split[split])
        assert_no_leakage(rows_by_split)
        variant_rows[variant] = rows_by_split

        written: list[Path] = []
        for split in SPLITS:
            written.extend(write_split(output_variant, split, rows_by_split[split]))
        manifest_path = output_variant / "split_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for split in SPLITS:
                for row in rows_by_split[split]:
                    handle.write(
                        json.dumps(
                            {
                                key: value
                                for key, value in row.items()
                                if key != "teamgym_datapoint_json"
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        written.append(manifest_path)
        source_hashes = {
            str(path): file_sha256(path) for path in source_paths.values()
        }
        leakage_audit = {
            "schema_version": "teamgym_leakage_audit_v1",
            "passed": True,
            "split_group": "source_group_id",
            "interleave_datasets": False,
            "input_hashes": source_hashes,
            "manifest_sha256": file_sha256(manifest_path),
            "row_counts": {
                split: len(rows_by_split[split]) for split in SPLITS
            },
            "split_group_counts": {
                split: len(identity_sets(rows_by_split, "source_group_id")[split])
                for split in SPLITS
            },
            "canonical_task_counts": {
                split: len(identity_sets(rows_by_split, "canonical_task_id")[split])
                for split in SPLITS
            },
            "overlaps": {
                "train_validation": [],
                "train_test": [],
                "validation_test": [],
            },
        }
        leakage_path = output_variant / "leakage_audit.json"
        leakage_path.write_text(
            json.dumps(leakage_audit, indent=2) + "\n", encoding="utf-8"
        )
        written.append(leakage_path)
        report["variants"][variant] = {
            "output_dir": str(output_variant),
            "source_hashes": source_hashes,
            "splits": split_reports,
            "row_counts": {
                split: len(rows_by_split[split]) for split in SPLITS
            },
            "dataset_counts": {
                split: count_rows(rows_by_split[split], "dataset")
                for split in SPLITS
            },
            "allocation_counts": {
                split: count_rows(rows_by_split[split], "allocation")
                for split in SPLITS
            },
            "participant_count_distributions": {
                split: count_rows(rows_by_split[split], "participant_count")
                for split in SPLITS
            },
            "output_hashes": {
                path.name: file_sha256(path) for path in written
            },
        }

    for split in SPLITS:
        riu_rows = variant_rows["riu_only"][split]
        full_rows = variant_rows["full_context"][split]
        if [identity(row) for row in riu_rows] != [identity(row) for row in full_rows]:
            raise ValueError(f"Matched variant identity/order mismatch in {split}")
        for riu_row, full_row in zip(riu_rows, full_rows, strict=True):
            riu = json.loads(riu_row["teamgym_datapoint_json"])
            full = json.loads(full_row["teamgym_datapoint_json"])
            riu_owners = {
                str(row["participant_id"]): sorted(row.get("owned_rui_ids", []))
                for row in riu["participants"]
            }
            full_owners = {
                str(row["participant_id"]): sorted(row.get("owned_rui_ids", []))
                for row in full["participants"]
            }
            if riu_owners != full_owners:
                raise ValueError(
                    f"Matched ownership mismatch for {riu_row['instance_id']}"
                )
    report["matched_variant_checks"] = {
        "identity_order_equal": True,
        "participant_rui_ownership_equal": True,
    }
    report_path = output_root / "preparation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
