#!/usr/bin/env python3
"""Prepare the finance participant-count transfer experiment.

Training combines FindVer training tasks with two- and three-participant
FinLongDocQA tasks.  Both datasets can contribute source-group-disjoint
validation examples.  Four-participant FinLongDocQA is kept locked as the OOD
test set, and the single six-participant task is materialized separately for
qualitative use.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import pandas as pd

from teamgym.core.dataset import (
    iter_common_datapoints,
)

from ..data import (
    allocation_mode,
    answer_type,
    canonical_dataset_name,
    canonical_task_id,
    file_sha256,
    require_source_group_id,
    stable_json_hash,
)


DEFAULT_INPUT = Path(os.environ.get("TEAMGYM_FINANCE_NORMALIZED_DATA", "generated/finance_participant_handoff_rl/normalized/finance_riu_only.jsonl")).expanduser()
DEFAULT_OUTPUT = Path(os.environ.get("TEAMGYM_FINANCE_PREPARED_ROOT", "generated/finance_participant_handoff_rl")).expanduser() / "findver_plus_finlong23_to_finlong4"
SPLITS = ("train", "validation", "test", "test_finlongdocqa_6p")


def _without_loader_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"_input_path", "_input_line_number"}
    }


def _prepared_row(
    row: dict[str, Any], *, split: str, seed: int
) -> dict[str, Any]:
    datapoint = _without_loader_metadata(row)
    return {
        "instance_id": str(row.get("instance_id") or ""),
        "dataset": canonical_dataset_name(row),
        "canonical_task_id": canonical_task_id(row),
        "source_group_id": require_source_group_id(row),
        "allocation": allocation_mode(row),
        "answer_type": answer_type(row),
        "participant_count": len(row.get("participant_directory", [])),
        "context_variant": str(
            ((row.get("_cross") or {}).get("context_variant")) or "unknown"
        ),
        "paper_standard_163": bool(
            (row.get("_cross") or {}).get("paper_standard_163", False)
        ),
        "source_path": str(row.get("_input_path") or ""),
        "source_line": int(row.get("_input_line_number") or 0),
        "split": split,
        "split_seed": int(seed),
        "datapoint_sha256": stable_json_hash(datapoint),
        "teamgym_datapoint_json": json.dumps(
            datapoint, sort_keys=True, ensure_ascii=False
        ),
    }


def _write_split(
    root: Path, split: str, rows: list[dict[str, Any]]
) -> dict[str, str]:
    jsonl_path = root / f"{split}.jsonl"
    parquet_path = root / f"{split}.parquet"
    verl_path = root / f"{split}_verl.parquet"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    pd.DataFrame(
        [
            {
                "prompt": [
                    {"role": "user", "content": "TeamGym workflow task"}
                ],
                "reward_model": {"style": "rule", "ground_truth": None},
                "extra_info": row,
            }
            for row in rows
        ]
    ).to_parquet(verl_path, index=False)
    return {
        "jsonl": str(jsonl_path),
        "jsonl_sha256": file_sha256(jsonl_path),
        "parquet": str(parquet_path),
        "parquet_sha256": file_sha256(parquet_path),
        "verl_parquet": str(verl_path),
        "verl_parquet_sha256": file_sha256(verl_path),
    }


def _identity_set(
    rows: list[dict[str, Any]], field: str
) -> set[tuple[str, str]]:
    return {(str(row["dataset"]), str(row[field])) for row in rows}


def _assert_no_overlap(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    for field in ("source_group_id", "canonical_task_id"):
        identities = {
            split: _identity_set(rows, field)
            for split, rows in rows_by_split.items()
        }
        for index, left in enumerate(SPLITS):
            for right in SPLITS[index + 1 :]:
                overlap = identities[left] & identities[right]
                if overlap:
                    raise ValueError(
                        f"{field} overlap between {left} and {right}: "
                        f"{sorted(overlap)[:3]}"
                    )


def prepare_transfer(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 17,
    findver_validation_fraction: float = 0.1,
    finlongdocqa_validation_fraction: float = 0.0,
) -> dict[str, Any]:
    if not 0.0 < findver_validation_fraction < 1.0:
        raise ValueError("findver_validation_fraction must be in (0, 1)")
    if not 0.0 <= finlongdocqa_validation_fraction < 1.0:
        raise ValueError(
            "finlongdocqa_validation_fraction must be in [0, 1)"
        )
    source_path = Path(input_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    rows = list(iter_common_datapoints(source_path))
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[canonical_dataset_name(row)].append(row)
    expected = {"findver", "finlongdocqa"}
    if set(by_dataset) != expected:
        raise ValueError(
            f"Expected exactly {sorted(expected)}, found {sorted(by_dataset)}"
        )

    findver_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_dataset["findver"]:
        findver_groups[require_source_group_id(row)].append(row)
    group_ids = sorted(findver_groups)
    random.Random(seed).shuffle(group_ids)
    validation_group_count = min(
        len(group_ids) - 1,
        max(1, int(round(len(group_ids) * findver_validation_fraction))),
    )
    validation_group_ids = set(group_ids[:validation_group_count])

    finlong_groups_by_count: dict[int, dict[str, list[dict[str, Any]]]] = {
        2: defaultdict(list),
        3: defaultdict(list),
    }
    for row in by_dataset["finlongdocqa"]:
        participant_count = len(row.get("participant_directory", []))
        if participant_count in finlong_groups_by_count:
            finlong_groups_by_count[participant_count][require_source_group_id(row)].append(
                row
            )
    finlong_validation_group_ids: set[str] = set()
    finlong_validation_group_counts: dict[str, int] = {}
    for participant_count, groups in finlong_groups_by_count.items():
        candidate_ids = sorted(groups)
        random.Random(seed + 100 + participant_count).shuffle(candidate_ids)
        if finlongdocqa_validation_fraction > 0.0:
            held_out_count = min(
                len(candidate_ids) - 1,
                max(
                    1,
                    int(
                        round(
                            len(candidate_ids)
                            * finlongdocqa_validation_fraction
                        )
                    ),
                ),
            )
        else:
            held_out_count = 0
        selected = set(candidate_ids[:held_out_count])
        finlong_validation_group_ids.update(selected)
        finlong_validation_group_counts[str(participant_count)] = len(selected)

    normalized_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
        "test_finlongdocqa_6p": [],
    }
    for row in by_dataset["findver"]:
        target = (
            "validation"
            if require_source_group_id(row) in validation_group_ids
            else "train"
        )
        normalized_by_split[target].append(row)
    for row in by_dataset["finlongdocqa"]:
        participant_count = len(row.get("participant_directory", []))
        if participant_count in {2, 3}:
            target = (
                "validation"
                if require_source_group_id(row) in finlong_validation_group_ids
                else "train"
            )
            normalized_by_split[target].append(row)
        elif participant_count == 4:
            normalized_by_split["test"].append(row)
        elif participant_count == 6:
            normalized_by_split["test_finlongdocqa_6p"].append(row)
        else:
            raise ValueError(
                "Unexpected FinLongDocQA participant count "
                f"{participant_count} for {row.get('instance_id')}"
            )

    prepared_by_split: dict[str, list[dict[str, Any]]] = {}
    for offset, split in enumerate(SPLITS):
        values = normalized_by_split[split]
        random.Random(seed + offset).shuffle(values)
        prepared_by_split[split] = [
            _prepared_row(row, split=split, seed=seed) for row in values
        ]
    _assert_no_overlap(prepared_by_split)

    actual_counts = {
        split: len(values) for split, values in prepared_by_split.items()
    }
    if sum(actual_counts.values()) != len(rows):
        raise ValueError(
            "Split rows do not cover the input exactly: "
            f"{actual_counts} versus {len(rows)} input rows"
        )
    for split, count in actual_counts.items():
        if count == 0:
            raise ValueError(f"Required split {split!r} is empty")

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        split: _write_split(output_root, split, prepared_by_split[split])
        for split in SPLITS
    }
    manifest_path = output_root / "split_manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(
                {
                    key: value
                    for key, value in row.items()
                    if key != "teamgym_datapoint_json"
                },
                ensure_ascii=False,
            )
            + "\n"
            for split in SPLITS
            for row in prepared_by_split[split]
        ),
        encoding="utf-8",
    )

    dataset_counts = {
        split: dict(
            sorted(Counter(row["dataset"] for row in values).items())
        )
        for split, values in prepared_by_split.items()
    }
    participant_counts = {
        split: {
            str(key): value
            for key, value in sorted(
                Counter(row["participant_count"] for row in values).items()
            )
        }
        for split, values in prepared_by_split.items()
    }
    audit = {
        "schema_version": "finance_participant_transfer_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "input": str(source_path),
        "input_sha256": file_sha256(source_path),
        "output_dir": str(output_root),
        "design": {
            "train": (
                "FindVer train plus FinLongDocQA participant counts 2 and 3 "
                "after source-group validation holdout"
            ),
            "validation": (
                "FindVer plus stratified FinLongDocQA participant counts 2 and 3"
                if finlongdocqa_validation_fraction > 0.0
                else "FindVer only"
            ),
            "test": "FinLongDocQA participant count 4, locked",
            "qualitative_test": "FinLongDocQA participant count 6",
        },
        "validation_fractions": {
            "findver": findver_validation_fraction,
            "finlongdocqa_2p_3p": finlongdocqa_validation_fraction,
        },
        "validation_source_group_counts": {
            "findver": len(validation_group_ids),
            "finlongdocqa_by_participant_count": (
                finlong_validation_group_counts
            ),
        },
        "row_counts": actual_counts,
        "dataset_counts": dataset_counts,
        "participant_count_distribution": participant_counts,
        "source_group_overlap": False,
        "canonical_task_overlap": False,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "outputs": outputs,
    }
    audit_path = output_root / "transfer_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {**audit, "audit_path": str(audit_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--findver-validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--finlongdocqa-validation-fraction", type=float, default=0.0
    )
    args = parser.parse_args()
    report = prepare_transfer(
        args.input,
        args.output_dir,
        seed=args.seed,
        findver_validation_fraction=args.findver_validation_fraction,
        finlongdocqa_validation_fraction=(
            args.finlongdocqa_validation_fraction
        ),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
