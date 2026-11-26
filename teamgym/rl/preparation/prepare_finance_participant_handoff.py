#!/usr/bin/env python3
"""Prepare leak-closed PIU-only finance data for participant handoff RL."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any

from teamgym.core.dataset import iter_common_datapoints
from teamgym.interaction.rui_context import rui_only_datapoint

from ..data import prepare_data


DEFAULT_INPUT = Path(os.environ.get("TEAMGYM_FINANCE_ROLE_DATA", "generated/finance_participant_roles/role_augmented.jsonl")).expanduser()
DEFAULT_OUTPUT = Path(os.environ.get("TEAMGYM_FINANCE_PREPARED_ROOT", "generated/finance_participant_handoff_rl")).expanduser()


def finance_cross_metadata(row: dict[str, Any]) -> dict[str, Any]:
    dataset = str(row.get("dataset") or "").strip().casefold()
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    if dataset == "findver":
        source_identity = source.get("source_id") or source.get("report")
    elif dataset == "finlongdocqa":
        source_identity = source.get("report_id")
    else:
        raise ValueError(f"Unsupported finance dataset: {dataset!r}")
    if not source_identity:
        raise ValueError(
            f"Missing source identity for instance {row.get('instance_id')!r}"
        )
    return {
        "canonical_task_id": str(row.get("instance_id") or ""),
        "source_group_id": f"{dataset}:{source_identity}",
        "context_variant": "riu_only",
        "experiment_family": "finance_participant_handoff_rl_v1",
    }


def materialize_rui_only(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    datasets: dict[str, int] = {}
    participant_counts: dict[str, int] = {}
    with destination.open("w", encoding="utf-8") as handle:
        for row in iter_common_datapoints(input_path):
            transformed = rui_only_datapoint(row)
            transformed["_cross"] = {
                **(
                    transformed.get("_cross")
                    if isinstance(transformed.get("_cross"), dict)
                    else {}
                ),
                **finance_cross_metadata(transformed),
            }
            handle.write(json.dumps(transformed, ensure_ascii=False) + "\n")
            count += 1
            dataset = str(transformed.get("dataset") or "unknown")
            datasets[dataset] = datasets.get(dataset, 0) + 1
            participant_count = len(transformed.get("participant_directory", []))
            participant_counts[str(participant_count)] = (
                participant_counts.get(str(participant_count), 0) + 1
            )
            if limit is not None and count >= max(0, int(limit)):
                break
    if not count:
        raise ValueError(f"No finance datapoints found in {input_path}")
    return {
        "normalized_path": str(destination),
        "row_count": count,
        "dataset_counts": datasets,
        "participant_count_distribution": participant_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    normalized_path = output_root / "normalized" / "finance_riu_only.jsonl"
    materialized = materialize_rui_only(
        args.input, normalized_path, limit=args.limit
    )
    prepared = prepare_data(
        input_paths=normalized_path,
        output_dir=output_root / "riu_only",
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        split_group="source_group",
        interleave_datasets=True,
    )
    report = {**materialized, "prepared": prepared}
    report_path = output_root / "preparation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report, "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
