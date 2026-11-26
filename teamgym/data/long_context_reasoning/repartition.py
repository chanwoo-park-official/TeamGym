#!/usr/bin/env python3
"""Create a validated PIU-only partition variant of common datapoints."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

from teamgym.core.information_allocation import (
    assign_reference_units,
    owners_by_reference_unit,
)
from teamgym.core.io import read_jsonl, write_json, write_jsonl
from teamgym.core.schema import validate_common_datapoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allocation-mode", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    source_rows = read_jsonl(args.input)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        try:
            row = repartition(source, mode=args.allocation_mode, seed=args.seed + index)
            validation = validate_common_datapoint(row)
            row["format_validation"] = validation
            if not validation["valid"]:
                raise ValueError("common datapoint validation failed")
            kept.append(row)
        except Exception as exc:  # noqa: BLE001
            rejected.append({"instance_id": source.get("instance_id"), "error": f"{type(exc).__name__}: {exc}"})

    output = Path(args.output_root)
    write_jsonl(output / "common_datapoints.jsonl", kept)
    write_jsonl(output / "rejected.jsonl", rejected)
    write_json(
        output / "report.json",
        {
            "input_count": len(source_rows),
            "kept_count": len(kept),
            "rejected_count": len(rejected),
            "allocation_mode": args.allocation_mode,
            "context_variant": "rui_only",
        },
    )


def repartition(source: dict[str, Any], *, mode: str, seed: int) -> dict[str, Any]:
    row = copy.deepcopy(source)
    ruis = [item for item in row.get("reference_information_units", []) if isinstance(item, dict)]
    rui_ids = [str(item["rui_id"]) for item in ruis]
    oracle = row.get("oracle") if isinstance(row.get("oracle"), dict) else {}
    sufficient_sets = [
        [str(item) for item in candidate.get("rui_ids", [])]
        for candidate in oracle.get("smallest_sufficient_rui_sets", [])
        if isinstance(candidate, dict) and candidate.get("rui_ids")
    ] or [[str(item) for item in oracle.get("required_rui_ids", rui_ids)]]
    allocation = assign_reference_units(
        rui_ids,
        mode=mode,
        rng=random.Random(seed),
        min_participants=max(2, len(row.get("participants", []))),
        smallest_sufficient_sets=sufficient_sets,
    )
    participant_ids = [_participant_id(index) for index in range(len(allocation.private_assignment))]
    source_ids = sorted(allocation.private_assignment)
    private_assignment = {
        participant_ids[index]: list(allocation.private_assignment[source_id])
        for index, source_id in enumerate(source_ids)
    }
    common_ids = list(allocation.common_rui_ids)
    effective_assignment = {
        participant_id: list(dict.fromkeys([*private_ids, *common_ids]))
        for participant_id, private_ids in private_assignment.items()
    }
    owners = owners_by_reference_unit(effective_assignment)
    rui_by_id = {str(item["rui_id"]): item for item in ruis}

    participants = []
    contexts = []
    for participant_id in participant_ids:
        visible_ids = effective_assignment[participant_id]
        context_id = f"{participant_id}_riu_only_context"
        text = "\n\n".join(str(rui_by_id[rui_id].get("text") or "").strip() for rui_id in visible_ids).strip()
        participants.append(
            {
                "participant_id": participant_id,
                "role_name": "Document analyst",
                "private_context_unit_ids": [context_id],
                "owned_rui_ids": visible_ids,
            }
        )
        contexts.append(
            {
                "context_unit_id": context_id,
                "assigned_to": [participant_id],
                "text": text,
                "hidden_contains_rui_ids": visible_ids,
                "metadata": {"context_kind": "riu_only_partition"},
            }
        )

    for rui in ruis:
        rui["owned_by"] = owners.get(str(rui["rui_id"]), [])
    row["participants"] = participants
    row["participant_directory"] = [
        {"participant_id": item["participant_id"], "role_name": item["role_name"]}
        for item in participants
    ]
    row["private_context_units"] = contexts
    metadata = row.setdefault("generation_metadata", {})
    metadata["allocation_mode"] = allocation.mode
    metadata["requested_allocation_mode"] = mode
    metadata["common_rui_ids"] = common_ids
    metadata["context_variant"] = "rui_only_partition"
    metadata["private_information_allocation"] = {
        **allocation.to_metadata(),
        "mapped_private_assignment": private_assignment,
        "mapped_effective_assignment": effective_assignment,
    }
    row.setdefault("dataset_specific", {})["partition_mode"] = allocation.mode
    row["dataset_specific"]["context_variant"] = "rui_only_partition"
    return row


def _participant_id(index: int) -> str:
    if index >= 26:
        raise ValueError("at most 26 participants are supported")
    return f"participant_{chr(ord('a') + index)}"


if __name__ == "__main__":
    main()
