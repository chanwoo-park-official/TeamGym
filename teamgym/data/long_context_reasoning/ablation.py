#!/usr/bin/env python
"""Create leave-one-PIU-out variants for single-agent necessity checks."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from teamgym.core.io import read_jsonl, write_json, write_jsonl
from teamgym.core.schema import normalize_release_datapoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selected-only", action="store_true", help="Ablate selected_rui_ids only when available.")
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = [normalize_release_datapoint(row) for row in read_jsonl(input_jsonl)]
    ablated_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for source_row_index, row in enumerate(rows):
        ruis = [rui for rui in row.get("reference_information_units", []) if isinstance(rui, dict)]
        selected = selected_rui_ids(row)
        candidate_ids = selected if args.selected_only and selected else [str(rui.get("rui_id")) for rui in ruis]
        for missing_id in candidate_ids:
            if missing_id not in {str(rui.get("rui_id")) for rui in ruis}:
                continue
            out = deepcopy(row)
            out["instance_id"] = f"{row.get('instance_id')}_minus_{safe_id(missing_id)}"
            out["reference_information_units"] = [
                rui for rui in out.get("reference_information_units", []) if str(rui.get("rui_id")) != missing_id
            ]
            out["participants"] = prune_participants(out.get("participants", []), missing_id)
            out["dataset_specific"] = {
                **(out.get("dataset_specific") or {}),
                "ablation_type": "leave_one_rui_out",
                "source_instance_id": row.get("instance_id"),
                "missing_rui_id": missing_id,
            }
            out["source"] = {
                **(out.get("source") or {}),
                "ablation_type": "leave_one_rui_out",
                "source_instance_id": row.get("instance_id"),
                "missing_rui_id": missing_id,
            }
            ablated_rows.append(out)
            manifest_rows.append(
                {
                    "source_row_index": source_row_index,
                    "source_dataset": row.get("dataset"),
                    "source_partition_mode": partition_mode(row),
                    "source_instance_id": row.get("instance_id"),
                    "ablated_instance_id": out["instance_id"],
                    "missing_rui_id": missing_id,
                    "remaining_rui_count": len(out["reference_information_units"]),
                }
            )

    write_jsonl(output_root / "common_datapoints.jsonl", ablated_rows)
    write_jsonl(output_root / "ablation_manifest.jsonl", manifest_rows)
    report = {
        "schema": "leave_one_rui_out_dataset_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(input_jsonl),
        "output_root": str(output_root),
        "source_row_count": len(rows),
        "ablated_row_count": len(ablated_rows),
        "selected_only": bool(args.selected_only),
        "common_datapoints": str(output_root / "common_datapoints.jsonl"),
    }
    write_json(output_root / "build_report.json", report)
    print(json.dumps(report, indent=2))


def selected_rui_ids(row: dict[str, Any]) -> list[str]:
    dataset_specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    selected = dataset_specific.get("selected_rui_ids")
    if isinstance(selected, list):
        return [str(item) for item in selected]
    sets = (row.get("oracle") or {}).get("smallest_sufficient_rui_sets") or []
    if sets and isinstance(sets[0], dict) and isinstance(sets[0].get("rui_ids"), list):
        return [str(item) for item in sets[0]["rui_ids"]]
    return []


def partition_mode(row: dict[str, Any]) -> str:
    generation = row.get("generation_metadata") if isinstance(row.get("generation_metadata"), dict) else {}
    dataset_specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    return str(generation.get("allocation_mode") or dataset_specific.get("partition_mode") or "unknown")


def prune_participants(participants: Any, missing_id: str) -> list[dict[str, Any]]:
    out = []
    for participant in participants if isinstance(participants, list) else []:
        if not isinstance(participant, dict):
            continue
        copy = deepcopy(participant)
        for key in ("private_context_unit_ids", "owned_rui_ids", "reference_information_unit_ids"):
            if isinstance(copy.get(key), list):
                copy[key] = [item for item in copy[key] if str(item) != missing_id]
        out.append(copy)
    return out


def safe_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")[:120]


if __name__ == "__main__":
    main()
