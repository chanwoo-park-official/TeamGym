#!/usr/bin/env python3
"""Append corrected deterministic numeric rescoring to existing judgments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from teamgym.core.io import read_jsonl
from teamgym.scoring.short_answers import (
    JUDGE_SCHEMA,
    deterministic_numeric_evaluation,
    judge_passing_score,
    normalize_judgment,
    numeric_final_step_ids,
    requires_all_steps_for_pass,
)
from teamgym.core.dataset import iter_common_datapoints


RESCORE_VERSION = "final_amount_extraction_v5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--judge-root", required=True, type=Path)
    parser.add_argument("--dataset", default="sara")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datapoints = {
        str(row.get("instance_id")): row
        for row in iter_common_datapoints(args.inputs)
        if dataset_name(row) == args.dataset
    }
    rescored = 0
    changed = 0
    skipped = 0
    for path in sorted(args.judge_root.rglob("judgments.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        row = rows[-1]
        instance_id = str(row.get("instance_id") or "")
        datapoint = datapoints.get(instance_id)
        if datapoint is None:
            continue
        if (
            row.get("schema") == JUDGE_SCHEMA
            and row.get("deterministic_rescore", {}).get("version") == RESCORE_VERSION
        ):
            skipped += 1
            continue
        old_judgment = row.get("judgment") if isinstance(row.get("judgment"), dict) else {}
        raw_response = old_judgment.get("raw_response")
        if not isinstance(raw_response, dict):
            skipped += 1
            continue
        oracle = datapoint.get("oracle") if isinstance(datapoint.get("oracle"), dict) else {}
        specific = datapoint.get("dataset_specific") if isinstance(datapoint.get("dataset_specific"), dict) else {}
        required_steps = (
            oracle.get("required_decision_steps")
            or oracle.get("reasoning_steps")
            or specific.get("reasoning_steps")
            or []
        )
        numeric = deterministic_numeric_evaluation(datapoint, str(row.get("candidate_answer") or ""))
        new_judgment = normalize_judgment(
            raw_response,
            required_steps=required_steps,
            require_all_steps_for_pass=requires_all_steps_for_pass(datapoint),
            numeric_evaluation=numeric,
            numeric_final_step_ids=numeric_final_step_ids(required_steps),
            passing_score=judge_passing_score(datapoint),
        )
        new_row = {
            **row,
            "schema": JUDGE_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "judgment": new_judgment,
            "deterministic_rescore": {
                "version": RESCORE_VERSION,
                "previous_passed": bool(old_judgment.get("holistic_passed", old_judgment.get("passed"))),
            },
        }
        if not args.dry_run:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(new_row, ensure_ascii=False) + "\n")
        rescored += 1
        changed += bool(new_judgment.get("holistic_passed")) != bool(
            old_judgment.get("holistic_passed", old_judgment.get("passed"))
        )
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "rescored": rescored,
                "pass_status_changed": changed,
                "skipped": skipped,
            },
            indent=2,
        )
    )


def dataset_name(row: dict[str, Any]) -> str:
    specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    return str(specific.get("benchmark_dataset") or row.get("dataset") or "")


if __name__ == "__main__":
    main()
