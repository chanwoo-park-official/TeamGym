#!/usr/bin/env python3
"""Apply the repeated blind-solve criterion to SARA subproof candidates."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--minimum-passes", default=4, type=int)
    parser.add_argument("--target-count", default=30, type=int)
    args = parser.parse_args(argv)

    tracking = json.loads(
        (args.output_root / "tracking" / "generation_tracking.json").read_text(encoding="utf-8")
    )
    final_rows = []
    qualified_records: list[dict[str, Any]] = []
    for row in tracking:
        index = int(row["case_index"])
        path = args.output_root / "workers" / f"case_{index:03d}" / "record.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        generation_accepted = row.get("generation_accepted") is True
        trials = [bool(record.get("blind_all_riu_audit", {}).get("accepted"))] if generation_accepted else []
        runtime_failures = 0
        if generation_accepted:
            for trial in range(2, 6):
                trial_path = args.output_root / "qualification" / f"case_{index:03d}" / f"trial_{trial}.json"
                if not trial_path.is_file():
                    trials.append(False)
                    runtime_failures += 1
                    continue
                result = json.loads(trial_path.read_text(encoding="utf-8"))
                trials.append(result.get("passed") is True)
                runtime_failures += bool(result.get("runtime_error"))
        passes = sum(trials)
        qualified = generation_accepted and passes >= args.minimum_passes
        if qualified:
            qualified_records.append(record)
        final_rows.append(
            {
                **row,
                "qualification_passes": passes if generation_accepted else None,
                "qualification_trials": len(trials),
                "qualification_runtime_failures": runtime_failures,
                "final_qualified": qualified,
                "trial_pattern": "".join("P" if value else "F" for value in trials),
            }
        )

    qualified_records.sort(
        key=lambda row: (
            -sum(
                bool(item.get("passed"))
                for item in qualification_results(args.output_root, int(row["case_index"]))
            ),
            int(row["case_index"]),
        )
    )
    selected_records = qualified_records[: args.target_count]
    final_dir = args.output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(final_dir / "qualified_records.jsonl", qualified_records)
    write_jsonl(final_dir / "qualified_seed.jsonl", [row["seed_spec"] for row in qualified_records])
    write_jsonl(final_dir / "selected_seed.jsonl", [row["seed_spec"] for row in selected_records])
    write_jsonl(final_dir / "selected30_seed.jsonl", [row["seed_spec"] for row in qualified_records[:30]])
    write_tracking(args.output_root / "tracking", final_rows)
    report = {
        "schema": "sara_subproof_final_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_cases": len(final_rows),
        "generation_accepted": sum(bool(row["generation_accepted"]) for row in final_rows),
        "qualified_4_of_5": len(qualified_records),
        "selected_for_pilot": len(selected_records),
        "target_count_reached": len(selected_records) >= args.target_count,
        "single_all_riu_trial_pass_rate": trial_pass_rate(final_rows),
        "qualified_case_ids": [row.get("case_id") for row in qualified_records],
    }
    (final_dir / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def qualification_results(root: Path, case_index: int) -> list[dict[str, Any]]:
    results = []
    for trial in range(2, 6):
        path = root / "qualification" / f"case_{case_index:03d}" / f"trial_{trial}.json"
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def trial_pass_rate(rows: list[dict[str, Any]]) -> float | None:
    accepted = [row for row in rows if row["generation_accepted"]]
    if not accepted:
        return None
    return round(
        sum(int(row["qualification_passes"]) for row in accepted)
        / sum(int(row["qualification_trials"]) for row in accepted),
        6,
    )


def write_tracking(root: Path, rows: list[dict[str, Any]]) -> None:
    (root / "final_tracking.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (root / "final_tracking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
