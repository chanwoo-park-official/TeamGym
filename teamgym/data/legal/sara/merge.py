#!/usr/bin/env python3
"""Merge SARA subproof construction records and prepare qualification work."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .subproof_generation import TARGET_SLICED_POLICY


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    audited = json.loads(
        (args.output_root / "metadata" / "audited_cases.json").read_text(encoding="utf-8")
    )
    if args.limit is not None:
        audited = audited[: args.limit]
    records = load_records(args.output_root, audited)
    accepted = [row for row in records if record_is_accepted(row)]
    rejected = [row for row in records if not record_is_accepted(row)]
    seed_dir = args.output_root / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(seed_dir / "accepted_records.jsonl", accepted)
    write_jsonl(seed_dir / "rejected_records.jsonl", rejected)
    write_jsonl(seed_dir / "generation_seed.jsonl", [row["seed_spec"] for row in accepted])

    manifest = [
        {"case_index": int(row["case_index"]), "trial": trial}
        for row in accepted
        for trial in range(2, 6)
    ]
    (args.output_root / "metadata" / "qualification_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    tracking = [tracking_row(row) for row in records]
    write_tracking(args.output_root / "tracking", tracking)
    report = {
        "schema": "sara_subproof_generation_merge_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_cases": len(audited),
        "records": len(records),
        "runtime_failures": sum(bool(row.get("runtime_error")) for row in records),
        "failed_stage_counts": failed_stage_counts(records),
        "structural_passed": sum(bool(row.get("structural_audit", {}).get("accepted")) for row in records),
        "blind_trial_1_passed": sum(bool(row.get("blind_all_riu_audit", {}).get("accepted")) for row in records),
        "semantic_verification_passed": sum(
            row.get("semantic_verification", {}).get("accepted") is True for row in records
        ),
        "accepted_after_generation": len(accepted),
        "rejected_after_generation": len(rejected),
        "selected_target_counts": selected_target_counts(accepted),
        "qualification_calls": len(manifest),
    }
    (seed_dir / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def load_records(root: Path, audited: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for index, audit in enumerate(audited):
        path = root / "workers" / f"case_{index:03d}" / "record.json"
        if path.is_file():
            records.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            records.append(
                {
                    "schema": "sara_subproof_generation_record_v1",
                    "case_index": index,
                    "case_id": audit["case_id"],
                    "accepted": False,
                    "failed_stage": "missing_record",
                    "runtime_error": {
                        "type": "MissingRecord",
                        "message": "The generation worker did not produce a record.",
                    },
                }
            )
    return records


def record_is_accepted(record: dict[str, Any]) -> bool:
    if record.get("accepted") is not True:
        return False
    if record.get("generation_policy") != TARGET_SLICED_POLICY:
        return False
    return record.get("semantic_verification", {}).get("atomic_rius") is True


def tracking_row(record: dict[str, Any]) -> dict[str, Any]:
    issue_groups = (
        record.get("structural_audit", {}).get("issues", []),
        record.get("blind_all_riu_audit", {}).get("issues", []),
        record.get("semantic_verification", {}).get("issues", []),
    )
    issues = [str(issue) for group in issue_groups for issue in group]
    runtime = record.get("runtime_error")
    if isinstance(runtime, dict):
        issues.append(f"{runtime.get('type')}: {runtime.get('message')}")
    target = record.get("selected_target") or {}
    return {
        "case_index": record.get("case_index"),
        "case_id": record.get("case_id"),
        "target_predicate": target.get("predicate"),
        "target_answer": target.get("answer"),
        "structural_validation": bool(record.get("structural_audit", {}).get("accepted")),
        "blind_trial_1": bool(record.get("blind_all_riu_audit", {}).get("accepted")),
        "semantic_verification": record.get("semantic_verification", {}).get("accepted") is True,
        "generation_accepted": record_is_accepted(record),
        "riu_count": record.get("structural_audit", {}).get("riu_count"),
        "qualification_passes": None,
        "qualification_trials": None,
        "final_qualified": None,
        "issues": " | ".join(issues[:8]),
    }


def selected_target_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        predicate = str((row.get("selected_target") or {}).get("predicate") or "unknown")
        counts[predicate] = counts.get(predicate, 0) + 1
    return dict(sorted(counts.items()))


def failed_stage_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        stage = str(row.get("failed_stage") or "constructed_and_verified")
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def write_tracking(root: Path, rows: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "generation_tracking.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (root / "generation_tracking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
