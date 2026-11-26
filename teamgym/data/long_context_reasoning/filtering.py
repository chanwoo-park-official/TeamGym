#!/usr/bin/env python
"""Filter a legal JSONL dataset to examples passed by single all-PIU judging."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from teamgym.core.io import read_jsonl, write_json, write_jsonl
from teamgym.scoring.short_answers import strict_judgment_passed as strict_all_step_passed
from teamgym.data.long_context_reasoning.pipeline import (
    run_structural_leakage_funnel,
    run_structural_option_exclusion_funnel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Candidate common_datapoints JSONL.")
    parser.add_argument("--judgments-jsonl", required=True, help="Short-answer judge judgments JSONL.")
    parser.add_argument("--output-root", required=True, help="Directory for filtered dataset.")
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--require-all-steps",
        action="store_true",
        help="Keep only rows whose judgment covers every required reasoning step.",
    )
    parser.add_argument("--skip-structural-funnel", action="store_true")
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    judgments_jsonl = Path(args.judgments_jsonl).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_jsonl)
    row_by_id = {str(row.get("instance_id")): row for row in rows}
    judgments = read_jsonl(judgments_jsonl)

    kept_rows: list[dict[str, Any]] = []
    kept_judgments: list[dict[str, Any]] = []
    dropped_judgments: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    for judgment_row in judgments:
        instance_id = str(judgment_row.get("instance_id"))
        row = row_by_id.get(instance_id)
        if row is None:
            missing_ids.append(instance_id)
            continue
        judgment = judgment_row.get("judgment") or {}
        passed = bool(judgment.get("passed"))
        score = float(judgment.get("score") or 0.0)
        all_steps_passed = strict_all_step_passed(judgment)
        passes_filter = passed and score >= args.min_score
        if args.require_all_steps:
            passes_filter = passes_filter and all_steps_passed
        if passes_filter and len(kept_rows) < args.target_count:
            kept_rows.append(row)
            kept_judgments.append(judgment_row)
        else:
            dropped_judgments.append(judgment_row)

    if len(kept_rows) < args.target_count:
        raise SystemExit(
            f"Only found {len(kept_rows)} passing rows, target-count is {args.target_count}. "
            "Generate a larger candidate pool or lower the target count."
        )

    common_path = output_root / "common_datapoints.jsonl"
    write_jsonl(common_path, kept_rows)
    write_jsonl(output_root / "single_rui_passed_judgments.jsonl", kept_judgments)
    write_jsonl(output_root / "single_rui_dropped_judgments.jsonl", dropped_judgments)

    structural_report = None
    if not args.skip_structural_funnel:
        structural_report = run_structural_filters(output_root)

    report = {
        "schema": "single_rui_pass_filter_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(input_jsonl),
        "judgments_jsonl": str(judgments_jsonl),
        "output_root": str(output_root),
        "target_count": args.target_count,
        "min_score": args.min_score,
        "require_all_steps": args.require_all_steps,
        "input_row_count": len(rows),
        "judgment_count": len(judgments),
        "kept_count": len(kept_rows),
        "dropped_judgment_count": len(dropped_judgments),
        "missing_judgment_ids": missing_ids,
        "kept_instance_ids": [str(row.get("instance_id")) for row in kept_rows],
        "kept_score_stats": score_stats(kept_judgments),
        "structural_report": structural_report,
    }
    write_json(output_root / "filter_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def run_structural_filters(dataset_root: Path) -> dict[str, Any]:
    leakage_root = dataset_root / "02_leakage_deleted"
    leakage_report = run_structural_leakage_funnel(
        input_root=dataset_root / "common_datapoints.jsonl",
        output_root=leakage_root,
    )
    option_root = dataset_root / "03_option_exclusion"
    option_report = run_structural_option_exclusion_funnel(
        input_root=leakage_root / "leakage_deleted_common_datapoints.jsonl",
        output_root=option_root,
    )
    return {
        "leakage_deleted": leakage_report,
        "option_exclusion": option_report,
        "kept_common_datapoints": str(option_root / "solo_oracle_kept_common_datapoints.jsonl"),
}


def score_stats(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    scores = sorted(float((row.get("judgment") or {}).get("score") or 0.0) for row in judgments)
    if not scores:
        return {"n": 0}
    return {
        "n": len(scores),
        "mean": round(sum(scores) / len(scores), 6),
        "median": scores[len(scores) // 2],
        "min": scores[0],
        "max": scores[-1],
    }


if __name__ == "__main__":
    main()
