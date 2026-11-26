#!/usr/bin/env python3
"""Execute all SARA references and enumerate intermediate subproof targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from teamgym.data.legal.sara.reference import (
    execute_reference,
    source_payload,
)
from teamgym.data.legal.sara.subproof_generation import (
    execution_is_usable,
    target_options,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--swipl-image", required=True, type=Path)
    args = parser.parse_args(argv)

    rows = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 100:
        raise RuntimeError(f"Expected 100 SARA whole-split cases, found {len(rows)}")
    audit_root = args.output_root / "metadata" / "oracle_audit"
    audited = []
    for index, row in enumerate(rows):
        execution = execute_reference(
            row,
            work_dir=audit_root / f"case_{index:03d}_{row['id']}",
            swipl_image=args.swipl_image,
        )
        source = source_payload(row)
        options = target_options(execution, source)
        audited.append(
            {
                "case_index": index,
                "case_id": row.get("id"),
                "source_row": row,
                "oracle_execution": execution,
                "target_options": options,
                "execution_usable": execution_is_usable(execution),
            }
        )

    metadata = args.output_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "audited_cases.json").write_text(
        json.dumps(audited, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema": "sara_subproof_selection_v1",
        "source_cases": len(rows),
        "executable_traces": sum(bool(row["execution_usable"]) for row in audited),
        "cases_with_targets": sum(bool(row["target_options"]) for row in audited),
        "target_option_count": sum(len(row["target_options"]) for row in audited),
        "target_predicate_counts": predicate_counts(audited),
    }
    (metadata / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def predicate_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for option in row["target_options"]:
            predicate = str(option["predicate"])
            counts[predicate] = counts.get(predicate, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
