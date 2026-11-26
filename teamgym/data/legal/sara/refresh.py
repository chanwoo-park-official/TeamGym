#!/usr/bin/env python3
"""Recompute deterministic acceptance fields for target-sliced SARA records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from teamgym.data.legal.sara.subproof_generation import (
    TARGET_SLICED_POLICY,
    candidate_passes,
    candidate_to_seed,
    selected_target,
    structural_audit,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)

    refreshed = 0
    accepted = 0
    for path in sorted((args.output_root / "workers").glob("case_*/record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("generation_policy") != TARGET_SLICED_POLICY:
            continue
        candidate = record.get("candidate")
        source = record.get("source")
        execution = record.get("target_execution")
        options = record.get("target_options")
        if not all(isinstance(value, dict) for value in (candidate, source, execution)):
            continue
        if not isinstance(options, list):
            continue

        previous = record.get("structural_audit")
        structural = structural_audit(
            candidate,
            source,
            execution,
            options,
            policy=TARGET_SLICED_POLICY,
        )
        record["structural_audit"] = structural
        record["generation_initial_blind_pass"] = bool(
            record.get("blind_all_riu_audit", {}).get("accepted")
        )
        record["accepted"] = candidate_passes(
            structural,
            {"accepted": True},
            record.get("semantic_verification") or {},
            candidate,
            policy=TARGET_SLICED_POLICY,
            require_blind=False,
        )
        target = selected_target(candidate, options)
        record["seed_spec"] = (
            candidate_to_seed(
                source,
                candidate,
                target,
                policy=TARGET_SLICED_POLICY,
            )
            if record["accepted"] and target
            else None
        )
        record["deterministic_refresh"] = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_structural_audit": previous,
        }
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        refreshed += 1
        accepted += int(record["accepted"])

    print(json.dumps({"refreshed": refreshed, "accepted": accepted}, indent=2))


if __name__ == "__main__":
    main()
