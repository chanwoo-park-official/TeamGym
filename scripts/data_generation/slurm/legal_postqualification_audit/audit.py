#!/usr/bin/env python3
"""Audit task-level qualification evidence across partition-mode replicates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from teamgym.core.io import read_jsonl, write_json, write_jsonl
from teamgym.core.schema import normalize_release_datapoint


EXPECTED_MODES = {
    "bottleneck",
    "common_anchor",
    "one_each",
    "overlap",
    "paired",
    "redundant_player",
}


@dataclass(frozen=True)
class Profile:
    name: str
    require_complete_grid: bool
    min_all_riu_pass_rate: float
    min_ablation_observations_per_rui: int
    min_ablation_failure_rate: float


PROFILES = (
    Profile("strict", True, 1.0, 6, 1.0),
    Profile("robust", True, 5 / 6, 3, 0.8),
    Profile("screening", True, 4 / 6, 2, 2 / 3),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", default="postqualification_audit")
    args = parser.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    leakage_rows = read_jsonl(root / "leakage_survivors" / "common_datapoints.jsonl")
    qualified_rows = [normalize_release_datapoint(row) for row in read_jsonl(root / "all_riu_qualified" / "common_datapoints.jsonl")]
    necessity_rows = read_jsonl(root / "final_qualified" / "necessity_results.jsonl")

    leakage_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in leakage_rows:
        leakage_by_key[canonical_key(row)].append(row)

    qualified_by_key: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for source_index, row in enumerate(qualified_rows):
        qualified_by_key[canonical_key(row)].append((source_index, row))

    necessity_by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in necessity_rows:
        necessity_by_source[int(result["source_row_index"])].append(result)

    task_rows = []
    rui_rows = []
    profile_counts: Counter[tuple[str, str, str]] = Counter()
    rejection_counts: Counter[tuple[str, str]] = Counter()

    for key in sorted(leakage_by_key):
        dataset, source_instance_id = key
        available = leakage_by_key[key]
        qualified = qualified_by_key.get(key, [])
        available_modes = sorted({partition_mode(row) for row in available})
        passed_modes = sorted({partition_mode(row) for _, row in qualified})
        complete_grid = set(available_modes) == EXPECTED_MODES
        all_riu_rate = len(passed_modes) / len(available_modes) if available_modes else 0.0

        evidence_by_rui: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source_index, _row in qualified:
            for result in necessity_by_source.get(source_index, []):
                evidence_by_rui[str(result["missing_rui_id"])].append(result)

        per_rui = []
        for rui_id, evidence in sorted(evidence_by_rui.items()):
            failures = sum(bool(item.get("necessity_demonstrated")) for item in evidence)
            observations = len(evidence)
            rate = failures / observations if observations else 0.0
            record = {
                "dataset": dataset,
                "source_instance_id": source_instance_id,
                "rui_id": rui_id,
                "observations": observations,
                "ablation_failures": failures,
                "ablation_failure_rate": rate,
                "observed_partition_modes": sorted({str(item["partition_mode"]) for item in evidence}),
            }
            per_rui.append(record)
            rui_rows.append(record)

        profile_status = {}
        for profile in PROFILES:
            reasons = profile_rejection_reasons(profile, complete_grid, all_riu_rate, per_rui)
            passed = not reasons
            profile_status[profile.name] = {"passed": passed, "reasons": reasons}
            profile_counts[(profile.name, dataset, "passed" if passed else "failed")] += 1
            for reason in reasons:
                rejection_counts[(profile.name, reason)] += 1

        task_rows.append(
            {
                "dataset": dataset,
                "source_instance_id": source_instance_id,
                "available_partition_modes": available_modes,
                "available_partition_mode_count": len(available_modes),
                "complete_partition_grid": complete_grid,
                "all_riu_passed_partition_modes": passed_modes,
                "all_riu_pass_count": len(passed_modes),
                "all_riu_pass_rate": all_riu_rate,
                "rui_count_with_ablation_evidence": len(per_rui),
                "minimum_ablation_observations": min((item["observations"] for item in per_rui), default=0),
                "minimum_ablation_failure_rate": min(
                    (item["ablation_failure_rate"] for item in per_rui), default=0.0
                ),
                "profiles": profile_status,
            }
        )

    summary = {
        "schema": "0712_postqualification_audit_v1",
        "run_root": str(root),
        "expected_partition_modes": sorted(EXPECTED_MODES),
        "leakage_survivor_row_count": len(leakage_rows),
        "all_riu_qualified_row_count": len(qualified_rows),
        "necessity_judgment_count": len(necessity_rows),
        "canonical_task_count": len(task_rows),
        "complete_partition_grid_task_count": sum(row["complete_partition_grid"] for row in task_rows),
        "profiles": [asdict(profile) for profile in PROFILES],
        "profile_counts": [
            {"profile": profile, "dataset": dataset, "status": status, "count": count}
            for (profile, dataset, status), count in sorted(profile_counts.items())
        ],
        "rejection_reason_counts": [
            {"profile": profile, "reason": reason, "count": count}
            for (profile, reason), count in sorted(rejection_counts.items())
        ],
    }

    write_json(output / "summary.json", summary)
    write_jsonl(output / "task_evidence.jsonl", task_rows)
    write_jsonl(output / "rui_evidence.jsonl", rui_rows)
    write_profile_table(output / "profile_sensitivity.tsv", profile_counts)
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(render_report(summary))


def profile_rejection_reasons(
    profile: Profile,
    complete_grid: bool,
    all_riu_rate: float,
    per_rui: list[dict[str, Any]],
) -> list[str]:
    reasons = []
    if profile.require_complete_grid and not complete_grid:
        reasons.append("incomplete_partition_grid")
    if all_riu_rate < profile.min_all_riu_pass_rate:
        reasons.append("insufficient_all_riu_stability")
    if not per_rui:
        reasons.append("no_ablation_evidence")
        return reasons
    if any(item["observations"] < profile.min_ablation_observations_per_rui for item in per_rui):
        reasons.append("insufficient_per_rui_repeats")
    if any(item["ablation_failure_rate"] < profile.min_ablation_failure_rate for item in per_rui):
        reasons.append("insufficient_ablation_effect")
    return reasons


def canonical_key(row: dict[str, Any]) -> tuple[str, str]:
    dataset_specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    dataset = benchmark_dataset(row)
    source_instance_id = str(
        dataset_specific.get("source_instance_id")
        or source.get("source_instance_id")
        or row.get("instance_id")
        or "unknown"
    )
    return dataset, source_instance_id


def partition_mode(row: dict[str, Any]) -> str:
    generation = row.get("generation_metadata") if isinstance(row.get("generation_metadata"), dict) else {}
    dataset_specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    return str(generation.get("allocation_mode") or dataset_specific.get("partition_mode") or "unknown")


def benchmark_dataset(row: dict[str, Any]) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    dataset_specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    family = str(source.get("source_family") or dataset_specific.get("source_family") or "").lower()
    if family.startswith("mslr_"):
        return "mslr"
    if family.startswith("sara_"):
        return "sara"
    if family.startswith("ecfr_casebook"):
        return "casebook"
    if "multi-lexsum" in family or "multilexsum" in family:
        return "multilexsum"
    return str(dataset_specific.get("benchmark_dataset") or row.get("dataset") or "unknown")


def write_profile_table(path: Path, counts: Counter[tuple[str, str, str]]) -> None:
    datasets = sorted({dataset for _, dataset, _ in counts})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["profile", "dataset", "passed", "failed"])
        for profile in (item.name for item in PROFILES):
            for dataset in datasets:
                writer.writerow(
                    [
                        profile,
                        dataset,
                        counts[(profile, dataset, "passed")],
                        counts[(profile, dataset, "failed")],
                    ]
                )


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 0712 Post-qualification Audit",
        "",
        f"- Canonical tasks: {summary['canonical_task_count']}",
        f"- Complete six-mode grids: {summary['complete_partition_grid_task_count']}",
        f"- All-RIU-qualified rows: {summary['all_riu_qualified_row_count']}",
        f"- Reused necessity judgments: {summary['necessity_judgment_count']}",
        "",
        "## Profile sensitivity",
        "",
        "| Profile | Dataset | Passed | Failed |",
        "|---|---:|---:|---:|",
    ]
    indexed = {
        (item["profile"], item["dataset"], item["status"]): item["count"]
        for item in summary["profile_counts"]
    }
    datasets = sorted({item["dataset"] for item in summary["profile_counts"]})
    for profile in (item.name for item in PROFILES):
        for dataset in datasets:
            lines.append(
                f"| {profile} | {dataset} | {indexed.get((profile, dataset, 'passed'), 0)} "
                f"| {indexed.get((profile, dataset, 'failed'), 0)} |"
            )
    lines.extend(
        [
            "",
            "This audit does not mutate or relabel the generated dataset. It identifies the task-level evidence gaps",
            "that a follow-up repeat-verification submission must fill before freezing a matched experiment grid.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
