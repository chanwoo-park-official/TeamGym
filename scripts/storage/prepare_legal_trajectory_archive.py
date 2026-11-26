#!/usr/bin/env python3
"""Prepare validated paper-cell plans for the legal trajectory archive."""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

ARCHIVE_ROOT = Path(os.environ.get("TEAMGYM_ARCHIVE_ROOT", "artifacts/archives")).expanduser()
GENERATED_ROOT = Path(os.environ.get("TEAMGYM_GENERATED_ROOT", "generated")).expanduser()
SECONDARY_GENERATED_ROOT = Path(os.environ.get("TEAMGYM_SECONDARY_GENERATED_ROOT", str(GENERATED_ROOT))).expanduser()


DATASET_ARCHIVE = ARCHIVE_ROOT / 'legal_allocated_v1'
OUTPUT_ROOT = ARCHIVE_ROOT / 'legal_paper_release_v1'
LEGACY_MODEL_ACTORS = SECONDARY_GENERATED_ROOT / '0712_legal_model_matrix'
LEGACY_MODEL_REPORTS = GENERATED_ROOT / '0712_legal_model_matrix'
LEGACY_SCHEDULE_ACTORS = SECONDARY_GENERATED_ROOT / '0712_legal_schedule_matrix'
LEGACY_SCHEDULE_REPORTS = GENERATED_ROOT / '0712_legal_schedule_matrix'
MULTILEXSUM_ROOT = GENERATED_ROOT / 'multilexsum_full_context_rerun_0712'
SARA_ROOT = GENERATED_ROOT / 'sara_final_eval_0714'
CROSS_ROOT = SECONDARY_GENERATED_ROOT / 'legal_partition_schedule_cross_0714'
HUMAN_ROOT = SECONDARY_GENERATED_ROOT / 'human_constraint_grid_0714'
READING_ROOT = SECONDARY_GENERATED_ROOT / 'participant_reading_grid_corrected_0715'
READ16_ROOT = SECONDARY_GENERATED_ROOT / 'participant_reading_16_extension_0715'
SINGLE_FULL_ROOT = SECONDARY_GENERATED_ROOT / 'legal_single_full_context_0720'
JUDGE_CONSISTENCY_POOL = GENERATED_ROOT / 'judge_consistency_0721'
JUDGE_CONSISTENCY_SCRATCH = SECONDARY_GENERATED_ROOT / 'judge_consistency_0721'
LEGACY_INPUT = GENERATED_ROOT / '0712_legal_pilot/inputs/combined_gpt55_pilot.jsonl'
RETAINED_LEGACY_DATASETS = {"cuad", "maud", "casebook", "mslr"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-archive", type=Path, default=DATASET_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_archive = args.dataset_archive.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    plan_root = output_root / "plans"
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    elif output_root.exists():
        raise FileExistsError(output_root)
    plan_root.mkdir(parents=True)

    dataset_index = read_jsonl(dataset_archive / "unpacked/provenance/instance_index.jsonl")
    selection_index = read_jsonl(
        dataset_archive / "unpacked/paper_selections/selection_index.jsonl"
    )
    archive_by_id = {row["archive_instance_id"]: row for row in dataset_index}
    standard_ids = selection_ids(selection_index, "paper_standard", 163)
    allocation_ids = selection_ids(selection_index, "paper_allocation", 618)
    allocation_by_task = {
        (
            archive_by_id[archive_id]["dataset"],
            archive_by_id[archive_id]["source_group_id"],
            archive_by_id[archive_id]["allocation_mode"],
        ): archive_id
        for archive_id in allocation_ids
    }
    if len(allocation_by_task) != len(allocation_ids):
        raise RuntimeError("Allocation selection contains duplicate task/mode keys")
    lookup = build_dataset_lookup(dataset_index)

    model_cells = build_model_cells(standard_ids, archive_by_id)
    communication_cells = build_communication_cells(standard_ids, archive_by_id)
    communication_lookup = {
        communication_key(row): row["paper_cell_id"] for row in communication_cells
    }
    if len(communication_lookup) != len(communication_cells):
        raise RuntimeError("Communication lookup contains duplicate keys")

    sara_source_cells = build_sara_partition_source_cells(lookup)
    sara_source_lookup = {
        sara_partition_key(row): row["paper_cell_id"] for row in sara_source_cells
    }
    if len(sara_source_lookup) != len(sara_source_cells):
        raise RuntimeError("SARA partition lookup contains duplicate keys")

    allocation_cells = build_allocation_cells(
        allocation_ids,
        allocation_by_task,
        archive_by_id,
        communication_lookup,
        sara_source_lookup,
    )
    human_cells = build_human_cells(lookup)
    diagnostic_cells = build_diagnostic_cells(standard_ids, archive_by_id)
    consistency_sources, consistency_tasks = build_judge_consistency_sources(lookup)

    plans = {
        "model_evidence": model_cells,
        "communication_structure": communication_cells,
        "allocation_communication": allocation_cells,
        "allocation_source_cells": sara_source_cells,
        "human_constraints": human_cells,
        "diagnostics": diagnostic_cells,
        "judge_consistency_sources": consistency_sources,
    }
    for name, rows in plans.items():
        write_jsonl(plan_root / f"{name}.jsonl", rows)
    write_jsonl(plan_root / "judge_consistency_tasks.jsonl", consistency_tasks)

    expected = {
        "model_evidence": 3912,
        "communication_structure": 13040,
        "allocation_communication": 49440,
        "allocation_source_cells": 4320,
        "human_constraints": 1881,
        "diagnostics": 1304,
        "judge_consistency_sources": 100,
    }
    actual = {name: len(rows) for name, rows in plans.items()}
    if actual != expected:
        raise RuntimeError(f"Unexpected plan dimensions: {actual}")
    reused = Counter(row.get("result_source") for row in allocation_cells)
    if reused != Counter({"new": 35120, "reused": 14320}):
        raise RuntimeError(f"Unexpected allocation reuse counts: {reused}")
    reused_targets = Counter(
        row.get("reused_from_archive") for row in allocation_cells if row.get("result_source") == "reused"
    )
    summary = {
        "schema": "legal_trajectory_archive_plan_v1",
        "paper_cell_count": (
            len(model_cells)
            + len(communication_cells)
            + len(allocation_cells)
            + len(human_cells)
            + len(diagnostic_cells)
        ),
        "plan_counts": actual,
        "allocation_result_sources": dict(reused),
        "allocation_reused_targets": dict(reused_targets),
        "judge_consistency_family_counts": dict(
            Counter(row["diagnostic_family"] for row in consistency_sources)
        ),
        "dataset_archive": str(dataset_archive),
    }
    write_json(plan_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def build_model_cells(
    standard_ids: list[str], archive_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    cells = []
    legacy_rows = read_jsonl(LEGACY_MODEL_REPORTS / "reports/cell_results.jsonl")
    for row in legacy_rows:
        if row["dataset"] not in RETAINED_LEGACY_DATASETS:
            continue
        task_index = int(row["task_index"])
        archive_id = standard_ids[task_index]
        validate_dataset(row["dataset"], archive_id, archive_by_id)
        actor = (
            LEGACY_MODEL_ACTORS
            / "actors"
            / row["model_slug"]
            / row["condition"]
            / f"shard_{task_index:03d}"
        )
        judge = (
            LEGACY_MODEL_ACTORS
            / "judges"
            / row["model_slug"]
            / row["condition"]
            / f"shard_{task_index:03d}"
        )
        cells.append(
            cell_record(
                experiment="model_evidence",
                archive_id=archive_id,
                row=row,
                source_component="legacy_0712",
                actor=actor,
                judge=judge,
                qualifiers=[row["model_slug"], row["condition"]],
            )
        )

    multilex_rows = read_jsonl(MULTILEXSUM_ROOT / "model_matrix/reports/cell_results.jsonl")
    for row in multilex_rows:
        task_index = int(row["task_index"])
        archive_id = standard_ids[89 + task_index]
        validate_dataset("multilexsum", archive_id, archive_by_id)
        actor = (
            MULTILEXSUM_ROOT
            / "model_matrix/actors"
            / row["model_slug"]
            / row["condition"]
            / f"shard_{task_index:03d}"
        )
        judge = (
            MULTILEXSUM_ROOT
            / "model_matrix/judges"
            / row["model_slug"]
            / row["condition"]
            / f"shard_{task_index:03d}"
        )
        cells.append(
            cell_record(
                experiment="model_evidence",
                archive_id=archive_id,
                row=row,
                source_component="multilexsum_0712",
                actor=actor,
                judge=judge,
                qualifiers=[row["model_slug"], row["condition"]],
            )
        )

    sara_rows = [
        row
        for row in read_jsonl(SARA_ROOT / "reports/cell_results.jsonl")
        if row.get("experiment") == "model"
    ]
    metadata = keyed_cells(SARA_ROOT / "metadata/cells.jsonl")
    for row in sara_rows:
        archive_id = standard_ids[118 + int(row["task_index"])]
        validate_dataset("sara", archive_id, archive_by_id)
        cell = metadata[int(row["cell_index"])]
        cells.append(
            cell_record(
                experiment="model_evidence",
                archive_id=archive_id,
                row=row,
                source_component="sara_final_0714",
                actor=Path(cell["output_root"]),
                judge=Path(cell["judge_root"]),
                qualifiers=[row["model_slug"], row["condition"]],
            )
        )
    validate_unique_cells(cells, 3912, "model evidence")
    return cells


def build_communication_cells(
    standard_ids: list[str], archive_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    cells = []
    legacy_rows = read_jsonl(LEGACY_SCHEDULE_REPORTS / "reports/cell_results.jsonl")
    for row in legacy_rows:
        if row["dataset"] not in RETAINED_LEGACY_DATASETS:
            continue
        task_index = int(row["task_index"])
        archive_id = standard_ids[task_index]
        validate_dataset(row["dataset"], archive_id, archive_by_id)
        actor = (
            LEGACY_SCHEDULE_ACTORS
            / "actors"
            / row["model_slug"]
            / row["schedule_slug"]
            / row["condition"]
            / f"shard_{task_index:03d}"
        )
        judge = (
            LEGACY_SCHEDULE_ACTORS
            / "judges"
            / row["model_slug"]
            / row["schedule_slug"]
            / row["condition"]
            / f"shard_{task_index:03d}"
        )
        cells.append(
            cell_record(
                experiment="communication_structure",
                archive_id=archive_id,
                row=row,
                source_component="legacy_0712",
                actor=actor,
                judge=judge,
                qualifiers=[row["model_slug"], row["schedule_slug"], row["condition"]],
            )
        )

    multilex_rows = read_jsonl(MULTILEXSUM_ROOT / "schedule_matrix/reports/cell_results.jsonl")
    for row in multilex_rows:
        task_index = int(row["task_index"])
        archive_id = standard_ids[89 + task_index]
        validate_dataset("multilexsum", archive_id, archive_by_id)
        actor = (
            MULTILEXSUM_ROOT
            / "schedule_matrix/actors"
            / row["model_slug"]
            / row["schedule_slug"]
            / row["condition"]
            / f"shard_{task_index:03d}"
        )
        judge = (
            MULTILEXSUM_ROOT
            / "schedule_matrix/judges"
            / row["model_slug"]
            / row["schedule_slug"]
            / row["condition"]
            / f"shard_{task_index:03d}"
        )
        cells.append(
            cell_record(
                experiment="communication_structure",
                archive_id=archive_id,
                row=row,
                source_component="multilexsum_0712",
                actor=actor,
                judge=judge,
                qualifiers=[row["model_slug"], row["schedule_slug"], row["condition"]],
            )
        )

    sara_rows = [
        row
        for row in read_jsonl(SARA_ROOT / "reports/cell_results.jsonl")
        if row.get("experiment") == "schedule"
    ]
    metadata = keyed_cells(SARA_ROOT / "metadata/cells.jsonl")
    for row in sara_rows:
        archive_id = standard_ids[118 + int(row["task_index"])]
        validate_dataset("sara", archive_id, archive_by_id)
        cell = metadata[int(row["cell_index"])]
        cells.append(
            cell_record(
                experiment="communication_structure",
                archive_id=archive_id,
                row=row,
                source_component="sara_final_0714",
                actor=Path(cell["output_root"]),
                judge=Path(cell["judge_root"]),
                qualifiers=[row["model_slug"], row["schedule_slug"], row["condition"]],
            )
        )
    validate_unique_cells(cells, 13040, "communication structure")
    return cells


def build_sara_partition_source_cells(
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    task_rows = read_jsonl(SARA_ROOT / "inputs/partition_grid.jsonl")
    task_ids = [lookup_archive_id(row, "sara", lookup) for row in task_rows]
    metadata = keyed_cells(SARA_ROOT / "metadata/cells.jsonl")
    rows = [
        row
        for row in read_jsonl(SARA_ROOT / "reports/cell_results.jsonl")
        if row.get("experiment") == "partition"
    ]
    cells = []
    for row in rows:
        archive_id = task_ids[int(row["task_index"])]
        cell = metadata[int(row["cell_index"])]
        cells.append(
            cell_record(
                experiment="allocation_sara_source",
                archive_id=archive_id,
                row=row,
                source_component="sara_final_0714",
                actor=Path(cell["output_root"]),
                judge=Path(cell["judge_root"]),
                qualifiers=[row["model_slug"], row["partition_mode"], row["condition"]],
                record_type="support_cell",
            )
        )
    validate_unique_cells(cells, 4320, "SARA partition source")
    return cells


def build_allocation_cells(
    allocation_ids: list[str],
    allocation_by_task: dict[tuple[str, str, str], str],
    archive_by_id: dict[str, dict[str, Any]],
    communication_lookup: dict[tuple[str, str, str, str], str],
    sara_source_lookup: dict[tuple[str, str, str], str],
) -> list[dict[str, Any]]:
    rows = read_jsonl(CROSS_ROOT / "reports/cell_results.jsonl")
    metadata = keyed_cells(CROSS_ROOT / "metadata/cells.jsonl")
    cells = []
    for row in rows:
        if row.get("variant_index") is not None:
            archive_id = allocation_ids[int(row["variant_index"])]
        else:
            key = (
                row["dataset"],
                row["canonical_task_id"],
                row["partition_mode"],
            )
            archive_id = allocation_by_task.get(key)
            if archive_id is None:
                raise RuntimeError(f"No allocation instance for reused cross cell {key}")
        validate_dataset(row["dataset"], archive_id, archive_by_id)
        qualifiers = [
            row["model_slug"],
            row["partition_mode"],
            row["schedule_slug"],
            row["condition"],
        ]
        if row["result_source"] == "new":
            source = metadata[int(row["cell_index"])]
            cell = cell_record(
                experiment="allocation_communication",
                archive_id=archive_id,
                row=row,
                source_component="cross_0714",
                actor=Path(source["output_root"]),
                judge=Path(source["judge_root"]),
                qualifiers=qualifiers,
            )
            cell["result_source"] = "new"
        else:
            communication_key_value = (
                archive_id,
                row["model_slug"],
                row["schedule_slug"],
                row["condition"],
            )
            target = communication_lookup.get(communication_key_value)
            target_archive = "communication_structure"
            if target is None:
                target = sara_source_lookup.get(
                    (archive_id, row["model_slug"], row["condition"])
                )
                target_archive = "allocation_communication"
            if target is None:
                raise RuntimeError(f"No reused source for cross cell {row['cell_index']}")
            cell = cell_record(
                experiment="allocation_communication",
                archive_id=archive_id,
                row=row,
                source_component="reused",
                actor=None,
                judge=None,
                qualifiers=qualifiers,
            )
            cell.update(
                {
                    "result_source": "reused",
                    "reused_from_archive": target_archive,
                    "reused_from_cell_id": target,
                }
            )
        cells.append(cell)
    validate_unique_cells(cells, 49440, "allocation communication")
    return cells


def build_human_cells(
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    cells = []
    components = [
        ("bounded_messaging", HUMAN_ROOT, "reports/cell_results.jsonl"),
        ("bounded_reading", READING_ROOT, "reports/cell_results.jsonl"),
        ("bounded_reading_16", READ16_ROOT, "reports/cell_results.jsonl"),
    ]
    for name, root, report_name in components:
        metadata = keyed_cells(root / "metadata/cells.jsonl")
        for row in read_jsonl(root / report_name):
            source = metadata[int(row["cell_index"])]
            dataset = str(source.get("dataset") or row.get("dataset") or "multilexsum")
            task_row = {
                "instance_id": source.get("instance_id") or row.get("instance_id"),
                "dataset_specific": {
                    "partition_mode": source.get("partition_mode") or "one_each"
                },
            }
            archive_id = lookup_archive_id(task_row, dataset, lookup)
            qualifiers = [name, str(row["cell_index"])]
            cells.append(
                cell_record(
                    experiment=name,
                    archive_id=archive_id,
                    row=row,
                    source_component=root.name,
                    actor=Path(source["output_root"]),
                    judge=Path(source["judge_root"]),
                    qualifiers=qualifiers,
                )
            )
    validate_unique_cells(cells, 1881, "human constraints")
    return cells


def build_diagnostic_cells(
    standard_ids: list[str], archive_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    metadata = keyed_cells(SINGLE_FULL_ROOT / "metadata/cells.jsonl")
    rows = read_jsonl(SINGLE_FULL_ROOT / "reports/cell_results.jsonl")
    cells = []
    for row in rows:
        archive_id = standard_ids[int(row["task_index"])]
        validate_dataset(row["dataset"], archive_id, archive_by_id)
        source = metadata[int(row["cell_index"])]
        cells.append(
            cell_record(
                experiment="single_full_context",
                archive_id=archive_id,
                row=row,
                source_component="single_full_context_0720",
                actor=Path(source["output_root"]),
                judge=Path(source["judge_root"]),
                qualifiers=[row["model_slug"]],
            )
        )
    validate_unique_cells(cells, 1304, "single full context")
    return cells


def build_judge_consistency_sources(
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_jsonl(JUDGE_CONSISTENCY_POOL / "manifest.jsonl")
    legacy_tasks = {row["instance_id"]: row for row in read_jsonl(LEGACY_INPUT)}
    sources = []
    selected_tasks: dict[str, dict[str, Any]] = {}
    for row in manifest:
        instance_id = row["instance_id"]
        task = legacy_tasks.get(instance_id)
        if task is None:
            raise RuntimeError(f"Missing judge-consistency task: {instance_id}")
        family = diagnostic_family(instance_id)
        archive_id = None
        if family in {"casebook", "mslr"}:
            try:
                archive_id = lookup_archive_id(task, family, lookup)
            except RuntimeError:
                archive_id = None
        trajectory = resolve_moved_path(Path(row["trajectory_path"]))
        source_judgment = resolve_moved_path(Path(row["source_judgment_path"]))
        if not trajectory.is_file() or not source_judgment.is_file():
            raise FileNotFoundError(f"Missing consistency source for cell {row['cell_index']}")
        cell_index = int(row["cell_index"])
        gpt54_dir = JUDGE_CONSISTENCY_SCRATCH / f"judges/gpt54/cell_{cell_index:03d}"
        sources.append(
            {
                "cell_index": cell_index,
                "instance_id": instance_id,
                "archive_instance_id": archive_id,
                "diagnostic_family": family,
                "actor_model": row["actor_model"],
                "condition": row["condition"],
                "trajectory_source_file": str(trajectory),
                "source_judgment_file": str(source_judgment),
                "gpt54_judge_source_dir": str(gpt54_dir),
                "legacy_diagnostic_only": family in {"initial_sara24", "legacy_multilexsum"},
            }
        )
        selected_tasks.setdefault(instance_id, task)
    return sources, list(selected_tasks.values())


def cell_record(
    *,
    experiment: str,
    archive_id: str,
    row: dict[str, Any],
    source_component: str,
    actor: Path | None,
    judge: Path | None,
    qualifiers: list[str],
    record_type: str = "paper_cell",
) -> dict[str, Any]:
    safe_qualifiers = [slug(value) for value in qualifiers]
    paper_cell_id = ":".join([experiment, *safe_qualifiers, archive_id])
    archive_base = Path("episodes") / source_component / Path(*safe_qualifiers) / archive_id
    return {
        "schema": "legal_trajectory_cell_v1",
        "record_type": record_type,
        "experiment": experiment,
        "paper_cell_id": paper_cell_id,
        "archive_instance_id": archive_id,
        "dataset": row.get("dataset") or ("sara" if source_component == "sara_final_0714" else None),
        "model_slug": row.get("model_slug"),
        "condition": row.get("condition"),
        "schedule_slug": row.get("schedule_slug"),
        "allocation_mode": row.get("partition_mode") or "one_each",
        "status": row.get("status"),
        "success": row.get("success"),
        "strict_all_step_success": row.get("strict_all_step_success"),
        "runtime_error": row.get("runtime_error"),
        "source_component": source_component,
        "actor_source_dir": str(actor) if actor is not None else None,
        "judge_source_dir": str(judge) if judge is not None else None,
        "actor_archive_dir": str(archive_base / "actor") if actor is not None else None,
        "judge_archive_dir": str(archive_base / "judge") if judge is not None else None,
        "source_result": row,
    }


def communication_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        row["archive_instance_id"],
        row["model_slug"],
        row["schedule_slug"],
        row["condition"],
    )


def sara_partition_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["archive_instance_id"], row["model_slug"], row["condition"]


def build_dataset_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lookup[
            (row["dataset"], row["original_instance_id"], row["allocation_mode"])
        ].append(row)
    return lookup


def lookup_archive_id(
    task: dict[str, Any],
    dataset: str,
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> str:
    specific = task.get("dataset_specific") if isinstance(task.get("dataset_specific"), dict) else {}
    cross = task.get("_cross") if isinstance(task.get("_cross"), dict) else {}
    instance_id = str(
        cross.get("original_instance_id")
        or specific.get("cross_original_instance_id")
        or strip_partition(str(task.get("instance_id") or ""))
    )
    mode = allocation_mode(task)
    candidates = lookup.get((dataset, instance_id, mode), [])
    q_variant = specific.get("pilot_q_variant")
    if q_variant is not None:
        candidates = [row for row in candidates if row.get("question_variant") == q_variant]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one archive match for {dataset}/{instance_id}/{mode}; found {len(candidates)}"
        )
    return candidates[0]["archive_instance_id"]


def allocation_mode(task: dict[str, Any]) -> str:
    specific = task.get("dataset_specific") if isinstance(task.get("dataset_specific"), dict) else {}
    generation = task.get("generation_metadata") if isinstance(task.get("generation_metadata"), dict) else {}
    private = generation.get("private_information_allocation") if isinstance(generation.get("private_information_allocation"), dict) else {}
    partitioning = generation.get("partitioning") if isinstance(generation.get("partitioning"), dict) else {}
    mode = (
        (task.get("_cross") or {}).get("partition_mode")
        or specific.get("partition_mode")
        or generation.get("allocation_mode")
        or private.get("mode")
        or partitioning.get("strategy")
        or specific.get("partition_strategy")
        or specific.get("pilot_partition_mode")
        or "one_each"
    )
    return str(mode)


def selection_ids(rows: list[dict[str, Any]], selection: str, expected: int) -> list[str]:
    selected = sorted(
        (row for row in rows if row["selection"] == selection),
        key=lambda row: int(row["selection_row"]),
    )
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} {selection} rows, found {len(selected)}")
    return [row["archive_instance_id"] for row in selected]


def validate_dataset(
    dataset: str, archive_id: str, archive_by_id: dict[str, dict[str, Any]]
) -> None:
    actual = archive_by_id[archive_id]["dataset"]
    if dataset != actual:
        raise RuntimeError(f"Dataset mismatch for {archive_id}: {dataset} != {actual}")


def validate_unique_cells(cells: list[dict[str, Any]], expected: int, name: str) -> None:
    if len(cells) != expected:
        raise RuntimeError(f"Expected {expected} {name} cells, found {len(cells)}")
    ids = {row["paper_cell_id"] for row in cells}
    if len(ids) != expected:
        raise RuntimeError(f"{name} contains duplicate cell IDs")


def keyed_cells(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["cell_index"]): row for row in read_jsonl(path)}


def diagnostic_family(instance_id: str) -> str:
    if instance_id.startswith("sara_numeric_"):
        return "initial_sara24"
    if instance_id.startswith("multilexsum_"):
        return "legacy_multilexsum"
    if instance_id.startswith("ecfr_casebook"):
        return "casebook"
    if instance_id.startswith("mslr_"):
        return "mslr"
    return "other"


def resolve_moved_path(path: Path) -> Path:
    if path.is_file():
        return path
    # Recorded manifest prefix, retained for reading historical artifacts.
    old = "/orcd/pool/004/cpark97/collabgym_fully_collaborative/generated/0712_legal_model_matrix"
    new = str(LEGACY_MODEL_ACTORS)
    candidate = Path(str(path).replace(old, new))
    if candidate.is_file():
        return candidate
    return path


def strip_partition(instance_id: str) -> str:
    return instance_id.split("__partition_", 1)[0]


def slug(value: Any) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(":", "_")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
