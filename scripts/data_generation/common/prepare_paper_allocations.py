#!/usr/bin/env python3
"""Freeze the available legal partition variants and build the missing cross cells."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from teamgym.core.schema import normalize_release_datapoint, validate_common_datapoint


DATASETS = ("cuad", "maud", "casebook", "mslr", "multilexsum", "sara")
MODES = ("one_each", "paired", "overlap", "redundant_player", "common_anchor", "bottleneck")
CONDITIONS = ("multi_riu", "multi_full")
EXPECTED_VARIANTS = {
    "cuad": 44,
    "maud": 62,
    "casebook": 77,
    "mslr": 24,
    "multilexsum": 141,
    "sara": 270,
}
EXPECTED_REUSED_CELLS = 14_320
EXPECTED_NEW_CELLS = 35_120
EXPECTED_FULL_CELLS = 49_440


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--schedules", required=True, type=Path)
    parser.add_argument("--selected-input", required=True, type=Path)
    parser.add_argument("--qualified-variants", required=True, type=Path)
    parser.add_argument("--cuad-variants", required=True, type=Path)
    parser.add_argument("--maud-variants", required=True, type=Path)
    parser.add_argument("--multilexsum-full", required=True, type=Path)
    parser.add_argument("--sara-input-root", required=True, type=Path)
    parser.add_argument("--old-schedule-results", required=True, type=Path)
    parser.add_argument("--sara-results", required=True, type=Path)
    args = parser.parse_args()

    root = args.output_root.expanduser().resolve()
    models = read_tsv(args.models)
    schedules = read_tsv(args.schedules)
    if len(models) != 8 or len(schedules) != 5:
        raise RuntimeError(f"Expected 8 models and 5 schedules; found {len(models)} and {len(schedules)}")

    selected = selected_tasks(args.selected_input)
    variants, selected_modes = build_non_sara_variants(
        selected,
        qualified_path=args.qualified_variants,
        cuad_path=args.cuad_variants,
        maud_path=args.maud_variants,
        multilexsum_full_path=args.multilexsum_full,
    )
    sara_variants, sara_order = build_sara_variants(args.sara_input_root)
    variants.extend(sara_variants)
    variants.sort(key=lambda row: (DATASETS.index(row["_cross"]["dataset"]), row["_cross"]["canonical_task_id"], MODES.index(row["_cross"]["partition_mode"])))

    counts = Counter(row["_cross"]["dataset"] for row in variants)
    if dict(counts) != EXPECTED_VARIANTS:
        raise RuntimeError(f"Unexpected partition-variant counts: {dict(counts)}")
    if len(variants) != sum(EXPECTED_VARIANTS.values()):
        raise RuntimeError(f"Expected {sum(EXPECTED_VARIANTS.values())} variants, found {len(variants)}")

    for index, row in enumerate(variants):
        row["_cross"]["variant_index"] = index
    input_path = root / "inputs" / "partition_variants.jsonl"
    write_jsonl(input_path, variants)

    reused = build_reused_results(
        old_schedule_path=args.old_schedule_results,
        sara_results_path=args.sara_results,
        selected=selected,
        selected_modes=selected_modes,
        sara_order=sara_order,
    )
    reused_by_key = {result_key(row): row for row in reused}
    if len(reused_by_key) != EXPECTED_REUSED_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_REUSED_CELLS:,} unique reusable cells, found {len(reused_by_key):,}"
        )
    write_jsonl(root / "metadata" / "reused_results.jsonl", sorted(reused_by_key.values(), key=result_key))

    cells = []
    target_count = 0
    per_model = Counter()
    for model_index, (model_slug, model, provider, endpoint, base_url) in enumerate(models):
        for variant_index, row in enumerate(variants):
            cross = row["_cross"]
            for schedule_slug, structure, turn_policy in schedules:
                for condition in CONDITIONS:
                    target_count += 1
                    identity = {
                        "dataset": cross["dataset"],
                        "canonical_task_id": cross["canonical_task_id"],
                        "partition_mode": cross["partition_mode"],
                        "model_slug": model_slug,
                        "schedule_slug": schedule_slug,
                        "condition": condition,
                    }
                    if result_key(identity) in reused_by_key:
                        continue
                    cell_index = len(cells)
                    output_root = (
                        root
                        / "actors"
                        / model_slug
                        / cross["dataset"]
                        / cross["partition_mode"]
                        / schedule_slug
                        / condition
                        / f"variant_{variant_index:04d}"
                    )
                    judge_root = (
                        root
                        / "judges"
                        / model_slug
                        / cross["dataset"]
                        / cross["partition_mode"]
                        / schedule_slug
                        / condition
                        / f"variant_{variant_index:04d}"
                    )
                    cells.append(
                        {
                            **identity,
                            "cell_index": cell_index,
                            "model_index": model_index,
                            "model": model,
                            "provider": provider,
                            "endpoint": endpoint,
                            "base_url": base_url,
                            "variant_index": variant_index,
                            "variant_count": len(variants),
                            "input_path": str(input_path),
                            "answer_type": answer_type(row),
                            "communication_structure": structure,
                            "turn_policy": turn_policy,
                            "output_root": str(output_root),
                            "judge_root": str(judge_root),
                        }
                    )
                    per_model[model_slug] += 1

    if target_count != EXPECTED_FULL_CELLS:
        raise RuntimeError(f"Expected {EXPECTED_FULL_CELLS:,} target cells, found {target_count:,}")
    if len(cells) != EXPECTED_NEW_CELLS:
        raise RuntimeError(f"Expected {EXPECTED_NEW_CELLS:,} new cells, found {len(cells):,}")
    if set(per_model.values()) != {EXPECTED_NEW_CELLS // len(models)}:
        raise RuntimeError(f"Unequal cells per model: {dict(per_model)}")
    write_jsonl(root / "metadata" / "cells.jsonl", cells)

    mode_counts = Counter(
        (row["_cross"]["dataset"], row["_cross"]["partition_mode"]) for row in variants
    )
    task_counts = {
        dataset: len(
            {
                row["_cross"]["canonical_task_id"]
                for row in variants
                if row["_cross"]["dataset"] == dataset
            }
        )
        for dataset in DATASETS
    }
    summary = {
        "schema": "legal_partition_schedule_cross_preparation_v1",
        "variant_count": len(variants),
        "variant_counts_by_dataset": dict(sorted(counts.items())),
        "variant_counts_by_dataset_mode": [
            {"dataset": dataset, "partition_mode": mode, "count": count}
            for (dataset, mode), count in sorted(mode_counts.items())
        ],
        "task_counts_by_dataset": task_counts,
        "model_count": len(models),
        "schedule_count": len(schedules),
        "condition_count": len(CONDITIONS),
        "full_factorial_cell_count": target_count,
        "reused_cell_count": len(reused_by_key),
        "new_cell_count": len(cells),
        "cells_per_model": EXPECTED_NEW_CELLS // len(models),
        "new_cells_by_model": dict(sorted(per_model.items())),
        "judge_model": "gpt-5.4",
    }
    write_json(root / "metadata" / "preparation_summary.json", summary)
    print(json.dumps(summary, indent=2))


def selected_tasks(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    selected: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        dataset = dataset_name(row)
        if dataset not in {"cuad", "maud", "casebook", "mslr"}:
            continue
        selected[dataset][canonical_task_id(row, dataset)] = row
    expected = {"cuad": 30, "maud": 30, "casebook": 18, "mslr": 11}
    counts = {dataset: len(rows) for dataset, rows in selected.items()}
    if counts != expected:
        raise RuntimeError(f"Unexpected selected task counts: {counts}")
    return selected


def build_non_sara_variants(
    selected: dict[str, dict[str, dict[str, Any]]],
    *,
    qualified_path: Path,
    cuad_path: Path,
    maud_path: Path,
    multilexsum_full_path: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    variants: list[dict[str, Any]] = []
    selected_modes: dict[tuple[str, str], str] = {}

    for dataset, path, q_variant in (
        ("cuad", cuad_path, None),
        ("maud", maud_path, "maud_q01"),
    ):
        grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for wrapper in read_jsonl(path):
            if not bool((wrapper.get("validation") or {}).get("valid")):
                continue
            if q_variant and wrapper.get("q_variant") != q_variant:
                continue
            task_id = str(wrapper.get("source_id") or "").split("__", 1)[0]
            if task_id not in selected[dataset]:
                continue
            mode = str(wrapper.get("partition_strategy") or "")
            datapoint = wrapper.get("common_datapoint")
            if mode not in MODES or not isinstance(datapoint, dict):
                continue
            grouped[(task_id, mode)].append((str(wrapper.get("source_id") or ""), datapoint))

        for task_id, selected_row in selected[dataset].items():
            selected_mode = partition_mode(selected_row)
            selected_modes[(dataset, task_id)] = selected_mode
            selected_source = str((selected_row.get("dataset_specific") or {}).get("pilot_source_id") or "")
            for mode in MODES:
                candidates = grouped.get((task_id, mode), [])
                if not candidates:
                    continue
                candidates.sort(key=lambda item: (item[0] != selected_source, item[0]))
                row = deepcopy(candidates[0][1])
                row["dataset_specific"] = {
                    **(row.get("dataset_specific") or {}),
                    "benchmark_dataset": dataset,
                    "pilot_case_id": task_id,
                    "pilot_source_id": candidates[0][0],
                    "partition_mode": mode,
                }
                variants.append(finalize_variant(row, dataset, task_id, mode))

    qualified_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for public in read_jsonl(qualified_path):
        row = normalize_release_datapoint(public)
        dataset = dataset_name(row)
        if dataset in {"casebook", "mslr", "multilexsum"}:
            qualified_by_dataset[dataset].append(row)

    for dataset in ("casebook", "mslr"):
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in qualified_by_dataset[dataset]:
            task_id = canonical_task_id(row, dataset)
            if task_id in selected[dataset]:
                grouped[(task_id, partition_mode(row))].append(row)
        for task_id, selected_row in selected[dataset].items():
            selected_mode = partition_mode(selected_row)
            selected_modes[(dataset, task_id)] = selected_mode
            for mode in MODES:
                candidates = sorted(grouped.get((task_id, mode), []), key=lambda row: str(row.get("_input_line_number") or ""))
                if candidates:
                    variants.append(finalize_variant(deepcopy(candidates[0]), dataset, task_id, mode))

    full_multilexsum = {
        canonical_task_id(row, "multilexsum"): normalize_release_datapoint(row)
        for row in read_jsonl(multilexsum_full_path)
    }
    grouped_multi: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in qualified_by_dataset["multilexsum"]:
        task_id = canonical_task_id(row, "multilexsum")
        if task_id in full_multilexsum:
            grouped_multi[(task_id, partition_mode(row))].append(row)
    for task_id, base in sorted(full_multilexsum.items()):
        for mode in MODES:
            candidates = grouped_multi.get((task_id, mode), [])
            if candidates:
                projected = project_multilexsum_full_context(base, candidates[0], mode=mode)
                variants.append(finalize_variant(projected, "multilexsum", task_id, mode))
    return variants, selected_modes


def build_sara_variants(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    variants = []
    order: list[str] | None = None
    for mode in MODES:
        rows = read_jsonl(root / "by_partition" / f"{mode}.jsonl")
        task_ids = [canonical_task_id(row, "sara") for row in rows]
        if order is None:
            order = task_ids
        elif task_ids != order:
            raise RuntimeError(f"SARA task order differs for partition {mode}")
        for row, task_id in zip(rows, task_ids, strict=True):
            variants.append(finalize_variant(deepcopy(row), "sara", task_id, mode))
    if order is None or len(order) != 45:
        raise RuntimeError("Expected 45 ordered SARA tasks")
    return variants, order


def project_multilexsum_full_context(
    base: dict[str, Any], partition: dict[str, Any], *, mode: str
) -> dict[str, Any]:
    row = deepcopy(base)
    partition_participants = [
        item for item in partition.get("participants", []) if isinstance(item, dict)
    ]
    ownership = {
        str(item["participant_id"]): [str(rui_id) for rui_id in item.get("owned_rui_ids", [])]
        for item in partition_participants
    }
    if not ownership:
        raise RuntimeError(f"MultiLexSum {row.get('instance_id')} {mode} has no participants")

    participants = [
        {
            "participant_id": participant_id,
            "role_name": "Case-file analyst",
            "private_context_unit_ids": [],
            "owned_rui_ids": rui_ids,
            "metadata": {
                "full_context_partition_role": "original Multi-LexSum source windows plus unrelated case-file chunks"
            },
        }
        for participant_id, rui_ids in ownership.items()
    ]
    participant_by_id = {item["participant_id"]: item for item in participants}
    participant_ids = list(participant_by_id)
    units = [deepcopy(item) for item in row.get("private_context_units", []) if isinstance(item, dict)]
    decoy_index = 0
    for unit in units:
        hidden = [str(item) for item in unit.get("hidden_contains_rui_ids", [])]
        if len(hidden) > 1:
            raise RuntimeError(
                f"MultiLexSum source unit {unit.get('context_unit_id')} contains multiple RIUs"
            )
        if hidden:
            assigned = [
                participant_id
                for participant_id, owned in ownership.items()
                if hidden[0] in owned
            ]
            if not assigned:
                raise RuntimeError(
                    f"No owner for {hidden[0]} in MultiLexSum {row.get('instance_id')} {mode}"
                )
        else:
            assigned = [participant_ids[decoy_index % len(participant_ids)]]
            decoy_index += 1
        unit["assigned_to"] = assigned
        for participant_id in assigned:
            participant_by_id[participant_id]["private_context_unit_ids"].append(
                str(unit["context_unit_id"])
            )

    owners_by_rui = defaultdict(list)
    for participant_id, rui_ids in ownership.items():
        for rui_id in rui_ids:
            owners_by_rui[rui_id].append(participant_id)
    for rui in row.get("reference_information_units", []):
        if isinstance(rui, dict):
            rui["owned_by"] = owners_by_rui.get(str(rui.get("rui_id")), [])

    row["participants"] = participants
    row["participant_directory"] = [
        {"participant_id": item["participant_id"], "role_name": item["role_name"]}
        for item in participants
    ]
    row["private_context_units"] = units
    row["generation_metadata"] = {
        **(row.get("generation_metadata") or {}),
        "allocation_mode": mode,
        "context_variant": "full_original_source_packets",
        "partition_projection": "0712_qualified_riu_ownership_onto_restored_full_context",
    }
    row["dataset_specific"] = {
        **(row.get("dataset_specific") or {}),
        "benchmark_dataset": "multilexsum",
        "partition_mode": mode,
        "context_variant": "full_original_source_packets",
    }
    validation = validate_common_datapoint(row)
    if not validation.get("valid"):
        raise RuntimeError(
            f"Projected MultiLexSum row failed validation: {row.get('instance_id')} {mode}: {validation}"
        )
    row["format_validation"] = validation
    return row


def finalize_variant(
    row: dict[str, Any], dataset: str, task_id: str, mode: str
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unknown partition mode: {mode}")
    original_instance_id = str(row.get("instance_id") or task_id)
    suffix = re.sub(r"[^a-z0-9_]+", "_", mode.lower())
    row["instance_id"] = f"{original_instance_id}__partition_{suffix}"
    row["dataset_specific"] = {
        **(row.get("dataset_specific") or {}),
        "benchmark_dataset": dataset,
        "partition_mode": mode,
        "cross_original_instance_id": original_instance_id,
        "cross_canonical_task_id": task_id,
    }
    row["_cross"] = {
        "dataset": dataset,
        "canonical_task_id": task_id,
        "partition_mode": mode,
        "original_instance_id": original_instance_id,
    }
    return row


def build_reused_results(
    *,
    old_schedule_path: Path,
    sara_results_path: Path,
    selected: dict[str, dict[str, dict[str, Any]]],
    selected_modes: dict[tuple[str, str], str],
    sara_order: list[str],
) -> list[dict[str, Any]]:
    selected_by_instance = {}
    for dataset, tasks in selected.items():
        for task_id, row in tasks.items():
            selected_by_instance[(dataset, str(row.get("instance_id") or ""))] = (
                task_id,
                selected_modes[(dataset, task_id)],
            )

    reused = []
    for row in read_jsonl(old_schedule_path):
        dataset = str(row.get("dataset") or "")
        if dataset not in {"cuad", "maud", "casebook", "mslr"}:
            continue
        identity = selected_by_instance.get((dataset, str(row.get("instance_id") or "")))
        if identity is None:
            raise RuntimeError(
                f"Cannot map reusable {dataset} result {row.get('instance_id')}"
            )
        task_id, mode = identity
        reused.append(normalize_reused(row, task_id=task_id, mode=mode))

    sara_by_key = {}
    for row in read_jsonl(sara_results_path):
        experiment = str(row.get("experiment") or "")
        if experiment == "schedule":
            task_index = int(row["task_index"])
            mode = "one_each"
            schedule = str(row.get("schedule_slug") or "")
        elif experiment == "partition":
            task_index = int(row["task_index"]) % 45
            mode = str(row.get("partition_mode") or "")
            schedule = "coord_select_one"
        else:
            continue
        normalized = normalize_reused(
            {**row, "dataset": "sara", "schedule_slug": schedule},
            task_id=sara_order[task_index],
            mode=mode,
        )
        sara_by_key[result_key(normalized)] = normalized
    if len(sara_by_key) != 7_200:
        raise RuntimeError(f"Expected 7,200 reusable SARA cells, found {len(sara_by_key):,}")
    reused.extend(sara_by_key.values())
    return reused


def normalize_reused(
    row: dict[str, Any], *, task_id: str, mode: str
) -> dict[str, Any]:
    return {
        "dataset": str(row.get("dataset") or ""),
        "canonical_task_id": task_id,
        "partition_mode": mode,
        "model_slug": str(row.get("model_slug") or ""),
        "model": str(row.get("model") or ""),
        "schedule_slug": str(row.get("schedule_slug") or ""),
        "condition": str(row.get("condition") or ""),
        "status": str(row.get("status") or "actor_missing"),
        "success": parse_bool_or_none(row.get("success")),
        "strict_all_step_success": parse_bool_or_none(row.get("strict_all_step_success")),
        "runtime_error": row.get("runtime_error"),
        "result_source": "reused",
    }


def result_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("canonical_task_id") or ""),
        str(row.get("partition_mode") or ""),
        str(row.get("model_slug") or ""),
        str(row.get("schedule_slug") or ""),
        str(row.get("condition") or ""),
    )


def dataset_name(row: dict[str, Any]) -> str:
    specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    explicit = str(specific.get("benchmark_dataset") or row.get("dataset") or "").lower()
    if explicit in DATASETS:
        return explicit
    family = str(source.get("source_family") or specific.get("source_family") or "").lower()
    if family.startswith("ecfr_casebook"):
        return "casebook"
    if family.startswith("mslr_"):
        return "mslr"
    if family.startswith("sara_"):
        return "sara"
    if "multilexsum" in family or "multi-lexsum" in family:
        return "multilexsum"
    return explicit


def canonical_task_id(row: dict[str, Any], dataset: str) -> str:
    specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    if dataset in {"cuad", "maud"}:
        value = specific.get("pilot_case_id") or specific.get("pilot_source_id")
        if value:
            return str(value).split("__", 1)[0]
    return str(
        specific.get("source_instance_id")
        or source.get("source_instance_id")
        or specific.get("pilot_case_id")
        or row.get("instance_id")
        or ""
    )


def partition_mode(row: dict[str, Any]) -> str:
    generation = row.get("generation_metadata") if isinstance(row.get("generation_metadata"), dict) else {}
    specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    return str(
        generation.get("allocation_mode")
        or (generation.get("partitioning") or {}).get("strategy")
        or specific.get("pilot_partition_mode")
        or specific.get("partition_mode")
        or ""
    )


def answer_type(row: dict[str, Any]) -> str:
    task = row.get("task_description") if isinstance(row.get("task_description"), dict) else {}
    specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    if task.get("answer_type") == "short_answer_judged" or specific.get("answer_type") == "short_answer_judged":
        return "short_answer_judged"
    return "finite_choice"


def parse_bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return value is True or str(value).lower() == "true"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def read_tsv(path: Path) -> list[list[str]]:
    return [
        line.split("\t")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
