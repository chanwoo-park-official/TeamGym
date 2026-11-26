#!/usr/bin/env python3
"""Build the self-contained archive for the six legal allocated datasets."""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any, Iterable
import zipfile

ARCHIVE_ROOT = Path(os.environ.get("TEAMGYM_ARCHIVE_ROOT", "artifacts/archives")).expanduser()
GENERATED_ROOT = Path(os.environ.get("TEAMGYM_GENERATED_ROOT", "generated")).expanduser()
SECONDARY_GENERATED_ROOT = Path(os.environ.get("TEAMGYM_SECONDARY_GENERATED_ROOT", str(GENERATED_ROOT))).expanduser()


DATASETS = ("cuad", "maud", "casebook", "mslr", "multilexsum", "sara")
EXPECTED_COUNTS = {
    "cuad": 538,
    "maud": 557,
    "casebook": 77,
    "mslr": 24,
    "multilexsum": 141,
    "sara": 270,
}
EXPECTED_ALLOCATION_COUNTS = {
    "cuad": {
        "one_each": 321,
        "paired": 6,
        "overlap": 2,
        "redundant_player": 117,
        "common_anchor": 1,
        "bottleneck": 91,
    },
    "maud": {
        "one_each": 207,
        "paired": 89,
        "overlap": 93,
        "redundant_player": 64,
        "common_anchor": 32,
        "bottleneck": 72,
    },
    "casebook": {
        "one_each": 18,
        "paired": 16,
        "overlap": 10,
        "redundant_player": 18,
        "common_anchor": 12,
        "bottleneck": 3,
    },
    "mslr": {
        "one_each": 8,
        "paired": 4,
        "overlap": 2,
        "redundant_player": 6,
        "common_anchor": 3,
        "bottleneck": 1,
    },
    "multilexsum": {
        "one_each": 29,
        "paired": 28,
        "overlap": 28,
        "redundant_player": 28,
        "common_anchor": 27,
        "bottleneck": 1,
    },
    "sara": {
        "one_each": 45,
        "paired": 45,
        "overlap": 45,
        "redundant_player": 45,
        "common_anchor": 45,
        "bottleneck": 45,
    },
}
ALLOCATION_LABELS = {
    "one_each": "One RIU per participant",
    "paired": "Two-RIU concentration",
    "overlap": "Pairwise RIU overlap",
    "redundant_player": "Extra participant without an assigned RIU",
    "common_anchor": "One common RIU",
    "bottleneck": "All-but-one RIU concentration",
}
DEFAULT_CUAD = SECONDARY_GENERATED_ROOT / 'legal_full/cuad/cuad_full_document_partitions.jsonl'
DEFAULT_MAUD = GENERATED_ROOT / 'maud_q01_q02_all_partitions_20260615_182615/maud_q01_q02_all_partition_modes.jsonl'
DEFAULT_STANDARD = GENERATED_ROOT / 'sara_final_eval_0714/inputs/final_legal_163.jsonl'
DEFAULT_VARIANTS = SECONDARY_GENERATED_ROOT / 'legal_partition_schedule_cross_0714/inputs/partition_variants.jsonl'
DEFAULT_OUTPUT = ARCHIVE_ROOT / 'legal_allocated_v1'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuad", type=Path, default=DEFAULT_CUAD)
    parser.add_argument("--maud", type=Path, default=DEFAULT_MAUD)
    parser.add_argument("--paper-standard", type=Path, default=DEFAULT_STANDARD)
    parser.add_argument("--paper-variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_paths = {
        "cuad": args.cuad.expanduser().resolve(),
        "maud": args.maud.expanduser().resolve(),
        "paper_standard": args.paper_standard.expanduser().resolve(),
        "paper_variants": args.paper_variants.expanduser().resolve(),
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    output_root = args.output_root.expanduser().resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)

    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.", dir=output_root.parent) as tmp:
        build_root = Path(tmp) / output_root.name
        build_archive(build_root, source_paths)
        build_root.rename(output_root)

    print(json.dumps({"output_root": str(output_root), "status": "complete"}, indent=2))


def build_archive(root: Path, source_paths: dict[str, Path]) -> None:
    unpacked = root / "unpacked"
    data_root = unpacked / "data"
    selections_root = unpacked / "paper_selections"
    provenance_root = unpacked / "provenance"
    zip_root = root / "zips"
    for path in (data_root, selections_root, provenance_root, zip_root):
        path.mkdir(parents=True, exist_ok=True)

    entries: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in DATASETS}
    entries["cuad"] = load_wrapped_dataset(source_paths["cuad"], "cuad")
    entries["maud"] = load_wrapped_dataset(source_paths["maud"], "maud")
    variant_rows = read_jsonl(source_paths["paper_variants"])
    for line_number, row in enumerate(variant_rows, start=1):
        dataset = benchmark_dataset(row)
        if dataset in {"cuad", "maud"}:
            continue
        if dataset not in DATASETS:
            raise ValueError(f"Unexpected dataset in paper variants: {dataset}")
        entries[dataset].append(
            make_entry(
                dataset=dataset,
                task={key: deepcopy(value) for key, value in row.items() if key != "_cross"},
                source_path=source_paths["paper_variants"],
                source_line=line_number,
                construction_metadata={"cross": deepcopy(row.get("_cross") or {})},
                question_variant=None,
                validation=True,
                cross=deepcopy(row.get("_cross") or {}),
            )
        )

    validate_entry_counts(entries)
    assign_archive_ids(entries)

    standard_rows = read_jsonl(source_paths["paper_standard"])
    standard_matches = match_selection(entries, standard_rows, selection="paper_standard")
    variant_matches = match_selection(entries, variant_rows, selection="paper_allocation")
    if len(standard_matches) != 163:
        raise RuntimeError(f"Expected 163 standard matches, found {len(standard_matches)}")
    if len(variant_matches) != 618:
        raise RuntimeError(f"Expected 618 allocation matches, found {len(variant_matches)}")

    selected_standard = {item["archive_instance_id"] for item in standard_matches}
    selected_variants = {item["archive_instance_id"] for item in variant_matches}
    if len(selected_standard) != 163:
        raise RuntimeError("Standard selection does not map to 163 unique archive instances")
    if len(selected_variants) != 618:
        raise RuntimeError("Allocation selection does not map to 618 unique archive instances")

    all_indices: list[dict[str, Any]] = []
    dataset_summaries = []
    for dataset in DATASETS:
        dataset_dir = data_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        instances = [entry["task"] for entry in entries[dataset]]
        indices = []
        for row_number, entry in enumerate(entries[dataset], start=1):
            index = build_index(entry, row_number)
            archive_id = index["archive_instance_id"]
            index["paper_standard_163"] = archive_id in selected_standard
            index["paper_allocation_618"] = archive_id in selected_variants
            indices.append(index)
        summary = build_dataset_summary(dataset, indices)
        dataset_summaries.append(summary)
        all_indices.extend(indices)

        write_jsonl(dataset_dir / "instances.jsonl", instances)
        write_jsonl(dataset_dir / "index.jsonl", indices)
        write_json(dataset_dir / "summary.json", summary)
        write_id_file(
            dataset_dir / "paper_standard_163.ids",
            [item["archive_instance_id"] for item in indices if item["paper_standard_163"]],
        )
        write_id_file(
            dataset_dir / "paper_allocation_618.ids",
            [item["archive_instance_id"] for item in indices if item["paper_allocation_618"]],
        )
        (dataset_dir / "README.md").write_text(dataset_readme(dataset, summary), encoding="utf-8")
        write_checksums(dataset_dir, dataset_dir / "checksums.sha256")

    shutil.copy2(source_paths["paper_standard"], selections_root / "final_legal_163.jsonl")
    shutil.copy2(source_paths["paper_variants"], selections_root / "allocation_variants_618.jsonl")
    write_jsonl(selections_root / "selection_index.jsonl", standard_matches + variant_matches)
    write_jsonl(provenance_root / "instance_index.jsonl", all_indices)
    write_json(
        provenance_root / "source_files.json",
        {
            key: source_file_record(path)
            for key, path in source_paths.items()
        },
    )
    write_json(
        provenance_root / "dataset_statistics.json",
        {"schema": "legal_allocated_dataset_statistics_v1", "datasets": dataset_summaries},
    )
    write_json(
        provenance_root / "validation_summary.json",
        {
            "schema": "legal_allocated_validation_summary_v1",
            "instance_count": sum(len(items) for items in entries.values()),
            "all_source_records_validated": all(
                entry["validation"] for items in entries.values() for entry in items
            ),
            "expected_counts_verified": True,
            "paper_standard_match_count": len(standard_matches),
            "paper_allocation_match_count": len(variant_matches),
        },
    )

    archive_summary = {
        "schema": "legal_allocated_archive_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(DATASETS),
        "allocated_instance_count": sum(len(items) for items in entries.values()),
        "paper_standard_count": len(standard_matches),
        "paper_allocation_count": len(variant_matches),
        "datasets": dataset_summaries,
    }
    write_json(unpacked / "archive_summary.json", archive_summary)
    (unpacked / "README.md").write_text(root_readme(archive_summary), encoding="utf-8")
    write_checksums(unpacked, unpacked / "checksums.sha256")

    dataset_zip_records = []
    for dataset in DATASETS:
        zip_path = zip_root / f"{dataset}.zip"
        create_zip(data_root / dataset, zip_path, arc_prefix=f"data/{dataset}")
        verify_zip(zip_path)
        dataset_zip_records.append(file_record(zip_path))

    write_json(zip_root / "dataset_zips.json", {"archives": dataset_zip_records})
    write_zip_checksums(zip_root, zip_root / "dataset_zips.sha256", exclude={"legal_allocated_v1_all.zip"})
    combined_zip = zip_root / "legal_allocated_v1_all.zip"
    create_combined_zip(unpacked, zip_root, combined_zip)
    verify_zip(combined_zip)
    write_json(
        root / "archive_files.json",
        {
            "dataset_zips": dataset_zip_records,
            "combined_zip": file_record(combined_zip),
        },
    )


def load_wrapped_dataset(path: Path, dataset: str) -> list[dict[str, Any]]:
    entries = []
    for line_number, wrapper in enumerate(read_jsonl(path), start=1):
        validation = bool((wrapper.get("validation") or {}).get("valid"))
        if not validation:
            continue
        task = wrapper.get("common_datapoint")
        if not isinstance(task, dict):
            raise ValueError(f"{path}:{line_number}: missing common_datapoint")
        construction_metadata = {
            key: deepcopy(value)
            for key, value in wrapper.items()
            if key not in {"common_datapoint", "document_units", "source_document"}
        }
        entries.append(
            make_entry(
                dataset=dataset,
                task=deepcopy(task),
                source_path=path,
                source_line=line_number,
                construction_metadata=construction_metadata,
                question_variant=wrapper.get("q_variant"),
                validation=validation,
                cross={},
            )
        )
    return entries


def make_entry(
    *,
    dataset: str,
    task: dict[str, Any],
    source_path: Path,
    source_line: int,
    construction_metadata: dict[str, Any],
    question_variant: Any,
    validation: bool,
    cross: dict[str, Any],
) -> dict[str, Any]:
    source = task.get("source") if isinstance(task.get("source"), dict) else {}
    specific = task.get("dataset_specific") if isinstance(task.get("dataset_specific"), dict) else {}
    allocation_mode = extract_allocation_mode(task, construction_metadata, cross)
    stored_instance_id = str(task.get("instance_id") or "")
    original_instance_id = str(
        cross.get("original_instance_id")
        or specific.get("cross_original_instance_id")
        or strip_partition_suffix(stored_instance_id)
    )
    source_id = str(
        construction_metadata.get("source_id")
        or specific.get("pilot_source_id")
        or source.get("source_id")
        or ""
    )
    canonical_task_id = str(
        cross.get("canonical_task_id")
        or specific.get("cross_canonical_task_id")
        or source_id
        or original_instance_id
    )
    source_group_id = source_group(dataset, source_id, canonical_task_id, source)
    return {
        "dataset": dataset,
        "task": task,
        "source_path": source_path,
        "source_line": source_line,
        "construction_metadata": construction_metadata,
        "question_variant": str(question_variant) if question_variant is not None else None,
        "validation": validation,
        "allocation_mode": allocation_mode,
        "stored_instance_id": stored_instance_id,
        "original_instance_id": original_instance_id,
        "canonical_task_id": canonical_task_id,
        "source_group_id": source_group_id,
        "source_id": source_id,
    }


def assign_archive_ids(entries: dict[str, list[dict[str, Any]]]) -> None:
    seen: set[str] = set()
    for dataset in DATASETS:
        for entry in entries[dataset]:
            canonical = json.dumps(
                {
                    "dataset": dataset,
                    "task": entry["task"],
                    "question_variant": entry["question_variant"],
                    "allocation_mode": entry["allocation_mode"],
                    "source_line": entry["source_line"],
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            digest = hashlib.sha256(canonical).hexdigest()
            archive_id = f"{dataset}_{digest[:20]}"
            if archive_id in seen:
                raise RuntimeError(f"Duplicate archive ID: {archive_id}")
            seen.add(archive_id)
            entry["archive_instance_id"] = archive_id
            entry["content_sha256"] = hashlib.sha256(
                json.dumps(
                    entry["task"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()


def match_selection(
    entries: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    *,
    selection: str,
) -> list[dict[str, Any]]:
    by_identity: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for dataset_entries in entries.values():
        for entry in dataset_entries:
            by_identity[
                (entry["dataset"], entry["original_instance_id"], entry["allocation_mode"])
            ].append(entry)

    matches = []
    used: set[str] = set()
    for selection_row, row in enumerate(rows, start=1):
        dataset = benchmark_dataset(row)
        specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
        cross = row.get("_cross") if isinstance(row.get("_cross"), dict) else {}
        try:
            mode = extract_allocation_mode(row, {}, cross)
        except ValueError:
            if selection != "paper_standard":
                raise
            mode = "one_each"
        original_instance_id = str(
            cross.get("original_instance_id")
            or specific.get("cross_original_instance_id")
            or strip_partition_suffix(str(row.get("instance_id") or ""))
        )
        candidates = list(by_identity.get((dataset, original_instance_id, mode), []))
        requested_variant = specific.get("pilot_q_variant")
        if requested_variant is not None:
            candidates = [item for item in candidates if item["question_variant"] == str(requested_variant)]
        elif dataset == "maud" and len(candidates) > 1:
            candidates = [item for item in candidates if item["question_variant"] == "maud_q01"]
        if len(candidates) != 1:
            raise RuntimeError(
                f"{selection}:{selection_row}: expected one match for "
                f"{dataset}/{original_instance_id}/{mode}, found {len(candidates)}"
            )
        entry = candidates[0]
        archive_id = entry["archive_instance_id"]
        if archive_id in used:
            raise RuntimeError(f"{selection}: archive instance selected twice: {archive_id}")
        used.add(archive_id)
        matches.append(
            {
                "selection": selection,
                "selection_row": selection_row,
                "archive_instance_id": archive_id,
                "dataset": dataset,
                "canonical_task_id": entry["canonical_task_id"],
                "source_group_id": entry["source_group_id"],
                "allocation_mode": entry["allocation_mode"],
                "stored_instance_id": entry["stored_instance_id"],
                "original_instance_id": entry["original_instance_id"],
            }
        )
    return matches


def build_index(entry: dict[str, Any], row_number: int) -> dict[str, Any]:
    task = entry["task"]
    participants = task.get("participants") if isinstance(task.get("participants"), list) else []
    rius = (
        task.get("reference_information_units")
        if isinstance(task.get("reference_information_units"), list)
        else []
    )
    context_units = (
        task.get("private_context_units")
        if isinstance(task.get("private_context_units"), list)
        else []
    )
    common = task.get("initial_common_information")
    common_values = common if isinstance(common, list) else [common] if common else []
    return {
        "schema": "legal_allocated_instance_index_v1",
        "archive_instance_id": entry["archive_instance_id"],
        "dataset": entry["dataset"],
        "dataset_row": row_number,
        "stored_instance_id": entry["stored_instance_id"],
        "original_instance_id": entry["original_instance_id"],
        "canonical_task_id": entry["canonical_task_id"],
        "source_group_id": entry["source_group_id"],
        "source_id": entry["source_id"],
        "question_variant": entry["question_variant"],
        "allocation_mode": entry["allocation_mode"],
        "allocation_label": ALLOCATION_LABELS.get(entry["allocation_mode"], entry["allocation_mode"]),
        "participant_count": len(participants),
        "riu_count": len(rius),
        "private_context_unit_count": len(context_units),
        "private_context_word_count": sum(text_word_count(item) for item in context_units),
        "riu_word_count": sum(text_word_count(item) for item in rius),
        "initial_common_information_word_count": sum(text_word_count(item) for item in common_values),
        "validation_status": "valid" if entry["validation"] else "invalid",
        "content_sha256": entry["content_sha256"],
        "original_file": str(entry["source_path"]),
        "original_row": entry["source_line"],
    }


def build_dataset_summary(dataset: str, indices: list[dict[str, Any]]) -> dict[str, Any]:
    allocation_counts = Counter(item["allocation_mode"] for item in indices)
    question_counts = Counter(item["question_variant"] or "none" for item in indices)
    return {
        "schema": "legal_allocated_dataset_summary_v1",
        "dataset": dataset,
        "allocated_instance_count": len(indices),
        "canonical_task_count": len({item["canonical_task_id"] for item in indices}),
        "source_group_count": len({item["source_group_id"] for item in indices}),
        "source_id_count": len({item["source_id"] for item in indices}),
        "allocation_mode_counts": dict(sorted(allocation_counts.items())),
        "question_variant_counts": dict(sorted(question_counts.items())),
        "paper_standard_163_count": sum(bool(item["paper_standard_163"]) for item in indices),
        "paper_allocation_618_count": sum(bool(item["paper_allocation_618"]) for item in indices),
        "participant_count": numeric_summary(item["participant_count"] for item in indices),
        "riu_count": numeric_summary(item["riu_count"] for item in indices),
        "private_context_unit_count": numeric_summary(
            item["private_context_unit_count"] for item in indices
        ),
        "private_context_word_count": numeric_summary(
            item["private_context_word_count"] for item in indices
        ),
        "riu_word_count": numeric_summary(item["riu_word_count"] for item in indices),
        "all_records_valid": all(item["validation_status"] == "valid" for item in indices),
    }


def validate_entry_counts(entries: dict[str, list[dict[str, Any]]]) -> None:
    actual_counts = {dataset: len(entries[dataset]) for dataset in DATASETS}
    if actual_counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected dataset counts: {actual_counts}")
    for dataset in DATASETS:
        actual = Counter(entry["allocation_mode"] for entry in entries[dataset])
        expected = Counter(EXPECTED_ALLOCATION_COUNTS[dataset])
        if actual != expected:
            raise RuntimeError(
                f"Unexpected allocation counts for {dataset}: actual={dict(actual)}, expected={dict(expected)}"
            )
    if sum(actual_counts.values()) != 1607:
        raise RuntimeError(f"Expected 1,607 instances, found {sum(actual_counts.values())}")


def benchmark_dataset(row: dict[str, Any]) -> str:
    specific = row.get("dataset_specific") if isinstance(row.get("dataset_specific"), dict) else {}
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    explicit = str(specific.get("benchmark_dataset") or "").lower()
    if explicit:
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
    raw = str(row.get("dataset") or "").lower()
    if raw == "cuad":
        return "cuad"
    if raw == "maud":
        return "maud"
    raise ValueError(f"Cannot identify benchmark dataset for {row.get('instance_id')}")


def extract_allocation_mode(
    task: dict[str, Any], construction: dict[str, Any], cross: dict[str, Any]
) -> str:
    specific = task.get("dataset_specific") if isinstance(task.get("dataset_specific"), dict) else {}
    generation = task.get("generation_metadata") if isinstance(task.get("generation_metadata"), dict) else {}
    partitioning = generation.get("partitioning") if isinstance(generation.get("partitioning"), dict) else {}
    private = (
        generation.get("private_information_allocation")
        if isinstance(generation.get("private_information_allocation"), dict)
        else {}
    )
    mode = (
        cross.get("partition_mode")
        or specific.get("partition_mode")
        or generation.get("allocation_mode")
        or private.get("mode")
        or partitioning.get("strategy")
        or construction.get("partition_strategy")
        or specific.get("partition_strategy")
        or specific.get("pilot_partition_mode")
    )
    if not mode:
        raise ValueError(f"Missing allocation mode for {task.get('instance_id')}")
    return str(mode)


def source_group(
    dataset: str,
    source_id: str,
    canonical_task_id: str,
    source: dict[str, Any],
) -> str:
    if dataset in {"cuad", "maud"} and source_id:
        return source_id.split("__", 1)[0]
    return str(
        source.get("source_instance_id")
        or source.get("case_id")
        or canonical_task_id
        or source_id
    )


def strip_partition_suffix(instance_id: str) -> str:
    return instance_id.split("__partition_", 1)[0]


def text_word_count(value: Any) -> int:
    if isinstance(value, str):
        return len(value.split())
    if not isinstance(value, dict):
        return 0
    for key in ("text", "raw_text", "summary", "content"):
        text = value.get(key)
        if isinstance(text, str):
            return len(text.split())
    return 0


def numeric_summary(values: Iterable[int]) -> dict[str, float | int]:
    items = list(values)
    if not items:
        return {"min": 0, "mean": 0.0, "median": 0.0, "max": 0}
    return {
        "min": min(items),
        "mean": round(statistics.fmean(items), 3),
        "median": round(float(statistics.median(items)), 3),
        "max": max(items),
    }


def dataset_readme(dataset: str, summary: dict[str, Any]) -> str:
    modes = "\n".join(
        f"- `{mode}`: {count:,}"
        for mode, count in summary["allocation_mode_counts"].items()
    )
    return f"""# {dataset.upper()} allocated task instances

This directory contains {summary['allocated_instance_count']:,} validated, directly runnable
task instances derived from the {dataset.upper()} collection. Each task record contains the
question, answer information, reference information units, private context units, participant
assignments, and initially public information.

## Files

- `instances.jsonl`: complete task records in the common execution schema.
- `index.jsonl`: one compact metadata record per task instance.
- `summary.json`: aggregate counts and length statistics.
- `paper_standard_163.ids`: members of the paper's standard 163-instance evaluation set.
- `paper_allocation_618.ids`: members of the paper's 618-instance allocation evaluation set.
- `checksums.sha256`: checksums for this directory.

## Information allocation

{modes}

The machine-readable mode `redundant_player` corresponds to "Extra participant without an
assigned RIU" in the paper. Rows sharing a `canonical_task_id` are variants of the same decision task. Rows sharing
a `source_group_id` originate from the same underlying source and must remain in the same
train/evaluation split.
"""


def root_readme(summary: dict[str, Any]) -> str:
    lines = "\n".join(
        f"- `{item['dataset']}`: {item['allocated_instance_count']:,} allocated instances"
        for item in summary["datasets"]
    )
    return f"""# Legal allocated dataset archive v1

This archive contains {summary['allocated_instance_count']:,} validated allocated task instances
across six legal datasets. It is the complete reusable data pool underlying the paper's fixed
163-instance standard evaluation and 618-instance information-allocation evaluation.

## Datasets

{lines}

## Layout

- `data/<dataset>/instances.jsonl`: directly runnable task records.
- `data/<dataset>/index.jsonl`: allocation, source, validation, and size metadata.
- `paper_selections/final_legal_163.jsonl`: exact standard-evaluation input.
- `paper_selections/allocation_variants_618.jsonl`: exact allocation-evaluation input.
- `paper_selections/selection_index.jsonl`: mappings from paper rows to archive instance IDs.
- `provenance/`: combined index, source hashes, and validation summaries.

The 1,607 rows are allocated task instances, not 1,607 independent legal problems. Split data
by `source_group_id` to prevent variants of the same source from crossing train and evaluation.
The superseded initial 24-task SARA collection is not included.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_id_file(path: Path, ids: Iterable[str]) -> None:
    values = list(ids)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def source_file_record(path: Path) -> dict[str, Any]:
    record = file_record(path)
    record["line_count"] = sum(1 for _ in path.open("rb"))
    return record


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: Path, output: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != output):
        rows.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_zip_checksums(root: Path, output: Path, *, exclude: set[str]) -> None:
    rows = []
    for path in sorted(root.glob("*.zip")):
        if path.name in exclude:
            continue
        rows.append(f"{sha256_file(path)}  {path.name}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def create_zip(source_dir: Path, output: Path, *, arc_prefix: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for path in sorted(item for item in source_dir.rglob("*") if item.is_file()):
            archive.write(path, Path(arc_prefix) / path.relative_to(source_dir))


def create_combined_zip(unpacked: Path, zip_root: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for path in sorted(zip_root.glob("*.zip")):
            if path == output:
                continue
            archive.write(path, Path("dataset_zips") / path.name, compress_type=zipfile.ZIP_STORED)
        for name in ("dataset_zips.json", "dataset_zips.sha256"):
            path = zip_root / name
            archive.write(path, Path("dataset_zips") / name, compress_type=zipfile.ZIP_DEFLATED)
        for path in sorted((unpacked / "paper_selections").rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    Path("paper_selections") / path.relative_to(unpacked / "paper_selections"),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
        for path in sorted((unpacked / "provenance").rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    Path("provenance") / path.relative_to(unpacked / "provenance"),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
        for name in ("README.md", "archive_summary.json", "checksums.sha256"):
            path = unpacked / name
            archive.write(path, name, compress_type=zipfile.ZIP_DEFLATED)


def verify_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member in {path}: {bad}")


if __name__ == "__main__":
    main()
