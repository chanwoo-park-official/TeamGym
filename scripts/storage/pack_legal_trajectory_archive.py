#!/usr/bin/env python3
"""Pack one self-contained trajectory archive used by the legal paper."""

from __future__ import annotations

import argparse
import os
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile

ARCHIVE_ROOT = Path(os.environ.get("TEAMGYM_ARCHIVE_ROOT", "artifacts/archives")).expanduser()
GENERATED_ROOT = Path(os.environ.get("TEAMGYM_GENERATED_ROOT", "generated")).expanduser()
SECONDARY_GENERATED_ROOT = Path(os.environ.get("TEAMGYM_SECONDARY_GENERATED_ROOT", str(GENERATED_ROOT))).expanduser()


OUTPUT_ROOT = ARCHIVE_ROOT / 'legal_paper_release_v1'
POOL_GENERATED = GENERATED_ROOT
SCRATCH_GENERATED = SECONDARY_GENERATED_ROOT

ARCHIVES = {
    "model_evidence": {
        "plans": ["model_evidence"],
        "components": [
            (POOL_GENERATED / "0712_legal_model_matrix/metadata", "legacy_0712/metadata"),
            (POOL_GENERATED / "0712_legal_model_matrix/reports", "legacy_0712/reports"),
            (
                POOL_GENERATED / "multilexsum_full_context_rerun_0712/inputs",
                "multilexsum_0712/inputs",
            ),
            (
                POOL_GENERATED / "multilexsum_full_context_rerun_0712/metadata",
                "multilexsum_0712/metadata",
            ),
            (
                POOL_GENERATED / "multilexsum_full_context_rerun_0712/model_matrix/reports",
                "multilexsum_0712/reports",
            ),
            (POOL_GENERATED / "sara_final_eval_0714/inputs", "sara_final_0714/inputs"),
            (POOL_GENERATED / "sara_final_eval_0714/metadata", "sara_final_0714/metadata"),
            (POOL_GENERATED / "sara_final_eval_0714/reports", "sara_final_0714/reports"),
        ],
    },
    "communication_structure": {
        "plans": ["communication_structure"],
        "components": [
            (POOL_GENERATED / "0712_legal_schedule_matrix/metadata", "legacy_0712/metadata"),
            (POOL_GENERATED / "0712_legal_schedule_matrix/reports", "legacy_0712/reports"),
            (
                POOL_GENERATED / "multilexsum_full_context_rerun_0712/inputs",
                "multilexsum_0712/inputs",
            ),
            (
                POOL_GENERATED / "multilexsum_full_context_rerun_0712/metadata",
                "multilexsum_0712/metadata",
            ),
            (
                POOL_GENERATED / "multilexsum_full_context_rerun_0712/schedule_matrix/reports",
                "multilexsum_0712/reports",
            ),
            (POOL_GENERATED / "sara_final_eval_0714/inputs", "sara_final_0714/inputs"),
            (POOL_GENERATED / "sara_final_eval_0714/metadata", "sara_final_0714/metadata"),
            (POOL_GENERATED / "sara_final_eval_0714/reports", "sara_final_0714/reports"),
            (POOL_GENERATED / "legal_process_analysis_0714", "process_analysis_0714"),
        ],
    },
    "allocation_communication": {
        "plans": ["allocation_communication", "allocation_source_cells"],
        "components": [
            (SCRATCH_GENERATED / "legal_partition_schedule_cross_0714/inputs", "cross_0714/inputs"),
            (SCRATCH_GENERATED / "legal_partition_schedule_cross_0714/metadata", "cross_0714/metadata"),
            (SCRATCH_GENERATED / "legal_partition_schedule_cross_0714/reports", "cross_0714/reports"),
            (POOL_GENERATED / "sara_final_eval_0714/inputs", "sara_final_0714/inputs"),
            (POOL_GENERATED / "sara_final_eval_0714/metadata", "sara_final_0714/metadata"),
            (POOL_GENERATED / "sara_final_eval_0714/reports", "sara_final_0714/reports"),
        ],
    },
    "human_constraints": {
        "plans": ["human_constraints"],
        "components": [
            *[
                (
                    SCRATCH_GENERATED / root / section,
                    f"{root}/{section}",
                )
                for root in (
                    "human_constraint_grid_0714",
                    "participant_reading_grid_corrected_0715",
                    "participant_reading_16_extension_0715",
                )
                for section in ("inputs", "metadata", "reports")
            ],
        ],
    },
    "diagnostics": {
        "plans": ["diagnostics"],
        "components": [
            *[
                (
                    SCRATCH_GENERATED / "legal_single_full_context_0720" / section,
                    f"single_full_context_0720/{section}",
                )
                for section in ("inputs", "metadata", "reports")
            ],
            (POOL_GENERATED / "judge_consistency_0721", "judge_consistency_0721/gpt55"),
            (SCRATCH_GENERATED / "judge_consistency_0721", "judge_consistency_0721/gpt54"),
        ],
    },
}

BLOCKED_NAMES = {".env", "credentials", "credentials.json", "api_key", "api_key.txt"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", choices=sorted(ARCHIVES))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compression-level", type=int, default=6)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    plan_root = output_root / "plans"
    zip_root = output_root / "trajectory_zips"
    report_root = output_root / "reports"
    zip_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    destination = zip_root / f"{args.archive}.zip"
    temporary = destination.with_suffix(".zip.partial")
    if destination.exists() and not args.overwrite:
        raise FileExistsError(destination)
    destination.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)

    config = ARCHIVES[args.archive]
    plan_rows = []
    plan_counts = {}
    for name in config["plans"]:
        rows = read_jsonl(plan_root / f"{name}.jsonl")
        plan_rows.extend(rows)
        plan_counts[name] = len(rows)

    counters: Counter[str] = Counter()
    missing_sources = []
    archived_paths: set[str] = set()
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=args.compression_level,
        allowZip64=True,
    ) as archive:
        write_text(
            archive,
            "README.md",
            archive_readme(args.archive, plan_counts),
            archived_paths,
        )
        write_jsonl_to_zip(
            archive,
            "metadata/cell_index.jsonl",
            (portable_row(row) for row in plan_rows),
            archived_paths,
        )
        for name in config["plans"]:
            add_file(
                archive,
                plan_root / f"{name}.jsonl",
                f"metadata/source_plans/{name}.jsonl",
                archived_paths,
                counters,
            )

        for row in plan_rows:
            for source_key, archive_key in (
                ("actor_source_dir", "actor_archive_dir"),
                ("judge_source_dir", "judge_archive_dir"),
            ):
                source_value = row.get(source_key)
                if not source_value:
                    continue
                source = Path(source_value)
                archive_dir = row[archive_key]
                if not source.is_dir():
                    missing_sources.append(
                        {
                            "paper_cell_id": row.get("paper_cell_id"),
                            "source_kind": source_key,
                            "source": str(source),
                            "status": row.get("status"),
                            "expected": expected_missing_source(row, source_key),
                        }
                    )
                    continue
                add_tree(archive, source, archive_dir, archived_paths, counters)

        for source, archive_dir in config["components"]:
            if not source.exists():
                missing_sources.append(
                    {
                        "source_kind": "component",
                        "source": str(source),
                        "status": "missing",
                        "expected": False,
                    }
                )
                continue
            add_tree(
                archive,
                source,
                f"source_components/{archive_dir}",
                archived_paths,
                counters,
            )

        if args.archive == "diagnostics":
            add_judge_consistency_sources(
                archive, plan_root, archived_paths, counters, missing_sources
            )

        write_jsonl_to_zip(
            archive,
            "metadata/missing_sources.jsonl",
            missing_sources,
            archived_paths,
        )
        summary = {
            "schema": "legal_trajectory_zip_summary_v1",
            "archive": args.archive,
            "plan_counts": plan_counts,
            "indexed_cell_count": len(plan_rows),
            "archived_file_count": counters["files"],
            "archived_uncompressed_bytes": counters["bytes"],
            "missing_source_count": len(missing_sources),
            "unexpected_missing_source_count": sum(
                not item.get("expected", False) for item in missing_sources
            ),
        }
        write_text(
            archive,
            "metadata/archive_summary.json",
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            archived_paths,
        )

    temporary.replace(destination)
    verify_zip(destination)
    summary["zip_bytes"] = destination.stat().st_size
    summary["zip_sha256"] = sha256(destination)
    report_path = report_root / f"{args.archive}.json"
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_jsonl(report_root / f"{args.archive}_missing_sources.jsonl", missing_sources)
    print(json.dumps(summary, indent=2))


def expected_missing_source(row: dict[str, Any], source_kind: str) -> bool:
    if source_kind != "judge_source_dir":
        return False
    if row.get("status") != "evaluated":
        return True
    source_result = row.get("source_result")
    return isinstance(source_result, dict) and source_result.get("answer_type") == "finite_choice"


def add_judge_consistency_sources(
    archive: zipfile.ZipFile,
    plan_root: Path,
    archived_paths: set[str],
    counters: Counter[str],
    missing_sources: list[dict[str, Any]],
) -> None:
    source_rows = read_jsonl(plan_root / "judge_consistency_sources.jsonl")
    add_file(
        archive,
        plan_root / "judge_consistency_sources.jsonl",
        "metadata/source_plans/judge_consistency_sources.jsonl",
        archived_paths,
        counters,
    )
    add_file(
        archive,
        plan_root / "judge_consistency_tasks.jsonl",
        "source_components/judge_consistency_0721/exact_sample_tasks.jsonl",
        archived_paths,
        counters,
    )
    portable_sources = []
    for row in source_rows:
        index = int(row["cell_index"])
        base = f"source_components/judge_consistency_0721/source_samples/cell_{index:03d}"
        portable = {key: value for key, value in row.items() if not key.endswith("_file") and not key.endswith("_dir")}
        for key, filename in (
            ("trajectory_source_file", "trajectory.json"),
            ("source_judgment_file", "source_judgments.jsonl"),
        ):
            source = Path(row[key])
            if source.is_file():
                add_file(archive, source, f"{base}/{filename}", archived_paths, counters)
                portable[key.replace("_source_file", "_archive_file")] = f"{base}/{filename}"
            else:
                missing_sources.append(
                    {"cell_index": index, "source_kind": key, "source": str(source)}
                )
        judge_dir = Path(row["gpt54_judge_source_dir"])
        if judge_dir.is_dir():
            add_tree(archive, judge_dir, f"{base}/gpt54_judge", archived_paths, counters)
            portable["gpt54_judge_archive_dir"] = f"{base}/gpt54_judge"
        else:
            missing_sources.append(
                {"cell_index": index, "source_kind": "gpt54_judge", "source": str(judge_dir)}
            )
        portable_sources.append(portable)
    write_jsonl_to_zip(
        archive,
        "metadata/judge_consistency_sample_index.jsonl",
        portable_sources,
        archived_paths,
    )


def portable_row(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.pop("actor_source_dir", None)
    value.pop("judge_source_dir", None)
    result = value.get("source_result")
    if isinstance(result, dict):
        result = dict(result)
        result.pop("trajectory_path", None)
        value["source_result"] = result
    return value


def add_tree(
    archive: zipfile.ZipFile,
    source: Path,
    archive_dir: str,
    archived_paths: set[str],
    counters: Counter[str],
) -> None:
    if source.is_file():
        add_file(archive, source, archive_dir, archived_paths, counters)
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and allowed(path):
            relative = path.relative_to(source).as_posix()
            add_file(
                archive,
                path,
                f"{archive_dir.rstrip('/')}/{relative}",
                archived_paths,
                counters,
            )


def add_file(
    archive: zipfile.ZipFile,
    source: Path,
    archive_path: str,
    archived_paths: set[str],
    counters: Counter[str],
) -> None:
    normalized = archive_path.lstrip("/")
    if normalized in archived_paths:
        return
    if not source.is_file() or not allowed(source):
        return
    archive.write(source, normalized)
    archived_paths.add(normalized)
    counters["files"] += 1
    counters["bytes"] += source.stat().st_size


def allowed(path: Path) -> bool:
    name = path.name.lower()
    if name in BLOCKED_NAMES or name.endswith(".partial"):
        return False
    if name.endswith((".pyc", ".pyo")) or "__pycache__" in path.parts:
        return False
    return True


def write_jsonl_to_zip(
    archive: zipfile.ZipFile,
    archive_path: str,
    rows: Iterable[dict[str, Any]],
    archived_paths: set[str],
) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    write_text(archive, archive_path, text, archived_paths)


def write_text(
    archive: zipfile.ZipFile,
    archive_path: str,
    value: str,
    archived_paths: set[str],
) -> None:
    if archive_path in archived_paths:
        return
    archive.writestr(archive_path, value.encode("utf-8"))
    archived_paths.add(archive_path)


def archive_readme(name: str, plan_counts: dict[str, int]) -> str:
    return f"""# {name.replace('_', ' ').title()} Trajectory Archive

This ZIP preserves the exact transcripts, model outputs, judgments, result tables,
and run metadata used for this experiment family in the paper. `metadata/cell_index.jsonl`
maps each result cell to its dataset `archive_instance_id` and archived actor/judge files.
The matching task records are in `legal_allocated_v1_all.zip` in the outer release.

Plan counts: `{json.dumps(plan_counts, sort_keys=True)}`

Absolute source paths are retained only in `metadata/source_plans/` as provenance.
Portable paths are given in the cell index. Model API reruns may not reproduce generated
text byte-for-byte because hosted model behavior can be nondeterministic.
"""


def verify_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"CRC verification failed for {path}: {bad}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
