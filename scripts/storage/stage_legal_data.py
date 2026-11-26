#!/usr/bin/env python3
"""Stage the verified legal dataset archive as readable files on Scratch."""

from __future__ import annotations

import argparse
import os
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ARCHIVE_ROOT = Path(os.environ.get("TEAMGYM_ARCHIVE_ROOT", "artifacts/archives")).expanduser()
LEGAL_DATA_ROOT = Path(os.environ.get("TEAMGYM_LEGAL_DATA_ROOT", str(Path(os.environ.get("TEAMGYM_DATA_ROOT", "data")) / "legal"))).expanduser()


SOURCE = ARCHIVE_ROOT / 'legal_allocated_v1/zips/legal_allocated_v1_all.zip'
DESTINATION = LEGAL_DATA_ROOT
EXPECTED_MASTER_SHA256 = "1bda44942238793646ffb56d0d705e80d38d86ce94876e7e853ccaab01dc2c5b"
EXPECTED_COUNTS = {
    "casebook": 77,
    "cuad": 538,
    "maud": 557,
    "mslr": 24,
    "multilexsum": 141,
    "sara": 270,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_digest = sha256(source)
    if source_digest != EXPECTED_MASTER_SHA256:
        raise RuntimeError(f"Unexpected master ZIP digest: {source_digest}")
    if destination.exists() and not args.overwrite:
        raise FileExistsError(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        shutil.copy2(source, staging / "legal_allocated_v1_all.zip")
        extract_zip(source, staging)
        for dataset_zip in sorted((staging / "dataset_zips").glob("*.zip")):
            extract_zip(dataset_zip, staging)

        verify_checksum_file(staging / "checksums.sha256", staging)
        verify_checksum_file(
            staging / "dataset_zips/dataset_zips.sha256", staging / "dataset_zips"
        )

        observed_counts = {}
        combined = staging / "collections/all_task_allocations_1607.jsonl"
        combined.parent.mkdir(parents=True)
        with combined.open("w", encoding="utf-8") as output:
            for dataset, expected in EXPECTED_COUNTS.items():
                path = staging / f"data/{dataset}/instances.jsonl"
                count = 0
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            json.loads(line)
                            output.write(line)
                            count += 1
                if count != expected:
                    raise RuntimeError(f"Expected {expected} {dataset} rows, found {count}")
                observed_counts[dataset] = count

        selection_counts = {
            "paper_standard_163": count_jsonl(
                staging / "paper_selections/final_legal_163.jsonl"
            ),
            "paper_allocation_618": count_jsonl(
                staging / "paper_selections/allocation_variants_618.jsonl"
            ),
        }
        if selection_counts != {
            "paper_standard_163": 163,
            "paper_allocation_618": 618,
        }:
            raise RuntimeError(f"Unexpected selection counts: {selection_counts}")

        make_relative_symlink(
            staging / "paper_selections/final_legal_163.jsonl",
            staging / "collections/paper_tasks_163.jsonl",
        )
        make_relative_symlink(
            staging / "paper_selections/allocation_variants_618.jsonl",
            staging / "collections/paper_task_allocations_618.jsonl",
        )
        make_relative_symlink(staging / "data", staging / "collections/by_dataset")

        summary = {
            "schema": "legal_data_stage_v1",
            "source_zip_sha256": source_digest,
            "dataset_counts": observed_counts,
            "total_instance_count": sum(observed_counts.values()),
            "selection_counts": selection_counts,
            "combined_jsonl": "collections/all_task_allocations_1607.jsonl",
        }
        (staging / "stage_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "SCRATCH_README.md").write_text(readme(), encoding="utf-8")

        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
        current = destination.parent / "legal_current"
        current.unlink(missing_ok=True)
        current.symlink_to(destination.name)
        print(json.dumps({**summary, "destination": str(destination)}, indent=2))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def extract_zip(path: Path, destination: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def verify_checksum_file(checksum_file: Path, root: Path) -> None:
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = root / relative.strip()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {path}: {actual} != {expected}")


def make_relative_symlink(source: Path, link: Path) -> None:
    link.unlink(missing_ok=True)
    link.symlink_to(Path("../") / source.relative_to(link.parent.parent))


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readme() -> str:
    return """# Legal Data Working Copy v1

This directory contains both compressed and directly readable forms of the
verified legal dataset.

## Collections

- `collections/all_task_allocations_1607.jsonl`: every validated allocation instance.
- `collections/paper_tasks_163.jsonl`: fixed 163-task paper evaluation set.
- `collections/paper_task_allocations_618.jsonl`: paper allocation-analysis set.
- `collections/by_dataset/<dataset>/instances.jsonl`: dataset-specific records.

## Archives

- `legal_allocated_v1_all.zip`: master data-only archive.
- `dataset_zips/<dataset>.zip`: one ZIP per dataset.

Each task record already contains RIUs, private information, participant
allocation, the task question, answer metadata, and provenance. The trajectory
archives are intentionally separate from this data directory.
"""


if __name__ == "__main__":
    main()
