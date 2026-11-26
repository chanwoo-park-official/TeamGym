#!/usr/bin/env python3
"""Prepare private Hugging Face upload trees for legal data and paper results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ARCHIVE_ROOT = Path(os.environ.get("TEAMGYM_ARCHIVE_ROOT", "artifacts/archives")).expanduser()
LEGAL_DATA_ROOT = Path(os.environ.get("TEAMGYM_LEGAL_DATA_ROOT", str(Path(os.environ.get("TEAMGYM_DATA_ROOT", "data")) / "legal"))).expanduser()


OUTPUT_ROOT = ARCHIVE_ROOT / 'huggingface_upload_v1'
DATA_ROOT = LEGAL_DATA_ROOT
DATA_ARCHIVE_ROOT = ARCHIVE_ROOT / 'legal_allocated_v1'
RESULT_ROOT = ARCHIVE_ROOT / 'legal_paper_release_v1'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    data_output = output_root / "collabgym-legal"
    result_output = output_root / "collabgym-legal-results"
    data_output.mkdir(parents=True)
    result_output.mkdir(parents=True)

    prepare_data_repository(data_output)
    prepare_result_repository(result_output)
    summary = {
        "schema": "legal_huggingface_upload_v1",
        "repositories": {
            "Chanwoo97/collabgym-legal": summarize_tree(data_output),
            "Chanwoo97/collabgym-legal-results": summarize_tree(result_output),
        },
    }
    (output_root / "upload_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def prepare_data_repository(output: Path) -> None:
    mappings = {
        DATA_ROOT / "collections/all_task_allocations_1607.jsonl": output
        / "data/all_task_allocations_1607.jsonl",
        DATA_ROOT / "collections/paper_tasks_163.jsonl": output
        / "data/paper_tasks_163.jsonl",
        DATA_ROOT / "collections/paper_task_allocations_618.jsonl": output
        / "data/paper_task_allocations_618.jsonl",
        DATA_ROOT / "legal_allocated_v1_all.zip": output
        / "archives/legal_allocated_v1_all.zip",
        DATA_ARCHIVE_ROOT / "unpacked/archive_summary.json": output
        / "metadata/archive_summary.json",
        DATA_ARCHIVE_ROOT / "unpacked/provenance/dataset_statistics.json": output
        / "metadata/dataset_statistics.json",
        DATA_ARCHIVE_ROOT / "unpacked/provenance/validation_summary.json": output
        / "metadata/validation_summary.json",
        DATA_ARCHIVE_ROOT / "unpacked/paper_selections/selection_index.jsonl": output
        / "metadata/selection_index.jsonl",
    }
    for source, destination in mappings.items():
        copy_or_link(source, destination)
    (output / "README.md").write_text(data_card(), encoding="utf-8")
    write_checksums(output)


def prepare_result_repository(output: Path) -> None:
    mappings = {
        RESULT_ROOT / "legal_paper_release_v1.zip": output
        / "releases/v1/legal_paper_release_v1.zip",
        RESULT_ROOT / "release_summary.json": output / "releases/v1/release_summary.json",
        RESULT_ROOT / "verification_report.json": output
        / "releases/v1/verification_report.json",
        RESULT_ROOT / "checksums.sha256": output / "releases/v1/release_checksums.sha256",
    }
    for source, destination in mappings.items():
        copy_or_link(source, destination)
    (output / "README.md").write_text(result_card(), encoding="utf-8")
    write_checksums(output)


def copy_or_link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "FILES.sha256":
            rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "FILES.sha256").write_text("".join(rows), encoding="utf-8")


def summarize_tree(root: Path) -> dict:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": {
            path.relative_to(root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files)
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_card() -> str:
    return """---
pretty_name: TeamGym Legal Collaborative Reasoning
language:
- en
- zh
license: other
task_categories:
- question-answering
- text-classification
size_categories:
- 1K<n<10K
configs:
- config_name: all_task_allocations
  data_files: data/all_task_allocations_1607.jsonl
- config_name: paper_tasks
  data_files: data/paper_tasks_163.jsonl
- config_name: paper_task_allocations
  data_files: data/paper_task_allocations_618.jsonl
---

# TeamGym Legal Collaborative Reasoning

This private research repository contains six legal task collections converted
into collaborative decision problems with controlled information allocation.
Each record includes the task, reference information units, participant-private
information, answer metadata, validation status, and source provenance.

## Collections

| File | Description | Rows |
|---|---|---:|
| `data/all_task_allocations_1607.jsonl` | All validated task-allocation instances | 1,607 |
| `data/paper_tasks_163.jsonl` | Fixed paper evaluation tasks | 163 |
| `data/paper_task_allocations_618.jsonl` | Paper allocation-analysis instances | 618 |

The source collections are CUAD, MAUD, Casebook, MSLR, MultiLexSum, and SARA.
`archives/legal_allocated_v1_all.zip` is the checksum-verified data-only archive
and contains one nested ZIP per source collection.

The repository is private pending a source-by-source redistribution and license
review. `FILES.sha256` records the digest of every uploaded file.
"""


def result_card() -> str:
    return """---
pretty_name: TeamGym Legal Experiment Artifacts
license: other
---

# TeamGym Legal Experiment Artifacts

This private repository preserves the complete reproducibility release for the
legal collaborative-reasoning experiments. It contains the exact trajectories,
model outputs, semantic judgments, result tables, regression inputs, run
metadata, code snapshot, and the matching data archive.

## Release

- `releases/v1/legal_paper_release_v1.zip`: complete release (3.10 GB).
- `releases/v1/release_summary.json`: member sizes and SHA-256 digests.
- `releases/v1/verification_report.json`: independent post-build verification.
- `releases/v1/release_checksums.sha256`: source release checksums.

The release indexes 69,577 paper-result cells. Hosted-model reruns preserve the
experimental procedure but are not expected to reproduce generated text
byte-for-byte. The archived outputs are authoritative for exact table and
regression reproduction.

`FILES.sha256` records the digest of every uploaded file.
"""


if __name__ == "__main__":
    main()
