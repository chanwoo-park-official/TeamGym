#!/usr/bin/env python3
"""Apply audited PIU information features and participant roles to legal JSONL files.

The input JSONL files are never modified. CUAD and MAUD rows are copied byte for
byte because their construction already produces participant roles. Target rows
are written to new ``*_role_augmented.jsonl`` files under ``--output-root``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from teamgym.core.io import write_json, write_jsonl
from teamgym.core.participant_roles import (
    duplicate_role_names,
    reference_information_units,
    riu_text_sha256,
    role_name_preview,
    validate_feature,
)
from teamgym.data.legal.participant_role_augmentation import (
    EXCLUDED_SOURCE_CORPORA,
    PROMPT_VERSION,
    ExcludedSourceCorpusError,
    canonical_problem_instance_id,
    source_corpus,
)
from teamgym.core.schema import validate_common_datapoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Input legal JSONL; repeat to create multiple derived copies.",
    )
    parser.add_argument(
        "--features-root",
        required=True,
        type=Path,
        help="Root containing shard_*/information_features.jsonl files.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    input_paths = [path.expanduser().resolve() for path in args.input]
    features_root = args.features_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    feature_rows, features = load_features(features_root)
    merged_sidecar_path = output_root / f"information_features_{PROMPT_VERSION}.jsonl"
    assert_new_output(merged_sidecar_path)
    write_jsonl(merged_sidecar_path, feature_rows)

    file_reports = []
    for input_path in input_paths:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        output_path = output_root / f"{input_path.stem}_role_augmented{input_path.suffix}"
        assert_new_output(output_path)
        file_reports.append(
            apply_to_jsonl(
                input_path=input_path,
                output_path=output_path,
                features=features,
            )
        )

    used_problem_keys = {
        (corpus, problem_id)
        for report in file_reports
        for corpus, problem_id in report.pop("used_problem_keys")
    }
    available_problem_keys = {(corpus, problem_id) for corpus, problem_id, _ in features}
    unused_problem_keys = sorted(available_problem_keys - used_problem_keys)

    report = {
        "schema": "legal_participant_role_augmentation_application_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "features_root": str(features_root),
        "merged_feature_sidecar": str(merged_sidecar_path),
        "feature_count": len(feature_rows),
        "feature_problem_instance_count": len(available_problem_keys),
        "feature_counts_by_source_corpus": dict(
            sorted(Counter(row["source_corpus"] for row in feature_rows).items())
        ),
        "unused_feature_problem_instances": [
            {"source_corpus": corpus, "problem_instance_id": problem_id}
            for corpus, problem_id in unused_problem_keys
        ],
        "files": file_reports,
    }
    write_json(output_root / "application_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def load_features(
    features_root: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    paths = sorted(features_root.glob("shard_*/information_features.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No shard_*/information_features.jsonl files found under {features_root}"
        )

    features: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("prompt_version") != PROMPT_VERSION:
                    raise ValueError(
                        f"{path}:{line_number}: expected prompt version {PROMPT_VERSION!r}"
                    )
                audit = row.get("leakage_audit")
                if not isinstance(audit, dict) or audit.get("safe") is not True:
                    raise ValueError(f"{path}:{line_number}: feature did not pass audit")
                corpus = str(row.get("source_corpus") or "")
                problem_id = str(row.get("problem_instance_id") or "")
                rui_id = str(row.get("rui_id") or "")
                feature = str(row.get("information_feature") or "")
                validate_feature(feature, rui_id=rui_id)
                key = (corpus, problem_id, rui_id)
                if not all(key):
                    raise ValueError(f"{path}:{line_number}: incomplete feature key {key}")
                if key in features:
                    raise ValueError(f"Duplicate feature key across shards: {key}")
                features[key] = row

    ordered_rows = [features[key] for key in sorted(features)]
    return ordered_rows, features


def apply_to_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    features: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    temporary_path = output_path.with_name(f".{output_path.name}.partial")
    if temporary_path.exists():
        raise FileExistsError(
            f"Partial output already exists; inspect or remove it before retrying: {temporary_path}"
        )

    counts: Counter[str] = Counter()
    role_collisions: Counter[str] = Counter()
    used_problem_keys: set[tuple[str, str]] = set()
    input_digest = hashlib.sha256()
    excluded_input_digest = hashlib.sha256()
    excluded_output_digest = hashlib.sha256()

    try:
        with input_path.open("rb") as input_handle, temporary_path.open("wb") as output_handle:
            for line_number, raw_line in enumerate(input_handle, start=1):
                input_digest.update(raw_line)
                if not raw_line.strip():
                    output_handle.write(raw_line)
                    continue
                row = json.loads(raw_line)
                try:
                    corpus = source_corpus(row)
                except ExcludedSourceCorpusError:
                    corpus = excluded_corpus(row)
                    excluded_input_digest.update(raw_line)
                    output_handle.write(raw_line)
                    excluded_output_digest.update(raw_line)
                    counts[f"unchanged_{corpus}_rows"] += 1
                    counts["unchanged_excluded_rows"] += 1
                    counts["total_rows"] += 1
                    continue

                problem_id = canonical_problem_instance_id(row)
                units = reference_information_units(row)
                feature_by_rui_id: dict[str, str] = {}
                for unit in units:
                    rui_id = str(unit["rui_id"])
                    key = (corpus, problem_id, rui_id)
                    feature_row = features.get(key)
                    if feature_row is None:
                        raise ValueError(
                            f"{input_path}:{line_number}: missing feature for {key}"
                        )
                    actual_hash = riu_text_sha256(unit)
                    expected_hash = str(feature_row.get("riu_text_sha256") or "")
                    if actual_hash != expected_hash:
                        raise ValueError(
                            f"{input_path}:{line_number}: RIU text hash mismatch for {key}"
                        )
                    feature = str(feature_row["information_feature"])
                    unit["information_feature"] = feature
                    feature_by_rui_id[rui_id] = feature

                previews = role_name_preview(row, features_by_rui_id=feature_by_rui_id)
                preview_by_participant = {
                    str(preview["participant_id"]): preview for preview in previews
                }
                for participant in row.get("participants", []):
                    participant_id = str(participant.get("participant_id") or "")
                    participant["role_name"] = preview_by_participant[participant_id][
                        "proposed_role_name"
                    ]
                row["participant_directory"] = [
                    {
                        "participant_id": participant.get("participant_id"),
                        "role_name": participant.get("role_name"),
                    }
                    for participant in row.get("participants", [])
                ]
                generation = row.setdefault("generation_metadata", {})
                generation["participant_role_augmentation"] = {
                    "schema": "legal_participant_role_augmentation_v1",
                    "prompt_version": PROMPT_VERSION,
                    "source_corpus": corpus,
                    "problem_instance_id": problem_id,
                    "role_source": "privately_assigned_reference_information_units",
                    "common_reference_information_units_excluded": True,
                }

                validation = validate_common_datapoint(row)
                if not validation["valid"]:
                    failed = [
                        check["name"] for check in validation["checks"] if not check["passed"]
                    ]
                    raise ValueError(
                        f"{input_path}:{line_number}: common datapoint validation failed: {failed}"
                    )

                duplicates = duplicate_role_names(previews)
                if duplicates:
                    role_collisions[corpus] += 1
                used_problem_keys.add((corpus, problem_id))
                counts["augmented_rows"] += 1
                counts[f"augmented_{corpus}_rows"] += 1
                counts["augmented_rius"] += len(units)
                counts["augmented_participant_roles"] += len(previews)
                counts["total_rows"] += 1
                output_handle.write(
                    (json.dumps(row, ensure_ascii=False) + os.linesep).encode("utf-8")
                )
        os.replace(temporary_path, output_path)
    except Exception:
        # Retain a partial output for diagnosis rather than silently discarding work.
        raise

    if excluded_input_digest.digest() != excluded_output_digest.digest():
        raise AssertionError("CUAD/MAUD rows changed while writing the derived dataset.")

    output_digest = sha256_file(output_path)
    return {
        "input": str(input_path),
        "input_sha256": input_digest.hexdigest(),
        "output": str(output_path),
        "output_sha256": output_digest,
        "counts": dict(sorted(counts.items())),
        "problem_instance_count": len(used_problem_keys),
        "problem_instance_counts_by_source_corpus": dict(
            sorted(Counter(corpus for corpus, _ in used_problem_keys).items())
        ),
        "rows_with_duplicate_role_names_by_source_corpus": dict(
            sorted(role_collisions.items())
        ),
        "excluded_rows_copied_byte_for_byte": True,
        "used_problem_keys": sorted(used_problem_keys),
    }


def excluded_corpus(row: dict[str, Any]) -> str:
    text = " ".join(
        str(value or "").lower()
        for value in (
            row.get("dataset"),
            (row.get("_cross") or {}).get("dataset"),
            (row.get("dataset_specific") or {}).get("benchmark_dataset"),
        )
    )
    matches = [corpus for corpus in EXCLUDED_SOURCE_CORPORA if corpus in text]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify excluded source corpus for {row.get('instance_id')}")
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


if __name__ == "__main__":
    main()
