"""Permute decision labels in generated legal partition datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any

from teamgym.core.io import write_json, write_jsonl

from .decision_permutation import (
    apply_permutation_to_public_common,
    permute_common_datapoint_decisions,
)


MAP_SCHEMA_VERSION = "collabgym_fc_decision_option_permutation_map_v1"

DATASET_FILES = {
    "cuad": {
        "jsonl": Path("cuad/cuad_full_document_partitions.jsonl"),
        "pretty": Path("cuad/cuad_full_document_partitions.pretty.json"),
    },
    "maud": {
        "jsonl": Path("maud/maud_full_document_partitions.jsonl"),
        "pretty": Path("maud/maud_full_document_partitions.pretty.json"),
    },
}


def permute_dataset_root(
    *,
    input_root: Path,
    output_root: Path,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    all_maps: list[dict[str, Any]] = []
    report = {
        "schema_version": MAP_SCHEMA_VERSION,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": seed,
        "datasets": {},
    }

    for dataset, rels in DATASET_FILES.items():
        source_path = _source_path(input_root, rels)
        if source_path is None:
            continue
        rows = read_json_rows(source_path)
        target_answers = balanced_target_answers(rows, seed=f"{seed}:{dataset}")
        mixed_rows: list[dict[str, Any]] = []
        dataset_maps: list[dict[str, Any]] = []
        before = Counter()
        after = Counter()
        option_counts = Counter()
        for row_index, row in enumerate(rows):
            common = row.get("common_datapoint") if isinstance(row.get("common_datapoint"), dict) else None
            if common is None:
                mixed_rows.append(row)
                continue
            existing = ((common.get("generation_metadata") or {}).get("decision_option_permutation") or {})
            if existing and not force:
                raise ValueError(
                    f"{source_path} already has decision_option_permutation metadata. "
                    "Use --force only if you intentionally want to re-permute."
                )
            instance_id = str(common.get("instance_id") or row.get("instance_id") or row_index)
            old_answer = str((common.get("oracle") or {}).get("answer") or "")
            target_answer = target_answers.get(row_index)
            seed_material = f"{seed}:{dataset}:{instance_id}:{row_index}:decision_options"
            mixed_common, metadata = permute_common_datapoint_decisions(
                common,
                seed_material=seed_material,
                target_answer=target_answer,
            )
            mixed_row = dict(row)
            mixed_row["common_datapoint"] = mixed_common
            if isinstance(row.get("public_common"), dict):
                mixed_row["public_common"] = apply_permutation_to_public_common(
                    row["public_common"],
                    metadata,
                    source_common_datapoint=mixed_common,
                )
            mixed_rows.append(mixed_row)

            new_answer = str((mixed_common.get("oracle") or {}).get("answer") or "")
            before[old_answer] += 1
            after[new_answer] += 1
            option_counts[len(mixed_common.get("decision_set") or [])] += 1
            map_entry = {
                "schema_version": MAP_SCHEMA_VERSION,
                "dataset": dataset,
                "instance_id": instance_id,
                "row_index": row_index,
                "source_path": str(source_path.relative_to(input_root)),
                "old_answer": old_answer,
                "new_answer": new_answer,
                "old_to_new": metadata.get("old_to_new", {}),
                "new_to_old": metadata.get("new_to_old", {}),
                "decision_set": list(mixed_common.get("decision_set") or []),
                "option_count": len(mixed_common.get("decision_set") or []),
                "seed_material": seed_material,
            }
            dataset_maps.append(map_entry)
            all_maps.append(map_entry)

        write_partition_files(output_root, rels, mixed_rows)
        report["datasets"][dataset] = {
            "row_count": len(dataset_maps),
            "source_path": str(source_path.relative_to(input_root)),
            "jsonl_path": str(rels["jsonl"]),
            "pretty_path": str(rels["pretty"]),
            "option_counts": dict(sorted(option_counts.items())),
            "old_oracle_answer_distribution": dict(sorted(before.items())),
            "new_oracle_answer_distribution": dict(sorted(after.items())),
        }

    write_jsonl(output_root / "decision_option_permutation_map.jsonl", all_maps)
    write_json(output_root / "decision_option_permutation_map.pretty.json", all_maps)
    write_json(output_root / "decision_option_permutation_report.json", report)
    return report


def balanced_target_answers(rows: list[dict[str, Any]], *, seed: str) -> dict[int, str]:
    expected: dict[str, float] = defaultdict(float)
    eligible_rows: list[tuple[int, list[str]]] = []
    for row_index, row in enumerate(rows):
        common = row.get("common_datapoint") if isinstance(row.get("common_datapoint"), dict) else None
        if common is None:
            continue
        decision_set = [str(item) for item in common.get("decision_set") or [] if str(item)]
        if not decision_set:
            continue
        eligible_rows.append((row_index, decision_set))
        share = 1.0 / float(len(decision_set))
        for label in decision_set:
            expected[label] += share

    rng = random.Random(seed)
    rng.shuffle(eligible_rows)
    assigned: Counter[str] = Counter()
    targets: dict[int, str] = {}
    for row_index, decision_set in eligible_rows:
        candidates = list(decision_set)
        rng.shuffle(candidates)
        target = min(
            candidates,
            key=lambda label: (
                assigned[label] / expected[label] if expected[label] else float("inf"),
                assigned[label],
                label,
            ),
        )
        assigned[target] += 1
        targets[row_index] = target
    return targets


def _source_path(root: Path, rels: dict[str, Path]) -> Path | None:
    pretty = root / rels["pretty"]
    if pretty.is_file():
        return pretty
    jsonl = root / rels["jsonl"]
    if jsonl.is_file():
        return jsonl
    return None


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        return [row for row in payload if isinstance(row, dict)]
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        row, index = decoder.raw_decode(text, index)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_partition_files(root: Path, rels: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    write_jsonl(root / rels["jsonl"], rows)
    write_json(root / rels["pretty"], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Generated legal dataset root to read.")
    parser.add_argument("--output-root", default=None, help="Output root. Defaults to input root for in-place update.")
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--force", action="store_true", help="Allow re-permuting rows that already have permutation metadata.")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root) if args.output_root else input_root
    report = permute_dataset_root(
        input_root=input_root,
        output_root=output_root,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
