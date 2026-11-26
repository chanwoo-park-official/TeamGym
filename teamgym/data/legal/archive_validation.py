"""Offline integrity and reference checks for the frozen legal dataset bundle."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
from typing import Any

from teamgym.core.io import read_jsonl, write_json
from teamgym.core.schema import normalize_release_datapoint, validate_common_datapoint


DATASETS = ("cuad", "maud", "casebook", "mslr", "multilexsum", "sara")
SELECTIONS = {"paper_standard": "paper_standard_163.ids", "paper_allocation": "paper_allocation_618.ids"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def content_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def selection_source_identity(row: dict[str, Any]) -> str:
    """Resolve saved partition aliases using the archive builder's provenance fields."""
    cross = row.get("_cross") or {}
    specific = row.get("dataset_specific") or {}
    return str(cross.get("original_instance_id") or specific.get("cross_original_instance_id")
               or str(row.get("instance_id") or "").split("__partition_", 1)[0])


def validate_task(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Check integrity and return any extra PIU visibility from shared chunks."""
    row = normalize_release_datapoint(row)
    identity = str(row.get("instance_id") or "<missing>")
    schema = validate_common_datapoint(row)
    require(schema["valid"], f"{identity}: failed schema checks: {[c['name'] for c in schema['checks'] if not c['passed']]}")
    participants, contexts, units = row["participants"], row["private_context_units"], row["reference_information_units"]
    participant_ids = {unit["participant_id"] for unit in participants}
    rui_ids = {unit["rui_id"] for unit in units}
    public = {unit["rui_id"] for unit in row.get("initial_common_information", {}).get("common_reference_units", [])}
    require(public <= rui_ids, f"{identity}: unknown public PIU")
    require(set(row["oracle"].get("required_rui_ids", [])) <= rui_ids, f"{identity}: unknown required PIU")
    visibility = []
    for participant in participants:
        require(set(participant.get("owned_rui_ids", [])) <= rui_ids, f"{identity}: participant owns an unknown PIU")
    for context in contexts:
        require(set(context.get("hidden_contains_rui_ids", [])) <= rui_ids, f"{identity}: unknown PIU in source context")
    for unit in units:
        rui_id = unit["rui_id"]
        require(bool(unit.get("source_ref")), f"{identity}: missing source reference for {rui_id}")
        owners = {p["participant_id"] for p in participants if rui_id in p.get("owned_rui_ids", [])}
        # CUAD/MAUD encode ownership on participants and source contexts, without owned_by.
        if "owned_by" in unit:
            require(set(unit["owned_by"]) <= participant_ids and set(unit["owned_by"]) == owners,
                    f"{identity}: inconsistent declared ownership for {rui_id}")
        if rui_id not in public:
            visible = {pid for context in contexts if rui_id in context.get("hidden_contains_rui_ids", [])
                       for pid in context["assigned_to"]}
            require(owners <= visible, f"{identity}: PIU missing from its owner's source contexts: {rui_id}")
            if visible != owners:
                visibility.append({"rui_id": rui_id, "declared_owners": sorted(owners),
                                   "visible_to": sorted(visible), "extra_viewers": sorted(visible - owners)})
    return visibility


def verify_checksums(folder: Path, cache: dict[Path, str]) -> int:
    lines = (folder / "checksums.sha256").read_text().splitlines()
    require(bool(lines), f"Empty checksum manifest: {folder}")
    for line in lines:
        expected, relative = line.split(maxsplit=1)
        path = (folder / relative.strip()).resolve()
        require(path.is_relative_to(folder.resolve()), f"Checksum path leaves bundle directory: {relative}")
        if path not in cache:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            cache[path] = digest.hexdigest()
        require(cache[path] == expected, f"Checksum mismatch: {path}")
    return len(lines)


def validate_archive(root: str | Path, *, strict_visibility: bool = False) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    digests: dict[Path, str] = {}
    verify_checksums(root, digests)
    summary = json.loads((root / "archive_summary.json").read_text())
    all_index: dict[str, dict[str, Any]] = {}
    selections = {name: set() for name in SELECTIONS}
    reports = []
    visibility_diagnostics = []
    for dataset in DATASETS:
        folder = root / "data" / dataset
        verify_checksums(folder, digests)
        metadata = read_jsonl(folder / "index.jsonl")
        dataset_summary = json.loads((folder / "summary.json").read_text())
        stored_ids, scoped_ids, modes, canonical_ids, source_groups = set(), set(), Counter(), set(), set()
        with (folder / "instances.jsonl").open() as handle:
            rows = (json.loads(line) for line in handle if line.strip())
            for number, (row, index) in enumerate(zip_longest(rows, metadata), start=1):
                require(row is not None and index is not None, f"{dataset}: instance/index length mismatch")
                visibility = validate_task(row)
                require(index["dataset"] == dataset and index["dataset_row"] == number, f"{dataset}: invalid index ordering")
                require(index["stored_instance_id"] == row["instance_id"], f"{dataset}: index points to a different task")
                identity = (row["instance_id"], index["question_variant"])
                require(identity not in scoped_ids, f"{dataset}: duplicate instance ID within a question variant")
                require(content_hash(row) == index["content_sha256"], f"{dataset}: record content hash mismatch")
                archive_id = index["archive_instance_id"]
                require(archive_id not in all_index, f"Duplicate archive ID: {archive_id}")
                require(index["validation_status"] == "valid", f"{archive_id}: invalid saved status")
                all_index[archive_id] = index
                for finding in visibility:
                    visibility_diagnostics.append({
                        "dataset": dataset, "archive_instance_id": archive_id, "instance_id": row["instance_id"],
                        "allocation_mode": index["allocation_mode"], "paper_standard": index["paper_standard_163"],
                        "historical_paper_allocation": index["paper_allocation_618"], **finding,
                    })
                stored_ids.add(row["instance_id"])
                scoped_ids.add(identity)
                canonical_ids.add(index["canonical_task_id"])
                source_groups.add(index["source_group_id"])
                modes[index["allocation_mode"]] += 1
                for selection, filename in SELECTIONS.items():
                    flag = "paper_standard_163" if selection == "paper_standard" else "paper_allocation_618"
                    if index[flag]:
                        selections[selection].add(archive_id)
        require(len(scoped_ids) == dataset_summary["allocated_instance_count"], f"{dataset}: summary count mismatch")
        require(len(canonical_ids) == dataset_summary["canonical_task_count"], f"{dataset}: canonical task count mismatch")
        require(len(source_groups) == dataset_summary["source_group_count"], f"{dataset}: source-group count mismatch")
        require(dict(modes) == dataset_summary["allocation_mode_counts"], f"{dataset}: allocation counts mismatch")
        for selection, filename in SELECTIONS.items():
            ids = (folder / filename).read_text().splitlines()
            expected = {key for key in selections[selection] if all_index[key]["dataset"] == dataset}
            require(len(ids) == len(set(ids)) and set(ids) == expected, f"{dataset}: {selection} IDs mismatch")
            count_key = "paper_standard_163_count" if selection == "paper_standard" else "paper_allocation_618_count"
            require(len(ids) == dataset_summary[count_key], f"{dataset}: {selection} count mismatch")
        reports.append({"dataset": dataset, "records": len(scoped_ids), "unique_stored_instance_ids": len(stored_ids),
                        "instance_ids_reused_across_question_variants": len(scoped_ids) - len(stored_ids), "canonical_tasks": len(canonical_ids),
                        "source_groups": len(source_groups), "allocation_modes": dict(modes)})
    global_index = read_jsonl(root / "provenance" / "instance_index.jsonl")
    require(len(global_index) == len(all_index) and {r["archive_instance_id"]: r for r in global_index} == all_index,
            "Global provenance index does not match dataset indexes")
    selected_rows = read_jsonl(root / "paper_selections" / "selection_index.jsonl")
    seen = {name: set() for name in SELECTIONS}
    for row in selected_rows:
        name, archive_id = row["selection"], row["archive_instance_id"]
        require(name in seen and archive_id in all_index, "Unknown selection or archive ID")
        require(archive_id not in seen[name], f"Duplicate {name} archive ID: {archive_id}")
        for key in ["dataset", "canonical_task_id", "source_group_id", "allocation_mode", "stored_instance_id", "original_instance_id"]:
            require(row[key] == all_index[archive_id][key], f"Selection index mismatch for {archive_id}: {key}")
        seen[name].add(archive_id)
    require(seen == selections, "Paper selection index does not match dataset selection flags")
    for selection, filename in [("paper_standard", "final_legal_163.jsonl"), ("paper_allocation", "allocation_variants_618.jsonl")]:
        expected = sorted((row for row in selected_rows if row["selection"] == selection), key=lambda row: row["selection_row"])
        with (root / "paper_selections" / filename).open() as handle:
            actual = (json.loads(line) for line in handle if line.strip())
            for row, index in zip_longest(actual, expected):
                require(row is not None and index is not None, f"{selection}: saved selection length mismatch")
                require(selection_source_identity(row) == index["original_instance_id"],
                        f"{selection}: saved task does not match selection index")
    require(summary["dataset_count"] == len(reports) and summary["allocated_instance_count"] == len(all_index), "Archive totals mismatch")
    require(summary["paper_standard_count"] == len(selections["paper_standard"])
            and summary["paper_allocation_count"] == len(selections["paper_allocation"]), "Archive selection totals mismatch")
    return {
        "schema": "legal_frozen_archive_validation_v1",
        "valid": not strict_visibility or not visibility_diagnostics,
        "integrity_valid": True, "strict_visibility": strict_visibility,
        "archive_manifest_sha256": hashlib.sha256((root / "checksums.sha256").read_bytes()).hexdigest(),
        "record_content_hashes_verified": True,
        "checked_file_hashes": len(digests), "allocated_records": len(all_index),
        "paper_standard_records": len(selections["paper_standard"]),
        "historical_paper_allocation_records": len(selections["paper_allocation"]),
        "datasets": reports,
        "visibility_consistent": not visibility_diagnostics,
        "records_with_extra_piu_visibility": len({r["archive_instance_id"] for r in visibility_diagnostics}),
        "extra_piu_visibility": visibility_diagnostics,
        "scope": "Offline integrity and reference consistency; extra visibility from shared source chunks is reported separately. Model-based quality is not re-evaluated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--strict-visibility", action="store_true", help="Also fail if a source chunk exposes a PIU to non-owners.")
    args = parser.parse_args()
    try:
        report = validate_archive(args.archive_root, strict_visibility=args.strict_visibility)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report = {"schema": "legal_frozen_archive_validation_v1", "valid": False, "error": str(exc)}
    if args.report:
        require(not args.report.expanduser().resolve().is_relative_to(args.archive_root.expanduser().resolve()),
                "Write the report outside the immutable archive root")
        write_json(args.report, report)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
