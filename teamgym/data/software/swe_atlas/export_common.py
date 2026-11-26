"""Export benchmark-ready SWE-Atlas builds to the common interaction schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from teamgym.core.schema import build_common_datapoint
from teamgym.data.software.swe_atlas.source import load_qa_task
from teamgym.data.software.swe_atlas.riu_context import rui_only_datapoint
from .integrity import ConstructionIdentityError, validate_ready_receipt


INSTANCE_PREFIX = "swe_atlas_rubric_grounded_"


class ExportError(ValueError):
    """Raised when a construction output is not ready for evaluation export."""


def export_build(*, build_dir: str | Path, task_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return full-context and RIU-only common datapoints for one audited build."""

    build = Path(build_dir).expanduser().resolve()
    report = _read_json(build / "pipeline_report.json")
    if report.get("status") != "benchmark_ready":
        raise ExportError(f"{build.name}: pipeline status is not benchmark_ready")
    if report.get("all_none_loo_accept") is not True:
        raise ExportError(f"{build.name}: strict RIU audit did not pass")
    if report.get("semantic_leakage_accept") is not True:
        raise ExportError(f"{build.name}: semantic leakage audit did not pass")
    if report.get("salience_accept") is not True:
        raise ExportError(f"{build.name}: context salience audit did not pass")

    task = load_qa_task(task_dir)
    if task.task_id != report.get("task_id"):
        raise ExportError(f"{build.name}: task identity mismatch")
    try:
        validate_ready_receipt(build, task)
    except ConstructionIdentityError as exc:
        raise ExportError(str(exc)) from exc
    partition = _read_json(build / "07_partition" / "final_partition.hidden.json")
    final_rius = _read_json(build / "05_final_rius" / "final_rius.hidden.json").get("rius") or []
    if len(final_rius) < 2:
        raise ExportError(f"{build.name}: collaborative export needs at least two RIUs")

    ownership = {str(key): str(value) for key, value in partition["riu_ownership"].items()}
    participants = _participants(partition, ownership)
    contexts = _private_context_units(partition)
    reference_units = _reference_information_units(final_rius)
    all_riu_ids = [row["rui_id"] for row in reference_units]
    positive_rubrics = [row for row in task.evaluation.rubrics if row.polarity == "positive"]
    negative_rubrics = [row for row in task.evaluation.rubrics if row.polarity == "negative"]
    rubric_to_rius = {
        rubric.rubric_id: sorted(
            row["riu_id"]
            for row in final_rius
            if rubric.rubric_id in set(row.get("supports_rubric_ids") or [])
        )
        for rubric in positive_rubrics
    }
    if any(not ids for ids in rubric_to_rius.values()):
        missing = sorted(key for key, ids in rubric_to_rius.items() if not ids)
        raise ExportError(f"{build.name}: final RIUs miss positive rubrics {missing}")

    full = build_common_datapoint(
        instance_id=INSTANCE_PREFIX + task.task_id.removeprefix("task-"),
        dataset="swe_atlas",
        source={
            "task_id": task.task_id,
            "repository": task.repository.repository,
            "base_commit": task.repository.base_commit,
            "construction_output": str(build),
        },
        scenario_family="rubric_grounded_repository_qa",
        task_description={
            "question": task.public.question,
            "objective": "Answer the repository question using only the supplied private code and runtime evidence.",
            "answer_type": "short_answer_judged",
            "audience": "A developer without prior knowledge of this repository.",
            "response_format_note": (
                "Answer every requested part. Support factual conclusions with short exact excerpts "
                "from the supplied code or runtime material; state when the material does not establish a claim."
            ),
        },
        decision_set=["free_response"],
        decision_option_definitions={"free_response": "Submit the requested codebase analysis."},
        initial_common_information={
            "repository": task.repository.repository,
            "base_commit": task.repository.base_commit,
            "evidence_policy": (
                "Private packets contain immutable source files and Docker-verified runtime outputs "
                "from the stated base commit."
            ),
        },
        participants=participants,
        private_context_units=contexts,
        reference_information_units=reference_units,
        oracle={
            "answer": "free_response",
            "canonical_response": task.evaluation.reference_answer,
            "short_answer": task.evaluation.reference_answer,
            "required_decision_steps": [
                {
                    "step": index,
                    "step_id": rubric.rubric_id,
                    "claim": rubric.title,
                    "uses": rubric_to_rius[rubric.rubric_id],
                    "original_annotations": rubric.annotations,
                }
                for index, rubric in enumerate(positive_rubrics, start=1)
            ],
            "judge_rubric": {
                "require_all_steps_for_pass": True,
                "passing_score": 1.0,
                "source": "original SWE-Atlas tests/rubrics.json",
                "original_rubric_count": len(task.evaluation.rubrics),
            },
            "disqualifying_errors": [row.title for row in negative_rubrics],
            "semantic_judge": {
                "enabled": True,
                "primary_answer_field": "written_answer",
                "fallback_answer_fields": ["rationale"],
            },
            "smallest_sufficient_rui_sets": [
                {"rui_ids": all_riu_ids, "answer": "free_response"}
            ],
        },
        generation_metadata={
            "method": "swe_atlas_strict_source_first_pipeline_v1",
            "rubric_visibility": "hidden_from_solver",
            "strict_riu_audit": "all_none_leave_one_out",
            "semantic_leakage_audit": "participant_foreign_riu_reconstruction",
            "context_salience_audit": "passed",
            "allocation_policy": partition.get("allocation_policy"),
            "participant_context_token_cap": partition.get("participant_context_token_cap"),
            "solver_full_context_policy": (
                "raw_code_and_runtime_material_without_riu_labels_hidden_claims_or_gold_markers"
            ),
            "interaction_context_variant": "full_context_private_packet",
        },
        dataset_specific={
            "answer_type": "short_answer_judged",
            "original_rubrics": [
                {
                    "id": rubric.rubric_id,
                    "title": rubric.title,
                    "annotations": rubric.annotations,
                }
                for rubric in task.evaluation.rubrics
            ],
            "rubric_to_rui_ids": rubric_to_rius,
            "interaction_context_variant": "full_context",
        },
    )
    full["private_experiment_registry"] = _runtime_registry(
        task=task,
        final_rius=final_rius,
        ownership=ownership,
    )
    full["format_validation"] = _validated(full, build.name)

    riu_only = rui_only_datapoint(full)
    riu_only["private_experiment_registry"] = full["private_experiment_registry"]
    riu_only["format_validation"] = _validated(riu_only, build.name)
    _assert_solver_safe(full, riu_only)
    return full, riu_only


def export_package(
    *,
    build_dirs: Iterable[str | Path],
    task_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Export multiple benchmark-ready task builds into one evaluation package."""

    task_base = Path(task_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    full_rows: list[dict[str, Any]] = []
    riu_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for raw_build in build_dirs:
        build = Path(raw_build).expanduser().resolve()
        report = _read_json(build / "pipeline_report.json")
        task_id = str(report.get("task_id") or "")
        full, riu_only = export_build(build_dir=build, task_dir=task_base / task_id)
        full_rows.append(full)
        riu_rows.append(riu_only)
        manifest_rows.append(
            {
                "task_id": task_id,
                "instance_id": full["instance_id"],
                "participant_count": len(full["participants"]),
                "riu_count": len(full["reference_information_units"]),
                "repository": full["source"]["repository"],
                "build_dir": str(build),
            }
        )
    if not full_rows:
        raise ExportError("No benchmark-ready builds were supplied")
    _write_jsonl(output / "full" / "accepted.jsonl", full_rows)
    _write_jsonl(output / "riu_only" / "accepted.jsonl", riu_rows)
    manifest = {
        "schema": "swe_atlas_strict_evaluation_package_v1",
        "task_count": len(manifest_rows),
        "tasks": manifest_rows,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _participants(partition: dict[str, Any], ownership: dict[str, str]) -> list[dict[str, Any]]:
    contexts = {row["context_unit_id"]: row for row in partition["contexts"]}
    result = []
    for row in partition["participants"]:
        participant_id = str(row["participant_id"])
        unit_ids = [str(value) for value in row["private_context_unit_ids"]]
        result.append(
            {
                "participant_id": participant_id,
                "role_name": str(row.get("role_name") or "repository reviewer"),
                "private_context_unit_ids": unit_ids,
                "private_word_count": sum(
                    len(str(contexts[value].get("text") or "").split()) for value in unit_ids
                ),
                "owned_rui_ids": sorted(
                    riu_id for riu_id, owner in ownership.items() if owner == participant_id
                ),
            }
        )
    return result


def _private_context_units(partition: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in partition["contexts"]:
        source_ref: dict[str, Any] = {"context_type": row.get("context_type")}
        if row.get("context_type") == "code" and (row.get("source_ref") or {}).get("path"):
            source_ref["path"] = row["source_ref"]["path"]
        result.append(
            {
                "context_unit_id": str(row["context_unit_id"]),
                "text": str(row.get("text") or ""),
                "assigned_to": [str(value) for value in row.get("assigned_to") or []],
                "word_count": len(str(row.get("text") or "").split()),
                "source_ref": source_ref,
                "hidden_contains_rui_ids": [
                    str(value) for value in row.get("hidden_contains_riu_ids") or []
                ],
                "assignment": {"reason": "audited_disjoint_module_partition", "scores": {}},
            }
        )
    return result


def _reference_information_units(final_rius: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for riu in final_rius:
        blocks = []
        refs = []
        quotes = []
        for atom in riu.get("evidence_atoms") or []:
            quote = str(atom.get("quote") or "").strip()
            if not quote:
                continue
            quotes.append(quote)
            if atom.get("kind") == "source":
                path = str(atom.get("source_path") or "")
                blocks.append(f"[source]\npath: {path}\n{quote}".strip())
                refs.append({"kind": "source", "path": path, "line_range": atom.get("source_line_range")})
            else:
                command = str(atom.get("command") or "")
                blocks.append(f"[runtime]\ncommand: {command}\n{quote}".strip())
                refs.append({"kind": "runtime", "command": command})
        if not blocks:
            raise ExportError(f"{riu.get('riu_id')}: RIU contains no exact evidence text")
        result.append(
            {
                "rui_id": str(riu["riu_id"]),
                "text": "\n\n".join(blocks),
                "evidence_quotes": quotes,
                "supports_rubric_ids": [str(value) for value in riu.get("supports_rubric_ids") or []],
                "refutes_rubric_ids": [str(value) for value in riu.get("refutes_rubric_ids") or []],
                "source_ref": {"evidence_atoms": refs},
            }
        )
    return result


def _runtime_registry(*, task: Any, final_rius: list[dict[str, Any]], ownership: dict[str, str]) -> dict[str, Any]:
    registry: dict[str, Any] = {}
    for riu in final_rius:
        owner = ownership[str(riu["riu_id"])]
        for atom in riu.get("evidence_atoms") or []:
            if atom.get("kind") != "runtime" or not atom.get("command"):
                continue
            artifact_id = str(atom.get("artifact_id") or atom.get("atom_id"))
            key = "experiment_" + artifact_id
            entry = registry.setdefault(
                key,
                {
                    "artifact_id": artifact_id,
                    "kind": "runtime",
                    "owned_by": [],
                    "rui_ids": [],
                    "docker_image": task.runtime.docker_image,
                    "expected_commit": task.repository.base_commit,
                    "command": str(atom["command"]),
                    "expected_exit_code": 0,
                    "allow_internet": task.runtime.allow_internet,
                    "verified": True,
                    "verified_content_by_owner": {},
                },
            )
            if owner not in entry["owned_by"]:
                entry["owned_by"].append(owner)
            if riu["riu_id"] not in entry["rui_ids"]:
                entry["rui_ids"].append(riu["riu_id"])
            current = str(entry["verified_content_by_owner"].get(owner) or "")
            quote = str(atom.get("quote") or "")
            entry["verified_content_by_owner"][owner] = "\n\n".join(
                value for value in (current, quote) if value
            )
    return registry


def _validated(row: dict[str, Any], label: str) -> dict[str, Any]:
    validation = row.get("format_validation") or {}
    if validation.get("valid") is not True:
        failed = [item["name"] for item in validation.get("checks") or [] if not item.get("passed")]
        raise ExportError(f"{label}: common schema validation failed: {failed}")
    return validation


def _assert_solver_safe(full: dict[str, Any], riu_only: dict[str, Any]) -> None:
    for label, row in (("full", full), ("riu_only", riu_only)):
        serialized = json.dumps(row, ensure_ascii=False)
        if '"hidden_claim"' in serialized:
            raise ExportError(f"{label}: hidden claim leaked into solver-visible export")
    for unit in full["private_context_units"]:
        text = str(unit.get("text") or "")
        if text.startswith("RIU ") or "VERBATIM OWNED EVIDENCE" in text:
            raise ExportError("full-context packet contains an RIU-identifying display label")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExportError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", action="append", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = export_package(
        build_dirs=args.build_dir,
        task_root=args.task_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
