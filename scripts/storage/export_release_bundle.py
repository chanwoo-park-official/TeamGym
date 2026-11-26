#!/usr/bin/env python3
"""Package frozen paper RL splits and selected saved trajectories without model calls."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


SPLITS = {
    "legal": {"train": 334, "validation_cuad": 36, "validation_maud": 26},
    "finance": {"train": 538, "validation": 60, "test": 40},
}
METADATA = {
    "legal": ("written_answer_manifest.jsonl", "preparation_report.json", "dataset_audit.json"),
    "finance": ("split_manifest.jsonl", "transfer_audit.json"),
}
LEGAL_DATASETS = ("cuad", "maud", "casebook", "mslr", "sara", "multilexsum")
SUFFIXES = (".jsonl", ".parquet", "_verl.parquet")
VALIDATION_STATUS = {
    "checks": "File checksums, ZIP read-back, JSONL record counts, and example identity links.",
    "full_dataset_validation": "deferred",
    "gpu_training_validation": "deferred",
    "model_api_calls": 0,
}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(value: bytes) -> list[dict]:
    # Physical newlines preserve source strings containing Unicode line separators.
    rows = [json.loads(line) for line in io.BytesIO(value) if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL records must be objects")
    return rows


def checksums(files: dict[str, bytes]) -> bytes:
    return "".join(f"{sha256(value)}  {name}\n" for name, value in sorted(files.items())).encode()


def write_bundle(destination: Path, files: dict[str, bytes], manifest: dict, readme: str) -> dict:
    files = {**files, "README.md": readme.encode("utf-8")}
    details = {name: {"bytes": len(value), "sha256": sha256(value)} for name, value in sorted(files.items())}
    files["manifest.json"] = json_bytes({
        "schema_version": "teamgym_artifact_bundle_v1", **manifest,
        "validation_status": VALIDATION_STATUS, "files": details,
    })
    files["FILES.sha256"] = checksums(files)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, value in sorted(files.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value, compress_type=ZIP_DEFLATED, compresslevel=6)
    with ZipFile(destination) as archive:
        if len(archive.infolist()) != len(files):
            raise ValueError("Archive member count mismatch")
        for name, value in files.items():
            if sha256(archive.read(name)) != sha256(value):
                raise ValueError(f"Archive read-back checksum mismatch: {name}")
    return {"filename": destination.name, "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination), "kind": manifest["kind"],
            "file_count": len(files)}


def export_prepared(kind: str, data: Path, metadata: Path, destination: Path, revision: str | None) -> dict:
    files = {}
    counts = {}
    datasets = {}
    for split, expected in SPLITS[kind].items():
        for suffix in SUFFIXES:
            name = split + suffix
            value = (data / name).read_bytes()
            if not value:
                raise ValueError(f"Empty prepared file: {name}")
            files[f"prepared/{name}"] = value
        rows = jsonl_rows(files[f"prepared/{split}.jsonl"])
        if len(rows) != expected:
            raise ValueError(f"{kind}/{split}: expected {expected} frozen paper records, got {len(rows)}")
        counts[split] = len(rows)
        datasets[split] = dict(sorted(Counter(row["dataset"] for row in rows).items()))
    for name in METADATA[kind]:
        files[f"metadata/{name}"] = (metadata / name).read_bytes()
    optional = {}
    if kind == "finance":
        names = ["test_finlongdocqa_6p" + suffix for suffix in SUFFIXES]
        present = [(data / name).is_file() for name in names]
        if any(present) and not all(present):
            raise ValueError("The optional six-participant finance example requires all three formats")
        if all(present):
            for name in names:
                files[f"prepared/{name}"] = (data / name).read_bytes()
            rows = jsonl_rows(files[f"prepared/{names[0]}"])
            if len(rows) != 1:
                raise ValueError("Expected one optional six-participant finance example")
            optional = {"test_finlongdocqa_6p": 1}
    selection = (
        "Train on CUAD; select on validation_cuad; evaluate transfer on validation_maud. "
        "The combined generic validation view is omitted."
        if kind == "legal" else
        "Train on FinDVer and two/three-participant FinLong; select using the FinDVer "
        "validation tasks as specified by teamgym-paper. FinLong validation is monitoring data. "
        "The four-participant FinLong test is held out."
    )
    manifest = {"kind": f"{kind}_rl", "runtime_code_revision": revision,
                "split_counts": counts, "dataset_counts": datasets,
                "optional_qualitative_counts": optional,
                "checkpoint_selection": selection, "source_files_preserved_byte_for_byte": True}
    readme = (
        f"# TeamGym {kind} paper RL data\n\n"
        "`prepared/` contains the frozen JSONL, parquet, and VERL parquet views. "
        "`metadata/` contains the original preparation records. Original bytes, schemas, "
        "IDs, and historical provenance paths are preserved.\n\n"
        f"{selection}\n\n"
        "`manifest.json` records counts and file checksums. `FILES.sha256` also covers "
        "the manifest and this README. Parquet is copied as an opaque frozen artifact; "
        "the exporter does not decode it or rerun dataset construction or training.\n\n"
        "Use the extracted `prepared/` directory with `teamgym-paper`. "
        "Follow `teamgym/rl/README.md` in the code repository. "
        "Full dataset and GPU validation remain separate steps.\n"
    )
    result = write_bundle(destination, files, manifest, readme)
    result.update(split_counts=counts, optional_qualitative_counts=optional)
    return result


def export_examples(archive_path: Path, benchmark: Path, destination: Path,
                    model: str, condition: str, revision: str | None) -> dict:
    raw_tasks = {}
    for line in io.BytesIO(benchmark.read_bytes()):
        if not line.strip():
            continue
        task = json.loads(line)
        instance = task["instance_id"]
        if instance in raw_tasks:
            raise ValueError(f"Duplicate benchmark instance ID: {instance}")
        raw_tasks[instance] = (task, line.rstrip(b"\r\n") + b"\n")
    files = {}
    selected = []
    inputs = []
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Trajectory archive contains duplicate member names")
        rows = jsonl_rows(archive.read("metadata/cell_index.jsonl"))
        for dataset in LEGAL_DATASETS:
            chosen = False
            for index, row in enumerate(rows):
                if (row.get("dataset"), row.get("model_slug"), row.get("condition"), row.get("status")) != (
                    dataset, model, condition, "evaluated"
                ):
                    continue
                actor = row["actor_archive_dir"].rstrip("/") + "/"
                actor_names = [name for name in names if name.startswith(actor)]
                trajectories = [name for name in actor_names
                                if "/trajectories/" in name and name.endswith(".json")
                                and not name.endswith(".prompt_record.json")]
                if len(trajectories) != 1:
                    continue
                payload = json.loads(archive.read(trajectories[0]))
                instance = payload["evaluation"]["instance_id"]
                if instance not in raw_tasks:
                    continue
                task, raw = raw_tasks[instance]
                written = (task.get("task_description", {}).get("answer_type") == "short_answer_judged"
                           or task.get("decision_set") == ["free_response"])
                judge = (row.get("judge_archive_dir") or "").rstrip("/") + "/"
                judgments = [name for name in names if name.startswith(judge) and name.endswith("/judgments.jsonl")]
                if written and len(judgments) != 1:
                    continue
                if written:
                    judged = jsonl_rows(archive.read(judgments[0]))
                    if not judged or {entry.get("instance_id") for entry in judged} != {instance}:
                        continue
                base = f"examples/{dataset}"
                members = {f"{base}/trajectories/trajectory.json": trajectories[0]}
                complete = True
                for filename in ("run_manifest.json", "eval_results.jsonl", "eval_summary.json"):
                    candidates = [name for name in actor_names if name.endswith("/" + filename)]
                    if len(candidates) != 1:
                        complete = False
                        break
                    members[f"{base}/{filename}"] = candidates[0]
                if not complete:
                    continue
                evaluations = jsonl_rows(archive.read(members[f"{base}/eval_results.jsonl"]))
                if not evaluations or {entry.get("instance_id") for entry in evaluations} != {instance}:
                    continue
                if written:
                    members[f"{base}/judgments.jsonl"] = judgments[0]
                for name, member in members.items():
                    files[name] = archive.read(member)
                inputs.append(raw)
                selected.append({
                    "dataset": dataset, "instance_id": instance,
                    "paper_cell_id": row["paper_cell_id"], "archive_instance_id": row["archive_instance_id"],
                    "model_slug": model, "condition": condition,
                    "source_cell_index": index, "recorded_success": row.get("success"),
                    "recorded_strict_all_step_success": row.get("strict_all_step_success"),
                    "scoring": "saved_written_judgment" if written else "deterministic_finite_choice",
                    "members": members,
                })
                chosen = True
                break
            if not chosen:
                raise ValueError(f"No complete evaluated {model}/{condition} example with matching input/judgment for {dataset}")
    files["inputs.jsonl"] = b"".join(inputs)
    files["selection.jsonl"] = b"".join(json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n" for row in selected)
    manifest = {
        "kind": "selected_legal_trajectories", "runtime_code_revision": revision,
        "example_count": len(selected), "datasets": list(LEGAL_DATASETS),
        "source_archive_sha256": file_sha256(archive_path),
        "source_benchmark_sha256": file_sha256(benchmark),
        "selection_policy": "First complete evaluated cell per dataset in frozen cell-index order; no success filtering.",
        "model_slug": model, "condition": condition,
        "saved_episode_and_judgment_bytes_preserved": True,
    }
    readme = (
        "# Selected TeamGym legal trajectories\n\n"
        "One saved evaluated example from each of CUAD, MAUD, Casebook, MSLR, SARA, "
        "and Multi-LexSum. Selection follows frozen cell-index order and does not "
        "filter for success. This sample illustrates saved results; it is not the "
        "complete benchmark output or an estimate of benchmark performance.\n\n"
        "`inputs.jsonl` contains the matching frozen benchmark records. Each "
        "`examples/<dataset>/` directory preserves the trajectory, actor run manifest, "
        "evaluation results, and summary. Written tasks also include saved judgments; "
        "finite-choice tasks use their deterministic evaluation results. "
        "`selection.jsonl` maps these files to original archive cells and member paths.\n\n"
        "Original content, including historical schema labels and provenance paths, "
        "is preserved. Raw request logs, rendered reports, campaign tools, and code "
        "snapshots are omitted. No actors or judges were rerun.\n\n"
        "To browse an example: `python -m teamgym.visualization.interaction_trajectory_report "
        "--input examples/sara`. Other dataset directories use the same layout.\n"
    )
    result = write_bundle(destination, files, manifest, readme)
    result.update(example_count=len(selected), datasets=list(LEGAL_DATASETS))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legal-data-dir", type=Path, required=True)
    parser.add_argument("--legal-metadata-dir", type=Path, help="Defaults to the legal prepared directory's parent")
    parser.add_argument("--finance-data-dir", type=Path, required=True)
    parser.add_argument("--trajectory-archive", type=Path, required=True, help="The historical model_evidence.zip member")
    parser.add_argument("--benchmark-file", type=Path, required=True, help="Frozen standard-evaluation JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh local release directory")
    parser.add_argument("--model-slug", default="gpt54")
    parser.add_argument("--condition", choices=("single_all_riu", "multi_riu", "multi_full"), default="multi_full")
    parser.add_argument("--code-revision", help="Runtime code revision to record, without changing saved data")
    args = parser.parse_args(argv)
    try:
        destination = args.output_dir.expanduser().resolve()
        if destination.exists():
            raise ValueError(f"Output already exists: {destination}; choose a fresh --output-dir")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "release"
            staging.mkdir()
            legal = args.legal_data_dir.expanduser().resolve()
            metadata = args.legal_metadata_dir.expanduser().resolve() if args.legal_metadata_dir else legal.parent
            finance = args.finance_data_dir.expanduser().resolve()
            bundles = [
                export_prepared("legal", legal, metadata, staging / "teamgym_legal_rl_v1.zip", args.code_revision),
                export_prepared("finance", finance, finance, staging / "teamgym_finance_rl_v1.zip", args.code_revision),
                export_examples(args.trajectory_archive.expanduser(), args.benchmark_file.expanduser(),
                                staging / "teamgym_trajectory_examples_v1.zip", args.model_slug,
                                args.condition, args.code_revision),
            ]
            manifest = {"schema_version": "teamgym_release_bundle_v1", "runtime_code_revision": args.code_revision,
                        "validation_status": VALIDATION_STATUS, "artifacts": bundles}
            (staging / "release_manifest.json").write_bytes(json_bytes(manifest))
            (staging / "README.md").write_text(
                "# TeamGym release artifacts\n\n"
                "- `teamgym_legal_rl_v1.zip`: frozen legal prepared splits and preparation metadata.\n"
                "- `teamgym_finance_rl_v1.zip`: frozen finance splits, preparation metadata, and the optional six-participant example when present.\n"
                "- `teamgym_trajectory_examples_v1.zip`: six saved legal examples with matching inputs and applicable judgments.\n\n"
                "Verify the outer files with `sha256sum -c FILES.sha256`. Extract each archive "
                "into its own directory, then verify its internal `FILES.sha256` there. For example:\n\n"
                "```bash\npython -m zipfile -e teamgym_legal_rl_v1.zip legal_rl\n"
                "cd legal_rl\nsha256sum -c FILES.sha256\n```\n\n"
                "The prepared-data path is `legal_rl/prepared` (or `finance_rl/prepared`). "
                "Use `teamgym-paper` and the repository's training guide for the retained recipes. "
                "These prepared RL tasks are separate from the full standard-evaluation benchmark. "
                "Dataset reconstruction and GPU validation remain separate steps.\n",
                encoding="utf-8",
            )
            (staging / "FILES.sha256").write_bytes(checksums({path.name: path.read_bytes() for path in staging.iterdir()}))
            if destination.exists():
                raise ValueError(f"Output appeared during export: {destination}")
            staging.rename(destination)
        print(json.dumps({"output_dir": str(destination), **manifest}, indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
