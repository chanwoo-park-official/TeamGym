"""Load one source-grounded, boundary-qualified Medical task for redistribution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .certification import require_boundary_cases, validate_export_certification, construction_artifacts


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_allocation_source(dataset, source_run, *, source_dir=None):
    """Reuse successful construction audits, independently of packet acceptance.

    A failed packet distribution may still contain a valid task and PIU set.
    Revalidate its construction artifacts and exact upstream source before use;
    its packet audit or final-answer decisions are never inherited.
    """
    root = Path(source_run).expanduser().resolve()
    manifest = read_json(root / "run_manifest.json")
    if dataset == "evidencebench":
        from .evidencebench.source import load_records, sentence_blocks
        from .evidencebench.validation import validate_task_audit
        from .evidencebench.execution import export_artifacts
        summary = read_json(root / "summary.json")
        if summary.get("status") == "accepted":
            validate_export_certification(summary, export_artifacts(root))
        task = read_json(root / "02_task_rubric.json")
        audit = read_json(root / "03_task_audit.json")
        errors = validate_task_audit(audit, task_rubric=task)
        units_raw = read_json(root / "05_final_rius.hidden.json")
        boundaries = read_json(root / "04_boundaries.json")
        lineage = manifest["source_lineage"]
        source = Path(source_dir) / Path(lineage["source_file"]).name if source_dir else Path(lineage["source_file"])
        if file_hash(source) != lineage["source_file_sha256"]:
            raise ValueError("EvidenceBench source changed since construction")
        record = next((r for r in load_records(source.parent) if r["instance_id"] == lineage["instance_id"]), None)
        if record is None or hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != lineage["record_sha256"]:
            raise ValueError("EvidenceBench source record disagrees with its construction lineage")
        blocks = sentence_blocks(record)
        by_index = {r["sentence_index"]: r for r in blocks}
        contexts = [{"context_id": f"context_{r['sentence_index']:06d}", "text": r["text"],
                     "source_file": "paper", "document_order": 0, "source_order": r["sentence_index"],
                     "source_ref": r["source_ref"]} for r in blocks]
        units = []
        for unit in units_raw:
            indices = []
            for atom in unit["evidence_atoms"]:
                selected = atom["sentence_indices"]
                if atom["text"] != " ".join(str(by_index[i]["text"]).strip() for i in selected):
                    raise ValueError("EvidenceBench PIU differs from exact source sentences")
                indices.extend(selected)
            units.append({"unit_id": unit["riu_id"], "candidate_id": unit["candidate_id"],
                "text": "\n".join(a["text"] for a in unit["evidence_atoms"]),
                "context_ids": [f"context_{i:06d}" for i in sorted(set(indices))],
                "supports_claim_ids": unit["required_claim_ids"],
                "source_ref": {"instance_id": record["instance_id"], "sentence_indices": sorted(set(indices))}})
        instance_id = record["instance_id"]
        names = ["run_manifest.json", "00_source.json", "01_candidates.json", "02_task_rubric.json",
                 "03_task_audit.json", "04_boundaries.json", "05_final_rius.hidden.json"]
    elif dataset == "mlcr":
        from .mlcr.source import load_inspectable, build_case_indexes
        from .mlcr.construction import validate_grounding_audit
        from .mlcr.execution import export_artifacts
        report = read_json(root / "run_report.json")
        if report.get("status") == "accepted":
            validate_export_certification(report, export_artifacts(root))
        task, audit, boundaries = report["task"], report["grounding_audit"], report["boundary_audits"]
        errors = validate_grounding_audit(audit, task=task)
        if (read_json(root / "02_task.json")["task"] != task
                or read_json(root / "03_grounding_audit.json")["response"] != audit
                or read_json(root / "04_boundary_audits.json") != boundaries):
            raise ValueError("MLCR report disagrees with construction audit artifacts")
        lineage = report["source_lineage"]
        paths = {}
        for name in ("cases_summaries.jsonl", "questions.jsonl"):
            entry = lineage["source_files"][name]
            path = Path(source_dir) / name if source_dir else Path(entry["path"])
            if file_hash(path) != entry["sha256"]:
                raise ValueError("MLCR upstream source changed since construction")
            paths[name] = path
        summaries, questions = load_inspectable(paths["questions.jsonl"].parent)
        case = next(c for c in build_case_indexes(summaries, questions) if c["case_uuid"] == report["case_id"])
        atoms = {a["atom_id"]: a for d in case["documents"] for a in d["atoms"]}
        documents = {d["document_id"]: d for d in case["documents"]}
        contexts = [{"context_id": a["atom_id"], "text": a["text"], "source_file": d["summary_file"],
            "document_order": i, "source_order": a["atom_index"],
            "source_ref": {"document_id": d["document_id"], "raw_start": a["raw_start"],
                           "raw_end": a["raw_end"], "sha256": a["sha256"]}}
            for i, d in enumerate(case["documents"]) for a in d["atoms"]]
        units = []
        for u in task["evidence_units"]:
            text = documents[u["source_document_id"]]["text"][u["raw_start"]:u["raw_end"]]
            selected = [atoms[c] for c in u["atom_ids"]]
            if (text != u["text"] or hashlib.sha256(text.encode()).hexdigest() != u["sha256"]
                    or not selected or min(a["raw_start"] for a in selected) != u["raw_start"]
                    or max(a["raw_end"] for a in selected) != u["raw_end"]
                    or any(not c.startswith(u["source_document_id"] + ":") for c in u["atom_ids"])):
                raise ValueError("MLCR PIU no longer matches exact source atoms and offsets")
            units.append({"unit_id": u["unit_id"], "candidate_id": u["candidate_id"], "text": text,
                "context_ids": u["atom_ids"], "supports_claim_ids": u["supports_claim_ids"],
                "source_ref": {k: u[k] for k in ("source_document_id", "raw_start", "raw_end", "sha256")}})
        instance_id = report["instance_id"]
        names = ["run_manifest.json", "run_report.json", "02_task.json", "03_grounding_audit.json", "04_boundary_audits.json"]
    else:
        raise ValueError(f"Unsupported Medical dataset: {dataset}")
    status = summary.get("status") if dataset == "evidencebench" else report.get("status")
    if status != "accepted":
        path = root / "qualified_construction.json"
        if not path.is_file():
            raise ValueError("Uncertified construction: rerun with --construction-only before redistribution")
        qualified = read_json(path)
        validate_export_certification(qualified, construction_artifacts(root, dataset))
        expected_units = units_raw if dataset == "evidencebench" else task["evidence_units"]
        if (qualified.get("status") != "construction_qualified" or qualified.get("task") != task
                or qualified.get("units") != expected_units or qualified.get("source_lineage") != lineage
                or qualified.get("execution_identity") != manifest.get("execution_identity")):
            raise ValueError("Construction changed after its grounding/boundary certification")
        names.append("qualified_construction.json")
    if errors:
        raise ValueError(f"Construction grounding audit failed: {errors}")
    require_boundary_cases(boundaries, [u["candidate_id"] for u in units])
    if {u["candidate_id"] for u in units} != set(task["selected_candidate_ids"]):
        raise ValueError("Frozen PIUs disagree with the task's selected candidates")
    return {"schema": "medical_allocation_source_v1", "dataset": dataset, "instance_id": instance_id,
        "task": task, "units": units, "contexts": contexts, "source_lineage": lineage,
        "construction_audits": {"grounding": audit, "boundaries": boundaries},
        "producer_identity": manifest.get("execution_identity"),
        "source_artifacts": {name: {"path": str(root / name), "sha256": file_hash(root / name)} for name in names}}
