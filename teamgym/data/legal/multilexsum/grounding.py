#!/usr/bin/env python
"""Audit derived PIU propositions against their original Multi-LexSum source windows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from teamgym.core.io import read_jsonl, write_json, write_jsonl
from teamgym.llm.openai_client import OpenAIJsonClient, OpenAIJsonClientConfig


SYSTEM = """You are a strict legal dataset grounding auditor. For each derived proposition, decide whether its
associated original source window explicitly entails it. Do not use outside knowledge and do not treat a party's
allegation as a judicial finding. Also decide whether the canonical answer is fully supported by the collection of
supported propositions. Return JSON only."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    rows = read_jsonl(args.input_root)
    selected = [row for i, row in enumerate(rows) if i % args.num_shards == args.shard_index]
    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = OpenAIJsonClient(OpenAIJsonClientConfig(
        provider="bedrock_openai",
        model="gpt-5.5",
        base_url="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
        endpoint="/responses",
        api_key_env="AWS_BEARER_TOKEN_BEDROCK",
        credentials_file="api_key/api_key.py",
        temperature=0.0,
        max_output_tokens=4000,
        timeout_sec=300,
        request_retry_attempts=4,
        request_log_path=str(output / "requests.jsonl"),
        log_prompts=False,
        log_responses=False,
    ))
    audits = []
    for row in selected:
        windows = {}
        for unit in row.get("private_context_units", []):
            hidden = unit.get("hidden_contains_rui_ids") or []
            for rui_id in hidden:
                windows[str(rui_id)] = str(unit.get("text") or "")
        payload = {
            "question": row["task_description"]["question"],
            "canonical_answer": row["oracle"]["canonical_response"],
            "derived_units": [
                {"rui_id": rui["rui_id"], "proposition": rui["text"], "original_source_window": windows.get(rui["rui_id"], "")}
                for rui in row.get("reference_information_units", [])
            ],
            "response_format": {
                "atom_audits": [{"rui_id": "id", "supported": True, "reason": "brief source comparison"}],
                "canonical_fully_supported": True,
                "canonical_support_reason": "brief reason",
            },
        }
        response = client.generate_json(
            prompt_id=f"{row['instance_id']}.original_grounding_audit",
            system_prompt=SYSTEM,
            user_prompt=json.dumps(payload, indent=2, ensure_ascii=False),
        )
        expected = {rui["rui_id"] for rui in row.get("reference_information_units", [])}
        atom_rows = response.get("atom_audits") if isinstance(response.get("atom_audits"), list) else []
        by_id = {str(item.get("rui_id")): item for item in atom_rows if isinstance(item, dict)}
        unsupported = sorted(rui_id for rui_id in expected if (by_id.get(rui_id) or {}).get("supported") is not True)
        canonical_supported = response.get("canonical_fully_supported") is True
        audits.append({
            "instance_id": row["instance_id"],
            "fully_grounded": not unsupported and canonical_supported,
            "unsupported_rui_ids": unsupported,
            "atom_audits": atom_rows,
            "canonical_fully_supported": canonical_supported,
            "canonical_support_reason": response.get("canonical_support_reason"),
        })
    write_jsonl(output / "grounding_audits.jsonl", audits)
    print(json.dumps({"audited": len(audits), "fully_grounded": sum(x["fully_grounded"] for x in audits)}, indent=2))


def select_grounded(
    tasks: list[dict], audits: list[dict], *, limit: int | None = None,
) -> tuple[list[dict], list[str]]:
    """Select audited tasks in input order, requiring exact Boolean PIU coverage."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    task_ids = [row["instance_id"] for row in tasks]
    audit_ids = [row["instance_id"] for row in audits]
    if not task_ids or len(set(task_ids)) != len(task_ids) or len(set(audit_ids)) != len(audit_ids):
        raise ValueError("Tasks must be nonempty; task and audit IDs must be unique")
    if set(task_ids) != set(audit_ids):
        raise ValueError("Every task must have exactly one grounding audit, without extra IDs")
    by_id = {row["instance_id"]: row for row in audits}
    accepted, rejected = [], []
    for task in tasks:
        audit = by_id[task["instance_id"]]
        expected = {unit["rui_id"] for unit in task["reference_information_units"]}
        atoms = audit.get("atom_audits") or []
        ids = [item.get("rui_id") for item in atoms if isinstance(item, dict)]
        passed = (
            bool(expected) and len(ids) == len(atoms) == len(expected) and set(ids) == expected
            and all(item.get("supported") is True for item in atoms)
            and audit.get("fully_grounded") is True and audit.get("canonical_fully_supported") is True
            and audit.get("unsupported_rui_ids") == []
        )
        if passed:
            accepted.append(task)
        else:
            rejected.append(task["instance_id"])
    return accepted[:limit], rejected


def select_main() -> None:
    parser = argparse.ArgumentParser(description="Retain tasks with complete successful source-grounding audits.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--audits", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    tasks = read_jsonl(args.input_root)
    accepted, rejected = select_grounded(tasks, read_jsonl(args.audits), limit=args.limit)
    if not accepted:
        raise ValueError("No fully source-grounded tasks were retained")
    accepted = copy.deepcopy(accepted)
    audit_hash = hashlib.sha256(args.audits.read_bytes()).hexdigest()
    for row in accepted:
        row.setdefault("dataset_specific", {})["original_source_grounded"] = True
        for unit in row["reference_information_units"]:
            unit.setdefault("metadata", {})["original_source_grounded"] = True
        row.setdefault("generation_metadata", {}).update(
            requires_source_grounding_audit=False,
            source_grounding_audits_sha256=audit_hash,
        )
    write_jsonl(args.output_root / "common_datapoints.jsonl", accepted)
    write_json(args.output_root / "selection_report.json", {
        "schema": "multilexsum_grounded_selection_v1", "input_count": len(tasks),
        "retained_count": len(accepted), "grounding_rejected_ids": rejected,
        "grounding_audits_sha256": audit_hash,
        "grounded_but_not_selected_count": len(tasks) - len(rejected) - len(accepted),
        "policy": "Input order after complete Boolean grounding checks; fresh leakage and solvability checks still required",
    })


if __name__ == "__main__":
    main()
