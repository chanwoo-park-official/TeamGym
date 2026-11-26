"""Source-bound execution identity and atomic report commits for FinLongDocQA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .runtime import PROMPT_ROOT, REPO_ROOT, write_json, write_jsonl
from .source import PARSER_VERSION, ReportStore, index_report, qa_for_generation, validate_report_index


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_identity(args, qa_groups, index_by_report, report_ids):
    """Reparse the original ZIP member and compare the entire deterministic index."""
    identity = {"qa_sha256": file_digest(args.qa_path), "archive_sha256": file_digest(args.reports_zip),
                "parser_version": PARSER_VERSION, "reports": {}}
    store = ReportStore(args.reports_zip)
    try:
        for report_id in report_ids:
            qas, stored = qa_groups[report_id], index_by_report[report_id]
            text, report_source = store.load(str(qas[0]["company"]), str(qas[0]["year"]))
            errors = validate_report_index(stored, text)
            expected = index_report(report_id, text)
            if any(stored.get(key) != value for key, value in expected.items()):
                errors.append("index differs from deterministic reparse of the original source")
            source = stored.get("report_source") or {}
            if source.get("sha256") != report_source["sha256"] or source.get("member") != report_source["member"]:
                errors.append("index source member/hash disagrees with source archive")
            pages = {row["page_number"] for row in expected["pages"]}
            for qa in qas:
                public = qa_for_generation(qa)
                if not public["page_numbers"] or not set(public["page_numbers"]) <= pages:
                    errors.append(f"QA {public['qa_id']}: annotated pages absent from source")
            if errors:
                raise ValueError(f"{report_id}: " + "; ".join(dict.fromkeys(errors)))
            identity["reports"][report_id] = {"member": report_source["member"],
                "member_sha256": report_source["sha256"], "index_sha256": digest(expected),
                "qa_sha256": digest([qa_for_generation(qa) for qa in qas])}
    finally:
        store.close()
    return identity


def run_identity(args, sources, report_ids):
    domain = Path(__file__).parent
    paths = [*domain.glob("*.py"), *PROMPT_ROOT.glob("*.txt"),
             REPO_ROOT / "teamgym/llm/construction_client.py", REPO_ROOT / "teamgym/llm/openai_client.py",
             REPO_ROOT / "teamgym/core/information_allocation.py", REPO_ROOT / "teamgym/core/schema.py",
             REPO_ROOT / "teamgym/core/io.py", REPO_ROOT / "teamgym/core/release.py",
             REPO_ROOT / "teamgym/core/dataset.py", REPO_ROOT / "teamgym/tokenization.py",
             REPO_ROOT / "teamgym/interaction/llm_actors.py", REPO_ROOT / "teamgym/interaction/rui_context.py",
             REPO_ROOT / "teamgym/interaction/actor_utils.py", REPO_ROOT / "teamgym/interaction/types.py",
             REPO_ROOT / "teamgym/interaction/public_reading.py",
             REPO_ROOT / "teamgym/interaction/run_single_agent_rui_evaluation.py"]
    settings = {name: getattr(args, name, 40000 if name == "participant_context_token_budget" else None) for name in (
        "provider", "model", "base_url", "endpoint", "api_key_env", "region", "temperature",
        "reasoning_effort", "max_output_tokens", "timeout_sec", "request_retry_attempts",
        "allocation_mode", "sample_seed", "generation_attempts", "audit_workers",
        "packet_repair_rounds", "packet_audit_max_chars", "participant_context_token_budget")}
    value = {"schema": "finlongdocqa_run_identity_v1", "sources": sources,
             "selected_report_ids": report_ids, "settings": settings,
             "code_and_prompts": {str(path.relative_to(REPO_ROOT)): file_digest(path) for path in sorted(paths)}}
    return {**value, "fingerprint": digest(value)}


def establish_run(output, identity):
    path = output / "run_manifest.json"
    if path.exists():
        try:
            previous = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("invalid run manifest; use a fresh output directory") from exc
        if previous != identity:
            raise ValueError("run fingerprint changed; use a fresh output directory")
    else:
        if any((output / name).exists() for name in ("attempts.jsonl", "accepted.jsonl", "common_datapoints.jsonl")):
            raise ValueError("unmanifested previous results; use a fresh output directory")
        write_json(path, identity)


def read_commits(output, fingerprint, report_ids):
    commits, invalid = [], []
    selected = set(report_ids)
    for path in sorted((output / "_commits").glob("*/*.json")):
        try:
            row = json.loads(path.read_text())
            if row["fingerprint"] != fingerprint or row["report_id"] not in selected:
                continue
            if row["payload_sha256"] != digest({"result": row["result"], "common": row["common"]}):
                raise ValueError("payload hash mismatch")
            if row["result"].get("report_id") != row["report_id"]:
                raise ValueError("report mismatch")
            if row["result"].get("status") == "accepted":
                from .schema import validate_common_datapoint
                if not row["common"] or not validate_common_datapoint(row["common"])["valid"]:
                    raise ValueError("accepted commit has no valid common export")
            commits.append(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            invalid.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    return commits, invalid


def commit_report(output, fingerprint, report_id, attempt_number, result, common):
    if result.get("status") == "accepted" and common is None:
        raise ValueError("accepted report requires common export")
    payload = {"result": result, "common": common}
    row = {"schema": "finlongdocqa_report_commit_v1", "fingerprint": fingerprint,
           "report_id": report_id, "attempt_number": attempt_number,
           **payload, "payload_sha256": digest(payload)}
    write_json(output / "_commits" / report_id / f"{attempt_number:05d}.json", row)
    return row


def rebuild_exports(output, commits):
    attempts = [row["result"] for row in commits]
    # A committed accepted report is skipped thereafter. Keep the latest one if
    # an externally repaired run contains multiple historical accepted commits.
    accepted_by_id = {row["report_id"]: row for row in commits if row["result"].get("status") == "accepted"}
    accepted = [row["result"] for row in accepted_by_id.values()]
    common = [row["common"] for row in accepted_by_id.values()]
    write_jsonl(output / "attempts.jsonl", attempts)
    write_jsonl(output / "accepted.jsonl", accepted)
    write_jsonl(output / "common_datapoints.jsonl", common)
    return attempts, accepted, common
