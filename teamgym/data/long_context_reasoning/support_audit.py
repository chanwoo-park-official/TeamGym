#!/usr/bin/env python
"""Audit whether short-answer seed labels are supported by their PIUs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from teamgym.core.io import read_jsonl, write_json, write_jsonl
from teamgym.llm.openai_client import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)


SYSTEM_PROMPT = """You are a strict dataset-support auditor.

You receive a legal task, a canonical answer, and the reference information units (RIUs) that will be shown to a model.
Your job is to decide whether the canonical answer is fully supported by those RIUs.

Do not use outside legal knowledge. Do not fill in omitted statutory content from memory.
Treat a claim as unsupported if it appears in the canonical answer but is absent from, or only vaguely referenced by, the RIUs.
For example, if an RIU says "Article 5-2" but does not state what Article 5-2 requires, a canonical answer using the unstated
Article 5-2 requirement is unsupported.

Return JSON only.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-spec", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--endpoint", default=DEFAULT_LLM_ENDPOINT, choices=["/responses", "/chat/completions"])
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--credentials-file", default="api_key/api_key.py")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=2200)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--request-retry-attempts", type=int, default=4)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    seeds = read_jsonl(Path(args.seed_spec).expanduser())
    if args.limit is not None:
        seeds = seeds[: max(0, args.limit)]

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            credentials_file=args.credentials_file,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            timeout_sec=args.timeout_sec,
            request_retry_attempts=args.request_retry_attempts,
            request_log_path=str(output_root / "llm_requests.jsonl"),
            log_prompts=False,
            log_responses=False,
        )
    )

    rows = []
    errors = []
    for index, seed in enumerate(seeds, start=1):
        try:
            audit = audit_seed(seed, client=client)
            row = normalize_audit(seed, audit=audit)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            if not args.continue_on_error:
                raise
            errors.append(
                {
                    "instance_id": seed.get("instance_id"),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        write_json(output_root / "progress.json", {"completed": index, "total": len(seeds), "errors": len(errors)})

    kept = [row for row in rows if row["fully_supported"]]
    flagged = [row for row in rows if not row["fully_supported"]]
    write_jsonl(output_root / "support_audits.jsonl", rows)
    write_jsonl(output_root / "fully_supported_audits.jsonl", kept)
    write_jsonl(output_root / "flagged_unsupported_audits.jsonl", flagged)
    write_jsonl(output_root / "support_audit_errors.jsonl", errors)
    summary = summarize(rows, errors, args=args)
    write_json(output_root / "support_audit_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def audit_seed(seed: dict[str, Any], *, client: OpenAIJsonClient) -> dict[str, Any]:
    payload = {
        "audit_task": "canonical_answer_riu_support_check",
        "instance_id": seed.get("instance_id"),
        "question": seed.get("question"),
        "canonical_answer": seed.get("short_answer") or seed.get("canonical_response"),
        "rius": [
            {
                "rui_id": rui.get("rui_id"),
                "rui_type": rui.get("rui_type"),
                "source_span": rui.get("source_span"),
            }
            for rui in seed.get("rius", [])
        ],
        "response_format": {
            "fully_supported": "boolean",
            "support_confidence": "number from 0 to 1",
            "supported_claims": [
                {
                    "claim": "atomic claim from the canonical answer",
                    "supporting_rui_ids": ["rui ids"],
                    "evidence": "brief quote or paraphrase from RIUs",
                }
            ],
            "unsupported_claims": [
                {
                    "claim": "atomic claim from the canonical answer not supported by RIUs",
                    "reason": "why the RIUs do not support it",
                    "missing_information_needed": "what extra RIU would be needed",
                }
            ],
            "recommended_action": "keep | add_rius | rewrite_canonical_answer | drop",
            "reason": "brief overall reason",
        },
    }
    return client.generate_json(
        prompt_id=f"{safe_id(seed.get('instance_id'))}_support_audit",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
    )


def normalize_audit(seed: dict[str, Any], *, audit: dict[str, Any]) -> dict[str, Any]:
    unsupported = audit.get("unsupported_claims") if isinstance(audit.get("unsupported_claims"), list) else []
    supported = audit.get("supported_claims") if isinstance(audit.get("supported_claims"), list) else []
    fully_supported = bool(audit.get("fully_supported")) and not unsupported
    return {
        "schema": "short_answer_seed_support_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "instance_id": seed.get("instance_id"),
        "source_family": seed.get("source_family"),
        "fully_supported": fully_supported,
        "support_confidence": audit.get("support_confidence"),
        "supported_claim_count": len(supported),
        "unsupported_claim_count": len(unsupported),
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "recommended_action": audit.get("recommended_action") or ("keep" if fully_supported else "add_rius"),
        "reason": audit.get("reason"),
        "raw_audit": audit,
    }


def summarize(rows: list[dict[str, Any]], errors: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    actions = Counter(row.get("recommended_action") or "unknown" for row in rows)
    return {
        "schema": "short_answer_seed_support_audit_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seed_spec": args.seed_spec,
            "limit": args.limit,
            "model": args.model,
        },
        "aggregate": {
            "audited_count": len(rows),
            "error_count": len(errors),
            "fully_supported_count": sum(1 for row in rows if row.get("fully_supported")),
            "flagged_count": sum(1 for row in rows if not row.get("fully_supported")),
            "fully_supported_rate": safe_rate(sum(1 for row in rows if row.get("fully_supported")), len(rows)),
            "recommended_actions": dict(actions),
        },
        "flagged_preview": [
            {
                "instance_id": row.get("instance_id"),
                "unsupported_claim_count": row.get("unsupported_claim_count"),
                "recommended_action": row.get("recommended_action"),
                "reason": row.get("reason"),
                "unsupported_claims": row.get("unsupported_claims", [])[:3],
            }
            for row in rows
            if not row.get("fully_supported")
        ][:20],
        "errors": errors[:20],
    }


def safe_id(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "seed"))


def safe_rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


if __name__ == "__main__":
    main()
