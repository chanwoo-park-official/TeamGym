#!/usr/bin/env python3
"""Construct and validate one SARA subproof task under a selected policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from teamgym.data.legal.sara.subproof_generation import (
    GENERATION_POLICIES,
    TARGET_SLICED_POLICY,
    generate_case,
)
from teamgym.llm.openai_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audited-cases", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--policy", choices=sorted(GENERATION_POLICIES), default=TARGET_SLICED_POLICY)
    args = parser.parse_args(argv)

    rows = json.loads(args.audited_cases.read_text(encoding="utf-8"))
    audited = rows[args.index]
    worker_root = args.output_root / "workers" / f"case_{args.index:03d}"
    worker_root.mkdir(parents=True, exist_ok=True)
    client = make_client(args, args.output_root / "request_log.jsonl")
    try:
        result = generate_case(
            client,
            audited["source_row"],
            audited["oracle_execution"],
            trace_dir=worker_root / "traces",
            policy=args.policy,
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "schema": "sara_subproof_generation_record_v1",
            "case_id": audited.get("case_id"),
            "accepted": False,
            "runtime_error": {"type": type(exc).__name__, "message": str(exc)},
            "runtime_traceback": traceback.format_exc(),
            "failed_stage": "runtime",
        }
    result["case_index"] = args.index
    (worker_root / "record.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "index": args.index,
                "case_id": result.get("case_id"),
                "accepted": result.get("accepted"),
                "target": (result.get("selected_target") or {}).get("predicate"),
                "runtime_error": result.get("runtime_error"),
            }
        )
    )


def make_client(args: argparse.Namespace, request_log: Path) -> OpenAIJsonClient:
    return OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider="bedrock_openai",
            model=args.model,
            base_url="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
            endpoint="/responses",
            credentials_file="api_key/api_key.py",
            api_key_env="AWS_BEARER_TOKEN_BEDROCK",
            temperature=0,
            max_output_tokens=6000,
            timeout_sec=600,
            request_retry_attempts=5,
            request_log_path=str(request_log),
            log_prompts=False,
            log_responses=False,
        )
    )


if __name__ == "__main__":
    main()
