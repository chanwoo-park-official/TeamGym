#!/usr/bin/env python3
"""Run repeated blind Single-Agent All-PIU trials for accepted subproof tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from teamgym.data.legal.sara.subproof_generation import (
    solve_from_rius,
)
from teamgym.llm.openai_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=int)
    parser.add_argument("--worker-count", required=True, type=int)
    parser.add_argument("--model", default="gpt-5.4")
    args = parser.parse_args(argv)

    manifest = json.loads(
        (args.output_root / "metadata" / "qualification_manifest.json").read_text(encoding="utf-8")
    )
    client = make_client(args, args.output_root / "qualification_request_log.jsonl")
    for item in manifest[args.worker :: args.worker_count]:
        qualify_one(client, args.output_root, int(item["case_index"]), int(item["trial"]))


def qualify_one(client: OpenAIJsonClient, root: Path, case_index: int, trial: int) -> None:
    record = json.loads(
        (root / "workers" / f"case_{case_index:03d}" / "record.json").read_text(encoding="utf-8")
    )
    output = root / "qualification" / f"case_{case_index:03d}" / f"trial_{trial}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        solution, audit = solve_from_rius(
            client,
            record["source"],
            record["candidate"],
            record["target_options"],
            trace_dir=output.parent / f"trace_{trial}",
            suffix=f"qualification_{trial}",
        )
        result = {
            "case_index": case_index,
            "case_id": record.get("case_id"),
            "trial": trial,
            "passed": audit["accepted"],
            "solution": solution,
            "audit": audit,
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "case_index": case_index,
            "case_id": record.get("case_id"),
            "trial": trial,
            "passed": False,
            "runtime_error": {"type": type(exc).__name__, "message": str(exc)},
            "runtime_traceback": traceback.format_exc(),
        }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_index": case_index, "trial": trial, "passed": result["passed"]}))


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
            max_output_tokens=2200,
            timeout_sec=600,
            request_retry_attempts=5,
            request_log_path=str(request_log),
            log_prompts=False,
            log_responses=False,
        )
    )


if __name__ == "__main__":
    main()
