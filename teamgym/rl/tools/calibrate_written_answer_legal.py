#!/usr/bin/env python3
"""Parallel GPT calibration for a written-answer TeamGym validation split."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
from typing import Any

from teamgym.core.io import write_json, write_jsonl
from teamgym.scoring.short_answers import judge_one_answer
from teamgym.data.long_context_reasoning.judge import (
    build_report,
    write_html_report,
    write_markdown_report,
)
from teamgym.core.dataset import (
    iter_common_datapoints,
)
from teamgym.llm.openai_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)


class ThreadLocalClient:
    def __init__(self, args: argparse.Namespace, request_log: Path):
        self.args = args
        self.request_log = request_log
        self.local = threading.local()

    def get(self) -> OpenAIJsonClient:
        client = getattr(self.local, "client", None)
        if client is None:
            client = OpenAIJsonClient(
                OpenAIJsonClientConfig(
                    provider=self.args.provider,
                    model=self.args.model,
                    base_url=self.args.base_url,
                    endpoint=self.args.endpoint,
                    credentials_file=str(self.args.credentials_file),
                    api_key_env=self.args.api_key_env,
                    temperature=0,
                    max_output_tokens=self.args.max_output_tokens,
                    timeout_sec=self.args.timeout_sec,
                    request_retry_attempts=self.args.request_retry_attempts,
                    request_log_path=str(self.request_log),
                    log_prompts=False,
                    log_responses=False,
                )
            )
            self.local.client = client
        return client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-common-jsonl", type=Path, required=True)
    parser.add_argument("--negative-answers-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--provider", default="bedrock_openai")
    parser.add_argument(
        "--base-url",
        default="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
    )
    parser.add_argument("--endpoint", default="/responses")
    parser.add_argument("--credentials-file", type=Path, default=Path("api_key/api_key.py"))
    parser.add_argument("--api-key-env", default="AWS_BEARER_TOKEN_BEDROCK")
    parser.add_argument("--max-output-tokens", type=int, default=3000)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--request-retry-attempts", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    args.input_common_jsonl = args.input_common_jsonl.expanduser().resolve()
    args.negative_answers_jsonl = (
        args.negative_answers_jsonl.expanduser().resolve()
    )
    args.output_root = args.output_root.expanduser().resolve()
    args.credentials_file = args.credentials_file.expanduser().resolve()
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_condition(
    *,
    name: str,
    datapoints: list[dict[str, Any]],
    answer_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = args.output_root / name
    output.mkdir(parents=True, exist_ok=True)
    by_id = {str(row["instance_id"]): row for row in datapoints}
    clients = ThreadLocalClient(args, output / "llm_requests.jsonl")
    judgments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def judge_indexed(index_and_answer: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, answer = index_and_answer
        instance_id = str(answer.get("instance_id") or "")
        datapoint = by_id.get(instance_id)
        if datapoint is None:
            raise KeyError(f"missing datapoint {instance_id}")
        return judge_one_answer(
            client=clients.get(),
            datapoint=datapoint,
            answer_row=answer,
            trace_dir=None,
            index=index,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(judge_indexed, item): item
            for item in enumerate(answer_rows, start=1)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index, answer = futures[future]
            try:
                judgment = future.result()
                judgment["_calibration_index"] = index
                judgments.append(judgment)
                passed = bool((judgment.get("judgment") or {}).get("passed"))
                print(
                    json.dumps(
                        {
                            "condition": name,
                            "completed": completed,
                            "total": len(answer_rows),
                            "instance_id": answer.get("instance_id"),
                            "answer_source": answer.get("answer_source"),
                            "passed": passed,
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "calibration_index": index,
                        "instance_id": answer.get("instance_id"),
                        "answer_source": answer.get("answer_source"),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:1000],
                    }
                )
    judgments.sort(key=lambda row: int(row.pop("_calibration_index")))
    errors.sort(key=lambda row: int(row["calibration_index"]))
    write_jsonl(output / "judgments.jsonl", judgments)
    write_jsonl(output / "judge_errors.jsonl", errors)
    report = build_report(
        input_root=args.input_common_jsonl,
        output_root=output,
        judgments=judgments,
        errors=errors,
        answer_count=len(answer_rows),
    )
    write_json(output / "judge_report.json", report)
    write_markdown_report(output / "judge_report.md", report, judgments)
    write_html_report(output / "judge_report.html", report, judgments)
    return report


def main() -> None:
    args = parse_args()
    datapoints = list(iter_common_datapoints(args.input_common_jsonl))
    if not datapoints:
        raise ValueError("No common datapoints found")
    ids = [str(row["instance_id"]) for row in datapoints]
    if len(ids) != len(set(ids)):
        raise ValueError("Validation instance ids must be unique")
    oracle_answers = [
        {
            "instance_id": row["instance_id"],
            "answer_text": (row.get("oracle") or {}).get("canonical_response"),
            "answer_source": "oracle_self_check",
            "communication_structure": "oracle_self_check",
        }
        for row in datapoints
    ]
    negative_answers = read_jsonl(args.negative_answers_jsonl)
    validation_ids = set(ids)
    negative_answers = [
        row
        for row in negative_answers
        if str(row.get("instance_id")) in validation_ids
    ]
    if len(negative_answers) != 2 * len(datapoints):
        raise ValueError(
            f"Expected {2 * len(datapoints)} validation negative controls, "
            f"found {len(negative_answers)}"
        )
    oracle_report = run_condition(
        name="oracle",
        datapoints=datapoints,
        answer_rows=oracle_answers,
        args=args,
    )
    negative_report = run_condition(
        name="negative_controls",
        datapoints=datapoints,
        answer_rows=negative_answers,
        args=args,
    )
    summary = {
        "schema_version": "teamgym_written_answer_gpt55_calibration_v1",
        "input_common_jsonl": str(args.input_common_jsonl),
        "validation_task_count": len(datapoints),
        "model": args.model,
        "workers": args.workers,
        "oracle": oracle_report["aggregate"],
        "negative_controls": negative_report["aggregate"],
        "passed": (
            oracle_report["aggregate"]["passed_count"] == len(datapoints)
            and negative_report["aggregate"]["passed_count"] == 0
            and oracle_report["aggregate"]["error_count"] == 0
            and negative_report["aggregate"]["error_count"] == 0
        ),
    }
    write_json(args.output_root / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise RuntimeError("Calibration acceptance criteria failed")


if __name__ == "__main__":
    main()
