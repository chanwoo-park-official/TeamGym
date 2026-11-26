#!/usr/bin/env python3
"""Evaluate base-model finance participant handoff through an OpenAI server."""

from __future__ import annotations

import argparse
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine

from teamgym.core.dataset import iter_common_datapoints

from ..data import canonical_task_id
from ..rollout.workflow import TeamGymWorkflow


DEFAULT_INPUT = Path(os.environ.get("TEAMGYM_FINANCE_NORMALIZED_DATA", "generated/finance_participant_handoff_rl/normalized/finance_riu_only.jsonl")).expanduser()


class OpenAICompatibleRolloutEngine(RolloutEngine):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float,
        timeout_sec: float,
    ) -> None:
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_sec,
        )
        self.model = model
        self.temperature = float(temperature)

    async def get_model_response(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ModelOutput:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max(1, int(kwargs.get("max_tokens") or 1024)),
            temperature=self.temperature,
            top_p=0.95,
        )
        choice = response.choices[0]
        content = str(choice.message.content or "")
        reasoning = str(
            getattr(choice.message, "reasoning_content", None) or ""
        )
        usage = response.usage
        return ModelOutput(
            text=content,
            content=content,
            reasoning=reasoning,
            prompt_ids=[],
            completion_ids=[],
            prompt_length=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_length=int(
                getattr(usage, "completion_tokens", 0) or 0
            ),
            finish_reason=str(choice.finish_reason or ""),
        )


def select_balanced_examples(
    rows: list[dict[str, Any]],
    *,
    datasets: tuple[str, ...],
    per_dataset: int,
    minimum_participants: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset in datasets:
        eligible = [
            row
            for row in rows
            if str(row.get("dataset") or "").casefold() == dataset
            and len(row.get("participant_directory", []))
            >= minimum_participants
        ]
        eligible.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{dataset}:{row.get('instance_id')}".encode("utf-8")
            ).hexdigest()
        )
        if len(eligible) < per_dataset:
            raise ValueError(
                f"{dataset} has {len(eligible)} eligible examples; "
                f"requested {per_dataset}"
            )
        selected.extend(eligible[:per_dataset])
    return selected


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = list(iter_common_datapoints(args.input))
    datasets = tuple(
        value.strip().casefold()
        for value in args.datasets.split(",")
        if value.strip()
    )
    selected = select_balanced_examples(
        rows,
        datasets=datasets,
        per_dataset=args.per_dataset,
        minimum_participants=args.minimum_participants,
        seed=args.seed,
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    selection_path = output / "selected_examples.jsonl"
    selection_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in selected
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "finance_participant_handoff_eval_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "rollout_profile": "finance_riu_participant_handoff_15",
        "datasets": list(datasets),
        "per_dataset": args.per_dataset,
        "minimum_participants": args.minimum_participants,
        "seed": args.seed,
        "selected_instance_ids": [row.get("instance_id") for row in selected],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    engine = OpenAICompatibleRolloutEngine(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        timeout_sec=args.timeout_sec,
    )
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    results: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=max(4, args.concurrency * 2))

    async def run_one(index: int, datapoint: dict[str, Any]) -> None:
        async with semaphore:
            instance_id = str(datapoint.get("instance_id") or f"row_{index}")
            task = {
                "instance_id": instance_id,
                "canonical_task_id": canonical_task_id(datapoint),
                "dataset": datapoint.get("dataset"),
                "teamgym_datapoint_json": json.dumps(
                    datapoint, ensure_ascii=False
                ),
            }
            workflow = TeamGymWorkflow(
                rollout_engine=engine,
                executor=executor,
                rollout_profile="finance_riu_participant_handoff_15",
                role_training_mode="participant_only",
                rollout_lora_name=None,
                participant_backend="rollout",
                fixed_coordinator_backend="base_model",
                reward_config="written_holistic_score",
                written_judge_model=args.judge_model,
                written_judge_provider=args.judge_provider,
                written_judge_base_url=args.judge_base_url,
                written_judge_endpoint=args.judge_endpoint,
                written_judge_credentials_file=args.judge_credentials_file,
                written_judge_api_key_env=args.judge_api_key_env,
                written_judge_cache_dir=str(output / "judge_cache"),
                written_judge_fail_on_error=True,
                max_parse_retries=args.max_parse_retries,
                invalid_json_penalty=args.invalid_format_penalty,
                invalid_schema_penalty=args.invalid_format_penalty,
                format_credit_assignment="per_attempt",
                request_timeout_sec=args.timeout_sec,
                artifact_dir=str(artifacts),
                trace_prompts=True,
            )
            workflow.training_mode = "val"
            try:
                episode = await workflow.run(task, f"eval-{index:03d}-{instance_id}")
                teamgym = episode.info.get("teamgym", {})
                judgment = teamgym.get("written_answer_judgment") or {}
                normalized = judgment.get("judgment") or {}
                result = {
                    "index": index,
                    "instance_id": instance_id,
                    "dataset": datapoint.get("dataset"),
                    "participant_count": len(
                        datapoint.get("participant_directory", [])
                    ),
                    "correct": bool(episode.is_correct),
                    "holistic_score": float(normalized.get("score") or 0.0),
                    "format_error_count": episode.metrics.get(
                        "teamgym_format_error_count", 0.0
                    ),
                    "communication_turns": episode.metrics.get(
                        "teamgym_participant_communication_turns", 0.0
                    ),
                    "unique_speakers": episode.metrics.get(
                        "teamgym_participant_unique_speakers", 0.0
                    ),
                    "finalize_turn": episode.metrics.get(
                        "teamgym_participant_finalize_turn", 0.0
                    ),
                    "artifact": str(
                        (artifacts / f"eval-{index:03d}-{instance_id}.json").resolve()
                    ),
                    "error": teamgym.get("runtime_error"),
                }
            except BaseException as exc:  # noqa: BLE001
                result = {
                    "index": index,
                    "instance_id": instance_id,
                    "dataset": datapoint.get("dataset"),
                    "participant_count": len(
                        datapoint.get("participant_directory", [])
                    ),
                    "correct": False,
                    "holistic_score": 0.0,
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                }
            results.append(result)
            with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False), flush=True)

    try:
        await asyncio.gather(
            *(run_one(index, row) for index, row in enumerate(selected, 1))
        )
    finally:
        executor.shutdown(wait=True)

    results.sort(key=lambda row: int(row["index"]))
    by_dataset: dict[str, Any] = {}
    for dataset in datasets:
        subset = [row for row in results if row.get("dataset") == dataset]
        by_dataset[dataset] = _aggregate(subset)
    summary = {
        "schema_version": "finance_participant_handoff_eval_summary_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall": _aggregate(results),
        "by_dataset": by_dataset,
        "results": results,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    completed = [row for row in rows if not row.get("error")]
    correct_count = sum(bool(row.get("correct")) for row in rows)
    return {
        "selected_count": total,
        "completed_count": len(completed),
        "error_count": total - len(completed),
        "correct_count": correct_count,
        "accuracy": (
            correct_count / total
            if total
            else 0.0
        ),
        "completed_only_accuracy": (
            correct_count / len(completed) if completed else 0.0
        ),
        "mean_holistic_score": (
            sum(float(row.get("holistic_score") or 0.0) for row in rows)
            / total
            if total
            else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default="findver,finlongdocqa")
    parser.add_argument("--per-dataset", type=int, default=10)
    parser.add_argument("--minimum-participants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="teamgym-local")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=900.0)
    parser.add_argument("--max-parse-retries", type=int, default=1)
    parser.add_argument("--invalid-format-penalty", type=float, default=-0.1)
    parser.add_argument("--judge-model", default="gpt-5.5")
    parser.add_argument("--judge-provider", default="bedrock_openai")
    parser.add_argument(
        "--judge-base-url",
        default="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
    )
    parser.add_argument("--judge-endpoint", default="/responses")
    parser.add_argument(
        "--judge-credentials-file", default="api_key/api_key.py"
    )
    parser.add_argument(
        "--judge-api-key-env", default="AWS_BEARER_TOKEN_BEDROCK"
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(evaluate(parse_args()))


if __name__ == "__main__":
    main()
