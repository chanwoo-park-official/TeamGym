"""Cached semantic judging for TeamGym written-answer rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from teamgym.scoring.short_answers import (
    JUDGE_SCHEMA,
    judge_one_answer,
    normalize_judge_score,
    requires_all_steps_for_pass,
)
from teamgym.llm.openai_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)


CACHE_SCHEMA = "teamgym_written_reward_cache_v2"


@dataclass(frozen=True)
class WrittenJudgeConfig:
    model: str = "gpt-5.5"
    provider: str = "bedrock_openai"
    base_url: str = "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
    endpoint: str = "/responses"
    credentials_file: str = "api_key/api_key.py"
    api_key_env: str = "AWS_BEARER_TOKEN_BEDROCK"
    max_output_tokens: int = 3000
    timeout_sec: int = 600
    request_retry_attempts: int = 4
    cache_dir: str | None = None


def is_written_answer_task(datapoint: dict[str, Any]) -> bool:
    task = (
        datapoint.get("task_description")
        if isinstance(datapoint.get("task_description"), dict)
        else {}
    )
    specific = (
        datapoint.get("dataset_specific")
        if isinstance(datapoint.get("dataset_specific"), dict)
        else {}
    )
    answer_format = (
        task.get("answer_format")
        if isinstance(task.get("answer_format"), dict)
        else {}
    )
    return (
        task.get("answer_type") == "short_answer_judged"
        or specific.get("answer_type") == "short_answer_judged"
        or answer_format.get("type") == "free_response"
        or datapoint.get("decision_set") == ["free_response"]
    )


def written_answer_from_episode(episode: dict[str, Any] | None) -> str:
    if not isinstance(episode, dict):
        return ""
    final = (
        episode.get("final_decision")
        if isinstance(episode.get("final_decision"), dict)
        else {}
    )
    return str(
        final.get("written_answer")
        or final.get("rationale")
        or ""
    ).strip()


def holistic_score(judgment: dict[str, Any] | None) -> float:
    normalized = (
        judgment.get("judgment")
        if isinstance(judgment, dict)
        and isinstance(judgment.get("judgment"), dict)
        else {}
    )
    raw = normalized.get("raw_response")
    if isinstance(raw, dict):
        # Cached judgments from older runs can contain a clamped non-finite score.
        normalize_judge_score(raw.get("score"))
    return normalize_judge_score(normalized.get("score"))


def judge_written_episode(
    *,
    datapoint: dict[str, Any],
    episode: dict[str, Any],
    config: WrittenJudgeConfig,
) -> dict[str, Any]:
    """Judge the final written answer, using an atomic content-addressed cache."""

    answer_text = written_answer_from_episode(episode)
    if not answer_text:
        raise ValueError("written-answer episode has no non-empty final answer")
    requires_all_steps_for_pass(datapoint)
    answer_row = {
        "instance_id": datapoint.get("instance_id"),
        "answer_text": answer_text,
        "answer_source": "teamgym_rl_rollout",
        "communication_structure": (
            (episode.get("communication_structure") or {}).get("name")
            if isinstance(episode.get("communication_structure"), dict)
            else None
        ),
    }
    cache_path = _cache_path(
        datapoint=datapoint,
        answer_text=answer_text,
        config=config,
    )
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("cache_schema") == CACHE_SCHEMA
            and cached.get("judge_schema") == JUDGE_SCHEMA
            and isinstance(cached.get("judgment"), dict)
            and cached["judgment"].get("schema") == JUDGE_SCHEMA
        ):
            result = dict(cached["judgment"])
            holistic_score(result)
            result["cache_hit"] = True
            return result

    client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider=config.provider,
            model=config.model,
            base_url=config.base_url,
            endpoint=config.endpoint,
            credentials_file=config.credentials_file,
            api_key_env=config.api_key_env,
            temperature=0,
            max_output_tokens=config.max_output_tokens,
            timeout_sec=config.timeout_sec,
            request_retry_attempts=config.request_retry_attempts,
            request_log_path=None,
            log_prompts=False,
            log_responses=False,
        )
    )
    result = judge_one_answer(
        client=client,
        datapoint=datapoint,
        answer_row=answer_row,
        trace_dir=None,
        index=1,
    )
    result["cache_hit"] = False
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_schema": CACHE_SCHEMA,
            "judge_schema": JUDGE_SCHEMA,
            "judgment": result,
        }
        temporary = cache_path.with_name(
            f"{cache_path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, cache_path)
    return result


def _cache_path(
    *,
    datapoint: dict[str, Any],
    answer_text: str,
    config: WrittenJudgeConfig,
) -> Path | None:
    if not config.cache_dir:
        return None
    task = (
        datapoint.get("task_description")
        if isinstance(datapoint.get("task_description"), dict)
        else {}
    )
    oracle = (
        datapoint.get("oracle")
        if isinstance(datapoint.get("oracle"), dict)
        else {}
    )
    key_payload = {
        "cache_schema": CACHE_SCHEMA,
        "judge_schema": JUDGE_SCHEMA,
        "model": config.model,
        "provider": config.provider,
        "instance_id": datapoint.get("instance_id"),
        "question": task.get("question") or task.get("objective"),
        "canonical_response": (
            oracle.get("canonical_response")
            or oracle.get("short_answer")
            or oracle.get("answer_text")
        ),
        "required_decision_steps": oracle.get("required_decision_steps"),
        "judge_rubric": oracle.get("judge_rubric"),
        "disqualifying_errors": oracle.get("disqualifying_errors"),
        "candidate_answer": answer_text,
    }
    digest = hashlib.sha256(
        json.dumps(
            key_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return Path(config.cache_dir).expanduser().resolve() / f"{digest}.json"
