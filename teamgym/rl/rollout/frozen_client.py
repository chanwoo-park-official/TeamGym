"""JSON client for separately hosted, immutable TeamGym policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from teamgym.llm.openai_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)
from .action_protocol import _action_schema_error, _prompt_kind, _prompt_payload
from .policy_routing import BASE_LORA_NAME, _actor_role_for_prompt


class FrozenOpenAIJsonClient:
    """Record calls to a separately hosted, immutable OpenAI-compatible model."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        max_output_tokens: int,
        temperature: float,
        timeout_sec: float,
        attempt_records: list[dict[str, Any]],
        max_parse_retries: int = 0,
        policy_route: str = "frozen_external_base",
    ) -> None:
        self.client = OpenAIJsonClient(
            OpenAIJsonClientConfig(
                provider="openai_compatible",
                model=model,
                base_url=base_url,
                api_key=api_key,
                endpoint="/chat/completions",
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout_sec=max(1, int(timeout_sec)),
                request_retry_attempts=2,
                log_prompts=False,
                log_responses=False,
            )
        )
        self.attempt_records = attempt_records
        self.max_parse_retries = max(0, int(max_parse_retries))
        self.policy_route = policy_route
        self.last_request_meta: dict[str, Any] = {}

    def generate_json(
        self,
        *,
        prompt_id: str,
        system_prompt: str,
        user_prompt: str,
        trace_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        role = _actor_role_for_prompt(prompt_id)
        prompt_kind = _prompt_kind(prompt_id)
        retry_system_prompt = system_prompt
        response: dict[str, Any] | None = None
        schema_error: str | None = None
        for retry in range(self.max_parse_retries + 1):
            try:
                response = self.client.generate_json(
                    prompt_id=prompt_id,
                    system_prompt=retry_system_prompt,
                    user_prompt=user_prompt,
                    trace_dir=trace_dir,
                )
            except BaseException as exc:  # noqa: BLE001
                self.last_request_meta = dict(self.client.last_request_meta)
                self.attempt_records.append(
                    {
                        "episode_id": prompt_id.split(".", 1)[0],
                        "prompt_id": prompt_id,
                        "prompt_kind": prompt_kind,
                        "actor_role": role,
                        "retry_number": retry,
                        "parse_status": "request_error",
                        "parse_error": str(exc)[:500],
                        "raw_output": "",
                        "raw_output_token_count": 0,
                        "train_mask": False,
                        "policy_route": self.policy_route,
                        "local_reward_penalty": 0.0,
                        "lora_name": BASE_LORA_NAME,
                        "system_prompt": retry_system_prompt,
                        "user_prompt": user_prompt,
                        "parsed_response": None,
                    }
                )
                raise
            self.last_request_meta = dict(self.client.last_request_meta)
            schema_error = _action_schema_error(
                response,
                prompt_kind=prompt_kind,
                prompt_payload=_prompt_payload(user_prompt),
            )
            self.attempt_records.append(
                {
                    "episode_id": prompt_id.split(".", 1)[0],
                    "prompt_id": prompt_id,
                    "prompt_kind": prompt_kind,
                    "actor_role": role,
                    "retry_number": retry,
                    "parse_status": (
                        "ok" if schema_error is None else "invalid_schema"
                    ),
                    "parse_error": schema_error,
                    # The ordinary client exposes parsed JSON rather than the
                    # raw completion; preserve a canonical serialization.
                    "raw_output": json.dumps(response, ensure_ascii=False),
                    "raw_output_token_count": 0,
                    "train_mask": False,
                    "policy_route": self.policy_route,
                    "local_reward_penalty": 0.0,
                    "lora_name": BASE_LORA_NAME,
                    "system_prompt": retry_system_prompt,
                    "user_prompt": user_prompt,
                    "parsed_response": response,
                    "request_meta": self.last_request_meta,
                }
            )
            if schema_error is None:
                return response
            retry_system_prompt = (
                f"{system_prompt}\n"
                "Your previous JSON response violated response_format: "
                f"{schema_error}. Correct that error and return exactly one "
                "JSON object matching response_format."
            )
        assert response is not None
        # A frozen small-model finalizer sometimes returns the requested
        # object under a literal ``response_format`` wrapper. Keep this
        # visibly invalid in the format log, but unwrap it for runtime use so
        # a non-trainable environment component does not erase the
        # participant's semantic reward.
        nested = response.get("response_format")
        if isinstance(nested, dict) and _action_schema_error(
            nested,
            prompt_kind=prompt_kind,
            prompt_payload=_prompt_payload(user_prompt),
        ) is None:
            self.attempt_records[-1]["runtime_repair_method"] = (
                "unwrap_response_format"
            )
            return nested
        return response
