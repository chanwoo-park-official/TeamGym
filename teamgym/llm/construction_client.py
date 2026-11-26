"""Construction-specific options for the shared JSON client."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from .openai_client import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    OpenAIJsonClient as SharedJsonClient,
    OpenAIJsonClientConfig as SharedJsonClientConfig,
)


@dataclass
class OpenAIJsonClientConfig(SharedJsonClientConfig):
    env_file: str | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if (
            "moonshot" in self.model.lower()
            and self.base_url == "https://api.openai.com/v1"
            and self.api_key_env == "OPENAI_API_KEY"
        ):
            self.base_url = "https://api.moonshot.cn/v1"
            self.api_key_env = "MOONSHOT_API_KEY"


class OpenAIJsonClient(SharedJsonClient):
    def _resolve_api_key(self) -> str | None:
        values = {}
        if self.cfg.env_file:
            path = Path(self.cfg.env_file).expanduser()
            if path.is_file():
                for raw_line in path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        values[key.strip()] = value.strip().strip("\"'")
        loaded = self._load_credentials_file()
        for key in (
            self.cfg.api_key_env, "OPENAI_API_KEY", "AWS_BEARER_TOKEN_BEDROCK",
            "BEDROCK_OPENAI_API_KEY",
        ):
            if key and (value := os.environ.get(key) or values.get(key) or loaded.get(key)):
                return str(value)
        return None

    def _request(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if "json" not in system_prompt.lower() and "json" not in user_prompt.lower():
            system_prompt += "\n\nPlease respond in JSON format."
        if self.cfg.reasoning_effort and self.cfg.endpoint == "/responses" and self._client is not None:
            body = {
                "model": self.cfg.model,
                "input": [
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "text": {"format": {"type": "json_object"}, "verbosity": "low"},
                "max_output_tokens": max(1, int(self.cfg.max_output_tokens)),
                "reasoning": {"effort": self.cfg.reasoning_effort},
            }
            if self.cfg.temperature is not None:
                body["temperature"] = self.cfg.temperature
            try:
                response = self._client.responses.create(**body)
            except Exception as exc:
                if "temperature" not in body or "temperature" not in str(exc).lower():
                    raise
                body.pop("temperature")
                response = self._client.responses.create(**body)
            payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        else:
            payload = super()._request(system_prompt, user_prompt)
        self._validate_response_status(payload)
        return payload

    @staticmethod
    def _validate_response_status(payload: dict[str, Any]) -> None:
        if payload.get("status") == "incomplete":
            details = payload.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, dict) else None
            raise ValueError(f"LLM response was incomplete: {reason or 'unknown reason'}")
