"""Consistent explicit constructor/auditor model and endpoint settings."""
from __future__ import annotations

import os

from teamgym.llm.construction_client import OpenAIJsonClientConfig
from teamgym.llm.openai_client import (
    DEFAULT_LLM_PROVIDER, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_ENDPOINT,
)


def add_model_arguments(parser) -> None:
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--endpoint", default=DEFAULT_LLM_ENDPOINT)
    parser.add_argument("--reasoning-effort", default=os.environ.get("COLLABGYM_LLM_REASONING_EFFORT", "medium"))


def model_config(*, model, env_file, provider=None, base_url=None, api_key_env=None,
                 endpoint=None, reasoning_effort=None, max_output_tokens=16_000):
    return OpenAIJsonClientConfig(
        model=model, provider=provider or DEFAULT_LLM_PROVIDER,
        base_url=base_url or DEFAULT_LLM_BASE_URL,
        api_key_env=api_key_env or DEFAULT_LLM_API_KEY_ENV,
        endpoint=endpoint or DEFAULT_LLM_ENDPOINT, env_file=env_file,
        temperature=None, reasoning_effort=reasoning_effort or os.environ.get("COLLABGYM_LLM_REASONING_EFFORT", "medium"),
        timeout_sec=1200, request_retry_attempts=2, max_output_tokens=max_output_tokens)


def model_options(args) -> dict:
    return {key: getattr(args, key) for key in
            ("provider", "base_url", "api_key_env", "endpoint", "reasoning_effort")}
