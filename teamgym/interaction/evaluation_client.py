"""Model client configuration shared by the standard evaluation runners."""

from argparse import Namespace
from pathlib import Path
from typing import Any

from teamgym.llm.openai_client import OpenAIJsonClient, OpenAIJsonClientConfig


def evaluation_model_manifest(args: Namespace) -> dict[str, Any]:
    return {
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "api_key_env": args.api_key_env,
        "credentials_file": args.credentials_file,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "timeout_sec": args.timeout_sec,
        "request_retry_attempts": args.request_retry_attempts,
    }


def make_evaluation_client(args: Namespace, output_dir: Path) -> OpenAIJsonClient:
    return OpenAIJsonClient(
        OpenAIJsonClientConfig(
            **evaluation_model_manifest(args),
            request_log_path=str(output_dir / "llm_requests.jsonl"),
            log_prompts=bool(args.trace_prompts),
            log_responses=bool(args.trace_prompts),
        )
    )
