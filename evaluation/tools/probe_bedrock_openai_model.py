#!/usr/bin/env python3
"""Run a minimal structured-output request against one Bedrock model ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from teamgym.llm.openai_client import OpenAIJsonClient, OpenAIJsonClientConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--endpoint", choices=("/responses", "/chat/completions"), default="/responses")
    parser.add_argument("--base-url", default="https://bedrock-mantle.us-east-2.api.aws/openai/v1")
    args = parser.parse_args()
    client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider="bedrock_openai",
            model=args.model,
            base_url=args.base_url,
            endpoint=args.endpoint,
            credentials_file="api_key/api_key.py",
            temperature=0,
            max_output_tokens=64,
            request_retry_attempts=1,
        )
    )
    response = client.generate_json(
        prompt_id="model_probe",
        system_prompt="Return JSON only.",
        user_prompt='Return {"available": true}.',
    )
    print(
        json.dumps(
            {
                "requested_model": args.model,
                "resolved_model": client.cfg.model,
                "endpoint": args.endpoint,
                "base_url": args.base_url,
                "response": response,
            }
        )
    )


if __name__ == "__main__":
    main()
