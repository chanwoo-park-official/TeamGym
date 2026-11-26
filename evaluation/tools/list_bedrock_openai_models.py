#!/usr/bin/env python3
"""List model IDs exposed by the configured Bedrock OpenAI endpoint."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from teamgym.llm.openai_client import OpenAIJsonClient, OpenAIJsonClientConfig


def main() -> None:
    client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider="bedrock_openai",
            credentials_file="api_key/api_key.py",
            temperature=None,
        )
    )
    page = client._client.models.list()  # The wrapper does not otherwise expose model discovery.
    model_ids = sorted(item.id for item in page.data)
    print(json.dumps({"count": len(model_ids), "model_ids": model_ids}, indent=2))


if __name__ == "__main__":
    main()
