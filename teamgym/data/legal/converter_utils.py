"""Shared prompt-call helpers for legal dataset generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from teamgym.core.io import compact_text, prompt_text, render_template

from ...llm import OpenAIJsonClient


def call_legal_planner(
    *,
    dataset: str,
    prompt_name: str,
    variables: dict[str, Any],
    llm_client: OpenAIJsonClient,
    trace_root: str | Path,
    instance_id: str,
) -> dict[str, Any]:
    system_prompt = prompt_text(f"legal/{dataset}/system.txt")
    user_template = prompt_text(f"legal/{dataset}/{prompt_name}.txt")
    user_prompt = render_template(user_template, variables)
    trace_dir = Path(trace_root) / dataset / instance_id
    return llm_client.generate_json(
        prompt_id=f"{dataset}.{prompt_name}",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        trace_dir=trace_dir,
    )


def compact_json(payload: Any, *, max_chars: int) -> str:
    return compact_text(json.dumps(payload, indent=2, ensure_ascii=False), max_chars=max_chars)
