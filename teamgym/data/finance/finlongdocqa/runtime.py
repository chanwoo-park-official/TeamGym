"""I/O, prompt rendering, and LLM client construction."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from teamgym.llm.construction_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
PROMPT_ROOT = (
    REPO_ROOT
    / "teamgym"
    / "prompts"
    / "finance"
    / "finlongdocqa"
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    if not source.exists():
        return []
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_text(path: str | Path, payload: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + output.name, dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + os.linesep)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + os.linesep for row in rows))


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + os.linesep)


def load_env_value(path: str | Path, key: str) -> str:
    for raw in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'").strip('"')
    raise ValueError(f"{key} not found in {path}")


def build_client(args: Any, request_log_path: Path) -> Any:
    env_file = Path(args.env_file).expanduser()
    if not os.environ.get(args.api_key_env):
        os.environ[args.api_key_env] = load_env_value(
            env_file,
            args.api_key_env,
        )
    provider = args.provider
    base_url = args.base_url
    if provider == "bedrock_anthropic":
        provider = "bedrock_anthropic_runtime"
        base_url = f"https://bedrock-runtime.{args.region}.amazonaws.com"
    return OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider=provider,
            model=args.model,
            base_url=base_url,
            api_key_env=args.api_key_env,
            endpoint=args.endpoint,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            timeout_sec=args.timeout_sec,
            request_retry_attempts=args.request_retry_attempts,
            request_log_path=str(request_log_path),
            reasoning_effort=getattr(args, "reasoning_effort", None),
        )
    )


def call_json(
    client: Any,
    *,
    system_prompt_name: str,
    user_prompt_name: str,
    variables: dict[str, Any],
    trace_dir: Path,
) -> dict[str, Any]:
    system = _prompt_text(system_prompt_name)
    user = _render(_prompt_text(user_prompt_name), variables)
    return client.generate_json(
        prompt_id=f"finance.finlongdocqa.{Path(user_prompt_name).stem}",
        system_prompt=system,
        user_prompt=user,
        trace_dir=trace_dir,
    )


def _prompt_text(name: str) -> str:
    path = PROMPT_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _render(template: str, variables: dict[str, Any]) -> str:
    rendered = template
    for key, value in variables.items():
        replacement = (
            value
            if isinstance(value, str)
            else json.dumps(value, indent=2, ensure_ascii=False)
        )
        rendered = rendered.replace(f"{{{{{key}}}}}", replacement)
    return rendered
