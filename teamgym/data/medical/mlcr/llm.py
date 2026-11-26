"""MLCR-local prompt rendering and resumable structured-response cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable
from .execution import dependency_identity, assert_frozen_dependencies
from teamgym.tokenization import count_tokens


PROMPT_ROOT = Path(__file__).resolve().parents[3] / "prompts" / "medical" / "mlcr"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    with source.open(encoding="utf-8") as handle:
        rows = []
        for line in handle:
            if line.strip():
                try:
                    row = json.loads(line)
                    if isinstance(row, dict): rows.append(row)
                except ValueError:
                    continue
        return rows


def atomic_text(path: str | Path, payload: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + target.name, dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def inference_identity(client) -> dict:
    cfg = getattr(client, "cfg", None)
    return {key: getattr(cfg, key, None) for key in (
        "model", "provider", "base_url", "endpoint", "api_key_env", "temperature",
        "reasoning_effort", "max_output_tokens", "timeout_sec", "request_retry_attempts")}


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def load_prompt(name: str) -> str:
    path = PROMPT_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, variables: dict[str, Any]) -> str:
    rendered = template
    for key, value in variables.items():
        replacement = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
        rendered = rendered.replace("{{" + key + "}}", replacement)
    unresolved = sorted(set(part[2:-2] for part in _template_tokens(rendered)))
    if unresolved:
        raise ValueError(f"unresolved prompt variables: {unresolved}")
    return rendered


def _template_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"\{\{[A-Z0-9_]+\}\}", text)


class ResponseCache:
    def __init__(self, path: str | Path, *, max_input_tokens: int = 180_000):
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        self.max_input_tokens = max_input_tokens
        self.path = Path(path)
        self.dependencies = dependency_identity()
        self.rows = {str(row["cache_key"]): row for row in read_jsonl(self.path)
                     if isinstance(row.get("cache_key"), str) and isinstance(row.get("response"), dict)}

    def call(
        self,
        *,
        client: Any,
        stage: str,
        case_id: str,
        system_prompt: str,
        user_prompt: str,
        trace_dir: Path,
    ) -> dict[str, Any]:
        assert_frozen_dependencies(self.dependencies)
        digest = hashlib.sha256(
            json.dumps(
                {
                    "inference": inference_identity(client),
                    "dependencies_sha256": self.dependencies["sha256"],
                    "stage": stage,
                    "system": system_prompt,
                    "user": user_prompt,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        key = f"{case_id}:{stage}:{digest}"
        if key in self.rows:
            self.last_cache_event = {"cache_key": key, "status": "cache_hit"}
            return dict(self.rows[key]["response"])
        self.last_cache_event = {"cache_key": key, "status": "fresh"}
        response = client.generate_json(
            prompt_id=f"mlcr.{stage}",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_dir=trace_dir / case_id,
        )
        if not isinstance(response, dict):
            raise ValueError("structured LLM response must be a JSON object")
        self.rows[key] = {
            "cache_key": key,
            "case_id": case_id,
            "stage": stage,
            "model": str(getattr(getattr(client, "cfg", None), "model", "unknown")),
            "response": response,
            "inference": inference_identity(client),
            "request_meta": dict(getattr(client, "last_request_meta", {}) or {}),
        }
        write_jsonl(self.path, list(self.rows.values()))
        return response


def structured_call(
    *,
    client: Any,
    cache: ResponseCache,
    trace_root: Path,
    case_id: str,
    stage: str,
    user_prompt_name: str,
    variables: dict[str, Any],
    validator: Callable[[Any], list[str]],
    attempts: int,
    system_prompt_name: str = "construction_system.txt",
    initial_feedback: Any = "None",
    max_input_tokens: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if max_input_tokens is None:
        max_input_tokens = getattr(cache, "max_input_tokens", 180_000)
    response: dict[str, Any] = {}
    feedback: Any = initial_feedback
    errors = ["stage did not run"]
    for attempt in range(1, max(1, attempts) + 1):
        system = load_prompt(system_prompt_name)
        user = render_prompt(load_prompt(user_prompt_name), {**variables, "VALIDATION_FEEDBACK": feedback})
        tokens = count_tokens(system) + count_tokens(user)
        if tokens > max_input_tokens:
            return {}, [f"input budget exceeded: {tokens} tokens > {max_input_tokens}; no model call made"]
        try:
            response = cache.call(
                client=client,
                stage=f"{stage}.attempt_{attempt}",
                case_id=case_id,
                system_prompt=system,
                user_prompt=user,
                trace_dir=trace_root,
            )
            errors = validator(response)
        except (ValueError, TypeError) as exc:
            response, errors = {}, ["invalid model JSON: " + str(exc)]
        if not errors:
            return response, []
        feedback = {"previous_response_invalid": True, "errors": errors}
    return response, errors
