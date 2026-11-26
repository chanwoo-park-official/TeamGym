"""Prompt loading, structured LLM calls, and resumable response caching."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable
from .execution import dependency_identity, assert_frozen_dependencies
from teamgym.tokenization import count_tokens


PROMPT_ROOT = Path(__file__).resolve().parents[3] / "prompts" / "medical" / "evidencebench"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + os.linesep)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and isinstance(row.get("response"), dict) and isinstance(row.get("cache_key"), str):
                        rows.append(row)
                except ValueError:
                    continue
    return rows


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def inference_identity(client) -> dict:
    cfg = getattr(client, "cfg", None)
    return {key: getattr(cfg, key, None) for key in (
        "model", "provider", "base_url", "endpoint", "api_key_env", "temperature",
        "reasoning_effort", "max_output_tokens", "timeout_sec", "request_retry_attempts")}


def _prompt(name: str) -> str:
    path = PROMPT_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _render(template: str, variables: dict[str, Any]) -> str:
    rendered = template
    for key, value in variables.items():
        text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
        rendered = rendered.replace("{{" + key + "}}", text)
    return rendered


class ResponseCache:
    """Small resumable cache local to the EvidenceBench pipeline."""

    def __init__(self, path: Path, *, max_input_tokens: int = 180_000):
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        self.max_input_tokens = max_input_tokens
        self.path = path
        self.dependencies = dependency_identity()
        self.rows = {str(row["cache_key"]): row for row in _read_jsonl(path)}

    def call(
        self,
        *,
        client: Any,
        stage: str,
        instance_id: str,
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
            ).encode("utf-8")
        ).hexdigest()
        key = f"{instance_id}:{stage}:{digest}"
        if key in self.rows:
            self.last_cache_event = {"cache_key": key, "status": "cache_hit"}
            return dict(self.rows[key]["response"])
        self.last_cache_event = {"cache_key": key, "status": "fresh"}
        response = client.generate_json(
            prompt_id=f"evidencebench.{stage}",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_dir=trace_dir / instance_id,
        )
        if not isinstance(response, dict):
            raise ValueError("structured LLM response must be a JSON object")
        self.rows[key] = {
            "cache_key": key,
            "instance_id": instance_id,
            "stage": stage,
            "model": client.cfg.model,
            "inference": inference_identity(client),
            "response": response,
            "request_meta": dict(getattr(client, "last_request_meta", {}) or {}),
        }
        _write_jsonl(self.path, list(self.rows.values()))
        return response


def structured_call(
    *,
    client: Any,
    cache: ResponseCache,
    trace_root: Path,
    instance_id: str,
    stage: str,
    user_prompt_name: str,
    variables: dict[str, Any],
    validator: Callable[[Any], list[str]],
    attempts: int,
    system_prompt_name: str = "construction_system.txt",
    initial_feedback: Any = "None",
) -> tuple[dict[str, Any], list[str]]:
    feedback: Any = initial_feedback
    response: dict[str, Any] = {}
    errors = ["stage did not run"]
    for attempt in range(1, attempts + 1):
        system = _prompt(system_prompt_name)
        user = _render(_prompt(user_prompt_name), {**variables, "VALIDATION_FEEDBACK": feedback})
        tokens = count_tokens(system) + count_tokens(user)
        budget = getattr(cache, "max_input_tokens", 180_000)
        if tokens > budget:
            return {}, [f"input budget exceeded: {tokens} tokens > {budget}; no model call made"]
        try:
            response = cache.call(
                client=client,
                stage=f"{stage}.attempt_{attempt}",
                instance_id=instance_id,
                system_prompt=system,
                user_prompt=user,
                trace_dir=trace_root,
            )
            errors = validator(response)
        except (ValueError, TypeError) as exc:
            response, errors = {}, ["invalid model JSON: " + str(exc)]
        if not errors:
            return response, []
        feedback = {
            "previous_response_invalid": True,
            "errors": errors,
            "instruction": "Re-audit the unchanged input and return corrected JSON.",
        }
    return response, errors
