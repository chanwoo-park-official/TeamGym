"""Small OpenAI JSON client used by legal data generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import uuid


DEFAULT_LLM_PROVIDER = os.environ.get("COLLABGYM_LLM_PROVIDER", "bedrock_openai")
DEFAULT_LLM_MODEL = os.environ.get("COLLABGYM_LLM_MODEL", "gpt-5.5")
DEFAULT_LLM_BASE_URL = os.environ.get("COLLABGYM_LLM_BASE_URL", "https://bedrock-mantle.us-east-2.api.aws/openai/v1")
DEFAULT_LLM_ENDPOINT = os.environ.get("COLLABGYM_LLM_ENDPOINT", "/responses")
DEFAULT_LLM_API_KEY_ENV = os.environ.get("COLLABGYM_LLM_API_KEY_ENV", "AWS_BEARER_TOKEN_BEDROCK")


@dataclass
class OpenAIJsonClientConfig:
    provider: str = DEFAULT_LLM_PROVIDER
    model: str = DEFAULT_LLM_MODEL
    base_url: str = DEFAULT_LLM_BASE_URL
    api_key_env: str = DEFAULT_LLM_API_KEY_ENV
    api_key: str | None = None
    credentials_file: str | None = None
    endpoint: str = DEFAULT_LLM_ENDPOINT
    temperature: float | None = 0.2
    max_output_tokens: int = 6000
    timeout_sec: int = 240
    request_retry_attempts: int = 4
    request_log_path: str | None = None
    log_prompts: bool = True
    log_responses: bool = True


def normalize_model_for_provider(provider: str, model: str) -> str:
    if provider == "bedrock_openai" and model.startswith("gpt-"):
        return f"openai.{model}"
    return model


class OpenAIJsonClient:
    """Strict JSON-generation wrapper for OpenAI-compatible and Anthropic APIs.

    This class intentionally has no synthetic/offline fallback. If credentials,
    network access, or the model call fail, generation fails and the caller can
    skip or stop according to its explicit CLI policy.
    """

    _request_log_lock = threading.Lock()

    def __init__(self, cfg: OpenAIJsonClientConfig):
        self.cfg = cfg
        self.cfg.model = normalize_model_for_provider(cfg.provider, cfg.model)
        self.api_key = cfg.api_key or self._resolve_api_key()
        supported_providers = {
            "openai",
            "openai_compatible",
            "bedrock_openai",
            "bedrock_anthropic_mantle",
            "bedrock_anthropic_runtime",
        }
        if cfg.provider not in supported_providers:
            raise ValueError(f"Unsupported LLM provider: {cfg.provider}")
        if not self.api_key:
            raise ValueError(f"Missing API key. Set {cfg.api_key_env} or pass api_key explicitly.")
        self._client = None
        if cfg.provider in {"bedrock_anthropic_mantle", "bedrock_anthropic_runtime"}:
            self.last_request_meta: dict[str, Any] = {}
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the OpenAI SDK with `python -m pip install openai`.") from exc

        kwargs: dict[str, Any] = {"timeout": max(1, int(cfg.timeout_sec))}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self._client = OpenAI(**kwargs)
        self.last_request_meta: dict[str, Any] = {}

    def generate_json(
        self,
        *,
        prompt_id: str,
        system_prompt: str,
        user_prompt: str,
        trace_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        call_id = uuid.uuid4().hex
        trace_path = Path(trace_dir).expanduser().resolve() if trace_dir else None
        if trace_path is not None:
            trace_path.mkdir(parents=True, exist_ok=True)
            if self.cfg.log_prompts:
                (trace_path / f"{prompt_id}.{call_id}.system.txt").write_text(system_prompt, encoding="utf-8")
                (trace_path / f"{prompt_id}.{call_id}.user.txt").write_text(user_prompt, encoding="utf-8")

        started = time.perf_counter()
        self._log(
            {
                "event": "llm_call_started",
                "call_id": call_id,
                "prompt_id": prompt_id,
                "model": self.cfg.model,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "prompt_chars": len(system_prompt) + len(user_prompt),
            }
        )
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.cfg.request_retry_attempts) + 1):
            try:
                payload = self._request(system_prompt, user_prompt)
                output_text = self._extract_output_text(payload)
                if trace_path is not None and self.cfg.log_responses:
                    (trace_path / f"{prompt_id}.{call_id}.attempt_{attempt:02d}.raw_response.json").write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    (trace_path / f"{prompt_id}.{call_id}.attempt_{attempt:02d}.output.txt").write_text(
                        output_text,
                        encoding="utf-8",
                    )
                parsed = self._parse_json_object(output_text)
                elapsed = round(time.perf_counter() - started, 4)
                self.last_request_meta = {
                    "provider": self.cfg.provider,
                    "model": self.cfg.model,
                    "endpoint": self.cfg.endpoint,
                    "response_id": payload.get("id"),
                    "status": payload.get("status"),
                    "usage": payload.get("usage"),
                    "elapsed_sec": elapsed,
                    "attempt": attempt,
                }
                if trace_path is not None and self.cfg.log_responses:
                    (trace_path / f"{prompt_id}.{call_id}.parsed.json").write_text(
                        json.dumps(parsed, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                self._log(
                    {
                        "event": "llm_call_finished",
                        "call_id": call_id,
                        "prompt_id": prompt_id,
                        "request_meta": self.last_request_meta,
                    }
                )
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._log(
                    {
                        "event": "llm_call_attempt_failed",
                        "call_id": call_id,
                        "prompt_id": prompt_id,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                if attempt >= max(1, self.cfg.request_retry_attempts):
                    break
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
        assert last_error is not None
        raise last_error

    def _request(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self.cfg.provider in {"bedrock_anthropic_mantle", "bedrock_anthropic_runtime"}:
            return self._request_bedrock_anthropic(system_prompt, user_prompt)
        if self.cfg.endpoint == "/responses":
            body: dict[str, Any] = {
                "model": self.cfg.model,
                "input": [
                    {
                        "role": "developer",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "text": {"format": {"type": "json_object"}, "verbosity": "low"},
                "max_output_tokens": max(1, int(self.cfg.max_output_tokens)),
            }
            if self.cfg.temperature is not None:
                body["temperature"] = self.cfg.temperature
            try:
                response = self._client.responses.create(**body)
            except Exception as exc:  # noqa: BLE001
                if "temperature" in str(exc).lower():
                    body.pop("temperature", None)
                    response = self._client.responses.create(**body)
                else:
                    raise
            return response.model_dump() if hasattr(response, "model_dump") else dict(response)

        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max(1, int(self.cfg.max_output_tokens)),
        }
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        try:
            response = self._client.chat.completions.create(**body)
        except Exception as exc:  # noqa: BLE001
            text = str(exc).lower()
            if "max_completion_tokens" in text and "unsupported" in text:
                body["max_tokens"] = body.pop("max_completion_tokens")
                response = self._client.chat.completions.create(**body)
            elif "temperature" in text:
                body.pop("temperature", None)
                response = self._client.chat.completions.create(**body)
            else:
                raise
        return response.model_dump() if hasattr(response, "model_dump") else dict(response)

    def _request_bedrock_anthropic(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max(1, int(self.cfg.max_output_tokens)),
        }
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature

        headers = {"content-type": "application/json", "accept": "application/json"}
        if self.cfg.provider == "bedrock_anthropic_mantle":
            body["model"] = self.cfg.model
            headers["x-api-key"] = str(self.api_key)
            headers["anthropic-version"] = "2023-06-01"
            url = f"{self.cfg.base_url.rstrip('/')}/messages"
        else:
            body["anthropic_version"] = "bedrock-2023-05-31"
            headers["Authorization"] = f"Bearer {self.api_key}"
            model_id = quote(self.cfg.model, safe=".")
            url = f"{self.cfg.base_url.rstrip('/')}/model/{model_id}/invoke"
        return self._http_json_request(url=url, body=body, headers=headers)

    def _http_json_request(
        self,
        *,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        request = Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=max(1, int(self.cfg.timeout_sec))) as response:  # noqa: S310
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {self.cfg.provider}: {detail[:1000]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Request to {self.cfg.provider} failed: {exc}") from exc
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object from {self.cfg.provider}")
        return parsed

    def _extract_output_text(self, payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        if "choices" in payload:
            choices = payload.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    return content
        if isinstance(payload.get("content"), list):
            anthropic_chunks = [
                str(item.get("text"))
                for item in payload["content"]
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
            ]
            if anthropic_chunks:
                return "\n".join(anthropic_chunks)
        chunks: list[str] = []
        for item in payload.get("output", []) or []:
            for content in item.get("content", []) or []:
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(stripped[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        raise ValueError(f"LLM response did not contain a JSON object: {text[:500]}")

    def _resolve_api_key(self) -> str | None:
        loaded = self._load_credentials_file()
        candidates = [
            self.cfg.api_key_env,
            "OPENAI_API_KEY",
            "AWS_BEARER_TOKEN_BEDROCK",
            "BEDROCK_OPENAI_API_KEY",
        ]
        for key in candidates:
            if not key:
                continue
            value = os.environ.get(key) or loaded.get(key)
            if value:
                return str(value)
        return None

    def _load_credentials_file(self) -> dict[str, Any]:
        if not self.cfg.credentials_file:
            return {}
        path = Path(self.cfg.credentials_file).expanduser()
        if not path.is_file():
            return {}
        spec = importlib.util.spec_from_file_location("_collabgym_fc_credentials", path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {name: getattr(module, name) for name in dir(module) if name.isupper()}

    def _log(self, row: dict[str, Any]) -> None:
        if not self.cfg.request_log_path:
            return
        path = Path(self.cfg.request_log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._request_log_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
