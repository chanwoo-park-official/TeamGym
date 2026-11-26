"""Synchronous JSON adapter over the asynchronous TeamGym rollout engine."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
import json
import os
from pathlib import Path
from typing import Any

from rllm.agents.agent import Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine

from .action_protocol import _action_schema_error, _parse_exact_json_object, _prompt_kind, _prompt_payload
from .action_repair import _repair_compact_action, _repair_participant_format_counterfactual
from .policy_routing import (
    BASE_LORA_NAME, FULL_MODEL_NAME, VLLM_LORA_INT_ID, VLLM_LORA_NAME,
    _actor_role_for_prompt, _is_trainable,
)


class RolloutEngineJsonClient:
    """Synchronous JSON facade over an async RLLM rollout engine.

    The paper runtime remains synchronous. It runs in a worker thread, while
    this facade schedules each actual model request on the workflow event loop.
    Every attempt becomes an RLLM step; the ordinary actors still own prompt
    construction, parsing consequences, truncation, and fallback behavior.
    """

    def __init__(
        self,
        *,
        rollout_engine: RolloutEngine,
        event_loop: asyncio.AbstractEventLoop,
        uid: str,
        trainable_trajectory: Trajectory,
        fixed_trajectory: Trajectory,
        training_mode: str,
        rollout_lora_name: str | None,
        max_output_tokens: int,
        participant_max_output_tokens: int | None,
        max_parse_retries: int,
        invalid_json_penalty: float,
        invalid_schema_penalty: float,
        request_timeout_sec: float,
        allow_compact_repair: bool = False,
        counterfactual_participant_format_repair: bool = False,
        counterfactual_participant_format_repair_strategy: str = "wrap_raw",
        counterfactual_participant_max_utterance_tokens: int = 0,
        counterfactual_participant_tokenizer: Any | None = None,
    ) -> None:
        self.rollout_engine = rollout_engine
        self.event_loop = event_loop
        self.uid = uid
        self.trainable_trajectory = trainable_trajectory
        self.fixed_trajectory = fixed_trajectory
        self.training_mode = training_mode
        self.rollout_lora_name = rollout_lora_name
        self.max_output_tokens = max(1, int(max_output_tokens))
        self.participant_max_output_tokens = max(
            1,
            int(
                participant_max_output_tokens
                if participant_max_output_tokens is not None
                else max_output_tokens
            ),
        )
        self.max_parse_retries = max(0, int(max_parse_retries))
        self.invalid_json_penalty = float(invalid_json_penalty)
        self.invalid_schema_penalty = float(invalid_schema_penalty)
        self.request_timeout_sec = max(1.0, float(request_timeout_sec))
        self.allow_compact_repair = bool(allow_compact_repair)
        self.counterfactual_participant_format_repair = bool(
            counterfactual_participant_format_repair
        )
        self.counterfactual_participant_format_repair_strategy = str(
            counterfactual_participant_format_repair_strategy
        )
        self.counterfactual_participant_max_utterance_tokens = max(
            0, int(counterfactual_participant_max_utterance_tokens)
        )
        self.counterfactual_participant_tokenizer = (
            counterfactual_participant_tokenizer
        )
        self.last_request_meta: dict[str, Any] = {}
        self.attempt_records: list[dict[str, Any]] = []
        self.local_penalty_total = 0.0

    def generate_json(
        self,
        *,
        prompt_id: str,
        system_prompt: str,
        user_prompt: str,
        trace_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        del trace_dir  # PromptTraceRecorder owns optional trace persistence.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        role = _actor_role_for_prompt(prompt_id)
        prompt_kind = _prompt_kind(prompt_id)
        trainable = _is_trainable(role, self.training_mode)
        selected_lora_name = (
            (self.rollout_lora_name or FULL_MODEL_NAME)
            if trainable
            else BASE_LORA_NAME
        )
        last_error: BaseException | None = None

        for retry in range(self.max_parse_retries + 1):
            application_id = f"{self.uid}:{prompt_id}:retry_{retry}"
            request_kwargs: dict[str, Any] = {
                "application_id": application_id,
                "max_tokens": (
                    self.participant_max_output_tokens
                    if role == "participant"
                    and prompt_kind
                    in {"participant_response", "participant_public_message"}
                    else self.max_output_tokens
                ),
            }
            if selected_lora_name == "default":
                hydra_root = Path(os.environ.get("HYDRA_RUN_DIR", ".")).expanduser()
                request_kwargs["lora_request"] = {
                    "lora_name": VLLM_LORA_NAME,
                    "lora_int_id": VLLM_LORA_INT_ID,
                    "lora_path": str((hydra_root / "default").resolve()),
                }
            elif selected_lora_name not in {None, BASE_LORA_NAME, FULL_MODEL_NAME}:
                request_kwargs["lora_name"] = selected_lora_name
            future: Future[ModelOutput] = asyncio.run_coroutine_threadsafe(
                self.rollout_engine.get_model_response(messages, **request_kwargs),
                self.event_loop,
            )
            output = future.result(timeout=self.request_timeout_sec)
            # ``text`` is the authoritative decoded completion.  Qwen2.5's
            # RLLM chat parser can classify ordinary JSON as ``reasoning`` and
            # therefore leave ``content`` empty even though generation was
            # valid; parsing the full decoded text preserves the exact model
            # response across chat-template parsers.
            raw = str(output.text or output.content or "")
            parse_error: BaseException | None = None
            schema_error: str | None = None
            parsed: dict[str, Any] | None = None
            compact_repair_method: str | None = None
            counterfactual_repair_method: str | None = None
            try:
                # Training is deliberately stricter than the ordinary runtime:
                # markdown fences, surrounding prose, arrays, and JSON
                # substrings are format failures even when the evaluation
                # client could recover an object from them.
                parsed = _parse_exact_json_object(raw)
            except BaseException as exc:  # noqa: BLE001
                parse_error = exc
                last_error = exc
                if self.allow_compact_repair:
                    parsed, compact_repair_method = _repair_compact_action(
                        raw, prompt_kind=_prompt_kind(prompt_id)
                    )
                    if parsed is not None:
                        parse_error = None

            if parsed is not None and parse_error is None:
                schema_error = _action_schema_error(
                    parsed,
                    prompt_kind=_prompt_kind(prompt_id),
                    prompt_payload=_prompt_payload(user_prompt),
                )
                if schema_error is not None:
                    last_error = ValueError(schema_error)
            if compact_repair_method is not None:
                # Keep the parsed candidate for diagnostics, but never make a
                # repaired token sequence reward-equivalent to exact JSON.
                schema_error = (
                    "response required compact repair instead of exact JSON: "
                    + compact_repair_method
                )
                last_error = ValueError(schema_error)
            if (
                self.counterfactual_participant_format_repair
                and role == "participant"
                and (parse_error is not None or schema_error is not None)
            ):
                repaired = _repair_participant_format_counterfactual(
                    raw,
                    prompt_kind=_prompt_kind(prompt_id),
                    strategy=(
                        self.counterfactual_participant_format_repair_strategy
                    ),
                    max_utterance_tokens=(
                        self.counterfactual_participant_max_utterance_tokens
                    ),
                    tokenizer=self.counterfactual_participant_tokenizer,
                )
                if repaired is not None:
                    parsed = repaired
                    parse_error = None
                    schema_error = _action_schema_error(
                        repaired,
                        prompt_kind=_prompt_kind(prompt_id),
                        prompt_payload=_prompt_payload(user_prompt),
                    )
                    if schema_error is None:
                        compact_repair_method = None
                        counterfactual_repair_method = (
                            self.counterfactual_participant_format_repair_strategy
                        )
                    else:
                        last_error = ValueError(schema_error)

            penalty = (
                self.invalid_json_penalty
                if parse_error is not None and trainable
                else self.invalid_schema_penalty
                if schema_error is not None and trainable
                else 0.0
            )
            self.local_penalty_total += penalty
            step_info = {
                "episode_id": self.uid,
                "prompt_id": prompt_id,
                "prompt_kind": _prompt_kind(prompt_id),
                "actor_role": role,
                "retry_number": retry,
                "parse_status": (
                    "counterfactual_format_repair"
                    if counterfactual_repair_method is not None
                    else "invalid_schema"
                    if schema_error is not None
                    else "repaired_compact"
                    if compact_repair_method is not None
                    else "ok"
                    if parse_error is None
                    else "invalid_json"
                ),
                "compact_repair_method": compact_repair_method,
                "counterfactual_repair_method": counterfactual_repair_method,
                "counterfactual_repair_max_utterance_tokens": (
                    self.counterfactual_participant_max_utterance_tokens
                    if counterfactual_repair_method is not None
                    else 0
                ),
                "counterfactual_repair_utterance_token_count": (
                    len(
                        self.counterfactual_participant_tokenizer.encode(
                            str((parsed or {}).get("utterance") or ""),
                            add_special_tokens=False,
                        )
                    )
                    if counterfactual_repair_method is not None
                    and self.counterfactual_participant_tokenizer is not None
                    else None
                ),
                "parse_error": (
                    str(parse_error)[:500]
                    if parse_error is not None
                    else schema_error
                )
                if parse_error is not None or schema_error is not None
                else None,
                "raw_output": raw,
                "raw_output_token_count": int(
                    output.completion_length or len(output.completion_ids or [])
                ),
                "train_mask": bool(trainable),
                "training_mode": self.training_mode,
                "policy_route": (
                    "trainable_full_model"
                    if selected_lora_name == FULL_MODEL_NAME
                    else "trainable_adapter"
                    if trainable
                    else "frozen_base"
                ),
                "local_reward_penalty": penalty,
                "finish_reason": output.finish_reason,
                "lora_name": selected_lora_name,
            }
            target_trajectory = (
                self.trainable_trajectory if trainable else self.fixed_trajectory
            )
            target_trajectory.steps.append(
                Step(
                    chat_completions=messages
                    + [
                        {
                            "role": "assistant",
                            "content": raw,
                            "reasoning": output.reasoning,
                        }
                    ],
                    thought=str(output.reasoning or ""),
                    action=parsed,
                    model_response=raw,
                    model_output=output,
                    reward=penalty,
                    info=step_info,
                )
            )
            record = {
                **step_info,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "parsed_response": parsed,
            }
            self.attempt_records.append(record)
            self.last_request_meta = {
                "provider": "rllm_rollout_engine",
                "application_id": application_id,
                "attempt": retry + 1,
                "prompt_tokens": int(
                    output.prompt_length or len(output.prompt_ids or [])
                ),
                "completion_tokens": int(
                    output.completion_length or len(output.completion_ids or [])
                ),
                "finish_reason": output.finish_reason,
                "parse_status": step_info["parse_status"],
            }
            if parsed is not None and schema_error is None:
                return parsed
            if retry < self.max_parse_retries:
                failure_reason = (
                    str(parse_error)
                    if parse_error is not None
                    else str(schema_error)
                )
                output_limit = (
                    self.participant_max_output_tokens
                    if role == "participant"
                    and prompt_kind
                    in {"participant_response", "participant_public_message"}
                    else self.max_output_tokens
                )
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "That response was invalid: "
                            f"{failure_reason}. Return a corrected response as "
                            "exactly one JSON object matching response_format. "
                            f"The entire response, including JSON syntax, must "
                            f"fit within {output_limit} tokens; shorten text if "
                            "needed."
                        ),
                    },
                ]

        assert last_error is not None
        raise last_error
