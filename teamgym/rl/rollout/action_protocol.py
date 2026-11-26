"""Strict action JSON parsing and protocol checks for TeamGym rollouts."""

from __future__ import annotations

import json
from typing import Any


def _prompt_kind(prompt_id: str) -> str:
    return prompt_id.rsplit(".", 1)[-1]


def _prompt_payload(user_prompt: str) -> dict[str, Any]:
    """Return the JSON prompt payload used to define the exact action schema."""

    try:
        payload = json.loads(user_prompt)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_exact_json_object(raw: str) -> dict[str, Any]:
    """Parse an action only when the entire completion is one JSON object."""

    parsed = json.loads(str(raw).strip())
    if not isinstance(parsed, dict):
        raise ValueError("model response must be exactly one top-level JSON object")
    return parsed


def _action_schema_error(
    response: dict[str, Any],
    *,
    prompt_kind: str,
    prompt_payload: dict[str, Any] | None = None,
) -> str | None:
    """Return a compact protocol error after JSON syntax has parsed.

    The ordinary actors intentionally tolerate malformed fields by falling
    back to empty messages or the first legal action.  That behavior is useful
    for evaluation robustness but must not make malformed trainable outputs
    reward-equivalent to valid actions.
    """

    payload = prompt_payload or {}
    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        allowed_keys = {str(key) for key in response_format}
        extra_keys = sorted(set(response) - allowed_keys)
        if extra_keys:
            return (
                "response contains keys outside response_format: "
                + ", ".join(extra_keys)
            )
        required_keys = {
            str(key)
            for key, description in response_format.items()
            if "optional" not in str(description).casefold()
        }
        missing_keys = sorted(required_keys - set(response))
        if missing_keys:
            return (
                "response is missing required response_format keys: "
                + ", ".join(missing_keys)
            )

    if prompt_kind in {"participant_response", "participant_public_message"}:
        utterance = response.get("utterance")
        if not isinstance(utterance, str) or not utterance.strip():
            return "participant reply requires a non-empty top-level string field 'utterance'"
        recommended_next_actor_id = response.get("recommended_next_actor_id")
        allowed_participant_ids = payload.get("allowed_participant_ids")
        if recommended_next_actor_id is not None and (
            not isinstance(recommended_next_actor_id, str)
            or (
                isinstance(allowed_participant_ids, list)
                and recommended_next_actor_id not in allowed_participant_ids
            )
        ):
            return "recommended_next_actor_id must be an allowed participant id"
        action = response.get("action")
        next_actor_id = response.get("next_actor_id")
        if action is not None:
            if action not in {"handoff", "finalize"}:
                return "participant action must be 'handoff' or 'finalize'"
            if action == "handoff" and (
                not isinstance(next_actor_id, str)
                or not next_actor_id.strip()
                or (
                    isinstance(allowed_participant_ids, list)
                    and next_actor_id not in allowed_participant_ids
                )
            ):
                return (
                    "handoff requires next_actor_id from "
                    "allowed_participant_ids"
                )
            if action == "finalize" and next_actor_id is not None:
                return "finalize must omit next_actor_id"
        confidence = response.get("confidence")
        if confidence is not None and confidence not in {"low", "medium", "high"}:
            return "participant confidence must be low, medium, or high"
        return None
    if prompt_kind == "participant_private_think":
        think = response.get("think")
        if not isinstance(think, str) or not think.strip():
            return "participant private stage requires a non-empty top-level string field 'think'"
        return None
    if prompt_kind == "coordinator_query":
        participant_directory = payload.get("participant_directory")
        valid_ids = {
            str(row.get("participant_id"))
            for row in participant_directory
            if isinstance(row, dict) and row.get("participant_id")
        } if isinstance(participant_directory, list) else set()
        compact_target = response.get("target")
        compact_ask = response.get("ask")
        if (
            isinstance(compact_target, str)
            and compact_target.strip()
            and isinstance(compact_ask, str)
            and compact_ask.strip()
        ):
            if valid_ids and compact_target not in valid_ids:
                return "coordinator target must be an id from participant_directory"
            return None
        target_ids = response.get("target_ids")
        utterance = response.get("utterance")
        if (
            isinstance(target_ids, list)
            and target_ids
            and all(isinstance(item, str) and item.strip() for item in target_ids)
            and isinstance(utterance, str)
            and utterance.strip()
        ):
            if response.get("action") != "turn_choose":
                return "coordinator action must equal 'turn_choose'"
            if valid_ids and any(item not in valid_ids for item in target_ids):
                return "coordinator target_ids must come from participant_directory"
            reason = response.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                return "coordinator query requires a non-empty string field 'reason'"
            return None
        return (
            "coordinator query requires compact {'target','ask'} or "
            "full {'target_ids','utterance'} fields"
        )
    if prompt_kind in {
        "coordinator_final_decision",
        "final_synthesis",
        "participant_final_decision",
    }:
        answer = response.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return "final decision requires a non-empty top-level string field 'answer'"
        task = payload.get("task")
        decision_set = task.get("decision_set") if isinstance(task, dict) else None
        if isinstance(decision_set, list) and answer not in decision_set:
            return "final answer must be an id from decision_set"
        compact = isinstance(response_format, dict) and "why" in response_format
        if compact:
            why = response.get("why")
            if not isinstance(why, str) or not why.strip():
                return "compact final decision requires a non-empty string field 'why'"
        else:
            if response.get("action") != "final_decision":
                return "final action must equal 'final_decision'"
            if (
                not isinstance(response_format, dict)
                or "written_answer" not in response_format
            ):
                rationale = response.get("rationale")
                if not isinstance(rationale, str) or not rationale.strip():
                    return (
                        "final decision requires a non-empty string field "
                        "'rationale'"
                    )
            confidence = response.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                return "final confidence must be a number between 0 and 1"
        if "written_answer" in response and (
            not isinstance(response.get("written_answer"), str)
            or not str(response.get("written_answer")).strip()
        ):
            return "written_answer must be a non-empty string"
    return None


def _compact_action_has_required_field(
    fields: dict[str, Any], prompt_kind: str
) -> bool:
    if "final" in prompt_kind:
        return bool(str(fields.get("answer") or "").strip())
    return bool(str(fields.get("target") or "").strip())
