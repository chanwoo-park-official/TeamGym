"""LLM-backed actors for fully collaborative interaction episodes."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from teamgym.llm.openai_client import OpenAIJsonClient

from .actor_utils import (
    count_tokens,
    distributed_truncate_token_segments,
    participant_by_id,
    participant_ids,
    participant_private_context_units,
    truncate_tokens,
    truncate_tokens_strict,
)
from .public_reading import ParticipantPublicReader
from .types import BudgetSpec
from .rui_context import model_common_information


class PromptTraceRecorder:
    def __init__(self, trace_dir: str | Path | None = None):
        self.trace_dir = Path(trace_dir).expanduser().resolve() if trace_dir else None
        self.prompt_records: list[dict[str, Any]] = []

    def record(
        self,
        *,
        prompt_id: str,
        actor_id: str,
        actor_role: str,
        prompt_kind: str,
        system_prompt: str,
        user_prompt: str,
        response: dict[str, Any],
        request_meta: dict[str, Any] | None,
    ) -> None:
        record = {
            "prompt_id": prompt_id,
            "actor_id": actor_id,
            "actor_role": actor_role,
            "prompt_kind": prompt_kind,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response,
            "request_meta": request_meta or {},
        }
        self.prompt_records.append(record)
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.trace_dir / f"{_slug(prompt_id)}.prompt_record.json"
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class LLMParticipantActor:
    def __init__(
        self,
        client: OpenAIJsonClient,
        recorder: PromptTraceRecorder,
        *,
        role_name_mode: str = "hidden",
        include_confidence: bool = False,
        observation_mode: str = "full_public",
    ):
        if observation_mode not in {"full_public", "query_private_only"}:
            raise ValueError(
                "observation_mode must be 'full_public' or 'query_private_only'"
            )
        self.client = client
        self.recorder = recorder
        self.role_name_mode = role_name_mode
        self.include_confidence = include_confidence
        self.observation_mode = observation_mode
        self.public_reader = ParticipantPublicReader()

    def _participant_chat_history(
        self,
        *,
        datapoint: dict[str, Any],
        participant_id: str,
        state: dict[str, Any],
        request: dict[str, Any] | None,
        budgets: BudgetSpec,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self.public_reader.observe(
            datapoint_id=str(datapoint.get("instance_id") or ""),
            participant_id=participant_id,
            messages=[row for row in state.get("messages", []) if isinstance(row, dict)],
            request=request,
            policy=budgets.participant_public_read_policy,
            token_budget=budgets.participant_public_read_token_budget,
            seed=budgets.participant_public_read_seed,
            charge_request=budgets.public_read_charge_current_message,
        )

    def respond(
        self,
        *,
        datapoint: dict[str, Any],
        participant_id: str,
        state: dict[str, Any],
        request: dict[str, Any] | None,
        budgets: BudgetSpec,
        allow_participant_handoff: bool = False,
        allow_participant_finalize: bool = False,
    ) -> dict[str, Any]:
        participants = participant_by_id(datapoint)
        participant = participants[participant_id]
        role_name = _visible_role_name(participant, participant_id, self.role_name_mode)
        prompt_id = f"{datapoint.get('instance_id')}.turn_{state.get('turn')}.{participant_id}.participant_response"
        system_prompt = _participant_system_prompt(
            budgets,
            participant_count=len(participant_ids(datapoint)),
            no_coordinator=not isinstance(request, dict),
            observation_mode=self.observation_mode,
        )
        if self.observation_mode == "query_private_only":
            chat_history: list[dict[str, Any]] = []
            reading_metrics: dict[str, Any] = {}
        else:
            chat_history, reading_metrics = self._participant_chat_history(
                datapoint=datapoint,
                participant_id=participant_id,
                state=state,
                request=request,
                budgets=budgets,
            )
        response_format = {"utterance": "string public message"}
        if allow_participant_finalize:
            response_format = {
                "action": "handoff or finalize",
                "utterance": "string public message",
                "next_actor_id": (
                    "optional participant id from allowed_participant_ids; "
                    "required exactly when action is handoff"
                ),
            }
        elif allow_participant_handoff:
            response_format["recommended_next_actor_id"] = "optional participant id from allowed_participant_ids"
        allowed_participant_ids = participant_ids(datapoint) if allow_participant_handoff else None
        if self.observation_mode == "query_private_only":
            prompt_payload = {
                "participant_mode": "respond_to_private_source_question",
                "current_question": _participant_question(request),
                "private_source_material": _private_context_payload(
                    datapoint, participant, budgets
                ),
                "response_format": response_format,
            }
        else:
            prompt_payload = {
                "participant_mode": "respond_to_request" if isinstance(request, dict) else "scheduled_turn",
                "turn_instruction": _participant_turn_instruction(request),
                "task": _public_task_payload(datapoint),
                "participant": _participant_identity_payload(participant_id, role_name),
                "participant_directory": _participant_directory_payload(
                    datapoint, self.role_name_mode
                ),
                "private_source_material": _private_context_payload(datapoint, participant, budgets),
                "chat_history": chat_history,
                "response_format": response_format,
            }
        if allowed_participant_ids is not None:
            prompt_payload["allowed_participant_ids"] = allowed_participant_ids
        if self.include_confidence:
            response_format["confidence"] = "optional low|medium|high"
        response = _generate_and_record(
            client=self.client,
            recorder=self.recorder,
            prompt_id=prompt_id,
            actor_id=participant_id,
            actor_role="participant",
            prompt_kind="participant_response",
            system_prompt=system_prompt,
            prompt_payload=prompt_payload,
        )
        utterance = str(response.get("utterance") or "").strip()
        participant_action = (
            response.get("action") if allow_participant_finalize else None
        )
        if allow_participant_finalize:
            participant_action = (
                "finalize" if participant_action == "finalize" else "handoff"
            )
            next_actor = response.get("next_actor_id")
        else:
            next_actor = response.get("recommended_next_actor_id") if allow_participant_handoff else None
        if next_actor not in set(participant_ids(datapoint)):
            next_actor = None
        message = {
            "speaker_id": participant_id,
            "speaker_role": "participant",
            "content": truncate_tokens(utterance, budgets.participant_utterance_token_budget),
            "disclosed_rui_ids": [],
        }
        if budgets.participant_public_read_policy != "unlimited":
            message.update(reading_metrics)
        if allow_participant_handoff:
            message["recommended_next_actor_id"] = next_actor
        if allow_participant_finalize:
            message["participant_action"] = participant_action
        analysis = _utterance_analysis(utterance, datapoint)
        if analysis["kind"] != "message":
            message["utterance_analysis"] = analysis
        if self.include_confidence and response.get("confidence") is not None:
            message["confidence"] = response.get("confidence")
        if role_name is not None:
            message["role_name"] = role_name
        return message


class LLMTwoStageParticipantActor(LLMParticipantActor):
    """Participant with a private scratchpad stage before its public message."""

    def respond(
        self,
        *,
        datapoint: dict[str, Any],
        participant_id: str,
        state: dict[str, Any],
        request: dict[str, Any] | None,
        budgets: BudgetSpec,
        allow_participant_handoff: bool = False,
        allow_participant_finalize: bool = False,
    ) -> dict[str, Any]:
        participants = participant_by_id(datapoint)
        participant = participants[participant_id]
        role_name = _visible_role_name(participant, participant_id, self.role_name_mode)
        prompt_prefix = f"{datapoint.get('instance_id')}.turn_{state.get('turn')}.{participant_id}"
        chat_history, reading_metrics = self._participant_chat_history(
            datapoint=datapoint,
            participant_id=participant_id,
            state=state,
            request=request,
            budgets=budgets,
        )

        think_response = _generate_and_record(
            client=self.client,
            recorder=self.recorder,
            prompt_id=f"{prompt_prefix}.participant_private_think",
            actor_id=participant_id,
            actor_role="participant",
            prompt_kind="participant_private_think",
            system_prompt=_participant_private_think_system_prompt(budgets),
            prompt_payload={
                "task": _public_task_payload(datapoint),
                "participant": _participant_identity_payload(participant_id, role_name),
                "private_source_material": _private_context_payload(datapoint, participant, budgets),
                "chat_history": chat_history,
                "turn_instruction": (
                    "Prepare a private working note for your next public message. Identify the source facts, "
                    "rules, and implications that matter for the current request."
                ),
                "response_format": {"think": "private working note; not visible to other actors"},
            },
        )
        raw_think = str(think_response.get("think") or "").strip()
        private_think = truncate_tokens_strict(raw_think, budgets.participant_reasoning_token_budget)

        response_format = {"utterance": "string public message"}
        if allow_participant_finalize:
            response_format = {
                "action": "handoff or finalize",
                "utterance": "string public message",
                "next_actor_id": (
                    "optional participant id from allowed_participant_ids; "
                    "required exactly when action is handoff"
                ),
            }
        elif allow_participant_handoff:
            response_format["recommended_next_actor_id"] = (
                "optional participant id from allowed_participant_ids"
            )
        if self.include_confidence:
            response_format["confidence"] = "optional low|medium|high"
        prompt_payload: dict[str, Any] = {
            "participant_mode": "respond_to_request" if isinstance(request, dict) else "scheduled_turn",
            "turn_instruction": (
                "Write the next public response using the private working note and chat history. "
                "The private source is unavailable in this stage, and the private working note must not be quoted as a separate field."
            ),
            "task": _public_task_payload(datapoint),
            "participant": _participant_identity_payload(participant_id, role_name),
            "participant_directory": _participant_directory_payload(
                datapoint, self.role_name_mode
            ),
            "private_think": private_think,
            "chat_history": chat_history,
            "response_format": response_format,
        }
        if allow_participant_handoff:
            prompt_payload["allowed_participant_ids"] = participant_ids(datapoint)
        response = _generate_and_record(
            client=self.client,
            recorder=self.recorder,
            prompt_id=f"{prompt_prefix}.participant_public_message",
            actor_id=participant_id,
            actor_role="participant",
            prompt_kind="participant_public_message",
            system_prompt=_participant_public_message_system_prompt(
                budgets,
                participant_count=len(participant_ids(datapoint)),
                no_coordinator=not isinstance(request, dict),
            ),
            prompt_payload=prompt_payload,
        )

        utterance = str(response.get("utterance") or "").strip()
        participant_action = (
            response.get("action") if allow_participant_finalize else None
        )
        if allow_participant_finalize:
            participant_action = (
                "finalize" if participant_action == "finalize" else "handoff"
            )
            next_actor = response.get("next_actor_id")
        else:
            next_actor = response.get("recommended_next_actor_id") if allow_participant_handoff else None
        if next_actor not in set(participant_ids(datapoint)):
            next_actor = None
        message = {
            "speaker_id": participant_id,
            "speaker_role": "participant",
            "content": truncate_tokens_strict(utterance, budgets.participant_utterance_token_budget),
            "disclosed_rui_ids": [],
            "private_think_token_count": count_tokens(private_think),
            "private_think_raw_token_count": count_tokens(raw_think),
            "private_think_truncated": count_tokens(raw_think) > budgets.participant_reasoning_token_budget,
            "public_message_raw_token_count": count_tokens(utterance),
            "public_message_truncated": count_tokens(utterance) > budgets.participant_utterance_token_budget,
        }
        if budgets.participant_public_read_policy != "unlimited":
            message.update(reading_metrics)
        if allow_participant_handoff:
            message["recommended_next_actor_id"] = next_actor
        if allow_participant_finalize:
            message["participant_action"] = participant_action
        analysis = _utterance_analysis(utterance, datapoint)
        if analysis["kind"] != "message":
            message["utterance_analysis"] = analysis
        if self.include_confidence and response.get("confidence") is not None:
            message["confidence"] = response.get("confidence")
        if role_name is not None:
            message["role_name"] = role_name
        return message


def _generate_and_record(
    *,
    client: OpenAIJsonClient,
    recorder: PromptTraceRecorder,
    prompt_id: str,
    actor_id: str,
    actor_role: str,
    prompt_kind: str,
    system_prompt: str,
    prompt_payload: dict[str, Any],
) -> dict[str, Any]:
    user_prompt = json.dumps(prompt_payload, indent=2, ensure_ascii=False)
    response = client.generate_json(
        prompt_id=prompt_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        trace_dir=recorder.trace_dir,
    )
    recorder.record(
        prompt_id=prompt_id,
        actor_id=actor_id,
        actor_role=actor_role,
        prompt_kind=prompt_kind,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response,
        request_meta=dict(client.last_request_meta),
    )
    return response


def _utterance_analysis(utterance: str, datapoint: dict[str, Any]) -> dict[str, Any]:
    if not utterance:
        return {"kind": "empty"}
    if _is_bare_decision_id(utterance, datapoint):
        return {"kind": "decision_id_only"}
    return {"kind": "message"}


def _is_bare_decision_id(text: str, datapoint: dict[str, Any]) -> bool:
    normalized = text.strip().casefold().strip("\"'` .,:;")
    decision_ids = {
        str(item).casefold()
        for item in datapoint.get("decision_set", [])
        if isinstance(item, str)
    }
    return normalized in decision_ids


class LLMCoordinatorActor:
    def __init__(
        self,
        client: OpenAIJsonClient,
        recorder: PromptTraceRecorder,
        *,
        role_name_mode: str = "hidden",
        compact_output: bool = False,
        final_compact_output: bool | None = None,
        participants_query_private_only: bool = False,
        include_compact_envelope_example: bool = False,
    ):
        self.client = client
        self.recorder = recorder
        self.role_name_mode = role_name_mode
        self.compact_output = bool(compact_output)
        self.final_compact_output = (
            self.compact_output
            if final_compact_output is None
            else bool(final_compact_output)
        )
        self.participants_query_private_only = bool(
            participants_query_private_only
        )
        self.include_compact_envelope_example = bool(
            include_compact_envelope_example
        )
        self.public_reader = ParticipantPublicReader()

    def _chat_history(
        self,
        *,
        datapoint: dict[str, Any],
        state: dict[str, Any],
        budgets: BudgetSpec,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        if (
            budgets.public_read_scope != "all_actors"
            or budgets.participant_public_read_policy == "unlimited"
        ):
            return _public_transcript_payload(state), None
        return self.public_reader.observe(
            datapoint_id=str(datapoint.get("instance_id") or ""),
            participant_id="coordinator",
            messages=[row for row in state.get("messages", []) if isinstance(row, dict)],
            request=None,
            policy=budgets.participant_public_read_policy,
            token_budget=budgets.participant_public_read_token_budget,
            seed=budgets.participant_public_read_seed,
            prioritize_latest_other=budgets.public_read_charge_current_message,
        )

    def choose_targets(
        self,
        *,
        datapoint: dict[str, Any],
        state: dict[str, Any],
        max_targets: int,
        budgets: BudgetSpec,
    ) -> dict[str, Any]:
        prompt_id = f"{datapoint.get('instance_id')}.turn_{state.get('turn')}.coordinator_query"
        system_prompt = _coordinator_system_prompt(
            budgets,
            mode="query",
            compact_output=(
                self.compact_output
                and self.include_compact_envelope_example
            ),
            participants_query_private_only=(
                self.participants_query_private_only
            ),
        )
        valid_ids = participant_ids(datapoint)
        chat_history, reading_metrics = self._chat_history(
            datapoint=datapoint,
            state=state,
            budgets=budgets,
        )
        response_format = (
            {
                "target": "one id from participant_directory",
                "ask": (
                    "self-contained source question in at most 24 words"
                    if self.participants_query_private_only
                    else "public question in at most 12 words"
                ),
            }
            if self.compact_output
            else {
                "action": "turn_choose",
                "target_ids": _target_ids_schema(valid_ids, max_targets),
                "utterance": "string public coordinator message",
                "reason": "short public reason for the request",
            }
        )
        user_prompt = json.dumps(
            {
                "task": _public_task_payload(datapoint),
                "participant_directory": _participant_directory_payload(datapoint, self.role_name_mode),
                "chat_history": chat_history,
                "turn_instruction": (
                    "It is your turn to speak. Coordinate all participants through the public chat. "
                    "Provide a public response to the last message in chat_history, or ask target participant(s) "
                    "to respond next."
                ),
                "response_format": response_format,
            },
            indent=2,
            ensure_ascii=False,
        )
        response = self.client.generate_json(
            prompt_id=prompt_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_dir=self.recorder.trace_dir,
        )
        self.recorder.record(
            prompt_id=prompt_id,
            actor_id="coordinator",
            actor_role="coordinator",
            prompt_kind="coordinator_query",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            request_meta=dict(self.client.last_request_meta),
        )
        compact_target = response.get("target")
        response_target_ids = response.get("target_ids", [])
        if self.compact_output and isinstance(compact_target, str):
            response_target_ids = [compact_target]
        target_ids = [
            str(item)
            for item in response_target_ids
            if isinstance(item, str) and item in valid_ids
        ][: max(1, int(max_targets))]
        if not target_ids:
            target_ids = valid_ids[: max(1, int(max_targets))]
        utterance = str(
            response.get("utterance")
            or response.get("ask")
            or response.get("reason")
            or ""
        ).strip()
        if not utterance:
            utterance = (
                "What fact does your private source establish about the task? "
                "Include the controlling condition or exception."
                if self.participants_query_private_only
                else "Please share the source evidence most relevant to choosing among the listed decisions."
            )
        query = {
            "action": "turn_choose",
            "speaker_id": "coordinator",
            "speaker_role": "coordinator",
            "content": truncate_tokens(f"Coordinator: {utterance}", budgets.coordinator_utterance_token_budget),
            "target_ids": target_ids,
            "reason": response.get("reason"),
        }
        if reading_metrics is not None:
            query.update(reading_metrics)
        return query

    def decide(
        self,
        *,
        datapoint: dict[str, Any],
        state: dict[str, Any],
        budgets: BudgetSpec,
    ) -> dict[str, Any]:
        public_transcript, reading_metrics = self._chat_history(
            datapoint=datapoint,
            state=state,
            budgets=budgets,
        )
        decision = _llm_final_decision(
            client=self.client,
            recorder=self.recorder,
            datapoint=datapoint,
            state=state,
            budgets=budgets,
            actor_id="coordinator",
            actor_role="coordinator",
            prompt_kind="coordinator_final_decision",
            public_transcript=public_transcript,
            compact_output=self.final_compact_output,
            include_compact_envelope_example=(
                self.include_compact_envelope_example
            ),
            role_name_mode=self.role_name_mode,
        )
        if reading_metrics is not None:
            decision.update(reading_metrics)
        return decision


class LLMFinalDecisionActor:
    def __init__(
        self,
        client: OpenAIJsonClient,
        recorder: PromptTraceRecorder,
        *,
        compact_output: bool = False,
        role_name_mode: str = "hidden",
        participant_prompt_kind: str = "final_synthesis",
    ):
        self.client = client
        self.recorder = recorder
        self.compact_output = bool(compact_output)
        self.role_name_mode = role_name_mode
        self.participant_prompt_kind = participant_prompt_kind

    def decide(
        self,
        *,
        datapoint: dict[str, Any],
        state: dict[str, Any],
        actor_id: str,
        actor_role: str,
        budgets: BudgetSpec,
    ) -> dict[str, Any]:
        return _llm_final_decision(
            client=self.client,
            recorder=self.recorder,
            datapoint=datapoint,
            state=state,
            budgets=budgets,
            actor_id=actor_id,
            actor_role=actor_role,
            prompt_kind=(
                self.participant_prompt_kind
                if actor_role == "participant"
                else "final_synthesis"
            ),
            compact_output=self.compact_output,
            role_name_mode=self.role_name_mode,
        )


def _llm_final_decision(
    *,
    client: OpenAIJsonClient,
    recorder: PromptTraceRecorder,
    datapoint: dict[str, Any],
    state: dict[str, Any],
    budgets: BudgetSpec,
    actor_id: str,
    actor_role: str,
    prompt_kind: str,
    public_transcript: list[dict[str, Any]] | None = None,
    compact_output: bool = False,
    include_compact_envelope_example: bool = False,
    role_name_mode: str = "hidden",
) -> dict[str, Any]:
    prompt_id = f"{datapoint.get('instance_id')}.turn_{state.get('turn')}.{actor_id}.{prompt_kind}"
    system_prompt = _final_decision_system_prompt(
        budgets,
        actor_role=actor_role,
        compact_output=(
            compact_output and include_compact_envelope_example
        ),
    )
    response_format = (
        {
            "answer": "one id from decision_set",
            "why": "rationale in at most 12 words",
        }
        if compact_output
        else {
            "action": "final_decision",
            "answer": "one decision id from decision_set",
            "rationale": "short rationale using only information available in this prompt",
            "confidence": "number between 0 and 1",
        }
    )
    available_information = "task information and public transcript"
    prompt_payload: dict[str, Any] = {
        "task": _public_task_payload(datapoint),
        "public_transcript": (
            public_transcript if public_transcript is not None else _public_transcript_payload(state)
        ),
    }
    if role_name_mode != "hidden":
        prompt_payload["participant_directory"] = _participant_directory_payload(
            datapoint, role_name_mode
        )
    if actor_role == "participant":
        participant = participant_by_id(datapoint)[actor_id]
        role_name = _visible_role_name(
            participant, actor_id, role_name_mode
        )
        prompt_payload["participant"] = _participant_identity_payload(
            actor_id, role_name
        )
        prompt_payload["private_source_material"] = _private_context_payload(datapoint, participant, budgets)
        available_information += ", together with your own private source material"
    turn_instruction = f"Choose the final decision using the {available_information}."
    if _is_short_answer_judged(datapoint):
        if not compact_output:
            # A separate rationale duplicates a potentially long written
            # answer and consumes the small-model completion budget. The
            # written answer is itself the grounded rationale for this task.
            response_format.pop("rationale", None)
        response_format["written_answer"] = (
            "required conclusion in at most 12 words"
            if compact_output
            else "short written conclusion with the required decision steps, using only information available in this prompt"
        )
        turn_instruction = (
            f"Write the final short-answer conclusion using the {available_information}. "
            "Also set answer to a valid compatibility decision id from decision_set."
        )
    prompt_payload["turn_instruction"] = turn_instruction
    prompt_payload["response_format"] = response_format
    user_prompt = json.dumps(prompt_payload, indent=2, ensure_ascii=False)
    response = client.generate_json(
        prompt_id=prompt_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        trace_dir=recorder.trace_dir,
    )
    recorder.record(
        prompt_id=prompt_id,
        actor_id=actor_id,
        actor_role=actor_role,
        prompt_kind=prompt_kind,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response,
        request_meta=dict(client.last_request_meta),
    )
    answer = response.get("answer")
    if answer not in set(datapoint.get("decision_set", [])):
        answer = datapoint.get("decision_set", [""])[0]
    confidence = response.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "action": "final_decision",
        "actor_id": actor_id,
        "actor_role": actor_role,
        "answer": answer,
        "rationale": truncate_tokens(
            str(response.get("rationale") or response.get("why") or ""),
            budgets.final_rationale_token_budget,
        ),
        "written_answer": truncate_tokens(str(response.get("written_answer") or ""), budgets.final_rationale_token_budget),
        "confidence": confidence,
    }


def _participant_system_prompt(
    budgets: BudgetSpec,
    *,
    participant_count: int | None = None,
    no_coordinator: bool = False,
    observation_mode: str = "full_public",
) -> str:
    if observation_mode == "query_private_only":
        return "\n".join(
            [
                "You are a private-source participant in a distributed decision task.",
                "You can see only the coordinator's current question and your own private source material.",
                "You cannot see the task, answer choices, option labels, public transcript, or other participants' messages.",
                "Answer only what your private source supports. Do not guess the final answer or invent missing facts.",
                "Report the decision-relevant fact and preserve parties, modality, numbers, conditions, and exceptions when present.",
                "Prefer a short controlling source phrase over a generic interpretation.",
                _reasoning_budget_line(budgets.participant_reasoning_token_budget),
                f"Keep the public utterance at or under {budgets.participant_utterance_token_budget} tokens.",
                "Return exactly one JSON object with only the keys requested in response_format.",
                "Do not repeat the input JSON, current_question, or private_source_material.",
                "Return JSON only and follow response_format exactly.",
            ]
        )
    lines = [
        "You are a participant in this decision-making task.",
        'The input JSON has a "chat_history" key, which is the public chat you are currently participating in.',
        "When it is your turn, provide your next public response for that chat.",
        "Your public message must contain more than a bare option label.",
        "The final decision is made after the communication phase.",
        "Use your public message to share what your available information supports or rules out.",
        "You may mention an option label if useful, but do not present it as a final answer.",
    ]
    if no_coordinator:
        count = max(1, int(participant_count or 1))
        lines.append(
            f"You are one of {count} participants working together through the public chat to solve this task."
        )
    lines.extend(
        [
            _reasoning_budget_line(budgets.participant_reasoning_token_budget),
            f"Keep the public utterance at or under {budgets.participant_utterance_token_budget} tokens.",
            "Return exactly one JSON object with only the keys requested in response_format.",
            "Do not repeat the input JSON, task, options, chat_history, or private_source_material.",
            "Return JSON only and follow response_format exactly.",
        ]
    )
    return "\n".join(lines)


def _participant_private_think_system_prompt(budgets: BudgetSpec) -> str:
    return "\n".join(
        [
            "You are in the private deliberation stage of a distributed decision-making task.",
            "The working note is private and is not added to the public transcript.",
            f"Keep the private working note at or under {budgets.participant_reasoning_token_budget} tokens.",
            "Return exactly one JSON object with only the keys requested in response_format.",
            "Do not repeat the full input JSON, task, chat history, or source material.",
            "Return JSON only and follow response_format exactly.",
        ]
    )


def _participant_public_message_system_prompt(
    budgets: BudgetSpec,
    *,
    participant_count: int,
    no_coordinator: bool,
) -> str:
    lines = [
        "You are in the public-message stage of a distributed decision-making task.",
        "Use the supplied private working note to provide your next public response.",
        "Do not reveal or label private reasoning; return only the requested public JSON fields.",
        "Your public message must contain more than a bare option label.",
        "Use the public message to share what your information supports or rules out.",
    ]
    if no_coordinator:
        lines.append(
            f"You are one of {max(1, int(participant_count))} participants working together through the public chat."
        )
    lines.extend(
        [
            f"Keep the public utterance at or under {budgets.participant_utterance_token_budget} tokens.",
            "Return exactly one JSON object with only the keys requested in response_format.",
            "Do not repeat the input JSON, task, chat history, or private working note.",
            "Return JSON only and follow response_format exactly.",
        ]
    )
    return "\n".join(lines)


def _participant_turn_instruction(request: dict[str, Any] | None) -> str:
    if isinstance(request, dict):
        return "It is your turn to speak. Provide a public response to the last message in chat_history."
    return "It is your turn to speak. Provide a public response using private_source_material and chat_history."


def _participant_question(request: dict[str, Any] | None) -> str:
    if not isinstance(request, dict):
        return (
            "What decision-relevant fact does your private source establish? "
            "Include any controlling condition or exception."
        )
    question = str(request.get("content") or "").strip()
    return question or (
        "What decision-relevant fact does your private source establish? "
        "Include any controlling condition or exception."
    )


def _coordinator_system_prompt(
    budgets: BudgetSpec,
    *,
    mode: str,
    compact_output: bool = False,
    participants_query_private_only: bool = False,
) -> str:
    if mode == "query":
        body = [
            'The input JSON has a "chat_history" key, which is the public chat you are coordinating.',
            "When it is your turn, provide your next public coordinator response for that chat.",
            "Choose which participant(s) should speak next and ask a concrete public question.",
            *(
                [
                    "Participants can see only your current question and their own private source material.",
                    "They cannot see the task, answer choices, option definitions, labels, transcript, or other messages.",
                    "Ask a self-contained source-grounded question that spells out the legal or factual issue.",
                    "Never refer to an option or decision label, and never ask which option is correct.",
                    "Ask for the controlling fact, including a relevant party, condition, number, or exception.",
                ]
                if participants_query_private_only
                else []
            ),
            _reasoning_budget_line(budgets.coordinator_reasoning_token_budget),
            f"Keep the public coordinator utterance at or under {budgets.coordinator_utterance_token_budget} tokens.",
            *(
                [
                    'Required envelope example: {"target":"<id from participant_directory>",'
                    '"ask":"<self-contained source question>"}. Replace both angle-bracket placeholders.'
                ]
                if compact_output
                else []
            ),
        ]
    else:
        body = [
            "Choose the final decision using the task information and public transcript.",
            _reasoning_budget_line(budgets.final_reasoning_token_budget),
            f"Keep the final rationale at or under {budgets.final_rationale_token_budget} tokens.",
            *(
                [
                    'Required envelope example: {"answer":"<id from decision_set>",'
                    '"why":"<short rationale>"}. Replace both angle-bracket placeholders.'
                ]
                if compact_output
                else []
            ),
        ]
    return "\n".join(
        [
            "You are the coordinator for this decision-making task.",
            *body,
            "Return exactly one JSON object with only the keys requested in response_format.",
            "Do not repeat the input JSON, task, options, chat_history, source material, or public transcript.",
            "Return JSON only and follow response_format exactly.",
        ]
    )


def _final_decision_system_prompt(
    budgets: BudgetSpec,
    *,
    actor_role: str,
    compact_output: bool = False,
) -> str:
    if actor_role != "participant":
        return _coordinator_system_prompt(
            budgets,
            mode="final",
            compact_output=compact_output,
        )
    return "\n".join(
        [
            "You are the designated participant responsible for the group's final decision.",
            "Use the public transcript and your own private source material; you cannot access other participants' private material.",
            _reasoning_budget_line(budgets.final_reasoning_token_budget),
            f"Keep the final rationale at or under {budgets.final_rationale_token_budget} tokens.",
            "Return exactly one JSON object with only the keys requested in response_format.",
            "Do not repeat the input JSON, task, source material, or public transcript.",
            "Return JSON only and follow response_format exactly.",
        ]
    )


def _reasoning_budget_line(token_budget: int) -> str:
    if int(token_budget) <= 0:
        return "Return only the requested public JSON fields."
    return f"Use at most {int(token_budget)} private reasoning tokens before answering; return only the requested public JSON fields."


def _public_task_payload(datapoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_description": datapoint.get("task_description"),
        "decision_set": datapoint.get("decision_set"),
        "decision_option_definitions": datapoint.get("decision_option_definitions"),
        "initial_common_information": _strip_dataset_metadata(model_common_information(datapoint)),
    }


def _is_short_answer_judged(datapoint: dict[str, Any]) -> bool:
    task = datapoint.get("task_description") if isinstance(datapoint.get("task_description"), dict) else {}
    dataset_specific = datapoint.get("dataset_specific") if isinstance(datapoint.get("dataset_specific"), dict) else {}
    answer_format = task.get("answer_format") if isinstance(task.get("answer_format"), dict) else {}
    return (
        task.get("answer_type") == "short_answer_judged"
        or dataset_specific.get("answer_type") == "short_answer_judged"
        or answer_format.get("type") == "free_response"
        or datapoint.get("decision_set") == ["free_response"]
    )


def _target_ids_schema(valid_ids: list[str], max_targets: int) -> str:
    limit = max(1, int(max_targets))
    choices = " | ".join(valid_ids)
    example_ids = valid_ids[:limit] if valid_ids else ["participant_id"]
    example = json.dumps(example_ids, ensure_ascii=False)
    choices_text = f" Choose from: {choices}." if choices else ""
    if limit == 1:
        return f"list containing exactly one participant id to ask next.{choices_text} Example: {example}."
    return f"list containing one to {limit} participant ids to ask next.{choices_text} Example: {example}."


def _strip_dataset_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_dataset_metadata(item)
            for key, item in value.items()
            if "dataset" not in str(key).casefold()
        }
    if isinstance(value, list):
        return [_strip_dataset_metadata(item) for item in value]
    return value


def _private_context_payload(
    datapoint: dict[str, Any],
    participant: dict[str, Any],
    budgets: BudgetSpec,
) -> str:
    units = participant_private_context_units(datapoint, participant)
    segments = [
        (str(unit.get("text") or "").strip(), bool(unit.get("hidden_contains_rui_ids")))
        for unit in units
        if str(unit.get("text") or "").strip()
    ]
    return distributed_truncate_token_segments(segments, budgets.participant_context_token_budget)


def _participant_directory_payload(datapoint: dict[str, Any], role_name_mode: str) -> list[dict[str, str]]:
    participants = participant_by_id(datapoint)
    rows: list[dict[str, str]] = []
    for participant_id in participant_ids(datapoint):
        row = {"participant_id": participant_id}
        role_name = _visible_role_name(participants.get(participant_id, {}), participant_id, role_name_mode)
        if role_name is not None:
            row["role_name"] = role_name
        rows.append(row)
    return rows


def _participant_identity_payload(participant_id: str, role_name: str | None) -> dict[str, str]:
    payload = {"participant_id": participant_id}
    if role_name is not None:
        payload["role_name"] = role_name
    return payload


def _visible_role_name(participant: dict[str, Any], participant_id: str, role_name_mode: str) -> str | None:
    if role_name_mode == "hidden":
        return None
    if role_name_mode == "participant_id":
        return participant_id
    if role_name_mode == "generic":
        return "participant"
    return str(participant.get("role_name") or participant.get("public_role") or participant_id)


def _public_transcript_payload(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in state.get("messages", []):
        if not isinstance(message, dict):
            continue
        row = {
            "turn": message.get("turn"),
            "speaker_id": message.get("speaker_id"),
            "speaker_role": message.get("speaker_role"),
            "content": message.get("content"),
        }
        if message.get("role_name") is not None:
            row["role_name"] = message.get("role_name")
        rows.append(row)
    return rows


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)[:180]
