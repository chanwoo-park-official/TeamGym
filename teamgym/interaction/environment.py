"""Communication-structure runtime for normalized fully collaborative datapoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from teamgym.core.dataset import datapoint_variant

from .actor_utils import participant_ids
from .structures import CommunicationStructure
from .types import BudgetSpec


EPISODE_SCHEMA = "teamgym_episode"


def run_episode(
    datapoint: dict[str, Any],
    communication_structure: CommunicationStructure,
    *,
    budgets: BudgetSpec | None = None,
    participant_actor: Any | None = None,
    coordinator_actor: Any | None = None,
    final_actor: Any | None = None,
    initial_participant_id: str | None = None,
) -> dict[str, Any]:
    structure = communication_structure.checked()
    budgets = (budgets or BudgetSpec()).checked()
    _validate_explicit_actors(
        structure=structure,
        participant_actor=participant_actor,
        coordinator_actor=coordinator_actor,
        final_actor=final_actor,
    )
    prompted_actor_active = any(
        _uses_prompt_trace(actor)
        for actor in (participant_actor, coordinator_actor, final_actor)
    )

    valid_participant_ids = participant_ids(datapoint)
    if initial_participant_id is not None and initial_participant_id not in valid_participant_ids:
        raise ValueError(
            f"Unknown initial participant id: {initial_participant_id!r}"
        )
    first_speaker_id = initial_participant_id or _first_participant_id(datapoint)
    state: dict[str, Any] = {
        "turn": 0,
        "public_rui_ids": set(),
        "messages": [],
        "actions": [],
        "last_speaker_id": None,
        "next_speaker_id": first_speaker_id,
        "initial_speaker_id": first_speaker_id,
        "participant_requested_finalize": False,
        "final_participant_id": None,
    }
    turns: list[dict[str, Any]] = []
    max_communication_turns = max(0, budgets.turn_budget - 1)
    stopped_reason = "turn_budget_exhausted"

    for _ in range(max_communication_turns):
        state["turn"] = int(state["turn"]) + 1
        turn = _run_communication_turn(
            datapoint=datapoint,
            structure=structure,
            state=state,
            budgets=budgets,
            participant_actor=participant_actor,
            coordinator_actor=coordinator_actor,
        )
        turns.append(turn)
        if state.get("participant_requested_finalize"):
            stopped_reason = "participant_requested_finalize"
            break
        if turn["new_disclosed_rui_ids"]:
            stopped_reason = "turn_budget_exhausted"
        elif turns and not prompted_actor_active:
            stopped_reason = "no_new_public_information"
            break
        if (
            structure.turn_allocation_rule == "simultaneous_all"
            and not prompted_actor_active
        ):
            stopped_reason = "no_remaining_simultaneous_turns"
            break

    final_turn_number = min(budgets.turn_budget, int(state["turn"]) + 1)
    state["turn"] = final_turn_number
    final_decision = _make_final_decision(
        datapoint=datapoint,
        structure=structure,
        state=state,
        budgets=budgets,
        coordinator_actor=coordinator_actor,
        final_actor=final_actor,
    )
    turns.append(
        {
            "turn": final_turn_number,
            "turn_type": "final_decision",
            "actions": [
                {
                    "type": "final_decision",
                    "actor_id": final_decision["actor_id"],
                    "actor_role": final_decision["actor_role"],
                    "payload": final_decision,
                }
            ],
            "messages": [],
            "new_disclosed_rui_ids": [],
        }
    )

    # The episode records the task, interaction settings, and outcome. The
    # evaluation runner writes run-wide model settings to run_manifest.json.
    episode = {
        "schema_version": EPISODE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datapoint": _public_datapoint_view(datapoint),
        "communication_structure": structure.to_dict(),
        "budgets": budgets.__dict__,
        "turns": turns,
        "messages": list(state["messages"]),
        "public_rui_ids": sorted(state["public_rui_ids"]),
        "final_decision": final_decision,
        "prompt_records": _collect_prompt_records(participant_actor, coordinator_actor, final_actor),
        "stopped_reason": stopped_reason,
        "initial_speaker_id": first_speaker_id,
    }
    return episode


def _run_communication_turn(
    *,
    datapoint: dict[str, Any],
    structure: CommunicationStructure,
    state: dict[str, Any],
    budgets: BudgetSpec,
    participant_actor: Any,
    coordinator_actor: Any,
) -> dict[str, Any]:
    before = set(state["public_rui_ids"])
    actions: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []

    if structure.turn_allocation_rule == "coordinator_selects_one":
        query = coordinator_actor.choose_targets(datapoint=datapoint, state=state, max_targets=1, budgets=budgets)
        actions.append({"type": "query", "actor_id": structure.coordinator_id, "target_ids": query["target_ids"], "payload": query})
        messages.append(_message_from_query(state["turn"], query))
        for participant_id in query["target_ids"]:
            messages.append(_participant_message(datapoint, participant_id, state, query, budgets, participant_actor))
    elif structure.turn_allocation_rule == "coordinator_selects_multiple":
        ids = participant_ids(datapoint)
        query = coordinator_actor.choose_targets(datapoint=datapoint, state=state, max_targets=len(ids), budgets=budgets)
        actions.append({"type": "query", "actor_id": structure.coordinator_id, "target_ids": query["target_ids"], "payload": query})
        messages.append(_message_from_query(state["turn"], query))
        for participant_id in query["target_ids"]:
            messages.append(_participant_message(datapoint, participant_id, state, query, budgets, participant_actor))
    elif structure.turn_allocation_rule == "coordinator_requests_all":
        ids = participant_ids(datapoint)
        query = coordinator_actor.choose_targets(datapoint=datapoint, state=state, max_targets=len(ids), budgets=budgets)
        query["target_ids"] = ids
        actions.append({"type": "query", "actor_id": structure.coordinator_id, "target_ids": ids, "payload": query})
        messages.append(_message_from_query(state["turn"], query))
        for participant_id in ids:
            messages.append(_participant_message(datapoint, participant_id, state, query, budgets, participant_actor))
    elif structure.turn_allocation_rule == "round_robin":
        ids = participant_ids(datapoint)
        participant_id = ids[(int(state["turn"]) - 1) % len(ids)]
        actions.append({"type": "participant_turn", "actor_id": participant_id, "allocation_rule": "round_robin"})
        messages.append(_participant_message(datapoint, participant_id, state, None, budgets, participant_actor))
    elif structure.turn_allocation_rule == "participant_handoff":
        participant_id = str(state.get("next_speaker_id") or _first_participant_id(datapoint))
        actions.append({"type": "participant_turn", "actor_id": participant_id, "allocation_rule": "participant_handoff"})
        message = _participant_message(
            datapoint,
            participant_id,
            state,
            None,
            budgets,
            participant_actor,
            allow_participant_handoff=True,
            allow_participant_finalize=structure.participant_handoff_allows_finalize,
        )
        messages.append(message)
        if (
            structure.participant_handoff_allows_finalize
            and message.get("participant_action") == "finalize"
        ):
            state["participant_requested_finalize"] = True
            state["final_participant_id"] = participant_id
        else:
            state["next_speaker_id"] = message.get("recommended_next_actor_id") or _next_round_robin_id(datapoint, participant_id)
    elif structure.turn_allocation_rule == "simultaneous_all":
        ids = participant_ids(datapoint)
        actions.append({"type": "simultaneous_participant_turn", "actor_ids": ids})
        for participant_id in ids:
            messages.append(_participant_message(datapoint, participant_id, state, None, budgets, participant_actor))
    else:  # pragma: no cover - checked earlier
        raise ValueError(f"Unsupported turn allocation rule: {structure.turn_allocation_rule}")

    for message in messages:
        disclosed = [str(item) for item in message.get("disclosed_rui_ids", []) if isinstance(item, str)]
        state["public_rui_ids"].update(disclosed)
        state["messages"].append(message)
        if message.get("speaker_role") == "participant":
            state["last_speaker_id"] = message.get("speaker_id")
    new_disclosed = sorted(set(state["public_rui_ids"]) - before)
    return {
        "turn": state["turn"],
        "turn_type": "communication",
        "actions": actions,
        "messages": messages,
        "new_disclosed_rui_ids": new_disclosed,
    }


def _participant_message(
    datapoint: dict[str, Any],
    participant_id: str,
    state: dict[str, Any],
    request: dict[str, Any] | None,
    budgets: BudgetSpec,
    participant_actor: Any,
    *,
    allow_participant_handoff: bool = False,
    allow_participant_finalize: bool = False,
) -> dict[str, Any]:
    actor_kwargs = {
        "datapoint": datapoint,
        "participant_id": participant_id,
        "state": state,
        "request": request,
        "budgets": budgets,
        "allow_participant_handoff": allow_participant_handoff,
    }
    if allow_participant_finalize:
        actor_kwargs["allow_participant_finalize"] = True
    payload = participant_actor.respond(
        **actor_kwargs,
    )
    payload = dict(payload)
    payload["turn"] = state["turn"]
    payload["target_ids"] = []
    return payload


def _message_from_query(turn: int, query: dict[str, Any]) -> dict[str, Any]:
    message = {
        "turn": turn,
        "speaker_id": query.get("speaker_id", "coordinator"),
        "speaker_role": "coordinator",
        "content": query.get("content", ""),
        "target_ids": list(query.get("target_ids", [])),
        "disclosed_rui_ids": [],
    }
    for key in (
        "public_read_policy",
        "public_read_token_budget",
        "unread_public_tokens_available",
        "new_public_tokens_read",
        "priority_public_tokens_read",
        "cumulative_public_tokens_read",
    ):
        if key in query:
            message[key] = query[key]
    return message


def _make_final_decision(
    *,
    datapoint: dict[str, Any],
    structure: CommunicationStructure,
    state: dict[str, Any],
    budgets: BudgetSpec,
    coordinator_actor: Any,
    final_actor: Any,
) -> dict[str, Any]:
    if structure.final_decision_rule == "coordinator_decision":
        return coordinator_actor.decide(datapoint=datapoint, state=state, budgets=budgets)
    if structure.final_decision_rule == "designated_participant":
        participant_id = (
            (
                state.get("final_participant_id")
                or state.get("last_speaker_id")
            )
            if structure.participant_handoff_allows_finalize
            else structure.designated_final_participant_id
        ) or _first_participant_id(datapoint)
        return final_actor.decide(
            datapoint=datapoint,
            state=state,
            actor_id=participant_id,
            actor_role="participant",
            budgets=budgets,
        )
    return final_actor.decide(
        datapoint=datapoint,
        state=state,
        actor_id=structure.final_decision_rule,
        actor_role="final_synthesis",
        budgets=budgets,
    )


def _validate_explicit_actors(
    *,
    structure: CommunicationStructure,
    participant_actor: Any | None,
    coordinator_actor: Any | None,
    final_actor: Any | None,
) -> None:
    if participant_actor is None:
        raise ValueError("run_episode requires an explicit participant_actor.")
    if structure.coordinator_present and coordinator_actor is None:
        raise ValueError("run_episode requires an explicit coordinator_actor for coordinator-mediated structures.")
    if structure.final_decision_rule == "coordinator_decision" and coordinator_actor is None:
        raise ValueError("run_episode requires coordinator_actor for coordinator_decision.")
    if structure.final_decision_rule != "coordinator_decision" and final_actor is None:
        raise ValueError("run_episode requires an explicit final_actor for this final decision rule.")


def _uses_prompt_trace(actor: Any) -> bool:
    recorder = getattr(actor, "recorder", None)
    return hasattr(recorder, "prompt_records")


def _collect_prompt_records(*actors: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for actor in actors:
        recorder = getattr(actor, "recorder", None)
        if recorder is None or id(recorder) in seen:
            continue
        seen.add(id(recorder))
        prompt_records = getattr(recorder, "prompt_records", None)
        if isinstance(prompt_records, list):
            records.extend(row for row in prompt_records if isinstance(row, dict))
    return records


def _first_participant_id(datapoint: dict[str, Any]) -> str:
    ids = participant_ids(datapoint)
    if not ids:
        raise ValueError("Datapoint has no participants.")
    return ids[0]


def _next_round_robin_id(datapoint: dict[str, Any], participant_id: str) -> str:
    ids = participant_ids(datapoint)
    if not ids:
        raise ValueError("Datapoint has no participants.")
    try:
        index = ids.index(participant_id)
    except ValueError:
        return ids[0]
    return ids[(index + 1) % len(ids)]


def _public_datapoint_view(datapoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": datapoint.get("schema_version"),
        "instance_id": datapoint.get("instance_id"),
        "dataset": datapoint.get("dataset"),
        **datapoint_variant(datapoint),
        "scenario_family": datapoint.get("scenario_family"),
        "source": datapoint.get("source"),
        "task_description": datapoint.get("task_description"),
        "decision_set": datapoint.get("decision_set"),
        "decision_option_definitions": datapoint.get("decision_option_definitions"),
        "initial_common_information": datapoint.get("initial_common_information"),
        "participant_directory": datapoint.get("participant_directory"),
        "_input_path": datapoint.get("_input_path"),
        "_input_line_number": datapoint.get("_input_line_number"),
    }
