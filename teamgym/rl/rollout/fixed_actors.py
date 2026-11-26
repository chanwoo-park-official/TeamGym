"""Deterministic fixed actors for offline tests and the small GPU smoke run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from teamgym.interaction.actor_utils import (
    participant_by_id,
    participant_ids,
    participant_private_context_units,
    truncate_tokens_strict,
)
from teamgym.interaction.types import BudgetSpec


class SourceExcerptParticipantActor:
    """Publish assigned source text without consulting hidden evaluator fields.

    This deliberately simple policy is useful as a truly frozen participant in
    coordinator-only plumbing checks. It sees only the participant assignment
    and private-context text that the ordinary participant prompt builder sees.
    """

    recorder = None

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
        participant = participant_by_id(datapoint)[participant_id]
        text = "\n".join(
            str(unit.get("text") or "").strip()
            for unit in participant_private_context_units(datapoint, participant)
            if str(unit.get("text") or "").strip()
        )
        message: dict[str, Any] = {
            "speaker_id": participant_id,
            "speaker_role": "participant",
            "content": truncate_tokens_strict(
                text, budgets.participant_utterance_token_budget
            ),
            "disclosed_rui_ids": [],
            "fixed_actor_kind": "source_excerpt_v1",
        }
        if allow_participant_handoff:
            ids = participant_ids(datapoint)
            index = ids.index(participant_id)
            message["recommended_next_actor_id"] = ids[(index + 1) % len(ids)]
        if allow_participant_finalize:
            message["participant_action"] = "handoff"
        return message


class ReplayParticipantActor:
    """Replay a frozen participant-message bank for attribution evaluations."""

    recorder = None

    def __init__(self, replay_path: str | Path):
        path = Path(replay_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = payload.get("messages")
        if not isinstance(messages, dict):
            raise ValueError(f"Replay bank has no messages mapping: {path}")
        self.path = path
        self.messages = messages

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
        del state, request
        instance_id = str(datapoint.get("instance_id") or "")
        by_participant = self.messages.get(instance_id)
        if not isinstance(by_participant, dict):
            raise KeyError(f"Replay bank has no task {instance_id!r}")
        content = str(by_participant.get(participant_id) or "").strip()
        if not content:
            raise KeyError(
                f"Replay bank has no message for {instance_id}/{participant_id}"
            )
        message: dict[str, Any] = {
            "speaker_id": participant_id,
            "speaker_role": "participant",
            "content": truncate_tokens_strict(
                content, budgets.participant_utterance_token_budget
            ),
            "disclosed_rui_ids": [],
            "fixed_actor_kind": "participant_replay_v1",
            "replay_source": str(self.path),
        }
        if allow_participant_handoff:
            ids = participant_ids(datapoint)
            index = ids.index(participant_id)
            message["recommended_next_actor_id"] = ids[(index + 1) % len(ids)]
        if allow_participant_finalize:
            message["participant_action"] = "handoff"
        return message


class DeterministicCoordinatorActor:
    """Non-oracle coordinator fallback for offline participant-only tests."""

    recorder = None

    def choose_targets(
        self,
        *,
        datapoint: dict[str, Any],
        state: dict[str, Any],
        max_targets: int,
        budgets: BudgetSpec,
    ) -> dict[str, Any]:
        ids = participant_ids(datapoint)
        start = max(0, int(state.get("turn", 1)) - 1) % len(ids)
        count = max(1, min(int(max_targets), len(ids)))
        targets = [ids[(start + offset) % len(ids)] for offset in range(count)]
        return {
            "action": "turn_choose",
            "speaker_id": "coordinator",
            "speaker_role": "coordinator",
            "content": "Quote the decisive source clause, then state its implication.",
            "target_ids": targets,
            "reason": "fixed round-robin request",
        }

    def decide(
        self,
        *,
        datapoint: dict[str, Any],
        state: dict[str, Any],
        budgets: BudgetSpec,
    ) -> dict[str, Any]:
        decisions = [str(value) for value in datapoint.get("decision_set", [])]
        return {
            "action": "final_decision",
            "actor_id": "coordinator",
            "actor_role": "coordinator",
            "answer": decisions[0] if decisions else None,
            "rationale": "Fixed baseline decision based on the public transcript.",
            "written_answer": "",
            "confidence": 0.0,
        }
