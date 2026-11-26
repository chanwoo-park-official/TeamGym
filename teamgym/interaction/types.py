"""Shared typed objects for the fully collaborative interaction runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


TurnAllocationRule = Literal[
    "coordinator_selects_one",
    "coordinator_selects_multiple",
    "coordinator_requests_all",
    "round_robin",
    "participant_handoff",
    "simultaneous_all",
]

FinalDecisionRule = Literal[
    "coordinator_decision",
    "designated_participant",
    "final_synthesis",
]

ActorRole = Literal["coordinator", "participant", "final_synthesis"]


@dataclass(frozen=True)
class BudgetSpec:
    """Resource limits used by the interaction loop.

    One turn is one allocation step. A coordinator-mediated turn may include a
    coordinator query and one or more participant responses. The last turn is
    reserved for the final decision.
    """

    turn_budget: int = 6
    participant_utterance_token_budget: int = 80
    coordinator_utterance_token_budget: int = 80
    final_rationale_token_budget: int = 120
    participant_reasoning_token_budget: int = 128
    coordinator_reasoning_token_budget: int = 128
    final_reasoning_token_budget: int = 256
    participant_context_token_budget: int = 20000
    participant_public_read_policy: Literal["unlimited", "fifo", "stochastic_contiguous"] = "unlimited"
    participant_public_read_token_budget: int = 0
    participant_public_read_seed: int = 0
    public_read_scope: Literal["participant_only", "all_actors"] = "participant_only"
    public_read_charge_current_message: bool = False

    def checked(self) -> "BudgetSpec":
        read_policy = str(self.participant_public_read_policy)
        if read_policy not in {"unlimited", "fifo", "stochastic_contiguous"}:
            raise ValueError(f"Unknown participant public-read policy: {read_policy}")
        read_scope = str(self.public_read_scope)
        if read_scope not in {"participant_only", "all_actors"}:
            raise ValueError(f"Unknown public-read scope: {read_scope}")
        return BudgetSpec(
            turn_budget=max(1, int(self.turn_budget)),
            participant_utterance_token_budget=max(1, int(self.participant_utterance_token_budget)),
            coordinator_utterance_token_budget=max(1, int(self.coordinator_utterance_token_budget)),
            final_rationale_token_budget=max(1, int(self.final_rationale_token_budget)),
            participant_reasoning_token_budget=max(0, int(self.participant_reasoning_token_budget)),
            coordinator_reasoning_token_budget=max(0, int(self.coordinator_reasoning_token_budget)),
            final_reasoning_token_budget=max(0, int(self.final_reasoning_token_budget)),
            participant_context_token_budget=max(1, int(self.participant_context_token_budget)),
            participant_public_read_policy=read_policy,
            participant_public_read_token_budget=(
                0
                if read_policy == "unlimited"
                else max(1, int(self.participant_public_read_token_budget))
            ),
            participant_public_read_seed=int(self.participant_public_read_seed),
            public_read_scope=read_scope,
            public_read_charge_current_message=bool(self.public_read_charge_current_message),
        )


EpisodeMessage = dict[str, Any]
EpisodeAction = dict[str, Any]
Episode = dict[str, Any]
