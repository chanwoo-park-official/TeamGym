"""Communication-structure definitions.

The docs define communication structure as three separable choices:
coordinator presence, turn allocation, and final-decision rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .types import FinalDecisionRule, TurnAllocationRule


COMMUNICATION_STRUCTURE_SCHEMA = "teamgym_communication_structure"
_HISTORICAL_COMMUNICATION_STRUCTURE_SCHEMA = "collabgym_fc_communication_structure_v1"


TURN_ALLOCATION_RULES = {
    "coordinator_selects_one",
    "coordinator_selects_multiple",
    "coordinator_requests_all",
    "round_robin",
    "participant_handoff",
    "simultaneous_all",
}

FINAL_DECISION_RULES = {
    "coordinator_decision",
    "designated_participant",
    "final_synthesis",
}


@dataclass(frozen=True)
class CommunicationStructure:
    schema_version: str
    name: str
    coordinator_present: bool
    turn_allocation_rule: TurnAllocationRule
    final_decision_rule: FinalDecisionRule
    coordinator_id: str = "coordinator"
    designated_final_participant_id: str | None = None
    participant_handoff_allows_finalize: bool = False

    def checked(self) -> "CommunicationStructure":
        if self.schema_version not in {
            COMMUNICATION_STRUCTURE_SCHEMA,
            _HISTORICAL_COMMUNICATION_STRUCTURE_SCHEMA,
        }:
            raise ValueError(f"Unsupported communication structure schema: {self.schema_version}")
        if self.turn_allocation_rule not in TURN_ALLOCATION_RULES:
            raise ValueError(f"Unsupported turn allocation rule: {self.turn_allocation_rule}")
        if self.final_decision_rule not in FINAL_DECISION_RULES:
            raise ValueError(f"Unsupported final decision rule: {self.final_decision_rule}")
        if not self.coordinator_present and self.turn_allocation_rule.startswith("coordinator_"):
            raise ValueError("Coordinator turn-allocation rules require coordinator_present=True.")
        if not self.coordinator_present and self.final_decision_rule == "coordinator_decision":
            raise ValueError("coordinator_decision requires coordinator_present=True.")
        if (
            self.participant_handoff_allows_finalize
            and self.turn_allocation_rule != "participant_handoff"
        ):
            raise ValueError(
                "participant_handoff_allows_finalize requires participant_handoff."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.schema_version == _HISTORICAL_COMMUNICATION_STRUCTURE_SCHEMA:
            payload["schema_version"] = COMMUNICATION_STRUCTURE_SCHEMA
        return payload


def communication_structure_preset(name: str) -> CommunicationStructure:
    key = name.strip().casefold().replace("-", "_")
    presets = {
        "coordinator_select_one__coordinator_decision": CommunicationStructure(
            schema_version=COMMUNICATION_STRUCTURE_SCHEMA,
            name="coordinator_select_one__coordinator_decision",
            coordinator_present=True,
            turn_allocation_rule="coordinator_selects_one",
            final_decision_rule="coordinator_decision",
        ),
        "coordinator_select_multiple__coordinator_decision": CommunicationStructure(
            schema_version=COMMUNICATION_STRUCTURE_SCHEMA,
            name="coordinator_select_multiple__coordinator_decision",
            coordinator_present=True,
            turn_allocation_rule="coordinator_selects_multiple",
            final_decision_rule="coordinator_decision",
        ),
        "coordinator_ask_everybody__coordinator_decision": CommunicationStructure(
            schema_version=COMMUNICATION_STRUCTURE_SCHEMA,
            name="coordinator_ask_everybody__coordinator_decision",
            coordinator_present=True,
            turn_allocation_rule="coordinator_requests_all",
            final_decision_rule="coordinator_decision",
        ),
        "no_coordinator_round_robin__final_synthesis": CommunicationStructure(
            schema_version=COMMUNICATION_STRUCTURE_SCHEMA,
            name="no_coordinator_round_robin__final_synthesis",
            coordinator_present=False,
            turn_allocation_rule="round_robin",
            final_decision_rule="final_synthesis",
        ),
        "no_coordinator_handoff__final_synthesis": CommunicationStructure(
            schema_version=COMMUNICATION_STRUCTURE_SCHEMA,
            name="no_coordinator_handoff__final_synthesis",
            coordinator_present=False,
            turn_allocation_rule="participant_handoff",
            final_decision_rule="final_synthesis",
        ),
        "no_coordinator_simultaneous__final_synthesis": CommunicationStructure(
            schema_version=COMMUNICATION_STRUCTURE_SCHEMA,
            name="no_coordinator_simultaneous__final_synthesis",
            coordinator_present=False,
            turn_allocation_rule="simultaneous_all",
            final_decision_rule="final_synthesis",
        ),
        "no_coordinator_round_robin__designated_participant": CommunicationStructure(
            schema_version=COMMUNICATION_STRUCTURE_SCHEMA,
            name="no_coordinator_round_robin__designated_participant",
            coordinator_present=False,
            turn_allocation_rule="round_robin",
            final_decision_rule="designated_participant",
        ),
        "no_coordinator_handoff__designated_participant": CommunicationStructure(
            schema_version=COMMUNICATION_STRUCTURE_SCHEMA,
            name="no_coordinator_handoff__designated_participant",
            coordinator_present=False,
            turn_allocation_rule="participant_handoff",
            final_decision_rule="designated_participant",
        ),
        "no_coordinator_handoff_finalize__designated_participant": CommunicationStructure(
            schema_version=COMMUNICATION_STRUCTURE_SCHEMA,
            name="no_coordinator_handoff_finalize__designated_participant",
            coordinator_present=False,
            turn_allocation_rule="participant_handoff",
            final_decision_rule="designated_participant",
            participant_handoff_allows_finalize=True,
        ),
        "no_coordinator_simultaneous__designated_participant": CommunicationStructure(
            schema_version=COMMUNICATION_STRUCTURE_SCHEMA,
            name="no_coordinator_simultaneous__designated_participant",
            coordinator_present=False,
            turn_allocation_rule="simultaneous_all",
            final_decision_rule="designated_participant",
        ),
    }
    if key not in presets:
        raise ValueError(f"Unknown communication structure preset: {name}")
    return presets[key].checked()


def communication_structure_names() -> list[str]:
    return [
        "coordinator_select_one__coordinator_decision",
        "coordinator_select_multiple__coordinator_decision",
        "coordinator_ask_everybody__coordinator_decision",
        "no_coordinator_round_robin__final_synthesis",
        "no_coordinator_handoff__final_synthesis",
        "no_coordinator_simultaneous__final_synthesis",
        "no_coordinator_round_robin__designated_participant",
        "no_coordinator_handoff__designated_participant",
        "no_coordinator_handoff_finalize__designated_participant",
        "no_coordinator_simultaneous__designated_participant",
    ]
