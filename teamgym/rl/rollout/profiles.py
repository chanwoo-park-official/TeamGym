"""Named, versioned rollout profiles for TeamGym."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from teamgym.interaction.types import BudgetSpec


@dataclass(frozen=True)
class RolloutProfile:
    schema_version: str
    name: str
    communication_structure: str
    budgets: BudgetSpec
    participant_response_mode: str = "direct"
    participant_role_name_mode: str = "hidden"
    participant_observation_mode: str = "full_public"
    evaluation_temperature: float = 0.0
    model_max_output_tokens: int = 3600
    participant_model_max_output_tokens: int | None = None
    coordinator_query_compact_output: bool = False
    paper_compatible: bool = True
    dynamic_turn_budget: str | None = None
    initial_speaker_mode: str = "first"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["budgets"] = asdict(self.budgets.checked())
        return payload


_MAIN_BUDGETS = BudgetSpec(
    turn_budget=16,
    participant_context_token_budget=40_000,
    participant_utterance_token_budget=600,
    coordinator_utterance_token_budget=600,
    participant_reasoning_token_budget=1_200,
    coordinator_reasoning_token_budget=1_200,
    final_reasoning_token_budget=1_800,
    final_rationale_token_budget=1_200,
    participant_public_read_policy="unlimited",
    participant_public_read_token_budget=0,
    participant_public_read_seed=0,
    public_read_scope="participant_only",
    public_read_charge_current_message=False,
).checked()


def canonical_profile_name(name: str) -> str:
    """Normalize profile names and resolve aliases from earlier releases."""
    key = name.strip().casefold().replace("-", "_")
    return "short_context_smoke" if key == "smoke_h200_qwen25" else key


def get_rollout_profile(
    name: str, *, participant_count: int | None = None
) -> RolloutProfile:
    """Resolve a profile without mutating paper defaults."""

    key = canonical_profile_name(name)
    if key == "paper_main_full_context":
        return RolloutProfile(
            schema_version="teamgym_rollout_profile_v1",
            name=key,
            communication_structure="coordinator_select_one__coordinator_decision",
            budgets=_MAIN_BUDGETS,
            note="Main-paper full-context coordinator profile: 15 query turns and one final turn.",
        )
    if key == "paper_matched_participant_opportunities":
        if participant_count is None or int(participant_count) < 1:
            raise ValueError(
                "paper_matched_participant_opportunities requires participant_count >= 1"
            )
        return RolloutProfile(
            schema_version="teamgym_rollout_profile_v1",
            name=key,
            communication_structure="coordinator_select_one__coordinator_decision",
            budgets=replace(
                _MAIN_BUDGETS, turn_budget=int(participant_count) + 1
            ).checked(),
            note="Sequential matched-opportunity profile: N communication turns plus one final turn.",
        )
    if key == "short_context_smoke":
        return RolloutProfile(
            schema_version="teamgym_rollout_profile_v1",
            name=key,
            communication_structure="coordinator_select_one__coordinator_decision",
            budgets=BudgetSpec(
                turn_budget=3,
                participant_context_token_budget=2_048,
                participant_utterance_token_budget=192,
                coordinator_utterance_token_budget=128,
                final_rationale_token_budget=192,
                participant_reasoning_token_budget=0,
                coordinator_reasoning_token_budget=128,
                final_reasoning_token_budget=192,
                participant_public_read_policy="unlimited",
            ).checked(),
            model_max_output_tokens=512,
            paper_compatible=False,
            note=(
                "Operational smoke profile. It uses the paper runtime and prompts but reduces turns, "
                "field budgets, and generation capacity so optimizer plumbing can be checked quickly."
            ),
        )
    if key == "finance_riu_participant_handoff_15":
        return RolloutProfile(
            schema_version="teamgym_rollout_profile_v1",
            name=key,
            communication_structure=(
                "no_coordinator_handoff_finalize__designated_participant"
            ),
            budgets=BudgetSpec(
                # At most 15 public participant turns, followed by one final
                # decision from the participant who requests finalization or
                # speaks on the last available communication turn.
                turn_budget=16,
                participant_context_token_budget=40_000,
                participant_utterance_token_budget=600,
                coordinator_utterance_token_budget=0,
                final_rationale_token_budget=1_200,
                participant_reasoning_token_budget=1_200,
                coordinator_reasoning_token_budget=0,
                final_reasoning_token_budget=1_800,
                participant_public_read_policy="unlimited",
            ).checked(),
            participant_response_mode="direct",
            participant_role_name_mode="public",
            participant_observation_mode="full_public",
            # Final finance training and matched endpoints used the same cap.
            model_max_output_tokens=750,
            participant_model_max_output_tokens=750,
            paper_compatible=False,
            initial_speaker_mode="episode_hash",
            note=(
                "Finance RIU-only participant experiment: no coordinator; "
                "every speaker sees participant ids and public role names, "
                "then chooses a valid next participant or finalizes. The "
                "communication phase is capped at 15 turns."
            ),
        )
    if key == "written_64_query_private_full_context":
        if participant_count is not None and int(participant_count) < 1:
            raise ValueError(f"{key} requires participant_count >= 1")
        resolved_participant_count = int(participant_count or 1)
        return RolloutProfile(
            schema_version="teamgym_rollout_profile_v1",
            name=key,
            communication_structure=(
                "coordinator_select_one__coordinator_decision"
            ),
            budgets=BudgetSpec(
                turn_budget=resolved_participant_count + 1,
                participant_context_token_budget=28_000,
                # The hard 64-token completion cap includes the JSON
                # envelope. Reserve enough space for {"utterance":"..."} to
                # close instead of advertising an impossible 64-token field.
                participant_utterance_token_budget=48,
                coordinator_utterance_token_budget=64,
                final_rationale_token_budget=256,
                participant_reasoning_token_budget=0,
                coordinator_reasoning_token_budget=0,
                final_reasoning_token_budget=0,
                participant_public_read_policy="unlimited",
            ).checked(),
            participant_response_mode="direct",
            participant_role_name_mode="hidden",
            participant_observation_mode="query_private_only",
            model_max_output_tokens=384,
            participant_model_max_output_tokens=64,
            coordinator_query_compact_output=True,
            paper_compatible=False,
            dynamic_turn_budget="participant_count + 1",
            note=(
                "Written-answer RL profile. The coordinator sees the task and "
                "queries each private-source participant; participants see only "
                "the current question and private source. Participant completions "
                "are capped at 64 total tokens, with a 48-token utterance budget "
                "that leaves room for the JSON envelope. Final written synthesis "
                "has a 256-token field budget within a 384-token completion."
            ),
        )
    if key in {
        "efficient_64_coordinator",
        "efficient_64_full_context",
        "efficient_64_query_private_riu_only",
        "efficient_64_query_private_full_context",
        "efficient_32_participant_query_private_full_context",
    }:
        if participant_count is not None and int(participant_count) < 1:
            raise ValueError(
                f"{key} requires participant_count >= 1"
            )
        resolved_participant_count = int(participant_count or 1)
        full_context = key in {
            "efficient_64_full_context",
            "efficient_64_query_private_full_context",
            "efficient_32_participant_query_private_full_context",
        }
        query_private = key in {
            "efficient_64_query_private_riu_only",
            "efficient_64_query_private_full_context",
            "efficient_32_participant_query_private_full_context",
        }
        participant_output_tokens = (
            32
            if key == "efficient_32_participant_query_private_full_context"
            else 64
        )
        participant_context_token_budget = 28_000 if full_context else 40_000
        return RolloutProfile(
            schema_version="teamgym_rollout_profile_v1",
            name=key,
            communication_structure="coordinator_select_one__coordinator_decision",
            budgets=BudgetSpec(
                # Query every participant once, then reserve the final turn for
                # the coordinator's answer.
                turn_budget=resolved_participant_count + 1,
                participant_context_token_budget=participant_context_token_budget,
                participant_utterance_token_budget=participant_output_tokens,
                coordinator_utterance_token_budget=64,
                final_rationale_token_budget=64,
                participant_reasoning_token_budget=0,
                coordinator_reasoning_token_budget=0,
                final_reasoning_token_budget=0,
                participant_public_read_policy="unlimited",
            ).checked(),
            participant_response_mode="direct",
            participant_observation_mode=(
                "query_private_only" if query_private else "full_public"
            ),
            model_max_output_tokens=64,
            participant_model_max_output_tokens=(
                participant_output_tokens
                if participant_output_tokens != 64
                else None
            ),
            paper_compatible=False,
            dynamic_turn_budget="participant_count + 1",
            note=(
                "Efficiency-training profile: one direct query opportunity per participant, "
                "one direct final answer, no separate private-think stage, and a hard "
                f"{participant_output_tokens}-token participant model-completion cap "
                "including the JSON action. Coordinator queries and final decisions retain "
                "the 64-token model-completion cap. "
                + (
                    "Participants observe only the current coordinator question and their private source. "
                    if query_private
                    else ""
                )
                + (
                    "The full-context variant uses a 28,000-token private-source cap so this dataset remains "
                    "within Qwen2.5's native 32,768-token context window."
                    if full_context
                    else ""
                )
            ),
        )
    if key in {
        "retrieval_probe_full_context",
        "retrieval_probe_query_private_full_context",
    }:
        if participant_count is not None and int(participant_count) < 1:
            raise ValueError(f"{key} requires participant_count >= 1")
        resolved_participant_count = int(participant_count or 1)
        query_private = key == "retrieval_probe_query_private_full_context"
        return RolloutProfile(
            schema_version="teamgym_rollout_profile_v1",
            name=key,
            communication_structure="coordinator_select_one__coordinator_decision",
            budgets=BudgetSpec(
                turn_budget=resolved_participant_count + 1,
                participant_context_token_budget=27_000,
                participant_utterance_token_budget=256,
                coordinator_utterance_token_budget=64,
                final_rationale_token_budget=64,
                participant_reasoning_token_budget=0,
                coordinator_reasoning_token_budget=0,
                final_reasoning_token_budget=0,
                participant_public_read_policy="unlimited",
            ).checked(),
            participant_response_mode="direct",
            participant_role_name_mode="hidden",
            participant_observation_mode=(
                "query_private_only" if query_private else "full_public"
            ),
            model_max_output_tokens=256,
            paper_compatible=False,
            dynamic_turn_budget="participant_count + 1",
            note=(
                "Evaluation-only participant retrieval probe. A deterministic "
                "coordinator queries each participant once and the participant "
                "gets a 256-token direct response budget. Probe metrics must "
                "never be used as RL reward. "
                + (
                    "The participant sees only the probe question and its private source."
                    if query_private
                    else ""
                )
            ),
        )
    raise ValueError(f"Unknown TeamGym rollout profile: {name}")


def profile_names() -> tuple[str, ...]:
    return (
        "paper_main_full_context",
        "paper_matched_participant_opportunities",
        "short_context_smoke",
        "finance_riu_participant_handoff_15",
        "written_64_query_private_full_context",
        "efficient_64_coordinator",
        "efficient_64_full_context",
        "efficient_64_query_private_riu_only",
        "efficient_64_query_private_full_context",
        "retrieval_probe_full_context",
        "retrieval_probe_query_private_full_context",
    )
