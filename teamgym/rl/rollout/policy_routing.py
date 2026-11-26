"""Shared policy labels and prompt-role lookup for TeamGym model clients."""

from __future__ import annotations

BASE_LORA_NAME = "__base__"


def _actor_role_for_prompt(prompt_id: str) -> str:
    if ".participant_" in prompt_id:
        return "participant"
    if ".coordinator" in prompt_id:
        return "coordinator"
    return "final_synthesis"


# These are VERL's reserved identifiers for the synchronized training adapter.
# Keeping the tiny protocol constants local avoids importing vLLM server code
# merely to construct or unit-test the workflow.
FULL_MODEL_NAME = "__full__"
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"


def _is_trainable(role: str, mode: str) -> bool:
    return (
        mode == "joint"
        or (mode == "coordinator_only" and role == "coordinator")
        or (mode == "participant_only" and role == "participant")
    )
