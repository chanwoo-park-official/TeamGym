from __future__ import annotations

import asyncio

import numpy as np
import torch
from verl import DataProto

from teamgym.rl.training.trainer_compat import ACTOR_ROLE_KEY
from teamgym.rl.training.two_policy_trainer import (
    DualPolicyWorkerGroup,
    RoleRoutedRolloutEngine,
)


class _FakeRollout:
    def __init__(self, name: str):
        self.name = name
        self.validate = False
        self.tokenizer = object()
        self.processor = None
        self.chat_parser = object()
        self.wake_count = 0
        self.sleep_count = 0

    def get_chat_parser(self, enable_thinking):
        return (self.name, enable_thinking)

    async def get_model_response(self, messages, **kwargs):
        del messages, kwargs
        return self.name

    async def wake_up(self):
        self.wake_count += 1

    async def sleep(self):
        self.sleep_count += 1


class _FakeWorkerGroup:
    def __init__(self, value: float):
        self.value = value
        self.world_size = 4
        self.compute_batches = []
        self.update_batches = []

    def compute_log_prob(self, batch):
        self.compute_batches.append(batch)
        shape = batch.batch["response_mask"].shape
        return DataProto.from_dict(
            tensors={
                "old_log_probs": torch.full(shape, self.value),
                "entropys": torch.full(shape, self.value + 0.5),
            },
            meta_info={"temperature": 0.7},
        )

    def update_actor(self, batch):
        self.update_batches.append(batch)
        return DataProto(
            meta_info={"metrics": {"actor/loss": [self.value]}}
        )


def _batch() -> DataProto:
    rows = 5
    response_len = 3
    return DataProto.from_dict(
        tensors={
            "response_mask": torch.ones(rows, response_len),
            "attention_mask": torch.ones(rows, 7),
            "advantages": torch.ones(rows, response_len),
            "returns": torch.ones(rows, response_len),
        },
        non_tensors={
            ACTOR_ROLE_KEY: np.asarray(
                [
                    "coordinator",
                    "participant",
                    "coordinator",
                    "participant",
                    "coordinator",
                ],
                dtype=object,
            ),
            "is_pad_step": np.zeros(rows, dtype=bool),
            "is_valid": np.ones(rows, dtype=bool),
        },
    )


def test_role_routed_rollout_engine_routes_and_controls_both_policies() -> None:
    coordinator = _FakeRollout("coordinator")
    participant = _FakeRollout("participant")
    engine = RoleRoutedRolloutEngine(coordinator, participant)

    assert (
        asyncio.run(
            engine.get_model_response(
                [],
                application_id="episode.turn.participant_response:retry_0",
            )
        )
        == "participant"
    )
    assert (
        asyncio.run(
            engine.get_model_response(
                [],
                application_id="episode.turn.coordinator_query:retry_0",
            )
        )
        == "coordinator"
    )
    engine.validate = True
    assert coordinator.validate is True
    assert participant.validate is True
    asyncio.run(engine.wake_up())
    asyncio.run(engine.sleep())
    assert (coordinator.wake_count, participant.wake_count) == (1, 1)
    assert (coordinator.sleep_count, participant.sleep_count) == (1, 1)


def test_dual_policy_worker_group_splits_pads_and_recombines_rows() -> None:
    coordinator = _FakeWorkerGroup(1.0)
    participant = _FakeWorkerGroup(2.0)
    group = DualPolicyWorkerGroup(coordinator, participant)
    batch = _batch()

    old = group.compute_log_prob(batch)
    assert len(coordinator.compute_batches[0]) == 4
    assert len(participant.compute_batches[0]) == 4
    assert torch.all(
        coordinator.compute_batches[0].batch["response_mask"][3:] == 0
    )
    assert torch.all(
        participant.compute_batches[0].batch["response_mask"][2:] == 0
    )
    assert old.batch["old_log_probs"][:, 0].tolist() == [
        1.0,
        2.0,
        1.0,
        2.0,
        1.0,
    ]

    updated = group.update_actor(batch)
    metrics = updated.meta_info["metrics"]
    assert metrics["two_policy/coordinator_rows"] == [3.0]
    assert metrics["two_policy/participant_rows"] == [2.0]
    assert metrics["coordinator/actor/loss"] == [1.0]
    assert metrics["participant/actor/loss"] == [2.0]
