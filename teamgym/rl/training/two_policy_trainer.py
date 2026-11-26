"""True two-policy full-parameter PPO support for TeamGym.

The stock RLLM ``joint`` mode uses one shared actor for every role.  This
module instead colocates two independent ActorRolloutRefWorker instances on
each GPU, routes coordinator/final generations to one and participant
generations to the other, and dispatches PPO updates by the recorded role.
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.single_controller.ray.base import RayClassWithInitArgs
from verl.trainer.ppo.utils import Role
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.rollout.utils import is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.utils.episode_logger import EpisodeLogger

from .trainer_compat import (
    ACTOR_ROLE_KEY,
    TeamGymAgentWorkflowPPOTrainer,
)


COORDINATOR_POLICY = "coordinator_policy"
PARTICIPANT_POLICY = "participant_policy"
REFERENCE_POLICY = "reference_policy"


class OffloadedReferencePolicyWorker(ActorRolloutRefWorker):
    """Make VERL's ref ``param_offload`` flag effective for FSDP1.

    VERL 0.6.1 records the flag on reference-only workers but does not move
    ``ref_module_fsdp`` in either ``init_model`` or ``compute_ref_log_prob``.
    That leaves a full 7B reference resident during both policy backwards.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.ref_module_fsdp)

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(
            mesh_name="actor"
        )
    )
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.ref_module_fsdp)
        try:
            return super().compute_ref_log_prob(data)
        finally:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.ref_module_fsdp)


class _NamedTwoPolicyVLLMReplica(vLLMReplica):
    """vLLM replica with role-specific Ray actor names."""

    policy_prefix = ""

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to "
            f"world size {self.world_size}"
        )
        worker_node_ids = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: ray.get_runtime_context().get_node_id()
                )
                for worker in self.workers
            ]
        )

        nnodes, gpus_per_node = self.nnodes, self.gpus_per_node
        if self.config.data_parallel_size == 1:
            nnodes = 1
            gpus_per_node = self.world_size

        for node_rank in range(nnodes):
            workers = self.workers[
                node_rank * gpus_per_node : (node_rank + 1) * gpus_per_node
            ]
            node_id = worker_node_ids[node_rank * gpus_per_node]
            name = (
                f"teamgym_{self.policy_prefix}_vllm_server_"
                f"{self.replica_rank}_{node_rank}"
            )
            server = self.server_class.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_node,
                nnodes=nnodes,
            )
            self.servers.append(server)

        master_address, master_port = (
            await self.servers[0].get_master_address.remote()
        )
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address,
                    master_port=master_port,
                )
                for server in self.servers
            ]
        )
        server_address, server_port = (
            await self.servers[0].get_server_address.remote()
        )
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )


class _CoordinatorVLLMReplica(_NamedTwoPolicyVLLMReplica):
    policy_prefix = COORDINATOR_POLICY


class _ParticipantVLLMReplica(_NamedTwoPolicyVLLMReplica):
    policy_prefix = PARTICIPANT_POLICY


class _NoAgentWorkersLoopManager(AgentLoopManager):
    """TeamGym drives the HTTP servers directly, so no VERL loop actors."""

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []


class _CoordinatorLoopManager(_NoAgentWorkersLoopManager):
    rollout_replica_class = _CoordinatorVLLMReplica


class _ParticipantLoopManager(_NoAgentWorkersLoopManager):
    rollout_replica_class = _ParticipantVLLMReplica


class RoleRoutedRolloutEngine:
    """Route each TeamGym request to its role's independent rollout engine."""

    def __init__(self, coordinator: VerlEngine, participant: VerlEngine):
        self.coordinator = coordinator
        self.participant = participant
        self.tokenizer = coordinator.tokenizer
        self.processor = coordinator.processor
        self.chat_parser = coordinator.chat_parser
        self._validate = False

    @property
    def validate(self) -> bool:
        return self._validate

    @validate.setter
    def validate(self, value: bool) -> None:
        self._validate = bool(value)
        self.coordinator.validate = bool(value)
        self.participant.validate = bool(value)

    def get_chat_parser(self, enable_thinking: bool | None):
        return self.coordinator.get_chat_parser(enable_thinking)

    async def get_model_response(self, messages: list[dict], **kwargs):
        application_id = str(kwargs.get("application_id") or "")
        engine = (
            self.participant
            if ".participant_" in application_id
            else self.coordinator
        )
        return await engine.get_model_response(messages, **kwargs)

    async def wake_up(self):
        await asyncio.gather(
            self.coordinator.wake_up(),
            self.participant.wake_up(),
        )

    async def sleep(self):
        await asyncio.gather(
            self.coordinator.sleep(),
            self.participant.sleep(),
        )


def _clone_config(config):
    return OmegaConf.create(
        OmegaConf.to_container(config, resolve=False)
    )


def _zero_padding_loss(batch: DataProto, pad_size: int) -> None:
    if pad_size <= 0:
        return
    for key in ("response_mask", "advantages", "returns"):
        if key in batch.batch:
            batch.batch[key][-pad_size:] = 0
    if "is_pad_step" in batch.non_tensor_batch:
        batch.non_tensor_batch["is_pad_step"][-pad_size:] = True
    if "is_valid" in batch.non_tensor_batch:
        batch.non_tensor_batch["is_valid"][-pad_size:] = False


def _role_subbatch(
    batch: DataProto,
    indices: np.ndarray,
    world_size: int,
) -> tuple[DataProto, int]:
    subbatch = batch.select_idxs(indices)
    subbatch, pad_size = pad_dataproto_to_divisor(subbatch, world_size)
    _zero_padding_loss(subbatch, pad_size)
    if "attention_mask" in subbatch.batch:
        subbatch.meta_info["global_token_num"] = (
            torch.sum(subbatch.batch["attention_mask"], dim=-1).tolist()
        )
    return subbatch, pad_size


class DualPolicyWorkerGroup:
    """Dispatch policy operations to independent coordinator/participant WGs."""

    def __init__(self, coordinator, participant):
        if coordinator.world_size != participant.world_size:
            raise ValueError("Two policy worker groups must have equal world size")
        self.coordinator = coordinator
        self.participant = participant
        self.world_size = coordinator.world_size

    def _roles(self, batch: DataProto) -> np.ndarray:
        if ACTOR_ROLE_KEY not in batch.non_tensor_batch:
            raise RuntimeError(
                f"Missing required two-policy metadata: {ACTOR_ROLE_KEY}"
            )
        roles = np.asarray(
            batch.non_tensor_batch[ACTOR_ROLE_KEY],
            dtype=object,
        )
        unexpected = sorted(
            set(roles.tolist()) - {"coordinator", "participant"}
        )
        if unexpected:
            raise RuntimeError(f"Unexpected TeamGym actor roles: {unexpected}")
        return roles

    def compute_log_prob(self, batch: DataProto) -> DataProto:
        roles = self._roles(batch)
        outputs: list[tuple[np.ndarray, DataProto]] = []
        for role, worker_group in (
            ("coordinator", self.coordinator),
            ("participant", self.participant),
        ):
            indices = np.where(roles == role)[0]
            if not len(indices):
                continue
            subbatch, pad_size = _role_subbatch(
                batch, indices, worker_group.world_size
            )
            output = worker_group.compute_log_prob(subbatch)
            outputs.append((indices, unpad_dataproto(output, pad_size)))

        if not outputs:
            raise RuntimeError("Two-policy batch contains no policy rows")
        tensor_keys = set(outputs[0][1].batch.keys())
        tensors = {
            key: torch.zeros(
                (len(batch), *outputs[0][1].batch[key].shape[1:]),
                dtype=outputs[0][1].batch[key].dtype,
                device=outputs[0][1].batch[key].device,
            )
            for key in tensor_keys
        }
        for indices, output in outputs:
            for key in tensor_keys:
                tensors[key][indices] = output.batch[key]
        return DataProto.from_dict(
            tensors=tensors,
            meta_info=outputs[0][1].meta_info,
        )

    def update_actor(self, batch: DataProto) -> DataProto:
        roles = self._roles(batch)
        metrics: dict[str, list[Any]] = defaultdict(list)
        for role, worker_group in (
            ("coordinator", self.coordinator),
            ("participant", self.participant),
        ):
            indices = np.where(roles == role)[0]
            metrics[f"two_policy/{role}_rows"].append(float(len(indices)))
            if not len(indices):
                continue
            subbatch, _ = _role_subbatch(
                batch, indices, worker_group.world_size
            )
            output = worker_group.update_actor(subbatch)
            for key, values in output.meta_info.get("metrics", {}).items():
                if isinstance(values, (list, tuple)):
                    metrics[f"{role}/{key}"].extend(values)
                else:
                    metrics[f"{role}/{key}"].append(values)
        return DataProto(meta_info={"metrics": dict(metrics)})

    def save_checkpoint(
        self,
        local_path: str,
        remote_path: str | None,
        global_step: int,
        max_ckpt_to_keep: int | None = None,
    ):
        self.coordinator.save_checkpoint(
            os.path.join(local_path, "coordinator"),
            (
                os.path.join(remote_path, "coordinator")
                if remote_path is not None
                else None
            ),
            global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        self.participant.save_checkpoint(
            os.path.join(local_path, "participant"),
            (
                os.path.join(remote_path, "participant")
                if remote_path is not None
                else None
            ),
            global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )

    def load_checkpoint(
        self,
        local_path: str | None,
        del_local_after_load: bool = False,
    ):
        if local_path is None:
            self.coordinator.load_checkpoint(None)
            self.participant.load_checkpoint(None)
            return
        self.coordinator.load_checkpoint(
            os.path.join(local_path, "coordinator"),
            del_local_after_load=del_local_after_load,
        )
        self.participant.load_checkpoint(
            os.path.join(local_path, "participant"),
            del_local_after_load=del_local_after_load,
        )


class TeamGymTwoPolicyPPOTrainer(TeamGymAgentWorkflowPPOTrainer):
    """Two independent full-parameter policies trained from one team reward."""

    def _validate_config(self):
        super()._validate_config()
        if str(
            OmegaConf.select(
                self.config, "teamgym.training_mode", default=""
            )
        ) != "joint":
            raise ValueError("Two-policy trainer requires training_mode=joint")
        if int(self.config.actor_rollout_ref.model.get("lora_rank", 0)) != 0:
            raise ValueError("Two-policy trainer is for full training, not LoRA")
        if self.config.actor_rollout_ref.rollout.name != "vllm":
            raise ValueError("Two-policy trainer currently requires vLLM")
        if self.use_critic or self.use_rm:
            raise NotImplementedError(
                "Two-policy TeamGym currently supports GRPO without critic/RM"
            )

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        resource_pool = self.resource_pool_manager.get_resource_pool(
            Role.ActorRollout
        )
        # Run each policy and the frozen KL reference in its own Ray process.
        # All three processes for a rank share that rank's physical GPU via
        # fractional Ray resources. Separate PIDs are required because VERL's
        # async vLLM bridge binds one ZeroMQ endpoint per process.
        resource_pool.max_colocate_count = 3
        actor_cls = self.role_worker_mapping[Role.ActorRollout]
        wg_kwargs: dict[str, Any] = {"device_name": self.device_name}
        wait_timeout = OmegaConf.select(
            self.config.trainer, "ray_wait_register_center_timeout"
        )
        if wait_timeout is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = wait_timeout
        profile_steps = OmegaConf.select(
            self.config.global_profiler, "steps"
        )
        if profile_steps is not None:
            wg_kwargs["profile_steps"] = profile_steps
            if (
                OmegaConf.select(self.config.global_profiler, "tool")
                == "nsys"
            ):
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                )

        def make_worker_group(
            *,
            name: str,
            worker_cls,
            role: Role,
        ):
            return self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=RayClassWithInitArgs(
                    cls=worker_cls,
                    config=_clone_config(self.config.actor_rollout_ref),
                    role=str(role),
                ),
                name_prefix=f"teamgym_{name}_",
                **wg_kwargs,
            )

        if self.use_reference_policy:
            self.ref_policy_wg = make_worker_group(
                name=REFERENCE_POLICY,
                worker_cls=ray.remote(OffloadedReferencePolicyWorker),
                role=Role.RefPolicy,
            )
            self.ref_policy_wg.init_model()

        self.coordinator_actor_rollout_wg = make_worker_group(
            name=COORDINATOR_POLICY,
            worker_cls=actor_cls,
            role=Role.ActorRollout,
        )
        self.participant_actor_rollout_wg = make_worker_group(
            name=PARTICIPANT_POLICY,
            worker_cls=actor_cls,
            role=Role.ActorRollout,
        )
        self.coordinator_actor_rollout_wg.init_model()
        self.participant_actor_rollout_wg.init_model()
        self.actor_rollout_wg = DualPolicyWorkerGroup(
            self.coordinator_actor_rollout_wg,
            self.participant_actor_rollout_wg,
        )
        self.rm_wg = None
        self.async_rollout_mode = True

        coordinator_manager = _CoordinatorLoopManager(
            config=_clone_config(self.config),
            worker_group=self.coordinator_actor_rollout_wg,
            rm_wg=None,
        )
        participant_manager = _ParticipantLoopManager(
            config=_clone_config(self.config),
            worker_group=self.participant_actor_rollout_wg,
            rm_wg=None,
        )
        self.coordinator_rollout_manager = coordinator_manager
        self.participant_rollout_manager = participant_manager
        self.async_rollout_manager = coordinator_manager
        coordinator_engine = VerlEngine(
            config=self.config,
            rollout_manager=coordinator_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        participant_engine = VerlEngine(
            config=self.config,
            rollout_manager=participant_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        rollout_engine = RoleRoutedRolloutEngine(
            coordinator_engine,
            participant_engine,
        )

        episode_logger = None
        if self.config.trainer.get("log_episodes", False):
            episode_log_dir = self.config.trainer.get(
                "episode_log_dir",
                (
                    f"logs/{self.config.trainer.project_name}/"
                    f"{self.config.trainer.experiment_name}"
                ),
            )
            episode_logger = EpisodeLogger(
                base_dir=episode_log_dir,
                subdirectory="episodes",
            )
        sat_cfg = getattr(self.config, "sat", None)
        log_train_episodes = bool(
            getattr(sat_cfg, "debug_training_trajectory", True)
        )
        engine_cls = self._resolve_engine_class()
        self.agent_execution_engine = engine_cls(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=rollout_engine,
            config=self.config,
            n_parallel_tasks=self.config.rllm.workflow.n_parallel_tasks,
            retry_limit=self.config.rllm.workflow.retry_limit,
            episode_logger=episode_logger,
            log_train_episodes=log_train_episodes,
        )
        asyncio.run_coroutine_threadsafe(
            self.agent_execution_engine.initialize_pool(), self._loop
        ).result()
