"""
Copyright under Agentica Project.

Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf, open_dict
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer

# Local application imports
from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env


@hydra.main(config_path="../config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    run_ppo_agent(config)


def run_ppo_agent(config):
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # read off all the `ray_init` settings from the config
        if config is not None and hasattr(config, "ray_init"):
            ray_init_settings = {k: v for k, v in config.ray_init.items() if v is not None}
        else:
            ray_init_settings = {}
        ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if is_cuda_available and config.trainer.get("profile_steps") is not None and len(config.trainer.get("profile_steps", [])) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(
        self,
        config,
        workflow_class=None,
        workflow_args=None,
        workflow_trainer_cls=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        env_args=None,
        agent_run_func=None,
    ):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.single_controller.ray import RayWorkerGroup
        from verl.utils.fs import copy_to_local

        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config))

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Starting from verl 0.6.1, this cls has been standardized for both fsdp and megatron backends.
        ray_worker_group_cls = RayWorkerGroup
        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                # import warnings
                # warnings.warn(f"Legacy worker impl is going to be deprecated, will be removed in the future. \
                #   Please set trainer.use_legacy_worker_impl = false to switch to the new worker implementation.")
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # if config.rllm.workflow.use_workflow:
        if agent_run_func is not None:
            print("IMPORTANT: Using AgentSdkTrainer")
            from rllm.trainer.verl.agent_sdk_trainer import AgentSdkTrainer

            trainer = AgentSdkTrainer(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                agent_run_func=agent_run_func,
            )
        elif workflow_class is not None:
            workflow_args = workflow_args or {}
            if config.rllm.workflow.get("workflow_args") is not None:
                for key, value in config.rllm.workflow.get("workflow_args").items():
                    if value is not None:
                        if key in workflow_args and isinstance(workflow_args[key], dict):
                            workflow_args[key].update(value)
                        else:
                            workflow_args[key] = value

            trainer_cls = workflow_trainer_cls or AgentWorkflowPPOTrainer
            trainer = trainer_cls(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                workflow_class=workflow_class,
                workflow_args=workflow_args,
            )

        else:
            if env_class is None:
                env_class = ENV_CLASS_MAPPING[config.rllm.env.name]
            if agent_class is None:
                agent_class = AGENT_CLASS_MAPPING[config.rllm.agent.name]

            env_args = env_args or {}
            agent_args = agent_args or {}
            if config.rllm.env.get("env_args") is not None:
                env_args.update(config.rllm.env.get("env_args"))
            if config.rllm.agent.get("agent_args") is not None:
                agent_args.update(config.rllm.agent.get("agent_args"))

            trainer = AgentPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                env_class=env_class,
                agent_class=agent_class,
                env_args=env_args,
                agent_args=agent_args,
            )

        trainer.init_workers()
        try:
            trainer.fit_agent()
        finally:
            trainer.shutdown()


def _get_rllm_flag(config, name: str, default: bool = False) -> bool:
    if config is None:
        return default
    if hasattr(config, "rllm"):
        return bool(getattr(config.rllm, name, default))
    if isinstance(config, dict):
        return bool(config.get("rllm", {}).get(name, default))
    return default


def _get_verl_flag(config, name: str, default: bool = False) -> bool:
    if config is None:
        return default
    if hasattr(config, "verl"):
        return bool(getattr(config.verl, name, default))
    if isinstance(config, dict):
        return bool(config.get("verl", {}).get(name, default))
    return default


def _set_env_bool(name: str, value: bool) -> None:
    os.environ[name] = "1" if bool(value) else "0"


def _set_cfg_bool(node, key: str, value: bool) -> None:
    if node is None:
        return
    val = bool(value)
    try:
        with open_dict(node):
            setattr(node, key, val)
            return
    except Exception:
        pass
    try:
        node[key] = val
        return
    except Exception:
        pass
    try:
        setattr(node, key, val)
    except Exception:
        pass


def _env_flag(primary: str, legacy: str | None = None) -> bool:
    raw = os.getenv(primary)
    if raw is None and legacy is not None:
        raw = os.getenv(legacy)
    if raw is None:
        return False
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


_ORIG_TASKRUNNER_RUN = TaskRunner.run


def _patched_taskrunner_run(
    self,
    config,
    workflow_class=None,
    workflow_args=None,
    workflow_trainer_cls=None,
    agent_class=None,
    env_class=None,
    agent_args=None,
    env_args=None,
    agent_run_func=None,
):
    efficient_mode = _get_rllm_flag(config, "efficient_mode", False)
    multi_agent = _get_rllm_flag(config, "multi_agent", False)
    debug_lora = _get_rllm_flag(config, "debug_lora", _env_flag("RLLM_DEBUG_LORA")) or _get_verl_flag(
        config, "debug_lora", _env_flag("RLLM_DEBUG_LORA")
    )
    debug_balance_batch = _get_rllm_flag(
        config, "debug_balance_batch", _env_flag("RLLM_DEBUG_BALANCE_BATCH")
    )
    time_debug = _get_rllm_flag(config, "time_debug", _env_flag("RLLM_TIME_DEBUG", "RLLM_DEBUG_TIMING"))
    debug_add_env = _get_rllm_flag(config, "debug_add_env", _env_flag("RLLM_DEBUG_ADD_ENV", "RLLM_DEBUG_ENV"))
    debug_resume = _get_verl_flag(config, "debug_resume", _env_flag("VERL_DEBUG_RESUME")) or _get_rllm_flag(
        config, "debug_resume", _env_flag("VERL_DEBUG_RESUME")
    )

    _set_env_bool("RLLM_DEBUG_LORA", debug_lora)
    _set_env_bool("RLLM_DEBUG_BALANCE_BATCH", debug_balance_batch)
    _set_env_bool("RLLM_TIME_DEBUG", time_debug)
    _set_env_bool("RLLM_DEBUG_TIMING", time_debug)
    _set_env_bool("RLLM_DEBUG_ADD_ENV", debug_add_env)
    _set_env_bool("RLLM_DEBUG_ENV", debug_add_env)
    _set_env_bool("VERL_DEBUG_RESUME", debug_resume)

    if debug_lora:
        strategy = getattr(config.actor_rollout_ref.actor, "strategy", None)
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        print(f"[TaskRunner] actor strategy={strategy}, lora_rank={lora_rank}")

    if hasattr(config, "rllm"):
        _set_cfg_bool(config.rllm, "debug_lora", debug_lora)
    if hasattr(config, "verl"):
        _set_cfg_bool(config.verl, "debug_lora", debug_lora)
    if hasattr(config, "actor_rollout_ref"):
        _set_cfg_bool(config.actor_rollout_ref, "debug_lora", debug_lora)
        if hasattr(config.actor_rollout_ref, "actor"):
            _set_cfg_bool(config.actor_rollout_ref.actor, "debug_lora", debug_lora)
        if hasattr(config.actor_rollout_ref, "ref"):
            _set_cfg_bool(config.actor_rollout_ref.ref, "debug_lora", debug_lora)
        if hasattr(config.actor_rollout_ref, "rollout"):
            _set_cfg_bool(config.actor_rollout_ref.rollout, "debug_lora", debug_lora)

    if efficient_mode and multi_agent:
        mode = getattr(config.actor_rollout_ref.actor, "strategy", None)
        if mode not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("Efficient multi-agent mode only supports FSDP or FSDP2.")

    return _ORIG_TASKRUNNER_RUN(
        self,
        config,
        workflow_class=workflow_class,
        workflow_args=workflow_args,
        workflow_trainer_cls=workflow_trainer_cls,
        agent_class=agent_class,
        env_class=env_class,
        agent_args=agent_args,
        env_args=env_args,
        agent_run_func=agent_run_func,
    )


TaskRunner.run = _patched_taskrunner_run


if __name__ == "__main__":
    main()
