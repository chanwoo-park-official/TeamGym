from collections.abc import Callable
from typing import Any, Literal


def _get_rllm_config(config: Any | None) -> Any | None:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("rllm")
    return getattr(config, "rllm", None)


def _get_rllm_flag(config: Any | None, name: str, default: bool = False) -> bool:
    rllm_cfg = _get_rllm_config(config)
    if rllm_cfg is None:
        return default
    if isinstance(rllm_cfg, dict):
        return bool(rllm_cfg.get(name, default))
    return bool(getattr(rllm_cfg, name, default))


def _get_rllm_list(config: Any | None, name: str) -> list[str] | None:
    rllm_cfg = _get_rllm_config(config)
    if rllm_cfg is None:
        return None
    if isinstance(rllm_cfg, dict):
        value = rllm_cfg.get(name)
    else:
        value = getattr(rllm_cfg, name, None)
    return list(value) if value else None


def _set_config_field(config_node: Any | None, key: str, value: Any) -> None:
    if config_node is None:
        return
    if isinstance(config_node, dict):
        config_node[key] = value
        return
    try:
        setattr(config_node, key, value)
        return
    except Exception:
        pass
    try:
        from omegaconf import open_dict

        with open_dict(config_node):
            setattr(config_node, key, value)
    except Exception:
        pass

import ray

from rllm.data import Dataset


class AgentTrainer:
    """
    A wrapper class that allows users to easily train custom agents with custom environments
    without having to directly interact with the underlying training infrastructure.

    Supports two backends:
    - 'verl' (default): Standard training backend supporting both workflow and agent/env classes
    - 'fireworks': Pipeline-based training backend optimized for workflow-based training
    """

    def __init__(
        self,
        workflow_class: type | None = None,
        workflow_args: dict[str, Any] | None = None,
        workflow_trainer_cls: type | None = None,
        agent_class: type | None = None,
        env_class: type | None = None,
        agent_args: dict[str, Any] | None = None,
        env_args: dict[str, Any] | None = None,
        config: dict[str, Any] | list[str] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        val_in_dataset: Dataset | None = None,
        val_out_dataset: Dataset | None = None,
        backend: Literal["verl", "fireworks", "tinker"] = "verl",
        agent_run_func: Callable | None = None,
    ):
        """
        Initialize the AgentTrainer.

        Args:
            workflow_class: The workflow class to use for training
            workflow_args: Optional arguments to pass to the workflow class
            workflow_trainer_cls: Optional workflow trainer override for the verl backend
            agent_class: The custom agent class to use for training
            env_class: The custom environment class to use for training
            agent_args: Optional arguments to pass to the agent class
            env_args: Optional arguments to pass to the environment class
            config: Configuration overrides to apply to the default config
                   Can be a dictionary with dot notation keys (e.g., {"data.train_batch_size": 8})
                   or a list of strings in the format "key=value" (e.g., ["data.train_batch_size=8"])
            train_dataset: Optional train dataset to use
            val_dataset: Optional validation dataset to use (backward-compatible alias of val_in_dataset)
            val_in_dataset: Optional in-distribution validation dataset to use
            val_out_dataset: Optional out-of-distribution validation dataset to use
            backend: Training backend to use ('verl' or 'fireworks'). Default is 'verl'
        """
        # Validate backend
        assert backend in ["verl", "fireworks", "tinker"], f"Unsupported backend: {backend}, must be one of ['verl', 'fireworks', 'tinker']"

        self.backend = backend

        # Validate backend-specific requirements
        if backend == "fireworks":
            if agent_class is not None or env_class is not None:
                raise ValueError("The 'fireworks' backend only supports workflow_class. agent_class and env_class are not supported. Use workflow_args to configure agent and environment.")
            if agent_args is not None or env_args is not None:
                raise ValueError("The 'fireworks' backend does not support agent_args or env_args. Use workflow_args to configure the workflow.")

        if workflow_class is not None and config is not None and hasattr(config, "rllm") and hasattr(config.rllm, "workflow") and config.rllm.workflow.use_workflow:
            if agent_class is not None:
                raise ValueError("agent_class is not supported when using workflow, instead use workflow_args['agent_cls']")
            if agent_args is not None:
                raise ValueError("agent_args is not supported when using workflow, instead use workflow_args['agent_args']")
            if env_class is not None:
                raise ValueError("env_class is not supported when using workflow, instead use workflow_args['env_cls']")
            if env_args is not None:
                raise ValueError("env_args is not supported when using workflow, instead use workflow_args['env_args']")

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self.workflow_trainer_cls = workflow_trainer_cls

        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_args = agent_args or {}
        self.env_args = env_args or {}

        self.agent_run_func = agent_run_func

        self.config = config
        self.train_dataset = train_dataset
        self.val_in_dataset = val_in_dataset or val_dataset
        self.val_out_dataset = val_out_dataset
        # Keep backward-compatible attribute name used by non-workflow trainers.
        self.val_dataset = self.val_in_dataset
        self.backend = backend

        efficient_mode = _get_rllm_flag(self.config, "efficient_mode", False)
        multi_agent = _get_rllm_flag(self.config, "multi_agent", False)
        debug_lora = _get_rllm_flag(self.config, "debug_lora", False)
        if debug_lora:
            print(
                f"[AgentTrainer] backend={self.backend}, efficient_mode={efficient_mode}, multi_agent={multi_agent}"
            )
        if efficient_mode and not multi_agent:
            raise ValueError("efficient_mode=True requires multi_agent=True")
        if efficient_mode and self.backend != "verl":
            raise NotImplementedError("Efficient multi-agent mode only supports VERL backend.")
        
        # Validate lora_adapter_path configuration
        if hasattr(self.config, "actor_rollout_ref") and hasattr(self.config.actor_rollout_ref, "model"):
            lora_adapter_path = self.config.actor_rollout_ref.model.get("lora_adapter_path")
            if lora_adapter_path is not None and isinstance(lora_adapter_path, dict):
                # Dictionary lora_adapter_path requires efficient multi-agent mode
                if not multi_agent or not efficient_mode:
                    raise ValueError(
                        "Dictionary lora_adapter_path (multiple LoRA adapters) requires both "
                        "multi_agent=True and efficient_mode=True. "
                        "Set config.rllm.multi_agent=True and config.rllm.efficient_mode=True."
                    )
        
        # Validate multi_agent without efficient_mode
        if multi_agent and not efficient_mode:
            raise NotImplementedError(
                "multi_agent=True with efficient_mode=False is not currently supported. "
                "This would require multiple Ray actors and ActorRolloutRefWorker instances with different base models, "
                "which is not yet implemented. Please set efficient_mode=True to use multi-agent mode with a single "
                "base model and multiple LoRA adapters."
            )

        if efficient_mode and multi_agent:
            agent_names = _get_rllm_list(self.config, "multi_agent_names") or []
            if agent_names and hasattr(self.config, "actor_rollout_ref"):
                lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
                max_lora_rank = lora_rank if lora_rank and lora_rank > 0 else None
                max_loras = len(agent_names)
                try:
                    from omegaconf import open_dict

                    with open_dict(self.config.actor_rollout_ref.rollout):
                        self.config.actor_rollout_ref.rollout.max_loras = max_loras
                        if max_lora_rank is not None:
                            self.config.actor_rollout_ref.rollout.max_lora_rank = max_lora_rank
                except Exception:
                    try:
                        self.config.actor_rollout_ref.rollout.max_loras = max_loras
                        if max_lora_rank is not None:
                            self.config.actor_rollout_ref.rollout.max_lora_rank = max_lora_rank
                    except Exception:
                        pass

        if train_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.train_files = train_dataset.get_verl_data_path()
        if self.config is not None and hasattr(self.config, "data"):
            if self.val_in_dataset is not None:
                val_in_path = self.val_in_dataset.get_verl_data_path()
                _set_config_field(self.config.data, "val_files", val_in_path)
                _set_config_field(self.config.data, "val_in_files", val_in_path)
            if self.val_out_dataset is not None:
                val_out_path = self.val_out_dataset.get_verl_data_path()
                _set_config_field(self.config.data, "val_out_files", val_out_path)
            else:
                _set_config_field(self.config.data, "val_out_files", None)

    def train(self):
        if self.backend == "verl":
            self._train_verl()
        elif self.backend == "fireworks":
            self._train_fireworks()
        elif self.backend == "tinker":
            self._train_tinker()

    def _train_tinker(self):
        from rllm.trainer.tinker.tinker_agent_trainer import TinkerAgentTrainer
        from rllm.trainer.tinker.tinker_workflow_trainer import TinkerWorkflowTrainer

        if self.workflow_class is not None:
            trainer = TinkerWorkflowTrainer(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
            )
        else:
            trainer = TinkerAgentTrainer(
                config=self.config,
                agent_class=self.agent_class,
                env_class=self.env_class,
                agent_args=self.agent_args,
                env_args=self.env_args,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
            )
        trainer.fit_agent()

    def _train_verl(self):
        """
        Train using the standard verl backend.
        Supports both workflow-based and agent/env-based training.
        """
        from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
        from rllm.trainer.verl.train_agent_ppo import TaskRunner

        # Check if Ray is not initialized
        if not ray.is_initialized():
            # read off all the `ray_init` settings from the config
            if self.config is not None and hasattr(self.config, "ray_init"):
                ray_init_settings = {k: v for k, v in self.config.ray_init.items() if v is not None}
            else:
                ray_init_settings = {}
            ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

        runner = TaskRunner.remote()

        ray.get(
            runner.run.remote(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                workflow_trainer_cls=self.workflow_trainer_cls,
                agent_class=self.agent_class,
                env_class=self.env_class,
                agent_args=self.agent_args,
                env_args=self.env_args,
                agent_run_func=self.agent_run_func,
            )
        )

    def _train_fireworks(self):
        """
        Train using the fireworks (pipeline) backend.
        Optimized for workflow-based training with the Fireworks API.
        """
        if not ray.is_initialized():
            # TODO: check whether we need a separate function to retrieve the runtime environment (for fireworks)
            from verl.trainer.constants_ppo import get_ppo_ray_runtime_env as get_fireworks_ray_runtime_env

            ray.init(runtime_env=get_fireworks_ray_runtime_env(), num_cpus=self.config.ray_init.num_cpus)

        # Lazy import to avoid requiring fireworks package for users who don't use it
        from rllm.trainer.verl.train_workflow_pipeline import PipelineTaskRunner

        runner = PipelineTaskRunner.remote()

        ray.get(
            runner.run.remote(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
            )
        )
