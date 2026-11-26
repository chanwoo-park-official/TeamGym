import asyncio
import json
import math
import os
import threading
import uuid
from collections import Counter, defaultdict
from functools import reduce
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    agg_loss,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.debug import marked_timer

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.utils.episode_logger import EpisodeLogger
from rllm.workflows.workflow import TerminationReason

BASE_LORA_NAME = "__base__"


class AgentWorkflowPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        workflow_class=None,
        workflow_args=None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, processor=processor, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self._validate_config()

        self._loop = asyncio.new_event_loop()

        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _validate_config(self):
        assert self.workflow_class is not None, "workflow_class is required for agent workflow trainer"
        assert self.config.actor_rollout_ref.hybrid_engine is True, "Only hybrid engine is supported"
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported"
        assert self.use_rm is False, "Reward models are not supported. Rewards should be assigned using a reward function in the workflow or environment."
        if self.config.rllm.rejection_sample.multiplier != 1:
            assert self.config.rllm.rejection_sample.enable is True, "rejection sampling is disabled, but rejection_sample.multiplier is not 1"

        # TODO: revisit whether this is now supported by Verl
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("REMAX is not supported yet")

    def _resolve_engine_class(self):
        import importlib

        workflow_cfg = getattr(getattr(self.config, "rllm", None), "workflow", None)
        engine_cls = getattr(workflow_cfg, "engine_cls", None) if workflow_cfg else None
        if engine_cls is None:
            return AgentWorkflowEngine
        if isinstance(engine_cls, str):
            module_path, _, class_name = engine_cls.partition(":")
            if not module_path or not class_name:
                raise ValueError(
                    "rllm.workflow.engine_cls must be in the form 'module.path:ClassName'"
                )
            module = importlib.import_module(module_path)
            try:
                return getattr(module, class_name)
            except AttributeError as exc:
                raise ValueError(f"Engine class not found: {engine_cls}") from exc
        return engine_cls

    def init_workers(self):
        super().init_workers()

        rollout_engine = VerlEngine(
            config=self.config,
            rollout_manager=self.async_rollout_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        # Create episode logger if enabled in config
        episode_logger = None
        if self.config.trainer.get("log_episodes", False):
            # Get episode log directory from config, default to "logs/my_project/my_experiment"
            episode_log_dir = self.config.trainer.get("episode_log_dir", f"logs/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}")
            episode_logger = EpisodeLogger(base_dir=episode_log_dir, subdirectory="episodes")
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

        # init workflow workers
        asyncio.run_coroutine_threadsafe(self.agent_execution_engine.initialize_pool(), self._loop).result()
    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        rllm_cfg = getattr(self.config, "rllm", None)
        verl_cfg = getattr(self.config, "verl", None)
        debug_balance_batch = bool(getattr(rllm_cfg, "debug_balance_batch", False))
        debug_lora = bool(getattr(rllm_cfg, "debug_lora", False)) or bool(
            getattr(verl_cfg, "debug_lora", False)
        )
        debug_resume = bool(getattr(verl_cfg, "debug_resume", False)) or bool(
            getattr(rllm_cfg, "debug_resume", False)
        )
        time_debug = bool(getattr(rllm_cfg, "time_debug", False))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent(epoch=0)
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        if time_debug:
            print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        batch = None
        solve_none = 0
        solve_all = 0
        solve_partial = 0
        num_tasks = 0
        termination_counts = Counter()
        workflow_metrics = defaultdict(list)
        metrics = {}
        timing_raw = {}

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.get("profile_steps") is not None else False
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_tasks += len(new_batch.batch)

                new_batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n)

                new_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])

                # Update training step in engine for episode logging
                self.agent_execution_engine.set_training_step(self.global_steps, mode="train", epoch=epoch)

                with marked_timer("step", timing_raw):

                    # generate trajectories
                    final_gen_batch_output = self.generate_trajectories(batch=new_batch, timing_raw=timing_raw)

                    # need to repeat to make shape match
                    repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                    new_batch = new_batch.sample_level_repeat(repeat_counts)
                    final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                    new_batch = new_batch.union(final_gen_batch_output)
                    if len(new_batch) == 0:
                        zero_repeats = int(np.sum(np.asarray(repeat_counts) == 0))
                        print(
                            "[Warning] No valid trajectories in batch, skipping. "
                            f"repeat_counts_zero={zero_repeats}/{len(repeat_counts)}"
                        )
                        continue

                    model_cfg = self.config.actor_rollout_ref.model
                    uses_lora = bool(model_cfg.get("lora_adapter_path")) or int(
                        model_cfg.get("lora_rank", 0)
                    ) > 0

                    # Extract policy-route names from trajectory_ids if available.
                    # They are also needed without LoRA to remove frozen ``__base__``
                    # rows in participant-only training. Full-model training drops
                    # the names after filtering because Transformers exposes
                    # set_adapter even when PEFT is not loaded.
                    trajectory_ids = new_batch.non_tensor_batch.get("trajectory_ids")
                    if (
                        trajectory_ids is not None
                        and "lora_names" not in new_batch.non_tensor_batch
                    ):
                        lora_names = [str(traj_id).rsplit("_", 1)[-1] for traj_id in trajectory_ids]
                        # Store as object dtype to preserve Python strings for dictionary key lookups
                        new_batch.non_tensor_batch["lora_names"] = np.array(lora_names, dtype=object)
                    lora_names = new_batch.non_tensor_batch.get("lora_names")
                    if lora_names is not None:
                        lora_names_arr = np.asarray(lora_names)
                        keep_idxs = np.where(lora_names_arr != BASE_LORA_NAME)[0]
                        if len(keep_idxs) == 0:
                            continue
                        if len(keep_idxs) < len(lora_names_arr):
                            new_batch = new_batch.select_idxs(keep_idxs)
                    if not uses_lora:
                        new_batch.non_tensor_batch.pop("lora_names", None)

                    # rejection sampling
                    # we do rejection sampling at the episode level instead of the traj/step level
                    uids = new_batch.non_tensor_batch["task_ids"]
                    unique_uids = np.unique(uids)
                    is_correct = new_batch.non_tensor_batch["is_correct"]
                    drop_uids = set()

                    for uid in unique_uids:
                        candidate_rows = uids == uid
                        candidate_is_correct = is_correct[candidate_rows]

                        # Check if all episodes are correct or incorrect
                        if not candidate_is_correct.any():
                            drop_uids.add(uid)
                            solve_none += 1
                        elif candidate_is_correct.all():
                            drop_uids.add(uid)
                            solve_all += 1
                        else:
                            solve_partial += 1

                    # Build a view with a single item per episode_id for metrics/logging
                    seen_episodes = set()
                    episode_unique_idxs = []
                    for i, episode_id in enumerate(new_batch.non_tensor_batch["episode_ids"]):
                        if episode_id not in seen_episodes:
                            seen_episodes.add(episode_id)
                            episode_unique_idxs.append(i)
                    episode_unique_batch = new_batch.select_idxs(episode_unique_idxs)

                    # log metrics returned by workflows
                    for metric_dict in episode_unique_batch.non_tensor_batch["metrics"]:
                        for key, value in metric_dict.items():
                            workflow_metrics[key].append(value)

                    # collect and log termination reasons
                    termination_reasons = episode_unique_batch.non_tensor_batch["termination_reasons"]
                    termination_counts.update(termination_reasons)

                    # If no valid samples remain, skip this batch and get a new one
                    # if len(drop_uids) == len(unique_uids):
                    #     print("No valid samples remain, skipping batch")
                    #     continue

                    if not self.config.rllm.rejection_sample.enable:
                        batch = new_batch
                    else:
                        rejection_mask = np.isin(uids, list(drop_uids))
                        new_batch = new_batch[~rejection_mask]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        if solve_partial < self.config.data.train_batch_size:
                            continue
                        else:
                            # randomly select bsz task uids from batch, then filter batch to only contain these tasks
                            # TODO: add heuristic for selecting train_batch_size uids
                            uids = batch.non_tensor_batch["task_ids"]
                            unique_uids = np.unique(uids)
                            assert len(unique_uids) >= self.config.data.train_batch_size, "Not enough unique uids to sample from"
                            selected_uids = np.random.choice(unique_uids, size=self.config.data.train_batch_size, replace=False)
                            selected_mask = np.isin(uids, selected_uids)
                            batch = batch[selected_mask]

                    if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                        # need to make sure both number of last steps (number of uids) and number of total steps in the batch
                        # (batch size after processing) are both multiples of world size

                        # first we split the batch in two: one with only the last steps of each trajectory and the other with the remaining steps
                        is_last_step = batch.non_tensor_batch["is_last_step"]
                        valid_last_step_indices = np.where(is_last_step == True)[0]
                        not_last_step_indices = np.where(is_last_step == False)[0]
                        last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                        non_last_step_batch = batch.select_idxs(not_last_step_indices)

                        # round down last_step_batch to make sure its multiple of world size
                        num_trainer_replicas = self.actor_rollout_wg.world_size
                        max_batch_size = (last_step_batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas

                        size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                        size_mask[:max_batch_size] = True
                        last_step_batch = last_step_batch[size_mask]  # filtered last steps

                        # now we go through all the non_last_step_batch and keep everything that has same trajectory_id that exists in the filtered last steps
                        valid_last_step_trajectory_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
                        non_last_step_trajectory_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
                        non_last_step_mask = np.isin(non_last_step_trajectory_ids, valid_last_step_trajectory_ids)
                        non_last_step_batch = non_last_step_batch[non_last_step_mask]

                        # concatenate then pad
                        batch = DataProto.concat([last_step_batch, non_last_step_batch])
                        batch = self._pad_dataproto_to_world_size(batch)

                    else:
                        # then we just pad the batch size to a multiple of world size
                        batch = self._pad_dataproto_to_world_size(batch=batch)

                    if debug_balance_batch:
                        print(
                            f"[DEBUG] About to call _balance_batch, "
                            f"balance_batch={self.config.trainer.balance_batch}"
                        )
                    # Balance batch to ensure adapter-aware distribution
                    if self.config.trainer.balance_batch:
                        if debug_balance_batch:
                            print(f"[DEBUG] Calling _balance_batch now")
                        self._balance_batch(batch, metrics=metrics)
                        if debug_balance_batch:
                            print(f"[DEBUG] _balance_batch completed")

                    # recompute old_log_probs
                    if debug_balance_batch:
                        print(
                            "[DEBUG] About to call compute_log_prob, "
                            f"batch size={len(batch.batch['input_ids'])}"
                        )
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        if debug_balance_batch:
                            print(f"[DEBUG] Calling compute_log_prob now")
                        
                        # CRITICAL FIX for FSDP with multi-adapter:
                        # FSDP requires all workers to process the same adapter simultaneously
                        # because they share sharded parameters and need synchronized collective ops.
                        # Solution: Process each adapter sequentially, not in parallel.
                        
                        lora_names = batch.non_tensor_batch.get("lora_names")
                        if lora_names is not None:
                            lora_names_arr = np.asarray(lora_names)
                            unique_adapters = list(dict.fromkeys(lora_names_arr.tolist()))
                            if debug_balance_batch:
                                print(
                                    "[DEBUG] Multi-adapter mode: processing "
                                    f"{len(unique_adapters)} adapters sequentially: {unique_adapters}"
                                )
                            
                            # Process each adapter separately to avoid FSDP deadlock
                            old_log_prob_chunks = []
                            for adapter_name in unique_adapters:
                                adapter_indices = np.where(lora_names_arr == adapter_name)[0]
                                if debug_balance_batch:
                                    print(
                                        f"[DEBUG] Processing adapter '{adapter_name}' "
                                        f"with {len(adapter_indices)} samples"
                                    )
                                
                                # Extract samples for this adapter
                                adapter_batch = batch.select_idxs(adapter_indices)
                                
                                # All workers process this adapter together
                                adapter_old_log_prob = self.actor_rollout_wg.compute_log_prob(adapter_batch)
                                old_log_prob_chunks.append((adapter_indices, adapter_old_log_prob))
                            
                            # Reconstruct the full batch results in original order
                            if debug_balance_batch:
                                print(
                                    "[DEBUG] Reconstructing full batch from "
                                    f"{len(old_log_prob_chunks)} adapter chunks"
                                )
                            all_log_probs = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                            all_entropys = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                            
                            for adapter_indices, adapter_result in old_log_prob_chunks:
                                all_log_probs[adapter_indices] = adapter_result.batch["old_log_probs"]
                                all_entropys[adapter_indices] = adapter_result.batch["entropys"]
                            
                            # Create combined result
                            old_log_prob = DataProto.from_dict(
                                tensors={"old_log_probs": all_log_probs, "entropys": all_entropys},
                                meta_info={"temperature": adapter_result.meta_info.get("temperature", 1.0)},
                            )
                        else:
                            # Single adapter or no adapter - process normally
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        
                        if debug_balance_batch:
                            print(f"[DEBUG] compute_log_prob returned")
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        lora_names = batch.non_tensor_batch.get("lora_names")
                        if lora_names is not None:
                            lora_names = np.asarray(lora_names)
                            unique_names = list(dict.fromkeys(lora_names.tolist()))
                            mask = response_masks.float()
                            denom = torch.clamp(mask.sum(dim=1), min=1.0)
                            per_sample_entropy = (entropys * mask).sum(dim=1) / denom
                            for adapter_name in unique_names:
                                idxs = np.where(lora_names == adapter_name)[0]
                                if len(idxs) == 0:
                                    continue
                                metrics[f"actor/{adapter_name}/entropy"] = (
                                    per_sample_entropy[idxs].mean().detach().item()
                                )
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                lora_names = batch.non_tensor_batch.get("lora_names")
                                if lora_names is not None:
                                    lora_names_arr = np.asarray(lora_names)
                                    unique_adapters = list(dict.fromkeys(lora_names_arr.tolist()))
                                    if len(unique_adapters) > 1:
                                        if debug_balance_batch:
                                            print(
                                                "[DEBUG] Multi-adapter ref path: processing "
                                                f"{len(unique_adapters)} adapters sequentially: {unique_adapters}"
                                            )
                                        ref_log_prob_chunks = []
                                        for adapter_name in unique_adapters:
                                            adapter_indices = np.where(lora_names_arr == adapter_name)[0]
                                            adapter_batch = batch.select_idxs(adapter_indices)
                                            adapter_ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(
                                                adapter_batch
                                            )
                                            ref_log_prob_chunks.append((adapter_indices, adapter_ref_log_prob))

                                        all_ref_log_probs = torch.zeros_like(
                                            batch.batch["response_mask"], dtype=torch.float32
                                        )
                                        for adapter_indices, adapter_result in ref_log_prob_chunks:
                                            all_ref_log_probs[adapter_indices] = adapter_result.batch["ref_log_prob"]

                                        ref_log_prob = DataProto.from_dict(
                                            tensors={"ref_log_prob": all_ref_log_probs}
                                        )
                                    else:
                                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # step_ids is safe to always use for advantage computation
                        # if we're not using computing advantages stepwise (i.e., for cumulative agents or single turn workflows)
                        # then step_ids == trajectory_ids
                        # Keep as object array so later uid rewrites never get silently
                        # truncated by fixed-width NumPy string dtypes.
                        batch.non_tensor_batch["uid"] = np.asarray(
                            batch.non_tensor_batch["step_ids"], dtype=object
                        )

                        if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "per_step":
                            batch.batch["token_level_scores"] = batch.batch["step_rewards"]
                        else:
                            batch.batch["token_level_scores"] = batch.batch["traj_rewards"]

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                            is_last_step = batch.non_tensor_batch["is_last_step"]
                            last_step_indices = np.where(is_last_step == True)[0]
                            not_last_step_indices = np.where(is_last_step == False)[0]
                            non_last_step_batch = batch.select_idxs(not_last_step_indices)
                            batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
                            # last_step_batch contains no padded steps as it was rounded down (not padded) to a multiple of world size
                        else:
                            batch = self._remove_padding(batch)  # compute advantages over non-padded steps only

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        sat_regret_metrics = getattr(self, "_sat_last_regret_metrics", None)
                        if sat_regret_metrics:
                            metrics.update(sat_regret_metrics)
                            self._sat_last_regret_metrics = None

                        if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(batch, non_last_step_batch)
                            batch = DataProto.concat([batch, non_last_step_batch])

                    self._maybe_dump_training_examples(
                        batch=batch,
                        epoch=epoch,
                        stage="post_advantage_pre_filter",
                        timing_raw=timing_raw,
                    )

                    # remove invalid items filtered out due to compact filtering
                    is_valid = batch.non_tensor_batch["is_valid"]
                    valid_idxs = np.where(is_valid == True)[0]
                    batch = batch.select_idxs(valid_idxs)

                    # for backward compatibility
                    if self.config.rllm.mask_truncated_samples:
                        mask = batch.batch["attention_mask"][:, -1] == 1
                        batch = batch[~mask]

                    # re-pad batch size to world size for gradient update
                    batch = self._pad_dataproto_to_world_size(batch=batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            # CRITICAL FIX for FSDP with multi-adapter:
                            # Process each adapter sequentially to avoid FSDP deadlock
                            lora_names = batch.non_tensor_batch.get("lora_names")
                            if lora_names is not None:
                                lora_names_arr = np.asarray(lora_names)
                                unique_adapters = list(dict.fromkeys(lora_names_arr.tolist()))
                                
                                # Process each adapter separately
                                all_actor_outputs = []
                                for adapter_name in unique_adapters:
                                    adapter_indices = np.where(lora_names_arr == adapter_name)[0]
                                    
                                    # Extract samples for this adapter
                                    adapter_batch = batch.select_idxs(adapter_indices)
                                    
                                    # All workers update this adapter together
                                    adapter_actor_output = self.actor_rollout_wg.update_actor(adapter_batch)
                                    all_actor_outputs.append(adapter_actor_output)
                                
                                # Merge metrics from all adapters
                                if debug_balance_batch:
                                    print(
                                        "[DEBUG] update_actor: Merging metrics from "
                                        f"{len(all_actor_outputs)} adapter outputs"
                                    )
                                def _flatten_metric_values(values):
                                    if isinstance(values, (list, tuple)):
                                        flat = []
                                        for value in values:
                                            if isinstance(value, (list, tuple)):
                                                flat.extend(_flatten_metric_values(value))
                                            else:
                                                flat.append(value)
                                        return flat
                                    return [values]

                                merged_metrics = {}
                                for adapter_output in all_actor_outputs:
                                    adapter_metrics = adapter_output.meta_info.get("metrics", {})
                                    for key, values in adapter_metrics.items():
                                        if key not in merged_metrics:
                                            merged_metrics[key] = []
                                        merged_metrics[key].extend(_flatten_metric_values(values))
                                
                                actor_output_metrics = reduce_metrics(merged_metrics)
                            else:
                                # Single adapter or no adapter - process normally
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate_agent(epoch=epoch)
                        metrics.update(val_metrics)

                    # Debug checkpoint saving
                    save_freq = self.config.trainer.save_freq
                    if save_freq > 0:
                        steps_until_save = save_freq - (self.global_steps % save_freq)
                        if steps_until_save == save_freq:
                            steps_until_save = 0
                        if debug_resume:
                            print(
                                f"[DEBUG] Checkpoint: current_step={self.global_steps}, "
                                f"save_freq={save_freq}, "
                                "next_save_at_step="
                                f"{self.global_steps + steps_until_save if steps_until_save > 0 else self.global_steps}",
                                flush=True,
                            )
                        
                        if self.global_steps % save_freq == 0:
                            if debug_resume:
                                print(
                                    f"[DEBUG] SAVING CHECKPOINT NOW at step {self.global_steps}",
                                    flush=True,
                                )
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()
                            if debug_resume:
                                print(
                                    f"[DEBUG] Checkpoint saved successfully at step {self.global_steps}",
                                    flush=True,
                                )

                    # Visualize some sample trajectories
                    if (
                        os.environ.get(
                            "RLLM_DISABLE_TRAJECTORY_VISUALIZATION", ""
                        ).strip().lower()
                        not in {"1", "true", "yes", "on"}
                        and batch is not None
                        and len(batch) > 0
                    ):
                        # Randomly select a few samples to visualize
                        batch_size = len(batch)
                        num_samples = min(2, batch_size)  # Visualize up to 2 samples
                        if num_samples > 0:
                            sample_indices = np.random.choice(batch_size, size=num_samples, replace=False)
                            for idx in sample_indices:
                                self.visualize_trajectory_last_step(batch, sample_idx=idx, max_samples=1)

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                metrics["batch/solve_none"] = solve_none / num_tasks
                metrics["batch/solve_all"] = solve_all / num_tasks
                metrics["batch/solve_partial"] = solve_partial / num_tasks

                for key, value in workflow_metrics.items():
                    values = np.asarray(value, dtype=float)
                    metrics[f"batch/{key}"] = float(np.mean(values))
                    if key.startswith(("solver_prompt_tokens", "solver_response_tokens")):
                        metrics[f"batch/{key}/min"] = float(np.min(values))
                        metrics[f"batch/{key}/max"] = float(np.max(values))
                    if key.startswith(("teamgym_reward", "teamgym_task_success")):
                        metrics[f"batch/{key}/min"] = float(np.min(values))
                        metrics[f"batch/{key}/max"] = float(np.max(values))
                        metrics[f"batch/{key}/std"] = float(np.std(values))

                def _sum_workflow_metric(metric_key: str) -> float:
                    values = workflow_metrics.get(metric_key, [])
                    if not values:
                        return 0.0
                    return float(np.sum(np.asarray(values, dtype=float)))

                solver_action_count_sum = _sum_workflow_metric("solver_action_count")
                solver_invalid_action_count_sum = _sum_workflow_metric(
                    "solver_invalid_action_count"
                )
                solver_successful_action_count_sum = _sum_workflow_metric(
                    "solver_successful_action_count"
                )
                solver_valid_action_count_sum = _sum_workflow_metric(
                    "solver_valid_action_count"
                )
                if "solver_valid_action_count" not in workflow_metrics:
                    solver_valid_action_count_sum = solver_successful_action_count_sum
                solver_invalid_step_count_sum = _sum_workflow_metric(
                    "solver_invalid_step_count"
                )
                solver_invalid_action_penalized_count_sum = _sum_workflow_metric(
                    "solver_invalid_action_penalized_count"
                )
                solver_invalid_action_unpenalized_count_sum = _sum_workflow_metric(
                    "solver_invalid_action_unpenalized_count"
                )
                if (
                    solver_action_count_sum > 0.0
                    or "solver_action_count" in workflow_metrics
                ):
                    metrics["batch/solver_action_count"] = solver_action_count_sum
                    metrics["batch/solver_invalid_action_count"] = (
                        solver_invalid_action_count_sum
                    )
                    metrics["batch/solver_successful_action_count"] = (
                        solver_successful_action_count_sum
                    )
                    metrics["batch/solver_valid_action_count"] = (
                        solver_valid_action_count_sum
                    )
                    metrics["batch/solver_invalid_action_ratio"] = (
                        solver_invalid_action_count_sum / solver_action_count_sum
                        if solver_action_count_sum > 0.0
                        else 0.0
                    )
                    metrics["batch/solver_successful_action_ratio"] = (
                        solver_successful_action_count_sum / solver_action_count_sum
                        if solver_action_count_sum > 0.0
                        else 0.0
                    )
                    metrics["batch/solver_valid_action_ratio"] = (
                        solver_valid_action_count_sum / solver_action_count_sum
                        if solver_action_count_sum > 0.0
                        else 0.0
                    )
                    metrics["batch/solver_invalid_step_count"] = (
                        solver_invalid_step_count_sum
                    )
                    metrics["batch/solver_invalid_step_ratio"] = (
                        solver_invalid_step_count_sum / solver_action_count_sum
                        if solver_action_count_sum > 0.0
                        else 0.0
                    )
                    metrics["batch/solver_invalid_action_penalized_count"] = (
                        solver_invalid_action_penalized_count_sum
                    )
                    metrics["batch/solver_invalid_action_unpenalized_count"] = (
                        solver_invalid_action_unpenalized_count_sum
                    )
                    metrics["batch/solver_invalid_action_penalized_ratio"] = (
                        solver_invalid_action_penalized_count_sum
                        / solver_invalid_action_count_sum
                        if solver_invalid_action_count_sum > 0.0
                        else 0.0
                    )
                    metrics["batch/solver_invalid_action_unpenalized_ratio"] = (
                        solver_invalid_action_unpenalized_count_sum
                        / solver_invalid_action_count_sum
                        if solver_invalid_action_count_sum > 0.0
                        else 0.0
                    )

                auditor_step_count_sum = _sum_workflow_metric("auditor_step_count")
                auditor_valid_action_count_sum = _sum_workflow_metric(
                    "auditor_valid_action_count"
                )
                if "auditor_valid_action_count" not in workflow_metrics:
                    auditor_valid_action_count_sum = _sum_workflow_metric(
                        "auditor_format_ok_count"
                    )
                auditor_invalid_action_count_sum = _sum_workflow_metric(
                    "auditor_invalid_action_count"
                )
                if "auditor_invalid_action_count" not in workflow_metrics:
                    auditor_invalid_action_count_sum = max(
                        0.0, auditor_step_count_sum - auditor_valid_action_count_sum
                    )
                if (
                    auditor_step_count_sum > 0.0
                    or "auditor_step_count" in workflow_metrics
                ):
                    metrics["batch/auditor_step_count"] = auditor_step_count_sum
                    metrics["batch/auditor_valid_action_count"] = (
                        auditor_valid_action_count_sum
                    )
                    metrics["batch/auditor_invalid_action_count"] = (
                        auditor_invalid_action_count_sum
                    )
                    metrics["batch/auditor_valid_action_ratio"] = (
                        auditor_valid_action_count_sum / auditor_step_count_sum
                        if auditor_step_count_sum > 0.0
                        else 0.0
                    )
                    metrics["batch/auditor_invalid_action_ratio"] = (
                        auditor_invalid_action_count_sum / auditor_step_count_sum
                        if auditor_step_count_sum > 0.0
                        else 0.0
                    )

                if "solver_turn_with_limit_count" in workflow_metrics:
                    solver_turn_with_limit_count_sum = _sum_workflow_metric(
                        "solver_turn_with_limit_count"
                    )
                    solver_turn_not_reaching_count_sum = _sum_workflow_metric(
                        "solver_turn_not_reaching_max_minus_3_count"
                    )
                    solver_turn_reaching_count_sum = _sum_workflow_metric(
                        "solver_turn_reaching_max_minus_3_count"
                    )
                    metrics["batch/solver_turn_with_limit_count"] = (
                        solver_turn_with_limit_count_sum
                    )
                    metrics["batch/solver_turn_not_reaching_max_minus_3_count"] = (
                        solver_turn_not_reaching_count_sum
                    )
                    metrics["batch/solver_turn_reaching_max_minus_3_count"] = (
                        solver_turn_reaching_count_sum
                    )
                    metrics["batch/solver_turn_not_reaching_max_minus_3_ratio"] = (
                        solver_turn_not_reaching_count_sum
                        / solver_turn_with_limit_count_sum
                        if solver_turn_with_limit_count_sum > 0.0
                        else 0.0
                    )
                    metrics["batch/solver_turn_reaching_max_minus_3_ratio"] = (
                        solver_turn_reaching_count_sum
                        / solver_turn_with_limit_count_sum
                        if solver_turn_with_limit_count_sum > 0.0
                        else 0.0
                    )

                for r in TerminationReason:
                    metrics[f"batch/{r.value}"] = termination_counts[r.value] / len(set(new_batch.non_tensor_batch["episode_ids"]))

                metrics["batch/num_tasks"] = num_tasks

                lora_names = None
                if batch is not None:
                    lora_names = batch.non_tensor_batch.get("lora_names")
                reward_key = (
                    "step_rewards"
                    if self.config.rllm.stepwise_advantage.enable
                    and self.config.rllm.stepwise_advantage.mode == "per_step"
                    else "traj_rewards"
                )
                if batch is not None and reward_key in batch.batch:
                    rewards = batch.batch[reward_key]
                    if rewards.dim() > 1 and "response_mask" in batch.batch:
                        mask = batch.batch["response_mask"]
                        per_sample = (rewards * mask).sum(-1) / mask.sum(-1).clamp(min=1)
                    else:
                        per_sample = rewards.view(rewards.shape[0], -1).mean(-1)
                    if lora_names is not None:
                        for name in dict.fromkeys(lora_names.tolist()):
                            idxs = (lora_names == name)
                            if idxs.any():
                                metrics[f"reward/{name}"] = per_sample[idxs].mean().item()
                    trajectory_ids = batch.non_tensor_batch.get("trajectory_ids")
                    if trajectory_ids is not None:
                        role_names = np.asarray(
                            [str(traj_id).rsplit("_", 1)[-1] for traj_id in trajectory_ids],
                            dtype=object,
                        )
                        for role in ("solver", "auditor"):
                            idxs = role_names == role
                            if idxs.any():
                                metrics[f"reward/{role}"] = per_sample[idxs].mean().item()

                if debug_lora:
                    lora_names = None
                    if batch is not None:
                        lora_names = batch.non_tensor_batch.get("lora_names")
                    adapters = sorted(set(lora_names.tolist())) if lora_names is not None else []
                    deltas = {}
                    for name in adapters:
                        key = f"actor/lora_delta/{name}"
                        if key in metrics:
                            val = metrics[key]
                            if isinstance(val, (list, tuple)):
                                val = float(sum(val) / max(len(val), 1))
                            deltas[name] = val
                    if adapters or deltas:
                        print(f"[LoRA][driver] adapters={adapters} deltas={deltas}")

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                batch = None
                solve_none = 0
                solve_all = 0
                solve_partial = 0
                num_tasks = 0
                termination_counts = Counter()
                workflow_metrics = defaultdict(list)
                metrics = {}
                timing_raw = {}

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent(epoch=epoch)
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _build_split_val_dataloader(self, val_files):
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl.trainer.main_ppo import create_rl_dataset

        if not val_files:
            return None

        val_dataset = create_rl_dataset(
            val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            max_samples=self.config.data.get("val_max_samples", -1),
        )
        if val_dataset is None or len(val_dataset) == 0:
            return None

        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(val_dataset)

        collate_fn = getattr(self.val_dataloader, "collate_fn", None)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        return StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data["dataloader_num_workers"],
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

    def _save_validation_result(self, split: str, epoch: int, metrics: dict, summary: dict):
        episode_log_dir = self.config.trainer.get(
            "episode_log_dir",
            f"logs/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}",
        )
        result_dir = os.path.join(episode_log_dir, "validation_results", split)
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"step_{self.global_steps}_epoch_{epoch}.json")
        payload = {
            "step": int(self.global_steps),
            "epoch": int(epoch),
            "split": split,
            "metrics": metrics,
            "summary": summary,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"[Validation] Saved {split} result to {result_path}")

    def _validate_agent_single(self, val_dataloader, metric_prefix: str):
        is_correct_lst = []
        data_source_lst = []
        uid_lst = []
        workflow_metrics_by_source = defaultdict(lambda: defaultdict(list))
        pass_rates_by_uid = defaultdict(list)
        n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n

        def _sum_metric(metric_store: dict[str, list[float]], key: str) -> float:
            values = metric_store.get(key, [])
            if not values:
                return 0.0
            return float(np.sum(np.asarray(values, dtype=float)))

        for test_data in val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)

            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)

            test_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {"validate": True}

            test_output_gen_batch = self.generate_trajectories(batch=test_batch)
            repeat_counts = test_output_gen_batch.meta_info["repeat_counts"]
            # need to repeat to make shape match
            test_batch = test_batch.sample_level_repeat(repeat_counts)
            test_output_gen_batch.meta_info.pop("repeat_counts", None)  # no longer needed after this
            test_batch = test_batch.union(test_output_gen_batch)

            seen_episodes = set()
            selected_idxs = []
            for i, episode_id in enumerate(test_batch.non_tensor_batch["episode_ids"]):
                if episode_id not in seen_episodes:
                    seen_episodes.add(episode_id)
                    selected_idxs.append(i)
            test_batch = test_batch.select_idxs(selected_idxs)

            is_correct_lst.extend(test_batch.non_tensor_batch["is_correct"])
            uid_lst.extend(test_batch.non_tensor_batch["task_ids"])
            for is_correct, uid in zip(test_batch.non_tensor_batch["is_correct"], test_batch.non_tensor_batch["task_ids"], strict=False):
                pass_rates_by_uid[uid].append(bool(is_correct))

            data_sources = test_batch.non_tensor_batch.get("data_source", None)
            if data_sources is None:
                data_sources = ["unknown"] * len(test_batch)
            data_source_lst.extend(data_sources)

            # Collect workflow metrics per episode and data source
            for i, data_source in enumerate(data_sources):
                episode_metrics = test_batch.non_tensor_batch["metrics"][i]
                if episode_metrics is not None:
                    for key, value in episode_metrics.items():
                        workflow_metrics_by_source[data_source][key].append(float(value))

        metrics = {}
        if len(is_correct_lst) == 0:
            summary = {
                "num_episodes": 0,
                "num_tasks": 0,
                "num_correct": 0,
                "num_tasks_solved": 0,
                "pass@1": 0.0,
                f"pass@{n_val_samples}": 0.0,
            }
            metrics[f"{metric_prefix}/summary/num_episodes"] = 0.0
            metrics[f"{metric_prefix}/summary/num_tasks"] = 0.0
            metrics[f"{metric_prefix}/summary/pass@1"] = 0.0
            metrics[f"{metric_prefix}/summary/pass@{n_val_samples}"] = 0.0
            return metrics, summary

        is_correct_array = np.array(is_correct_lst)
        uid_array = np.array(uid_lst)
        data_source_array = np.array(data_source_lst)

        for data_source in np.unique(data_source_array):
            pass_rates = defaultdict(list)

            data_source_mask = data_source_array == data_source
            is_correct_data_source = is_correct_array[data_source_mask]
            uids_data_source = uid_array[data_source_mask]

            for is_correct, uid in zip(is_correct_data_source, uids_data_source, strict=False):
                pass_rates[uid].append(is_correct)

            metrics[f"{metric_prefix}/{data_source}/pass@1"] = np.mean(is_correct_data_source)
            metrics[f"{metric_prefix}/{data_source}/pass@{n_val_samples}"] = np.mean([1 if any(pass_rate) else 0 for pass_rate in pass_rates.values()])

            # Add workflow metrics for this data source
            if data_source in workflow_metrics_by_source:
                source_metrics = workflow_metrics_by_source[data_source]
                for key, values in source_metrics.items():
                    if values:  # Only add if we have values
                        metrics[f"{metric_prefix}/{data_source}/{key}"] = np.mean(values)

                solver_action_count_sum = _sum_metric(source_metrics, "solver_action_count")
                solver_invalid_action_count_sum = _sum_metric(
                    source_metrics, "solver_invalid_action_count"
                )
                solver_successful_action_count_sum = _sum_metric(
                    source_metrics, "solver_successful_action_count"
                )
                solver_valid_action_count_sum = _sum_metric(
                    source_metrics, "solver_valid_action_count"
                )
                if "solver_valid_action_count" not in source_metrics:
                    solver_valid_action_count_sum = solver_successful_action_count_sum
                solver_invalid_step_count_sum = _sum_metric(
                    source_metrics, "solver_invalid_step_count"
                )
                solver_invalid_action_penalized_count_sum = _sum_metric(
                    source_metrics, "solver_invalid_action_penalized_count"
                )
                solver_invalid_action_unpenalized_count_sum = _sum_metric(
                    source_metrics, "solver_invalid_action_unpenalized_count"
                )
                if (
                    solver_action_count_sum > 0.0
                    or "solver_action_count" in source_metrics
                ):
                    metrics[f"{metric_prefix}/{data_source}/solver_action_count"] = (
                        solver_action_count_sum
                    )
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_invalid_action_count"
                    ] = solver_invalid_action_count_sum
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_successful_action_count"
                    ] = solver_successful_action_count_sum
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_valid_action_count"
                    ] = solver_valid_action_count_sum
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_invalid_action_ratio"
                    ] = (
                        solver_invalid_action_count_sum / solver_action_count_sum
                        if solver_action_count_sum > 0.0
                        else 0.0
                    )
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_successful_action_ratio"
                    ] = (
                        solver_successful_action_count_sum / solver_action_count_sum
                        if solver_action_count_sum > 0.0
                        else 0.0
                    )
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_valid_action_ratio"
                    ] = (
                        solver_valid_action_count_sum / solver_action_count_sum
                        if solver_action_count_sum > 0.0
                        else 0.0
                    )
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_invalid_step_count"
                    ] = solver_invalid_step_count_sum
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_invalid_step_ratio"
                    ] = (
                        solver_invalid_step_count_sum / solver_action_count_sum
                        if solver_action_count_sum > 0.0
                        else 0.0
                    )
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_invalid_action_penalized_count"
                    ] = solver_invalid_action_penalized_count_sum
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_invalid_action_unpenalized_count"
                    ] = solver_invalid_action_unpenalized_count_sum
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_invalid_action_penalized_ratio"
                    ] = (
                        solver_invalid_action_penalized_count_sum
                        / solver_invalid_action_count_sum
                        if solver_invalid_action_count_sum > 0.0
                        else 0.0
                    )
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_invalid_action_unpenalized_ratio"
                    ] = (
                        solver_invalid_action_unpenalized_count_sum
                        / solver_invalid_action_count_sum
                        if solver_invalid_action_count_sum > 0.0
                        else 0.0
                    )

                auditor_step_count_sum = _sum_metric(
                    source_metrics, "auditor_step_count"
                )
                auditor_valid_action_count_sum = _sum_metric(
                    source_metrics, "auditor_valid_action_count"
                )
                if "auditor_valid_action_count" not in source_metrics:
                    auditor_valid_action_count_sum = _sum_metric(
                        source_metrics, "auditor_format_ok_count"
                    )
                auditor_invalid_action_count_sum = _sum_metric(
                    source_metrics, "auditor_invalid_action_count"
                )
                if "auditor_invalid_action_count" not in source_metrics:
                    auditor_invalid_action_count_sum = max(
                        0.0, auditor_step_count_sum - auditor_valid_action_count_sum
                    )
                if (
                    auditor_step_count_sum > 0.0
                    or "auditor_step_count" in source_metrics
                ):
                    metrics[f"{metric_prefix}/{data_source}/auditor_step_count"] = (
                        auditor_step_count_sum
                    )
                    metrics[
                        f"{metric_prefix}/{data_source}/auditor_valid_action_count"
                    ] = auditor_valid_action_count_sum
                    metrics[
                        f"{metric_prefix}/{data_source}/auditor_invalid_action_count"
                    ] = auditor_invalid_action_count_sum
                    metrics[
                        f"{metric_prefix}/{data_source}/auditor_valid_action_ratio"
                    ] = (
                        auditor_valid_action_count_sum / auditor_step_count_sum
                        if auditor_step_count_sum > 0.0
                        else 0.0
                    )
                    metrics[
                        f"{metric_prefix}/{data_source}/auditor_invalid_action_ratio"
                    ] = (
                        auditor_invalid_action_count_sum / auditor_step_count_sum
                        if auditor_step_count_sum > 0.0
                        else 0.0
                    )

                if "solver_turn_with_limit_count" in source_metrics:
                    solver_turn_with_limit_count_sum = _sum_metric(
                        source_metrics, "solver_turn_with_limit_count"
                    )
                    solver_turn_not_reaching_count_sum = _sum_metric(
                        source_metrics, "solver_turn_not_reaching_max_minus_3_count"
                    )
                    solver_turn_reaching_count_sum = _sum_metric(
                        source_metrics, "solver_turn_reaching_max_minus_3_count"
                    )
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_turn_with_limit_count"
                    ] = solver_turn_with_limit_count_sum
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_turn_not_reaching_max_minus_3_count"
                    ] = solver_turn_not_reaching_count_sum
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_turn_reaching_max_minus_3_count"
                    ] = solver_turn_reaching_count_sum
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_turn_not_reaching_max_minus_3_ratio"
                    ] = (
                        solver_turn_not_reaching_count_sum
                        / solver_turn_with_limit_count_sum
                        if solver_turn_with_limit_count_sum > 0.0
                        else 0.0
                    )
                    metrics[
                        f"{metric_prefix}/{data_source}/solver_turn_reaching_max_minus_3_ratio"
                    ] = (
                        solver_turn_reaching_count_sum
                        / solver_turn_with_limit_count_sum
                        if solver_turn_with_limit_count_sum > 0.0
                        else 0.0
                    )

        num_episodes = int(len(is_correct_array))
        num_correct = int(np.sum(is_correct_array))
        num_tasks = int(len(pass_rates_by_uid))
        num_tasks_solved = int(sum(1 if any(pass_rate) else 0 for pass_rate in pass_rates_by_uid.values()))
        summary = {
            "num_episodes": num_episodes,
            "num_tasks": num_tasks,
            "num_correct": num_correct,
            "num_tasks_solved": num_tasks_solved,
            "pass@1": float(num_correct / num_episodes) if num_episodes > 0 else 0.0,
            f"pass@{n_val_samples}": float(num_tasks_solved / num_tasks) if num_tasks > 0 else 0.0,
        }
        metrics[f"{metric_prefix}/summary/num_episodes"] = float(num_episodes)
        metrics[f"{metric_prefix}/summary/num_tasks"] = float(num_tasks)
        metrics[f"{metric_prefix}/summary/pass@1"] = summary["pass@1"]
        metrics[f"{metric_prefix}/summary/pass@{n_val_samples}"] = summary[f"pass@{n_val_samples}"]
        return metrics, summary

    def _validate_agent(self, epoch: int = 0):
        all_metrics = {}

        # in distribution validation
        self.agent_execution_engine.set_training_step(self.global_steps, mode="val_in", epoch=epoch)
        in_metrics, in_summary = self._validate_agent_single(self.val_dataloader, metric_prefix="val/in")
        all_metrics.update(in_metrics)
        self._save_validation_result("in", epoch, in_metrics, in_summary)

        # out of distribution validation
        val_out_files = self.config.data.get("val_out_files", None)
        if not val_out_files:
            return all_metrics
        if not hasattr(self, "_val_out_dataloader"):
            self._val_out_dataloader = self._build_split_val_dataloader(val_out_files)
        if self._val_out_dataloader is None:
            print(f"[Warning] val_out_files is configured but no dataloader could be created: {val_out_files}")
            return all_metrics

        self.agent_execution_engine.set_training_step(self.global_steps, mode="val_out", epoch=epoch)
        out_metrics, out_summary = self._validate_agent_single(self._val_out_dataloader, metric_prefix="val/out")
        all_metrics.update(out_metrics)
        self._save_validation_result("out", epoch, out_metrics, out_summary)

        # combined validation
        n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
        combined_num_episodes = in_summary["num_episodes"] + out_summary["num_episodes"]
        combined_num_tasks = in_summary["num_tasks"] + out_summary["num_tasks"]
        combined_num_correct = in_summary["num_correct"] + out_summary["num_correct"]
        combined_num_tasks_solved = in_summary["num_tasks_solved"] + out_summary["num_tasks_solved"]
        combined_summary = {
            "num_episodes": combined_num_episodes,
            "num_tasks": combined_num_tasks,
            "num_correct": combined_num_correct,
            "num_tasks_solved": combined_num_tasks_solved,
            "pass@1": float(combined_num_correct / combined_num_episodes) if combined_num_episodes > 0 else 0.0,
            f"pass@{n_val_samples}": float(combined_num_tasks_solved / combined_num_tasks) if combined_num_tasks > 0 else 0.0,
        }
        combined_metrics = {
            "val/combined/summary/num_episodes": float(combined_num_episodes),
            "val/combined/summary/num_tasks": float(combined_num_tasks),
            "val/combined/summary/pass@1": combined_summary["pass@1"],
            f"val/combined/summary/pass@{n_val_samples}": combined_summary[f"pass@{n_val_samples}"],
        }
        all_metrics.update(combined_metrics)
        self._save_validation_result("combined", epoch, combined_metrics, combined_summary)
        print(f"[Validation] Combined summary: {combined_summary}")
        return all_metrics

    def generate_trajectories(self, batch, timing_raw=None, **kwargs):
        """
        Generates trajectories asynchronously using the agent execution engine's excute tasks method.
        Post-processing is done in the engine as well.

        Args:
            batch: The input batch for trajectory generation
            timing_raw: Dictionary to store timing information for profiling
            **kwargs: Additional arguments to pass to trajectory_generator

        Returns:
            list: List of collected processed trajectories
        """
        if timing_raw is None:
            timing_raw = {}

        with marked_timer("generate_trajectories", timing_raw, color="red"):
            coro = self.agent_execution_engine.execute_tasks_verl(batch, **kwargs)
            final_gen_batch_output = asyncio.run_coroutine_threadsafe(coro, self._loop).result()

        return final_gen_batch_output

    def _stepwise_advantage_broadcast(self, last_step_batch, non_last_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps within the same episode and trajectory.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants

        src_traj_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
        src_eps_ids = last_step_batch.non_tensor_batch["episode_ids"]
        src_steps = last_step_batch.non_tensor_batch["step_nums"]
        src_mask = last_step_batch.batch["response_mask"]
        src_advantages = last_step_batch.batch["advantages"]

        tgt_traj_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
        tgt_eps_ids = non_last_step_batch.non_tensor_batch["episode_ids"]
        tgt_mask = non_last_step_batch.batch["response_mask"]

        # Build id -> scalar advantage
        traj_ep_to_scalar_adv = {}
        for i, (traj_id, eps_id) in enumerate(zip(src_traj_ids, src_eps_ids, strict=False)):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.rllm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            traj_ep_to_scalar_adv[(traj_id, eps_id)] = scalar

        # Create new tensor for non_last_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=traj_ep_to_scalar_adv[(traj_id, eps_id)], dtype=torch.float32) for i, (traj_id, eps_id) in enumerate(zip(tgt_traj_ids, tgt_eps_ids, strict=False))])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        non_last_step_batch.batch["advantages"] = final_advantage
        non_last_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        ref_policy_wg = getattr(self, "ref_policy_wg", None)
        if (
            self.use_reference_policy
            and not self.ref_in_actor
            and ref_policy_wg is not None
            and ref_policy_wg.world_size != 0
        ):
            world_sizes.append(ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if hasattr(self, "rollout_wg") and self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        batch = self._remove_padding(batch)  # Remove any padded steps from the batch (just in case)
        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch["is_last_step"][idx] = False
            batch.non_tensor_batch["is_pad_step"][idx] = True
            batch.non_tensor_batch["is_valid"][idx] = False

        return batch

    def _remove_padding(self, batch):
        """Removes padded steps from the batch"""
        is_pad_step = batch.non_tensor_batch["is_pad_step"]
        non_pad_step_indices = np.where(is_pad_step == False)[0]
        batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
        return batch

    def _maybe_dump_training_examples(
        self,
        batch,
        epoch: int,
        stage: str,
        timing_raw: dict | None = None,
    ) -> None:
        cfg = OmegaConf.select(
            self.config,
            "rllm.debug_dump_training_examples",
            default=None,
        )
        if cfg is None or not bool(OmegaConf.select(cfg, "enable", default=False)):
            return

        max_steps = int(OmegaConf.select(cfg, "max_steps", default=1) or 0)
        if max_steps > 0 and self.global_steps > max_steps:
            return

        timer = timing_raw if timing_raw is not None else {}
        with marked_timer("dump_training_examples", timer, color="green"):
            self._dump_training_examples(batch=batch, cfg=cfg, epoch=epoch, stage=stage)

    def _dump_training_examples(self, batch, cfg, epoch: int, stage: str) -> None:
        dump_dir = OmegaConf.select(
            cfg,
            "dir",
            default=f"logs/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/training_examples",
        )
        max_rows = int(OmegaConf.select(cfg, "max_rows", default=16) or 0)
        include_token_vectors = bool(
            OmegaConf.select(cfg, "include_token_vectors", default=False)
        )
        include_chat_completions = bool(
            OmegaConf.select(cfg, "include_chat_completions", default=False)
        )
        skip_special_tokens = bool(OmegaConf.select(cfg, "skip_special_tokens", default=True))

        os.makedirs(dump_dir, exist_ok=True)
        filename = os.path.join(
            dump_dir,
            f"step_{self.global_steps:06d}_{stage}.jsonl",
        )

        is_pad_step = batch.non_tensor_batch.get("is_pad_step")
        if is_pad_step is None:
            candidate_indices = list(range(len(batch)))
        else:
            candidate_indices = [
                idx for idx, is_pad in enumerate(np.asarray(is_pad_step).tolist()) if not bool(is_pad)
            ]
        if max_rows > 0:
            candidate_indices = candidate_indices[:max_rows]

        prompt_width = int(batch.batch["prompts"].shape[1])
        response_width = int(batch.batch["responses"].shape[1])
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)

        lines = []
        for row_index in candidate_indices:
            response_mask = batch.batch.get("response_mask")
            row_mask = (
                response_mask[row_index].bool()
                if response_mask is not None
                else torch.ones(response_width, dtype=torch.bool)
            )
            prompt_ids = self._active_prompt_ids(batch, row_index, prompt_width, pad_token_id)
            response_ids = self._active_response_ids(batch, row_index, prompt_width, pad_token_id)
            trained_response_ids = batch.batch["responses"][row_index][
                row_mask.to(device=batch.batch["responses"].device).bool()
            ]

            entry = {
                "global_step": int(self.global_steps),
                "epoch": int(epoch),
                "stage": stage,
                "row_index": int(row_index),
                "prompt": self.tokenizer.decode(
                    prompt_ids.detach().cpu().tolist(),
                    skip_special_tokens=skip_special_tokens,
                ),
                "response": self.tokenizer.decode(
                    response_ids.detach().cpu().tolist(),
                    skip_special_tokens=skip_special_tokens,
                ),
                "trained_response": self.tokenizer.decode(
                    trained_response_ids.detach().cpu().tolist(),
                    skip_special_tokens=skip_special_tokens,
                ),
                "prompt_token_count": int(prompt_ids.numel()),
                "response_token_count": int(response_ids.numel()),
                "trained_response_token_count": int(row_mask.sum().item()),
                "reward": self._row_sum(batch, "token_level_scores", row_index),
                "token_level_reward": self._row_sum(batch, "token_level_rewards", row_index),
                "trajectory_reward": self._row_sum(batch, "traj_rewards", row_index),
                "step_reward": self._row_sum(batch, "step_rewards", row_index),
                "advantage": self._masked_tensor_stats(batch, "advantages", row_index, row_mask),
                "return": self._masked_tensor_stats(batch, "returns", row_index, row_mask),
                "old_log_prob": self._masked_tensor_stats(batch, "old_log_probs", row_index, row_mask),
                "ref_log_prob": self._masked_tensor_stats(batch, "ref_log_prob", row_index, row_mask),
                "metadata": self._training_example_metadata(batch, row_index),
            }
            local_penalty = self._row_optional_scalar(
                batch,
                "local_advantage_penalties",
                row_index,
            )
            if local_penalty is not None:
                entry["local_advantage_penalty"] = local_penalty
            if include_token_vectors:
                entry["token_vectors"] = self._token_vectors(batch, row_index, row_mask)
            if include_chat_completions and "chat_completions" in batch.non_tensor_batch:
                entry["chat_completions"] = self._json_safe(
                    batch.non_tensor_batch["chat_completions"][row_index]
                )
            lines.append(json.dumps(self._json_safe(entry), ensure_ascii=False))

        with open(filename, "w", encoding="utf-8") as file:
            file.write("\n".join(lines))
            if lines:
                file.write("\n")

        print(
            "[debug] dumped PPO training examples "
            f"path={filename} rows={len(lines)} stage={stage}",
            flush=True,
        )

    def _active_prompt_ids(self, batch, row_index: int, prompt_width: int, pad_token_id):
        prompt_ids = batch.batch["prompts"][row_index]
        attention_mask = batch.batch.get("attention_mask")
        if attention_mask is not None:
            return prompt_ids[attention_mask[row_index, :prompt_width].bool()]
        if pad_token_id is not None:
            return prompt_ids[prompt_ids != pad_token_id]
        return prompt_ids

    def _active_response_ids(self, batch, row_index: int, prompt_width: int, pad_token_id):
        response_ids = batch.batch["responses"][row_index]
        attention_mask = batch.batch.get("attention_mask")
        if attention_mask is not None:
            return response_ids[
                attention_mask[
                    row_index,
                    prompt_width : prompt_width + response_ids.shape[0],
                ].bool()
            ]
        if pad_token_id is not None:
            return response_ids[response_ids != pad_token_id]
        return response_ids

    def _row_sum(self, batch, key: str, row_index: int) -> float | None:
        if key not in batch.batch.keys():
            return None
        value = batch.batch[key][row_index]
        return float(value.detach().float().sum().cpu().item())

    def _row_optional_scalar(self, batch, key: str, row_index: int) -> float | None:
        if key not in batch.batch.keys():
            return None
        value = batch.batch[key][row_index]
        if value.numel() == 0:
            return None
        if value.numel() == 1:
            return float(value.detach().float().cpu().item())
        return float(value.detach().float().sum().cpu().item())

    def _masked_tensor_stats(self, batch, key: str, row_index: int, mask) -> dict | None:
        if key not in batch.batch.keys():
            return None
        values = batch.batch[key][row_index]
        active = values[mask.to(device=values.device).bool()].detach().float().cpu()
        if active.numel() == 0:
            return {
                "mean": 0.0,
                "sum": 0.0,
                "min": 0.0,
                "max": 0.0,
                "token_count": 0,
            }
        return {
            "mean": float(active.mean().item()),
            "sum": float(active.sum().item()),
            "min": float(active.min().item()),
            "max": float(active.max().item()),
            "token_count": int(active.numel()),
        }

    def _token_vectors(self, batch, row_index: int, mask) -> dict[str, list[float]]:
        vectors = {}
        for key in (
            "advantages",
            "returns",
            "token_level_scores",
            "token_level_rewards",
            "old_log_probs",
            "ref_log_prob",
            "response_mask",
        ):
            if key not in batch.batch.keys():
                continue
            values = batch.batch[key][row_index]
            active = values[mask.to(device=values.device).bool()].detach().float().cpu()
            vectors[key] = [float(value) for value in active.tolist()]
        return vectors

    def _training_example_metadata(self, batch, row_index: int) -> dict:
        metadata = {}
        for key in (
            "task_ids",
            "uid",
            "episode_ids",
            "trajectory_ids",
            "step_ids",
            "step_nums",
            "is_last_step",
            "is_valid",
            "is_correct",
            "is_pad_step",
            "termination_reasons",
            "lora_names",
            "adv_group_ids",
            "adv_action_group_ids",
            "orggym_broadcast_uid",
            "metrics",
        ):
            if key in batch.non_tensor_batch:
                metadata[key] = self._json_safe(batch.non_tensor_batch[key][row_index])
        return metadata

    def _json_safe(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return self._json_safe(value.detach().cpu().tolist())
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def shutdown(self):
        """A cleanup method to gracefully stop the background event loop."""
        if hasattr(self, "agent_execution_engine") and self.agent_execution_engine is not None:
            self.agent_execution_engine.shutdown()
            self.agent_execution_engine = None
        if hasattr(self, "_loop") and self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_thread") and self._thread is not None:
            self._thread.join()

    def visualize_trajectory_last_step(self, tensor_batch, sample_idx=0, max_samples=1):
        """
        Visualize last steps from a workflow rollout using the shared visualization utility.
        """
        from rllm.utils.visualization import visualize_trajectories

        # Select only last steps if stepwise-advantage is enabled
        if "is_last_step" in tensor_batch.non_tensor_batch:
            is_last = tensor_batch.non_tensor_batch["is_last_step"]
            if is_last is not None and len(is_last) == len(tensor_batch):
                tensor_batch = tensor_batch[is_last]

        if len(tensor_batch) == 0:
            return

        end_idx = min(sample_idx + max_samples, len(tensor_batch))
        indices = list(range(sample_idx, end_idx))

        visualize_trajectories(
            batch=tensor_batch,
            tokenizer=self.tokenizer,
            sample_indices=indices,
            mask_key="response_mask",
            reward_key="step_rewards" if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "per_step" else "traj_rewards",
            show_workflow_metadata=True,
        )
