"""Narrow compatibility layer for role-level response masks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer

from .checkpoint_retention import (
    checkpoint_step,
    prune_model_checkpoints,
    verify_model_checkpoint,
    write_json_atomic,
)


FORMAT_ADVANTAGE_OVERRIDE_KEY = "teamgym_format_advantage_overrides"
FORMAT_PARSE_STATUS_KEY = "teamgym_parse_status"
FORMAT_PROMPT_ID_KEY = "teamgym_prompt_id"
FORMAT_RETRY_NUMBER_KEY = "teamgym_retry_number"
ACTOR_ROLE_KEY = "teamgym_actor_role"
_INVALID_FORMAT_STATUSES = frozenset({"invalid_json", "invalid_schema"})


def _apply_format_advantage_overrides(batch) -> None:
    """Give invalid attempts their own local advantage.

    RLLM's broadcast mode computes one GRPO advantage from the trajectory
    reward and copies it to every step.  That is appropriate for valid
    participant messages whose semantic quality is judged at episode level,
    but it gives an invalid first attempt the same credit as its valid retry.
    The aligned override vector marks only invalid, trainable attempts.  Their
    active response tokens receive the configured local penalty; all other
    rows keep the ordinary trajectory-level semantic advantage.
    """

    overrides = batch.batch.get(FORMAT_ADVANTAGE_OVERRIDE_KEY)
    if overrides is None or "advantages" not in batch.batch:
        return
    response_mask = batch.batch["response_mask"].bool()
    overrides = overrides.to(
        device=batch.batch["advantages"].device,
        dtype=batch.batch["advantages"].dtype,
    )
    active_override = torch.isfinite(overrides).unsqueeze(-1) & response_mask
    replacement = overrides.unsqueeze(-1).expand_as(
        batch.batch["advantages"]
    )
    batch.batch["advantages"] = torch.where(
        active_override, replacement, batch.batch["advantages"]
    )
    batch.batch["returns"] = torch.where(
        active_override, replacement, batch.batch["returns"]
    )


class TeamGymAgentWorkflowPPOTrainer(AgentWorkflowPPOTrainer):
    """TeamGym-specific validation and per-attempt credit assignment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._teamgym_checkpoint_scores: dict[int, float] = {}
        self._teamgym_best_checkpoint_step: int | None = None
        self._teamgym_best_checkpoint_score: float | None = None
        self._teamgym_previous_best_checkpoint_step: int | None = None
        self._load_teamgym_checkpoint_selection()
        if bool(
            OmegaConf.select(self.config, "teamgym.disable_validation", default=False)
        ):
            self.val_reward_fn = None

    def _checkpoint_retention_enabled(self) -> bool:
        generic = OmegaConf.select(
            self.config,
            "teamgym.retain_best_and_latest",
            default=None,
        )
        if generic is not None:
            return bool(generic)
        return bool(
            OmegaConf.select(
                self.config,
                "teamgym.retain_cuad_best_and_latest",
                default=False,
            )
        )

    def _checkpoint_selection_path(self) -> Path:
        return Path(str(self.config.trainer.default_local_dir)) / "checkpoint_selection.json"

    def _load_teamgym_checkpoint_selection(self) -> None:
        if not self._checkpoint_retention_enabled():
            return
        path = self._checkpoint_selection_path()
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        self._teamgym_checkpoint_scores = {
            int(step): float(score)
            for step, score in state.get("scores", {}).items()
        }
        best_step = state.get("best_step")
        best_score = state.get("best_score")
        self._teamgym_best_checkpoint_step = (
            int(best_step) if best_step is not None else None
        )
        self._teamgym_best_checkpoint_score = (
            float(best_score) if best_score is not None else None
        )
        root = path.parent
        complete_steps = {
            step
            for checkpoint in root.glob("global_step_*")
            if (step := checkpoint_step(checkpoint)) is not None
            and (checkpoint / "_SUCCESS").is_file()
        }
        if self._teamgym_best_checkpoint_step not in complete_steps:
            eligible = [
                step
                for step in complete_steps
                if step in self._teamgym_checkpoint_scores
            ]
            if eligible:
                recovered_step = min(
                    eligible,
                    key=lambda step: (
                        -self._teamgym_checkpoint_scores[step],
                        step,
                    ),
                )
                self._teamgym_best_checkpoint_step = recovered_step
                self._teamgym_best_checkpoint_score = (
                    self._teamgym_checkpoint_scores[recovered_step]
                )
                print(
                    "[TeamGym] Recovered CUAD selection from complete checkpoint "
                    f"step {recovered_step}"
                )
            else:
                self._teamgym_best_checkpoint_step = None
                self._teamgym_best_checkpoint_score = None

    def _write_teamgym_checkpoint_selection(self, *, checkpoint_complete: bool) -> None:
        metric = str(
            OmegaConf.select(
                self.config,
                "teamgym.checkpoint_selection_metric",
                default="val/in/unknown/teamgym_holistic_score_cuad",
            )
        )
        write_json_atomic(
            self._checkpoint_selection_path(),
            {
                "selection_split": str(
                    OmegaConf.select(
                        self.config,
                        "teamgym.checkpoint_selection_split",
                        default="validation_cuad",
                    )
                ),
                "selection_metric": metric,
                "tie_break": "earliest_step",
                "best_step": self._teamgym_best_checkpoint_step,
                "best_score": self._teamgym_best_checkpoint_score,
                "latest_observed_step": max(self._teamgym_checkpoint_scores, default=None),
                "latest_checkpoint_complete": checkpoint_complete,
                "scores": {
                    str(step): score
                    for step, score in sorted(self._teamgym_checkpoint_scores.items())
                },
            },
        )

    def _validate_agent(self, epoch: int = 0):
        metrics = super()._validate_agent(epoch=epoch)
        if not self._checkpoint_retention_enabled() or self.global_steps <= 0:
            return metrics

        # The upstream trainer performs one extra validation after incrementing
        # beyond the last optimizer update. There is no checkpoint for that
        # observation (for example, step 81 after an 80-update run), so it must
        # not replace a real, complete CUAD-selected checkpoint.
        save_freq = int(self.config.trainer.save_freq)
        if save_freq <= 0 or self.global_steps % save_freq != 0:
            return metrics

        metric = str(
            OmegaConf.select(
                self.config,
                "teamgym.checkpoint_selection_metric",
                default="val/in/unknown/teamgym_holistic_score_cuad",
            )
        )
        if metric not in metrics:
            raise RuntimeError(
                f"Checkpoint-selection metric {metric!r} is missing from validation"
            )
        score = float(metrics[metric])
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite checkpoint-selection score: {score}")

        step = int(self.global_steps)
        self._teamgym_checkpoint_scores[step] = score
        if (
            self._teamgym_best_checkpoint_score is None
            or score > self._teamgym_best_checkpoint_score
        ):
            self._teamgym_previous_best_checkpoint_step = (
                self._teamgym_best_checkpoint_step
            )
            self._teamgym_best_checkpoint_step = step
            self._teamgym_best_checkpoint_score = score
        self._write_teamgym_checkpoint_selection(checkpoint_complete=False)
        return metrics

    def _save_checkpoint(self):
        if not self._checkpoint_retention_enabled():
            return super()._save_checkpoint()

        step = int(self.global_steps)
        if step not in self._teamgym_checkpoint_scores:
            raise RuntimeError(
                "Best-checkpoint retention requires validation and save frequencies "
                f"to align; no selection score exists for step {step}"
            )
        root = Path(str(self.config.trainer.default_local_dir))
        if self._teamgym_best_checkpoint_step == step:
            protected_before_save = {
                self._teamgym_previous_best_checkpoint_step
            }
        else:
            protected_before_save = {self._teamgym_best_checkpoint_step}
        protected_before_save.discard(None)
        removed_before = prune_model_checkpoints(root, protected_before_save)
        for path in removed_before:
            print(f"[TeamGym] Pruned superseded checkpoint before save: {path}")

        super()._save_checkpoint()
        checkpoint_dir = root / f"global_step_{step}"
        verification = verify_model_checkpoint(
            checkpoint_dir,
            world_size=int(self.actor_rollout_wg.world_size),
            two_policy=bool(
                OmegaConf.select(self.config, "teamgym.two_policy", default=False)
            ),
        )
        verification.update(
            {
                "selection_metric": str(
                    OmegaConf.select(
                        self.config,
                        "teamgym.checkpoint_selection_metric",
                        default="val/in/unknown/teamgym_holistic_score_cuad",
                    )
                ),
                "selection_score": self._teamgym_checkpoint_scores[step],
            }
        )
        write_json_atomic(checkpoint_dir / "_SUCCESS", verification)

        keep_after_save = {step, self._teamgym_best_checkpoint_step}
        keep_after_save.discard(None)
        removed_after = prune_model_checkpoints(root, keep_after_save)
        for path in removed_after:
            print(f"[TeamGym] Pruned superseded checkpoint after save: {path}")
        self._teamgym_previous_best_checkpoint_step = None
        self._write_teamgym_checkpoint_selection(checkpoint_complete=True)

    def _stepwise_advantage_broadcast(self, last_step_batch, non_last_step_batch):
        super()._stepwise_advantage_broadcast(
            last_step_batch, non_last_step_batch
        )
        if (
            str(
                OmegaConf.select(
                    self.config,
                    "teamgym.format_credit_assignment",
                    default="trajectory",
                )
            )
            == "per_attempt"
        ):
            _apply_format_advantage_overrides(last_step_batch)
            _apply_format_advantage_overrides(non_last_step_batch)

    def _token_vectors(self, batch, row_index: int, mask) -> dict:
        vectors = super()._token_vectors(batch, row_index, mask)
        if FORMAT_ADVANTAGE_OVERRIDE_KEY in batch.batch:
            value = batch.batch[FORMAT_ADVANTAGE_OVERRIDE_KEY][row_index]
            vectors[FORMAT_ADVANTAGE_OVERRIDE_KEY] = float(
                value.detach().float().cpu()
            )
        return vectors

    def _training_example_metadata(self, batch, row_index: int) -> dict:
        metadata = super()._training_example_metadata(batch, row_index)
        for key in (
            FORMAT_PARSE_STATUS_KEY,
            FORMAT_PROMPT_ID_KEY,
            FORMAT_RETRY_NUMBER_KEY,
        ):
            if key in batch.non_tensor_batch:
                metadata[key] = self._json_safe(
                    batch.non_tensor_batch[key][row_index]
                )
        return metadata


class TeamGymAgentWorkflowEngine(AgentWorkflowEngine):
    """Apply ``Step.info.train_mask`` to the current RLLM response mask.

    Current RLLM (the repository revision used here) tokenizes every workflow
    step but does not yet consume a role-level train mask. The standard
    stepwise-broadcast implementation already keys by both trajectory and
    sampled episode, so only this small transform extension is required.
    """

    def transform_results_for_verl(self, episodes, task_ids):
        batch = super().transform_results_for_verl(episodes, task_ids)
        train_masks: list[float] = []
        format_advantage_overrides: list[float] = []
        parse_statuses: list[str] = []
        prompt_ids: list[str] = []
        retry_numbers: list[int] = []
        actor_roles: list[str] = []
        for episode in episodes:
            if episode is None:
                continue
            if all(not trajectory.steps for trajectory in episode.trajectories):
                continue
            for trajectory in episode.trajectories:
                for step in trajectory.steps:
                    trainable = step.info.get("train_mask") is not False
                    parse_status = str(step.info.get("parse_status") or "unknown")
                    train_masks.append(1.0 if trainable else 0.0)
                    parse_statuses.append(parse_status)
                    prompt_ids.append(str(step.info.get("prompt_id") or ""))
                    retry_numbers.append(int(step.info.get("retry_number", -1)))
                    actor_role = str(step.info.get("actor_role") or "")
                    actor_roles.append(
                        "participant"
                        if actor_role == "participant"
                        else "coordinator"
                    )
                    format_advantage_overrides.append(
                        float(step.info.get("local_reward_penalty", 0.0))
                        if trainable
                        and parse_status in _INVALID_FORMAT_STATUSES
                        else float("nan")
                    )
        actual = int(batch.batch["response_mask"].shape[0])
        if len(train_masks) != actual:
            raise RuntimeError(
                "TeamGym train-mask row mismatch: "
                f"metadata_rows={len(train_masks)} batch_rows={actual}"
            )
        row_mask = torch.tensor(
            train_masks,
            dtype=batch.batch["response_mask"].dtype,
            device=batch.batch["response_mask"].device,
        ).unsqueeze(-1)
        batch.batch["response_mask"] = batch.batch["response_mask"] * row_mask
        batch.batch[FORMAT_ADVANTAGE_OVERRIDE_KEY] = torch.tensor(
            format_advantage_overrides,
            dtype=torch.float32,
            device=batch.batch["response_mask"].device,
        )
        batch.non_tensor_batch[FORMAT_PARSE_STATUS_KEY] = np.asarray(
            parse_statuses, dtype=object
        )
        batch.non_tensor_batch[FORMAT_PROMPT_ID_KEY] = np.asarray(
            prompt_ids, dtype=object
        )
        batch.non_tensor_batch[FORMAT_RETRY_NUMBER_KEY] = np.asarray(
            retry_numbers, dtype=np.int64
        )
        batch.non_tensor_batch[ACTOR_ROLE_KEY] = np.asarray(
            actor_roles, dtype=object
        )
        return batch
