"""RLLM workflow that executes the ordinary TeamGym paper runtime."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from .rollout_client import RolloutEngineJsonClient as RolloutEngineJsonClient
from .action_repair import (
    _UTTERANCE_FIELD_RE as _UTTERANCE_FIELD_RE,
    _repair_compact_action as _repair_compact_action,
    _nested_utterance as _nested_utterance,
    _partial_utterance_value as _partial_utterance_value,
    _participant_counterfactual_utterance as _participant_counterfactual_utterance,
    _repair_participant_format_counterfactual as _repair_participant_format_counterfactual,
)

from .frozen_client import FrozenOpenAIJsonClient as FrozenOpenAIJsonClient
from .policy_routing import (
    BASE_LORA_NAME as BASE_LORA_NAME,
    _actor_role_for_prompt as _actor_role_for_prompt,
    FULL_MODEL_NAME as FULL_MODEL_NAME,
    VLLM_LORA_INT_ID as VLLM_LORA_INT_ID,
    VLLM_LORA_NAME as VLLM_LORA_NAME,
    _is_trainable as _is_trainable,
)

from .action_protocol import (
    _prompt_kind as _prompt_kind,
    _prompt_payload as _prompt_payload,
    _parse_exact_json_object as _parse_exact_json_object,
    _action_schema_error as _action_schema_error,
    _compact_action_has_required_field as _compact_action_has_required_field,
)

from teamgym.core.dataset import extract_common_datapoint
from teamgym.interaction.environment import run_episode
from teamgym.interaction.actor_utils import participant_ids
from teamgym.interaction.evaluation import (
    build_failed_evaluation,
    evaluate_episode,
)
from teamgym.interaction.llm_actors import (
    LLMCoordinatorActor,
    LLMFinalDecisionActor,
    LLMParticipantActor,
    LLMTwoStageParticipantActor,
    PromptTraceRecorder,
)
from teamgym.interaction.structures import (
    communication_structure_preset,
)
from teamgym.llm.openai_client import OpenAIJsonClient
from teamgym.llm.openai_client import (
    OpenAIJsonClientConfig,
)
from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.workflows.workflow import TerminationReason, Workflow
from .fixed_actors import (
    DeterministicCoordinatorActor,
    ReplayParticipantActor,
    SourceExcerptParticipantActor,
)
from .profiles import get_rollout_profile
from .rewards import format_scores, scalarize_evaluation
from teamgym.scoring.short_answers import strict_judgment_passed
from teamgym.scoring.written_answers import (
    WrittenJudgeConfig,
    holistic_score,
    is_written_answer_task,
    judge_written_episode,
)


TRAINING_MODES = {"coordinator_only", "participant_only", "joint"}
FIXED_COORDINATOR_BACKENDS = {"base_model", "deterministic", "external_openai"}
class TeamGymWorkflow(Workflow):
    """Execute one standard TeamGym episode and expose model calls to VERL."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor,
        *,
        rollout_profile: str = "paper_main_full_context",
        role_training_mode: str = "coordinator_only",
        rollout_lora_name: str | None = None,
        reward_config: str = "task_success",
        written_judge_model: str = "gpt-5.5",
        written_judge_provider: str = "bedrock_openai",
        written_judge_base_url: str = (
            "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
        ),
        written_judge_endpoint: str = "/responses",
        written_judge_credentials_file: str = "api_key/api_key.py",
        written_judge_api_key_env: str = "AWS_BEARER_TOKEN_BEDROCK",
        written_judge_max_output_tokens: int = 3000,
        written_judge_timeout_sec: int = 600,
        written_judge_request_retry_attempts: int = 4,
        written_judge_fail_on_error: bool = False,
        written_judge_cache_dir: str | None = None,
        participant_backend: str = "scripted",
        participant_replay_path: str | None = None,
        fixed_participant_base_url: str | None = None,
        fixed_participant_model: str | None = None,
        fixed_participant_api_key: str = "teamgym-local",
        fixed_participant_temperature: float = 0.7,
        fixed_participant_val_temperature: float = 0.0,
        fixed_coordinator_backend: str = "base_model",
        fixed_coordinator_base_url: str | None = None,
        fixed_coordinator_model: str | None = None,
        fixed_coordinator_api_key: str = "teamgym-local",
        fixed_coordinator_temperature: float = 0.7,
        fixed_coordinator_val_temperature: float = 0.0,
        max_parse_retries: int = 2,
        invalid_json_penalty: float = -0.1,
        invalid_schema_penalty: float = -0.1,
        format_credit_assignment: str = "trajectory",
        request_timeout_sec: float = 900.0,
        artifact_dir: str | None = None,
        trace_prompts: bool = True,
        participant_format_repair_for_eval: bool = False,
        participant_format_repair_strategy: str = "wrap_raw",
        participant_format_repair_max_utterance_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)
        if role_training_mode not in TRAINING_MODES:
            raise ValueError(f"Unsupported TeamGym training mode: {role_training_mode}")
        if participant_backend not in {
            "scripted",
            "rollout",
            "replay",
            "external_openai",
        }:
            raise ValueError(
                "participant_backend must be 'scripted', 'rollout', 'replay', "
                "or 'external_openai'"
            )
        if participant_backend == "replay" and not participant_replay_path:
            raise ValueError(
                "participant_backend='replay' requires participant_replay_path"
            )
        if participant_backend == "external_openai" and (
            not fixed_participant_base_url or not fixed_participant_model
        ):
            raise ValueError(
                "participant_backend='external_openai' requires "
                "fixed_participant_base_url and fixed_participant_model"
            )
        if fixed_coordinator_backend not in FIXED_COORDINATOR_BACKENDS:
            raise ValueError(
                "fixed_coordinator_backend must be 'base_model', 'deterministic', "
                "or 'external_openai'"
            )
        if fixed_coordinator_backend == "external_openai" and (
            not fixed_coordinator_base_url or not fixed_coordinator_model
        ):
            raise ValueError(
                "fixed_coordinator_backend='external_openai' requires "
                "fixed_coordinator_base_url and fixed_coordinator_model"
            )
        if (
            role_training_mode == "participant_only"
            and rollout_lora_name is None
            and fixed_coordinator_backend == "base_model"
            and communication_structure_preset(
                get_rollout_profile(
                    rollout_profile, participant_count=1
                ).communication_structure
            ).coordinator_present
        ):
            raise ValueError(
                "Full-parameter participant-only training cannot freeze the "
                "coordinator through the in-process base_model route: VERL "
                "synchronizes updated full weights into that vLLM server. Use "
                "fixed_coordinator_backend='external_openai'."
            )
        if (
            participant_backend in {"scripted", "replay", "external_openai"}
            and role_training_mode in {"participant_only", "joint"}
        ):
            raise ValueError(
                f"{role_training_mode} requires participant_backend='rollout'"
            )
        self.rollout_profile_name = rollout_profile
        # AgentWorkflowEngine owns ``Workflow.training_mode`` and rewrites it
        # to "train"/"val" for logging.  Keep role optimization policy in a
        # separate field so that engine bookkeeping cannot change train masks.
        self.role_training_mode = role_training_mode
        self.rollout_lora_name = rollout_lora_name
        self.reward_config = reward_config
        self.written_judge_config = WrittenJudgeConfig(
            model=written_judge_model,
            provider=written_judge_provider,
            base_url=written_judge_base_url,
            endpoint=written_judge_endpoint,
            credentials_file=written_judge_credentials_file,
            api_key_env=written_judge_api_key_env,
            max_output_tokens=int(written_judge_max_output_tokens),
            timeout_sec=int(written_judge_timeout_sec),
            request_retry_attempts=int(
                written_judge_request_retry_attempts
            ),
            cache_dir=written_judge_cache_dir,
        )
        self.written_judge_fail_on_error = bool(written_judge_fail_on_error)
        self.participant_backend = participant_backend
        self.participant_replay_path = participant_replay_path
        self.fixed_participant_base_url = fixed_participant_base_url
        self.fixed_participant_model = fixed_participant_model
        self.fixed_participant_api_key = fixed_participant_api_key
        self.fixed_participant_temperature = float(fixed_participant_temperature)
        self.fixed_participant_val_temperature = float(
            fixed_participant_val_temperature
        )
        self.fixed_coordinator_backend = fixed_coordinator_backend
        self.fixed_coordinator_base_url = fixed_coordinator_base_url
        self.fixed_coordinator_model = fixed_coordinator_model
        self.fixed_coordinator_api_key = fixed_coordinator_api_key
        self.fixed_coordinator_temperature = float(fixed_coordinator_temperature)
        self.fixed_coordinator_val_temperature = float(
            fixed_coordinator_val_temperature
        )
        self.max_parse_retries = max_parse_retries
        self.invalid_json_penalty = invalid_json_penalty
        self.invalid_schema_penalty = invalid_schema_penalty
        if format_credit_assignment not in {"trajectory", "per_attempt"}:
            raise ValueError(
                "format_credit_assignment must be 'trajectory' or 'per_attempt'"
            )
        self.format_credit_assignment = format_credit_assignment
        self.request_timeout_sec = request_timeout_sec
        self.artifact_dir = Path(artifact_dir).expanduser() if artifact_dir else None
        self.trace_prompts = bool(trace_prompts)
        self.participant_format_repair_for_eval = bool(
            participant_format_repair_for_eval
        )
        if participant_format_repair_strategy not in {
            "wrap_raw",
            "extract_utterance_suffix",
        }:
            raise ValueError(
                "participant_format_repair_strategy must be 'wrap_raw' or "
                "'extract_utterance_suffix'"
            )
        self.participant_format_repair_strategy = (
            participant_format_repair_strategy
        )
        self.participant_format_repair_max_utterance_tokens = max(
            0, int(participant_format_repair_max_utterance_tokens)
        )
        if (
            self.participant_format_repair_strategy
            == "extract_utterance_suffix"
            and self.participant_format_repair_max_utterance_tokens <= 0
        ):
            raise ValueError(
                "extract_utterance_suffix requires a positive "
                "participant_format_repair_max_utterance_tokens"
            )

    async def run(self, task: dict[str, Any], uid: str, **kwargs: Any) -> Episode:
        del kwargs
        self.reset(task=task, uid=uid)
        datapoint = _datapoint_from_task(task)
        profile = get_rollout_profile(
            self.rollout_profile_name,
            participant_count=len(datapoint.get("participant_directory", [])),
        )
        structure = communication_structure_preset(
            profile.communication_structure
        )
        trainable_trajectory = Trajectory(
            name=self.rollout_lora_name or FULL_MODEL_NAME,
            task={"instance_id": datapoint.get("instance_id")},
        )
        fixed_trajectory = Trajectory(
            name=BASE_LORA_NAME,
            task={"instance_id": datapoint.get("instance_id")},
        )
        event_loop = asyncio.get_running_loop()
        client = RolloutEngineJsonClient(
            rollout_engine=self.rollout_engine,
            event_loop=event_loop,
            uid=uid,
            trainable_trajectory=trainable_trajectory,
            fixed_trajectory=fixed_trajectory,
            training_mode=self.role_training_mode,
            rollout_lora_name=self.rollout_lora_name,
            max_output_tokens=profile.model_max_output_tokens,
            participant_max_output_tokens=(
                profile.participant_model_max_output_tokens
            ),
            max_parse_retries=self.max_parse_retries,
            invalid_json_penalty=self.invalid_json_penalty,
            invalid_schema_penalty=self.invalid_schema_penalty,
            request_timeout_sec=self.request_timeout_sec,
            allow_compact_repair=profile.model_max_output_tokens <= 64,
            counterfactual_participant_format_repair=(
                self.participant_format_repair_for_eval
                and str(getattr(self, "training_mode", "train")).startswith("val")
            ),
            counterfactual_participant_format_repair_strategy=(
                self.participant_format_repair_strategy
            ),
            counterfactual_participant_max_utterance_tokens=(
                self.participant_format_repair_max_utterance_tokens
            ),
            counterfactual_participant_tokenizer=getattr(
                self.rollout_engine, "tokenizer", None
            ),
        )
        recorder = PromptTraceRecorder(None)
        fixed_client: Any = client
        is_validation = str(
            getattr(self, "training_mode", "train")
        ).startswith("val")
        if self.fixed_coordinator_backend == "external_openai":
            assert self.fixed_coordinator_base_url is not None
            assert self.fixed_coordinator_model is not None
            fixed_client = FrozenOpenAIJsonClient(
                base_url=self.fixed_coordinator_base_url,
                model=self.fixed_coordinator_model,
                api_key=self.fixed_coordinator_api_key,
                max_output_tokens=profile.model_max_output_tokens,
                temperature=(
                    self.fixed_coordinator_val_temperature
                    if is_validation
                    else self.fixed_coordinator_temperature
                ),
                timeout_sec=self.request_timeout_sec,
                attempt_records=client.attempt_records,
                max_parse_retries=self.max_parse_retries,
                policy_route="frozen_external_coordinator",
            )
        fixed_participant_client: Any = None
        if self.participant_backend == "external_openai":
            assert self.fixed_participant_base_url is not None
            assert self.fixed_participant_model is not None
            fixed_participant_client = FrozenOpenAIJsonClient(
                base_url=self.fixed_participant_base_url,
                model=self.fixed_participant_model,
                api_key=self.fixed_participant_api_key,
                max_output_tokens=(
                    profile.participant_model_max_output_tokens
                    or profile.model_max_output_tokens
                ),
                temperature=(
                    self.fixed_participant_val_temperature
                    if is_validation
                    else self.fixed_participant_temperature
                ),
                timeout_sec=self.request_timeout_sec,
                attempt_records=client.attempt_records,
                max_parse_retries=self.max_parse_retries,
                policy_route="frozen_external_participant",
            )

        participant_actor: Any
        if self.participant_backend == "scripted":
            participant_actor = SourceExcerptParticipantActor()
        elif self.participant_backend == "replay":
            assert self.participant_replay_path is not None
            participant_actor = ReplayParticipantActor(
                self.participant_replay_path
            )
        else:
            participant_class = (
                LLMTwoStageParticipantActor
                if profile.participant_response_mode == "two_stage"
                else LLMParticipantActor
            )
            participant_actor = participant_class(
                fixed_participant_client
                if self.participant_backend == "external_openai"
                else client,
                recorder,
                role_name_mode=profile.participant_role_name_mode,
                observation_mode=profile.participant_observation_mode,
            )

        if (
            self.role_training_mode == "participant_only"
            and self.fixed_coordinator_backend == "deterministic"
        ):
            coordinator_actor: Any = DeterministicCoordinatorActor()
        else:
            compact_output = (
                profile.model_max_output_tokens <= 64
                or profile.coordinator_query_compact_output
            )
            coordinator_actor = LLMCoordinatorActor(
                fixed_client
                if self.role_training_mode == "participant_only"
                else client,
                recorder,
                role_name_mode=profile.participant_role_name_mode,
                compact_output=compact_output,
                final_compact_output=(
                    profile.model_max_output_tokens <= 64
                ),
                include_compact_envelope_example=(
                    self.role_training_mode != "participant_only"
                ),
                participants_query_private_only=(
                    profile.participant_observation_mode
                    == "query_private_only"
                ),
            )
        final_actor = LLMFinalDecisionActor(
            client
            if (
                self.role_training_mode == "participant_only"
                and structure.final_decision_rule
                == "designated_participant"
            )
            else fixed_client
            if self.role_training_mode == "participant_only"
            else client,
            recorder,
            compact_output=profile.model_max_output_tokens <= 64,
            role_name_mode=profile.participant_role_name_mode,
            participant_prompt_kind=(
                "participant_final_decision"
                if structure.participant_handoff_allows_finalize
                else "final_synthesis"
            ),
        )

        paper_episode: dict[str, Any] | None = None
        runtime_error: BaseException | None = None
        try:
            initial_participant_id = _initial_participant_id(
                datapoint,
                mode=profile.initial_speaker_mode,
                episode_id=uid,
                validation=is_validation,
            )
            paper_episode = await self.run_in_executor(
                run_episode,
                datapoint,
                structure,
                budgets=profile.budgets,
                participant_actor=participant_actor,
                coordinator_actor=coordinator_actor,
                final_actor=final_actor,
                initial_participant_id=initial_participant_id,
            )
            evaluation = evaluate_episode(paper_episode, datapoint)
        except BaseException as exc:  # noqa: BLE001
            runtime_error = exc
            evaluation = build_failed_evaluation(
                datapoint,
                communication_structure=profile.communication_structure,
                error=exc,
            )

        written_judgment: dict[str, Any] | None = None
        written_judge_error: str | None = None
        written_task = is_written_answer_task(datapoint)
        if written_task and paper_episode is not None:
            try:
                written_judgment = await self.run_in_executor(
                    judge_written_episode,
                    datapoint=datapoint,
                    episode=paper_episode,
                    config=self.written_judge_config,
                )
            except BaseException as exc:  # noqa: BLE001
                written_judge_error = f"{type(exc).__name__}: {exc}"[:1000]
                if self.written_judge_fail_on_error:
                    raise RuntimeError(
                        "Written-answer judge failed after configured retries"
                    ) from exc
        elif self.reward_config == "written_holistic_score":
            written_judge_error = (
                "Written holistic reward requested but no completed "
                "written-answer episode was available."
            )

        base_reward, reward_components = scalarize_evaluation(
            evaluation,
            self.reward_config,
            written_judgment=written_judgment,
        )
        penalized_score = float(base_reward + client.local_penalty_total)
        per_attempt_format_credit = (
            self.format_credit_assignment == "per_attempt"
        )
        # In per-attempt mode, GRPO compares episodes using the semantic task
        # score only. The trainer later replaces the broadcast advantage on
        # each invalid trainable attempt with that attempt's local penalty.
        # This prevents a valid retry from sharing its invalid predecessor's
        # credit while preserving stable trajectory-level semantic groups.
        training_reward = (
            float(base_reward)
            if per_attempt_format_credit
            else penalized_score
        )
        trajectories = [
            trajectory
            for trajectory in (trainable_trajectory, fixed_trajectory)
            if trajectory.steps
        ]
        for trajectory in trajectories:
            if per_attempt_format_credit:
                for step in trajectory.steps:
                    if (
                        step.info.get("train_mask") is not False
                        and step.info.get("parse_status")
                        not in {"invalid_json", "invalid_schema"}
                    ):
                        step.reward = float(base_reward)
            else:
                trajectory.steps[-1].reward = training_reward
            trajectory.steps[-1].done = True
            trajectory.reward = training_reward
            trajectory.info = {
                "episode_id": uid,
                "instance_id": datapoint.get("instance_id"),
                "canonical_task_id": task.get("canonical_task_id"),
                "role_training_mode": self.role_training_mode,
                "rollout_mode": getattr(self, "training_mode", "standalone"),
                "reward_config": self.reward_config,
                "base_reward": base_reward,
                "local_penalty_total": client.local_penalty_total,
                "penalized_score": penalized_score,
                "format_credit_assignment": self.format_credit_assignment,
                "policy_route": (
                    "frozen_base"
                    if trajectory.name == BASE_LORA_NAME
                    else "trainable_adapter"
                ),
            }

        normalized_judgment = (
            written_judgment.get("judgment")
            if isinstance(written_judgment, dict)
            and isinstance(written_judgment.get("judgment"), dict)
            else {}
        )
        success = (
            bool(normalized_judgment.get("holistic_passed"))
            if written_task
            else evaluation.get("primary_eval", {}).get("task_success") is True
        )
        total_steps = int(normalized_judgment.get("total_step_count") or 0)
        covered_steps = int(
            normalized_judgment.get("covered_step_count") or 0
        )
        strict_success = strict_judgment_passed(normalized_judgment)
        step_coverage = (
            float(covered_steps) / float(total_steps)
            if total_steps > 0
            else 0.0
        )
        trainable_format = format_scores(
            client.attempt_records, trainable_only=True
        )
        all_actor_format = format_scores(
            client.attempt_records, trainable_only=False
        )
        handoff_dynamics = _participant_handoff_dynamics(
            paper_episode, client.attempt_records
        )
        dataset_name = str(task.get("dataset") or datapoint.get("dataset") or "unknown")
        dataset_metric_suffix = re.sub(r"[^a-z0-9]+", "_", dataset_name.casefold()).strip("_")
        metrics = {
            "teamgym_reward": training_reward,
            "teamgym_base_reward": base_reward,
            "teamgym_local_penalty": client.local_penalty_total,
            "teamgym_penalized_score": penalized_score,
            "teamgym_per_attempt_format_credit": (
                1.0 if per_attempt_format_credit else 0.0
            ),
            "teamgym_holistic_score": holistic_score(written_judgment),
            "teamgym_strict_success": 1.0 if strict_success else 0.0,
            "teamgym_required_step_coverage": step_coverage,
            "teamgym_written_judge_error": (
                1.0 if written_judge_error else 0.0
            ),
            "teamgym_written_judge_api_error": (
                1.0
                if written_judge_error and paper_episode is not None
                else 0.0
            ),
            "teamgym_written_judge_cache_hit": (
                1.0
                if isinstance(written_judgment, dict)
                and written_judgment.get("cache_hit") is True
                else 0.0
            ),
            "teamgym_format_score": trainable_format["format_score"],
            "teamgym_first_pass_format_score": trainable_format[
                "first_pass_format_score"
            ],
            "teamgym_format_invocation_count": trainable_format[
                "invocation_count"
            ],
            "teamgym_all_actor_format_score": all_actor_format[
                "format_score"
            ],
            "teamgym_task_success": 1.0 if success else 0.0,
            "teamgym_informed_success": (
                1.0
                if evaluation.get("primary_eval", {}).get("informed_success") is True
                else 0.0
            ),
            "teamgym_complete_participant_probe": (
                1.0
                if evaluation.get("participation_eval", {}).get(
                    "complete_participant_probe"
                )
                is True
                else 0.0
            ),
            "teamgym_complete_query_schedule": (
                1.0
                if evaluation.get("participation_eval", {}).get(
                    "complete_query_schedule"
                )
                is True
                else 0.0
            ),
            "teamgym_model_step_count": float(
                len(client.attempt_records)
            ),
            "teamgym_trainable_step_count": float(
                len(trainable_trajectory.steps)
            ),
            "teamgym_frozen_model_step_count": float(
                sum(not bool(record.get("train_mask")) for record in client.attempt_records)
            ),
            "teamgym_compact_repair_count": float(
                sum(
                    record.get("compact_repair_method") is not None
                    for record in client.attempt_records
                )
            ),
            "teamgym_participant_format_counterfactual_repair_count": float(
                sum(
                    record.get("counterfactual_repair_method") is not None
                    for record in client.attempt_records
                )
            ),
            "teamgym_invalid_json_count": float(
                sum(
                    record.get("parse_status") == "invalid_json"
                    for record in client.attempt_records
                )
            ),
            "teamgym_invalid_schema_count": float(
                sum(
                    record.get("parse_status") == "invalid_schema"
                    for record in client.attempt_records
                )
            ),
            "teamgym_format_error_count": float(
                sum(
                    record.get("parse_status")
                    in {"invalid_json", "invalid_schema"}
                    for record in client.attempt_records
                )
            ),
            "teamgym_trainable_invalid_json_count": float(
                sum(
                    bool(record.get("train_mask"))
                    and record.get("parse_status") == "invalid_json"
                    for record in client.attempt_records
                )
            ),
            "teamgym_trainable_invalid_schema_count": float(
                sum(
                    bool(record.get("train_mask"))
                    and record.get("parse_status") == "invalid_schema"
                    for record in client.attempt_records
                )
            ),
            "teamgym_trainable_format_error_count": float(
                sum(
                    bool(record.get("train_mask"))
                    and record.get("parse_status")
                    in {"invalid_json", "invalid_schema"}
                    for record in client.attempt_records
                )
            ),
            "teamgym_frozen_format_error_count": float(
                sum(
                    not bool(record.get("train_mask"))
                    and record.get("parse_status")
                    in {"invalid_json", "invalid_schema"}
                    for record in client.attempt_records
                )
            ),
            "teamgym_participant_communication_turns": float(
                handoff_dynamics["communication_turns"]
            ),
            "teamgym_participant_unique_speakers": float(
                handoff_dynamics["unique_speaker_count"]
            ),
            "teamgym_participant_handoff_count": float(
                handoff_dynamics["handoff_count"]
            ),
            "teamgym_participant_self_handoff_rate": float(
                handoff_dynamics["self_handoff_rate"]
            ),
            "teamgym_participant_speaker_entropy": float(
                handoff_dynamics["speaker_entropy"]
            ),
            "teamgym_participant_finalize_turn": float(
                handoff_dynamics["finalize_turn"]
            ),
            "teamgym_invalid_next_actor_count": float(
                handoff_dynamics["invalid_next_actor_attempt_count"]
            ),
            f"teamgym_reward_{dataset_metric_suffix}": training_reward,
            f"teamgym_holistic_score_{dataset_metric_suffix}": holistic_score(
                written_judgment
            ),
            f"teamgym_format_score_{dataset_metric_suffix}": trainable_format[
                "format_score"
            ],
            f"teamgym_task_success_{dataset_metric_suffix}": 1.0 if success else 0.0,
        }
        policy_output_error = bool(
            runtime_error is not None
            and isinstance(runtime_error, (ValueError, json.JSONDecodeError))
            and client.attempt_records
        )
        episode = Episode(
            id=uid,
            task={
                "instance_id": datapoint.get("instance_id"),
                "canonical_task_id": task.get("canonical_task_id"),
            },
            termination_reason=(
                TerminationReason.ENV_DONE
                if runtime_error is None or policy_output_error
                else TerminationReason.ERROR
            ),
            is_correct=success,
            trajectories=trajectories,
            metrics=metrics,
            info={
                "teamgym": {
                    "schema_version": "teamgym_rllm_episode_view_v1",
                    "profile": profile.to_dict(),
                    "paper_episode": paper_episode,
                    "evaluation": evaluation,
                    "written_answer_judgment": written_judgment,
                    "written_answer_judge_error": written_judge_error,
                    "reward": training_reward,
                    "reward_components": reward_components,
                    "local_penalty_total": client.local_penalty_total,
                    "penalized_score": penalized_score,
                    "format_credit_assignment": self.format_credit_assignment,
                    "generation_attempts": client.attempt_records
                    if self.trace_prompts
                    else [],
                    "participant_handoff_dynamics": handoff_dynamics,
                    "runtime_error": str(runtime_error)[:1000]
                    if runtime_error
                    else None,
                }
            },
        )
        if self.artifact_dir is not None:
            await self.run_in_executor(self._write_artifact, episode, datapoint)
        return episode

    def _write_artifact(self, episode: Episode, datapoint: dict[str, Any]) -> None:
        assert self.artifact_dir is not None
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode.id)[:180]
        payload = {
            "schema_version": "teamgym_rl_rollout_artifact_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "episode_id": episode.id,
            "termination_reason": episode.termination_reason.value,
            "is_correct": episode.is_correct,
            "metrics": episode.metrics,
            "normalized_datapoint": datapoint,
            **episode.info["teamgym"],
            "rllm_trajectories": [
                trajectory.to_dict() for trajectory in episode.trajectories
            ],
        }
        (self.artifact_dir / f"{slug}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _datapoint_from_task(task: dict[str, Any]) -> dict[str, Any]:
    value = task.get("teamgym_datapoint_json")
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    value = task.get("datapoint")
    if isinstance(value, dict):
        return value
    datapoint = extract_common_datapoint(task)
    if datapoint is not None:
        return datapoint
    raise ValueError("TeamGym task does not contain a normalized common datapoint")


def _initial_participant_id(
    datapoint: dict[str, Any],
    *,
    mode: str,
    episode_id: str,
    validation: bool,
) -> str:
    ids = participant_ids(datapoint)
    if not ids:
        raise ValueError("TeamGym datapoint has no participants")
    if mode == "first":
        return ids[0]
    if mode != "episode_hash":
        raise ValueError(f"Unknown initial speaker mode: {mode}")
    instance_id = str(datapoint.get("instance_id") or "")
    key = instance_id if validation else f"{instance_id}:{episode_id}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return ids[int.from_bytes(digest[:8], "big") % len(ids)]


def _participant_handoff_dynamics(
    episode: dict[str, Any] | None,
    attempt_records: list[dict[str, Any]],
) -> dict[str, Any]:
    messages = (
        [
            row
            for row in episode.get("messages", [])
            if isinstance(row, dict) and row.get("speaker_role") == "participant"
        ]
        if isinstance(episode, dict)
        else []
    )
    speakers = [str(row.get("speaker_id") or "") for row in messages]
    counts = {speaker: speakers.count(speaker) for speaker in sorted(set(speakers))}
    total = len(speakers)
    entropy = 0.0
    if total:
        entropy = -sum(
            (count / total) * math.log(count / total)
            for count in counts.values()
        )
    handoffs = [row for row in messages if row.get("participant_action") == "handoff"]
    self_handoffs = sum(
        row.get("recommended_next_actor_id") == row.get("speaker_id")
        for row in handoffs
    )
    finalize_rows = [
        row for row in messages if row.get("participant_action") == "finalize"
    ]
    transitions = [
        {
            "turn": row.get("turn"),
            "speaker_id": row.get("speaker_id"),
            "speaker_role_name": row.get("role_name"),
            "action": row.get("participant_action"),
            "next_actor_id": row.get("recommended_next_actor_id"),
        }
        for row in messages
    ]
    invalid_next_actor_attempt_count = sum(
        record.get("parse_status") == "invalid_schema"
        and "next_actor" in str(record.get("parse_error") or "")
        for record in attempt_records
    )
    return {
        "initial_speaker_id": (
            episode.get("initial_speaker_id")
            if isinstance(episode, dict)
            else None
        ),
        "communication_turns": total,
        "unique_speaker_count": len(counts),
        "speaker_turn_counts": counts,
        "speaker_entropy": entropy,
        "handoff_count": len(handoffs),
        "self_handoff_count": self_handoffs,
        "self_handoff_rate": self_handoffs / len(handoffs) if handoffs else 0.0,
        "finalize_turn": int(finalize_rows[-1].get("turn") or 0)
        if finalize_rows
        else 0,
        "invalid_next_actor_attempt_count": invalid_next_actor_attempt_count,
        "transitions": transitions,
    }
