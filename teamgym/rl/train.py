"""Train TeamGym roles with the repository's RLLM/VERL revision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from rllm.trainer.agent_trainer import AgentTrainer

from .data import load_prepared_dataset
from .training.manifest import build_run_manifest
from .rollout.profiles import get_rollout_profile
from .training.trainer_compat import TeamGymAgentWorkflowPPOTrainer
from .training.two_policy_trainer import TeamGymTwoPolicyPPOTrainer
from .rollout.workflow import TeamGymWorkflow


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config: Any) -> None:
    data_dir = OmegaConf.select(config, "teamgym.prepared_data_dir", default=None)
    if not data_dir:
        raise ValueError(
            "+teamgym.prepared_data_dir=/path/to/prepared_data is required"
        )
    data_dir = to_absolute_path(str(data_dir))
    train_split = str(OmegaConf.select(config, "teamgym.train_split", default="train"))
    validation_split = str(
        OmegaConf.select(config, "teamgym.validation_split", default="validation")
    )
    train_dataset = load_prepared_dataset(data_dir, train_split)
    # VERL constructs a validation dataloader even when validation calls are
    # disabled. Always provide the real prepared split; the narrow trainer
    # subclass below disables execution by clearing val_reward_fn.
    validation_dataset = load_prepared_dataset(data_dir, validation_split)

    profile_name = str(
        OmegaConf.select(
            config, "teamgym.rollout_profile", default="paper_main_full_context"
        )
    )
    training_mode = str(
        OmegaConf.select(config, "teamgym.training_mode", default="coordinator_only")
    )
    participant_backend = str(
        OmegaConf.select(config, "teamgym.participant_backend", default="scripted")
    )
    participant_replay_path = OmegaConf.select(
        config, "teamgym.participant_replay_path", default=None
    )
    fixed_participant_base_url = OmegaConf.select(
        config, "teamgym.fixed_participant_base_url", default=None
    )
    fixed_participant_model = OmegaConf.select(
        config, "teamgym.fixed_participant_model", default=None
    )
    fixed_participant_api_key = str(
        OmegaConf.select(
            config, "teamgym.fixed_participant_api_key", default="teamgym-local"
        )
    )
    fixed_coordinator_backend = str(
        OmegaConf.select(
            config, "teamgym.fixed_coordinator_backend", default="base_model"
        )
    )
    fixed_coordinator_base_url = OmegaConf.select(
        config, "teamgym.fixed_coordinator_base_url", default=None
    )
    fixed_coordinator_model = OmegaConf.select(
        config, "teamgym.fixed_coordinator_model", default=None
    )
    fixed_coordinator_api_key = str(
        OmegaConf.select(
            config, "teamgym.fixed_coordinator_api_key", default="teamgym-local"
        )
    )
    reward_config = str(
        OmegaConf.select(config, "teamgym.reward_config", default="task_success")
    )
    written_judge_model = str(
        OmegaConf.select(config, "teamgym.written_judge_model", default="gpt-5.5")
    )
    written_judge_provider = str(
        OmegaConf.select(
            config, "teamgym.written_judge_provider", default="bedrock_openai"
        )
    )
    written_judge_base_url = str(
        OmegaConf.select(
            config,
            "teamgym.written_judge_base_url",
            default="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
        )
    )
    written_judge_endpoint = str(
        OmegaConf.select(
            config, "teamgym.written_judge_endpoint", default="/responses"
        )
    )
    written_judge_credentials_file = to_absolute_path(
        str(
            OmegaConf.select(
                config,
                "teamgym.written_judge_credentials_file",
                default="api_key/api_key.py",
            )
        )
    )
    written_judge_api_key_env = str(
        OmegaConf.select(
            config,
            "teamgym.written_judge_api_key_env",
            default="AWS_BEARER_TOKEN_BEDROCK",
        )
    )
    written_judge_max_output_tokens = int(
        OmegaConf.select(
            config,
            "teamgym.written_judge_max_output_tokens",
            default=3000,
        )
    )
    written_judge_timeout_sec = int(
        OmegaConf.select(
            config, "teamgym.written_judge_timeout_sec", default=600
        )
    )
    written_judge_request_retry_attempts = int(
        OmegaConf.select(
            config,
            "teamgym.written_judge_request_retry_attempts",
            default=4,
        )
    )
    written_judge_fail_on_error = bool(
        OmegaConf.select(
            config,
            "teamgym.written_judge_fail_on_error",
            default=False,
        )
    )
    configured_written_judge_cache_dir = OmegaConf.select(
        config, "teamgym.written_judge_cache_dir", default=None
    )
    max_parse_retries = int(
        OmegaConf.select(config, "teamgym.max_parse_retries", default=2)
    )
    invalid_json_penalty = float(
        OmegaConf.select(config, "teamgym.invalid_json_penalty", default=-0.1)
    )
    invalid_schema_penalty = float(
        OmegaConf.select(config, "teamgym.invalid_schema_penalty", default=-0.1)
    )
    format_credit_assignment = str(
        OmegaConf.select(
            config,
            "teamgym.format_credit_assignment",
            default="trajectory",
        )
    )
    trace_prompts = bool(
        OmegaConf.select(config, "teamgym.trace_prompts", default=True)
    )
    participant_format_repair_for_eval = bool(
        OmegaConf.select(
            config,
            "teamgym.participant_format_repair_for_eval",
            default=False,
        )
    )
    participant_format_repair_strategy = str(
        OmegaConf.select(
            config,
            "teamgym.participant_format_repair_strategy",
            default="wrap_raw",
        )
    )
    participant_format_repair_max_utterance_tokens = int(
        OmegaConf.select(
            config,
            "teamgym.participant_format_repair_max_utterance_tokens",
            default=0,
        )
    )
    run_dir = Path(
        to_absolute_path(
            str(
                OmegaConf.select(
                    config,
                    "teamgym.run_dir",
                    default=f"outputs/teamgym/{config.trainer.experiment_name}",
                )
            )
        )
    )
    artifact_dir = run_dir / "rollouts"
    written_judge_cache_dir = (
        Path(to_absolute_path(str(configured_written_judge_cache_dir)))
        if configured_written_judge_cache_dir
        else run_dir / "written_judge_cache"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    profile = get_rollout_profile(profile_name)
    manifest = build_run_manifest(
        profile=profile,
        prepared_data_dir=data_dir,
        train_split=train_split,
        validation_split=validation_split,
        model=str(config.actor_rollout_ref.model.path),
        training_mode=training_mode,
        participant_backend=participant_backend,
        participant_replay_path=(
            str(participant_replay_path)
            if participant_replay_path is not None
            else None
        ),
        fixed_participant_base_url=(
            str(fixed_participant_base_url)
            if fixed_participant_base_url is not None
            else None
        ),
        fixed_participant_model=(
            str(fixed_participant_model)
            if fixed_participant_model is not None
            else None
        ),
        fixed_coordinator_backend=fixed_coordinator_backend,
        fixed_coordinator_base_url=(
            str(fixed_coordinator_base_url)
            if fixed_coordinator_base_url is not None
            else None
        ),
        fixed_coordinator_model=(
            str(fixed_coordinator_model)
            if fixed_coordinator_model is not None
            else None
        ),
        reward_config=reward_config,
        invalid_json_penalty=invalid_json_penalty,
        invalid_schema_penalty=invalid_schema_penalty,
        format_credit_assignment=format_credit_assignment,
        max_parse_retries=max_parse_retries,
        config=config,
    )
    manifest["written_answer_judge"] = {
        "model": written_judge_model,
        "provider": written_judge_provider,
        "base_url": written_judge_base_url,
        "endpoint": written_judge_endpoint,
        "credentials_file": written_judge_credentials_file,
        "api_key_env": written_judge_api_key_env,
        "max_output_tokens": written_judge_max_output_tokens,
        "timeout_sec": written_judge_timeout_sec,
        "request_retry_attempts": written_judge_request_retry_attempts,
        "fail_on_error": written_judge_fail_on_error,
        "cache_dir": str(written_judge_cache_dir),
        "temperature": 0,
    }
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        previous_manifest = manifest_path.read_text(encoding="utf-8").rstrip()
        if previous_manifest:
            with (run_dir / "run_manifest_history.jsonl").open(
                "a", encoding="utf-8"
            ) as history_file:
                history_file.write(previous_manifest.replace("\n", " ") + "\n")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[teamgym] run manifest: {manifest_path}", flush=True)

    two_policy = bool(
        OmegaConf.select(config, "teamgym.two_policy", default=False)
    )
    trainer = AgentTrainer(
        workflow_class=TeamGymWorkflow,
        workflow_trainer_cls=(
            TeamGymTwoPolicyPPOTrainer
            if two_policy
            else TeamGymAgentWorkflowPPOTrainer
        ),
        workflow_args={
            "rollout_profile": profile_name,
            "role_training_mode": training_mode,
            "rollout_lora_name": OmegaConf.select(
                config, "teamgym.rollout_lora_name", default=None
            ),
            "participant_backend": participant_backend,
            "participant_replay_path": (
                str(participant_replay_path)
                if participant_replay_path is not None
                else None
            ),
            "fixed_participant_base_url": (
                str(fixed_participant_base_url)
                if fixed_participant_base_url is not None
                else None
            ),
            "fixed_participant_model": (
                str(fixed_participant_model)
                if fixed_participant_model is not None
                else None
            ),
            "fixed_participant_api_key": fixed_participant_api_key,
            "fixed_participant_temperature": float(
                OmegaConf.select(
                    config,
                    "teamgym.fixed_participant_temperature",
                    default=0.7,
                )
            ),
            "fixed_participant_val_temperature": float(
                OmegaConf.select(
                    config,
                    "teamgym.fixed_participant_val_temperature",
                    default=0.0,
                )
            ),
            "fixed_coordinator_backend": fixed_coordinator_backend,
            "fixed_coordinator_base_url": (
                str(fixed_coordinator_base_url)
                if fixed_coordinator_base_url is not None
                else None
            ),
            "fixed_coordinator_model": (
                str(fixed_coordinator_model)
                if fixed_coordinator_model is not None
                else None
            ),
            "fixed_coordinator_api_key": fixed_coordinator_api_key,
            "fixed_coordinator_temperature": float(
                OmegaConf.select(
                    config,
                    "teamgym.fixed_coordinator_temperature",
                    default=0.7,
                )
            ),
            "fixed_coordinator_val_temperature": float(
                OmegaConf.select(
                    config,
                    "teamgym.fixed_coordinator_val_temperature",
                    default=0.0,
                )
            ),
            "reward_config": reward_config,
            "written_judge_model": written_judge_model,
            "written_judge_provider": written_judge_provider,
            "written_judge_base_url": written_judge_base_url,
            "written_judge_endpoint": written_judge_endpoint,
            "written_judge_credentials_file": (
                written_judge_credentials_file
            ),
            "written_judge_api_key_env": written_judge_api_key_env,
            "written_judge_max_output_tokens": (
                written_judge_max_output_tokens
            ),
            "written_judge_timeout_sec": written_judge_timeout_sec,
            "written_judge_request_retry_attempts": (
                written_judge_request_retry_attempts
            ),
            "written_judge_fail_on_error": written_judge_fail_on_error,
            "written_judge_cache_dir": str(written_judge_cache_dir),
            "max_parse_retries": max_parse_retries,
            "invalid_json_penalty": invalid_json_penalty,
            "invalid_schema_penalty": invalid_schema_penalty,
            "format_credit_assignment": format_credit_assignment,
            "request_timeout_sec": float(
                OmegaConf.select(config, "teamgym.request_timeout_sec", default=900.0)
            ),
            "artifact_dir": str(artifact_dir),
            "trace_prompts": trace_prompts,
            "participant_format_repair_for_eval": (
                participant_format_repair_for_eval
            ),
            "participant_format_repair_strategy": (
                participant_format_repair_strategy
            ),
            "participant_format_repair_max_utterance_tokens": (
                participant_format_repair_max_utterance_tokens
            ),
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        backend="verl",
    )
    trainer.train()


if __name__ == "__main__":
    main()
