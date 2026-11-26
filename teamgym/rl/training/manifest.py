"""Self-contained run-manifest helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from ..rollout.profiles import RolloutProfile


def build_run_manifest(
    *,
    profile: RolloutProfile,
    prepared_data_dir: str,
    train_split: str,
    validation_split: str,
    model: str,
    training_mode: str,
    participant_backend: str,
    participant_replay_path: str | None,
    fixed_participant_base_url: str | None,
    fixed_participant_model: str | None,
    fixed_coordinator_backend: str,
    fixed_coordinator_base_url: str | None,
    fixed_coordinator_model: str | None,
    reward_config: str,
    invalid_json_penalty: float,
    invalid_schema_penalty: float,
    format_credit_assignment: str,
    max_parse_retries: int,
    config: Any,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    prepared_root = Path(prepared_data_dir).expanduser().resolve()
    full_finetune = (
        int(OmegaConf.select(config, "actor_rollout_ref.model.lora_rank", default=0))
        == 0
    )
    two_policy = bool(
        OmegaConf.select(config, "teamgym.two_policy", default=False)
    )
    audit_path = prepared_root / "leakage_audit.json"
    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.is_file()
        else {}
    )
    selected_split_files: dict[str, dict[str, Any]] = {}
    for role, split in (
        ("train", train_split),
        ("validation", validation_split),
    ):
        selected_split_files[role] = {
            "split": split,
            "files": {
                suffix: {
                    "path": str(path),
                    "sha256": _file_sha256(path),
                }
                for suffix, path in (
                    ("jsonl", prepared_root / f"{split}.jsonl"),
                    ("parquet", prepared_root / f"{split}.parquet"),
                    ("verl_parquet", prepared_root / f"{split}_verl.parquet"),
                )
            },
        }
    revision, dirty = _git_state()
    return {
        "schema_version": "teamgym_rl_run_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_data_dir": str(prepared_root),
        "selected_split_files": selected_split_files,
        "input_hashes": audit.get("input_hashes", {}),
        "split_manifest_sha256": audit.get("manifest_sha256"),
        "code_revision": revision,
        "dirty_tree": dirty,
        "packages": {
            name: _package_version(name)
            for name in ("rllm", "verl", "torch", "transformers", "vllm", "ray")
        },
        "python": sys.version,
        "hardware": {
            "slurm_job_id": _optional_env("SLURM_JOB_ID"),
            "slurm_job_name": _optional_env("SLURM_JOB_NAME"),
            "slurm_node": _optional_env("SLURMD_NODENAME"),
            "slurm_gpus_on_node": _optional_env("SLURM_GPUS_ON_NODE"),
            "gpu_allocation_policy": _optional_env(
                "TEAMGYM_GPU_ALLOCATION_POLICY"
            ),
            "requested_gpus": _optional_env_int("TEAMGYM_REQUESTED_GPUS"),
            "detected_gpus": _optional_env_int("TEAMGYM_DETECTED_GPUS"),
            "used_gpus": _optional_env_int("TEAMGYM_USED_GPUS"),
            "used_gpu_ids": _optional_env("TEAMGYM_USED_GPU_IDS"),
        },
        "model": model,
        "model_revision": _cached_model_revision(model),
        "rollout_profile": profile.to_dict(),
        "training_mode": training_mode,
        "participant_backend": participant_backend,
        "participant_replay": (
            {
                "path": str(
                    Path(participant_replay_path).expanduser().resolve()
                ),
                "sha256": _file_sha256(
                    Path(participant_replay_path).expanduser().resolve()
                ),
            }
            if participant_replay_path
            else None
        ),
        "fixed_participant": {
            "backend": participant_backend,
            "base_url": fixed_participant_base_url,
            "model": fixed_participant_model,
            "separate_immutable_server": participant_backend
            == "external_openai",
        },
        "fixed_coordinator_backend": fixed_coordinator_backend,
        "fixed_coordinator": {
            "backend": fixed_coordinator_backend,
            "base_url": fixed_coordinator_base_url,
            "model": fixed_coordinator_model,
            "separate_immutable_server": (
                fixed_coordinator_backend == "external_openai"
            ),
        },
        "role_assignments": {
            "coordinator": "trainable"
            if training_mode in {"coordinator_only", "joint"}
            else "fixed",
            "participant": "trainable"
            if training_mode in {"participant_only", "joint"}
            else "fixed",
        },
        "role_gradient_masks": {
            "coordinator": training_mode in {"coordinator_only", "joint"},
            "participant": training_mode in {"participant_only", "joint"},
            "final_synthesis": training_mode in {"coordinator_only", "joint"},
        },
        "shared_weight_semantics": {
            "full_finetune": full_finetune,
            "two_policy": two_policy,
            "roles_share_model_weights": not two_policy and not (
                (
                    training_mode == "participant_only"
                    and fixed_coordinator_backend == "external_openai"
                )
                or (
                    training_mode == "coordinator_only"
                    and participant_backend == "external_openai"
                )
            ),
            "caveat": (
                "Coordinator/final and participant generations use independent "
                "full-parameter policies, rollout servers, PPO updates, and "
                "checkpoint subtrees; both policies receive the same episode-level "
                "semantic reward before role-local format overrides."
                if two_policy
                else "The participant policy is full-parameter trained while coordinator "
                "queries and final synthesis are generated by a separately hosted "
                "immutable base checkpoint."
                if (
                    full_finetune
                    and training_mode == "participant_only"
                    and fixed_coordinator_backend == "external_openai"
                )
                else "The coordinator/final policy is full-parameter trained "
                "while participant responses are generated by a separately "
                "hosted immutable participant checkpoint."
                if (
                    full_finetune
                    and training_mode == "coordinator_only"
                    and participant_backend == "external_openai"
                )
                else "Role-only denotes which role completions receive policy gradients; "
                "with full finetuning, an update changes the shared in-process model "
                "used by all model-backed roles on later rollouts."
                if full_finetune
                else "Role adapters can isolate direct parameter updates by role."
            ),
        },
        "reward_config": reward_config,
        "local_penalties": {
            "invalid_json": float(invalid_json_penalty),
            "invalid_action_schema": float(invalid_schema_penalty),
            "credit_assignment": format_credit_assignment,
            "strict_format_contract": {
                "exact_single_json_object": True,
                "markdown_or_surrounding_prose_is_invalid": True,
                "compact_repair_is_penalized": True,
                "only_response_format_keys": True,
                "required_keys_types_and_allowed_values": True,
            },
        },
        "max_parse_retries": int(max_parse_retries),
        "sampling": _redact_secrets(
            OmegaConf.to_container(config.actor_rollout_ref.rollout, resolve=True)
        ),
        "seed": OmegaConf.select(config, "data.seed", default=None),
        "campaign_seed": OmegaConf.select(
            config, "teamgym.campaign_seed", default=None
        ),
        "seeds": {
            "data": OmegaConf.select(config, "data.seed", default=None),
            "rollout": OmegaConf.select(
                config, "actor_rollout_ref.rollout.seed", default=None
            ),
            "trainer": OmegaConf.select(config, "trainer.seed", default=None),
        },
        "resume": {
            "mode": OmegaConf.select(config, "trainer.resume_mode", default="disable"),
            "from_path": OmegaConf.select(
                config, "trainer.resume_from_path", default=None
            ),
        },
        "command": " ".join(_shell_quote(value) for value in sys.argv),
        # Some VERL/RLLM resolvers (notably ``mul``) are registered only when
        # the trainer starts. Preserve those expressions in this early
        # provenance snapshot instead of resolving the manifest prematurely.
        "config_snapshot": _redact_secrets(
            OmegaConf.to_container(config, resolve=False)
        ),
    }


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _optional_env_int(name: str) -> int | None:
    value = _optional_env(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        if name == "rllm":
            return "repository_worktree"
        return None


def _cached_model_revision(model: str) -> str | None:
    try:
        from huggingface_hub import constants

        model_dir = Path(constants.HF_HUB_CACHE) / f"models--{model.replace('/', '--')}"
        ref = model_dir / "refs" / "main"
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else None
    except Exception:  # noqa: BLE001
        return None


def _shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "_./:=+,-" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


_SECRET_KEY_PARTS = ("api_key", "password", "secret", "token")


def _redact_secrets(value: Any) -> Any:
    """Remove credentials from provenance snapshots without mutating config."""
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if any(part in str(key).lower() for part in _SECRET_KEY_PARTS)
                else _redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value
