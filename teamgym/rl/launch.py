"""Train on prepared TeamGym data; preview configurations without GPUs or data reads."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit

from teamgym.interaction.structures import communication_structure_preset
from .rollout.profiles import canonical_profile_name, get_rollout_profile, profile_names
from .rollout.rewards import REWARD_CONFIGS


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TRAINING_PROFILES = tuple(
    name for name in profile_names()
    if not name.startswith("retrieval_probe_") and name != "paper_matched_participant_opportunities"
)
TRAINING_MODES = ("joint", "participant_only", "coordinator_only")


def _validate_options(args: argparse.Namespace, *, require_fixed_roles: bool = True) -> dict:
    profile = get_rollout_profile(args.profile)
    structure = communication_structure_preset(profile.communication_structure)
    devices = [value.strip() for value in args.devices.split(",")]
    if not all(devices) or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain distinct, nonempty CUDA device IDs")
    positive = (
        "updates", "epochs", "batch_size", "rollouts", "tensor_parallel_size",
        "save_every", "validate_every", "cpus", "lora_rank", "lora_alpha",
    )
    if any(getattr(args, name) < 1 for name in positive):
        raise ValueError("Update, epoch, batch, rollout, parallelism, interval, CPU, and LoRA counts must be positive")
    if args.rollouts < 2:
        raise ValueError("GRPO training requires --rollouts >= 2")
    if len(devices) % args.tensor_parallel_size:
        raise ValueError("Device count must be divisible by --tensor-parallel-size")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    splits = [args.train_split, args.validation_split] + ([args.test_split] if args.test_split else [])
    if len(set(splits)) != len(splits) or any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name) for name in splits):
        raise ValueError("Split names must be distinct filenames without directory components")
    maximum = max(profile.model_max_output_tokens, profile.participant_model_max_output_tokens or 0)
    response_length = args.max_response_length if args.max_response_length is not None else maximum
    prompt_length = args.max_prompt_length if args.max_prompt_length is not None else max(
        4096, profile.budgets.participant_context_token_budget + 4096
    )
    if prompt_length < 1 or response_length < maximum:
        raise ValueError(f"Prompt length must be positive and response length must be at least the profile cap ({maximum})")
    if not structure.coordinator_present and args.training_mode != "participant_only":
        raise ValueError("A profile without a coordinator requires --training-mode participant_only")
    participant_backend = args.participant_backend or (
        "external_openai" if args.training_mode == "coordinator_only" else "rollout"
    )
    coordinator_fixed = structure.coordinator_present and args.training_mode == "participant_only"
    participant_fixed = args.training_mode == "coordinator_only" and participant_backend == "external_openai"
    if args.training_mode == "coordinator_only":
        if participant_backend == "rollout":
            raise ValueError("Coordinator-only training requires scripted or external_openai participants")
    elif participant_backend != "rollout":
        raise ValueError("Participant-only and joint training require rollout participants")
    for role, needed in (("coordinator", coordinator_fixed), ("participant", participant_fixed)):
        url = getattr(args, f"fixed_{role}_base_url")
        model = getattr(args, f"fixed_{role}_model")
        if (url or model) and not needed:
            raise ValueError(f"Fixed {role} endpoint settings are unused in this training configuration")
        if bool(url) != bool(model) or (needed and require_fixed_roles and not url):
            raise ValueError(f"Fixed {role} requires both --fixed-{role}-base-url and --fixed-{role}-model")
        if url:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(f"Fixed {role} base URL must be an HTTP(S) endpoint")
    return dict(profile=profile, devices=devices, participant_backend=participant_backend,
                fixed_coordinator_backend="external_openai" if coordinator_fixed else "deterministic",
                max_prompt_length=prompt_length, max_response_length=response_length)


def build_plan(args: argparse.Namespace) -> dict:
    """Resolve explicit launcher settings without reading datasets or model files."""
    options = _validate_options(args)
    if not args.model or args.run_dir is None:
        raise ValueError("--model and --run-dir are required for training or --dry-run")
    run_dir = args.run_dir.expanduser().resolve()
    import_roots = [str(REPO_ROOT)]
    if (REPO_ROOT / "verl/verl/__init__.py").is_file():
        import_roots.insert(0, str(REPO_ROOT / "verl"))
    full = args.finetuning == "full"
    logical_batch = args.batch_size * args.rollouts
    token_length = options["max_prompt_length"] + options["max_response_length"]
    env = {
        "TEAMGYM_PYTHON": sys.executable,
        "PYTHONPATH": os.pathsep.join(import_roots),
        "MODEL_PATH": str(Path(args.model).expanduser().resolve()),
        "PREPARED_DATA_DIR": str(args.prepared_data_dir.expanduser().resolve()),
        "TRAIN_SPLIT": args.train_split, "VALIDATION_SPLIT": args.validation_split,
        "RUN_NAME": run_dir.name, "RUN_DIR": str(run_dir),
        "CHECKPOINT_DIR": str(run_dir / "checkpoints"),
        "HYDRA_RUN_DIR": str(run_dir / "hydra"),
        "CUDA_VISIBLE_DEVICES": ",".join(options["devices"]),
        "N_GPUS_PER_NODE": len(options["devices"]),
        "ROLLOUT_TENSOR_PARALLEL_SIZE": args.tensor_parallel_size,
        "TRAINING_MODE": args.training_mode,
        "PARTICIPANT_BACKEND": options["participant_backend"],
        "FIXED_COORDINATOR_BACKEND": options["fixed_coordinator_backend"],
        "FIXED_COORDINATOR_BASE_URL": args.fixed_coordinator_base_url or "",
        "FIXED_COORDINATOR_MODEL": args.fixed_coordinator_model or "",
        "FIXED_PARTICIPANT_BASE_URL": args.fixed_participant_base_url or "",
        "FIXED_PARTICIPANT_MODEL": args.fixed_participant_model or "",
        "PARTICIPANT_REPLAY_PATH": "",
        "FIXED_COORDINATOR_TEMPERATURE": "0.0", "FIXED_COORDINATOR_VAL_TEMPERATURE": "0.0",
        "FIXED_PARTICIPANT_TEMPERATURE": "0.0", "FIXED_PARTICIPANT_VAL_TEMPERATURE": "0.0",
        "ROLLOUT_PROFILE": options["profile"].name, "REWARD_CONFIG": args.reward_config,
        "FULL_FINETUNE": int(full), "LORA_RANK": args.lora_rank, "LORA_ALPHA": args.lora_alpha,
        "ACTOR_MODEL_DTYPE": "bfloat16" if full else "fp32",
        "ACTOR_PARAM_OFFLOAD": full, "ACTOR_OPTIMIZER_OFFLOAD": full,
        "REF_PARAM_OFFLOAD": False, "USE_KL_LOSS": False, "KL_LOSS_COEF": "0.001",
        "ACTOR_LR": args.learning_rate,
        "TRAIN_BATCH_SIZE": args.batch_size, "ROLLOUT_N": args.rollouts,
        "PPO_MINI_BATCH_SIZE": logical_batch, "N_PARALLEL_TASKS": logical_batch,
        "DATA_MAX_PROMPT_LENGTH": options["max_prompt_length"],
        "DATA_MAX_RESPONSE_LENGTH": options["max_response_length"],
        "PPO_MAX_TOKEN_LEN_PER_GPU": max(16384, token_length),
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS": max(8192, token_length),
        "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.35" if full else "0.45",
        "ROLLOUT_ENFORCE_EAGER": False, "RAY_NUM_CPUS": args.cpus,
        "TOTAL_TRAINING_STEPS": args.updates + 1, "TOTAL_EPOCHS": args.epochs,
        "SAVE_FREQ": args.save_every, "TEST_FREQ": args.validate_every,
        "TRAINER_LOGGER": "['console']", "VAL_BEFORE_TRAIN": False, "VAL_ONLY": False,
        "DISABLE_VALIDATION": False, "DATA_SHUFFLE": True, "VALIDATION_SHUFFLE": False,
        "RESUME_MODE": "disable", "RESUME_FROM_PATH": "",
        "FORMAT_CREDIT_ASSIGNMENT": "per_attempt", "MAX_PARSE_RETRIES": "1",
        "INVALID_JSON_PENALTY": "-0.1", "INVALID_SCHEMA_PENALTY": "-0.1",
        "PARTICIPANT_FORMAT_REPAIR_FOR_EVAL": False,
        "WRITTEN_JUDGE_FAIL_ON_ERROR": True,
        "WRITTEN_JUDGE_CACHE_DIR": str(run_dir / "written_judge_cache"),
        "RLLM_DISABLE_TRAJECTORY_VISUALIZATION": "1",
    }
    for name in ("CAMPAIGN_SEED", "DATA_SEED", "ROLLOUT_SEED", "TRAINER_SEED"):
        env[name] = args.seed
    return {
        "schema_version": "teamgym_training_launch_v1",
        "cwd": str(Path.cwd().resolve()),
        "command": ["bash", str(SCRIPT_DIR / "train_backend.sh")],
        "environment": {name: str(value) for name, value in env.items()},
        "profile": options["profile"].to_dict(),
        "policy_weights": "shared" if args.training_mode == "joint" else "one trainable role",
        "optimizer_update_limit": args.updates,
        "test_split_for_checks_only": args.test_split,
        "requires_written_judge": args.reward_config == "written_holistic_score",
        "paper_recipe": False,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data-dir", type=Path, required=True)
    parser.add_argument("--model", help="Local immutable model snapshot; required except with --check-data")
    parser.add_argument("--run-dir", type=Path, help="Fresh output directory; required except with --check-data")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--test-split", help="Optional held-out split checked for overlap; never used for selection")
    parser.add_argument("--profile", type=canonical_profile_name, choices=TRAINING_PROFILES, required=True)
    parser.add_argument("--reward-config", choices=sorted(REWARD_CONFIGS), required=True)
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default="joint",
                        help="joint trains shared policy weights for both roles")
    parser.add_argument("--participant-backend", choices=("rollout", "scripted", "external_openai"))
    parser.add_argument("--fixed-coordinator-base-url")
    parser.add_argument("--fixed-coordinator-model", help="Model ID served by the fixed coordinator endpoint")
    parser.add_argument("--fixed-participant-base-url")
    parser.add_argument("--fixed-participant-model", help="Model ID served by the fixed participant endpoint")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=2, help="Optimizer update limit; epochs can stop training earlier")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--finetuning", choices=("lora", "full"), default="lora")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-prompt-length", type=int)
    parser.add_argument("--max-response-length", type=int)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=8)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Print settings without data/model reads or job startup")
    action.add_argument("--check-data", action="store_true", help="Check prepared files locally with RL preparation dependencies")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.check_data:
            _validate_options(args, require_fixed_roles=False)
            plan = None
        else:
            plan = build_plan(args)
            if args.dry_run:
                print(json.dumps(plan, indent=2, sort_keys=True))
                return
        from .preparation.check_training_data import check_prepared_splits

        report = check_prepared_splits(
            args.prepared_data_dir, train_split=args.train_split,
            validation_split=args.validation_split, test_split=args.test_split,
            reward_config=args.reward_config,
        )
        if args.check_data:
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        run_dir = Path(plan["environment"]["RUN_DIR"])
        if run_dir.exists():
            raise ValueError(f"Run directory already exists: {run_dir}; choose a fresh --run-dir")
        if not (Path(plan["environment"]["MODEL_PATH"]) / "config.json").is_file():
            raise ValueError("--model must point to a local snapshot containing config.json")
        run_dir.mkdir(parents=True)
        (run_dir / "launch_plan.json").write_text(json.dumps({"plan": plan, "data_check": report}, indent=2) + "\n")
        print(json.dumps({"training_plan": plan, "data_check": report}, indent=2, sort_keys=True), flush=True)
        result = subprocess.run(plan["command"], cwd=plan["cwd"],
                                env={**os.environ, **plan["environment"]}, check=False)
        raise SystemExit(result.returncode)
    except ImportError as error:
        parser.error(f"Data checks need the RL preparation environment: {error}")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
