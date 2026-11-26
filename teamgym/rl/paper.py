"""Portable entry point for the retained paper training recipes.

Dry runs use only the standard library. The shell backends remain responsible
for immutable actor servers, FSDP, rollout collection, and checkpoint storage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
REGIMES = ("legal-participant", "legal-coordinator", "legal-joint", "finance-handoff")


def build_plan(args: argparse.Namespace) -> dict:
    legal = args.regime.startswith("legal-")
    if (args.eval_checkpoint or args.eval_split) and not args.evaluate:
        raise ValueError("--eval-checkpoint and --eval-split require --evaluate")
    if args.eval_split and args.eval_split not in (("validation_cuad", "validation_maud") if legal else ("validation", "test")):
        raise ValueError("--eval-split does not belong to this regime")
    if args.regime == "legal-coordinator" and not args.fixed_participant_model:
        raise ValueError("legal-coordinator requires --fixed-participant-model (merged Stage B weights)")
    devices = args.devices.split(",")
    if len(devices) != 4 or any(not device.strip() for device in devices) or len(set(devices)) != 4:
        raise ValueError("The retained paper recipes require four distinct GPU device IDs")
    if args.updates < 1 or args.updates % 10:
        raise ValueError("--updates must be a positive multiple of 10 (checkpoint/selection interval)")
    run_dir = args.run_dir.expanduser().resolve()
    data_dir = args.prepared_data_dir.expanduser().resolve()
    import_roots = [str(REPO_ROOT)]
    if (REPO_ROOT / "verl" / "verl" / "__init__.py").is_file():
        import_roots.insert(0, str(REPO_ROOT / "verl"))
    mode = {
        "legal-participant": "participant_only",
        "legal-coordinator": "coordinator_only",
        "legal-joint": "joint",
        "finance-handoff": "participant_only",
    }[args.regime]
    env = {
        "TEAMGYM_PYTHON": sys.executable,
        "PYTHONPATH": os.pathsep.join(import_roots),
        "MODEL_PATH": str(Path(args.model).expanduser().resolve()),
        "PREPARED_DATA_DIR": str(data_dir),
        "RUN_NAME": run_dir.name,
        "RUN_DIR": str(run_dir),
        "CHECKPOINT_DIR": str(run_dir / "checkpoints"),
        "WANDB_DIR": str(run_dir / "wandb"),
        "TRAINER_LOGGER": "['console']",
        "CUDA_VISIBLE_DEVICES": args.devices,
        "TEAMGYM_CUDA_VISIBLE_DEVICES": args.devices,
        "N_GPUS_PER_NODE": "4",
        "FULL_FINETUNE": "1",
        "TRAINING_MODE": mode,
        "PARTICIPANT_BACKEND": "rollout",
        "TRAIN_SPLIT": "train",
        "VALIDATION_SPLIT": "validation_cuad" if legal else "validation",
        "CONTEXT_VARIANT": "full_context" if legal else "riu_only",
        "ROLLOUT_PROFILE": "written_64_query_private_full_context" if legal else "finance_riu_participant_handoff_15",
        "REWARD_CONFIG": "written_holistic_score",
        "FORMAT_CREDIT_ASSIGNMENT": "per_attempt",
        "INVALID_JSON_PENALTY": "-0.1",
        "INVALID_SCHEMA_PENALTY": "-0.1",
        "MAX_PARSE_RETRIES": "1",
        "PARTICIPANT_FORMAT_REPAIR_FOR_EVAL": "False",
        "TRAIN_BATCH_SIZE": "8",
        "ROLLOUT_N": "8" if legal else "16",
        "PPO_MINI_BATCH_SIZE": "64" if legal else "128",
        "N_PARALLEL_TASKS": ("8" if mode == "joint" else "16") if legal else "64",
        "RAY_NUM_CPUS": "32" if legal else "64",
        "ACTOR_MODEL_DTYPE": "fp32" if legal else "bfloat16",
        "ACTOR_LR": "1e-6",
        "ACTOR_PARAM_OFFLOAD": "True",
        "ACTOR_OPTIMIZER_OFFLOAD": "True",
        "REF_PARAM_OFFLOAD": "True" if legal else "False",
        "USE_KL_LOSS": "True" if legal else "False",
        "KL_LOSS_COEF": "0.001",
        "DATA_MAX_PROMPT_LENGTH": "32640" if legal else "16384",
        "DATA_MAX_RESPONSE_LENGTH": "64" if legal else "750",
        "PPO_MAX_TOKEN_LEN_PER_GPU": "32768" if legal else "24576",
        "ROLLOUT_TENSOR_PARALLEL_SIZE": "1",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS": "32768",
        "ROLLOUT_ENFORCE_EAGER": "False",
        "ROLLOUT_GPU_MEMORY_UTILIZATION": ("0.18" if mode == "joint" else "0.35") if legal else "0.45",
        "TOTAL_TRAINING_STEPS": str(args.updates + 1),
        "TOTAL_EPOCHS": "3" if legal else "20",
        "SAVE_FREQ": "10",
        "TEST_FREQ": "10",
        "MAX_ACTOR_CKPTS_TO_KEEP": "12",
        "VAL_BEFORE_TRAIN": "True" if legal else "False",
        "VAL_ONLY": "False",
        "DISABLE_VALIDATION": "False",
        "DATA_SHUFFLE": "True",
        "VALIDATION_SHUFFLE": "False",
        "RESUME_MODE": "disable",
        "RESUME_FROM_PATH": "",
        "WRITTEN_JUDGE_FAIL_ON_ERROR": "True",
        "WRITTEN_JUDGE_CACHE_DIR": str(run_dir / "written_judge_cache"),
        "RLLM_DISABLE_TRAJECTORY_VISUALIZATION": "1",
    }
    for name in ("CAMPAIGN_SEED", "DATA_SEED", "ROLLOUT_SEED", "TRAINER_SEED", "FROZEN_COORDINATOR_SEED", "FROZEN_PARTICIPANT_SEED"):
        env[name] = str(args.seed)
    env["FIXED_COORDINATOR_BACKEND"] = "base_model" if legal else "deterministic"
    # Never inherit a server from a previous experiment for these fresh runs.
    env["FIXED_COORDINATOR_BASE_URL"] = ""
    env["FIXED_COORDINATOR_MODEL"] = env["MODEL_PATH"]
    env["FIXED_PARTICIPANT_BASE_URL"] = ""
    env["FIXED_PARTICIPANT_MODEL"] = str(Path(args.fixed_participant_model).expanduser().resolve()) if args.fixed_participant_model else env["MODEL_PATH"]
    env["PARTICIPANT_REPLAY_PATH"] = ""
    env["FIXED_COORDINATOR_TEMPERATURE"] = "0.0"
    env["FIXED_COORDINATOR_VAL_TEMPERATURE"] = "0.0"
    env["FIXED_PARTICIPANT_TEMPERATURE"] = "0.7"
    env["FIXED_PARTICIPANT_VAL_TEMPERATURE"] = "0.0"
    if legal and mode != "joint":
        # The immutable role shares the last physical GPU with the trainer.
        env["FROZEN_COORDINATOR_GPU"] = devices[-1]
        env["FROZEN_PARTICIPANT_GPU"] = devices[-1]
    if args.regime == "legal-coordinator":
        env["PARTICIPANT_BACKEND"] = "external_openai"
        env["N_PARALLEL_TASKS"] = "8"
        env["FROZEN_PARTICIPANT_GPU_MEMORY_UTILIZATION"] = "0.14"
        env["FROZEN_PARTICIPANT_ENFORCE_EAGER"] = "True"
    script = "train_legal_roles.sh" if legal else "train_backend.sh"
    command = ["bash", str(SCRIPT_DIR / script)]
    command += [
        "actor_rollout_ref.actor.optim.weight_decay=0.01",
        "actor_rollout_ref.actor.clip_ratio=0.2",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.2",
        "actor_rollout_ref.actor.clip_ratio_c=3.0",
    ]
    if args.evaluate:
        env.update({
            "VAL_ONLY": "True", "VAL_BEFORE_TRAIN": "True",
            "VALIDATION_SPLIT": args.eval_split or ("validation_maud" if legal else "test"),
            "SAVE_FREQ": "-1", "TEST_FREQ": "-1", "ROLLOUT_N": "1",
            "TRAIN_BATCH_SIZE": "8" if legal else "1",
            "PPO_MINI_BATCH_SIZE": "8" if legal else "4",
        })
        if args.eval_checkpoint:
            env["RESUME_MODE"] = "resume_path"
            env["RESUME_FROM_PATH"] = str(args.eval_checkpoint.expanduser().resolve())
        command.append("actor_rollout_ref.actor.checkpoint.load_contents=['model']")
    elif legal:
        command += [
            "+teamgym.retain_cuad_best_and_latest=True",
            "+teamgym.checkpoint_selection_metric=val/in/unknown/teamgym_holistic_score_cuad",
        ]
    else:
        command += [
            "+teamgym.retain_best_and_latest=True",
            "+teamgym.checkpoint_selection_split=validation_findver_and_finlongdocqa",
            "+teamgym.checkpoint_selection_metric=val/in/unknown/teamgym_holistic_score_findver",
        ]
    if legal and mode == "joint":
        command.append("+teamgym.two_policy=True")
    return {"regime": args.regime, "optimizer_updates": 0 if args.evaluate else args.updates, "cwd": str(Path.cwd().resolve()), "command": command, "environment": env}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--prepared-data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Immutable local model snapshot; see README for paper revisions")
    parser.add_argument("--fixed-participant-model", help="Merged Stage B model for legal-coordinator")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--evaluate", action="store_true", help="Greedy endpoint evaluation; omit --eval-checkpoint for the base model")
    parser.add_argument("--eval-checkpoint", type=Path, help="Native global_step_N directory; load model weights only for evaluation")
    parser.add_argument("--eval-split", help="Default: held-out MAUD for legal; four-participant FinLong for finance")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Print command and explicit settings without reading data or starting a job")
    action.add_argument("--check-data", action="store_true", help="Validate local prepared data without loading a model or contacting a judge")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return
        from teamgym.rl.preparation.check_paper_data import check_prepared_data

        report = check_prepared_data(args.prepared_data_dir, finance=args.regime == "finance-handoff")
        if args.check_data:
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        run_dir = Path(plan["environment"]["RUN_DIR"])
        if run_dir.exists():
            raise ValueError(f"Run directory already exists: {run_dir}; choose a fresh --run-dir")
        models = [plan["environment"]["MODEL_PATH"]] + ([plan["environment"]["FIXED_PARTICIPANT_MODEL"]] if args.fixed_participant_model else [])
        for model in models:
            if not (Path(model).expanduser() / "config.json").is_file():
                raise ValueError(f"Missing local model snapshot config.json: {model}")
        if args.eval_checkpoint:
            actor = args.eval_checkpoint.expanduser().resolve() / "actor"
            roles = [actor / "coordinator", actor / "participant"] if args.regime == "legal-joint" else [actor]
            if not args.eval_checkpoint.name.startswith("global_step_") or any(len(list(role.glob("model_world_size_4_rank_*.pt"))) != 4 for role in roles):
                raise ValueError("Expected a native global_step_N checkpoint with four model shards per policy")
        print(json.dumps({"training_plan": plan, "data_check": report}, indent=2, sort_keys=True), flush=True)
        result = subprocess.run(plan["command"], cwd=plan["cwd"], env={**os.environ, **plan["environment"]}, check=False)
        raise SystemExit(result.returncode)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
