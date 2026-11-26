"""Run fully collaborative interaction evaluation on normalized legal datapoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from teamgym.core.io import write_json
from teamgym.core.dataset import iter_common_datapoints
from teamgym.interaction.environment import run_episode
from teamgym.interaction.evaluation import (
    aggregate_evaluations,
    build_failed_evaluation,
    evaluate_episode,
)
from teamgym.interaction.evaluation_client import make_evaluation_client as _make_llm_client
from teamgym.interaction.evaluation_client import evaluation_model_manifest
from teamgym.interaction.llm_actors import (
    LLMCoordinatorActor,
    LLMFinalDecisionActor,
    LLMParticipantActor,
    LLMTwoStageParticipantActor,
    PromptTraceRecorder,
)
from teamgym.interaction.structures import communication_structure_names, communication_structure_preset
from teamgym.interaction.types import BudgetSpec
from teamgym.llm.openai_client import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)
from teamgym.interaction.rui_context import rui_only_datapoint
from teamgym.visualization.interaction_trajectory_report import build_report


def main() -> None:
    args = _parse_args()
    if args.context_variant == "piu_only":
        run_interaction_evaluation(
            args,
            datapoint_transform=rui_only_datapoint,
            run_label="rui_context_interaction_eval",
            manifest_extra={
                "context_variant": "rui_only",
                "context_variant_description": "Participant private context units contain only owned RIU text.",
            },
        )
        return
    run_interaction_evaluation(args)


def run_interaction_evaluation(
    args: argparse.Namespace,
    *,
    datapoint_transform: Any | None = None,
    run_label: str = "interaction_eval",
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    datapoints, input_diagnostics = load_evaluation_datapoints(args)
    if datapoint_transform is not None:
        datapoints = [datapoint_transform(datapoint) for datapoint in datapoints]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser() / _run_output_dir_name(
        stamp,
        run_label,
        args.num_shards,
        args.shard_index,
    )
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=False)

    structures = [communication_structure_preset(name) for name in args.communication_structure]
    budgets = BudgetSpec(
        turn_budget=args.turn_budget,
        participant_utterance_token_budget=args.participant_utterance_token_budget,
        coordinator_utterance_token_budget=args.coordinator_utterance_token_budget,
        final_rationale_token_budget=args.final_rationale_token_budget,
        participant_reasoning_token_budget=args.participant_reasoning_token_budget,
        coordinator_reasoning_token_budget=args.coordinator_reasoning_token_budget,
        final_reasoning_token_budget=args.final_reasoning_token_budget,
        participant_context_token_budget=args.participant_context_token_budget,
        participant_public_read_policy=args.participant_public_read_policy,
        participant_public_read_token_budget=args.participant_public_read_token_budget,
        participant_public_read_seed=args.participant_public_read_seed,
        public_read_scope=args.public_read_scope,
        public_read_charge_current_message=args.public_read_charge_current_message,
    ).checked()
    manifest = {
        "schema_version": "teamgym_interaction_eval_run",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": args.input_root,
        "output_dir": str(output_dir),
        "datapoint_count": len(datapoints),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "communication_structures": [structure.to_dict() for structure in structures],
        "budgets": budgets.__dict__,
        "actor_runtime": (
            "llm_two_stage_participant"
            if args.participant_response_mode == "two_stage"
            else "llm"
        ),
        "llm": {
            **evaluation_model_manifest(args),
            "participant_role_name_mode": args.participant_role_name_mode,
            "participant_confidence": args.participant_confidence,
            "participant_response_mode": args.participant_response_mode,
        },
        "input_diagnostics": input_diagnostics,
        "prompt_tracing_enabled": bool(args.trace_prompts),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    write_json(output_dir / "run_manifest.json", manifest)

    llm_client = _make_llm_client(args, output_dir)
    evaluations: list[dict[str, Any]] = []
    for datapoint_index, datapoint in enumerate(datapoints, start=1):
        for structure in structures:
            slug = _slug(f"{datapoint_index:04d}_{datapoint.get('dataset')}_{datapoint.get('instance_id')}_{structure.name}")
            recorder = PromptTraceRecorder(trajectories_dir / f"{slug}.prompts" if args.trace_prompts else None)
            try:
                episode = run_episode(
                    datapoint,
                    structure,
                    budgets=budgets,
                    participant_actor=(
                        LLMTwoStageParticipantActor
                        if args.participant_response_mode == "two_stage"
                        else LLMParticipantActor
                    )(
                        llm_client,
                        recorder,
                        role_name_mode=args.participant_role_name_mode,
                        include_confidence=args.participant_confidence,
                    ),
                    coordinator_actor=LLMCoordinatorActor(
                        llm_client,
                        recorder,
                        role_name_mode=args.participant_role_name_mode,
                    ),
                    final_actor=LLMFinalDecisionActor(
                        llm_client,
                        recorder,
                        role_name_mode=args.participant_role_name_mode,
                    ),
                )
                evaluation = evaluate_episode(episode, datapoint)
                trajectory_payload = {"episode": episode, "evaluation": evaluation}
            except Exception as exc:  # noqa: BLE001
                if not args.continue_on_error:
                    raise
                evaluation = build_failed_evaluation(
                    datapoint,
                    communication_structure=structure.name,
                    error=exc,
                )
                trajectory_payload = {
                    "episode": None,
                    "evaluation": evaluation,
                    "prompt_records": list(recorder.prompt_records),
                }
            evaluations.append(evaluation)
            write_json(trajectories_dir / f"{slug}.json", trajectory_payload)
            _append_jsonl(output_dir / "eval_results.jsonl", evaluation)

    summary = {
        "input_diagnostics": input_diagnostics,
        "aggregate": aggregate_evaluations(evaluations),
        "results": evaluations,
    }
    write_json(output_dir / "eval_summary.json", summary)
    report_path = build_report(input_path=output_dir) if args.render_html else None
    result = {
        "output_dir": str(output_dir),
        "report_path": str(report_path) if report_path else None,
        "aggregate": summary["aggregate"],
        "input_diagnostics": input_diagnostics,
    }
    print(json.dumps(result, indent=2))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--context-variant", choices=("full_context", "piu_only"), default="full_context")
    parser.add_argument("--output-dir", default="generated/teamgym_interaction_eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--communication-structure",
        action="append",
        choices=communication_structure_names(),
        default=None,
        help="May be repeated. Required unless --all-communication-structures is passed.",
    )
    parser.add_argument(
        "--all-communication-structures",
        action="store_true",
        help="Run every prompt-debug communication structure supported by this runner.",
    )
    parser.add_argument("--turn-budget", type=int, default=3)
    parser.add_argument("--participant-utterance-token-budget", type=int, default=80)
    parser.add_argument("--coordinator-utterance-token-budget", type=int, default=80)
    parser.add_argument("--final-rationale-token-budget", type=int, default=120)
    parser.add_argument("--participant-reasoning-token-budget", type=int, default=128)
    parser.add_argument("--coordinator-reasoning-token-budget", type=int, default=128)
    parser.add_argument("--final-reasoning-token-budget", type=int, default=256)
    parser.add_argument("--participant-context-token-budget", type=int, default=20000)
    parser.add_argument(
        "--participant-public-read-policy",
        choices=["unlimited", "fifo", "stochastic_contiguous"],
        default="unlimited",
        help="Policy for bounded observation of previously unread public text.",
    )
    parser.add_argument("--participant-public-read-token-budget", type=int, default=0)
    parser.add_argument("--participant-public-read-seed", type=int, default=0)
    parser.add_argument(
        "--public-read-scope",
        choices=["participant_only", "all_actors"],
        default="participant_only",
        help="Apply bounded public reading only to participants or to participants and coordinators.",
    )
    parser.add_argument(
        "--public-read-charge-current-message",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Charge the current directed message against the receiving actor's public-read budget.",
    )
    parser.add_argument(
        "--participant-response-mode",
        choices=["direct", "two_stage"],
        default="direct",
        help=(
            "Use one direct participant response or a private-think call followed by a public-message call. "
            "The two-stage mode uses participant reasoning and utterance budgets as independent hard field caps."
        ),
    )
    parser.add_argument(
        "--participant-role-name-mode",
        choices=["hidden", "datapoint", "participant_id", "generic"],
        default="hidden",
        help="Controls actor-visible participant role names in participant and coordinator payloads.",
    )
    parser.add_argument(
        "--participant-confidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request and save optional participant confidence values. Off by default.",
    )
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_LLM_ENDPOINT,
        choices=["/responses", "/chat/completions", "/messages", "/invoke"],
    )
    parser.add_argument("--credentials-file", default="api_key/api_key.py")
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--request-retry-attempts", type=int, default=3)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record a failed evaluation row and continue when a datapoint/model call fails.",
    )
    parser.add_argument(
        "--trace-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the exact system/user prompts and raw LLM outputs beside each trajectory.",
    )
    parser.add_argument(
        "--render-html",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render interaction_report.html in the output directory.",
    )
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.communication_structure and args.all_communication_structures:
        parser.error("Use either --communication-structure or --all-communication-structures, not both.")
    if args.all_communication_structures:
        args.communication_structure = communication_structure_names()
    if args.communication_structure is None:
        parser.error("Provide --communication-structure at least once, or pass --all-communication-structures.")
    if args.max_output_tokens < (
        max(args.participant_utterance_token_budget, args.coordinator_utterance_token_budget, args.final_rationale_token_budget)
        + max(args.participant_reasoning_token_budget, args.coordinator_reasoning_token_budget, args.final_reasoning_token_budget)
        + 64
    ):
        raise ValueError(
            "--max-output-tokens should be large enough to cover reasoning budget, public response budget, and JSON overhead."
        )
    return args


def load_evaluation_datapoints(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    skipped: dict[str, list[int]] = {}

    def record_skip(path: Path, line_number: int) -> None:
        skipped.setdefault(str(path), []).append(line_number)

    rows = list(iter_common_datapoints(args.input_root, on_skipped_row=record_skip))
    selected = _select_shard(rows, args.num_shards, args.shard_index)
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]
    skipped_count = sum(len(lines) for lines in skipped.values())
    if not selected:
        raise ValueError(
            f"No common_datapoint rows selected from {args.input_root} "
            f"({len(rows)} normalized, {skipped_count} unsupported across {len(skipped)} files)"
        )
    return selected, {
        "normalized_row_count": len(rows),
        "selected_row_count": len(selected),
        "skipped_row_count": skipped_count,
        "skipped_rows_by_file": [
            {"path": path, "row_count": len(lines), "line_numbers": lines}
            for path, lines in skipped.items()
        ],
    }


def _select_shard(rows: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_output_dir_name(stamp: str, run_label: str, num_shards: int, shard_index: int) -> str:
    if num_shards <= 1:
        return f"{stamp}_{run_label}"
    width = max(2, len(str(num_shards - 1)))
    return f"{stamp}_shard{shard_index:0{width}d}of{num_shards:0{width}d}_{run_label}"


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)[:180]


if __name__ == "__main__":
    main()
