"""Run centralized single-agent legal baselines with controlled evidence access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from teamgym.core.io import write_json
from teamgym.interaction.actor_utils import (
    count_tokens,
    distributed_truncate_token_segments,
    truncate_tokens,
)
from teamgym.core.dataset import DEFAULT_LEGAL_DATASET_PATH
from teamgym.interaction.environment import EPISODE_SCHEMA
from teamgym.interaction.evaluation import (
    aggregate_evaluations,
    build_failed_evaluation,
    evaluate_episode,
)
from teamgym.interaction.evaluation_client import evaluation_model_manifest
from teamgym.interaction.evaluation_client import make_evaluation_client as _make_client
from teamgym.interaction.llm_actors import PromptTraceRecorder
from teamgym.interaction.rui_context import model_common_information, rui_context_units
from teamgym.interaction.run_evaluation import (
    _append_jsonl,
    _run_output_dir_name,
    _slug,
    load_evaluation_datapoints,
)
from teamgym.llm.openai_client import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    OpenAIJsonClient,
)
from teamgym.visualization.interaction_trajectory_report import build_report


CONTEXT_ALL_RIUS = "all_rius"
CONTEXT_FULL_SOURCE = "full_source"


def main() -> None:
    """Run the backward-compatible Single-Agent All-RIU baseline."""

    _main(CONTEXT_ALL_RIUS)


def main_full_context() -> None:
    """Run the Single-Agent Full-Context baseline."""

    _main(CONTEXT_FULL_SOURCE)


def _main(context_variant: str) -> None:
    args = _parse_args(context_variant)
    variant = _context_variant_metadata(context_variant)
    datapoints, input_diagnostics = load_evaluation_datapoints(args)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser() / _run_output_dir_name(
        stamp,
        variant["run_slug"],
        args.num_shards,
        args.shard_index,
    )
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=False)
    # The manifest records the selected context variant and model settings.
    # Full-context support changed this manifest independently of the episode.
    manifest = {
        "schema_version": "teamgym_single_agent_eval_run",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": args.input_root,
        "output_dir": str(output_dir),
        "datapoint_count": len(datapoints),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "baseline": variant["baseline"],
        "context_variant": variant["manifest_context_variant"],
        "llm": evaluation_model_manifest(args),
        "input_diagnostics": input_diagnostics,
        "context_token_budget": args.context_token_budget,
        "final_rationale_token_budget": args.final_rationale_token_budget,
        "final_reasoning_token_budget": args.final_reasoning_token_budget,
        "prompt_tracing_enabled": bool(args.trace_prompts),
    }
    write_json(output_dir / "run_manifest.json", manifest)

    client = _make_client(args, output_dir)
    evaluations: list[dict[str, Any]] = []
    for datapoint_index, datapoint in enumerate(datapoints, start=1):
        slug = _slug(
            f"{datapoint_index:04d}_{datapoint.get('dataset')}_"
            f"{datapoint.get('instance_id')}_{variant['baseline']}"
        )
        recorder = PromptTraceRecorder(trajectories_dir / f"{slug}.prompts" if args.trace_prompts else None)
        try:
            episode = run_single_agent_episode(
                datapoint,
                client=client,
                recorder=recorder,
                args=args,
                context_variant=context_variant,
            )
            evaluation = evaluate_episode(episode, datapoint)
            evaluation["communication_structure"] = variant["baseline"]
            trajectory_payload = {"episode": episode, "evaluation": evaluation}
        except Exception as exc:  # noqa: BLE001
            if not args.continue_on_error:
                raise
            evaluation = build_failed_evaluation(
                datapoint,
                communication_structure=variant["baseline"],
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
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "report_path": str(report_path) if report_path else None,
                "aggregate": summary["aggregate"],
                "input_diagnostics": input_diagnostics,
            },
            indent=2,
        )
    )


def run_single_agent_rui_episode(
    datapoint: dict[str, Any],
    *,
    client: OpenAIJsonClient,
    recorder: PromptTraceRecorder,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one backward-compatible Single-Agent All-RIU episode."""

    return run_single_agent_episode(
        datapoint,
        client=client,
        recorder=recorder,
        args=args,
        context_variant=CONTEXT_ALL_RIUS,
    )


def run_single_agent_full_context_episode(
    datapoint: dict[str, Any],
    *,
    client: OpenAIJsonClient,
    recorder: PromptTraceRecorder,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one Single-Agent Full-Context episode."""

    return run_single_agent_episode(
        datapoint,
        client=client,
        recorder=recorder,
        args=args,
        context_variant=CONTEXT_FULL_SOURCE,
    )


def run_single_agent_episode(
    datapoint: dict[str, Any],
    *,
    client: OpenAIJsonClient,
    recorder: PromptTraceRecorder,
    args: argparse.Namespace,
    context_variant: str,
) -> dict[str, Any]:
    variant = _context_variant_metadata(context_variant)
    rui_ids = [
        str(row.get("rui_id"))
        for row in datapoint.get("reference_information_units", [])
        if isinstance(row, dict) and row.get("rui_id")
    ]
    prompt_id = f"{datapoint.get('instance_id')}.{variant['baseline']}"
    short_answer_judged = _is_short_answer_judged(datapoint)
    if context_variant == CONTEXT_ALL_RIUS:
        context_text = _budgeted_all_rui_context(datapoint, args.context_token_budget)
        context_payload = {
            "reference_information_unit_count": len(rui_ids),
            "all_rui_context_text": context_text,
        }
        context_grounding = "the reference information units"
        context_instruction = "all reference information units"
        disclosed_rui_ids = rui_ids
    else:
        context_text, context_unit_count, context_token_count = full_source_context_text(
            datapoint,
            context_token_budget=args.context_token_budget,
        )
        context_payload = {
            "source_unit_count": context_unit_count,
            "source_token_count": context_token_count,
            "all_source_context_text": context_text,
        }
        context_grounding = "the source material"
        context_instruction = "all available source material"
        disclosed_rui_ids = []
    system_prompt = "\n".join(
        [
            "You are a single legal decision-maker.",
            variant["system_context_statement"],
            f"Use at most {int(args.final_reasoning_token_budget)} private reasoning tokens before answering.",
            f"Keep the final rationale at or under {int(args.final_rationale_token_budget)} tokens.",
            "Return exactly one JSON object with only the keys requested in response_format.",
            f"Do not repeat the input JSON, task, options, or {variant['context_payload_key']}.",
            "Return JSON only and follow response_format exactly.",
        ]
    )
    response_format = {
        "action": "final_decision",
        "answer": "one decision id from decision_set",
        "rationale": f"short rationale grounded in {context_grounding}",
        "confidence": "number between 0 and 1",
    }
    turn_instruction = f"Choose the final answer using the task and {context_instruction}."
    if short_answer_judged:
        response_format["written_answer"] = (
            f"short written conclusion with the required decision steps grounded in {context_grounding}"
        )
        turn_instruction = (
            f"Write the final short-answer conclusion using the task and {context_instruction}. "
            "Also set answer to a valid compatibility decision id from decision_set."
        )
    user_payload = {
        "task": {
            "task_description": datapoint.get("task_description"),
            "decision_set": datapoint.get("decision_set"),
            "decision_option_definitions": datapoint.get("decision_option_definitions"),
            "initial_common_information": model_common_information(datapoint),
        },
        **context_payload,
        "turn_instruction": turn_instruction,
        "response_format": response_format,
    }
    user_prompt = json.dumps(user_payload, indent=2, ensure_ascii=False)
    response = client.generate_json(
        prompt_id=prompt_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        trace_dir=recorder.trace_dir,
    )
    recorder.record(
        prompt_id=prompt_id,
        actor_id="single_agent",
        actor_role="single_agent",
        prompt_kind="single_agent_final_decision",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response,
        request_meta=dict(client.last_request_meta),
    )
    answer = response.get("answer")
    if answer not in set(datapoint.get("decision_set", [])):
        answer = datapoint.get("decision_set", [""])[0]
    confidence = response.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    final_decision = {
        "action": "final_decision",
        "actor_id": "single_agent",
        "actor_role": "single_agent",
        "answer": answer,
        "rationale": truncate_tokens(str(response.get("rationale") or ""), args.final_rationale_token_budget),
        "written_answer": truncate_tokens(str(response.get("written_answer") or ""), args.final_rationale_token_budget),
        "confidence": confidence,
    }
    message = {
        "turn": 1,
        "speaker_id": "single_agent",
        "speaker_role": "single_agent",
        "content": final_decision["written_answer"] or final_decision["rationale"],
        "target_ids": [],
        "disclosed_rui_ids": disclosed_rui_ids,
    }
    return {
        "schema_version": EPISODE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datapoint": {
            "instance_id": datapoint.get("instance_id"),
            "dataset": datapoint.get("dataset"),
            "scenario_family": datapoint.get("scenario_family"),
            "task_description": datapoint.get("task_description"),
            "decision_set": datapoint.get("decision_set"),
        },
        "communication_structure": {
            "schema_version": "teamgym_single_agent_baseline",
            "name": variant["baseline"],
            "coordinator_present": False,
            "turn_allocation_rule": variant["baseline"],
            "final_decision_rule": "single_agent",
        },
        "budgets": {
            "context_token_budget": args.context_token_budget,
            "final_rationale_token_budget": args.final_rationale_token_budget,
            "final_reasoning_token_budget": args.final_reasoning_token_budget,
        },
        "turns": [
            {
                "turn": 1,
                "turn_type": "single_agent_final_decision",
                "actions": [{"type": "final_decision", "actor_id": "single_agent", "payload": final_decision}],
                "messages": [message],
                "new_disclosed_rui_ids": disclosed_rui_ids,
            }
        ],
        "messages": [message],
        "public_rui_ids": disclosed_rui_ids,
        "final_decision": final_decision,
        "prompt_records": list(recorder.prompt_records),
        "stopped_reason": "single_agent_answered",
    }


def _budgeted_all_rui_context(datapoint: dict[str, Any], context_token_budget: int) -> str:
    segments = [
        (str(unit.get("text") or "").strip(), True)
        for unit in rui_context_units(datapoint)
        if str(unit.get("text") or "").strip()
    ]
    return distributed_truncate_token_segments(segments, context_token_budget)


def full_source_context_units(datapoint: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ordered union of source units assigned to any participant."""

    participant_ids = {
        str(row.get("participant_id"))
        for row in datapoint.get("participants", [])
        if isinstance(row, dict) and row.get("participant_id")
    }
    assigned_unit_ids = {
        str(unit_id)
        for participant in datapoint.get("participants", [])
        if isinstance(participant, dict)
        for unit_id in participant.get("private_context_unit_ids", [])
        if isinstance(unit_id, str)
    }
    output: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    for unit in datapoint.get("private_context_units", []):
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("context_unit_id") or "")
        text = str(unit.get("text") or "").strip()
        visible_by_id = bool(unit_id and unit_id in assigned_unit_ids)
        visible_by_assignment = bool(
            participant_ids.intersection(
                str(item) for item in unit.get("assigned_to", []) if isinstance(item, str)
            )
        )
        if not text or not (visible_by_id or visible_by_assignment):
            continue
        if not unit_id:
            raise ValueError("Assigned private context unit is missing context_unit_id")
        previous_text = seen_ids.get(unit_id)
        if previous_text is not None:
            if previous_text != text:
                raise ValueError(f"Conflicting text for private context unit {unit_id}")
            continue
        seen_ids[unit_id] = text
        output.append(unit)
    if not output:
        raise ValueError("No participant-assigned source units found for full-context baseline")
    return output


def full_source_context_text(
    datapoint: dict[str, Any],
    *,
    context_token_budget: int,
) -> tuple[str, int, int]:
    """Build an untruncated centralized source context and verify its declared budget."""

    units = full_source_context_units(datapoint)
    text = "\n\n".join(str(unit.get("text") or "").strip() for unit in units)
    token_count = count_tokens(text)
    if token_count > int(context_token_budget):
        raise ValueError(
            f"Full source requires {token_count} tokens, exceeding the "
            f"{int(context_token_budget)}-token experiment budget; no truncation was applied"
        )
    return text, len(units), token_count


def _is_short_answer_judged(datapoint: dict[str, Any]) -> bool:
    task = datapoint.get("task_description") if isinstance(datapoint.get("task_description"), dict) else {}
    dataset_specific = datapoint.get("dataset_specific") if isinstance(datapoint.get("dataset_specific"), dict) else {}
    return task.get("answer_type") == "short_answer_judged" or dataset_specific.get("answer_type") == "short_answer_judged"


def _parse_args(context_variant: str) -> argparse.Namespace:
    variant = _context_variant_metadata(context_variant)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_LEGAL_DATASET_PATH))
    parser.add_argument("--output-dir", default=f"outputs/teamgym/evaluation/{variant['baseline']}")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--context-token-budget", type=int, default=20000)
    parser.add_argument("--final-rationale-token-budget", type=int, default=120)
    parser.add_argument("--final-reasoning-token-budget", type=int, default=256)
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
    parser.add_argument("--trace-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-html", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    return args


def _context_variant_metadata(context_variant: str) -> dict[str, str]:
    if context_variant == CONTEXT_ALL_RIUS:
        return {
            "baseline": "single_agent_all_rius",
            "run_slug": "single_agent_rui_eval",
            "manifest_context_variant": "all_reference_information_units_only",
            "system_context_statement": "You can see all reference information units for the task.",
            "context_payload_key": "all_rui_context_text",
        }
    if context_variant == CONTEXT_FULL_SOURCE:
        return {
            "baseline": "single_agent_full_context",
            "run_slug": "single_agent_full_context_eval",
            "manifest_context_variant": "all_participant_source_units_centralized",
            "system_context_statement": (
                "You can see the complete source material available across all participants. "
                "Locate the decision-relevant evidence before answering."
            ),
            "context_payload_key": "all_source_context_text",
        }
    raise ValueError(f"Unknown single-agent context variant: {context_variant}")


if __name__ == "__main__":
    main()
