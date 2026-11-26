"""Run a participant-private solo baseline for normalized legal datapoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from typing import Any

from teamgym.core.io import write_json
from teamgym.interaction.actor_utils import (
    distributed_truncate_token_segments,
    participant_by_id,
    participant_ids,
    participant_private_context_units,
    truncate_tokens,
)
from teamgym.core.dataset import datapoint_variant
from teamgym.interaction.environment import EPISODE_SCHEMA
from teamgym.interaction.evaluation import (
    EPISODE_EVALUATION_SCHEMA,
    RUNTIME_FAILURE_SCHEMA,
    aggregate_evaluations,
    build_primary_evaluation,
    evaluate_episode,
)
from teamgym.interaction.evaluation_client import evaluation_model_manifest
from teamgym.interaction.evaluation_client import make_evaluation_client as _make_client
from teamgym.interaction.llm_actors import PromptTraceRecorder
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
from teamgym.scoring.written_answers import is_written_answer_task


COMMUNICATION_STRUCTURE = "single_participant_private_context"


def main() -> None:
    args = _parse_args()
    run_single_participant_evaluation(args)


def run_single_participant_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    datapoints, input_diagnostics = load_evaluation_datapoints(args)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser() / _run_output_dir_name(
        stamp,
        "single_participant_eval",
        args.num_shards,
        args.shard_index,
    )
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": "teamgym_single_participant_eval_run",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": args.input_root,
        "output_dir": str(output_dir),
        "datapoint_count": len(datapoints),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "baseline": COMMUNICATION_STRUCTURE,
        "baseline_note": (
            "Each participant answers alone using only common task information and that "
            "participant's private source material. The oracle-max metric is computed "
            "afterward as an upper bound over participants per instance."
        ),
        "llm": evaluation_model_manifest(args),
        "input_diagnostics": input_diagnostics,
        "context_token_budget": args.context_token_budget,
        "final_rationale_token_budget": args.final_rationale_token_budget,
        "final_reasoning_token_budget": args.final_reasoning_token_budget,
        "participant_role_name_mode": args.participant_role_name_mode,
        "prompt_tracing_enabled": bool(args.trace_prompts),
    }
    write_json(output_dir / "run_manifest.json", manifest)

    client = _make_client(args, output_dir)
    evaluations: list[dict[str, Any]] = []
    for datapoint_index, datapoint in enumerate(datapoints, start=1):
        for participant_index, participant_id in enumerate(participant_ids(datapoint)):
            slug = _slug(
                f"{datapoint_index:04d}_{datapoint.get('dataset')}_{datapoint.get('instance_id')}_{participant_id}"
            )
            recorder = PromptTraceRecorder(trajectories_dir / f"{slug}.prompts" if args.trace_prompts else None)
            try:
                episode = run_single_participant_episode(
                    datapoint,
                    participant_id=participant_id,
                    participant_index=participant_index,
                    client=client,
                    recorder=recorder,
                    args=args,
                )
                evaluation = evaluate_episode(episode, datapoint)
                _attach_single_participant_eval(
                    evaluation,
                    datapoint=datapoint,
                    participant_id=participant_id,
                    participant_index=participant_index,
                    raw_answer=episode.get("final_decision", {}).get("raw_answer"),
                    valid_answer=episode.get("final_decision", {}).get("answer") is not None,
                )
                trajectory_payload = {"episode": episode, "evaluation": evaluation}
            except Exception as exc:  # noqa: BLE001
                if not args.continue_on_error:
                    raise
                evaluation = _failed_single_participant_evaluation(
                    datapoint,
                    participant_id=participant_id,
                    participant_index=participant_index,
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
        "schema_version": "teamgym_single_participant_eval_summary",
        "input_diagnostics": input_diagnostics,
        "aggregate": aggregate_evaluations(evaluations),
        "single_participant_aggregate": aggregate_single_participant_evaluations(evaluations),
        "results": evaluations,
    }
    write_json(output_dir / "eval_summary.json", summary)
    result = {
        "output_dir": str(output_dir),
        "aggregate": summary["aggregate"],
        "input_diagnostics": input_diagnostics,
        "single_participant_aggregate": summary["single_participant_aggregate"],
    }
    print(json.dumps(result, indent=2))
    return result


def run_single_participant_episode(
    datapoint: dict[str, Any],
    *,
    participant_id: str,
    participant_index: int,
    client: OpenAIJsonClient,
    recorder: PromptTraceRecorder,
    args: argparse.Namespace,
) -> dict[str, Any]:
    participants = participant_by_id(datapoint)
    participant = participants[participant_id]
    role_name = _visible_role_name(participant, participant_id, args.participant_role_name_mode)
    owned_rui_ids = _owned_rui_ids(participant)
    short_answer_judged = is_written_answer_task(datapoint)
    prompt_id = f"{datapoint.get('instance_id')}.{participant_id}.single_participant_private_context"
    system_lines = [
        "You are one participant answering a legal decision task.",
        "You can see only the common task information and your own private source material.",
        "You are answering alone; there is no group discussion and no coordinator.",
        "Do not assume hidden source material you cannot see.",
        f"Use at most {int(args.final_reasoning_token_budget)} private reasoning tokens before answering.",
        f"Keep the rationale at or under {int(args.final_rationale_token_budget)} tokens.",
        "Return exactly one JSON object with only the keys requested in response_format.",
        "Do not repeat the input JSON, task, options, or private_source_material.",
        "Return JSON only and follow response_format exactly.",
    ]
    if short_answer_judged:
        system_lines.insert(3, "Write the short-answer conclusion that your available information best supports.")
        system_lines.insert(4, "If your information is incomplete, say what conclusion is supported and what is missing.")
    else:
        system_lines.insert(3, "Choose the final answer that your available information best supports.")
        system_lines.insert(4, "If your private source material is incomplete, choose the best-supported answer from the visible information.")
    system_prompt = "\n".join(system_lines)
    response_format = {
        "action": "participant_final_decision",
        "answer": "one decision id from decision_set",
        "rationale": "short rationale grounded only in visible information",
        "confidence": "number between 0 and 1",
    }
    turn_instruction = "Choose the final answer using only the task and private_source_material."
    if short_answer_judged:
        response_format["written_answer"] = (
            "short written conclusion with the decision steps visible from this participant's private source material"
        )
        turn_instruction = (
            "Write the final short-answer conclusion using only the task and private_source_material. "
            "Also set answer to a valid compatibility decision id from decision_set."
        )
    user_payload = {
        "task": _public_task_payload(datapoint),
        "participant": _participant_identity_payload(participant_id, role_name),
        "private_source_material": _private_context_payload(datapoint, participant, args.context_token_budget),
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
        actor_id=participant_id,
        actor_role="participant",
        prompt_kind="single_participant_final_decision",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response,
        request_meta=dict(client.last_request_meta),
    )
    raw_answer = response.get("answer")
    answer = _coerce_decision_id(raw_answer, datapoint.get("decision_set", []))
    confidence = response.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    final_decision = {
        "action": "participant_final_decision",
        "actor_id": participant_id,
        "actor_role": "participant",
        "answer": answer,
        "raw_answer": raw_answer,
        "rationale": truncate_tokens(str(response.get("rationale") or ""), args.final_rationale_token_budget),
        "written_answer": truncate_tokens(str(response.get("written_answer") or ""), args.final_rationale_token_budget),
        "confidence": confidence,
    }
    if role_name is not None:
        final_decision["role_name"] = role_name
    message = {
        "turn": 1,
        "speaker_id": participant_id,
        "speaker_role": "participant",
        "content": final_decision["written_answer"] or final_decision["rationale"],
        "target_ids": [],
        "disclosed_rui_ids": owned_rui_ids,
    }
    if role_name is not None:
        message["role_name"] = role_name
    return {
        "schema_version": EPISODE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datapoint": {
            "instance_id": datapoint.get("instance_id"),
            "dataset": datapoint.get("dataset"),
            **datapoint_variant(datapoint),
            "scenario_family": datapoint.get("scenario_family"),
            "task_description": datapoint.get("task_description"),
            "decision_set": datapoint.get("decision_set"),
        },
        "communication_structure": {
            "schema_version": "teamgym_single_participant_baseline",
            "name": COMMUNICATION_STRUCTURE,
            "coordinator_present": False,
            "turn_allocation_rule": "one_private_participant_answers_alone",
            "final_decision_rule": "participant",
        },
        "participant_baseline": {
            "participant_id": participant_id,
            "participant_index": int(participant_index),
            "participant_owned_rui_ids": owned_rui_ids,
            "private_context_unit_ids": list(participant.get("private_context_unit_ids", [])),
            "private_word_count": participant.get("private_word_count"),
        },
        "budgets": {
            "context_token_budget": args.context_token_budget,
            "final_rationale_token_budget": args.final_rationale_token_budget,
            "final_reasoning_token_budget": args.final_reasoning_token_budget,
        },
        "turns": [
            {
                "turn": 1,
                "turn_type": "single_participant_final_decision",
                "actions": [{"type": "final_decision", "actor_id": participant_id, "payload": final_decision}],
                "messages": [message],
                "new_disclosed_rui_ids": owned_rui_ids,
            }
        ],
        "messages": [message],
        "public_rui_ids": owned_rui_ids,
        "final_decision": final_decision,
        "prompt_records": list(recorder.prompt_records),
        "stopped_reason": "single_participant_answered",
    }


def aggregate_single_participant_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    by_instance: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluations:
        key = (
            str(row.get("dataset") or ""),
            str(row.get("instance_id") or ""),
            str(row.get("evidence_condition") or ""),
            str(row.get("distribution_key") or ""),
        )
        by_instance[key].append(row)

    oracle_success_count = 0
    finite_choice_instance_count = 0
    all_failed_count = 0
    for rows in by_instance.values():
        successes = _finite_choice_successes(rows)
        finite_choice_instance_count += bool(successes)
        if any(successes):
            oracle_success_count += 1
        if rows and all(_has_runtime_error(row) for row in rows):
            all_failed_count += 1

    per_participant_id = _participant_group_summary(evaluations, key="participant_id")
    per_participant_index = _participant_group_summary(evaluations, key="participant_index")
    attempt_successes = _finite_choice_successes(evaluations)
    valid_answers = [
        bool((row.get("single_participant_eval") or {}).get("valid_answer"))
        for row in evaluations
    ]
    instance_count = len(by_instance)
    return {
        "schema_version": "teamgym_single_participant_aggregate",
        "instance_count": instance_count,
        "attempt_count": len(evaluations),
        "attempt_success_count": int(sum(attempt_successes)),
        "attempt_success_rate": _safe_ratio(sum(attempt_successes), len(attempt_successes)),
        "oracle_max_instance_success_count": int(oracle_success_count),
        "oracle_max_instance_success_rate": _safe_ratio(oracle_success_count, finite_choice_instance_count),
        "runtime_failure_count": sum(1 for row in evaluations if _has_runtime_error(row)),
        "invalid_answer_count": sum(
            1 for row in evaluations if not _has_runtime_error(row) and not bool((row.get("single_participant_eval") or {}).get("valid_answer"))
        ),
        "valid_answer_count": int(sum(valid_answers)),
        "valid_answer_rate": _safe_ratio(sum(valid_answers), len(valid_answers)),
        "all_participants_failed_instance_count": all_failed_count,
        "per_participant_id": per_participant_id,
        "per_participant_index": per_participant_index,
    }


def _attach_single_participant_eval(
    evaluation: dict[str, Any],
    *,
    datapoint: dict[str, Any],
    participant_id: str,
    participant_index: int,
    raw_answer: Any,
    valid_answer: bool,
) -> None:
    participant = participant_by_id(datapoint).get(participant_id, {})
    owned_rui_ids = _owned_rui_ids(participant)
    evaluation["single_participant_eval"] = {
        "participant_id": participant_id,
        "participant_index": int(participant_index),
        "participant_role_name": participant.get("role_name"),
        "participant_owned_rui_ids": owned_rui_ids,
        "participant_owned_rui_count": len(owned_rui_ids),
        "private_context_unit_count": len(participant.get("private_context_unit_ids", []) or []),
        "private_word_count": participant.get("private_word_count"),
        "private_sufficient_set_covered": _sufficient_set_covered(datapoint, set(owned_rui_ids)),
        "raw_answer": raw_answer,
        "valid_answer": bool(valid_answer),
        "oracle_max_note": "Use max over participants for the same task and distribution as a post-hoc upper bound.",
    }


def _failed_single_participant_evaluation(
    datapoint: dict[str, Any],
    *,
    participant_id: str,
    participant_index: int,
    error: BaseException,
) -> dict[str, Any]:
    participant = participant_by_id(datapoint).get(participant_id, {})
    owned_rui_ids = _owned_rui_ids(participant)
    row = {
        "schema_version": EPISODE_EVALUATION_SCHEMA,
        "instance_id": datapoint.get("instance_id"),
        "dataset": datapoint.get("dataset"),
        **datapoint_variant(datapoint),
        "communication_structure": COMMUNICATION_STRUCTURE,
        "primary_eval": build_primary_evaluation(datapoint),
        "information_flow_eval": {
            "reference_information_unit_count": len(datapoint.get("reference_information_units", []) or []),
            "public_reference_information_unit_count": 0,
            "public_reference_information_unit_recall": 0.0,
            "sufficient_set_covered": False,
            "smallest_sufficient_set_count": len(_smallest_sufficient_sets(datapoint)),
            "unsupported_disclosed_rui_ids": [],
            "unsupported_disclosed_rui_count": 0,
            "matching_method": None,
        },
        "reference_information_matching": {
            "schema_version": RUNTIME_FAILURE_SCHEMA,
            "matched_rui_ids": [],
        },
        "communication_eval": {
            "turn_count": 0,
            "message_count": 0,
            "message_count_by_participant": {pid: 0 for pid in participant_ids(datapoint)},
            "message_count_by_nonparticipant": 0,
            "word_count": 0,
        },
        "participation_eval": {
            "participant_count": len(participant_ids(datapoint)),
            "participant_message_count_total": 0,
            "participant_message_shares": {pid: 0.0 for pid in participant_ids(datapoint)},
            "participant_message_balance_variance": 0.0,
        },
        "final_decision": None,
        "single_participant_eval": {
            "participant_id": participant_id,
            "participant_index": int(participant_index),
            "participant_role_name": participant.get("role_name"),
            "participant_owned_rui_ids": owned_rui_ids,
            "participant_owned_rui_count": len(owned_rui_ids),
            "private_context_unit_count": len(participant.get("private_context_unit_ids", []) or []),
            "private_word_count": participant.get("private_word_count"),
            "private_sufficient_set_covered": _sufficient_set_covered(datapoint, set(owned_rui_ids)),
            "raw_answer": None,
            "valid_answer": False,
        },
        "runtime_error": {
            "error_type": type(error).__name__,
            "error_message": str(error)[:500],
        },
    }
    return row


def _private_context_payload(datapoint: dict[str, Any], participant: dict[str, Any], context_token_budget: int) -> str:
    units = participant_private_context_units(datapoint, participant)
    segments = [
        (str(unit.get("text") or "").strip(), bool(unit.get("hidden_contains_rui_ids")))
        for unit in units
        if str(unit.get("text") or "").strip()
    ]
    return distributed_truncate_token_segments(segments, context_token_budget)


def _public_task_payload(datapoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_description": datapoint.get("task_description"),
        "decision_set": datapoint.get("decision_set"),
        "decision_option_definitions": datapoint.get("decision_option_definitions"),
        "initial_common_information": _strip_dataset_metadata(datapoint.get("initial_common_information")),
    }


def _strip_dataset_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_dataset_metadata(item)
            for key, item in value.items()
            if "dataset" not in str(key).casefold()
        }
    if isinstance(value, list):
        return [_strip_dataset_metadata(item) for item in value]
    return value


def _participant_identity_payload(participant_id: str, role_name: str | None) -> dict[str, str]:
    payload = {"participant_id": participant_id}
    if role_name is not None:
        payload["role_name"] = role_name
    return payload


def _visible_role_name(participant: dict[str, Any], participant_id: str, role_name_mode: str) -> str | None:
    if role_name_mode == "hidden":
        return None
    if role_name_mode == "participant_id":
        return participant_id
    if role_name_mode == "generic":
        return "participant"
    return str(participant.get("role_name") or participant.get("public_role") or participant_id)


def _coerce_decision_id(value: Any, decision_set: Any) -> str | None:
    valid = [str(item) for item in decision_set if isinstance(item, str)]
    valid_set = set(valid)
    if isinstance(value, str):
        stripped = value.strip().strip("\"'` ")
        if stripped in valid_set:
            return stripped
        normalized = stripped.casefold()
        for option in valid:
            if normalized == option.casefold():
                return option
        for option in valid:
            if option.casefold() in normalized:
                return option
    return None


def _owned_rui_ids(participant: dict[str, Any]) -> list[str]:
    return [str(item) for item in participant.get("owned_rui_ids", []) if isinstance(item, str)]


def _smallest_sufficient_sets(datapoint: dict[str, Any]) -> list[set[str]]:
    oracle = datapoint.get("oracle") if isinstance(datapoint.get("oracle"), dict) else {}
    rows = oracle.get("smallest_sufficient_rui_sets") if isinstance(oracle.get("smallest_sufficient_rui_sets"), list) else []
    out: list[set[str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rui_ids = {str(item) for item in row.get("rui_ids", []) if isinstance(item, str)}
        if rui_ids:
            out.append(rui_ids)
    return out


def _sufficient_set_covered(datapoint: dict[str, Any], rui_ids: set[str]) -> bool:
    return any(rui_set <= rui_ids for rui_set in _smallest_sufficient_sets(datapoint))


def _participant_group_summary(evaluations: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluations:
        single = row.get("single_participant_eval") if isinstance(row.get("single_participant_eval"), dict) else {}
        value = single.get(key)
        if value is None:
            continue
        groups[str(value)].append(row)
    result = {}
    for name, rows in sorted(groups.items()):
        successes = _finite_choice_successes(rows)
        result[name] = {
            "attempt_count": len(rows),
            "success_count": int(sum(successes)),
            "success_rate": _safe_ratio(sum(successes), len(successes)),
            "runtime_failure_count": sum(1 for row in rows if _has_runtime_error(row)),
            "valid_answer_rate": _safe_ratio(
                sum(bool((row.get("single_participant_eval") or {}).get("valid_answer")) for row in rows),
                len(rows),
            ),
            "mean_private_rui_count": round(
                mean(
                    float((row.get("single_participant_eval") or {}).get("participant_owned_rui_count") or 0)
                    for row in rows
                ),
                4,
            )
            if rows
            else None,
        }
    return result


def _finite_choice_successes(rows: list[dict[str, Any]]) -> list[bool]:
    successes = []
    for row in rows:
        primary = row.get("primary_eval")
        if isinstance(primary, dict) and isinstance(primary.get("task_success"), bool):
            successes.append(primary["task_success"])
    return successes


def _has_runtime_error(row: dict[str, Any]) -> bool:
    return isinstance(row.get("runtime_error"), dict)


def _safe_ratio(numerator: float, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="TeamGym JSONL file or directory.")
    parser.add_argument("--output-dir", default="generated/single_participant_eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--context-token-budget", type=int, default=20000)
    parser.add_argument("--final-rationale-token-budget", type=int, default=120)
    parser.add_argument("--final-reasoning-token-budget", type=int, default=256)
    parser.add_argument(
        "--participant-role-name-mode",
        choices=["hidden", "datapoint", "participant_id", "generic"],
        default="hidden",
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
        help="Record a failed participant row and continue when a datapoint/model call fails.",
    )
    parser.add_argument("--trace-prompts", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    return args


if __name__ == "__main__":
    main()
