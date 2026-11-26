"""Episode scoring for the fully collaborative interaction runtime."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from teamgym.scoring.written_answers import is_written_answer_task

from .rui_matching import match_public_rius


EPISODE_EVALUATION_SCHEMA = "teamgym_episode_evaluation"
RUNTIME_FAILURE_SCHEMA = "teamgym_runtime_failure"


def build_primary_evaluation(
    datapoint: dict[str, Any], final_answer: Any = None, *, sufficient_set_covered: bool = False
) -> dict[str, Any]:
    oracle = datapoint.get("oracle") if isinstance(datapoint.get("oracle"), dict) else {}
    oracle_answer = oracle.get("answer")
    matches = final_answer is not None and final_answer == oracle_answer
    metrics = {"final_answer": final_answer, "oracle_answer": oracle_answer}
    if is_written_answer_task(datapoint):
        metrics["decision_label_matches"] = matches
        metrics["decision_label_matches_with_evidence"] = matches and sufficient_set_covered
    else:
        metrics["task_success"] = matches
        metrics["informed_success"] = matches and sufficient_set_covered
    return metrics


def evaluate_episode(episode: dict[str, Any], datapoint: dict[str, Any]) -> dict[str, Any]:
    final_decision = episode.get("final_decision") if isinstance(episode.get("final_decision"), dict) else {}
    final_answer = final_decision.get("answer")
    reference_ids = {
        str(row.get("rui_id"))
        for row in datapoint.get("reference_information_units", [])
        if isinstance(row, dict) and isinstance(row.get("rui_id"), str)
    }
    rui_match_eval = match_public_rius(episode, datapoint)
    public_rui_ids = {
        str(item)
        for item in rui_match_eval.get("matched_rui_ids", [])
        if isinstance(item, str)
    }
    smallest_sets = _smallest_sufficient_sets(datapoint)
    sufficient_set_covered = any(rui_set <= public_rui_ids for rui_set in smallest_sets)
    unsupported = sorted(public_rui_ids - reference_ids)
    participant_ids = [
        str(row["participant_id"])
        for row in datapoint.get("participant_directory", [])
        if isinstance(row, dict) and isinstance(row.get("participant_id"), str)
    ]
    query_target_ids = _query_target_ids(episode)
    messages = [row for row in episode.get("messages", []) if isinstance(row, dict)]
    by_speaker = Counter(str(row.get("speaker_id")) for row in messages)
    by_participant = {participant_id: int(by_speaker.get(participant_id, 0)) for participant_id in participant_ids}
    coordinator_messages = sum(
        count
        for speaker, count in by_speaker.items()
        if speaker not in set(participant_ids)
    )
    participant_total = sum(by_participant.values())
    complete_participant_probe = bool(participant_ids) and all(
        by_participant[participant_id] == 1 for participant_id in participant_ids
    )
    complete_query_schedule = bool(participant_ids) and Counter(
        query_target_ids
    ) == Counter(participant_ids)
    shares = {
        participant_id: (float(count) / participant_total if participant_total else 0.0)
        for participant_id, count in by_participant.items()
    }
    target_share = 1.0 / len(participant_ids) if participant_ids else 0.0
    balance_variance = (
        mean((share - target_share) ** 2 for share in shares.values())
        if shares
        else 0.0
    )
    return {
        "schema_version": EPISODE_EVALUATION_SCHEMA,
        "instance_id": datapoint.get("instance_id"),
        "dataset": datapoint.get("dataset"),
        "communication_structure": episode.get("communication_structure", {}).get("name"),
        "primary_eval": build_primary_evaluation(
            datapoint, final_answer, sufficient_set_covered=sufficient_set_covered
        ),
        "information_flow_eval": {
            "reference_information_unit_count": len(reference_ids),
            "public_reference_information_unit_count": len(public_rui_ids & reference_ids),
            "public_reference_information_unit_recall": _safe_ratio(len(public_rui_ids & reference_ids), len(reference_ids)),
            "sufficient_set_covered": bool(sufficient_set_covered),
            "smallest_sufficient_set_count": len(smallest_sets),
            "unsupported_disclosed_rui_ids": unsupported,
            "unsupported_disclosed_rui_count": len(unsupported),
            "matching_method": rui_match_eval.get("schema_version"),
        },
        "reference_information_matching": rui_match_eval,
        "communication_eval": {
            "turn_count": len(episode.get("turns", [])),
            "message_count": len(messages),
            "message_count_by_participant": by_participant,
            "message_count_by_nonparticipant": int(coordinator_messages),
            "word_count": sum(len(str(row.get("content", "")).split()) for row in messages),
        },
        "participation_eval": {
            "participant_count": len(participant_ids),
            "participant_message_count_total": participant_total,
            "complete_participant_probe": complete_participant_probe,
            "query_target_ids": query_target_ids,
            "complete_query_schedule": complete_query_schedule,
            "participant_message_shares": {key: round(value, 4) for key, value in shares.items()},
            "participant_message_balance_variance": round(balance_variance, 6),
        },
        "final_decision": final_decision,
    }


def build_failed_evaluation(
    datapoint: dict[str, Any],
    *,
    communication_structure: str,
    error: BaseException,
) -> dict[str, Any]:
    reference_ids = {
        str(row.get("rui_id"))
        for row in datapoint.get("reference_information_units", [])
        if isinstance(row, dict) and isinstance(row.get("rui_id"), str)
    }
    participant_ids = [
        str(row["participant_id"])
        for row in datapoint.get("participant_directory", [])
        if isinstance(row, dict) and isinstance(row.get("participant_id"), str)
    ]
    return {
        "schema_version": EPISODE_EVALUATION_SCHEMA,
        "instance_id": datapoint.get("instance_id"),
        "dataset": datapoint.get("dataset"),
        "communication_structure": communication_structure,
        "primary_eval": build_primary_evaluation(datapoint),
        "information_flow_eval": {
            "reference_information_unit_count": len(reference_ids),
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
            "message_count_by_participant": {participant_id: 0 for participant_id in participant_ids},
            "message_count_by_nonparticipant": 0,
            "word_count": 0,
        },
        "participation_eval": {
            "participant_count": len(participant_ids),
            "participant_message_count_total": 0,
            "complete_participant_probe": False,
            "query_target_ids": [],
            "complete_query_schedule": False,
            "participant_message_shares": {participant_id: 0.0 for participant_id in participant_ids},
            "participant_message_balance_variance": 0.0,
        },
        "final_decision": None,
        "runtime_error": {
            "error_type": type(error).__name__,
            "error_message": str(error)[:500],
        },
    }


def aggregate_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "teamgym_episode_evaluation_aggregate",
        "episode_count": len(evaluations),
        "mean_task_success": _mean_path(evaluations, ["primary_eval", "task_success"]),
        "mean_informed_success": _mean_path(evaluations, ["primary_eval", "informed_success"]),
        "mean_decision_label_matches": _mean_path(evaluations, ["primary_eval", "decision_label_matches"]),
        "mean_decision_label_matches_with_evidence": _mean_path(
            evaluations, ["primary_eval", "decision_label_matches_with_evidence"]
        ),
        "mean_public_reference_information_unit_recall": _mean_path(
            evaluations,
            ["information_flow_eval", "public_reference_information_unit_recall"],
        ),
        "mean_turn_count": _mean_path(evaluations, ["communication_eval", "turn_count"]),
        "mean_message_count": _mean_path(evaluations, ["communication_eval", "message_count"]),
        "mean_complete_participant_probe": _mean_path(
            evaluations, ["participation_eval", "complete_participant_probe"]
        ),
        "mean_complete_query_schedule": _mean_path(
            evaluations, ["participation_eval", "complete_query_schedule"]
        ),
        "by_dataset": _aggregate_by_key(evaluations, "dataset"),
        "by_communication_structure": _aggregate_by_key(evaluations, "communication_structure"),
    }


def _aggregate_by_key(evaluations: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in evaluations:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return {
        name: {
            "episode_count": len(rows),
            "mean_task_success": _mean_path(rows, ["primary_eval", "task_success"]),
            "mean_informed_success": _mean_path(rows, ["primary_eval", "informed_success"]),
            "mean_decision_label_matches": _mean_path(rows, ["primary_eval", "decision_label_matches"]),
            "mean_decision_label_matches_with_evidence": _mean_path(
                rows, ["primary_eval", "decision_label_matches_with_evidence"]
            ),
            "mean_public_reference_information_unit_recall": _mean_path(
                rows,
                ["information_flow_eval", "public_reference_information_unit_recall"],
            ),
            "mean_complete_participant_probe": _mean_path(
                rows, ["participation_eval", "complete_participant_probe"]
            ),
            "mean_complete_query_schedule": _mean_path(
                rows, ["participation_eval", "complete_query_schedule"]
            ),
        }
        for name, rows in sorted(groups.items())
    }


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


def _query_target_ids(episode: dict[str, Any]) -> list[str]:
    """Return targets selected by coordinator query actions in execution order."""
    target_ids: list[str] = []
    for turn in episode.get("turns", []):
        if not isinstance(turn, dict):
            continue
        for action in turn.get("actions", []):
            if not isinstance(action, dict) or action.get("type") != "query":
                continue
            target_ids.extend(
                str(item)
                for item in action.get("target_ids", [])
                if isinstance(item, str)
            )
    return target_ids


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _mean_path(rows: list[dict[str, Any]], path: list[str]) -> float | None:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return round(mean(values), 6)
