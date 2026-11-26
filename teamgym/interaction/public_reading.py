"""Participant-specific bounded observation of the public transcript."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import random
from typing import Any

from .actor_utils import count_tokens, slice_tokens


class ParticipantPublicReader:
    """Track public text observed by each participant across activations."""

    def __init__(self) -> None:
        self._read_intervals: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(dict)
        self._activation_counts: dict[str, int] = defaultdict(int)
        self._free_message_fingerprints: dict[str, set[tuple[Any, ...]]] = defaultdict(set)

    def observe(
        self,
        *,
        datapoint_id: str,
        participant_id: str,
        messages: list[dict[str, Any]],
        request: dict[str, Any] | None,
        policy: str,
        token_budget: int,
        seed: int,
        charge_request: bool = False,
        prioritize_latest_other: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if policy == "unlimited":
            history = [_message_payload(row) for row in messages if isinstance(row, dict)]
            if isinstance(request, dict):
                history.append(_request_payload(request))
            return history, {
                "public_read_policy": "unlimited",
                "public_read_token_budget": None,
                "unread_public_tokens_available": 0,
                "new_public_tokens_read": 0,
                "cumulative_public_tokens_read": 0,
            }

        budget = max(1, int(token_budget))
        intervals = self._read_intervals[participant_id]
        free_fingerprints = self._free_message_fingerprints[participant_id]
        effective_messages = [row for row in messages if isinstance(row, dict)]
        request_index: int | None = None

        # A current directed request is the activation trigger and is visible
        # without consuming the participant's peer-reading budget unless the
        # strict all-actor accounting mode explicitly charges it.
        if isinstance(request, dict) and charge_request:
            request_index = len(effective_messages)
            effective_messages.append(_request_payload(request))
        elif isinstance(request, dict):
            free_fingerprints.add(_message_fingerprint(request))

        candidates: list[tuple[int, int, int]] = []
        token_counts: dict[int, int] = {}
        free_message_indices: set[int] = set()
        for index, message in enumerate(effective_messages):
            content = str(message.get("content") or "")
            token_count = count_tokens(content)
            token_counts[index] = token_count
            if str(message.get("speaker_id") or "") == participant_id:
                continue
            if _message_fingerprint(message) in free_fingerprints:
                intervals[index] = [(0, token_count)] if token_count else []
                free_message_indices.add(index)
                continue
            for start, end in _complement(intervals.get(index, []), token_count):
                candidates.append((index, start, end))

        available = sum(end - start for _, start, end in candidates)
        priority_index = request_index
        if priority_index is None and prioritize_latest_other:
            priority_index = next(
                (
                    index
                    for index in range(len(effective_messages) - 1, -1, -1)
                    if str(effective_messages[index].get("speaker_id") or "") != participant_id
                ),
                None,
            )
        selected = _select_priority_then_backlog(
            candidates,
            budget=budget,
            policy=policy,
            seed=_stable_seed(
                seed,
                datapoint_id,
                participant_id,
                self._activation_counts[participant_id],
            ),
            priority_index=priority_index,
        )
        self._activation_counts[participant_id] += 1
        for index, start, end in selected:
            intervals[index] = _merge([*intervals.get(index, []), (start, end)])

        history: list[dict[str, Any]] = []
        for index, message in enumerate(effective_messages):
            own_message = str(message.get("speaker_id") or "") == participant_id
            visible_intervals = [(0, token_counts.get(index, 0))] if own_message else intervals.get(index, [])
            if not visible_intervals:
                continue
            row = _message_payload(message)
            row["content"] = _visible_content(
                str(message.get("content") or ""),
                visible_intervals,
                token_counts.get(index, 0),
            )
            if visible_intervals != [(0, token_counts.get(index, 0))]:
                row["partially_observed"] = True
            history.append(row)
        if isinstance(request, dict) and not charge_request:
            history.append(_request_payload(request))

        cumulative = sum(
            end - start
            for index, index_intervals in intervals.items()
            if index not in free_message_indices
            for start, end in index_intervals
        )
        return history, {
            "public_read_policy": policy,
            "public_read_token_budget": budget,
            "unread_public_tokens_available": available,
            "new_public_tokens_read": sum(end - start for _, start, end in selected),
            "priority_public_tokens_read": sum(
                end - start for index, start, end in selected if index == priority_index
            ),
            "cumulative_public_tokens_read": cumulative,
        }


def _select_priority_then_backlog(
    candidates: list[tuple[int, int, int]],
    *,
    budget: int,
    policy: str,
    seed: int,
    priority_index: int | None,
) -> list[tuple[int, int, int]]:
    if priority_index is None:
        return _select_intervals(candidates, budget=budget, policy=policy, seed=seed)

    priority = [row for row in candidates if row[0] == priority_index]
    backlog = [row for row in candidates if row[0] != priority_index]
    selected = _select_intervals(priority, budget=budget, policy="fifo", seed=seed)
    remaining = max(0, int(budget) - sum(end - start for _, start, end in selected))
    if remaining:
        selected.extend(_select_intervals(backlog, budget=remaining, policy=policy, seed=seed))
    return selected


def _select_intervals(
    candidates: list[tuple[int, int, int]],
    *,
    budget: int,
    policy: str,
    seed: int,
) -> list[tuple[int, int, int]]:
    total = sum(end - start for _, start, end in candidates)
    if total <= 0:
        return []
    window = min(total, max(1, int(budget)))
    if policy == "fifo":
        offset = 0
    elif policy == "stochastic_contiguous":
        offset = random.Random(seed).randrange(total - window + 1)
    else:
        raise ValueError(f"Unknown participant public-read policy: {policy}")

    selected: list[tuple[int, int, int]] = []
    remaining_offset = offset
    remaining_window = window
    for index, start, end in candidates:
        length = end - start
        if remaining_offset >= length:
            remaining_offset -= length
            continue
        selected_start = start + remaining_offset
        selected_end = min(end, selected_start + remaining_window)
        selected.append((index, selected_start, selected_end))
        remaining_window -= selected_end - selected_start
        remaining_offset = 0
        if remaining_window <= 0:
            break
    return selected


def _visible_content(text: str, intervals: list[tuple[int, int]], total: int) -> str:
    merged = _merge(intervals)
    pieces: list[str] = []
    if merged and merged[0][0] > 0:
        pieces.append("[...]")
    for position, (start, end) in enumerate(merged):
        if position > 0 and start > merged[position - 1][1]:
            pieces.append("[...]")
        pieces.append(slice_tokens(text, start, end).strip())
    if merged and merged[-1][1] < total:
        pieces.append("[...]")
    return "\n".join(piece for piece in pieces if piece)


def _complement(intervals: list[tuple[int, int]], total: int) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    cursor = 0
    for start, end in _merge(intervals):
        start = max(0, min(total, start))
        end = max(start, min(total, end))
        if cursor < start:
            output.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total:
        output.append((cursor, total))
    return output


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for start, end in sorted((int(start), int(end)) for start, end in intervals if end > start):
        if not output or start > output[-1][1]:
            output.append((start, end))
        else:
            output[-1] = (output[-1][0], max(output[-1][1], end))
    return output


def _message_payload(message: dict[str, Any]) -> dict[str, Any]:
    row = {
        "turn": message.get("turn"),
        "speaker_id": message.get("speaker_id"),
        "speaker_role": message.get("speaker_role"),
        "content": message.get("content"),
    }
    if message.get("role_name") is not None:
        row["role_name"] = message.get("role_name")
    return row


def _request_payload(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn": request.get("turn"),
        "speaker_id": request.get("speaker_id"),
        "speaker_role": request.get("speaker_role"),
        "content": request.get("content"),
        "target_ids": request.get("target_ids"),
    }


def _message_fingerprint(message: dict[str, Any]) -> tuple[Any, ...]:
    return (
        message.get("speaker_id"),
        message.get("speaker_role"),
        str(message.get("content") or ""),
        tuple(message.get("target_ids") or []),
    )


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
