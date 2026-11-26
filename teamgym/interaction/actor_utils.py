"""Shared actor helpers used by the interaction runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def participant_ids(datapoint: dict[str, Any]) -> list[str]:
    return [
        str(row["participant_id"])
        for row in datapoint.get("participant_directory", [])
        if isinstance(row, dict) and isinstance(row.get("participant_id"), str)
    ]


def participant_by_id(datapoint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["participant_id"]): row
        for row in datapoint.get("participants", [])
        if isinstance(row, dict) and isinstance(row.get("participant_id"), str)
    }


def participant_private_context_units(
    datapoint: dict[str, Any], participant: dict[str, Any]
) -> list[dict[str, Any]]:
    participant_id = participant["participant_id"]
    allowed_ids = set(participant.get("private_context_unit_ids", []))
    units: dict[str, dict[str, Any]] = {}
    for unit in datapoint.get("private_context_units", []):
        if not isinstance(unit, dict) or unit.get("context_unit_id") not in allowed_ids:
            continue
        unit_id = unit["context_unit_id"]
        if unit_id in units:
            raise ValueError(f"Duplicate private context unit {unit_id!r}")
        assigned_to = unit.get("assigned_to")
        if not isinstance(assigned_to, list) or participant_id not in assigned_to:
            raise ValueError(f"Private context unit {unit_id!r} is not assigned to {participant_id!r}")
        units[unit_id] = unit
    missing = allowed_ids - units.keys()
    if missing:
        raise ValueError(f"Missing private context units for {participant_id!r}: {sorted(missing)}")
    return list(units.values())


def truncate_tokens(text: str, max_tokens: int) -> str:
    text = str(text)
    max_tokens = max(1, int(max_tokens))
    encode, decode = _token_codec()
    tokens = encode(text)
    if len(tokens) <= max_tokens:
        return text
    return decode(tokens[:max_tokens]).rstrip() + " ..."


def count_tokens(text: str) -> int:
    encode, _ = _token_codec()
    return len(encode(str(text)))


def slice_tokens(text: str, start: int, end: int) -> str:
    """Decode a half-open token interval from text."""

    encode, decode = _token_codec()
    tokens = encode(str(text))
    lower = max(0, min(len(tokens), int(start)))
    upper = max(lower, min(len(tokens), int(end)))
    return decode(tokens[lower:upper])


def truncate_tokens_strict(text: str, max_tokens: int) -> str:
    """Truncate without an appended marker so the result cannot exceed the cap."""

    text = str(text)
    max_tokens = max(1, int(max_tokens))
    encode, decode = _token_codec()
    tokens = encode(text)
    return text if len(tokens) <= max_tokens else decode(tokens[:max_tokens]).rstrip()


def distributed_truncate_tokens(text: str, max_tokens: int) -> str:
    text = str(text)
    max_tokens = max(1, int(max_tokens))
    encode, decode = _token_codec()
    tokens = encode(text)
    if len(tokens) <= max_tokens:
        return text

    marker = "\n\n[...]\n\n"
    marker_tokens = encode(marker)
    window_count = min(32, max(3, max_tokens // 256))
    while window_count > 1 and len(marker_tokens) * (window_count - 1) >= max_tokens:
        window_count -= 1
    if window_count <= 1:
        return decode(tokens[:max_tokens]).rstrip() + " ..."

    usable_tokens = max_tokens - len(marker_tokens) * (window_count - 1)
    base_window_size = max(1, usable_tokens // window_count)
    extra = usable_tokens % window_count
    max_start = max(0, len(tokens) - base_window_size)
    intervals: list[tuple[int, int]] = []
    for index in range(window_count):
        window_size = base_window_size + (1 if index < extra else 0)
        if window_count == 1:
            start = 0
        else:
            start = round(index * max_start / (window_count - 1))
        intervals.append((start, min(len(tokens), start + window_size)))

    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))

    pieces = [decode(tokens[start:end]).strip() for start, end in merged if end > start]
    return marker.join(piece for piece in pieces if piece).strip()


def distributed_truncate_token_segments(
    segments: list[tuple[str, bool]],
    max_tokens: int,
) -> str:
    max_tokens = max(1, int(max_tokens))
    encode, decode = _token_codec()
    separator_tokens = encode("\n\n")
    tokens: list[Any] = []
    protected_intervals: list[tuple[int, int]] = []
    for index, (text, protected) in enumerate(segments):
        text = str(text).strip()
        if not text:
            continue
        if index > 0 and tokens:
            tokens.extend(separator_tokens)
        start = len(tokens)
        tokens.extend(encode(text))
        end = len(tokens)
        if protected and end > start:
            protected_intervals.append((start, end))

    if len(tokens) <= max_tokens:
        return decode(tokens)
    if not protected_intervals:
        return distributed_truncate_tokens(decode(tokens), max_tokens)

    protected_intervals = _merge_intervals(protected_intervals)
    protected_token_count = sum(end - start for start, end in protected_intervals)
    if protected_token_count >= max_tokens:
        return _decode_intervals(tokens, protected_intervals, decode)

    remaining = max_tokens - protected_token_count
    distributed_intervals = _distributed_intervals(len(tokens), remaining)
    return _decode_intervals(tokens, _merge_intervals([*protected_intervals, *distributed_intervals]), decode)


def _distributed_intervals(token_count: int, max_tokens: int) -> list[tuple[int, int]]:
    max_tokens = max(1, int(max_tokens))
    if token_count <= max_tokens:
        return [(0, token_count)]
    window_count = min(32, max(3, max_tokens // 256))
    if window_count <= 1:
        return [(0, max_tokens)]
    base_window_size = max(1, max_tokens // window_count)
    extra = max_tokens % window_count
    max_start = max(0, token_count - base_window_size)
    intervals: list[tuple[int, int]] = []
    for index in range(window_count):
        window_size = base_window_size + (1 if index < extra else 0)
        start = round(index * max_start / (window_count - 1))
        intervals.append((start, min(token_count, start + window_size)))
    return intervals


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _decode_intervals(
    tokens: list[Any],
    intervals: list[tuple[int, int]],
    decode: Callable[[list[Any]], str],
) -> str:
    marker = "\n\n[...]\n\n"
    pieces = [decode(tokens[start:end]).strip() for start, end in intervals if end > start]
    return marker.join(piece for piece in pieces if piece).strip()


def _token_codec() -> tuple[Callable[[str], list[Any]], Callable[[list[Any]], str]]:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.encode, encoding.decode
    except Exception:  # noqa: BLE001
        return _fallback_encode, _fallback_decode


def _fallback_encode(text: str) -> list[str]:
    return text.split()


def _fallback_decode(tokens: list[Any]) -> str:
    return " ".join(str(token) for token in tokens)
