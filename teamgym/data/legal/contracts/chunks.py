"""Evidence-preserving chunk boundaries for CUAD and MAUD contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Protocol

from teamgym.core.information_allocation import owners_by_reference_unit


SECTION_HEADING_PATTERN = re.compile(
    r"(?m)^\s*(?:SECTION\s+\d+|ARTICLE\s+[A-Z0-9IVXLC]+|\d+(?:\.\d+)*[.)]?\s+[A-Z][^\n]{0,140})"
)


class EvidenceSpan(Protocol):
    """The source offsets needed to keep evidence intact during chunking."""

    @property
    def unit_id(self) -> str: ...

    @property
    def raw_start(self) -> int: ...

    @property
    def raw_end(self) -> int: ...


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    text: str
    word_count: int
    contains_rui_ids: tuple[str, ...]
    section_key: str = ""


def chunk_context(
    context: str,
    rui_ranges: Sequence[EvidenceSpan],
    *,
    target_words: int,
    min_words: int,
    soft_max_words: int,
    hard_max_words: int,
) -> list[Chunk]:
    word_spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", context)]
    if not word_spans:
        return []
    boundaries = _candidate_boundaries(context, word_spans, rui_ranges)
    chunks: list[Chunk] = []
    start = 0
    while start < len(context):
        if not context[start:].strip():
            chunks.append(_make_chunk(context, start, len(context), rui_ranges))
            break
        start_word = _word_index_at_or_after(word_spans, start)
        if start_word is None:
            break
        min_end = _word_end_at_count(word_spans, start_word, min_words)
        target_end = _word_end_at_count(word_spans, start_word, target_words)
        soft_end = _word_end_at_count(word_spans, start_word, soft_max_words)
        hard_end = _word_end_at_count(word_spans, start_word, hard_max_words)
        if hard_end is None:
            hard_end = len(context)
        if soft_end is None:
            soft_end = len(context)
        if target_end is None:
            target_end = len(context)
        min_end = min_end or target_end
        boundary = _choose_boundary(boundaries, start, min_end, soft_end, target_end)
        if boundary is None:
            boundary = _safe_boundary_at_or_after(hard_end, rui_ranges, len(context))
        boundary = max(boundary, start + 1)
        chunks.append(_make_chunk(context, start, boundary, rui_ranges))
        start = boundary
    return _merge_tiny_chunks(context, chunks, rui_ranges, min_words=min_words, hard_max_words=hard_max_words)


def _candidate_boundaries(
    context: str,
    word_spans: list[tuple[int, int]],
    rui_ranges: Sequence[EvidenceSpan],
) -> dict[int, int]:
    boundaries: dict[int, int] = {0: 0, len(context): 0}
    for match in SECTION_HEADING_PATTERN.finditer(context):
        _add_boundary(boundaries, match.start(), 1, rui_ranges)
    for match in re.finditer(r"\n\s*\n+", context):
        _add_boundary(boundaries, match.end(), 3, rui_ranges)
    for match in re.finditer(r"[.!?][\"')\]]?\s+", context):
        _add_boundary(boundaries, match.end(), 4, rui_ranges)
    for _, end in word_spans:
        _add_boundary(boundaries, end, 5, rui_ranges)
    return boundaries


def _add_boundary(boundaries: dict[int, int], offset: int, priority: int, rui_ranges: Sequence[EvidenceSpan]) -> None:
    if _splits_rui(offset, rui_ranges):
        return
    previous = boundaries.get(offset)
    if previous is None or priority < previous:
        boundaries[offset] = priority


def _choose_boundary(
    boundaries: dict[int, int],
    start: int,
    min_end: int,
    soft_end: int,
    target_end: int,
) -> int | None:
    candidates = [
        (priority, abs(offset - target_end), offset)
        for offset, priority in boundaries.items()
        if start < offset <= soft_end and offset >= min_end
    ]
    if not candidates:
        candidates = [
            (priority, abs(offset - target_end), offset)
            for offset, priority in boundaries.items()
            if start < offset <= soft_end
        ]
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _safe_boundary_at_or_after(offset: int, rui_ranges: Sequence[EvidenceSpan], context_len: int) -> int:
    boundary = min(offset, context_len)
    for rui in rui_ranges:
        if rui.raw_start < boundary < rui.raw_end:
            boundary = rui.raw_end
    return min(boundary, context_len)


def _make_chunk(context: str, start: int, end: int, rui_ranges: Sequence[EvidenceSpan]) -> Chunk:
    text = context[start:end]
    contains = tuple(
        rui.unit_id
        for rui in rui_ranges
        if start <= rui.raw_start and rui.raw_end <= end
    )
    return Chunk(
        start=start,
        end=end,
        text=text,
        word_count=word_count(text),
        contains_rui_ids=contains,
        section_key=section_key_at(context, start),
    )


def _merge_tiny_chunks(
    context: str,
    chunks: list[Chunk],
    rui_ranges: Sequence[EvidenceSpan],
    *,
    min_words: int,
    hard_max_words: int,
) -> list[Chunk]:
    if len(chunks) <= 1:
        return chunks
    out: list[Chunk] = []
    idx = 0
    while idx < len(chunks):
        current = chunks[idx]
        if current.word_count >= min_words or idx == len(chunks) - 1:
            out.append(current)
            idx += 1
            continue
        nxt = chunks[idx + 1]
        merged_words = word_count(context[current.start : nxt.end])
        if merged_words <= hard_max_words:
            out.append(_make_chunk(context, current.start, nxt.end, rui_ranges))
            idx += 2
        else:
            out.append(current)
            idx += 1
    return out


def split_conflicting_piu_chunks(
    context: str,
    chunks: list[Chunk],
    rui_ranges: Sequence[EvidenceSpan],
    rui_assignment: dict[str, list[str]],
    *,
    min_words: int,
) -> list[Chunk]:
    owners_by_rui = owners_by_reference_unit(rui_assignment)
    rui_by_id = {rui.unit_id: rui for rui in rui_ranges}
    out: list[Chunk] = []
    for chunk in chunks:
        owners = {
            tuple(owners_by_rui.get(rui_id, []))
            for rui_id in chunk.contains_rui_ids
        }
        owners.discard(())
        if len(owners) <= 1:
            out.append(chunk)
            continue
        splits = [chunk.start]
        contained = sorted((rui_by_id[rui_id] for rui_id in chunk.contains_rui_ids), key=lambda rui: rui.raw_start)
        for left, right in zip(contained, contained[1:]):
            if owners_by_rui.get(left.unit_id) == owners_by_rui.get(right.unit_id):
                continue
            boundary = _best_internal_split(context, left.raw_end, right.raw_start, min_words=min_words)
            if boundary is not None and chunk.start < boundary < chunk.end:
                splits.append(boundary)
        splits.append(chunk.end)
        splits = sorted(set(splits))
        for start, end in zip(splits, splits[1:]):
            out.append(_make_chunk(context, start, end, rui_ranges))
    return out


def _best_internal_split(context: str, left_end: int, right_start: int, *, min_words: int) -> int | None:
    if right_start <= left_end:
        return None
    segment = context[left_end:right_start]
    candidates: list[int] = []
    for match in re.finditer(r"\n\s*\n+", segment):
        candidates.append(left_end + match.end())
    for match in re.finditer(r"[.!?][\"')\]]?\s+", segment):
        candidates.append(left_end + match.end())
    if not candidates:
        for match in re.finditer(r"\s+", segment):
            candidates.append(left_end + match.end())
    if not candidates:
        candidates.append(left_end)
    if not candidates:
        return None
    midpoint = left_end + (right_start - left_end) // 2
    return min(candidates, key=lambda offset: abs(offset - midpoint))


def _word_index_at_or_after(word_spans: list[tuple[int, int]], offset: int) -> int | None:
    for idx, (_, end) in enumerate(word_spans):
        if end > offset:
            return idx
    return None


def _word_end_at_count(word_spans: list[tuple[int, int]], start_word: int, count: int) -> int | None:
    target = start_word + max(1, count) - 1
    if target >= len(word_spans):
        return None
    return word_spans[target][1]


def _splits_rui(offset: int, rui_ranges: Sequence[EvidenceSpan]) -> bool:
    return any(rui.raw_start < offset < rui.raw_end for rui in rui_ranges)


def section_key_at(context: str, offset: int) -> str:
    current = "document_preamble"
    for match in SECTION_HEADING_PATTERN.finditer(context):
        if match.start() > offset:
            break
        heading = " ".join(match.group(0).split())
        if heading:
            current = heading[:160].lower()
    return current


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))
