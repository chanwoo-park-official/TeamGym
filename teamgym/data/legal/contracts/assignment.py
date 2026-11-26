"""Participant roles and chunk assignment for CUAD and MAUD contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
import random
import re
from typing import Any, Protocol

from teamgym.core.information_allocation import owners_by_reference_unit

from .chunks import Chunk, EvidenceSpan


ASSIGNMENT_SCORE_WEIGHTS = {
    "role_overlap": 1.0,
    "feature_overlap": 1.5,
    "proximity": 1.0,
    "section_match": 0.5,
    "load_penalty": 0.8,
    "related_semantic_scale": 0.25,
    "unrelated_semantic_scale": 1.0,
    "word_balance": 1.2,
    "chunk_balance": 0.7,
    "owned_related_rui_penalty": 8.0,
    "random_jitter": 0.02,
    "softmax_temperature": 0.45,
}


STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "and",
    "agreement",
    "between",
    "clause",
    "contract",
    "could",
    "each",
    "from",
    "for",
    "have",
    "into",
    "must",
    "only",
    "or",
    "other",
    "party",
    "profile",
    "review",
    "reviewer",
    "shall",
    "should",
    "span",
    "such",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "with",
    "within",
    "would",
}


class AssignmentEvidence(EvidenceSpan, Protocol):
    """Evidence text and features used to score assignments."""

    @property
    def clean_text(self) -> str: ...

    @property
    def raw_text(self) -> str: ...

    @property
    def legal_feature(self) -> str: ...

    @property
    def summary(self) -> str: ...

    @property
    def section_key(self) -> str: ...


def participants_from_assignment(
    rui_assignment: dict[str, list[str]],
    rui_ranges: Sequence[AssignmentEvidence],
    *,
    role_style: str,
) -> list[dict[str, Any]]:
    by_id = {rui.unit_id: rui for rui in rui_ranges}
    participants: list[dict[str, Any]] = []
    for idx, (participant_id, rui_ids) in enumerate(sorted(rui_assignment.items()), start=1):
        features = [by_id[rui_id].legal_feature for rui_id in rui_ids if rui_id in by_id]
        role = f"reviewer_{idx}" if role_style == "generic" else _role_from_features(features, idx)
        role_terms = sorted(_keyword_set(" ".join([role, *features, *[by_id[r].summary for r in rui_ids if r in by_id]])))
        feature_terms = sorted(_keyword_set(" ".join([*features, *[by_id[r].summary for r in rui_ids if r in by_id]])))
        participants.append(
            {
                "participant_id": participant_id,
                "public_role": role,
                "owned_rui_ids": rui_ids,
                "role_terms": role_terms,
                "owned_rui_feature_terms": feature_terms,
                "owned_rui_centers": [
                    (by_id[rui_id].raw_start + by_id[rui_id].raw_end) / 2
                    for rui_id in rui_ids
                    if rui_id in by_id
                ],
                "owned_rui_section_keys": sorted(
                    {
                        by_id[rui_id].section_key
                        for rui_id in rui_ids
                        if rui_id in by_id and by_id[rui_id].section_key
                    }
                ),
                "private_doc_unit_ids": [],
                "private_word_count": 0,
            }
        )
    return participants


def _role_from_features(features: list[str], idx: int) -> str:
    cleaned = [feature.replace("_", " ").replace("-", " ").strip().lower() for feature in features if feature.strip()]
    if not cleaned:
        return f"context reviewer {idx}"
    if len(cleaned) == 1:
        return f"{cleaned[0]} reviewer"
    first_terms = []
    for feature in cleaned[:2]:
        tokens = [
            token
            for token in _tokens(feature)
            if token not in STOPWORDS and token not in {"reviewer"} and len(token) > 2
        ]
        if tokens:
            first_terms.append(" ".join(tokens[:2]))
    return f"{' and '.join(first_terms)} reviewer" if first_terms else f"reviewer {idx}"


def assign_chunks(
    chunks: list[Chunk],
    participants: list[dict[str, Any]],
    rui_assignment: dict[str, list[str]],
    rui_ranges: Sequence[AssignmentEvidence],
    *,
    common_rui_ids: set[str],
    target_words: int,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    rng = rng or random.Random(0)
    owners_by_rui = owners_by_reference_unit(rui_assignment)
    participant_by_id = {participant["participant_id"]: participant for participant in participants}
    rui_by_id = {rui.unit_id: rui for rui in rui_ranges}
    doc_units_by_index: dict[int, dict[str, Any]] = {}
    neutral_chunks: list[tuple[int, Chunk]] = []
    for idx, chunk in enumerate(chunks, start=1):
        if chunk.contains_rui_ids:
            common_in_chunk = common_rui_ids & set(chunk.contains_rui_ids)
            assigned = sorted(
                {
                    participant_id
                    for rui_id in chunk.contains_rui_ids
                    for participant_id in owners_by_rui.get(rui_id, [])
                }
            )
            if common_in_chunk:
                assigned = sorted({*assigned, *participant_by_id})
                reason = (
                    "common_anchor"
                    if len(common_in_chunk) == len(chunk.contains_rui_ids)
                    else "rui_owner_with_common_anchor"
                )
            else:
                reason = "rui_owner"
            scores = {}
        else:
            neutral_chunks.append((idx, chunk))
            continue
        if not assigned and participants:
            assigned = [min(participants, key=lambda p: int(p.get("private_word_count") or 0))["participant_id"]]
            reason = "least_loaded_balance_assignment"
        _record_participant_assignment(participant_by_id, assigned, f"du_{idx:04d}", chunk.word_count)
        doc_units_by_index[idx] = _document_unit_record(
            idx=idx,
            chunk=chunk,
            assigned=assigned,
            reason=reason,
            scores=scores,
            rui_by_id=rui_by_id,
        )

    assignment_order = list(neutral_chunks)
    rng.shuffle(assignment_order)
    for idx, chunk in assignment_order:
        assigned, scores = _balanced_participant_for_neutral_chunk(
            chunk,
            participants,
            rui_ranges=rui_ranges,
            rui_assignment=rui_assignment,
            target_words=target_words,
            rng=rng,
        )
        reason = "balanced_neutral_context"
        if not assigned and participants:
            assigned = [min(participants, key=lambda p: int(p.get("private_word_count") or 0))["participant_id"]]
            reason = "least_loaded_balance_assignment"
        _record_participant_assignment(participant_by_id, assigned, f"du_{idx:04d}", chunk.word_count)
        doc_units_by_index[idx] = _document_unit_record(
            idx=idx,
            chunk=chunk,
            assigned=assigned,
            reason=reason,
            scores=scores,
            rui_by_id=rui_by_id,
        )
    return [doc_units_by_index[idx] for idx in sorted(doc_units_by_index)]


def _record_participant_assignment(
    participant_by_id: dict[str, dict[str, Any]],
    assigned: list[str],
    doc_unit_id: str,
    word_count: int,
) -> None:
    for participant_id in assigned:
        participant = participant_by_id[participant_id]
        participant["private_doc_unit_ids"].append(doc_unit_id)
        participant["private_word_count"] += word_count


def _document_unit_record(
    *,
    idx: int,
    chunk: Chunk,
    assigned: list[str],
    reason: str,
    scores: dict[str, dict[str, float]],
    rui_by_id: dict[str, Any],
) -> dict[str, Any]:
    return {
        "doc_unit_id": f"du_{idx:04d}",
        "start": chunk.start,
        "end": chunk.end,
        "word_count": chunk.word_count,
        "section_key": chunk.section_key,
        "text": chunk.text,
        "assigned_to": assigned,
        "hidden_contains_rui_ids": list(chunk.contains_rui_ids),
        "hidden_contains_rui_features": [
            rui_by_id[rui_id].legal_feature
            for rui_id in chunk.contains_rui_ids
            if rui_id in rui_by_id
        ],
        "assignment_reason": reason,
        "assignment_scores": scores,
    }


def _balanced_participant_for_neutral_chunk(
    chunk: Chunk,
    participants: list[dict[str, Any]],
    *,
    rui_ranges: Sequence[AssignmentEvidence] | None = None,
    rui_assignment: dict[str, list[str]] | None = None,
    target_words: int,
    rng: random.Random | None = None,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    related = _related_rui_ids_for_chunk(chunk, rui_ranges or [], target_words=target_words)
    owner_map = owners_by_reference_unit(rui_assignment or {})
    candidate_ids = _safe_candidate_ids_for_related_ruis(participants, related, owner_map)
    if not candidate_ids:
        candidate_ids = {str(participant.get("participant_id")) for participant in participants}
    safe_candidate_ids = set(candidate_ids)
    counts = participant_doc_unit_counts(participants)
    words = {
        str(participant.get("participant_id")): int(participant.get("private_word_count") or 0)
        for participant in participants
    }
    scoped_counts = {pid: counts.get(pid, 0) for pid in candidate_ids}
    min_count = min(scoped_counts.values()) if scoped_counts else 0
    candidates = [
        participant
        for participant in participants
        if str(participant.get("participant_id")) in candidate_ids
        and counts.get(str(participant.get("participant_id")), 0) == min_count
    ]
    candidate_ids = {str(participant.get("participant_id")) for participant in candidates}
    _, scores = _best_participant_for_chunk(
        chunk,
        participants,
        target_words=target_words,
        related_rui_ids=related,
        rui_assignment=rui_assignment,
        candidate_participant_ids=candidate_ids,
        rng=rng,
    )
    if not scores:
        return [], {}
    scores["_assignment_context"] = {
        "related_rui_ids": related,
        "safe_candidate_ids": sorted(safe_candidate_ids),
        "balanced_candidate_ids": sorted(candidate_ids),
    }
    for participant in participants:
        participant_id = str(participant.get("participant_id"))
        if participant_id not in scores:
            continue
        scores[participant_id]["chunk_count_before"] = float(counts.get(participant_id, 0))
        scores[participant_id]["word_count_before"] = float(words.get(participant_id, 0))
        scores[participant_id]["balance_priority"] = 1.0 if participant in candidates else 0.0
    chosen_id = _sample_participant_id(
        [str(participant.get("participant_id")) for participant in candidates],
        scores,
        rng=rng,
    )
    return ([chosen_id] if chosen_id else []), scores


def _best_participant_for_chunk(
    chunk: Chunk,
    participants: list[dict[str, Any]],
    *,
    target_words: int,
    related_rui_ids: Iterable[str] | None = None,
    rui_assignment: dict[str, list[str]] | None = None,
    candidate_participant_ids: set[str] | None = None,
    rng: random.Random | None = None,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    chunk_terms = _keyword_set(chunk.text)
    scores: dict[str, dict[str, float]] = {}
    related_set = {str(item) for item in (related_rui_ids or []) if str(item)}
    owners_by_rui = owners_by_reference_unit(rui_assignment or {})
    owned_related_by_pid = _owned_related_rui_ids_by_participant(participants, related_set, owners_by_rui)
    candidate_participant_ids = set(candidate_participant_ids) if candidate_participant_ids else None
    counts = participant_doc_unit_counts(participants)
    words = {
        str(participant.get("participant_id")): int(participant.get("private_word_count") or 0)
        for participant in participants
    }
    min_words = min(words.values()) if words else 0
    min_count = min(counts.values()) if counts else 0
    for participant in participants:
        participant_id = str(participant["participant_id"])
        role_terms = set(participant.get("role_terms") or [])
        feature_terms = set(participant.get("owned_rui_feature_terms") or [])
        role_overlap = _fraction_overlap(role_terms, chunk_terms)
        feature_overlap = _fraction_overlap(feature_terms, chunk_terms)
        proximity = _proximity_to_owned_rui(chunk, participant, target_words=target_words)
        section_match = _section_match(chunk, participant)
        load_penalty = float(participant.get("private_word_count") or 0) / max(1, target_words * 2)
        semantic_raw = (
            ASSIGNMENT_SCORE_WEIGHTS["role_overlap"] * role_overlap
            + ASSIGNMENT_SCORE_WEIGHTS["feature_overlap"] * feature_overlap
            + ASSIGNMENT_SCORE_WEIGHTS["proximity"] * proximity
            + ASSIGNMENT_SCORE_WEIGHTS["section_match"] * section_match
            - ASSIGNMENT_SCORE_WEIGHTS["load_penalty"] * load_penalty
        )
        semantic_scale = (
            ASSIGNMENT_SCORE_WEIGHTS["related_semantic_scale"]
            if related_set
            else ASSIGNMENT_SCORE_WEIGHTS["unrelated_semantic_scale"]
        )
        word_balance_penalty = max(0.0, float(words.get(participant_id, 0) - min_words) / max(1, target_words))
        chunk_balance_penalty = max(0.0, float(counts.get(participant_id, 0) - min_count))
        owned_related_ids = owned_related_by_pid.get(participant_id, [])
        unsafe_penalty = 1.0 if owned_related_ids else 0.0
        jitter = (rng.random() * 2 - 1) * ASSIGNMENT_SCORE_WEIGHTS["random_jitter"] if rng else 0.0
        final = (
            semantic_scale * semantic_raw
            - ASSIGNMENT_SCORE_WEIGHTS["word_balance"] * word_balance_penalty
            - ASSIGNMENT_SCORE_WEIGHTS["chunk_balance"] * chunk_balance_penalty
            - ASSIGNMENT_SCORE_WEIGHTS["owned_related_rui_penalty"] * unsafe_penalty
            + jitter
        )
        scores[participant_id] = {
            "final": round(final, 4),
            "semantic_raw": round(semantic_raw, 4),
            "semantic_scale": round(semantic_scale, 4),
            "role_overlap": round(role_overlap, 4),
            "feature_overlap": round(feature_overlap, 4),
            "proximity": round(proximity, 4),
            "section_match": round(section_match, 4),
            "load_penalty": round(load_penalty, 4),
            "word_balance_penalty": round(word_balance_penalty, 4),
            "chunk_balance_penalty": round(chunk_balance_penalty, 4),
            "is_riu_related_context": 1.0 if related_set else 0.0,
            "owns_related_rui": 1.0 if owned_related_ids else 0.0,
            "candidate_after_safety_filter": (
                1.0 if candidate_participant_ids is None or participant_id in candidate_participant_ids else 0.0
            ),
        }
    if not scores:
        return [], {}
    eligible_scores = {
        pid: score
        for pid, score in scores.items()
        if candidate_participant_ids is None or pid in candidate_participant_ids
    }
    if not eligible_scores:
        eligible_scores = scores
    best = max(eligible_scores, key=lambda pid: eligible_scores[pid]["final"])
    if scores[best]["final"] <= 0:
        eligible_participants = [
            participant
            for participant in participants
            if str(participant.get("participant_id")) in eligible_scores
        ]
        best = min(eligible_participants or participants, key=lambda p: int(p.get("private_word_count") or 0))[
            "participant_id"
        ]
    return [best], scores


def _related_rui_ids_for_chunk(
    chunk: Chunk,
    rui_ranges: Sequence[AssignmentEvidence],
    *,
    target_words: int,
) -> list[str]:
    """Classify non-RUI chunks that may explain or neighbor a required RIU."""

    if chunk.contains_rui_ids:
        return []
    chunk_terms = _keyword_set(chunk.text)
    related: list[str] = []
    for rui in rui_ranges:
        rui_terms = _keyword_set(" ".join([rui.clean_text, rui.raw_text, rui.legal_feature, rui.summary]))
        keyword_overlap = max(
            _fraction_overlap(rui_terms, chunk_terms),
            _fraction_overlap(chunk_terms, rui_terms),
        )
        distance = _char_distance_between_ranges(chunk.start, chunk.end, rui.raw_start, rui.raw_end)
        approximate_word_distance = distance / 6.0
        proximity = 1.0 / (1.0 + approximate_word_distance / max(1, target_words))
        same_section = bool(chunk.section_key and rui.section_key and chunk.section_key == rui.section_key)
        if proximity >= 0.62 or keyword_overlap >= 0.2 or (same_section and keyword_overlap >= 0.08):
            related.append(rui.unit_id)
    return sorted(_dedupe(related))


def _safe_candidate_ids_for_related_ruis(
    participants: list[dict[str, Any]],
    related_rui_ids: list[str],
    owners_by_rui: dict[str, list[str]],
) -> set[str]:
    all_ids = {str(participant.get("participant_id")) for participant in participants}
    if not related_rui_ids:
        return all_ids
    unsafe = {
        str(participant_id)
        for rui_id in related_rui_ids
        for participant_id in owners_by_rui.get(str(rui_id), [])
    }
    safe = all_ids - unsafe
    return safe or all_ids


def _owned_related_rui_ids_by_participant(
    participants: list[dict[str, Any]],
    related_rui_ids: set[str],
    owners_by_rui: dict[str, list[str]],
) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {
        str(participant.get("participant_id")): set()
        for participant in participants
    }
    if not related_rui_ids:
        return {pid: [] for pid in out}
    for rui_id in related_rui_ids:
        owner_ids = owners_by_rui.get(str(rui_id))
        if owner_ids:
            for participant_id in owner_ids:
                out.setdefault(str(participant_id), set()).add(str(rui_id))
            continue
        for participant in participants:
            participant_id = str(participant.get("participant_id"))
            if rui_id in {str(item) for item in participant.get("owned_rui_ids") or []}:
                out.setdefault(participant_id, set()).add(str(rui_id))
    return {pid: sorted(values) for pid, values in out.items()}


def _sample_participant_id(
    candidate_ids: list[str],
    scores: dict[str, dict[str, float]],
    *,
    rng: random.Random | None,
) -> str:
    if not candidate_ids:
        return ""
    ordered = [pid for pid in sorted(candidate_ids) if pid in scores]
    if not ordered:
        return sorted(candidate_ids)[0]
    if rng is None or len(ordered) == 1:
        return max(ordered, key=lambda pid: scores[pid]["final"])
    temperature = max(0.01, float(ASSIGNMENT_SCORE_WEIGHTS["softmax_temperature"]))
    raw_scores = [float(scores[pid]["final"]) / temperature for pid in ordered]
    max_score = max(raw_scores)
    weights = [math.exp(score - max_score) for score in raw_scores]
    total = sum(weights)
    if total <= 0:
        return max(ordered, key=lambda pid: scores[pid]["final"])
    draw = rng.random() * total
    cumulative = 0.0
    for pid, weight in zip(ordered, weights):
        cumulative += weight
        if draw <= cumulative:
            return pid
    return ordered[-1]


def _char_distance_between_ranges(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    if left_end < right_start:
        return right_start - left_end
    if right_end < left_start:
        return left_start - right_end
    return 0


def _fraction_overlap(query_terms: set[str], chunk_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & chunk_terms) / len(query_terms)


def _proximity_to_owned_rui(
    chunk: Chunk,
    participant: dict[str, Any],
    *,
    target_words: int,
) -> float:
    centers = []
    for value in participant.get("owned_rui_centers") or []:
        try:
            centers.append(float(value))
        except (TypeError, ValueError):
            continue
    if not centers:
        return 0.0
    chunk_center = (chunk.start + chunk.end) / 2
    nearest_char_distance = min(abs(chunk_center - center) for center in centers)
    approximate_word_distance = nearest_char_distance / 6.0
    return 1.0 / (1.0 + approximate_word_distance / max(1, target_words))


def _section_match(chunk: Chunk, participant: dict[str, Any]) -> float:
    owned_sections = set(participant.get("owned_rui_section_keys") or [])
    if not chunk.section_key or not owned_sections:
        return 0.0
    return 1.0 if chunk.section_key in owned_sections else 0.0


def participant_doc_unit_counts(participants: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(participant.get("participant_id")): len(participant.get("private_doc_unit_ids") or [])
        for participant in participants
    }


def _keyword_set(text: str) -> set[str]:
    return {
        token
        for token in _tokens(text)
        if len(token) >= 4 and token not in STOPWORDS
    }


def _tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    if keep_stopwords:
        return re.findall(r"\S+", str(text or ""))
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]*", str(text or ""))]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
