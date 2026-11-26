"""Cheap lexical hints for semantic packet leakage audits."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{2,}|\d+(?:\.\d+)?", re.IGNORECASE)


def packet_bm25_hints(
    packet: dict[str, Any],
    riu_rows: list[dict[str, Any]],
    *,
    unauthorized_rui_ids: set[str],
    top_k: int = 4,
) -> list[dict[str, Any]]:
    contexts = [
        row
        for row in packet.get("contexts") or []
        if isinstance(row, dict) and str(row.get("text") or "").strip()
    ]
    if not contexts:
        return []
    docs = [_tokens(row.get("text")) for row in contexts]
    lengths = [max(1, len(tokens)) for tokens in docs]
    average_length = sum(lengths) / len(lengths)
    document_frequency: Counter[str] = Counter()
    for tokens in docs:
        document_frequency.update(set(tokens))

    hints: list[dict[str, Any]] = []
    for rui in riu_rows:
        rui_id = str(rui.get("riu_id", ""))
        if rui_id not in unauthorized_rui_ids:
            continue
        query_parts: list[str] = []
        for key in ("text", "information_feature", "entity_scope", "period_scope"):
            query_parts.append(str(rui.get(key, "")))
        query = _tokens(" ".join(query_parts))
        scored: list[tuple[float, int]] = []
        for index, tokens in enumerate(docs):
            score = _bm25(
                query,
                tokens,
                document_frequency=document_frequency,
                document_count=len(docs),
                document_length=lengths[index],
                average_length=average_length,
            )
            if score > 0:
                scored.append((score, index))
        for score, index in sorted(scored, reverse=True)[: max(1, int(top_k))]:
            context = contexts[index]
            hints.append(
                {
                    "riu_id": rui_id,
                    "context_id": (context.get("source_ref") or {}).get("context_id"),
                    "bm25_score": round(score, 4),
                    "matched_tokens": sorted(
                        set(query) & set(docs[index])
                    )[:20],
                }
            )
    return hints


def _bm25(
    query: list[str],
    document: list[str],
    *,
    document_frequency: Counter[str],
    document_count: int,
    document_length: int,
    average_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    frequencies = Counter(document)
    score = 0.0
    for token in set(query):
        frequency = frequencies.get(token, 0)
        if not frequency:
            continue
        df = document_frequency.get(token, 0)
        inverse_frequency = math.log(
            1 + (document_count - df + 0.5) / (df + 0.5)
        )
        denominator = frequency + k1 * (
            1 - b + b * document_length / max(1.0, average_length)
        )
        score += inverse_frequency * frequency * (k1 + 1) / denominator
    return score


def _tokens(value: Any) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(str(value or ""))]
