"""Post-episode lexical matching from public transcript text to hidden RIUs."""

from __future__ import annotations

import re
from typing import Any


STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "also",
    "among",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "could",
    "does",
    "from",
    "have",
    "into",
    "more",
    "most",
    "only",
    "other",
    "over",
    "should",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "which",
    "while",
    "with",
    "would",
}


def match_public_rius(episode: dict[str, Any], datapoint: dict[str, Any]) -> dict[str, Any]:
    public_text = _episode_public_text(episode)
    public_tokens = set(_tokens(public_text))
    matched: list[dict[str, Any]] = []
    for rui in datapoint.get("reference_information_units", []):
        if not isinstance(rui, dict) or not isinstance(rui.get("rui_id"), str):
            continue
        source_text = _rui_match_text(rui)
        rui_tokens = set(_tokens(source_text))
        if not rui_tokens:
            continue
        overlap = sorted(rui_tokens & public_tokens)
        recall = len(overlap) / len(rui_tokens)
        matched_by = None
        if _normalized(source_text) and _normalized(source_text) in _normalized(public_text):
            matched_by = "exact_normalized_substring"
        elif recall >= 0.55:
            matched_by = "token_recall_ge_0.55"
        elif len(overlap) >= 8 and recall >= 0.4:
            matched_by = "token_overlap_ge_8_and_recall_ge_0.4"
        if matched_by is not None:
            matched.append(
                {
                    "rui_id": rui["rui_id"],
                    "matched_by": matched_by,
                    "token_recall": round(recall, 4),
                    "overlap_token_count": len(overlap),
                    "sample_overlap_tokens": overlap[:20],
                }
            )
    explicit = {
        str(item)
        for message in episode.get("messages", [])
        if isinstance(message, dict)
        for item in message.get("disclosed_rui_ids", [])
        if isinstance(item, str)
    }
    matched_ids = {row["rui_id"] for row in matched}
    return {
        "schema_version": "teamgym_rui_lexical_match",
        "matched_rui_ids": sorted(explicit | matched_ids),
        "explicit_disclosed_rui_ids": sorted(explicit),
        "lexically_matched_rui_ids": sorted(matched_ids),
        "lexical_matches": matched,
        "method_note": (
            "RIUs are matched after the episode from public transcript/final-rationale text. "
            "Actors do not receive RIU ids, RUI text, smallest sufficient sets, or oracle labels."
        ),
    }


def _episode_public_text(episode: dict[str, Any]) -> str:
    chunks: list[str] = []
    for message in episode.get("messages", []):
        if isinstance(message, dict):
            chunks.append(str(message.get("content") or ""))
    final_decision = episode.get("final_decision")
    if isinstance(final_decision, dict):
        chunks.append(str(final_decision.get("rationale") or ""))
    return "\n".join(chunks)


def _rui_match_text(rui: dict[str, Any]) -> str:
    parts = [
        str(rui.get("summary") or ""),
        str(rui.get("text") or ""),
    ]
    return "\n".join(part for part in parts if part.strip())


def _tokens(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", str(text).casefold()):
        if token not in STOPWORDS:
            tokens.append(token)
    return tokens


def _normalized(text: str) -> str:
    return " ".join(_tokens(text))
