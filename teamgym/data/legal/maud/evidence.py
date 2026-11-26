"""MAUD row-evidence splitting and agreement-text grounding."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any

_SPLIT_RE = re.compile(
    r"(?:<\s*omitted\s*>|\[\[\s*omitted\s*\]\]|\[\s*omitted\s*\]|\.{3,}|\n\s*\n|Page\s+\d+)",
    re.IGNORECASE,
)
_PAGE_CITE_RE = re.compile(r"\s*\((?:Page|Pages)\s+[^)]*\)\s*$", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_CLAUSE_MARKER_RE = re.compile(r"\(([a-z]|\d+|[ivxlcdm]{1,6})\)\s+", re.IGNORECASE)
_TRAILING_SECTION_HEADING_RE = re.compile(
    r"\s+\d+(?:\.\d+)+\.\s+[A-Z][A-Za-z0-9 ,;:/&()'-]{5,120}\.?\s*$"
)


@dataclass(frozen=True)
class _NormalizedText:
    text: str
    original_index_by_norm_index: list[int]


def split_maud_row_text(text: str) -> tuple[list[str], list[str]]:
    """Split a MAUD row evidence packet into usable source-evidence blocks."""

    blocks: list[str] = []
    dropped: list[str] = []
    pending_prefix = ""
    original_cleaned = _clean_block(text)
    for raw_part in _SPLIT_RE.split(text or ""):
        part = _clean_block(raw_part)
        if not part:
            continue
        if pending_prefix:
            part = f"{pending_prefix} {part}".strip()
            pending_prefix = ""
        words = _word_count(part)
        if words < 8:
            if blocks:
                blocks[-1] = f"{blocks[-1]} {part}".strip()
            else:
                pending_prefix = part
            continue
        if _is_bad_fragment(part):
            dropped.append(part)
            continue
        blocks.append(part)
    if pending_prefix:
        if blocks:
            blocks[-1] = f"{blocks[-1]} {pending_prefix}".strip()
        elif pending_prefix != original_cleaned:
            dropped.append(pending_prefix)
    if not blocks and original_cleaned:
        return [original_cleaned], dropped
    return _dedupe_preserve_order(blocks), dropped


def match_evidence_block(
    block: str,
    contract_text: str,
    normalized_contract: _NormalizedText | None = None,
    chunks: list[dict[str, Any]] | None = None,
    aggressive_contract: _NormalizedText | None = None,
) -> dict[str, Any]:
    """Match one cleaned evidence block to the full agreement text."""

    normalized_contract = normalized_contract or _normalize_with_offsets(contract_text)
    chunks = chunks if chunks is not None else _agreement_chunks(contract_text)
    aggressive_contract = aggressive_contract or _aggressive_normalize_with_offsets(contract_text)

    for variant, variant_kind in _match_variants(block):
        raw_start = contract_text.find(variant)
        if raw_start >= 0:
            return _match_result(
                "exact_raw" if variant_kind == "original" else f"exact_raw_{variant_kind}",
                1.0,
                raw_start,
                raw_start + len(variant),
                contract_text[raw_start : raw_start + len(variant)],
            )

        normalized_block = _normalize_with_offsets(variant).text
        norm_start = normalized_contract.text.find(normalized_block)
        if norm_start >= 0 and normalized_block:
            start = normalized_contract.original_index_by_norm_index[norm_start]
            end_norm_idx = norm_start + len(normalized_block) - 1
            end = normalized_contract.original_index_by_norm_index[end_norm_idx] + 1
            status = "exact_whitespace_normalized" if variant_kind == "original" else f"exact_whitespace_normalized_{variant_kind}"
            return _match_result(
                status,
                1.0,
                start,
                end,
                contract_text[start:end],
            )

    for variant, variant_kind in _match_variants(block):
        aggressive_block = _aggressive_normalize_with_offsets(variant).text
        if len(aggressive_block.split()) < 10:
            continue
        aggr_start = aggressive_contract.text.find(aggressive_block)
        if aggr_start >= 0:
            start = aggressive_contract.original_index_by_norm_index[aggr_start]
            end_norm_idx = aggr_start + len(aggressive_block) - 1
            end = aggressive_contract.original_index_by_norm_index[end_norm_idx] + 1
            start, end = _expand_aggressive_bounds(contract_text, start, end)
            status = "exact_aggressive" if variant_kind == "original" else f"exact_aggressive_{variant_kind}"
            return _match_result(
                status,
                1.0,
                start,
                end,
                contract_text[start:end],
            )

    normalized_block = _normalize_with_offsets(block).text
    fuzzy = _best_fuzzy_chunk(normalized_block, chunks)
    if fuzzy["score"] >= 0.95:
        status = "fuzzy_high"
    elif fuzzy["score"] >= 0.85:
        status = "fuzzy_medium"
    elif fuzzy["score"] >= 0.70:
        status = "partial_overlap"
    else:
        status = "row_excerpt_only"

    if status in {"fuzzy_high", "fuzzy_medium"}:
        return _match_result(
            status,
            fuzzy["score"],
            fuzzy.get("start"),
            fuzzy.get("end"),
            fuzzy.get("text") or block,
        )
    return _match_result(status, fuzzy["score"], None, None, block)


def _clean_block(text: str) -> str:
    text = str(text or "").replace("\u00a0", " ")
    text = _PAGE_CITE_RE.sub("", text.strip())
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _match_variants(block: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = [(block, "original")]
    stripped_tail = _strip_trailing_section_heading(block)
    if stripped_tail != block:
        variants.append((stripped_tail, "trimmed_heading"))
    marker = _CLAUSE_MARKER_RE.search(block)
    if marker and 0 < marker.start() <= 160:
        suffix = block[marker.start() :].strip()
        variants.append((suffix, "trimmed_heading"))
        suffix_stripped_tail = _strip_trailing_section_heading(suffix)
        if suffix_stripped_tail != suffix:
            variants.append((suffix_stripped_tail, "trimmed_heading"))
    return _dedupe_variant_order(variants)


def _strip_trailing_section_heading(block: str) -> str:
    stripped = _TRAILING_SECTION_HEADING_RE.sub("", block).strip()
    return stripped or block


def _dedupe_variant_order(variants: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for text, kind in variants:
        key = _normalize_with_offsets(text).text
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((text, kind))
    return out


def _normalize_with_offsets(text: str) -> _NormalizedText:
    chars: list[str] = []
    mapping: list[int] = []
    in_space = False
    for idx, char in enumerate(text):
        normalized = _normalize_char(char)
        if normalized.isspace():
            if chars and not in_space:
                chars.append(" ")
                mapping.append(idx)
            in_space = True
            continue
        chars.append(normalized)
        mapping.append(idx)
        in_space = False
    while chars and chars[-1] == " ":
        chars.pop()
        mapping.pop()
    return _NormalizedText("".join(chars), mapping)


def _normalize_char(char: str) -> str:
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
    return replacements.get(char, char).lower()


def _aggressive_normalize_with_offsets(text: str) -> _NormalizedText:
    chars: list[str] = []
    mapping: list[int] = []
    in_space = False
    for idx, char in enumerate(text):
        normalized = _normalize_char(char)
        if normalized.isalnum():
            chars.append(normalized)
            mapping.append(idx)
            in_space = False
            continue
        if chars and not in_space:
            chars.append(" ")
            mapping.append(idx)
            in_space = True
    while chars and chars[-1] == " ":
        chars.pop()
        mapping.pop()
    return _NormalizedText("".join(chars), mapping)


def _expand_aggressive_bounds(contract_text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and not contract_text[start - 1].isalnum() and not contract_text[start - 1].isspace():
        start -= 1
    while end < len(contract_text) and not contract_text[end].isalnum() and not contract_text[end].isspace():
        end += 1
    return start, end


def _agreement_chunks(contract_text: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    cursor = 0
    for raw_line in contract_text.splitlines(keepends=True):
        line_start = cursor
        cursor += len(raw_line)
        cleaned = raw_line.strip()
        if _word_count(cleaned) >= 8:
            content_start = line_start + raw_line.find(cleaned)
            chunks.append(
                {
                    "text": cleaned,
                    "normalized": _normalize_with_offsets(cleaned).text,
                    "start": content_start,
                    "end": content_start + len(cleaned),
                }
            )
    if chunks:
        return chunks
    cleaned = contract_text.strip()
    return [
        {
            "text": cleaned,
            "normalized": _normalize_with_offsets(cleaned).text,
            "start": 0,
            "end": len(cleaned),
        }
    ]


def _best_fuzzy_chunk(normalized_block: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    block_words = normalized_block.split()
    distinctive = {word for word in block_words if len(word) >= 7}
    best = {"score": 0.0, "text": "", "start": None, "end": None}
    for chunk in chunks:
        normalized_chunk = str(chunk.get("normalized") or "")
        chunk_words = normalized_chunk.split()
        if distinctive:
            overlap = distinctive & {word for word in chunk_words if len(word) >= 7}
            if len(overlap) < min(3, len(distinctive)):
                continue
        score = _chunk_similarity(normalized_block, normalized_chunk)
        if score > float(best["score"]):
            best = {
                "score": score,
                "text": chunk.get("text") or "",
                "start": chunk.get("start"),
                "end": chunk.get("end"),
            }
    return best


def _chunk_similarity(normalized_block: str, normalized_chunk: str) -> float:
    block_words = normalized_block.split()
    chunk_words = normalized_chunk.split()
    if not normalized_block or not normalized_chunk:
        return 0.0
    if len(chunk_words) <= max(len(block_words) * 2, len(block_words) + 20):
        return SequenceMatcher(None, normalized_block, normalized_chunk).ratio()
    window_size = max(1, len(block_words))
    step = max(1, window_size // 3)
    best = 0.0
    for start in range(0, max(1, len(chunk_words) - window_size + 1), step):
        window = " ".join(chunk_words[start : start + window_size])
        best = max(best, SequenceMatcher(None, normalized_block, window).ratio())
    return best


def _match_result(
    status: str,
    score: float,
    start: int | None,
    end: int | None,
    text: str,
) -> dict[str, Any]:
    return {
        "match_status": status,
        "match_score": round(float(score), 4),
        "agreement_start": start,
        "agreement_end": end,
        "matched_text": text,
    }


def _word_count(text: str) -> int:
    return len(_SPACE_RE.sub(" ", str(text or "")).strip().split())


def _is_bad_fragment(text: str) -> bool:
    stripped = text.strip()
    words = _word_count(stripped)
    if words < 12 and stripped.endswith(("(", ",", ";", ":")):
        return True
    if words < 12 and stripped[:1].islower() and stripped.lower().startswith(("from ", "to ", "and ", "or ", "of ")):
        return True
    return False


def _dedupe_preserve_order(blocks: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for block in blocks:
        key = _normalize_with_offsets(block).text
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
    return out
